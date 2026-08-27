"""webtorch.vl -- vision-language (Qwen2.5-VL) support on top of webtorch.llm.

Adds a quantized ViT vision tower + M-RoPE + multimodal sequence assembly so a
Qwen2.5-VL model can caption/answer-about an image in the browser:

    import vl
    model = await vl.VLCausalLM.from_qwen2_5_vl("/models/qwen2.5-vl-3b")
    out = model.generate("Describe this image.", image=pil_image)

The vision tower reproduces the HF Qwen2.5-VL ViT exactly (validated offline to
~1e-4 vs transformers): patch_embed(Conv3d==Linear) -> 2D-RoPE + window reorder
-> 32 blocks (block-diagonal full/window attention, gated-SiLU MLP) -> 2x2 patch
merger. Big linears are quantized to int4 (reusing webtorch.QuantizedLinear); the
LM half reuses webtorch.llm.CausalLM. M-RoPE (3D position ids) is applied in the
LM attention for image tokens.
"""
import json, math, time
import numpy as np
from . import _core as wt
from .llm import CausalLM, GenResult, BPETokenizer

xp = wt.xp


def _bf16_to_f32(raw):
    u16 = np.frombuffer(raw, np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


class _Lin:
    """Plain fp32 linear (y = x @ W.T + b), same call interface as QuantizedLinear.
    Used for correctness checks; real models use int4 QuantizedLinear."""
    def __init__(self, W_out_in, bias):
        self.W = wt.Tensor(np.ascontiguousarray(W_out_in.T.astype(np.float32)))  # (in,out)
        self.b = None if bias is None else wt.Tensor(np.asarray(bias, np.float32))

    def __call__(self, x):
        y = x.matmul(self.W)
        return y if self.b is None else y + self.b


def _quant_linear(W_out_in, bias, gs=128, bits=4):
    """fp32 (out,in) weight -> int4/int8 QuantizedLinear (pad to divide gs/pack)."""
    if bits is None:
        return _Lin(W_out_in, bias)
    W = np.ascontiguousarray(W_out_in.T)                     # (in,out)
    K, N = W.shape; per = 32 // bits
    kmul = gs if gs % per == 0 else gs * per
    Kp = K + (-K) % kmul; Np = N + (-N) % per
    if Kp != K or Np != N:
        W = np.pad(W, ((0, Kp - K), (0, Np - N)))
    qw, qz, sc, _, _ = wt._gptq_quantize(W, gs, bits)
    b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
    return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, gs, bits)


# ============================ image preprocessing ===========================
# Port of Qwen2VLImageProcessor: smart-resize to a multiple of (patch*merge),
# CLIP normalize, patchify into (grid_t*grid_h*grid_w, C*tp*p*p) rows.
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def _smart_resize(h, w, factor, min_pixels, max_pixels):
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        s = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / s / factor) * factor)
        wb = max(factor, math.floor(w / s / factor) * factor)
    elif hb * wb < min_pixels:
        s = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * s / factor) * factor
        wb = math.ceil(w * s / factor) * factor
    return hb, wb


def _resize_rgb(a, Hn, Wn):
    """Bilinear resize of an (H,W,3) float array. Used when the image arrives as pixels
    rather than as a PIL object -- a browser has already decoded it by then, and requiring
    PIL would mean shipping an image library to re-do work the page has done."""
    H0, W0 = a.shape[:2]
    if (H0, W0) == (Hn, Wn):
        return a
    yi = (np.arange(Hn, dtype=np.float32) + 0.5) * (H0 / Hn) - 0.5
    xi = (np.arange(Wn, dtype=np.float32) + 0.5) * (W0 / Wn) - 0.5
    y0 = np.clip(np.floor(yi), 0, H0 - 1).astype(np.int64); y1 = np.clip(y0 + 1, 0, H0 - 1)
    x0 = np.clip(np.floor(xi), 0, W0 - 1).astype(np.int64); x1 = np.clip(x0 + 1, 0, W0 - 1)
    wy = np.clip(yi - y0, 0, 1)[:, None, None]; wx = np.clip(xi - x0, 0, 1)[None, :, None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def preprocess_image(img_in, patch=14, merge=2, temporal_patch=2,
                     min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28):
    """RGB image -> (pixel_values (seq,1176) f32, grid_thw (1,3) int).

    Takes a PIL image or an (H,W,3) uint8/float array, so a caller that already has decoded
    pixels -- which is every browser caller -- does not need PIL at all."""
    if hasattr(img_in, "convert"):                            # PIL
        from PIL import Image
        img = img_in.convert("RGB")
        W0, H0 = img.size
        factor = patch * merge
        Hn, Wn = _smart_resize(H0, W0, factor, min_pixels, max_pixels)
        a = np.asarray(img.resize((Wn, Hn), Image.BICUBIC), np.float32) / 255.0
    else:
        arr = np.asarray(img_in)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        arr = arr[:, :, :3].astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        H0, W0 = arr.shape[:2]
        factor = patch * merge
        Hn, Wn = _smart_resize(H0, W0, factor, min_pixels, max_pixels)
        a = _resize_rgb(arr, Hn, Wn)
    a = (a - IMAGE_MEAN) / IMAGE_STD
    a = a.transpose(2, 0, 1)                                 # (3,H,W)
    a = a[None]                                              # (1,3,H,W) single frame
    a = np.repeat(a, temporal_patch, axis=0)                # temporal pad -> (tp,3,H,W)
    gt, gh, gw = 1, Hn // patch, Wn // patch
    ch = 3
    # patchify to (gt, gh, gw, C, tp, p, p) then flatten patch dims
    a = a.reshape(gt, temporal_patch, ch, gh // merge, merge, patch, gw // merge, merge, patch)
    a = a.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)              # (gt,gh/m,gw/m,m,m,C,tp,p,p)
    flat = a.reshape(gt * gh * gw, ch * temporal_patch * patch * patch).astype(np.float32)
    grid = np.array([[gt, gh, gw]], np.int64)
    return flat, grid


# ============================= vision tower =================================
class VLVisionTower:
    def __init__(self, cfg, eps=1e-6):
        self.C = cfg                                         # config dict
        self.eps = eps
        self.H = cfg["hidden_size"]; self.depth = cfg["depth"]
        self.heads = cfg["num_heads"]; self.HD = self.H // self.heads
        self.merge = cfg["spatial_merge_size"]; self.patch = cfg["patch_size"]
        self.win = cfg["window_size"]
        self.full = set(cfg["fullatt_block_indexes"])
        self.out_hidden = cfg["out_hidden_size"]
        self.mu = self.merge * self.merge

    def _rms(self, x, w):
        return (x / ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()) * w

    def _rot(self, x):
        d = self.HD
        return wt.cat([wt._slice_last(x, d // 2, d) * (-1.0), wt._slice_last(x, 0, d // 2)], axis=-1)

    # ---- index/rope helpers (numpy; run once at prefill) ----
    def _rot_pos_emb(self, grid):
        m = self.merge; pos = []
        for t, h, w in grid.tolist():
            hp = np.arange(h)[:, None].repeat(w, 1)
            hp = hp.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            wp = np.arange(w)[None, :].repeat(h, 0)
            wp = wp.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            pos.append(np.stack([hp, wp], -1).repeat(t, 0) if t > 1 else np.stack([hp, wp], -1))
        pos = np.concatenate(pos, 0)                         # (seq,2)
        mg = int(grid[:, 1:].max())
        dim = self.HD // 2                                   # rotary dim per axis (=40)
        inv = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
        full = np.arange(mg)[:, None] * inv[None, :]         # (mg, dim/2=20)
        rpe = full[pos].reshape(pos.shape[0], -1)            # (seq,40)
        return rpe.astype(np.float32)

    def _window_index(self, grid):
        m, ws = self.merge, self.win // self.merge // self.patch
        widx = []; cuw = [0]; wid = 0
        for t, h, w in grid.tolist():
            lh, lw = h // m, w // m
            idx = np.arange(t * lh * lw).reshape(t, lh, lw)
            ph = (-lh) % ws; pw = (-lw) % ws
            nh = (lh + ph) // ws; nw = (lw + pw) // ws
            ip = np.pad(idx, ((0, 0), (0, ph), (0, pw)), constant_values=-100)
            ip = ip.reshape(t, nh, ws, nw, ws).transpose(0, 1, 3, 2, 4).reshape(t, nh * nw, ws, ws)
            seqlens = (ip != -100).sum((2, 3)).reshape(-1)
            ipf = ip.reshape(-1); newi = ipf[ipf != -100]
            widx.append(newi + wid)
            cu = (np.cumsum(seqlens) * self.mu + cuw[-1])
            cuw.extend(cu.tolist()); wid += t * lh * lw
        widx = np.concatenate(widx)
        cuw = np.array(cuw, np.int64)
        cuw = cuw[np.concatenate([[True], np.diff(cuw) != 0])]   # unique_consecutive
        return widx, cuw

    def _blockdiag_mask(self, cu, seq):
        """additive (seq,seq) mask: 0 within a segment, -inf across."""
        seg = np.zeros(seq, np.int32)
        for i in range(len(cu) - 1):
            seg[cu[i]:cu[i + 1]] = i
        m = np.where(seg[:, None] == seg[None, :], 0.0, -1e9).astype(np.float32)
        return wt.Tensor(m.reshape(1, seq, seq))

    def _attn(self, x, lay, mask, cos_t, sin_t, seq):
        qkv = lay["qkv"](x)                                  # (seq, 3*H); layout [q|k|v]
        H, hd, nh = self.H, self.HD, self.heads
        q = wt._contig(wt._slice_last(qkv, 0, H).reshape(seq, nh, hd).permute(1, 0, 2))
        k = wt._contig(wt._slice_last(qkv, H, 2 * H).reshape(seq, nh, hd).permute(1, 0, 2))
        v = wt._contig(wt._slice_last(qkv, 2 * H, 3 * H).reshape(seq, nh, hd).permute(1, 0, 2))
        q = q * cos_t + self._rot(q) * sin_t
        k = k * cos_t + self._rot(k) * sin_t
        sc = wt.bmm(q, wt.transpose_last2(k)) * (self.HD ** -0.5) + mask
        a = wt.softmax(sc)
        o = wt.bmm(a, v).permute(1, 0, 2).reshape(seq, self.H)
        return lay["proj"](wt._contig(o))

    def forward(self, pixel_values, grid):
        seq = pixel_values.shape[0]; m = self.merge
        rpe = self._rot_pos_emb(grid)                        # (seq,40)
        widx, cuw = self._window_index(grid)
        # full-attn cu_seqlens = cumsum(h*w per frame)
        cu = np.concatenate([[0], np.cumsum(
            np.repeat((grid[:, 1] * grid[:, 2]), grid[:, 0].tolist()))]).astype(np.int64)
        # patch_embed (Conv3d kernel==input -> linear, no bias)
        h = wt.Tensor(pixel_values).matmul(self.patch_embed)  # (seq,1280)
        # reorder hidden + rope by window_index on merge-units
        order = widx                                         # (seq/mu,)
        h = wt._contig(h.reshape(seq // self.mu, self.mu, -1))
        h = wt.Tensor(h.numpy()[order].reshape(seq, -1))     # gather (host, once)
        rp = rpe.reshape(seq // self.mu, self.mu, -1)[order].reshape(seq, -1)  # (seq,40)
        emb = np.concatenate([rp, rp], -1)                   # (seq,80)
        cos_t = wt.Tensor(np.cos(emb).astype(np.float32))    # (seq,80) broadcast over heads
        sin_t = wt.Tensor(np.sin(emb).astype(np.float32))
        mask_full = self._blockdiag_mask(cu, seq)
        mask_win = self._blockdiag_mask(cuw, seq)
        for i, lay in enumerate(self.blocks):
            mask = mask_full if i in self.full else mask_win
            h = h + self._attn(self._rms(h, lay["norm1"]), lay, mask, cos_t, sin_t, seq)
            x = self._rms(h, lay["norm2"])
            h = h + lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))
        # merger: ln_q -> reshape(-1,H*mu) -> mlp0 -> gelu -> mlp2
        hm = self._rms(h, self.merger_ln).reshape(seq // self.mu, self.H * self.mu)
        hm = wt.gelu(self.merger0(hm))
        hm = self.merger2(hm)                                # (seq/mu, out_hidden)
        rev = np.argsort(order)
        return wt.Tensor(hm.numpy()[rev])                    # reverse window reorder

    # ---- build from a synchronous weight-getter (bits=None -> fp32 check) ----
    def _build(self, g, gs=128, bits=4):
        """g(name)->fp32 np array. name uses HF 'visual.*' keys."""
        self.gs = gs; self.bits = bits
        pe = g("visual.patch_embed.proj.weight")             # (H,3,2,14,14)
        self.patch_embed = wt.Tensor(pe.reshape(self.H, -1).T.copy())  # (1176,H)
        self.blocks = []
        for i in range(self.depth):
            p = "visual.blocks.%d." % i
            self.blocks.append({
                "norm1": wt.Tensor(g(p + "norm1.weight")),
                "norm2": wt.Tensor(g(p + "norm2.weight")),
                "qkv": _quant_linear(g(p + "attn.qkv.weight"), g(p + "attn.qkv.bias"), gs, bits),
                "proj": _quant_linear(g(p + "attn.proj.weight"), g(p + "attn.proj.bias"), gs, bits),
                "gate": _quant_linear(g(p + "mlp.gate_proj.weight"), g(p + "mlp.gate_proj.bias"), gs, bits),
                "up": _quant_linear(g(p + "mlp.up_proj.weight"), g(p + "mlp.up_proj.bias"), gs, bits),
                "down": _quant_linear(g(p + "mlp.down_proj.weight"), g(p + "mlp.down_proj.bias"), gs, bits),
            })
        self.merger_ln = wt.Tensor(g("visual.merger.ln_q.weight"))
        self.merger0 = _quant_linear(g("visual.merger.mlp.0.weight"), g("visual.merger.mlp.0.bias"), gs, bits)
        self.merger2 = _quant_linear(g("visual.merger.mlp.2.weight"), g("visual.merger.mlp.2.bias"), gs, bits)
        return self

    @classmethod
    def from_weights(cls, cfg, weights, gs=128, bits=4):
        """Build from an in-memory {name: np.ndarray} dict (offline npz or test)."""
        return cls(cfg)._build(lambda n: weights[n], gs, bits)

    @classmethod
    async def from_stream(cls, cfg, aget, gs=128, bits=4):
        """Build by streaming; aget(name) is async -> fp32 np array. Each big
        weight is quantized to int`bits` as it arrives (2.5GB never materializes)."""
        self = cls(cfg); self.gs = gs; self.bits = bits
        pe = await aget("visual.patch_embed.proj.weight")
        self.patch_embed = wt.Tensor(pe.reshape(self.H, -1).T.copy()); del pe
        self.blocks = []
        for i in range(self.depth):
            p = "visual.blocks.%d." % i
            async def q(nm):
                b = await aget(p + nm + ".bias") if True else None
                return _quant_linear(await aget(p + nm + ".weight"), b, gs, bits)
            self.blocks.append({
                "norm1": wt.Tensor(await aget(p + "norm1.weight")),
                "norm2": wt.Tensor(await aget(p + "norm2.weight")),
                "qkv": await q("attn.qkv"), "proj": await q("attn.proj"),
                "gate": await q("mlp.gate_proj"), "up": await q("mlp.up_proj"), "down": await q("mlp.down_proj"),
            })
        self.merger_ln = wt.Tensor(await aget("visual.merger.ln_q.weight"))
        self.merger0 = _quant_linear(await aget("visual.merger.mlp.0.weight"), await aget("visual.merger.mlp.0.bias"), gs, bits)
        self.merger2 = _quant_linear(await aget("visual.merger.mlp.2.weight"), await aget("visual.merger.mlp.2.bias"), gs, bits)
        return self


# ============================ VL causal LM =================================
class VLCausalLM(CausalLM):
    """Qwen2.5-VL: quantized ViT vision tower + Qwen2 LM with M-RoPE.

        model = await vl.VLCausalLM.from_qwen2_5_vl("/models/qwen2.5-vl-3b")
        out = model.generate("Describe this image.", image=pil_image)
    """
    async def _embed_stream(self, name, chunk=64 << 20):
        """Stream a (V,H) BF16/F16 embed to host f16 in row-blocks.

        Blocks are kept separate rather than assembled into one array: a 152k x 2048 fp16
        table is 594MB as a single object, which 32-bit WASM refuses however much memory the
        machine has. `_ChunkRows` indexes across the blocks and reads like an ndarray, which
        is what the engine wants from an embedding table anyway."""
        return await self._f16_chunked(name, chunk_bytes=chunk)

    @classmethod
    async def from_qwen2_5_vl(cls, base, lmax=None, bits=4):
        self = cls(base); self.lmax = lmax; self.bits = bits; self.gs = 128
        self._shard_hdr = {}
        from . import webio
        cfg = await webio.read_json(self.base + "config.json")
        tc = cfg.get("text_config", cfg); vc = cfg["vision_config"]
        self.H = tc["hidden_size"]; self.L = tc["num_hidden_layers"]
        self.NH = tc["num_attention_heads"]; self.NKV = tc["num_key_value_heads"]
        self.HD = self.H // self.NH; self.VOCAB = tc["vocab_size"]
        self.eps = tc["rms_norm_eps"]
        self.theta = tc.get("rope_theta") or tc["rope_scaling"].get("rope_theta", 1000000.0)
        self.mrope = tc["rope_scaling"]["mrope_section"]
        # An image costs hundreds of prompt tokens, so a fixed small context is the difference
        # between working and refusing. Take it from the model, capped by KV memory, exactly
        # as the text path does.
        self.lmax = self._auto_lmax(lmax, tc.get("max_position_embeddings"))
        self.merge = vc["spatial_merge_size"]
        self.IMG_TOK = cfg["image_token_id"]; self.VSTART = cfg["vision_start_token_id"]
        self.VEND = cfg["vision_end_token_id"]
        vcfg = dict(hidden_size=vc["hidden_size"], depth=vc["depth"], num_heads=vc["num_heads"],
                    spatial_merge_size=vc["spatial_merge_size"], patch_size=vc["patch_size"],
                    temporal_patch_size=vc["temporal_patch_size"], window_size=vc["window_size"],
                    fullatt_block_indexes=vc["fullatt_block_indexes"],
                    out_hidden_size=vc["out_hidden_size"], in_channels=vc.get("in_channels", 3))

        # Build the tokenizer the same way every other HF model does. Reading tokenizer.json's
        # `model.vocab` directly misses `added_tokens`, which is where a Qwen model keeps
        # <|im_end|> and the rest -- without them the end-of-turn id was wrong (the probe
        # matched <s>/</s>, which this family does not use) and generation never stopped
        # where it should. The shared path also picks up the chat template and eos ids.
        self.tok = await self._hf_tokenizer(cfg)
        self.tok.SPECIALS.update({"<|vision_start|>": self.VSTART, "<|vision_end|>": self.VEND,
                                  "<|image_pad|>": self.IMG_TOK})
        for t, i in self.tok.SPECIALS.items():
            self.tok.dec[i] = t

        imeta = await webio.read_json(self.base + "model.safetensors.index.json")
        self._idx = imeta.get("weight_map")
        if not self._idx:
            h, _ = await self._hdr("model.safetensors"); self._idx = {k: "model.safetensors" for k in h}

        t0 = time.perf_counter()
        self.embed = await self._embed_stream("model.embed_tokens.weight")
        self.head = self._quantize_head(self.embed)                 # tied
        self.layers = []
        for i in range(self.L):
            p = "model.layers.%d." % i
            async def qb(nm):
                b = await self._np(p + nm + ".bias") if (p + nm + ".bias") in self._idx else None
                return self._gquant(await self._np(p + nm + ".weight"), b)
            self.layers.append({
                "in_ln": wt.Tensor(await self._np(p + "input_layernorm.weight")),
                "post_ln": wt.Tensor(await self._np(p + "post_attention_layernorm.weight")),
                "q": await qb("self_attn.q_proj"), "k": await qb("self_attn.k_proj"),
                "v": await qb("self_attn.v_proj"), "o": await qb("self_attn.o_proj"),
                "gate": await qb("mlp.gate_proj"), "up": await qb("mlp.up_proj"), "down": await qb("mlp.down_proj")})
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.vision = await VLVisionTower.from_stream(
            vcfg, lambda n: self._np(n), self.gs, bits)
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._init_state()
        return self

    # ---- M-RoPE ----
    def _mrope_cos_sin(self, pos3):
        HD = self.HD
        inv = 1.0 / (self.theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        freqs = pos3[:, :, None] * inv[None, None, :]           # (3,seq,HD/2)
        emb = np.concatenate([freqs, freqs], -1)                # (3,seq,HD)
        cos = np.cos(emb); sin = np.sin(emb)
        sec = list(self.mrope) * 2                              # [16,24,24,16,24,24]
        cc = []; ss = []; st = 0
        for i, w in enumerate(sec):
            cc.append(cos[i % 3, :, st:st + w]); ss.append(sin[i % 3, :, st:st + w]); st += w
        return (np.concatenate(cc, -1).astype(np.float32),
                np.concatenate(ss, -1).astype(np.float32))

    def _rope_index(self, ids, grid):
        m = self.merge; ids = np.asarray(ids); seq = len(ids); cur = 0; out = []; idx = 0
        while idx < seq:
            if ids[idx] == self.IMG_TOK:
                gt, gh, gw = [int(x) for x in grid[0]]; lh, lw = gh // m, gw // m; n = lh * lw
                t = np.full(n, cur, np.int64)
                h = cur + np.repeat(np.arange(lh), lw)
                w = cur + np.tile(np.arange(lw), lh)
                out.append(np.stack([t, h, w])); idx += n; cur += max(lh, lw)
            else:
                out.append(np.array([[cur], [cur], [cur]], np.int64)); idx += 1; cur += 1
        pos = np.concatenate(out, 1)
        return pos, int(pos.max()) + 1

    def _encode_vl(self, user, n_img, system="You are a helpful assistant."):
        S = self.tok.SPECIALS; E = self.tok.encode
        seq = [S["<|im_start|>"]] + E("system\n" + system) + [S["<|im_end|>"]] + E("\n")
        seq += [S["<|im_start|>"]] + E("user\n")
        seq += [self.VSTART] + [self.IMG_TOK] * n_img + [self.VEND] + E(user)
        seq += [S["<|im_end|>"]] + E("\n") + [S["<|im_start|>"]] + E("assistant\n")
        return seq

    # ---- multimodal prefill ----
    def _vl_prefill(self, h, cos, sin):
        T = h.shape[0]; H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.lmax
        cos_t, sin_t = wt.Tensor(cos), wt.Tensor(sin)
        m = np.triu(np.full((T, LMAX), -1e9, np.float32), 1); m[:, T:] = -1e9
        mask = wt.Tensor(m.reshape(1, T, LMAX)); sc = 1.0 / math.sqrt(HD)
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(T, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
            self.Kc[i].data = wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, T, NKV, HD, LMAX)
            self.Vc[i].data = wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, T, NKV, HD, LMAX)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], mask, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(T, H))
            x = self._rms(h, lay["post_ln"])
            h = h + lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))

    def _set_inputs_vl(self, token, kv_pos, rope_pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.lmax
        self.h_in.data.buffer.set_data(self.embed[token].astype(np.float32))
        c, s = self._rope_np(rope_pos)                          # text decode = 1D rope
        self.cos_b.data.buffer.set_data(c.reshape(-1)); self.sin_b.data.buffer.set_data(s.reshape(-1))
        m = np.zeros((1, 1, LMAX), np.float32); m[0, 0, kv_pos + 1:] = -1e9
        self.mask_b.data.buffer.set_data(m)
        self.ctl.buffer.set_data(np.array([kv_pos, 1, NKV, HD, LMAX], np.int32))

    kind = "multimodal"

    def generate(self, prompt, image=None, max_new=64, system="You are a helpful assistant.",
                 max_pixels=384 * 28 * 28, media=None, **kw):
        # `media=` is what the generic multimodal surface calls it (MultimodalLM.generate),
        # and a caller should not have to know which of the two it is holding.
        if image is None:
            image = media
        if isinstance(image, (list, tuple)):
            image = image[0] if image else None
        # max_pixels caps the image resolution: vision full-attention is O(seq^2)
        # (seq = pixels/patch^2), so large images blow the WASM heap. 384*28*28 ->
        # ~1536 patches, a ~150MB score matrix. Raise it if you have headroom.
        if image is None:
            return super().generate(prompt, max_new=max_new, system=system, **kw)
        eot = self.tok.eot
        # Same sampling surface as every other model: temperature, penalties, stop strings and
        # constraints all come through `_pick`. This path used to take the argmax outright, so
        # every one of those parameters was silently ignored once an image was attached.
        pv, grid = preprocess_image(image, max_pixels=max_pixels)
        img_embeds = self.vision.forward(pv, grid).numpy()      # (n_img, H)
        n_img = img_embeds.shape[0]
        ids = self._encode_vl(prompt, n_img, system); P = len(ids)
        # `max_new=None` means "as much as the model allows", the same as everywhere else --
        # the page sends None when the user sets no limit. A prompt that does not fit is a
        # hard error here rather than something to trim from the front: trimming would cut
        # the image placeholder tokens, which must stay in step with the image embeddings.
        if P >= self.lmax:
            raise ValueError("prompt plus image needs %d tokens but the context is %d — "
                             "load with a larger lmax, or send a smaller image" % (P, self.lmax))
        if max_new is None:
            d = getattr(self, "gen_defaults", {}) or {}
            max_new = int(d.get("max_new_tokens") or 0) or (self.lmax - P)
        max_new = max(1, min(int(max_new), self.lmax - P))
        self._set_sampling(prompt_ids=ids, **kw)
        pos3, next_pos = self._rope_index(ids, grid)
        cos, sin = self._mrope_cos_sin(pos3)                    # (P,HD)
        # embed tokens, splice image embeds at IMG_TOK positions
        emb = self.embed[np.asarray(ids, np.int64)].astype(np.float32)
        emb[np.asarray(ids) == self.IMG_TOK] = img_embeds
        if not self._gpu:
            return self._generate_webgl(ids, emb, cos, sin, next_pos, P, eot, max_new)

        for c in self.Kc:
            c.data[:] = 0.0
        for v in self.Vc:
            v.data[:] = 0.0
        t0 = time.perf_counter()
        g0 = self._vl_prefill(wt.Tensor(emb), cos, sin)
        ttft = time.perf_counter() - t0
        plat = wt._adam_kernel["platform"]
        self._set_inputs_vl(g0, P, next_pos)
        plat.beginCapture("vldecode"); lt = self._decode_fwd(); lt.numpy(); plat.endCapture()
        self.capture_ready = True
        gen = [g0]; nxt = g0; kvp = P; rp = next_pos; steps = 0; td = time.perf_counter()
        while len(gen) < max_new:
            self._set_inputs_vl(nxt, kvp, rp); plat.replay("vldecode")
            nxt = self._pick(lt.numpy()[0]); kvp += 1; rp += 1; steps += 1
            if nxt == eot:
                break
            gen.append(nxt)
            if self._stop_now():
                break
        dec = time.perf_counter() - td
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), round(steps / max(dec, 1e-9), 2))

    def _generate_webgl(self, ids, emb, cos, sin, next_pos, P, eot, max_new):
        cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
        t0 = time.perf_counter()
        g0 = self._fresh_fwd(emb, cos, sin, 0, cache)
        ttft = time.perf_counter() - t0
        gen = [g0]; nxt = g0; pos = P; rp = next_pos; steps = 0; td = time.perf_counter()
        while len(gen) < max_new:
            c, s = self._rope_np(rp)
            e = self.embed[np.asarray([nxt], np.int64)].astype(np.float32)
            nxt = self._fresh_fwd(e, c, s, pos, cache); pos += 1; rp += 1; steps += 1
            if nxt == eot:
                break
            gen.append(nxt)
        dec = time.perf_counter() - td
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), round(steps / max(dec, 1e-9), 2))

    def _fresh_fwd(self, emb, cos, sin, pos, cache):
        T = emb.shape[0]; H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        cos_t, sin_t = wt.Tensor(cos), wt.Tensor(sin); h = wt.Tensor(emb); sc = 1.0 / math.sqrt(HD)
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(T, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
            o = cache.attn(i, q, k, v, pos, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(T, H))
            x = self._rms(h, lay["post_ln"])
            h = h + lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))
