"""webtorch.llm -- LLM inference engine with graph-capture acceleration.

High-level API for running quantized (AutoGPTQ int4) LLMs in the browser on top
of webtorch/WgPy, with a WebGPU graph-capture fast path for decode.

    import llm
    model = await llm.GPTQModel.load("/models/qwen7b-gptq")
    out = model.generate("Give me three tips for staying focused.")
    print(out.text, "|", out.decode_tok_s, "tok/s")

What it does:
  * streams an AutoGPTQ safetensors model (single- or multi-shard) tensor-by-
    tensor via HTTP Range, quantizes embed/lm_head to int4 on the GPU (so the
    >1GB vocab matrices never sit in the WASM heap);
  * ChatML prompt formatting + a byte-level BPE tokenizer;
  * a fixed-capacity in-place KV cache written by a scatter kernel, so the whole
    decode step is shape-static and can be graph-captured;
  * WebGPU: captures the decode step once and replays it per token (removes the
    per-op Python dispatch overhead -- ~20x faster decode). WebGL falls back to a
    correct, un-captured growing-cache path (WebGL can't do in-place capture).

Architecturally identical for any qwen2 / llama AutoGPTQ model; tested on
Qwen2.5-3B/7B-Instruct-GPTQ-Int4.
"""
import json, math, time
import numpy as np
from . import _core as wt

xp = wt.xp


# ----------------------------- tokenizer ------------------------------------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    """GPT-2/Qwen-style byte-level BPE (vocab.json + merges.txt)."""
    SPECIALS = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}

    def __init__(self, vocab, merges):
        self.enc = vocab
        self.dec = {i: t for t, i in vocab.items()}
        self.b2u = _bytes_to_unicode(); self.u2b = {v: k for k, v in self.b2u.items()}
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        try:
            import regex as _re
            pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        except Exception:
            import re as _re
            pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
        self.re = _re; self.pat = _re.compile(pat)
        for t, i in self.SPECIALS.items():
            self.dec[i] = t

    def _bpe(self, tok):
        word = list(tok)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]): i for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            i = pairs[best]; word = word[:i] + [best[0] + best[1]] + word[i + 2:]
        return word

    def encode(self, text):
        ids = []
        for chunk in self.re.findall(self.pat, text):
            s = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            ids.extend(self.enc[p] for p in self._bpe(s))
        return ids

    def encode_chat(self, user, system="You are a helpful assistant."):
        seq = []
        for role, content in [("system", system), ("user", user)]:
            seq += [self.SPECIALS["<|im_start|>"]] + self.encode(role + "\n" + content)
            seq += [self.SPECIALS["<|im_end|>"]] + self.encode("\n")
        return seq + [self.SPECIALS["<|im_start|>"]] + self.encode("assistant\n")

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.dec.get(int(i), "")
            if t in self.SPECIALS:
                continue
            for ch in t:
                buf.append(self.u2b.get(ch, 32))
        return buf.decode("utf-8", "replace")


# ----------------------------- result --------------------------------------
class GenResult:
    def __init__(self, text, tokens, ttft_s, decode_tok_s):
        self.text = text; self.tokens = tokens
        self.ttft_s = ttft_s; self.decode_tok_s = decode_tok_s

    def __repr__(self):
        return "GenResult(ttft=%.2fs, %.1f tok/s)\n%s" % (self.ttft_s, self.decode_tok_s, self.text)


# ----------------------------- model ---------------------------------------
class CausalLM:
    """Quantized causal LM (qwen2 family) with capture-accelerated decode.

    Load from either format -- the inference engine (KV cache, capture/replay,
    generate) is shared; only loading differs:
        CausalLM.from_gptq("/models/qwen7b-gptq")     # AutoGPTQ int4 safetensors
        CausalLM.from_gguf("/models/qwen3b-gguf/model.gguf")   # llama.cpp GGUF
    """

    def __init__(self, base):
        self.base = (base.rstrip("/") + "/") if base else ""
        self.capture_ready = False
        self.rope_style = "hf"          # qwen2 = HF rotate_half (both GPTQ and GGUF)
        self.gs = 128; self.bits = 4    # int4 kernel params (GGUF is requantized to this)

    # ---- range helpers (all reads go through the global IO callback) ----
    async def _rng(self, fn, a, b):
        from . import webio
        return await webio.io_read(self.base + fn, a, b - a + 1)     # a..b inclusive

    async def _hdr(self, shard):
        if shard not in self._shard_hdr:
            n = int.from_bytes(await self._rng(shard, 0, 7), "little")
            h = json.loads((await self._rng(shard, 8, 8 + n - 1)).decode())
            h.pop("__metadata__", None)
            self._shard_hdr[shard] = (h, 8 + n)
        return self._shard_hdr[shard]

    async def _np(self, name):
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]
        raw = await self._rng(shard, base + a, base + z - 1)
        dt = info["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, np.float16).astype(np.float32)
        elif dt == "I32":
            arr = np.frombuffer(raw, np.int32)
        else:
            arr = np.frombuffer(raw, np.float32)
        return arr.reshape(info["shape"])

    async def _qlin(self, prefix):
        b = await self._np(prefix + ".bias") if (prefix + ".bias") in self._idx else None
        return wt.QuantizedLinear.from_autogptq(
            await self._np(prefix + ".qweight"), await self._np(prefix + ".qzeros"),
            await self._np(prefix + ".scales"), b, self.gs, self.bits)

    async def _f16_chunked(self, name, chunk_bytes=64 << 20):
        """Stream a big 2-D weight (embed / lm_head) in row blocks -> host fp16, so the full
        fp32 tensor never materializes. dtype-aware: handles F16 / BF16 / F32 sources."""
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]; V, Hh = info["shape"]; dt = info["dtype"]
        esz = 2 if dt in ("F16", "BF16") else 4; row = Hh * esz
        out = np.empty((V, Hh), np.float16); rpc = max(1, chunk_bytes // row)
        for v0 in range(0, V, rpc):
            v1 = min(V, v0 + rpc)
            raw = await self._rng(shard, base + a + v0 * row, base + a + v1 * row - 1)
            if dt == "BF16":
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, np.float16)
            else:
                arr = np.frombuffer(raw, np.float32)
            out[v0:v1] = arr.reshape(v1 - v0, Hh).astype(np.float16)
        return out

    def _quantize_head(self, WVH, blk=4096):
        V, Hh = WVH.shape; blocks = []
        for v0 in range(0, V, blk):
            v1 = min(V, v0 + blk)
            WbT = np.ascontiguousarray(WVH[v0:v1].astype(np.float32).T)
            qw, qz, sc, _, _ = wt._gptq_quantize(WbT, self.gs, self.bits)
            blocks.append(wt.QuantizedLinear(qw, qz, sc, np.zeros((v1 - v0,), np.float32),
                                             Hh, v1 - v0, Hh, v1 - v0, self.gs, self.bits)); del WbT
        return blocks

    async def _head_int4(self, name, V, Hh, blk=4096):
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]; row = Hh * 2; blocks = []
        for v0 in range(0, V, blk):
            v1 = min(V, v0 + blk)
            raw = await self._rng(shard, base + a + v0 * row, base + a + v1 * row - 1)
            Wb = np.frombuffer(raw, np.float16).astype(np.float32).reshape(v1 - v0, Hh)
            blocks.extend(self._quantize_head(Wb, blk=blk)); del Wb
        return blocks

    @classmethod
    async def from_gptq(cls, base, lmax=320):
        """Stream + build from a served AutoGPTQ int4 dir (e.g. '/models/qwen7b-gptq')."""
        from . import webio
        self = cls(base)
        self.lmax = lmax
        self._shard_hdr = {}
        cfg = await webio.read_json(self.base + "config.json")
        self.H = cfg["hidden_size"]; self.L = cfg["num_hidden_layers"]
        self.NH = cfg["num_attention_heads"]; self.NKV = cfg["num_key_value_heads"]
        self.HD = self.H // self.NH; self.VOCAB = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]; self.theta = cfg["rope_theta"]
        self.gs = cfg["quantization_config"]["group_size"]; self.bits = cfg["quantization_config"]["bits"]
        tie = cfg["tie_word_embeddings"]

        vocab = await webio.read_json(self.base + "vocab.json")
        merges = (await webio.read_text(self.base + "merges.txt")).split("\n")
        self.tok = BPETokenizer(vocab, [m for m in merges[1:] if m and not m.startswith("#")])

        imeta = await webio.read_json(self.base + "model.safetensors.index.json")
        self._idx = imeta.get("weight_map")
        if not self._idx:
            h, _ = await self._hdr("model.safetensors")
            self._idx = {k: "model.safetensors" for k in h}

        t0 = time.perf_counter()
        if not tie:
            self.head = await self._head_int4("lm_head.weight", self.VOCAB, self.H)
            self.embed = await self._f16_chunked("model.embed_tokens.weight")
        else:
            self.embed = await self._f16_chunked("model.embed_tokens.weight")
            self.head = self._quantize_head(self.embed)
        self.layers = []
        for i in range(self.L):
            p = "model.layers.%d." % i
            self.layers.append({
                "in_ln": wt.Tensor(await self._np(p + "input_layernorm.weight")),
                "post_ln": wt.Tensor(await self._np(p + "post_attention_layernorm.weight")),
                "q": await self._qlin(p + "self_attn.q_proj"), "k": await self._qlin(p + "self_attn.k_proj"),
                "v": await self._qlin(p + "self_attn.v_proj"), "o": await self._qlin(p + "self_attn.o_proj"),
                "gate": await self._qlin(p + "mlp.gate_proj"), "up": await self._qlin(p + "mlp.up_proj"),
                "down": await self._qlin(p + "mlp.down_proj")})
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()        # webgpu -> capture path available
        self._init_state()
        return self

    # ---- GGUF (llama.cpp) loading: dequant -> requant to int4 -> same engine ----
    async def _grng(self, a, b):
        from . import webio
        return await webio.io_read(self._gguf, a, b - a + 1)     # a..b inclusive

    async def _gload(self, name):
        """Dequant a GGUF tensor to fp32, shape (out,in) [ggml dims are reversed]."""
        from . import ggufload as G
        t = self._ginfo[name]; n = 1
        for d in t["dims"]:
            n *= int(d)
        nb = G.tensor_nbytes(t["type"], n)
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb - 1)
        return G.dequant(t["type"], raw, n).reshape(tuple(reversed(t["dims"])))

    def _gquant(self, W_out_in, bias=None):
        """fp32 (out,in) weight -> int4 QuantizedLinear (pad to divide gs / pack)."""
        W = np.ascontiguousarray(W_out_in.T)                 # (in, out)
        K, N = W.shape; per = 32 // self.bits
        kmul = self.gs if self.gs % per == 0 else self.gs * per
        Kp = K + (-K) % kmul; Np = N + (-N) % per
        if Kp != K or Np != N:
            W = np.pad(W, ((0, Kp - K), (0, Np - N)))
        qw, qz, sc, _, _ = wt._gptq_quantize(W, self.gs, self.bits)
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, self.gs, self.bits)

    @classmethod
    async def from_gguf(cls, url, lmax=320, bits=4):
        """Load a llama.cpp GGUF (qwen2 arch), dequantizing + requantizing weights
        to int`bits` (4 or 8) so they run on the same capture-accelerated engine.
        `url` is the served .gguf file, e.g. '/models/qwen3b-gguf/model.gguf'."""
        from . import ggufload as G
        self = cls(None); self.lmax = lmax; self._gguf = url
        self.bits = bits                                      # int4 or int8 (same kernels/optims)
        size = 12 << 20
        while True:
            buf = await self._grng(0, size - 1)
            try:
                _, meta, infos, ds = G.parse_header(buf); break
            except EOFError:
                size <<= 1
                if size > (128 << 20):
                    raise
        arch = meta.get("general.architecture")
        if arch != "qwen2":
            raise NotImplementedError("from_gguf supports qwen2 GGUF (got %r)" % arch)
        self.H = meta["qwen2.embedding_length"]; self.L = meta["qwen2.block_count"]
        self.NH = meta["qwen2.attention.head_count"]; self.NKV = meta["qwen2.attention.head_count_kv"]
        self.HD = self.H // self.NH; self.VOCAB = len(meta["tokenizer.ggml.tokens"])
        self.eps = meta["qwen2.attention.layer_norm_rms_epsilon"]
        self.theta = meta["qwen2.rope.freq_base"]
        self.tok = BPETokenizer({t: i for i, t in enumerate(meta["tokenizer.ggml.tokens"])},
                                meta["tokenizer.ggml.merges"])
        self._ginfo = {t["name"]: t for t in infos}; self._gds = ds

        t0 = time.perf_counter()
        # token_embd -> host f16 (streamed in row-blocks so the full 1.2GB fp32
        # never materializes); if tied, build the int4 head from the same blocks.
        et = self._ginfo["token_embd.weight"]
        H = int(et["dims"][0]); V = int(et["dims"][1])
        erow = G.tensor_nbytes(et["type"], H); eoff = self._gds + et["offset"]
        self.embed = np.empty((V, H), np.float16)
        tied = "output.weight" not in self._ginfo
        self.head = [] if tied else None
        for v0 in range(0, V, 4096):
            v1 = min(V, v0 + 4096)
            raw = await self._grng(eoff + v0 * erow, eoff + v1 * erow - 1)
            fb = G.dequant(et["type"], raw, (v1 - v0) * H).reshape(v1 - v0, H)
            self.embed[v0:v1] = fb.astype(np.float16)
            if tied:
                self.head.append(self._gquant(fb))            # (blk,H) = (out,in)
            del fb
        if not tied:                                          # separate lm_head, also streamed
            ot = self._ginfo["output.weight"]
            orow = G.tensor_nbytes(ot["type"], H); ooff = self._gds + ot["offset"]
            self.head = []
            for v0 in range(0, V, 4096):
                v1 = min(V, v0 + 4096)
                raw = await self._grng(ooff + v0 * orow, ooff + v1 * orow - 1)
                fb = G.dequant(ot["type"], raw, (v1 - v0) * H).reshape(v1 - v0, H)
                self.head.append(self._gquant(fb)); del fb
        self.layers = []
        for i in range(self.L):
            p = "blk.%d." % i
            async def qb(nm):                                  # weight + optional bias -> QuantizedLinear
                b = (await self._gload(p + nm + ".bias")) if (p + nm + ".bias") in self._ginfo else None
                return self._gquant(await self._gload(p + nm + ".weight"), b)
            self.layers.append({
                "in_ln": wt.Tensor(await self._gload(p + "attn_norm.weight")),
                "post_ln": wt.Tensor(await self._gload(p + "ffn_norm.weight")),
                "q": await qb("attn_q"), "k": await qb("attn_k"), "v": await qb("attn_v"),
                "o": await qb("attn_output"),
                "gate": await qb("ffn_gate"), "up": await qb("ffn_up"), "down": await qb("ffn_down")})
        self.final_norm = wt.Tensor(await self._gload("output_norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._init_state()
        return self

    # ---- fp16 (plain HF safetensors) loading: run UNquantized, or quantize on load ----
    def _mklin(self, W_out_in, bias=None):
        """One linear at the chosen precision: fp16 (UnquantizedLinear) or int4/int8."""
        if self._qbits:
            return self._gquant(W_out_in, bias)            # quantize-on-load -> QuantizedLinear
        return wt.UnquantizedLinear(W_out_in, bias)        # fp16 weights, fp32 compute

    async def _lin(self, prefix):
        b = await self._np(prefix + ".bias") if (prefix + ".bias") in self._idx else None
        return self._mklin(await self._np(prefix + ".weight"), b)

    @classmethod
    async def from_fp16(cls, base, lmax=320, quantize=None):
        """Load a plain fp16/bf16 HF safetensors dir (no `quantization_config`). Runs the
        model UNquantized by default (fp16 weights, fp32 compute); pass `quantize=4` or `8`
        to instead quantize every linear to int on load (same served fp16 model → int
        inference). Same engine (KV cache, capture/replay, generate) as the int loaders."""
        from . import webio
        self = cls(base); self.lmax = lmax; self._shard_hdr = {}
        self._qbits = int(quantize) if quantize else 0
        if self._qbits:
            self.bits = self._qbits                        # gs keeps the default 128
        cfg = await webio.read_json(self.base + "config.json")
        self.H = cfg["hidden_size"]; self.L = cfg["num_hidden_layers"]
        self.NH = cfg["num_attention_heads"]; self.NKV = cfg["num_key_value_heads"]
        self.HD = self.H // self.NH; self.VOCAB = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]; self.theta = cfg["rope_theta"]
        tie = cfg.get("tie_word_embeddings", False)

        vocab = await webio.read_json(self.base + "vocab.json")
        merges = (await webio.read_text(self.base + "merges.txt")).split("\n")
        self.tok = BPETokenizer(vocab, [m for m in merges[1:] if m and not m.startswith("#")])

        try:                                                   # sharded models have an index
            imeta = await webio.read_json(self.base + "model.safetensors.index.json")
            self._idx = imeta.get("weight_map")
        except Exception:                                      # single-file model: no index.json
            self._idx = None
        if not self._idx:
            h, _ = await self._hdr("model.safetensors")
            self._idx = {k: "model.safetensors" for k in h}

        t0 = time.perf_counter()
        self.embed = await self._f16_chunked("model.embed_tokens.weight")     # (V,H) fp16
        headW = self.embed if tie else await self._f16_chunked("lm_head.weight")
        self.head = self._quantize_head(headW) if self._qbits else [wt.UnquantizedLinear(headW)]
        self.layers = []
        for i in range(self.L):
            p = "model.layers.%d." % i
            self.layers.append({
                "in_ln": wt.Tensor(await self._np(p + "input_layernorm.weight")),
                "post_ln": wt.Tensor(await self._np(p + "post_attention_layernorm.weight")),
                "q": await self._lin(p + "self_attn.q_proj"), "k": await self._lin(p + "self_attn.k_proj"),
                "v": await self._lin(p + "self_attn.v_proj"), "o": await self._lin(p + "self_attn.o_proj"),
                "gate": await self._lin(p + "mlp.gate_proj"), "up": await self._lin(p + "mlp.up_proj"),
                "down": await self._lin(p + "mlp.down_proj")})
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._init_state()
        return self

    # ---- math ----
    def _rms(self, x, w):
        return (x / ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()) * w

    def _rot(self, x):
        hd = self.HD
        return wt.cat([wt._slice_last(x, hd // 2, hd) * (-1.0), wt._slice_last(x, 0, hd // 2)], axis=-1)

    def _rope_np(self, pos, T=1):
        inv = 1.0 / (self.theta ** (np.arange(0, self.HD, 2, dtype=np.float64) / self.HD))
        ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], -1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def _head_argmax(self, hlast):
        return int(wt.cat([blk(hlast) for blk in self.head], axis=-1).numpy()[0].argmax())

    def _init_state(self):
        L, NKV, HD, LMAX, H = self.L, self.NKV, self.HD, self.lmax, self.H
        if self._gpu:                                  # capture path: fixed scatter cache + persistent inputs
            self.Kc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(L)]
            self.Vc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(L)]
            self.h_in = wt.Tensor(np.zeros((1, H), np.float32))
            self.cos_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.sin_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.mask_b = wt.Tensor(np.zeros((1, 1, LMAX), np.float32))
            self.ctl = xp.asarray(np.array([0, 1, NKV, HD, LMAX], np.int32))

    # ---- prefill (fresh, fills KV 0..P-1) ----
    def _prefill(self, ids):
        T = len(ids); H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.lmax
        c, s = self._rope_np(0, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        m = np.triu(np.full((T, LMAX), -1e9, np.float32), 1); m[:, T:] = -1e9
        mask = wt.Tensor(m.reshape(1, T, LMAX))
        h = wt.Tensor(self.embed[np.asarray(ids, np.int64)].astype(np.float32))
        sc = 1.0 / math.sqrt(HD)
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(T, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(T, NKV, HD).permute(1, 0, 2)
            q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
            wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, T, NKV, HD, LMAX)
            wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, T, NKV, HD, LMAX)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], mask, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(T, H))
            x = self._rms(h, lay["post_ln"])
            h = h + lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))

    def _set_inputs(self, token, pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.lmax
        self.h_in.data.buffer.set_data(self.embed[token].astype(np.float32))
        c, s = self._rope_np(pos)
        self.cos_b.data.buffer.set_data(c.reshape(-1)); self.sin_b.data.buffer.set_data(s.reshape(-1))
        m = np.zeros((1, 1, LMAX), np.float32); m[0, 0, pos + 1:] = -1e9
        self.mask_b.data.buffer.set_data(m)
        self.ctl.buffer.set_data(np.array([pos, 1, NKV, HD, LMAX], np.int32))

    def _decode_fwd(self):
        H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.lmax
        sc = 1.0 / math.sqrt(HD); h = self.h_in
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            q = lay["q"](x).reshape(1, NH, HD).permute(1, 0, 2)
            k = lay["k"](x).reshape(1, NKV, HD).permute(1, 0, 2)
            v = lay["v"](x).reshape(1, NKV, HD).permute(1, 0, 2)
            q = q * self.cos_b + self._rot(q) * self.sin_b
            k = k * self.cos_b + self._rot(k) * self.sin_b
            wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], self.mask_b, scale=sc)
            h = h + lay["o"](o.permute(1, 0, 2).reshape(1, H))
            x = self._rms(h, lay["post_ln"])
            h = h + lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))
        return wt.cat([blk(self._rms(h, self.final_norm)) for blk in self.head], axis=-1)

    # ---- WebGL path: growing KVCache, fresh forward (no in-place capture) ----
    def _kv_forward(self, ids, pos, cache):
        T = len(ids); H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        c, s = self._rope_np(pos, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        h = wt.Tensor(self.embed[np.asarray(ids, np.int64)].astype(np.float32))
        sc = 1.0 / math.sqrt(HD)
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

    def stream(self, prompt, max_new=48, system="You are a helpful assistant."):
        """Streaming decode: yield each new token's text as it is produced (render live,
        bounded memory). WebGPU replays a captured step per token; WebGL grows a cache."""
        eot = self.tok.SPECIALS["<|im_end|>"]
        ids = self.tok.encode_chat(prompt, system); P = len(ids)
        if not self._gpu:
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            nxt = self._kv_forward(ids, 0, cache); pos = P; n = 0
            while n < max_new and nxt != eot:
                yield self.tok.decode([nxt]); n += 1
                nxt = self._kv_forward([nxt], pos, cache); pos += 1
            return
        for c in self.Kc: c.data[:] = 0.0
        for v in self.Vc: v.data[:] = 0.0
        g0 = self._prefill(ids); plat = wt._adam_kernel["platform"]
        self._set_inputs(g0, P)
        plat.beginCapture("decode"); logits_t = self._decode_fwd(); logits_t.numpy(); plat.endCapture()
        self.capture_ready = True; nxt = g0; pos = P; n = 0
        while n < max_new and nxt != eot:
            yield self.tok.decode([nxt]); n += 1
            self._set_inputs(nxt, pos); plat.replay("decode")
            nxt = int(logits_t.numpy()[0].argmax()); pos += 1

    def generate(self, prompt, max_new=48, system="You are a helpful assistant."):
        """ChatML prompt -> greedy decode -> GenResult. WebGPU replays a captured
        decode step per token (~20x); WebGL uses a correct growing-cache forward.
        For live token-by-token output use `stream(...)`."""
        eot = self.tok.SPECIALS["<|im_end|>"]
        ids = self.tok.encode_chat(prompt, system)
        P = len(ids)

        if not self._gpu:                              # WebGL fallback
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            t0 = time.perf_counter(); g0 = self._kv_forward(ids, 0, cache)
            ttft = time.perf_counter() - t0
            gen = [g0]; nxt = g0; pos = P; steps = 0; td = time.perf_counter()
            while len(gen) < max_new:
                nxt = self._kv_forward([nxt], pos, cache); pos += 1; steps += 1
                if nxt == eot:
                    break
                gen.append(nxt)
            dec = time.perf_counter() - td
            return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                             round(ttft, 3), round(steps / max(dec, 1e-9), 2))

        # WebGPU capture path
        for c in self.Kc:
            c.data[:] = 0.0
        for v in self.Vc:
            v.data[:] = 0.0
        t0 = time.perf_counter(); g0 = self._prefill(ids)
        ttft = time.perf_counter() - t0
        plat = wt._adam_kernel["platform"]
        self._set_inputs(g0, P)
        plat.beginCapture("decode")
        logits_t = self._decode_fwd(); logits_t.numpy()
        plat.endCapture(); self.capture_ready = True
        gen = [g0]; nxt = g0; pos = P; steps = 0; td = time.perf_counter()
        while len(gen) < max_new:
            self._set_inputs(nxt, pos)
            plat.replay("decode")
            nxt = int(logits_t.numpy()[0].argmax()); pos += 1; steps += 1
            if nxt == eot:
                break
            gen.append(nxt)
        dec = time.perf_counter() - td
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), round(steps / max(dec, 1e-9), 2))
