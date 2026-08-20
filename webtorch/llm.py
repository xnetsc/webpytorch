"""webtorch.llm -- LLM inference engine with graph-capture acceleration.

High-level API for running LLMs in the browser on top of webtorch/WgPy — AutoGPTQ
safetensors or llama.cpp GGUF, at int4 / int8 / fp16 — with a WebGPU graph-capture fast
path for decode.

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

Nothing here is tied to a model family. Shapes, head dims, rope, MoE and hybrid
(state-space / linear-attention) blocks all come from the model's own config or GGUF
metadata, optional pieces (QK-norm, fused QKV, fused Q+output-gate, sparse MoE, recurrent
blocks) are detected from the tensors the file actually contains, and the tokenizer's special
tokens and chat layout are read from the vocabulary the model ships.
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


# Chat formats, keyed by a marker token the MODEL's own vocabulary declares. This is discovery,
# not a model whitelist: whichever marker a tokenizer actually contains selects the format, so
# a model family the SDK has never heard of still works as long as it uses one of these
# conventions (and falls back to a plain transcript otherwise).
_CHAT_FORMATS = (
    ("<|im_start|>", "chatml"),            # Qwen, Yi, and many others
    ("<|start_header_id|>", "llama3"),     # Llama 3.x
    ("<start_of_turn>", "gemma"),          # Gemma
    ("[INST]", "mistral"),                 # Mistral / Mixtral
)
# Tokens that end a turn, in the order we prefer them. Again: whichever the model HAS.
_EOT_CANDIDATES = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "</s>",
                   "<|endoftext|>", "<|end_of_text|>")


class BPETokenizer:
    """Byte-level BPE (vocab + merges), with the special tokens and chat format discovered
    from the model itself.

    Nothing here is tied to a model family: the special-token ids come from the vocabulary the
    model ships, the end-of-turn id from the model's own metadata (or the first end-of-turn
    marker its vocabulary defines), and the chat layout from whichever marker token the
    vocabulary contains. A model using none of the known conventions still works through a
    plain `role: content` transcript."""

    def __init__(self, vocab, merges, eos_ids=None, chat_format=None):
        self.enc = vocab
        self.dec = {i: t for t, i in vocab.items()}
        # discover specials that this vocabulary actually defines
        self.SPECIALS = {t: vocab[t] for t in
                         ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|eot_id|>",
                          "<|start_header_id|>", "<|end_header_id|>", "<start_of_turn>",
                          "<end_of_turn>", "<|end_of_text|>", "<|end|>", "<s>", "</s>")
                         if t in vocab}
        self.chat_format = chat_format or next(
            (f for tok, f in _CHAT_FORMATS if tok in vocab), "plain")
        eos = [int(i) for i in (eos_ids or []) if i is not None]
        eos += [vocab[t] for t in _EOT_CANDIDATES if t in vocab]
        # de-duplicate, keep order; may be empty -> generation then stops at max_new
        self.eos_ids = list(dict.fromkeys(eos))
        self.eot = self.eos_ids[0] if self.eos_ids else -1
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
            for p in self._bpe(s):
                i = self.enc.get(p)
                if i is None:                     # incomplete vocab: fall back to its bytes
                    for ch in p:
                        j = self.enc.get(ch)
                        if j is not None: ids.append(j)
                else:
                    ids.append(i)
        return ids

    def _sp(self, name):
        return self.SPECIALS.get(name)

    def encode_chat(self, user, system="You are a helpful assistant."):
        """Render a chat prompt in whatever format this model's vocabulary indicates."""
        f = self.chat_format
        if f == "chatml":
            ims, ime = self._sp("<|im_start|>"), self._sp("<|im_end|>")
            seq = []
            for role, content in (("system", system), ("user", user)):
                seq += [ims] + self.encode(role + "\n" + content) + [ime] + self.encode("\n")
            return seq + [ims] + self.encode("assistant\n")
        if f == "llama3":
            sh, eh = self._sp("<|start_header_id|>"), self._sp("<|end_header_id|>")
            eot = self._sp("<|eot_id|>")
            seq = []
            bos = self._sp("<|begin_of_text|>")
            if bos is not None: seq.append(bos)
            for role, content in (("system", system), ("user", user)):
                seq += [sh] + self.encode(role) + [eh] + self.encode("\n\n" + content) + [eot]
            return seq + [sh] + self.encode("assistant") + [eh] + self.encode("\n\n")
        if f == "gemma":
            sot, eot = self._sp("<start_of_turn>"), self._sp("<end_of_turn>")
            # Gemma has no system role: fold it into the first user turn
            body = (system + "\n\n" + user) if system else user
            return ([sot] + self.encode("user\n" + body) + [eot] + self.encode("\n")
                    + [sot] + self.encode("model\n"))
        if f == "mistral":
            bos = self._sp("<s>")
            seq = [bos] if bos is not None else []
            body = (system + "\n\n" + user) if system else user
            return seq + self.encode("[INST] " + body + " [/INST]")
        # plain transcript — works for a base model or an unknown convention
        pre = (system + "\n\n") if system else ""
        return self.encode(pre + "User: " + user + "\nAssistant:")

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
# One range request spans several of the IO layer's 16 MB chunks so they are fetched in
# parallel; the fp32 expansion is done in _HEAD_BLK-row sub-blocks so it stays bounded
# whatever the read size.
_READ_BYTES = 64 << 20
_HEAD_BLK = 4096


class _RowTable:
    """A 2-D weight that is only ever read by row, exposed with ndarray-style indexing.

    Embedding and lm_head tables are indexed by token id -- a handful of rows per step, or
    one block at a time when quantizing the head. Materializing the whole thing as a dense
    (vocab, hidden) fp16 array is ~1.5 GB on a 27B, which 32-bit WASM refuses to allocate
    as a single object. Subclasses supply `_rows`; indexing semantics match ndarray, so a
    scalar gives (H,), a sequence gives (n, H) and a slice gives (n, H).
    """

    def __init__(self, nrows, H):
        self.shape = (nrows, H)
        self.dtype = np.float16

    def __len__(self):
        return self.shape[0]

    def _rows(self, idx):
        """Dense (len(idx), H) fp16 for the given row numbers."""
        raise NotImplementedError

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._rows(np.arange(*idx.indices(self.shape[0])))
        ids = np.asarray(idx, np.int64)
        flat = np.atleast_1d(ids).ravel()
        # Repeated ids are common in a prompt; fetch each distinct row once.
        uniq, inv = np.unique(flat, return_inverse=True)
        out = self._rows(uniq)[inv]
        return out.reshape(ids.shape + (self.shape[1],)) if ids.ndim else out[0]


class _QuantRows(_RowTable):
    """Rows kept in their GGUF-quantized form and dequantized on lookup."""

    dtype = np.float32                                         # rows arrive as fp32

    def __init__(self, buf, gtype, H, row_nbytes, dequant):
        raw = np.frombuffer(buf, np.uint8)
        _RowTable.__init__(self, raw.size // row_nbytes, H)
        self.raw = raw.reshape(self.shape[0], row_nbytes)      # a view, not a copy
        self.gtype, self._dequant = gtype, dequant

    def _rows(self, idx):
        idx = np.asarray(idx, np.int64)
        # Each row is an independent, block-aligned run of bytes, so gathering the wanted
        # rows and dequantizing the result in ONE call is equivalent to a call per row --
        # and avoids paying numpy's per-call overhead once per token, which dominates
        # everything else at prompt length (measured 3.1s vs 12ms for 320 ids).
        sel = self.raw[idx]                                    # (n, row_nbytes), contiguous
        n, H = idx.size, self.shape[1]
        # `sel` is handed to the dequantizer directly (it reads through the buffer
        # protocol) and the fp32 it returns is handed on as-is: the caller wants fp32, so
        # a detour through fp16 would be two conversions and two copies per token.
        return self._dequant(self.gtype, sel, n * H).reshape(n, H)


class _ChunkRows(_RowTable):
    """Rows held as fp16 in several bounded blocks rather than one oversized array."""

    def __init__(self, chunks, rpc, nrows, H):
        _RowTable.__init__(self, nrows, H)
        self.chunks, self.rpc = chunks, rpc

    def _rows(self, idx):
        idx = np.asarray(idx, np.int64)
        out = np.empty((idx.size, self.shape[1]), np.float16)
        # One vectorized gather per source block rather than a copy per row.
        ch = idx // self.rpc
        for c in np.unique(ch):
            m = ch == c
            out[m] = self.chunks[int(c)][idx[m] - int(c) * self.rpc]
        return out


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

    def _apply_cfg(self, cfg):
        """Read an HF `config.json` into the engine's shape parameters. Purely config-driven —
        no model-name special cases — so it covers the whole Llama-style decoder family
        (Llama / Mistral / Qwen2 / Qwen3 / …), dense or MoE:
          * `head_dim` is honoured when present (Qwen3 has head_dim * num_heads != hidden_size);
            it falls back to hidden_size // num_heads for models that omit it.
          * QK-norm (Qwen3) is detected from the weights at load time, not from the model name.
          * MoE fields (num_experts / num_experts_per_tok / …) drive the sparse path."""
        # Multimodal/composite configs nest the decoder under `text_config` (and the vision
        # tower under `vision_config`); use the text sub-config when present.
        if "text_config" in cfg and "hidden_size" in (cfg.get("text_config") or {}):
            self.full_cfg = cfg
            cfg = dict(cfg["text_config"])
        else:
            self.full_cfg = cfg
        self.H = cfg["hidden_size"]; self.L = cfg["num_hidden_layers"]
        self.NH = cfg["num_attention_heads"]
        self.NKV = cfg.get("num_key_value_heads", self.NH)
        self.HD = int(cfg.get("head_dim") or (self.H // self.NH))
        self.VOCAB = cfg["vocab_size"]
        self.eps = cfg.get("rms_norm_eps", 1e-6); self.theta = cfg.get("rope_theta", 10000.0)
        # MoE (0 => dense); mirrors lm_engine.build_lm's config contract
        self.n_experts = int(cfg.get("num_experts", cfg.get("num_local_experts", 0)) or 0)
        self.top_k = int(cfg.get("num_experts_per_tok", 0) or 0)
        self.norm_topk = bool(cfg.get("norm_topk_prob", True))
        self.sparse_step = int(cfg.get("decoder_sparse_step", 1) or 1)
        self.mlp_only = set(cfg.get("mlp_only_layers", []) or [])
        self.has_shared_expert = bool(cfg.get("shared_expert_intermediate_size", 0))
        # ---- hybrid linear-attention models (Gated DeltaNet family) ----
        # `layer_types` marks each layer "full_attention" or "linear_attention"; when absent,
        # `full_attention_interval` (every Nth layer is full) is used; otherwise all-full.
        types = cfg.get("layer_types") or []
        interval = int(cfg.get("full_attention_interval", 0) or 0)
        if types:
            self.layer_types = list(types)
        elif interval > 1:
            self.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                for i in range(self.L)]
        else:
            self.layer_types = ["full_attention"] * self.L
        self.is_hybrid = any(t != "full_attention" for t in self.layer_types)
        self.linear_cfg = {
            "n_k_heads": int(cfg.get("linear_num_key_heads", 0) or 0),
            "n_v_heads": int(cfg.get("linear_num_value_heads", 0) or 0),
            "k_head_dim": int(cfg.get("linear_key_head_dim", 0) or 0),
            "v_head_dim": int(cfg.get("linear_value_head_dim", 0) or 0),
            "conv_kernel_dim": int(cfg.get("linear_conv_kernel_dim", 0) or 0),
            "hidden": self.H,
        }
        # partial rope: only the first `partial_rotary_factor` of head_dim is rotated
        rp = cfg.get("rope_parameters") or {}
        self.theta = float(rp.get("rope_theta", self.theta))
        self.partial_rotary = float(cfg.get("partial_rotary_factor",
                                            rp.get("partial_rotary_factor", 1.0)) or 1.0)
        self.rope_dim = int(self.HD * self.partial_rotary) // 2 * 2 or self.HD
        self.attn_output_gate = bool(cfg.get("attn_output_gate", False))
        # resolved from the text sub-config (nested configs put it there, not at the top level)
        self.tie_embeddings = bool(cfg.get("tie_word_embeddings", False))

    def _is_moe_layer(self, i):
        return (self.n_experts > 0 and i not in self.mlp_only
                and ((i + 1) % self.sparse_step == 0))

    async def _opt_norm(self, name):
        """Load an optional norm weight (e.g. Qwen3's q_norm/k_norm); None when absent."""
        return wt.Tensor(await self._np(name)) if name in self._idx else None

    async def _build_linear_attn(self, i, p, lin):
        """Build one linear-attention (Gated DeltaNet) layer. Weight names follow the
        `linear_attn.*` convention of this model family; every piece is optional so variants
        with/without conv, gate or output norm all load."""
        from . import linear_attn as la
        async def opt_lin(nm):
            return await lin(p + nm) if (p + nm + ".weight") in self._idx else None
        async def opt_arr(nm):
            return await self._np(p + nm) if (p + nm) in self._idx else None
        base = "linear_attn."
        w = {"q": await opt_lin(base + "q_proj"), "k": await opt_lin(base + "k_proj"),
             "v": await opt_lin(base + "v_proj"), "o": await opt_lin(base + "out_proj"),
             "a": await opt_lin(base + "a_proj"), "dt": await opt_lin(base + "dt_proj"),
             "g": await opt_lin(base + "g_proj"),
             "conv_w": await opt_arr(base + "conv1d.weight"),
             "conv_b": await opt_arr(base + "conv1d.bias"),
             "norm": await opt_arr(base + "norm.weight")}
        if w["conv_w"] is not None and w["conv_w"].ndim == 3:   # (C,1,W) -> (C,W)
            w["conv_w"] = w["conv_w"].reshape(w["conv_w"].shape[0], -1)
        return la.LinearAttention(dict(self.linear_cfg), w, eps=self.eps)

    async def _build_layer(self, i, lin):
        """Build one decoder layer generically from the served weights. `lin(prefix)` is the
        format's linear factory (int4 for AutoGPTQ, fp16/quantized for plain HF), so this is
        shared by every loader. Optional pieces are detected from the weights, not the model
        name: QK-norm (`self_attn.{q,k}_norm.weight`, Qwen3) and a sparse-MoE block."""
        p = "model.layers.%d." % i
        lay = {"in_ln": wt.Tensor(await self._np(p + "input_layernorm.weight")),
               "post_ln": wt.Tensor(await self._np(p + "post_attention_layernorm.weight"))}
        if self._is_linear_layer(i):                    # recurrent linear-attention layer
            lay["linear"] = await self._build_linear_attn(i, p, lin)
        else:                                           # softmax attention layer
            lay.update({
                "q": await lin(p + "self_attn.q_proj"), "k": await lin(p + "self_attn.k_proj"),
                "v": await lin(p + "self_attn.v_proj"), "o": await lin(p + "self_attn.o_proj"),
                "qn": await self._opt_norm(p + "self_attn.q_norm.weight"),
                "kn": await self._opt_norm(p + "self_attn.k_norm.weight")})
        if self._is_moe_layer(i):                       # sparse layer: router + experts
            moe = {"gate": await lin(p + "mlp.gate"), "top_k": self.top_k,
                   "norm_topk": self.norm_topk,
                   "experts": [{"gate": await lin(p + "mlp.experts.%d.gate_proj" % e),
                                "up": await lin(p + "mlp.experts.%d.up_proj" % e),
                                "down": await lin(p + "mlp.experts.%d.down_proj" % e)}
                               for e in range(self.n_experts)]}
            if self.has_shared_expert:
                moe["shared"] = {"gate": await lin(p + "mlp.shared_expert.gate_proj"),
                                 "up": await lin(p + "mlp.shared_expert.up_proj"),
                                 "down": await lin(p + "mlp.shared_expert.down_proj")}
                moe["shared_gate"] = await lin(p + "mlp.shared_expert_gate")
            lay["moe"] = moe
        else:                                           # dense SwiGLU
            lay["gate"] = await lin(p + "mlp.gate_proj")
            lay["up"] = await lin(p + "mlp.up_proj")
            lay["down"] = await lin(p + "mlp.down_proj")
        return lay

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
        rpc = max(1, chunk_bytes // row); chunks = []
        for v0 in range(0, V, rpc):
            v1 = min(V, v0 + rpc)
            raw = await self._rng(shard, base + a + v0 * row, base + a + v1 * row - 1)
            if dt == "BF16":
                arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, np.float16)
            else:
                arr = np.frombuffer(raw, np.float32)
            chunks.append(arr.reshape(v1 - v0, Hh).astype(np.float16))
        return _ChunkRows(chunks, rpc, V, Hh)

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
        self._apply_cfg(cfg)
        self.gs = cfg["quantization_config"]["group_size"]; self.bits = cfg["quantization_config"]["bits"]
        tie = self.tie_embeddings

        vocab = await webio.read_json(self.base + "vocab.json")
        merges = (await webio.read_text(self.base + "merges.txt")).split("\n")
        _e = cfg.get("eos_token_id")
        _eos = _e if isinstance(_e, list) else ([_e] if _e is not None else [])
        self.tok = BPETokenizer(vocab, [m for m in merges[1:] if m and not m.startswith("#")],
                                eos_ids=_eos)

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
        self.layers = [await self._build_layer(i, self._qlin) for i in range(self.L)]
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

    async def _gexperts(self, name):
        """Yield each expert of a stacked MoE tensor as fp32 (out,in), one at a time.

        llama.cpp packs a layer's experts into a single `ffn_*_exps` tensor. Dequantizing
        the whole stack costs ~800 MB of fp32 on a 128-expert layer, so experts are
        converted individually -- but the packed bytes are fetched in ONE range request,
        because a read per expert would turn three reads per layer into hundreds. Peak
        cost is therefore the packed tensor (tens of MB) plus a single fp32 expert.
        Quantization blocks run along `in`, and out*in is a whole number of blocks, so
        every per-expert offset is block-aligned.
        """
        from . import ggufload as G
        t = self._ginfo[name]
        ne, out_d, in_d = (int(d) for d in reversed(t["dims"]))
        per = out_d * in_d
        nb = G.tensor_nbytes(t["type"], per)                  # bytes per expert
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb * ne - 1)
        for e in range(ne):
            yield G.dequant(t["type"], raw[e * nb:(e + 1) * nb], per).reshape(out_d, in_d)

    def _gquant(self, W_out_in, bias=None):
        """fp32 (out,in) weight -> a linear layer. Quantized to int`bits` by default; with
        `self._qbits == 0` (dtype="fp16") it stays unquantized, which also makes GGUF models
        runnable without a GPU (the int kernel is GPU-only)."""
        if getattr(self, "_qbits", None) == 0:
            return wt.UnquantizedLinear(W_out_in, bias)
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
    async def from_gguf(cls, url, lmax=320, bits=4, quantize=True):
        """Load a llama.cpp GGUF, dequantizing + requantizing weights to int`bits` (4 or 8) so
        they run on the same capture-accelerated engine. `url` is the served .gguf file.

        **Architecture-generic**: shape/rope/eps are read from GGUF's own `{arch}.*` metadata
        keys (the format's convention), and optional pieces are detected from the tensors —
        QK-norm (`attn_q_norm`/`attn_k_norm`, e.g. Qwen3) and sparse-MoE (`ffn_gate_inp` +
        `ffn_*_exps`). So any Llama-style decoder GGUF (llama / qwen2 / qwen3 / mistral / …)
        loads without a per-model branch. A GGUF whose weights use a quantization type this
        SDK cannot dequantize (e.g. the IQ i-quants) is rejected with a clear error."""
        from . import ggufload as G
        self = cls(None); self.lmax = lmax; self._gguf = url
        self.bits = bits                                      # int4 or int8 (same kernels/optims)
        # `quantize=False` (dtype="fp16") keeps the dequantized weights unquantized, so a GGUF
        # runs on the numpy/CPU path too — the int kernel is GPU-only.
        self._qbits = bits if quantize else 0
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
        if not arch:
            raise ValueError("GGUF has no general.architecture")
        A = arch + "."                                        # GGUF namespaces its keys by arch
        def m(key, *alts, default=None, required=True):
            for k in (key,) + alts:
                if A + k in meta: return meta[A + k]
            if default is not None or not required: return default
            raise NotImplementedError(
                "GGUF arch %r is missing %r — unsupported architecture for from_gguf" % (arch, key))
        self.H = int(m("embedding_length")); self.L = int(m("block_count"))
        self.NH = int(m("attention.head_count"))
        self.NKV = int(m("attention.head_count_kv", default=self.NH, required=False) or self.NH)
        self.HD = int(m("attention.key_length", default=0, required=False) or (self.H // self.NH))
        self.VOCAB = len(meta["tokenizer.ggml.tokens"])
        self.eps = float(m("attention.layer_norm_rms_epsilon", default=1e-6, required=False))
        self.theta = float(m("rope.freq_base", default=10000.0, required=False))
        # MoE (0 => dense), same contract as the safetensors path
        self.n_experts = int(m("expert_count", default=0, required=False) or 0)
        self.top_k = int(m("expert_used_count", default=0, required=False) or 0)
        self.norm_topk = True; self.sparse_step = 1; self.mlp_only = set()
        self.has_shared_expert = False
        # ---- hybrid / state-space blocks, read from the file's own metadata ----
        # A GGUF declares these under its arch namespace whenever some blocks are recurrent
        # (linear attention / SSM) instead of softmax attention. Nothing here names a model:
        # a file that declares them gets the recurrent path, a file that does not gets the
        # plain decoder path.
        interval = int(m("full_attention_interval", default=0, required=False) or 0)
        recurrent = m("attention.recurrent_layers", default=None, required=False)
        if recurrent:
            self.layer_types = ["linear_attention" if bool(r) else "full_attention"
                                for r in list(recurrent)[:self.L]]
        elif interval > 1:
            self.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                for i in range(self.L)]
        else:
            self.layer_types = ["full_attention"] * self.L
        self.is_hybrid = any(t != "full_attention" for t in self.layer_types)
        d_state = int(m("ssm.state_size", default=0, required=False) or 0)
        d_inner = int(m("ssm.inner_size", default=0, required=False) or 0)
        n_group = int(m("ssm.group_count", default=0, required=False) or 0)
        dt_rank = int(m("ssm.time_step_rank", default=0, required=False) or 0)
        self.linear_cfg = {
            "n_k_heads": n_group, "n_v_heads": dt_rank,
            "k_head_dim": d_state,
            "v_head_dim": (d_inner // dt_rank) if dt_rank else 0,
            "conv_kernel_dim": int(m("ssm.conv_kernel", default=0, required=False) or 0),
            "hidden": self.H}
        # partial rope: the file says how many dims are rotated
        n_rot = int(m("rope.dimension_count", default=0, required=False) or 0)
        self.rope_dim = n_rot if n_rot else self.HD
        self.partial_rotary = (self.rope_dim / self.HD) if self.HD else 1.0
        self.attn_output_gate = False        # set below if the weights show a gate
        _eos = [meta.get("tokenizer.ggml.eos_token_id"), meta.get("tokenizer.ggml.eot_token_id")]
        self.tok = BPETokenizer({t: i for i, t in enumerate(meta["tokenizer.ggml.tokens"])},
                                meta.get("tokenizer.ggml.merges", []))
        self._ginfo = {t["name"]: t for t in infos}; self._gds = ds

        # Preflight: report every unsupported quantization up front, from the header alone,
        # instead of failing part-way through a multi-GB download. Mixed ("dynamic") files
        # combine many types, so list all of them with how many tensors each covers.
        from collections import Counter
        bad = Counter(G.GGML_NAMES.get(t["type"], "type%d" % t["type"])
                      for t in infos if not G.is_supported(t["type"]))
        if bad:
            raise NotImplementedError(
                "this GGUF uses quantization types webtorch cannot dequantize yet: %s. "
                "Supported: %s. Pick a build using only supported types (e.g. Q4_K_M, Q5_K_M, "
                "Q6_K, Q8_0, IQ4_XS)." % (
                    ", ".join("%s (%d tensors)" % (k, v) for k, v in sorted(bad.items())),
                    ", ".join(sorted(G.SUPPORTED_NAMES))))

        # Preflight the architecture from the header: a block must be something this engine
        # can run — softmax attention (separate q/k/v or a fused qkv) or a recurrent
        # linear-attention / state-space block (ssm_* tensors). Report a mismatch now rather
        # than after downloading gigabytes.
        names = set(self._ginfo)
        if "blk.0.attn_norm.weight" in names:
            post = any(("blk.0." + n + ".weight") in names
                       for n in ("ffn_norm", "post_attention_norm"))
            attn = (all(("blk.0.attn_%s.weight" % x) in names for x in ("q", "k", "v"))
                    or "blk.0.attn_qkv.weight" in names)
            recurrent = any(n.startswith("blk.0.ssm_") for n in names)
            if not post or not (attn or recurrent):
                missing = []
                if not post: missing.append("a post-attention norm (ffn_norm/post_attention_norm)")
                if not (attn or recurrent):
                    missing.append("attention (attn_q/k/v or attn_qkv) or recurrent (ssm_*) tensors")
                raise NotImplementedError(
                    "this GGUF's %r blocks are not a shape this engine implements: layer 0 "
                    "lacks %s." % (arch, "; ".join(missing)))
        bad = sorted({G.GGML_NAMES.get(t["type"], "type%d" % t["type"])
                      for t in infos if t["type"] not in G.SUPPORTED_TYPES})
        if bad:                                               # e.g. IQ4_XS / IQ2_M i-quants
            raise NotImplementedError(
                "GGUF uses quantization type(s) %s which this SDK cannot dequantize "
                "(supported: %s). Use a supported quant (Q4_K/Q6_K/Q8_0/Q4_0/Q4_1/F16/F32)."
                % (", ".join(bad), ", ".join(sorted(G.SUPPORTED_NAMES))))

        t0 = time.perf_counter()
        # token_embd -> host f16 (streamed in row-blocks so the full 1.2GB fp32
        # never materializes); if tied, build the int4 head from the same blocks.
        et = self._ginfo["token_embd.weight"]
        H = int(et["dims"][0]); V = int(et["dims"][1])
        erow = G.tensor_nbytes(et["type"], H); eoff = self._gds + et["offset"]
        # The table is kept PACKED and read a row at a time (see _PackedRows): it is only
        # ever indexed by token id, and a dense (V,H) fp16 copy is ~1.5 GB on a 27B --
        # more than 32-bit WASM will hand out for a single array.
        ebuf = np.empty(V * erow, np.uint8)
        tied = "output.weight" not in self._ginfo
        self.head = [] if tied else None
        # Read in big spans, not row counts: the IO layer splits a request into 16 MB
        # pieces and runs several at once, so one large read keeps every fetch slot busy
        # where a 4096-row read (a few MB) leaves them idle and pays the round trip per
        # chunk. Head quantization still walks the span in bounded sub-blocks, because it
        # is the fp32 expansion -- not the read -- that has to stay small.
        rpr = max(_HEAD_BLK, (_READ_BYTES // erow) // _HEAD_BLK * _HEAD_BLK)
        for v0 in range(0, V, rpr):
            v1 = min(V, v0 + rpr)
            raw = await self._grng(eoff + v0 * erow, eoff + v1 * erow - 1)
            ebuf[v0 * erow:v1 * erow] = np.frombuffer(raw, np.uint8)
            if tied:                                          # (blk,H) = (out,in)
                for b0 in range(v0, v1, _HEAD_BLK):
                    b1 = min(v1, b0 + _HEAD_BLK)
                    fb = G.dequant(et["type"], raw[(b0 - v0) * erow:(b1 - v0) * erow],
                                   (b1 - b0) * H).reshape(b1 - b0, H)
                    self.head.append(self._gquant(fb))
                    del fb
        self.embed = _QuantRows(memoryview(ebuf.data).cast("B"), et["type"], H, erow, G.dequant)
        if not tied:                                          # separate lm_head, also streamed
            ot = self._ginfo["output.weight"]
            orow = G.tensor_nbytes(ot["type"], H); ooff = self._gds + ot["offset"]
            self.head = []
            orpr = max(_HEAD_BLK, (_READ_BYTES // orow) // _HEAD_BLK * _HEAD_BLK)
            for v0 in range(0, V, orpr):                      # large reads, small expansions
                v1 = min(V, v0 + orpr)
                raw = await self._grng(ooff + v0 * orow, ooff + v1 * orow - 1)
                for b0 in range(v0, v1, _HEAD_BLK):
                    b1 = min(v1, b0 + _HEAD_BLK)
                    fb = G.dequant(ot["type"], raw[(b0 - v0) * orow:(b1 - v0) * orow],
                                   (b1 - b0) * H).reshape(b1 - b0, H)
                    self.head.append(self._gquant(fb)); del fb
        self.layers = []
        for i in range(self.L):
            p = "blk.%d." % i
            async def qb(nm, _p=p):                            # weight + optional bias -> QuantizedLinear
                b = (await self._gload(_p + nm + ".bias")) if (_p + nm + ".bias") in self._ginfo else None
                return self._gquant(await self._gload(_p + nm + ".weight"), b)
            async def opt_norm(nm, _p=p):                      # QK-norm (qwen3-style) if present
                n = _p + nm + ".weight"
                return wt.Tensor(await self._gload(n)) if n in self._ginfo else None
            # The post-attention norm is called `ffn_norm` by most converters and
            # `post_attention_norm` by others; take whichever this file has.
            post_name = next((n for n in ("ffn_norm", "post_attention_norm")
                              if (p + n + ".weight") in self._ginfo), None)
            if post_name is None:
                raise NotImplementedError(
                    "GGUF block %d has no post-attention norm (looked for ffn_norm / "
                    "post_attention_norm)" % i)
            lay = {"in_ln": wt.Tensor(await self._gload(p + "attn_norm.weight")),
                   "post_ln": wt.Tensor(await self._gload(p + post_name + ".weight"))}
            has_ssm = any(n.startswith(p + "ssm_") for n in self._ginfo)
            if has_ssm or self._is_linear_layer(i):
                # Recurrent (linear-attention / state-space) block. Every piece is optional and
                # located by name, so any converter's layout loads as long as the tensors exist.
                async def opt_g(nm, _p=p):
                    n = _p + nm
                    return await self._gload(n) if n in self._ginfo else None
                async def opt_qb(nm, _p=p):
                    return (await qb(nm, _p)) if (_p + nm + ".weight") in self._ginfo else None
                w = {"qkv": await opt_qb("attn_qkv"), "g": await opt_qb("attn_gate"),
                     "q": await opt_qb("attn_q"), "k": await opt_qb("attn_k"),
                     "v": await opt_qb("attn_v"),
                     "beta": await opt_qb("ssm_beta"), "alpha": await opt_qb("ssm_alpha"),
                     "o": await opt_qb("ssm_out"),
                     "A": await opt_g("ssm_a"), "dt_bias": await opt_g("ssm_dt.bias"),
                     "norm": await opt_g("ssm_norm.weight"),
                     "conv_w": await opt_g("ssm_conv1d.weight"),
                     "conv_b": await opt_g("ssm_conv1d.bias")}
                if w["conv_w"] is not None and w["conv_w"].ndim > 2:
                    w["conv_w"] = w["conv_w"].reshape(w["conv_w"].shape[0], -1)
                from . import linear_attn as la
                lay["linear"] = la.LinearAttention(dict(self.linear_cfg), w, eps=self.eps)
                self.layer_types[i] = "linear_attention"
            else:
                lay.update({
                    "q": await qb("attn_q"), "k": await qb("attn_k"), "v": await qb("attn_v"),
                    "o": await qb("attn_output"),
                    "qn": await opt_norm("attn_q_norm"), "kn": await opt_norm("attn_k_norm")})
            if (p + "ffn_gate_inp.weight") in self._ginfo:     # sparse-MoE block
                # llama.cpp stacks experts: ffn_{gate,up,down}_exps = (n_experts, out, in).
                # Taken one expert at a time -- see _gload_expert; the full stack does not
                # fit, and only the packed result is kept.
                ne = int(self._ginfo[p + "ffn_gate_exps.weight"]["dims"][-1])
                experts = [{} for _ in range(ne)]
                for key, nm in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"),
                                ("down", "ffn_down_exps")):
                    e = 0
                    async for W in self._gexperts(p + nm + ".weight"):
                        experts[e][key] = self._gquant(W); e += 1
                lay["moe"] = {"gate": self._gquant(await self._gload(p + "ffn_gate_inp.weight")),
                              "top_k": self.top_k or 2, "norm_topk": True, "experts": experts}
            else:
                lay["gate"] = await qb("ffn_gate"); lay["up"] = await qb("ffn_up")
                lay["down"] = await qb("ffn_down")
            self.layers.append(lay)
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
        self._apply_cfg(cfg)
        tie = self.tie_embeddings

        vocab = await webio.read_json(self.base + "vocab.json")
        merges = (await webio.read_text(self.base + "merges.txt")).split("\n")
        _e = cfg.get("eos_token_id")
        _eos = _e if isinstance(_e, list) else ([_e] if _e is not None else [])
        self.tok = BPETokenizer(vocab, [m for m in merges[1:] if m and not m.startswith("#")],
                                eos_ids=_eos)

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
        self.layers = [await self._build_layer(i, self._lin) for i in range(self.L)]
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._init_state()
        return self

    # ---- math ----
    def _rms(self, x, w):
        return (x / ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()) * w

    def _rot(self, x):
        """rotate_half. With partial rope the swap happens WITHIN the rotated prefix
        (`rope_dim`); the pass-through tail is appended unchanged (it is multiplied by sin=0,
        so its value is irrelevant — only the layout must line up)."""
        hd = self.HD; rd = getattr(self, "rope_dim", hd)
        half = rd // 2
        parts = [wt._slice_last(x, half, rd) * (-1.0), wt._slice_last(x, 0, half)]
        if rd < hd:
            parts.append(wt._slice_last(x, rd, hd))
        return wt.cat(parts, axis=-1)

    def _rope_np(self, pos, T=1):
        """rope cos/sin of shape (T, HD). With `partial_rotary_factor < 1` (Qwen3.5/3.8 style)
        only the first `rope_dim` dims are rotated: the tail gets cos=1, sin=0, which is the
        identity under `x*cos + rot(x)*sin`, so the forward paths need no special case."""
        rd = getattr(self, "rope_dim", self.HD)
        inv = 1.0 / (self.theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
        ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([ang, ang], -1)                  # (T, rd)
        cos = np.cos(emb).astype(np.float32); sin = np.sin(emb).astype(np.float32)
        if rd < self.HD:                                      # pad the un-rotated tail: identity
            pad = self.HD - rd
            cos = np.concatenate([cos, np.ones((cos.shape[0], pad), np.float32)], -1)
            sin = np.concatenate([sin, np.zeros((sin.shape[0], pad), np.float32)], -1)
        return cos, sin

    def _logits(self, hlast):
        return wt.cat([blk(hlast) for blk in self.head], axis=-1).numpy()[0]

    def _head_argmax(self, hlast):
        return self._pick(self._logits(hlast))

    def _pick(self, logits):
        """Turn logits into a token id using the current sampling parameters. Greedy when
        `temperature <= 0` (or no sampling params were given), otherwise temperature +
        top-k/top-p nucleus sampling. Parameters come from `generate(...)` / `load(...)`."""
        sp = getattr(self, "_sampling", None)
        if not sp or not sp.get("do_sample"):
            return int(np.asarray(logits).argmax())
        from . import lm_engine
        return int(lm_engine.sample_nucleus(
            np.asarray(logits, np.float32) / max(float(sp.get("temperature", 1.0)), 1e-5),
            top_p=float(sp.get("top_p", 1.0)), top_k=int(sp.get("top_k", 0) or 10 ** 9),
            rng=sp.get("rng")))

    def _set_sampling(self, temperature=None, top_p=None, top_k=None, seed=None, do_sample=None,
                      **_ignored):
        """Install generation parameters. Defaults set at load time are kept; anything passed to
        generate() overrides them for that call."""
        base = dict(getattr(self, "gen_defaults", {}) or {})
        for k, v in (("temperature", temperature), ("top_p", top_p),
                     ("top_k", top_k), ("seed", seed), ("do_sample", do_sample)):
            if v is not None:
                base[k] = v
        if base.get("do_sample") is None:
            base["do_sample"] = float(base.get("temperature", 0) or 0) > 0
        if base.get("seed") is not None:
            base["rng"] = np.random.default_rng(int(base["seed"]))
        self._sampling = base
        return base

    def _init_state(self):
        L, NKV, HD, LMAX, H = self.L, self.NKV, self.HD, self.lmax, self.H
        # recurrent states for linear-attention layers (fixed size, independent of length)
        self.lin_state = [lay["linear"].new_state() if lay.get("linear") else None
                          for lay in getattr(self, "layers", [])]
        if self._gpu:                                  # capture path: fixed scatter cache + persistent inputs
            self.Kc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(L)]
            self.Vc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(L)]
            self.h_in = wt.Tensor(np.zeros((1, H), np.float32))
            self.cos_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.sin_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.mask_b = wt.Tensor(np.zeros((1, 1, LMAX), np.float32))
            self.ctl = xp.asarray(np.array([0, 1, NKV, HD, LMAX], np.int32))

    # ---- prefill (fresh, fills KV 0..P-1) ----
    def _mlp(self, lay, x):
        """Dense SwiGLU, or generic sparse-MoE (router top-k + optional shared expert) when the
        layer carries `moe` — same layer dict shape as `lm_engine.build_lm`, so the proven
        generic MoE path is reused instead of a second implementation."""
        if lay.get("moe"):
            from . import lm_engine
            return lm_engine.moe_mlp(self, lay, x)
        return lay["down"](wt.silu(lay["gate"](x)) * lay["up"](x))

    def _qkv(self, lay, x, T):
        """q,k,v projections -> (heads, T, HD). Applies per-head QK-norm (RMSNorm over head_dim,
        before rope) when `lay` carries `qn`/`kn` weights.

        Some decoders fuse an output GATE into the query projection: it then emits
        2 * n_heads * head_dim, laid out per head as [q | gate]. That is detected from the
        projection's own width — not from a model name — and the gate is stashed on the layer
        for `_attn_out` to apply."""
        qraw = lay["q"](x)
        wide = int(np.prod(qraw.shape)) // max(T, 1) >= 2 * self.NH * self.HD
        if wide:                                        # fused [q | gate] per head
            qg = qraw.reshape(T, self.NH, 2 * self.HD)
            q = wt._slice_last(qg, 0, self.HD)
            lay["_gate"] = wt._slice_last(qg, self.HD, 2 * self.HD).reshape(T, self.NH * self.HD)
        else:
            q = qraw.reshape(T, self.NH, self.HD)
            lay["_gate"] = None
        k = lay["k"](x).reshape(T, self.NKV, self.HD)
        if lay.get("qn") is not None: q = self._rms(q, lay["qn"])
        if lay.get("kn") is not None: k = self._rms(k, lay["kn"])
        v = lay["v"](x).reshape(T, self.NKV, self.HD).permute(1, 0, 2)
        return q.permute(1, 0, 2), k.permute(1, 0, 2), v

    def _attn_out(self, lay, o, T):
        """Attention output -> hidden. Applies the sigmoid output gate when the query projection
        carried one (see `_qkv`), then the output projection."""
        y = o.permute(1, 0, 2).reshape(T, self.NH * self.HD)
        g = lay.get("_gate")
        if g is not None:
            y = y * g.sigmoid()
        return lay["o"](y)

    def _embed_ids(self, ids, embeds=None):
        """Input embeddings for a prompt. `embeds` (T,H) overrides the token lookup — the hook
        multimodal models use to splice image/audio embeddings into the sequence."""
        if embeds is not None:
            return wt.Tensor(np.asarray(embeds, np.float32))
        return wt.Tensor(np.asarray(self.embed[np.asarray(ids, np.int64)], np.float32))

    def _reset_linear_state(self):
        """Clear every linear-attention layer's recurrent state (start of a generation)."""
        for st in getattr(self, "lin_state", []) or []:
            if st is not None:
                st.reset()

    def _is_linear_layer(self, i):
        """True when layer `i` is a linear-attention (recurrent state) layer rather than softmax
        attention. Driven by the config's `layer_types` / `full_attention_interval`."""
        lt = getattr(self, "layer_types", None)
        return bool(lt) and i < len(lt) and lt[i] != "full_attention"

    def _linear_mixer(self, i, lay, x, T):
        """Run a linear-attention layer's recurrence -> (T, H) Tensor. The per-layer state lives
        in `self.lin_state[i]`, so prefill and incremental decode share one implementation."""
        st = self.lin_state[i]
        y = lay["linear"].forward(x.numpy() if hasattr(x, "numpy") else np.asarray(x), st)
        return wt.Tensor(np.asarray(y, np.float32))

    def _prefill(self, ids, embeds=None):
        T = len(ids); H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.lmax
        c, s = self._rope_np(0, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        m = np.triu(np.full((T, LMAX), -1e9, np.float32), 1); m[:, T:] = -1e9
        mask = wt.Tensor(m.reshape(1, T, LMAX))
        h = self._embed_ids(ids, embeds)
        sc = 1.0 / math.sqrt(HD)
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            if self._is_linear_layer(i):                       # recurrent (fixed-state) layer
                h = h + self._linear_mixer(i, lay, x, T)
            else:                                              # softmax attention layer
                q, k, v = self._qkv(lay, x, T)
                q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
                wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, T, NKV, HD, LMAX)
                wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, T, NKV, HD, LMAX)
                o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], mask, scale=sc)
                h = h + self._attn_out(lay, o, T)
            x = self._rms(h, lay["post_ln"])
            h = h + self._mlp(lay, x)
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))

    def _set_inputs(self, token, pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.lmax
        self.h_in.data.buffer.set_data(np.asarray(self.embed[token], np.float32))
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
            q, k, v = self._qkv(lay, x, 1)
            q = q * self.cos_b + self._rot(q) * self.sin_b
            k = k * self.cos_b + self._rot(k) * self.sin_b
            wt.kv_write(self.Kc[i].data, wt._contig(k).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            wt.kv_write(self.Vc[i].data, wt._contig(v).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
            o = wt.gqa_attention(q, self.Kc[i], self.Vc[i], self.mask_b, scale=sc)
            h = h + self._attn_out(lay, o, 1)
            x = self._rms(h, lay["post_ln"])
            h = h + self._mlp(lay, x)
        return wt.cat([blk(self._rms(h, self.final_norm)) for blk in self.head], axis=-1)

    # ---- WebGL path: growing KVCache, fresh forward (no in-place capture) ----
    def _kv_forward(self, ids, pos, cache, embeds=None):
        T = len(ids); H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        c, s = self._rope_np(pos, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        h = self._embed_ids(ids, embeds)
        sc = 1.0 / math.sqrt(HD)
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            if self._is_linear_layer(i):                       # recurrent (fixed-state) layer
                h = h + self._linear_mixer(i, lay, x, T)
            else:
                q, k, v = self._qkv(lay, x, T)
                q = q * cos_t + self._rot(q) * sin_t; k = k * cos_t + self._rot(k) * sin_t
                o = cache.attn(i, q, k, v, pos, scale=sc)
                h = h + self._attn_out(lay, o, T)
            x = self._rms(h, lay["post_ln"])
            h = h + self._mlp(lay, x)
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))

    def stream(self, prompt, max_new=48, system="You are a helpful assistant.",
               temperature=None, top_p=None, top_k=None, seed=None, do_sample=None, **_kw):
        """Streaming decode: yield each new token's text as it is produced (render live,
        bounded memory). WebGPU replays a captured step per token; WebGL grows a cache.
        Takes the same generation parameters as `generate` (temperature/top_p/top_k/seed)."""
        eot = self.tok.eot
        self._set_sampling(temperature, top_p, top_k, seed, do_sample)
        self._reset_linear_state()
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
            nxt = self._pick(logits_t.numpy()[0]); pos += 1

    def release(self):
        """Free this model's weights (layers, embeddings, head, KV cache). The object must not
        be used afterwards; load it again to use it. See `webtorch.release`."""
        from . import _sdk
        _sdk._free(self)
        return self

    def _check_live(self):
        if self.__dict__.get("_released"):
            raise RuntimeError("this model has been released; load it again to use it")

    def generate(self, prompt, max_new=48, system="You are a helpful assistant.",
                 ids=None, embeds=None, temperature=None, top_p=None, top_k=None, seed=None,
                 do_sample=None, **_kw):
        """ChatML prompt -> greedy decode -> GenResult. WebGPU replays a captured
        decode step per token (~20x); WebGL uses a correct growing-cache forward.
        For live token-by-token output use `stream(...)`.

        `ids`/`embeds` override the prompt encoding: pass prebuilt token ids and/or (T,H) input
        embeddings to decode from a sequence assembled elsewhere. That is the generic hook used
        for multimodality (image/audio embeddings spliced into the token embeddings) — see
        `webtorch.MultimodalLM` — so no model-specific decode path is needed."""
        self._check_live()
        eot = self.tok.eot
        if ids is None:
            ids = self.tok.encode_chat(prompt, system)
        P = len(ids)
        self._set_sampling(temperature, top_p, top_k, seed, do_sample)
        self._reset_linear_state()                     # fresh recurrent state per generation

        if not self._gpu:                              # WebGL fallback
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            t0 = time.perf_counter(); g0 = self._kv_forward(ids, 0, cache, embeds=embeds)
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
        t0 = time.perf_counter(); g0 = self._prefill(ids, embeds=embeds)
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
            nxt = self._pick(logits_t.numpy()[0]); pos += 1; steps += 1
            if nxt == eot:
                break
            gen.append(nxt)
        dec = time.perf_counter() - td
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), round(steps / max(dec, 1e-9), 2))
