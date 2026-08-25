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


def _tpl_raise(msg):
    raise ValueError(msg)


class BPETokenizer:
    """Byte-level BPE (vocab + merges), with the special tokens and chat format discovered
    from the model itself.

    Nothing here is tied to a model family: the special-token ids come from the vocabulary the
    model ships, the end-of-turn id from the model's own metadata (or the first end-of-turn
    marker its vocabulary defines), and the chat layout from whichever marker token the
    vocabulary contains. A model using none of the known conventions still works through a
    plain `role: content` transcript."""

    def __init__(self, vocab, merges, eos_ids=None, chat_format=None,
                 chat_template=None, control=None):
        self.enc = vocab
        # A model ships its own prompt layout. Probing the vocabulary for a known marker
        # gets the family right but not the details -- Qwen3 opens its turn with <think>,
        # which no amount of "this looks like ChatML" will tell you, and without it the
        # model answers by completing the template and stopping. So when the file carries a
        # chat template, that template is the source of truth; the probe is the fallback.
        self.chat_template = chat_template
        self._tpl = None
        # Tokens BPE must never split. From the file's own token_type where it has one --
        # anything else is guesswork about what "looks special".
        self.control = sorted((t for t in (control or ()) if t in vocab), key=len, reverse=True)
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

    async def prepare_template(self):
        """Compile the model's chat template, if it has one. Async because jinja2 is loaded
        on demand -- it ships with Pyodide, so this is a local package load, not a download,
        and a model without a template never pays for it. Rendering itself stays sync."""
        if self._tpl is not None or not self.chat_template:
            return self._tpl
        try:
            try:
                import jinja2
            except ImportError:
                import micropip
                await micropip.install("jinja2")
                import jinja2
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = _tpl_raise
            env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
            self._tpl = env.from_string(self.chat_template)
        except Exception as e:                     # no jinja2, or a template we cannot parse
            self._tpl = False
            print("webtorch: chat template unavailable (%s); using the detected format" % e)
        return self._tpl

    def encode_special(self, text):
        """Encode text, emitting each control token as its own id instead of BPE-ing it."""
        if not self.control:
            return self.encode(text)
        parts = self.re.split("(" + "|".join(self.re.escape(t) for t in self.control) + ")", text)
        out = []
        for p in parts:
            if not p:
                continue
            i = self.enc.get(p)
            out.append(i) if (i is not None and p in self.control) else out.extend(self.encode(p))
        return out

    def render_chat(self, messages, add_generation_prompt=True, **kw):
        """The model's own template applied to a message list -> token ids, or None if the
        model has no usable template."""
        if not self._tpl:
            return None
        try:
            txt = self._tpl.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                   bos_token=self._tok_str(self.SPECIALS.get("<s>")),
                                   eos_token=self._tok_str(self.eot), **kw)
        except Exception as e:
            print("webtorch: chat template failed to render (%s); using the detected format" % e)
            self._tpl = False
            return None
        return self.encode_special(txt)

    def _tok_str(self, i):
        return self.dec.get(int(i), "") if i is not None and int(i) >= 0 else ""

    def encode_chat(self, user=None, system="You are a helpful assistant.", messages=None,
                    tools=None, **tpl_kw):
        """Render a chat prompt: the model's own template when it has one, else whatever
        format its vocabulary indicates.

        `messages` is a full conversation ([{role, content}, ...]); `user`/`system` are the
        one-turn shorthand for it. `tools` and any other keyword are handed to the template
        unchanged, which is what makes tool calling, thinking toggles and the rest generic:
        a model that documents them in its template gets them, and one that does not
        ignores them. Nothing here knows what any particular model calls its options."""
        msgs = messages if messages is not None else (
            ([{"role": "system", "content": system}] if system else [])
            + ([{"role": "user", "content": user}] if user is not None else []))
        got = self.render_chat(msgs, tools=tools, **tpl_kw)
        if got is not None:
            return got
        # No template: fall back to the layout the vocabulary implies. Tools cannot be
        # expressed without one, so say so rather than dropping them silently.
        if tools:
            raise ValueError("this model ships no chat template, so tool definitions cannot "
                             "be rendered; pass them inside a message instead")
        f = self.chat_format
        sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
        turns = [m for m in msgs if m["role"] != "system"]
        if f == "chatml":
            ims, ime = self._sp("<|im_start|>"), self._sp("<|im_end|>")
            seq = []
            for m in msgs:
                seq += ([ims] + self.encode(m["role"] + "\n" + m["content"]) + [ime]
                        + self.encode("\n"))
            return seq + [ims] + self.encode("assistant\n")
        if f == "llama3":
            sh, eh = self._sp("<|start_header_id|>"), self._sp("<|end_header_id|>")
            eot = self._sp("<|eot_id|>")
            seq = []
            bos = self._sp("<|begin_of_text|>")
            if bos is not None: seq.append(bos)
            for m in msgs:
                seq += ([sh] + self.encode(m["role"]) + [eh]
                        + self.encode("\n\n" + m["content"]) + [eot])
            return seq + [sh] + self.encode("assistant") + [eh] + self.encode("\n\n")
        if f == "gemma":
            sot, eot = self._sp("<start_of_turn>"), self._sp("<end_of_turn>")
            seq = []
            for i, m in enumerate(turns):     # Gemma has no system role: fold it into turn 1
                body = m["content"]
                if i == 0 and sys_txt:
                    body = sys_txt + "\n\n" + body
                role = "model" if m["role"] == "assistant" else "user"
                seq += [sot] + self.encode(role + "\n" + body) + [eot] + self.encode("\n")
            return seq + [sot] + self.encode("model\n")
        if f == "mistral":
            bos = self._sp("<s>")
            seq = [bos] if bos is not None else []
            for i, m in enumerate(turns):
                body = m["content"]
                if i == 0 and sys_txt:
                    body = sys_txt + "\n\n" + body
                seq += (self.encode("[INST] " + body + " [/INST]") if m["role"] != "assistant"
                        else self.encode(body))
            return seq
        # plain transcript -- works for a base model or an unknown convention
        pre = (sys_txt + "\n\n") if sys_txt else ""
        body = "".join("%s: %s\n" % ("User" if m["role"] != "assistant" else "Assistant",
                                      m["content"]) for m in turns)
        return self.encode(pre + body + "Assistant:")

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
# Reasoning models emit their scratchpad inside a tagged span, and several of them OPEN that
# span in the chat template rather than in the generated text -- Qwen3 ends its prompt with
# "<think>\n", so the model itself only ever emits the CLOSING tag. Anything that pairs tags
# by looking at the output alone therefore renders raw reasoning as if it were the answer,
# right up until the closing tag finally arrives. Only the prompt settles which channel a
# stream starts in, so resolve it here, once, and hand callers text that is already labelled.
# Pairs are matched against the vocabulary and the rendered prompt, never assumed per model.
_THINK_TAGS = (("<think>", "</think>"), ("<thinking>", "</thinking>"),
               ("<reasoning>", "</reasoning>"), ("<reason>", "</reason>"),
               ("<|thinking|>", "<|/thinking|>"), ("<|think|>", "<|/think|>"))


class _Channels:
    """Split a decoded token stream into 'thinking' and 'content'.

    `opened` is the closing tag the prompt already committed us to, if the template opened a
    reasoning span itself. Tags survive token boundaries: a tag can arrive split across any
    number of chunks, so text that is still a possible prefix of one is held back rather than
    emitted into the wrong channel."""

    def __init__(self, opened=None):
        self.close = opened
        self.ch = "thinking" if opened else "content"
        self.buf = ""

    @staticmethod
    def _prefix_len(s, tags):
        """How many trailing chars of `s` could still grow into one of `tags`."""
        for n in range(min(len(s), max(len(t) for t in tags) - 1), 0, -1):
            tail = s[-n:]
            if any(t.startswith(tail) for t in tags):
                return n
        return 0

    def feed(self, text):
        """Consume a chunk, return a list of (channel, text) with empty pieces dropped."""
        self.buf += text
        out = []
        while True:
            tags = [self.close] if self.close else [o for o, _ in _THINK_TAGS]
            hit = min(((self.buf.find(t), t) for t in tags if self.buf.find(t) >= 0),
                      default=None)
            if hit is None:
                break
            i, tag = hit
            if i:
                out.append((self.ch, self.buf[:i]))
            self.buf = self.buf[i + len(tag):]
            if self.close:                       # closing a span: back to the answer
                self.close = None; self.ch = "content"
            else:                                # the model opened one itself
                self.close = dict(_THINK_TAGS)[tag]; self.ch = "thinking"
        keep = self._prefix_len(self.buf, [self.close] if self.close
                                else [o for o, _ in _THINK_TAGS])
        if len(self.buf) > keep:
            out.append((self.ch, self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return [(c, t) for c, t in out if t]

    def flush(self):
        """Whatever is still held back when the stream ends -- a partial tag that never
        completed is just text."""
        out = [(self.ch, self.buf)] if self.buf else []
        self.buf = ""
        return out


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
# On the native path the head is never expanded to fp32, so the block size is not bounded by
# a working set. 61 small matmuls cost far more than a few large ones -- the head is the
# biggest tensor in the model and is multiplied every token -- but a block is one GPU buffer,
# so budget it in BYTES rather than rows: the same row count is 115 MB for one quantization
# and 230 MB for another, and the large end runs into what the backend will stage.
_HEAD_BYTES_NATIVE = 128 << 20


def _source_bits(infos, G):
    """Int width that preserves what a GGUF actually stores.

    The engine's kernels read one packed layout, so GGUF weights are converted into it --
    which means a width has to be chosen. Deriving it from the file's own bits-per-weight
    keeps `dtype="auto"` meaning the same thing it means for an AutoGPTQ directory: run at
    the precision that was downloaded, rather than silently taking an 8-bit file to 4.
    Measured from the block type holding the most weights, via its own byte count, so it
    needs no per-format table and no per-model knowledge.
    """
    weight = {}
    for t in infos:
        if not t["name"].startswith("blk."):
            continue
        n = 1
        for d in t["dims"]:
            n *= int(d)
        weight[t["type"]] = weight.get(t["type"], 0) + n
    if not weight:
        return 4
    # Weighted by element count, not tensor count: a block holds a handful of huge weight
    # matrices and many tiny F32 norms, so counting tensors would let the norms decide.
    dom = max(weight, key=weight.get)
    bpw = G.tensor_nbytes(dom, 4096) * 8.0 / 4096
    return 8 if bpw > 4.5 else 4


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
        # Sampling state, so a forward called outside generate() still works.
        self.mtp = None             # multi-token-prediction head, when the file has one
        self._seen = []
        self._gen_start = 0
        self._con_text = ""
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
    async def from_gptq(cls, base, lmax=None):
        """Stream + build from a served AutoGPTQ int4 dir (e.g. '/models/qwen7b-gptq')."""
        self = cls(base)
        try:
            return await self._from_gptq(lmax)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_gptq(self, lmax=None):
        from . import webio
        self.lmax = lmax
        self._shard_hdr = {}
        cfg = await webio.read_json(self.base + "config.json")
        self._apply_cfg(cfg)
        self.lmax = self._auto_lmax(lmax, cfg.get("max_position_embeddings"))
        self.gs = cfg["quantization_config"]["group_size"]; self.bits = cfg["quantization_config"]["bits"]
        tie = self.tie_embeddings

        self.tok = await self._hf_tokenizer(cfg)

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

    def _head_blk(self, ttype, H):
        """Rows per head block: a byte budget when nothing is expanded, a row count when it is."""
        from . import ggufload as G
        nm = G.GGML_NAMES.get(ttype)
        native = (getattr(self, "_weights", "native") == "native"
                  and wt.ggml_native_supported(nm) and H % wt._GGML_TYPES[nm][2] == 0)
        if not native:
            return _HEAD_BLK
        row = G.tensor_nbytes(ttype, H)
        rows = max(1, _HEAD_BYTES_NATIVE // max(row, 1) // _HEAD_BLK) * _HEAD_BLK
        return rows

    def _head_block(self, ttype, chunk, rows, H):
        """One block of the LM head as a Linear. Native when the type allows it, which keeps
        the head off the conversion path too -- it is (vocab, H), the largest single tensor
        in most models, and expanding it to fp32 is what forces the block-at-a-time walk."""
        from . import ggufload as G
        nm = G.GGML_NAMES.get(ttype)
        if (getattr(self, "_weights", "native") == "native"
                and wt.ggml_native_supported(nm) and H % wt._GGML_TYPES[nm][2] == 0):
            return wt.GGMLLinear(chunk, nm, H, rows)
        return self._gquant(G.dequant(ttype, chunk, rows * H).reshape(rows, H))

    async def _hf_tokenizer(self, cfg):
        """Tokenizer for an HF directory: vocab, merges, and the model's own chat template.

        tokenizer_config.json is where a served model states its prompt layout and its
        added tokens; reading it is what makes an unfamiliar model's turns come out right
        instead of approximately right."""
        from . import webio
        vocab = await webio.read_json(self.base + "vocab.json")
        merges = (await webio.read_text(self.base + "merges.txt")).split("\n")
        _e = cfg.get("eos_token_id")
        eos = _e if isinstance(_e, list) else ([_e] if _e is not None else [])
        tc = {}
        try:
            tc = await webio.read_json(self.base + "tokenizer_config.json")
        except Exception:
            pass                                   # optional -- fall back to the format probe
        added = tc.get("added_tokens_decoder") or {}
        ctrl = [v.get("content") for v in added.values()
                if isinstance(v, dict) and v.get("content")]
        tok = BPETokenizer(vocab, [m for m in merges[1:] if m and not m.startswith("#")],
                           eos_ids=eos, chat_template=tc.get("chat_template"), control=ctrl)
        await tok.prepare_template()
        await self._load_gen_defaults()
        return tok

    async def _load_gen_defaults(self):
        """Sampling settings the model ships with (generation_config.json).

        A model states what it should be sampled at -- Qwen3 asks for temperature 0.6 /
        top_p 0.95 / top_k 20 -- and ignoring that makes it look worse than it is. These
        are defaults: anything passed to generate() still wins. Read by name, so a field
        the sampler does not implement is simply not picked up."""
        from . import webio
        try:
            gc = await webio.read_json(self.base + "generation_config.json")
        except Exception:
            return                                  # optional file
        keys = ("temperature", "top_p", "top_k", "do_sample", "repetition_penalty", "min_p",
                "max_new_tokens", "max_length", "min_new_tokens", "presence_penalty",
                "frequency_penalty", "stop")
        got = {k: gc[k] for k in keys if gc.get(k) is not None}
        if got:
            self.gen_defaults = dict(getattr(self, "gen_defaults", {}) or {}, **got)

    async def _gexperts(self, name):
        """Yield each expert of a stacked MoE tensor as a ready Linear, one at a time.

        llama.cpp packs a layer's experts into a single `ffn_*_exps` tensor. The packed
        bytes are fetched in ONE range request -- a read per expert would turn three reads
        per layer into hundreds -- and then split, which is exact: quantization blocks run
        along `in` and out*in is a whole number of blocks, so every per-expert offset is
        block-aligned.

        On the native path each expert's slice goes straight to the GPU in its own encoding.
        Otherwise it is dequantized and requantized one expert at a time, because the whole
        stack as fp32 is ~800 MB on a 128-expert layer; peak stays at one expert.
        """
        from . import ggufload as G
        t = self._ginfo[name]
        ne, out_d, in_d = (int(d) for d in reversed(t["dims"]))
        per = out_d * in_d
        nb = G.tensor_nbytes(t["type"], per)                  # bytes per expert
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb * ne - 1)
        nm = G.GGML_NAMES.get(t["type"])
        native = (getattr(self, "_weights", "native") == "native"
                  and wt.ggml_native_supported(nm)
                  and in_d % wt._GGML_TYPES[nm][2] == 0)
        for e in range(ne):
            chunk = raw[e * nb:(e + 1) * nb]
            if native:
                yield wt.GGMLLinear(chunk, nm, in_d, out_d)
            else:
                yield self._gquant(G.dequant(t["type"], chunk, per).reshape(out_d, in_d))

    async def _gload_native(self, name, bias=None):
        """Upload a GGUF weight unchanged, for a kernel that decodes it while multiplying.

        No dequantize, no requantize, no fp32 intermediate: the packed tensor is a few tens
        of MB where its fp32 expansion would be hundreds, so it goes up in one piece.
        Returns None when the tensor is not something the native kernels handle -- a type
        without a decode fragment, a non-2D tensor, or a K that is not block-aligned -- and
        the caller falls back to converting it."""
        from . import ggufload as G
        t = self._ginfo[name]
        dims = [int(d) for d in reversed(t["dims"])]
        if len(dims) != 2:
            return None
        nm = G.GGML_NAMES.get(t["type"])
        if not wt.ggml_native_supported(nm):
            return None
        N, K = dims
        if K % wt._GGML_TYPES[nm][2]:
            return None
        row = G.tensor_nbytes(t["type"], K)
        off = self._gds + t["offset"]
        return wt.GGMLLinear(await self._grng(off, off + N * row - 1), nm, K, N, bias)

    async def _gload_quant(self, name, bias=None):
        """Read a GGUF weight and quantize it without ever holding it whole.

        A 27B feed-forward weight is 89M values: 356 MB as fp32, and the i-quant
        dequantizers build an intermediate of the same size again, which a 32-bit heap will
        not give. Nothing needs the whole thing -- each output column is quantized
        independently, and GGUF blocks run along the input dimension, so a band of output
        rows is both self-contained and block-aligned. This walks those bands: read, expand,
        quantize, keep only the packed result. Peak is one band, whatever the tensor's size.
        """
        from . import ggufload as G
        if getattr(self, "_weights", "native") == "native":
            got = await self._gload_native(name, bias)
            if got is not None:
                return got
        if getattr(self, "_qbits", None) == 0:                # fp16 path wants it whole
            return self._gquant(await self._gload(name), bias)
        t = self._ginfo[name]
        dims = [int(d) for d in reversed(t["dims"])]
        if len(dims) != 2:
            return self._gquant(await self._gload(name), bias)
        N, K = dims                                            # (out, in)
        per = 32 // self.bits
        kmul = self.gs if self.gs % per == 0 else self.gs * per
        Kp = K + (-K) % kmul
        Np = N + (-N) % per
        row = G.tensor_nbytes(t["type"], K)                    # bytes per output row
        off = self._gds + t["offset"]
        band = max(per, (_READ_BYTES // max(K * 4, 1)) // per * per) or per
        qws, qzs, scs = [], [], []
        for r0 in range(0, Np, band):
            r1 = min(Np, r0 + band)
            live = max(0, min(N, r1) - r0)                     # real rows in this band
            if live:
                raw = await self._grng(off + r0 * row, off + (r0 + live) * row - 1)
                W = G.dequant(t["type"], raw, live * K).reshape(live, K)
                del raw
            else:
                W = np.zeros((0, K), np.float32)
            if live < r1 - r0 or Kp != K:                      # pad to the packing multiple
                W = np.pad(W, ((0, (r1 - r0) - live), (0, Kp - K)))
            qw, qz, sc, _, _ = wt._gptq_quantize(W, self.gs, self.bits, from_out_in=True)
            del W
            qws.append(qw); qzs.append(qz); scs.append(sc)
        qw = np.concatenate(qws, axis=1); del qws
        qz = np.concatenate(qzs, axis=1); del qzs
        sc = np.concatenate(scs, axis=1); del scs
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, self.gs, self.bits)

    def _gquant(self, W_out_in, bias=None):
        """fp32 (out,in) weight -> a linear layer. Quantized to int`bits` by default; with
        `self._qbits == 0` (dtype="fp16") it stays unquantized, which also makes GGUF models
        runnable without a GPU (the int kernel is GPU-only)."""
        if getattr(self, "_qbits", None) == 0:
            return wt.UnquantizedLinear(W_out_in, bias)
        # Handed over in its (out, in) layout: the quantizer transposes one column block at
        # a time, so the full (in, out) copy -- 356 MB on a 27B feed-forward weight, on top
        # of the 356 MB the tensor already occupies -- is never built.
        N, K = W_out_in.shape; per = 32 // self.bits
        kmul = self.gs if self.gs % per == 0 else self.gs * per
        Kp = K + (-K) % kmul; Np = N + (-N) % per
        src = W_out_in
        if Kp != K or Np != N:
            src = np.pad(src, ((0, Np - N), (0, Kp - K)))
        qw, qz, sc, _, _ = wt._gptq_quantize(src, self.gs, self.bits, from_out_in=True)
        del src
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, self.gs, self.bits)

    @classmethod
    async def from_gguf(cls, url, lmax=None, bits=None, quantize=True, weights="native"):
        """Load a llama.cpp GGUF, dequantizing + requantizing weights to int`bits` (4 or 8) so
        they run on the same capture-accelerated engine. `url` is the served .gguf file.

        `lmax=None` runs at the context length the file itself declares (`{arch}.context_length`),
        trimmed only if the KV cache for it would exceed the memory budget; pass a number to
        force a specific size.

        `bits=None` (the default) takes the width from the file itself, so an 8-bit GGUF
        runs at int8 instead of being quietly reduced to int4. Note that the conversion
        itself is a kernel-format limitation, not a property of the format: the engine
        reads one packed layout, so k-quant super-blocks are re-approximated by the
        kernel's group scale/zero scheme. Running GGUF blocks natively would need a
        kernel per quantization type.

        **Architecture-generic**: shape/rope/eps are read from GGUF's own `{arch}.*` metadata
        keys (the format's convention), and optional pieces are detected from the tensors —
        QK-norm (`attn_q_norm`/`attn_k_norm`, e.g. Qwen3) and sparse-MoE (`ffn_gate_inp` +
        `ffn_*_exps`). So any Llama-style decoder GGUF (llama / qwen2 / qwen3 / mistral / …)
        loads without a per-model branch. A GGUF whose weights use a quantization type this
        SDK cannot dequantize (e.g. the IQ i-quants) is rejected with a clear error."""
        self = cls(None)
        try:
            return await self._from_gguf(url, lmax, bits, quantize, weights)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_gguf(self, url, lmax=None, bits=None, quantize=True, weights="native"):
        from . import ggufload as G
        self.lmax = lmax; self._gguf = url
        size = 12 << 20
        while True:
            buf = await self._grng(0, size - 1)
            try:
                _, meta, infos, ds = G.parse_header(buf); break
            except EOFError:
                size <<= 1
                if size > (128 << 20):
                    raise
        # `bits=None` follows the file (see _source_bits); an explicit width still wins.
        self.bits = int(bits) if bits else _source_bits(infos, G)
        # `quantize=False` (dtype="fp16") keeps the dequantized weights unquantized, so a GGUF
        # runs on the numpy/CPU path too — the int kernel is GPU-only.
        self._qbits = self.bits if quantize else 0
        # "native" multiplies straight out of the file's own encoding; "requant" converts
        # every weight to the int4/int8 kernel format first, which is what this did before
        # the native kernels existed. Native is the default: it skips the conversion (the
        # bulk of a load) and the second rounding that came with it. Types without a native
        # decode fall back to conversion per tensor either way.
        if weights not in ("native", "requant"):
            raise ValueError("weights must be 'native' or 'requant', got %r" % (weights,))
        self._weights = weights
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
        self.H = int(m("embedding_length"))
        # `block_count` counts multi-token-prediction blocks too, but those are a separate
        # head (their tensors carry a `nextn.` prefix), not part of the decoder stack --
        # running one as a normal layer would silently corrupt every generation. The count
        # is declared in the file, so this needs no per-model knowledge.
        self.L = int(m("block_count")) - int(m("nextn_predict_layers", default=0,
                                               required=False) or 0)
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
        # The size the model itself declares (`{arch}.context_length`), capped only by the
        # KV-cache memory budget; an explicit `lmax` wins over both.
        self.lmax = self._auto_lmax(lmax, m("context_length", default=None, required=False))
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
        _toks = meta["tokenizer.ggml.tokens"]
        _tt = meta.get("tokenizer.ggml.token_type") or []
        # GGML_TOKEN_TYPE_CONTROL / _USER_DEFINED: the file says which tokens are markers,
        # so BPE never has to guess from what a token looks like.
        _ctrl = [_toks[i] for i, t in enumerate(_tt) if int(t) in (3, 4)]
        self.tok = BPETokenizer({t: i for i, t in enumerate(_toks)},
                                meta.get("tokenizer.ggml.merges", []), eos_ids=_eos,
                                chat_template=meta.get("tokenizer.chat_template"), control=_ctrl)
        await self.tok.prepare_template()
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
        _ehb = self._head_blk(et["type"], H)
        rpr = max(_ehb, (_READ_BYTES // erow) // _ehb * _ehb)
        for v0 in range(0, V, rpr):
            v1 = min(V, v0 + rpr)
            raw = await self._grng(eoff + v0 * erow, eoff + v1 * erow - 1)
            ebuf[v0 * erow:v1 * erow] = np.frombuffer(raw, np.uint8)
            if tied:                                          # (blk,H) = (out,in)
                hb = self._head_blk(et["type"], H)
                for b0 in range(v0, v1, hb):
                    b1 = min(v1, b0 + hb)
                    self.head.append(self._head_block(
                        et["type"], raw[(b0 - v0) * erow:(b1 - v0) * erow], b1 - b0, H))
        self.embed = _QuantRows(memoryview(ebuf.data).cast("B"), et["type"], H, erow, G.dequant)
        if not tied:                                          # separate lm_head, also streamed
            ot = self._ginfo["output.weight"]
            orow = G.tensor_nbytes(ot["type"], H); ooff = self._gds + ot["offset"]
            self.head = []
            _ohb = self._head_blk(ot["type"], H)
            orpr = max(_ohb, (_READ_BYTES // orow) // _ohb * _ohb)
            for v0 in range(0, V, orpr):                      # large reads, small expansions
                v1 = min(V, v0 + orpr)
                raw = await self._grng(ooff + v0 * orow, ooff + v1 * orow - 1)
                hb = self._head_blk(ot["type"], H)
                for b0 in range(v0, v1, hb):
                    b1 = min(v1, b0 + hb)
                    self.head.append(self._head_block(
                        ot["type"], raw[(b0 - v0) * orow:(b1 - v0) * orow], b1 - b0, H))
        self.layers = []
        for i in range(self.L):
            p = "blk.%d." % i
            async def qb(nm, _p=p):                            # weight + optional bias -> QuantizedLinear
                b = (await self._gload(_p + nm + ".bias")) if (_p + nm + ".bias") in self._ginfo else None
                return await self._gload_quant(_p + nm + ".weight", b)
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
                    async for lin in self._gexperts(p + nm + ".weight"):
                        experts[e][key] = lin; e += 1
                lay["moe"] = {"gate": await self._gload_quant(p + "ffn_gate_inp.weight"),
                              "top_k": self.top_k or 2, "norm_topk": True, "experts": experts}
            else:
                lay["gate"] = await qb("ffn_gate"); lay["up"] = await qb("ffn_up")
                lay["down"] = await qb("ffn_down")
            self.layers.append(lay)
        self.final_norm = wt.Tensor(await self._gload("output_norm.weight"))
        self.mtp = await self._gload_mtp()
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._init_state()
        return self

    async def _gload_mtp(self):
        """Load a multi-token-prediction head, if the file ships one. None otherwise.

        Found by tensor name rather than by architecture: a NextN/MTP block is an extra
        decoder block past the trunk whose distinguishing tensor is `nextn.eh_proj`, and
        DeepSeek-V3, Qwen3.5 and GLM all lay it out that way. Everything else in the block
        is an ordinary attention layer, and the pieces it does not carry -- its own
        embedding table, its own LM head -- fall back to the trunk's.
        """
        pre = next(("blk.%d." % i for i in range(self.L, self.L + 8)
                    if ("blk.%d.nextn.eh_proj.weight" % i) in self._ginfo), None)
        if pre is None:
            return None

        async def qb(nm):
            b = ((await self._gload(pre + nm + ".bias"))
                 if (pre + nm + ".bias") in self._ginfo else None)
            return await self._gload_quant(pre + nm + ".weight", b)

        async def opt_qb(nm):
            return (await qb(nm)) if (pre + nm + ".weight") in self._ginfo else None

        async def opt_t(nm):
            n = pre + nm + ".weight"
            return wt.Tensor(await self._gload(n)) if n in self._ginfo else None

        post = next((n for n in ("ffn_norm", "post_attention_norm")
                     if (pre + n + ".weight") in self._ginfo), None)
        m = {"eh": await qb("nextn.eh_proj"),
             "enorm": await opt_t("nextn.enorm"), "hnorm": await opt_t("nextn.hnorm"),
             "in_ln": await opt_t("attn_norm"),
             "post_ln": (await opt_t(post)) if post else None,
             "qkv": await opt_qb("attn_qkv"),
             "q": await opt_qb("attn_q"), "k": await opt_qb("attn_k"),
             "v": await opt_qb("attn_v"), "o": await opt_qb("attn_output"),
             "qn": await opt_t("attn_q_norm"), "kn": await opt_t("attn_k_norm"),
             "gate": await opt_qb("ffn_gate"), "up": await opt_qb("ffn_up"),
             "down": await opt_qb("ffn_down"),
             "head_norm": await opt_t("nextn.shared_head_norm"),
             "head": await opt_qb("nextn.shared_head_head")}
        if m["eh"] is None or m["q"] is None or m["o"] is None:
            return None                       # incomplete; run without it
        return m

    def mtp_logits(self, h_last, token, pos, cache):
        """Predict the token AFTER `token`, from the trunk's last hidden state.

        `h_last` is the trunk's final hidden for the position that produced `token`; the
        head combines it with that token's embedding and runs one more decoder block. This
        is the draft step of speculative decoding -- one layer against the trunk's many.
        """
        m = self.mtp
        if m is None:
            return None
        e = wt.Tensor(np.asarray(self.embed[int(token)], np.float32).reshape(1, self.H))
        pair = wt.cat([self._rms(e, m["enorm"]) if m["enorm"] is not None else e,
                       self._rms(h_last, m["hnorm"]) if m["hnorm"] is not None else h_last],
                      axis=-1)
        h = m["eh"](pair)
        lay = {"q": m["q"], "k": m["k"], "v": m["v"], "o": m["o"],
               "qn": m["qn"], "kn": m["kn"], "gate": m["gate"], "up": m["up"],
               "down": m["down"]}
        x = self._rms(h, m["in_ln"]) if m["in_ln"] is not None else h
        c, sn = self._rope_np(pos, 1)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(sn)
        q, k, v = self._qkv(lay, x, 1)
        q = q * cos_t + self._rot(q) * sin_t
        k = k * cos_t + self._rot(k) * sin_t
        sc = 1.0 / math.sqrt(self.HD)
        h = h + self._attn_out(lay, cache.attn(0, q, k, v, pos, scale=sc), 1)
        if m["post_ln"] is not None:
            x = self._rms(h, m["post_ln"])
            h = h + self._mlp(lay, x)
        hn = m["head_norm"] if m["head_norm"] is not None else self.final_norm
        fin = self._rms(h, hn)
        if m["head"] is not None:
            return np.asarray(m["head"](fin).numpy()).reshape(-1)
        return self._logits(wt.Tensor(wt._contig(fin.data[-1:])))

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
    async def from_fp16(cls, base, lmax=None, quantize=None):
        """Load a plain fp16/bf16 HF safetensors dir (no `quantization_config`). Runs the
        model UNquantized by default (fp16 weights, fp32 compute); pass `quantize=4` or `8`
        to instead quantize every linear to int on load (same served fp16 model → int
        inference). Same engine (KV cache, capture/replay, generate) as the int loaders."""
        self = cls(base)
        try:
            return await self._from_fp16(lmax, quantize)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_fp16(self, lmax=None, quantize=None):
        from . import webio
        self.lmax = lmax; self._shard_hdr = {}
        self._qbits = int(quantize) if quantize else 0
        if self._qbits:
            self.bits = self._qbits                        # gs keeps the default 128
        cfg = await webio.read_json(self.base + "config.json")
        self._apply_cfg(cfg)
        self.lmax = self._auto_lmax(lmax, cfg.get("max_position_embeddings"))
        tie = self.tie_embeddings

        self.tok = await self._hf_tokenizer(cfg)

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
        # One fused dispatch where the backend has it. Written as an expression this is six
        # kernels, and on this stack a dispatch costs about the same whatever its size, so
        # the two norms in every layer add up to a large share of a decode step.
        if getattr(self, "_gpu", False) and wt._RMS_FUSED:   # _gpu is unset while building
            r = wt.rmsnorm(x, w, self.eps)
            if r is not None:
                return r
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

    def _rope1(self, t):
        """Rotary embedding for the single decode position, fused into one dispatch where
        the backend allows it (see wt.rope_decode). The expression form is ~8 launches."""
        if getattr(self, "_gpu", False) and wt._ROPE_FUSED:
            r = wt.rope_decode(t, self.cos_b, self.sin_b, self.HD,
                               getattr(self, "rope_dim", self.HD))
            if r is not None:
                return r
        return t * self.cos_b + self._rot(t) * self.sin_b

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
        """Turn logits into a token id using the current sampling parameters.

        Greedy when `temperature <= 0` (or no sampling params were given), otherwise
        temperature + top-k/top-p/min-p sampling, with an optional repetition penalty and
        an optional output constraint. Parameters come from `generate(...)`, from the
        model's own generation_config.json, or from `load(...)`."""
        sp = getattr(self, "_sampling", None) or {}
        lg = np.asarray(logits, np.float32)
        rp = float(sp.get("repetition_penalty", 1.0) or 1.0)
        if rp != 1.0 and self._seen:
            idx = np.unique(np.asarray(self._seen, np.int64))
            idx = idx[(idx >= 0) & (idx < lg.size)]
            v = lg[idx]
            lg = lg.copy()
            lg[idx] = np.where(v > 0, v / rp, v * rp)   # the usual asymmetric form
        # OpenAI-style additive penalties, which are a different knob from the multiplicative
        # repetition_penalty above and compose with it: presence is a flat charge for having
        # appeared at all, frequency scales with the count.
        pp = float(sp.get("presence_penalty", 0.0) or 0.0)
        fp = float(sp.get("frequency_penalty", 0.0) or 0.0)
        if (pp or fp) and self._seen:
            ids = np.asarray(self._seen, np.int64)
            ids = ids[(ids >= 0) & (ids < lg.size)]
            if ids.size:
                idx, cnt = np.unique(ids, return_counts=True)
                lg = lg.copy()
                lg[idx] -= pp + fp * cnt.astype(np.float32)
        mn = int(sp.get("min_new_tokens", 0) or 0)
        if mn and len(self._seen) - self._gen_start < mn and self.tok.eos_ids:
            eos = np.asarray(self.tok.eos_ids, np.int64)
            eos = eos[(eos >= 0) & (eos < lg.size)]
            lg = lg.copy()
            lg[eos] = -np.inf                           # not allowed to stop yet
        con = sp.get("constraint")
        tok = (self._pick_constrained(lg, con, sp) if con is not None
               else (self._sample(lg, sp) if sp.get("do_sample") else int(lg.argmax())))
        self._seen.append(tok)
        if con is not None:
            self._con_text += self.tok.decode([tok])
        return tok

    def _sample(self, lg, sp):
        from . import lm_engine
        mp = float(sp.get("min_p", 0.0) or 0.0)
        if mp > 0:                              # keep only what is within min_p of the top
            m = lg.max()
            keep = lg >= m + math.log(max(mp, 1e-9))
            lg = np.where(keep, lg, -np.inf)
        return int(lm_engine.sample_nucleus(
            lg / max(float(sp.get("temperature", 1.0) or 1.0), 1e-5),
            top_p=float(sp.get("top_p", 1.0) or 1.0),
            top_k=int(sp.get("top_k", 0) or 10 ** 9), rng=sp.get("rng")))

    def _pieces(self):
        """id -> text for the whole vocabulary, built once. Only the widened constraint search
        needs it, and only for models that are actually asked to satisfy a constraint."""
        if getattr(self, "_piece_tab", None) is None:
            dec = self.tok.decode
            self._piece_tab = [dec([i]) for i in range(len(self.tok.dec))]
        return self._piece_tab

    def _pick_constrained(self, lg, con, sp):
        """Sample from just the candidates the constraint accepts.

        The likely handful is checked first: scoring a 250k-token vocabulary through a
        constraint costs more than the model step that produced the logits, and usually buys
        nothing, since what matters is which tokens are allowed rather than the exact
        ordering of ones the model was never going to pick.

        But the window WIDENS instead of giving up. The opening tokens of a constrained reply
        are often ones the model ranks nowhere -- asked in prose for JSON it wants to answer
        in prose, and `{` may sit far outside the top 64. Falling back to an unconstrained
        pick there does not just lose one token: that token makes every later prefix invalid,
        so no candidate is ever acceptable again and the constraint is silently dropped for
        the whole reply. Widening to the full vocabulary is the difference between a
        constraint that holds and one that only appears to. The plain pick remains the last
        resort, for a constraint nothing at all can satisfy -- that must not hang."""
        n = int(lg.size)
        k0 = max(1, int(sp.get("constraint_candidates", 64) or 64))
        allowed, order = [], None
        for k in (min(k0, n), min(k0 * 16, n), n):
            if k < n:
                idx = np.argpartition(-lg, k - 1)[:k]
                order = idx[np.argsort(-lg[idx])]
                pieces = None
            else:
                order = np.argsort(-lg)
                pieces = self._pieces()
            allowed = [int(t) for t in order
                       if con.allows(self._con_text,
                                     pieces[int(t)] if pieces is not None and int(t) < len(pieces)
                                     else self.tok.decode([int(t)]))]
            if allowed:
                break
        if not allowed:
            return int(lg.argmax())
        if not sp.get("do_sample"):
            return allowed[0]
        sub = np.full_like(lg, -np.inf)
        sub[np.asarray(allowed, np.int64)] = lg[np.asarray(allowed, np.int64)]
        return self._sample(sub, sp)

    # The KV cache is the only memory that grows with context: fp32, K and V, NKV × HD per
    # token per full-attention layer (recurrent layers keep a fixed-size state instead). A
    # context nobody asked for must not eat memory nobody budgeted, so an undeclared size
    # is trimmed to this; an explicit `lmax` always wins.
    _KV_BUDGET = 2 << 30

    # Decode cost used to scale with the context because a step scanned the whole KV buffer
    # and re-uploaded a full-length mask every token. Both are gone -- the fused attention
    # reads the live length from the step control block, and the mask is only built for the
    # general path -- so a step now costs the same at 512 as at 4096 (measured: 35.6 vs
    # 35.0 ms). Context is therefore a memory question again, and only memory caps it.

    def _auto_lmax(self, requested, declared):
        """The context size to load with: an explicit request, else what the model declares,
        capped by the KV memory budget."""
        n_full = sum(1 for t in getattr(self, "layer_types", []) if t == "full_attention")
        n_full = n_full or self.L
        per_tok = 2 * n_full * self.NKV * self.HD * 4
        cap = max(512, self._KV_BUDGET // per_tok)
        if requested:
            got = max(512, min(int(requested), cap))
            if got < int(requested):
                print("webtorch: context %d -> %d tokens (KV cache budget %.1f GB)"
                      % (int(requested), got, self._KV_BUDGET / 2 ** 30))
            return got
        want = int(declared) if declared else 2048
        got = max(512, min(want, cap))
        if got < want:
            print("webtorch: context %d -> %d tokens (KV cache budget %.1f GB)"
                  % (want, got, self._KV_BUDGET / 2 ** 30))
        return got

    def _plan_length(self, ids, max_new, max_length, truncate):
        """Reconcile the requested lengths with what this model can actually hold.

        `lmax` -- the KV cache the model was loaded with -- is a hard ceiling on prompt plus
        generation, and running past it silently corrupts the cache. A prompt that does not
        fit is either truncated from the front (a chat history wants its most recent turns)
        or refused, and `max_new` is clipped to whatever room is left.

        No `max_new` at all means "as much as the model allows": its own generation_config
        when it ships one, otherwise the rest of the context — the model still stops itself
        at its end token; the number only bounds it."""
        lmax = int(self.lmax)
        d = getattr(self, "gen_defaults", {}) or {}
        if max_new is not None:
            mx = int(max_new)
        elif d.get("max_new_tokens"):
            mx = int(d["max_new_tokens"])
        else:
            mx = max(1, lmax - len(ids))
        if max_length is None:
            max_length = d.get("max_length")
        if max_length:
            mx = min(mx, int(max_length) - len(ids))
        ids = list(ids)
        if len(ids) >= lmax:
            if not truncate:
                raise ValueError(
                    "prompt is %d tokens but this model was loaded with a %d-token context; "
                    "load it with a larger lmax, shorten the prompt, or pass truncate=True"
                    % (len(ids), lmax))
            keep = max(1, lmax - max(1, min(mx, lmax // 4)))
            ids = ids[-keep:]
        return ids, max(1, min(mx, lmax - len(ids)))

    def _stop_now(self):
        """True when an output constraint says the text is complete."""
        con = (getattr(self, "_sampling", None) or {}).get("constraint")
        return bool(con is not None and con.finished(self._con_text))

    def _set_sampling(self, temperature=None, top_p=None, top_k=None, seed=None, do_sample=None,
                      repetition_penalty=None, min_p=None, constraint=None,
                      min_new_tokens=None, prompt_ids=None, presence_penalty=None,
                      frequency_penalty=None, stop=None, **_ignored):
        """Install generation parameters. Defaults from the model's generation_config.json (or
        from load()) are kept; anything passed to generate() overrides them for that call."""
        from . import constrain
        base = dict(getattr(self, "gen_defaults", {}) or {})
        for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k),
                     ("seed", seed), ("do_sample", do_sample), ("min_p", min_p),
                     ("repetition_penalty", repetition_penalty),
                     ("presence_penalty", presence_penalty),
                     ("frequency_penalty", frequency_penalty),
                     ("min_new_tokens", min_new_tokens)):
            if v is not None:
                base[k] = v
        if base.get("do_sample") is None:
            base["do_sample"] = float(base.get("temperature", 0) or 0) > 0
        if base.get("seed") is not None:
            base["rng"] = np.random.default_rng(int(base["seed"]))
        stops = stop if stop is not None else base.get("stop")
        if isinstance(stops, str):
            stops = [stops]
        base["stop"] = stops
        con = constrain.build(constraint if constraint is not None else base.get("constraint"))
        if stops:
            sc = constrain.StopConstraint(stops)
            con = sc if con is None else constrain.AllOf([con, sc])
        if con is not None:
            con.reset()
        base["constraint"] = con
        self._sampling = base
        # Repetition penalty counts the prompt too, as it does elsewhere; the constraint
        # sees only what this generation produced.
        self._seen = list(prompt_ids or [])
        self._gen_start = len(self._seen)
        self._con_text = ""
        return base

    def _init_state(self):
        L, NKV, HD, LMAX, H = self.L, self.NKV, self.HD, self.lmax, self.H
        NH = self.NH
        # recurrent states for linear-attention layers (fixed size, independent of length)
        self.lin_state = [lay["linear"].new_state() if lay.get("linear") else None
                          for lay in getattr(self, "layers", [])]
        # Only full-attention layers hold a context-length K/V cache; a recurrent layer
        # carries its fixed-size state instead, so allocating a cache for it would be pure
        # waste — on a stack that is mostly linear that is most of the KV memory.
        self._kv_i = []; n = 0
        for i in range(L):
            if self._is_linear_layer(i):
                self._kv_i.append(-1)
            else:
                self._kv_i.append(n); n += 1
        if self._gpu:                                  # capture path: fixed scatter cache + persistent inputs
            self.Kc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(n)]
            self.Vc = [wt.Tensor(wt._zeros((NKV, LMAX, HD))) for _ in range(n)]
            self.h_in = wt.Tensor(np.zeros((1, H), np.float32))
            self.cos_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.sin_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.mask_b = wt.Tensor(np.zeros((1, 1, LMAX), np.float32))
            self.ctl = xp.asarray(np.array([0, 1, NKV, HD, LMAX], np.int32))
            # Does the fused decode attention cover this model's shapes? If it does, it reads
            # the length from `ctl` and never looks at the mask -- and the mask is the single
            # most expensive thing in a decode step, because keeping it current means
            # allocating and uploading LMAX floats EVERY token. That upload, not the
            # attention itself, is what made a large context slow down short conversations.
            self._fused_attn = False
            if wt._GQA_FUSED and n:
                probe = wt.gqa_decode(wt.Tensor(np.zeros((NH, 1, HD), np.float32)),
                                      self.Kc[0], self.Vc[0], self.mask_b, 1.0, ctl=self.ctl)
                self._fused_attn = probe is not None

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
        # Hand the Tensor over as-is: the layer's decode path stays on the device, and
        # pulling it to the host here would undo that.
        y = lay["linear"].forward(x, st)
        return y if isinstance(y, wt.Tensor) else wt.Tensor(np.asarray(y, np.float32))

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
                K, V = self.Kc[self._kv_i[i]], self.Vc[self._kv_i[i]]
                wt.kv_write(K.data, wt._contig(k).data, 0, T, NKV, HD, LMAX)
                wt.kv_write(V.data, wt._contig(v).data, 0, T, NKV, HD, LMAX)
                o = wt.gqa_attention(q, K, V, mask, scale=sc)
                h = h + self._attn_out(lay, o, T)
            x = self._rms(h, lay["post_ln"])
            h = h + self._mlp(lay, x)
        return self._head_argmax(wt.Tensor(wt._contig(self._rms(h, self.final_norm).data[-1:])))

    def _set_inputs(self, token, pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.lmax
        self.h_in.data.buffer.set_data(np.asarray(self.embed[token], np.float32))
        c, s = self._rope_np(pos)
        self.cos_b.data.buffer.set_data(c.reshape(-1)); self.sin_b.data.buffer.set_data(s.reshape(-1))
        if not getattr(self, "_fused_attn", False):    # only the general path reads the mask
            m = np.zeros((1, 1, LMAX), np.float32); m[0, 0, pos + 1:] = -1e9
            self.mask_b.data.buffer.set_data(m)
        self.ctl.buffer.set_data(np.array([pos, 1, NKV, HD, LMAX], np.int32))

    def _decode_fwd(self):
        H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.lmax
        sc = 1.0 / math.sqrt(HD); h = self.h_in
        for i, lay in enumerate(self.layers):
            x = self._rms(h, lay["in_ln"])
            if self._is_linear_layer(i):
                # A recurrent mixer is capturable once its step runs on the device: the
                # commands are the same every token, and the state it reads and writes in
                # place is a buffer like any other.
                h = h + self._linear_mixer(i, lay, x, 1)
            else:
                q, k, v = self._qkv(lay, x, 1)
                q = self._rope1(q); k = self._rope1(k)
                K, V = self.Kc[self._kv_i[i]], self.Vc[self._kv_i[i]]
                wt.kv_write(K.data, wt._contig(k).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
                wt.kv_write(V.data, wt._contig(v).data, 0, 1, NKV, HD, LMAX, ctl=self.ctl)
                # One dispatch for the single decode position; falls back for anything the
                # fused kernel does not cover.
                # `ctl` carries the position, so the fused kernel scans only what the
                # conversation has actually filled -- decode speed follows the conversation,
                # not the context the model was loaded with.
                o = (wt.gqa_decode(q, K, V, self.mask_b, sc, ctl=self.ctl)
                     if wt._GQA_FUSED else None)
                if o is None:
                    o = wt.gqa_attention(q, K, V, self.mask_b, scale=sc)
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

    def stream(self, prompt=None, max_new=None, system="You are a helpful assistant.",
               messages=None, tools=None, ids=None, temperature=None, top_p=None,
               top_k=None, seed=None, do_sample=None, repetition_penalty=None, min_p=None,
               constraint=None, max_length=None, min_new_tokens=None, truncate=True,
               enable_thinking=None, presence_penalty=None, frequency_penalty=None,
               stop=None, channels=False, **chat_kw):
        """Streaming decode: yield each new token's text as it is produced (render live,
        bounded memory). WebGPU replays a captured step per token; WebGL grows a cache.
        Takes the same parameters as `generate`.

        `channels=True` yields `{"channel": "thinking"|"content", "text": ...}` instead of
        bare text. A reader cannot work this out for itself: models whose template opens the
        reasoning span emit only the CLOSING tag, so pairing tags in the output alone shows
        reasoning as if it were the answer until that tag lands. Tags are also split across
        token boundaries at will, and are held back here rather than leaked into the wrong
        channel. Models without reasoning simply yield everything as "content".

        When the stream ends, `self.last_stream` holds `{n, truncated, ttft_s, tok_s}` —
        `truncated` says the length limit, not the model, ended the reply."""
        box = {}
        it = self._stream_raw(prompt, max_new, system, messages, tools, ids, temperature,
                              top_p, top_k, seed, do_sample, repetition_penalty, min_p,
                              constraint, max_length, min_new_tokens, truncate,
                              enable_thinking, presence_penalty, frequency_penalty, stop,
                              box, **chat_kw)
        return self._stream_channels(it, box) if channels else it

    def _stream_channels(self, it, box):
        """Label a raw text stream by channel. `box` is filled by the generator once it has
        the prompt, which is the only thing that says whether reasoning is already open."""
        ch = None
        for piece in it:
            if ch is None:
                ch = _Channels(box.get("close"))
            for c, t in ch.feed(piece):
                yield {"channel": c, "text": t}
        for c, t in (ch.flush() if ch is not None else ()):
            yield {"channel": c, "text": t}

    def _stream_raw(self, prompt=None, max_new=None, system="You are a helpful assistant.",
                    messages=None, tools=None, ids=None, temperature=None, top_p=None,
                    top_k=None, seed=None, do_sample=None, repetition_penalty=None,
                    min_p=None, constraint=None, max_length=None, min_new_tokens=None,
                    truncate=True, enable_thinking=None, presence_penalty=None,
                    frequency_penalty=None, stop=None, box=None, **chat_kw):
        """The decode loop itself: yields plain text per token."""
        self._check_live()
        eot = self.tok.eot
        self._reset_linear_state()
        if enable_thinking is None:
            enable_thinking = bool((getattr(self, "gen_defaults", {}) or {})
                                   .get("enable_thinking", False))
        if ids is None:
            ids = self.tok.encode_chat(prompt, system, messages=messages, tools=tools,
                                       enable_thinking=enable_thinking, **chat_kw)
        ids, max_new = self._plan_length(ids, max_new, max_length, truncate)
        P = len(ids)
        if box is not None:
            # Does the rendered prompt END inside a reasoning span? That is what decides the
            # starting channel, and only the template knows -- so ask the template's output.
            tail = self.tok.decode(ids[-16:]).rstrip() if ids else ""
            for o, c in _THINK_TAGS:
                if tail.endswith(o):
                    box["close"] = c
                    break
        self._set_sampling(temperature, top_p, top_k, seed, do_sample, repetition_penalty,
                           min_p, constraint, min_new_tokens, prompt_ids=ids,
                           presence_penalty=presence_penalty,
                           frequency_penalty=frequency_penalty, stop=stop)
        t0 = time.perf_counter()

        def _stats(n, truncated, steps, td):
            dec = time.perf_counter() - td
            self.last_stream = {"n": n, "truncated": bool(truncated),
                                "ttft_s": round(getattr(self, "_stream_ttft", dec), 3),
                                "tok_s": round(steps / max(dec, 1e-9), 2)}
            return self.last_stream

        # Same capture condition as `generate`: a decode step is capturable only when every
        # recurrent layer can run its step on the device; otherwise grow a cache instead.
        rec = [i for i in range(len(self.layers)) if self._is_linear_layer(i)]
        on_gpu = all(self.layers[i]["linear"]._gpu_step_ok() for i in rec)
        if not self._gpu or (rec and not on_gpu):
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            nxt = self._kv_forward(ids, 0, cache)
            self._stream_ttft = time.perf_counter() - t0
            pos = P; n = 0; steps = 0
            td = time.perf_counter()
            while n < max_new and nxt != eot:
                yield self.tok.decode([nxt]); n += 1; steps += 1
                if self._stop_now():
                    break                       # just-yielded text satisfies the constraint
                nxt = self._kv_forward([nxt], pos, cache); pos += 1
            _stats(n, n >= max_new and nxt != eot, steps, td)
            return
        for c in self.Kc: c.data[:] = 0.0
        for v in self.Vc: v.data[:] = 0.0
        g0 = self._prefill(ids); self._stream_ttft = time.perf_counter() - t0
        plat = wt._adam_kernel["platform"]
        self._set_inputs(g0, P)
        plat.beginCapture("decode"); logits_t = self._decode_fwd(); logits_t.numpy(); plat.endCapture()
        self.capture_ready = True; nxt = g0; pos = P; n = 0; steps = 0
        td = time.perf_counter()
        while n < max_new and nxt != eot:
            yield self.tok.decode([nxt]); n += 1; steps += 1
            if self._stop_now():
                break                           # just-yielded text satisfies the constraint
            self._set_inputs(nxt, pos); plat.replay("decode")
            nxt = self._pick(logits_t.numpy()[0]); pos += 1
        _stats(n, n >= max_new and nxt != eot, steps, td)

    def release(self):
        """Free this model's weights (layers, embeddings, head, KV cache). The object must not
        be used afterwards; load it again to use it. See `webtorch.release`.

        Several entry points can share one loaded model (`load()`, `pipeline()`,
        `from_pretrained` dedup on the weights); this releases ONE handle, and the weights
        drop only when the last handle does."""
        from . import _sdk
        _sdk._impl_release(self)
        return self

    def _abort_build(self):
        """A load that dies part-way — stopped, or a bad/unsupported file — drops the weights
        it had already uploaded, so a failed load never strands half a model in GPU memory.
        The object was never published to the impl cache, so this is a plain drop."""
        try:
            self.release()
        except Exception:
            pass

    def _check_live(self):
        if self.__dict__.get("_released"):
            raise RuntimeError("this model has been released; load it again to use it")

    def generate(self, prompt=None, max_new=None, system="You are a helpful assistant.",
                 messages=None, tools=None, ids=None, embeds=None,
                 temperature=None, top_p=None, top_k=None, seed=None, do_sample=None,
                 repetition_penalty=None, min_p=None, constraint=None,
                 max_length=None, min_new_tokens=None, truncate=True,
                 stream=False, enable_thinking=None, presence_penalty=None,
                 frequency_penalty=None, stop=None, channels=False, **chat_kw):
        """Chat prompt -> decode -> GenResult. WebGPU replays a captured decode step per
        token (~20x); WebGL uses a correct growing-cache forward.

        `stream=True` yields each token's text as it is produced instead of returning one
        GenResult at the end (same parameters either way; `stream(...)` is a shorthand for
        it). Render live, keep the reply visible while it is being written.

        `enable_thinking` defaults to False unless the model or `load(...)` set otherwise: a
        model whose chat template knows the option
        answers directly instead of writing out its reasoning first; pass True to get the
        reasoning too. A template that does not know the option ignores it.

        `messages` is a full conversation instead of the `prompt`/`system` shorthand.

        `stop="..."` or `stop=[...]` ends the reply at any of those strings, on top of the
        model's own end-of-turn token. `presence_penalty`/`frequency_penalty` are the
        additive OpenAI-style knobs; `repetition_penalty` is the multiplicative HF one, and
        they compose. Every sampling parameter here can also be set once at load time (as
        `load(..., temperature=...)`) or come from the model's own generation_config.json;
        what is passed to this call wins for this call.

        Any other keyword goes to the model's chat template unchanged -- `tools=[...]`,
        `reasoning_effort="low"`, whatever that model documents. This is why those work
        without the SDK knowing a thing about them: the template ships with the model and
        is the only place that knows what its options are called. A model whose template
        ignores a keyword simply ignores it.

        `ids`/`embeds` override the prompt encoding: pass prebuilt token ids and/or (T,H)
        input embeddings to decode from a sequence assembled elsewhere. That is the generic
        hook used for multimodality (image/audio embeddings spliced into the token
        embeddings) -- see `webtorch.MultimodalLM` -- so no model-specific decode path is
        needed."""
        self._check_live()
        if stream:
            if embeds is not None:
                raise ValueError("stream=True does not take embeds= — prefill embeddings "
                                 "need the batch path")
            return self.stream(prompt, max_new=max_new, system=system, messages=messages,
                               tools=tools, ids=ids, temperature=temperature, top_p=top_p,
                               top_k=top_k, seed=seed, do_sample=do_sample,
                               repetition_penalty=repetition_penalty, min_p=min_p,
                               constraint=constraint, max_length=max_length,
                               min_new_tokens=min_new_tokens, truncate=truncate,
                               enable_thinking=enable_thinking,
                               presence_penalty=presence_penalty,
                               frequency_penalty=frequency_penalty, stop=stop,
                               channels=channels, **chat_kw)
        eot = self.tok.eot
        if enable_thinking is None:
            enable_thinking = bool((getattr(self, "gen_defaults", {}) or {})
                                   .get("enable_thinking", False))
        if ids is None:
            ids = self.tok.encode_chat(prompt, system, messages=messages, tools=tools,
                                       enable_thinking=enable_thinking, **chat_kw)
        ids, max_new = self._plan_length(ids, max_new, max_length, truncate)
        P = len(ids)
        self._set_sampling(temperature, top_p, top_k, seed, do_sample, repetition_penalty,
                           min_p, constraint, min_new_tokens, prompt_ids=ids,
                           presence_penalty=presence_penalty,
                           frequency_penalty=frequency_penalty, stop=stop)
        self._reset_linear_state()                     # fresh recurrent state per generation

        # A decode step can only be captured if it is the same sequence of GPU commands every
        # token. That rules out a recurrent mixer whose state advances on the HOST -- but not
        # one whose step runs on the device, where the state is just a buffer being read and
        # written in place. So hybrids are captured when every recurrent layer can do that,
        # and fall back to the growing-cache forward when one cannot.
        rec = [i for i in range(len(self.layers)) if self._is_linear_layer(i)]
        on_gpu = all(self.layers[i]["linear"]._gpu_step_ok() for i in rec)
        if not self._gpu or (rec and not on_gpu):      # WebGL fallback, or a host-side mixer
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            t0 = time.perf_counter(); g0 = self._kv_forward(ids, 0, cache, embeds=embeds)
            ttft = time.perf_counter() - t0
            gen = [g0]; nxt = g0; pos = P; steps = 0; td = time.perf_counter()
            while len(gen) < max_new and not self._stop_now():
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
        # The capture ran the step for real, so its logits belong to this position. Replaying
        # it with the same inputs would be harmless for attention -- the KV write is
        # idempotent -- but would advance a recurrent state a second time, so consume the
        # captured result and start replaying from the next position.
        td = time.perf_counter()
        gen = [g0]; steps = 1
        nxt = self._pick(logits_t.numpy()[0]); pos = P + 1
        while nxt != eot and len(gen) < max_new:
            gen.append(nxt)
            if len(gen) >= max_new or self._stop_now():
                break
            self._set_inputs(nxt, pos)
            plat.replay("decode")
            nxt = self._pick(logits_t.numpy()[0]); pos += 1; steps += 1
        dec = time.perf_counter() - td
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), round(steps / max(dec, 1e-9), 2))
