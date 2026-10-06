"""Bidirectional text encoders — the whole shape of one read out of its own files.

A decoder answers by producing tokens one after another; an encoder answers by reading the
WHOLE sequence once and handing back a vector per position. That single difference is why
this is a separate engine rather than a flag on `llm.py`: there is no KV cache, no sampler and
no loop, and the fast paths that make decoding quick are built for a cache this never has.

Nothing here knows any model's name. Sizes, layer types, window width, rope frequencies,
activation and epsilon are read from `config.json`; the parts a config does not state -- fused
versus split QKV, gated versus plain MLP, which norms carry a bias, which layers have a
pre-attention norm at all -- are read from the WEIGHTS, because the tensors that exist and
their shapes say what the graph is more reliably than any field does. A model whose file says
`intermediate_size: 2624` and ships an `mlp.Wi` with 5248 rows has a gated MLP; that is not an
opinion about the model, it is arithmetic.

What a caller gets is `encode(ids) -> (T, hidden)`. What it does with those vectors -- pool
them, score some of them, classify them -- is not this module's business.
"""
import time

import numpy as np

from . import _core as wt
from ._core import Tensor, bmm, cat, gelu, layernorm, silu, softmax, transpose_last2


# Activations a config may name, and what each one actually computes. `gelu` is the exact
# erf form in PyTorch; `_core.gelu` is the tanh approximation. Measured on a 421M encoder
# across 28 layers, the difference moves a final probability by at most 8e-4 and changed no
# answer, so the approximation is used for both spellings rather than carrying an erf
# polynomial into every backend for the fourth decimal.
_ACT = {
    "gelu": gelu,
    "gelu_new": gelu,
    "gelu_pytorch_tanh": gelu,
    "silu": silu,
    "swish": silu,
    "relu": lambda x: wt.ReLU()(x),
}



# `_replayed_packed`: this batch cannot be laid end to end; the padded pass takes it.
_NOT_PACKED = object()

class EncoderConfig(object):
    """An HF `config.json`, read into the shapes this engine needs.

    Purely config-driven, exactly as `llm.py` is for decoders: no model-name branches. The
    fields below are the ones that change the arithmetic; anything else in the file is
    training or tooling metadata and is ignored on purpose.
    """

    def __init__(self, cfg):
        self.raw = dict(cfg or {})
        c = self.raw
        self.hidden = int(c["hidden_size"])
        self.layers = int(c["num_hidden_layers"])
        self.heads = int(c["num_attention_heads"])
        self.head_dim = int(c.get("head_dim") or (self.hidden // self.heads))
        self.ffn = int(c["intermediate_size"])
        self.vocab = int(c["vocab_size"])
        # Two spellings for the same number are in circulation; a file may carry either.
        self.eps = float(c.get("norm_eps", c.get("layer_norm_eps", 1e-5)))
        self.act = str(c.get("hidden_activation", c.get("hidden_act", "gelu"))).lower()
        self.max_positions = int(c.get("max_position_embeddings", 0) or 0)

        # Per-layer attention type. `layer_types` states it outright; `global_attn_every_n_layers`
        # is the older way of saying the same thing. Absent both, every layer sees everything.
        types = list(c.get("layer_types") or [])
        every = int(c.get("global_attn_every_n_layers", 0) or 0)
        if types:
            self.layer_types = types
        elif every > 1:
            self.layer_types = ["full_attention" if i % every == 0 else "sliding_attention"
                                for i in range(self.layers)]
        else:
            self.layer_types = ["full_attention"] * self.layers

        # Half-width of the local window: a position attends to `window` on each side. Files
        # state the FULL width (`local_attention`), and some state the half directly.
        local = int(c.get("local_attention", 0) or 0)
        self.window = int(c.get("sliding_window", 0) or 0) or (local // 2 if local else 0)

        # Rope frequency, per attention type when the file gives one per type.
        rp = c.get("rope_parameters") or c.get("rope_scaling") or {}
        base = float(c.get("rope_theta", 10000.0))
        self.theta = {}
        for t in set(self.layer_types):
            spec = rp.get(t) if isinstance(rp, dict) else None
            if isinstance(spec, dict):
                self.theta[t] = float(spec.get("rope_theta", base))
            else:
                self.theta[t] = float(rp.get("rope_theta", base)) if isinstance(rp, dict) else base

        # Token ids a caller needs to build a sequence. Read here so nobody re-reads the file.
        self.cls_id = c.get("cls_token_id")
        self.sep_id = c.get("sep_token_id")
        self.pad_id = c.get("pad_token_id")
        self.bos_id = c.get("bos_token_id")
        self.eos_id = c.get("eos_token_id")

    def __repr__(self):
        return ("<EncoderConfig hidden=%d layers=%d heads=%d ffn=%d vocab=%d window=%s>"
                % (self.hidden, self.layers, self.heads, self.ffn, self.vocab, self.window))


def _f32(a):
    """Weights arrive as whatever the file stored (fp16 is usual); the maths is fp32."""
    return np.ascontiguousarray(wt.materialize_weight(a, dtype=np.float32))


class TextEncoder(wt.Module):
    """A bidirectional encoder, built from a config and a bag of named weights.

    `weights` maps the names as they appear in the checkpoint to arrays. The structure is
    taken from which of those names exist:

      * `<p>attn.Wqkv`           one projection for q, k and v; otherwise `q_proj`/`k_proj`/`v_proj`
      * `<p>mlp.Wi` with 2x ffn  a gated MLP (half is transformed, half multiplies)
      * `<p>attn_norm` missing   that layer takes its input unnormalised (the embedding norm
                                 already did it, which is why it is usually layer 0)
      * any `.bias` absent       that projection or norm has no bias
    """

    def __init__(self, cfg, weights, prefix="encoder."):
        self.cfg = cfg
        self.p = prefix
        # What EXISTS is the structure, and it is needed after the arrays are gone.
        self.have = set(weights)
        self.shape_of = {k: wt.weight_shape(v) for k, v in weights.items()}
        self._src = dict(weights)          # emptied as tensors are built
        self._ten = {}
        self.act = _ACT.get(cfg.act, gelu)
        self._rope = {}
        self._emb_host = None
        # One recorded command list per sequence-length bucket, and a latch for a backend
        # that cannot record at all.
        self._cap = {}
        self._cap_seen = set()
        self._cap_off = False
        missing = [n for n in (prefix + "embeddings.tok_embeddings.weight",
                               prefix + "final_norm.weight") if n not in self.have]
        if missing:
            raise ValueError("this checkpoint has no %s -- it does not look like an encoder "
                             "this engine can build" % ", ".join(missing))

    def release(self):
        """Drop the weights, and everything recorded against them.

        `webtorch.release()` looks for this before falling back to a list of attribute names
        kept elsewhere, which is the only arrangement that works: what is heavy here is
        `_ten`, the embedding rows and the recorded graphs, and a list somewhere else cannot
        be expected to learn those names every time a model family is added. Dropping `_cap`
        matters as much as the weights -- a recorded pass pins every buffer it touched, and
        the backend only lets go of those when the release reaches it.
        """
        self.__dict__.update(_ten={}, _src={}, _emb_host=None, _rope={}, _cap={},
                             _cap_seen=set())
        self.__dict__["_released"] = True
        return self

    # ---- weight access -------------------------------------------------------------
    # Each weight becomes a Tensor exactly ONCE. Building one copies it to wherever the
    # backend keeps arrays, so building inside the forward would re-upload every weight on
    # every call -- and the host copy would sit beside the device copy for the model's whole
    # life. The source array is dropped as soon as its tensor exists, which keeps the peak at
    # one copy plus the one being converted.
    def _t(self, name, transposed=False):
        key = (name, transposed)
        got = self._ten.get(key)
        if got is None:
            a = self._src.get(name)
            if a is None:
                raise KeyError(name)
            half = np.asarray(a).dtype == np.float16
            a = _f32(a)
            if transposed:
                matrix = np.ascontiguousarray(a.T)
                # Half-width texture storage only for a weight the file stores at half
                # width; a float32 weight keeps float32 (see `half_weight`).
                got = wt.webgl_half_matrix(matrix) if half else None
                if got is None:
                    got = Tensor(matrix)
            else:
                got = Tensor(a)
            self._ten[key] = got
            if not (transposed and (name, False) in self._ten):
                self._src.pop(name, None)          # the file's copy is no longer needed
        return got

    def _has(self, name):
        return name in self.have

    def _packed(self, name):
        """The weight as packed half precision, built once, or None when it cannot be.

        Only this copy is kept: building it from the source array and dropping the array
        leaves ONE copy of the weight on the device at half the width, so the model takes
        half the memory rather than one and a half times it. A weight that cannot be packed
        (a width that is not a multiple of eight) falls back and is kept as f32.
        """
        key = (name, "f16")
        if key in self._ten:
            return self._ten[key]
        if self._ten.get((name, "nof16")):
            return None
        # Asked BEFORE the source array is dropped, and asked in full: packing a weight the
        # matmul will then refuse throws the only copy away and leaves the fallback with
        # nothing to read. So every condition that path checks is checked here -- the
        # backend, and the widths it needs -- rather than discovering one of them later.
        src = self._src.get(name)
        if src is None:
            return None
        t = wt.half_weight(src)              # None unless stored as float16 and holdable
        if t is None:
            self._ten[(name, "nof16")] = True
            return None
        self._ten[key] = t
        self._src.pop(name, None)
        return t

    def _lin(self, x, name):
        """`y = x @ W^T + b`, with the file's own layout: checkpoints store Linear weights as
        (out, in).

        The weight is read at half width where the backend can do it. What a matmul spends
        here is dominated by reading the weight -- 1.23 GB of them for one pass of a 421M
        encoder -- and halving that measured 1.33x across the four shapes in a layer, with
        the answer moving in the fourth decimal at most. Anything training keeps the f32
        path, which is also where a weight that will not pack ends up.
        """
        wn = name + ".weight"
        y = None
        key = (wn, "stored-linear")
        linear = self._ten.get(key)
        if linear is None:
            src = self._src.get(wn)
            native = wt.stored_linear(src)
            if native is not None:
                linear = native
                self._ten[key] = linear
                self._src.pop(wn, None)
        if linear is not None:
            y = linear(x)
        if not x.requires_grad:
            pk = None if y is not None else self._packed(wn)
            if pk is not None:
                n_out, n_in = self.shape_of[wn]
                y = wt.matmul_f16w(x, pk, n_in, n_out)
        if y is None:
            y = x.matmul(self._t(wn, transposed=True))
        return y + self._t(name + ".bias") if self._has(name + ".bias") else y

    def _norm(self, x, name):
        """LayerNorm whose affine part is whatever the file carries. A norm with a weight and
        no bias is not a different kind of norm, it is this one with beta fixed at zero."""
        g = self._t(name + ".weight")
        b = (self._t(name + ".bias") if self._has(name + ".bias") else self._zero())
        return layernorm(x, g, b, self.cfg.eps)

    def _add_norm(self, x, y, name):
        """`x + y` and that sum's norm `name`, fused where the backend can."""
        g = self._t(name + ".weight")
        b = (self._t(name + ".bias") if self._has(name + ".bias") else self._zero())
        return wt.add_layernorm(x, y, g, b, self.cfg.eps)

    def _zero(self):
        if "__zero__" not in self._ten:
            self._ten["__zero__"] = Tensor(np.zeros((self.cfg.hidden,), np.float32))
        return self._ten["__zero__"]

    # ---- pieces --------------------------------------------------------------------
    def _rope_tables(self, T, kind):
        key = (T, kind)
        if key not in self._rope:
            cos, sin = wt.rope_tables(T, self.cfg.head_dim, self.cfg.theta[kind])
            self._rope[key] = (Tensor(cos), Tensor(sin))
        return self._rope[key]

    def _mask(self, T, valid, kind, B=1):
        """Additive attention mask: 0 where a position may be read, a large negative where it
        may not. Two reasons a position may not be: it is padding, or this layer only looks a
        fixed distance to each side.

        The window is symmetric and inclusive -- `|i - j| <= window` -- which is what the
        reference implementation's mask actually does; it was read off a built mask rather
        than off the config field, because the field states the full width and the attention
        module and the mask builder disagree by one about what to do with it.
        """
        return Tensor(self._mask_array(T, valid, kind, B))

    def _mask_array(self, T, valid, kind, B=1):
        """Build the mask once on the host when filling a captured input buffer.

        A capture input must be uploaded, but creating a temporary device Tensor and
        reading it straight back first adds two transfers with no arithmetic benefit.
        The ordinary uncaptured path wraps the same values in a Tensor via ``_mask``.
        """
        base = np.zeros((T, T), dtype=np.float32)
        if kind == "sliding_attention" and self.cfg.window:
            idx = np.arange(T)
            base[np.abs(idx[:, None] - idx[None, :]) > self.cfg.window] = -1e9
        if B == 1:
            m = base
            if valid is not None:
                m = base.copy()
                m[:, np.asarray(valid[0] if np.ndim(valid) == 2 else valid) == 0] = -1e9
            return m
        # One mask per sequence, repeated for that sequence's heads, because what is padding
        # differs between them. The scores this is added to are (B*heads, T, T), so the rows
        # have to line up with that ordering: sequence outermost, head inside it.
        h = self.cfg.heads
        out = np.empty((B * h, T, T), dtype=np.float32)
        for b in range(B):
            mb = base.copy()
            if valid is not None:
                mb[:, np.asarray(valid[b]) == 0] = -1e9
            out[b * h:(b + 1) * h] = mb
        return out

    def _attn(self, x, layer, kind, mask, B=1, packed=None):
        h, hd = self.cfg.heads, self.cfg.head_dim
        p = "%slayers.%d.attn." % (self.p, layer)
        if packed is not None:
            rows = int(x.shape[0])
            cos, sin = self._rope_tables(rows, kind)
            o = wt.fused_attention_packed(
                self._lin(x, p + "Wqkv"), h, hd, packed["seg"], rows, 1.0 / (hd ** 0.5),
                self.cfg.window if kind == "sliding_attention" else 0, cos, sin,
                positions=packed["pos"])
            if o is None:
                raise RuntimeError("packed attention is unavailable for this encoder")
            return self._lin(o, p + "Wo" if self._has(p + "Wo.weight") else p + "o_proj")
        T = x.shape[0] // B
        cos, sin = self._rope_tables(T, kind)
        if self._has(p + "Wqkv.weight"):
            # One projection holding q, k and v back to back. The reference views it as
            # (T, 3, heads, head_dim) and unbinds axis 1, which in memory is exactly three
            # consecutive slices of the flat row.
            qkv = self._lin(x, p + "Wqkv")                     # (T, 3*h*hd)
            # The whole attention in two dispatches where the backend has it: q and k
            # rotated, then one kernel from the projection to the out-projection's rows.
            # None means this backend or shape keeps the expression below.
            o = (None if x.requires_grad else
                 wt.fused_attention(qkv, h, hd, T, 1.0 / (hd ** 0.5), mask,
                                    self.cfg.window if kind == "sliding_attention" else 0,
                                    B, cos, sin))
            if o is not None:
                return self._lin(o, p + "Wo" if self._has(p + "Wo.weight") else p + "o_proj")
            # Taken, transposed and rotated in one pass each. Written out, this is a slice
            # and a transpose before the rotation, and both are strided COPIES of the whole
            # tensor -- the backend has no view of a slice of a row -- so three of them per
            # layer cost more than every multiply in the pass put together.
            qq = None if x.requires_grad else wt.qkv_take(qkv, 0, h, hd, T, cos, sin, B)
            if qq is not None:
                q = qq
                k = wt.qkv_take(qkv, 1, h, hd, T, cos, sin, B)
                v = wt.qkv_take(qkv, 2, h, hd, T, B=B)
                return self._out(q, k, v, mask, p, T, h, hd, B, kind)
            d = h * hd
            q = wt._slice_last(qkv, 0, d).reshape(B * T, h, hd)
            k = wt._slice_last(qkv, d, 2 * d).reshape(B * T, h, hd)
            v = wt._slice_last(qkv, 2 * d, 3 * d).reshape(B * T, h, hd)
        else:
            q = self._lin(x, p + "q_proj").reshape(B * T, h, hd)
            k = self._lin(x, p + "k_proj").reshape(B * T, h, hd)
            v = self._lin(x, p + "v_proj").reshape(B * T, h, hd)
        # rope wants (..., T, hd): put heads first so T is the second-to-last axis, and with
        # a batch the sequence axis has to come out from under it first -- the layout the
        # attention below reads is (sequence, head) flattened, in that order.
        def heads_first(t):
            if B == 1:
                return t.permute(1, 0, 2)                      # (h, T, hd)
            return t.reshape(B, T, h, hd).permute(0, 2, 1, 3).reshape(B * h, T, hd)
        q, k = heads_first(q), heads_first(k)
        # One dispatch each where the backend has the fused form. Written out of primitives
        # this is eight dispatches per tensor per layer -- the largest block of them left in
        # the pass once the matmuls stopped being the whole of the time. The fused kernel
        # carries no gradient, so anything training keeps the expression.
        fq = None if (q.requires_grad or k.requires_grad) else wt.rope_decode(q, cos, sin, hd, hd, T)
        if fq is not None:
            q, k = fq, wt.rope_decode(k, cos, sin, hd, hd, T)
        else:
            q, k = wt.apply_rope(q, cos, sin), wt.apply_rope(k, cos, sin)
        v = heads_first(v)
        return self._out(q, k, v, mask, p, T, h, hd, B, kind)

    def _out(self, q, k, v, mask, p, T, h, hd, B=1, kind="full_attention"):
        """Attention over q, k, v laid out as (B*heads, T, head_dim)."""
        scale = 1.0 / (hd ** 0.5)
        if wt._adam_backend_ready() and not (q.requires_grad or k.requires_grad
                                            or mask.requires_grad):
            scores = wt.banded_qk_scores(
                q, k, mask, scale,
                self.cfg.window if kind == "sliding_attention" else 0)
            if scores is None:
                raise RuntimeError("WebGPU fused QK does not support this attention layout")
        else:
            # WebGL/CPU and autograd have an explicit equivalent implementation;
            # a selected WebGPU inference kernel is never silently replaced.
            scores = bmm(q, transpose_last2(k)) * scale + mask
        probabilities = softmax(scores)
        # `banded_pv` remains available to lower-level callers. A standalone
        # primitive win is insufficient to select it for this enclosing layer;
        # its full-encoder route still needs a stable positive measurement.
        o = bmm(probabilities, v)                              # (B*h, T, hd)
        if B == 1:
            o = o.permute(1, 0, 2).reshape(T, h * hd)
        else:
            o = o.reshape(B, h, T, hd).permute(0, 2, 1, 3).reshape(B * T, h * hd)
        return self._lin(o, p + "Wo" if self._has(p + "Wo.weight") else p + "o_proj")

    def packed_attention_probe(self, kind):
        """`probe(m)`: one `kind` layer's packed attention over m rows holding three
        sequences, for the load-time route ladder; None where a batch is not run packed."""
        if not self._packed_ok():
            return None
        h, hd = self.cfg.heads, self.cfg.head_dim
        window = self.cfg.window if kind == "sliding_attention" else 0
        rng = np.random.default_rng(0)

        def probe(m):
            rows = int(m)
            n = [rows // 3, rows // 3, rows - 2 * (rows // 3)]
            if min(n) < 1:
                return
            off = np.cumsum([0] + n[:-1])
            seg = wt.xp.asarray(np.stack([off, n], 1).reshape(-1).astype(np.int32))
            pos = wt.xp.asarray(np.concatenate([np.arange(k) for k in n]).astype(np.int32))
            qkv = Tensor(rng.standard_normal((rows, 3 * h * hd)).astype(np.float32))
            cos, sin = self._rope_tables(rows, kind)
            o = wt.fused_attention_packed(qkv, h, hd, seg, rows, 1.0 / (hd ** 0.5), window,
                                          cos, sin, positions=pos)
            if o is not None:
                wt._sync_small(o)
        return probe

    def attention_probe(self, kind):
        """`probe(m)`: one `kind` layer's attention at m tokens, for the load-time route
        ladder; None where the backend has no fused attention to race."""
        if not wt._adam_backend_ready():
            return None
        h, hd = self.cfg.heads, self.cfg.head_dim
        window = self.cfg.window if kind == "sliding_attention" else 0
        rng = np.random.default_rng(0)

        def probe(m):
            T = int(m)
            qkv = Tensor(rng.standard_normal((T, 3 * h * hd)).astype(np.float32))
            cos, sin = self._rope_tables(T, kind)
            o = wt.fused_attention(qkv, h, hd, T, 1.0 / (hd ** 0.5), self._mask(T, None, kind),
                                   window, 1, cos, sin)
            if o is not None:
                wt._sync_small(o)
        return probe

    def _mlp(self, x, layer):
        p = "%slayers.%d.mlp." % (self.p, layer)
        if self._has(p + "Wi.weight"):
            rows = self.shape_of[p + "Wi.weight"][0]
            y = self._lin(x, p + "Wi")
            if rows == 2 * self.cfg.ffn:
                # Gated: the first half is transformed, the second half multiplies it. The
                # order is not guessable from the shape -- it is the reference's `chunk(2)`
                # with the activation on the FIRST piece.
                #
                # Taken whole where the backend can. Sliced, each half is a strided COPY of
                # half the tensor before any arithmetic runs, and those two copies measured
                # as costly as the three QKV copies together. The fused form reads both
                # halves where they already are; it carries no gradient, so training keeps
                # the slices.
                g = (None if (y.requires_grad or self.act is not gelu)
                     else wt.geglu_split(y, self.cfg.ffn))
                if g is not None:
                    y = g
                else:
                    a = wt._slice_last(y, 0, self.cfg.ffn)
                    b = wt._slice_last(y, self.cfg.ffn, 2 * self.cfg.ffn)
                    y = self.act(a) * b
            else:
                y = self.act(y)
            return self._lin(y, p + "Wo")
        y = self.act(self._lin(x, p + "up_proj" if self._has(p + "up_proj.weight") else p + "dense_in"))
        return self._lin(y, p + "down_proj" if self._has(p + "down_proj.weight") else p + "dense_out")

    # ---- the pass ------------------------------------------------------------------
    # An embedding table big enough that putting it on the device is the wrong trade. The
    # number is the staging buffer, not the device: uploading one goes through a host-side
    # copy of the whole table, and a 256k x 768 vocabulary is 786 MB of f32, which failed to
    # allocate outright. Below the line a table is uploaded once and every lookup happens on
    # the device; above it, the table stays on the host and only the rows a sequence actually
    # uses are sent -- a few hundred kilobytes against hundreds of megabytes, for a lookup
    # that never needed the other quarter of a million rows.
    _EMB_DEVICE_MAX = 64 << 20          # in elements: 64M floats, 256 MB

    def _embed(self, ids):
        rows = self._host_embedding_rows(ids)
        if rows is not None:
            return Tensor(rows)
        name = self.p + "embeddings.tok_embeddings.weight"
        return wt.embedding(self._t(name), ids)

    def _host_embedding_rows(self, ids):
        """Return gathered host rows when the vocabulary is kept off-device."""
        name = self.p + "embeddings.tok_embeddings.weight"
        if self._emb_host is None:
            rows, dim = self.shape_of[name]
            if rows * dim <= self._EMB_DEVICE_MAX:
                return None
            src = self._src.pop(name, None)
            if src is None:
                return None
            src = wt.materialize_weight(src)
            self._emb_host = np.ascontiguousarray(src)       # left at the file's own width
        return _f32(self._emb_host[ids])

    def encode(self, ids, valid=None):
        """Token ids in, one vector per position out. `valid` is 1 for a real token and 0 for
        padding; padded columns are unreadable by every position."""
        return self._run(np.asarray(ids, dtype=np.int64), int(np.size(ids)), 1,
                         [valid] if valid is not None else None)

    def _encode_many_device(self, seqs):
        """Several padded sequences in one pass, leaving the result on the backend device.

        The returned tensor is still flattened as ``(B * L, hidden)``. Callers that can
        consume it there avoid reading every hidden row to the host merely to upload the
        same rows again for a downstream head.
        """
        if not seqs:
            raise ValueError("encode_many needs at least one sequence")
        L = max(len(s) for s in seqs)
        B = len(seqs)
        ids = np.zeros((B, L), dtype=np.int64)
        valid = np.zeros((B, L), dtype=np.int64)
        pad = self.cfg.pad_id or 0
        ids[:] = pad
        lengths = []
        for b, s in enumerate(seqs):
            lengths.append(len(s))
            ids[b, :len(s)] = np.asarray(s, dtype=np.int64)
            valid[b, :len(s)] = 1
        return self._run(ids.reshape(-1), L, B, valid), lengths, L

    def encode_many(self, seqs):
        """Several sequences in one pass, each trimmed back to its own length on the way out.

        One pass, not one per sequence, and a real batch axis rather than the sequences laid
        end to end: concatenated, every position would see every other one, and attention
        would cost the square of the total instead of the sum of the squares. Padded to the
        longest, each sequence reads only itself.

        Whether this is worth doing depends on how long the sequences are, and the caller
        decides that -- see `batch_pays`.
        """
        out, lengths, L = self._encode_many_device(seqs)
        B = len(lengths)
        flat = out.numpy().reshape(B, L, -1)
        return [Tensor(np.ascontiguousarray(flat[b, :length]))
                for b, length in enumerate(lengths)]

    # Sequence lengths are rounded up to a multiple of this before a pass is captured.
    #
    # A capture is only reusable for the shape it recorded, and decision sequences are never
    # the same length twice: measured over 24 real ones, 14 distinct lengths between 105 and
    # 185 tokens. Keyed by the exact length, four slots fill with lengths that occur once and
    # the other ten are never captured at all -- which is WORSE than not capturing, because
    # every one of them pays to record a pass that is then never replayed. Median of three
    # runs over those 24 calls, total and the warm second half:
    #
    #     bucket      total     warm      captures
    #     exact       1451 ms   692 ms    4   -- slots wasted on lengths seen once
    #     none        1131 ms   553 ms    0
    #     16          1345 ms   481 ms    4   -- too many buckets, slots run out
    #     32          1024 ms   440 ms    3
    #     48          1000 ms   453 ms    2
    #     64          1121 ms   471 ms    2   -- 20% of the positions are padding
    #
    # So the size is a trade between how often a bucket is hit and how much padding is then
    # carried through a quadratic attention. 32 rather than 48 because the padding share is
    # what grows with length, and these were short sequences.
    _BUCKET = 32

    # Rounding is this method's own, so `_replayed` trims the answer back to the length it was
    # asked about on the device. It has to: padding changes NOTHING on the real positions -- measured on
    # this checkpoint at 105 to 225 tokens, padding to a multiple of 64 and trimming back
    # leaves the hidden states bit-identical, max absolute difference 0.0 at every length
    # against values of magnitude 42 -- but handing the padded ROWS on is a different thing
    # entirely. A decision head's attention has no mask, so it reads them, which moved its
    # logits by up to 0.012. That is what once got the rounding here removed, as if padding
    # were numerically unsound; it is the missing trim. `encode_many` never had the bug, for
    # the same reason: it cuts every sequence back to its own length on the way out.

    # Each retained graph pins its intermediates, so the slot count is bounded. Retire the
    # least recently used graph when a repeatedly seen new shape needs a slot; otherwise a
    # session whose question lengths drift permanently falls back to uncaptured execution.
    _CAP_MAX = 4

    @staticmethod
    def _capture_platform():
        if wt._adam_backend_ready():
            return wt._adam_kernel["platform"]
        return None

    def _replayed(self, ids, T, B, valid):
        """One encoder pass through a recorded command list, or None if that is not on.

        Measured on this backend, one 82-token pass of a 22-layer 768-wide encoder: 36.2 ms
        of it is the HOST issuing 550 dispatches and 0.7 ms is waiting for the GPU, which is
        idle almost the whole time. The arithmetic is not the cost; reaching the device 550
        times from Python inside wasm is. A captured pass issues one command instead.

        Only where the shape can be held still, with no autograd and a backend that records.
        Independent question rows use a separate (batch, length) capture: the captured
        masks preserve each row's own attention boundary and padding.
        """
        self._last_capture_timing = None
        if not self._capture_ok():
            return None
        # One question too: its inputs staged by JS in one call and its embedding looked up
        # on the device, where the padded path read the rows back to upload them again and
        # built its masks in Python -- 1.5 ms of host work a request, 6-7 ms once the CPU
        # has dropped its clock between requests.
        if self._packed_ok():
            got = self._replayed_packed(ids, T, B, valid)
            if got is not _NOT_PACKED:
                return got                  # the result, or None: seen once, run it eagerly
        Tb = int(((T + self._BUCKET - 1) // self._BUCKET) * self._BUCKET)
        if self.cfg.max_positions and Tb > self.cfg.max_positions:
            return None
        # A large batch's B*heads*T*T mask would be pinned by recording. Bound that
        # allocation; larger shapes keep the ordinary (uncaptured) implementation.
        if B > 1 and B * self.cfg.heads * Tb * Tb * 4 * len(set(self.cfg.layer_types)) > (32 << 20):
            return None
        key = Tb if B == 1 else (B, Tb)
        slot = self._cap.get(key)
        if slot is None:
            if key not in self._cap_seen:
                self._cap_seen.add(key)
                return None
            if len(self._cap) >= self._CAP_MAX:
                old_key = next(iter(self._cap))
                old = self._cap[old_key]
                plat = self._capture_platform()
                if not hasattr(plat, "releaseCapture"):
                    raise RuntimeError("WebGPU capture eviction is unavailable")
                plat.releaseCapture(old["name"])
                del self._cap[old_key]
            slot = self._cap_make(Tb) if B == 1 else self._cap_make(Tb, B=B)
            if slot is None:
                return None
        else:
            # A repeated shape is hot; keep its recording when the next new shape arrives.
            self._cap.pop(key)
            self._cap[key] = slot
        stage_start = time.perf_counter()
        if B == 1:
            self._cap_write(slot, ids, T, valid)
        else:
            self._cap_write(slot, ids, T, valid, B=B)
        write_ms = (time.perf_counter() - stage_start) * 1000
        plat = self._capture_platform()
        stage_start = time.perf_counter()
        if slot["recorded"]:
            plat.replay(slot["name"])
        else:
            plat.beginCapture(slot["name"])
            out = self._layers(slot["x"], slot["masks"], B)
            # Kernel dispatches are enqueued eagerly in FIFO order. Reading all hidden rows
            # here only to upload them again for the decision head costs more than the
            # replay itself; the head's final small readback synchronises the whole graph.
            plat.endCapture()
            slot["out"] = out
            slot["recorded"] = True
        submit_ms = (time.perf_counter() - stage_start) * 1000
        # `encode` answers for the ids it was given: undo the rounding without bringing
        # every hidden row to the host. See the note on _BUCKET for why trimming is required.
        stage_start = time.perf_counter()
        if Tb == T:
            self._last_capture_timing = {"write_ms": round(write_ms, 3),
                                         "submit_ms": round(submit_ms, 3),
                                         "trim_ms": 0.0}
            return slot["out"]
        rows = (slot["out"][:T] if B == 1 else
                slot["out"].reshape(B, Tb, self.cfg.hidden)[:, :T]
                .reshape(B * T, self.cfg.hidden))
        self._last_capture_timing = {"write_ms": round(write_ms, 3),
                                     "submit_ms": round(submit_ms, 3),
                                     "trim_ms": round((time.perf_counter() - stage_start) * 1000, 3)}
        return rows

    def _capture_ok(self):
        if self._cap_off:
            return False
        plat = self._capture_platform()
        self._cap_off = not (hasattr(plat, "beginCapture") and hasattr(plat, "replay"))
        if wt._adam_backend_ready() and self._cap_off:
            raise RuntimeError("WebGPU encoder capture is unavailable after backend initialization")
        return not self._cap_off

    def _cap_make(self, Tb, B=1):
        """Buffers this length's recorded pass reads from. Written before every replay, never
        reallocated -- a capture binds the buffer it saw, so a fresh one would be invisible
        to it and the pass would answer with whatever the first call happened to contain."""
        compact_masks = (B > 1 and wt._adam_backend_ready()
                         and self._emb_host is not None
                         and self._emb_host.dtype in (np.float16, np.float32))
        mask_heads = 1 if compact_masks else self.cfg.heads
        x = Tensor(np.zeros((B * Tb, self.cfg.hidden), np.float32))
        masks = {k: Tensor(np.zeros((B * mask_heads, Tb, Tb), np.float32)
                           if B > 1 else np.zeros((Tb, Tb), np.float32))
                 for k in set(self.cfg.layer_types)}
        slot = {"x": x, "masks": masks, "out": None, "recorded": False,
                "name": "enc%d_%d_%d" % (id(self) & 0xffff, B, Tb), "T": Tb,
                "B": B, "mask_heads": mask_heads}
        self._cap[Tb if B == 1 else (B, Tb)] = slot
        return slot

    def _cap_write(self, slot, ids, T, valid, B=1):
        Tb = slot["T"]
        # The hot batched replay stages every mutable input in the worker's JS shared
        # arena.  Python supplies references and shape metadata only; JS gathers the
        # embedding rows, builds every per-question mask, and uploads in one call.
        if B > 1 and wt._adam_backend_ready() and self._emb_host is not None:
            dtype = self._emb_host.dtype
            if dtype == np.float16 or dtype == np.float32:
                import js
                mask_ids = {kind: int(mask.data.buffer.buffer_id)
                            for kind, mask in slot["masks"].items()}
                js.gpu.stageDecisionCapture(
                    int(slot["x"].data.buffer.buffer_id), mask_ids,
                    np.asarray(ids, dtype=np.int64).view(np.uint8),
                    np.asarray(valid, dtype=np.int64).view(np.uint8),
                    # Pyodide cannot expose a NumPy float16 buffer as Float16Array in
                    # browsers without native Float16 JS support. A uint8 *view* is the
                    # same memory, with no conversion/copy; JS decodes f16 from its bytes.
                    self._emb_host.view(np.uint8), "f16" if dtype == np.float16 else "f32",
                    B, T, Tb, self.cfg.hidden, self.cfg.vocab, self.cfg.pad_id or 0,
                    slot.get("mask_heads", self.cfg.heads), self.cfg.window or 0)
                return
        pad = self.cfg.pad_id or 0
        full = np.full((B, Tb), pad, dtype=np.int64)
        full[:, :T] = np.asarray(ids, dtype=np.int64).reshape(B, T)
        live = np.zeros((B, Tb), dtype=np.int64)
        live[:, :T] = (1 if valid is None else
                       np.asarray(valid, dtype=np.int64).reshape(B, T))
        emb = self._embed_rows(full.reshape(-1))
        slot["x"].data.buffer.set_data(np.ascontiguousarray(emb, dtype=np.float32).reshape(-1))
        for kind, m in slot["masks"].items():
            built = self._mask_array(Tb, live, kind, B)
            m.data.buffer.set_data(
                np.ascontiguousarray(built, dtype=np.float32).reshape(-1))

    # ---- several sequences end to end ---------------------------------------------------
    #
    # Padded, a batch of questions runs every one of them at the longest one's length and
    # then rounds that up: 160, 153 and 173 tokens became 3 x 192 = 576 rows for 486 real
    # ones, and every projection, norm and MLP ran over the 90 that are padding. Laid end to
    # end, the rows are the real tokens rounded once, on the TOTAL (512 here), and attention
    # reads each sequence alone at its own length (`wt.fused_attention_packed`) -- the sum of
    # the squares of the real lengths, not of the padded one. Every row's arithmetic is the
    # padded layout's, in the same order: the answers are bit-identical.

    def _packed_ok(self):
        """A batch can run end to end here: WebGPU, the fused q/k/v projection the packed
        attention reads in every layer, and a head dimension that kernel takes."""
        if getattr(self, "_packed_off", False) or not wt._adam_backend_ready():
            return False
        if self.cfg.head_dim % 32:
            return False
        return all(self._has("%slayers.%d.attn.Wqkv.weight" % (self.p, i))
                   for i in range(self.cfg.layers))

    def _replayed_packed(self, ids, T, B, valid):
        """`_replayed` for a batch laid end to end. `_NOT_PACKED` where this batch cannot be
        (a sequence whose real tokens are not a prefix); None the first time a shape is seen,
        so the caller runs it once eagerly, as `_replayed` does."""
        v = (np.ones((B, T), np.int64) if valid is None
             else np.asarray(valid, dtype=np.int64).reshape(B, T))
        lengths = v.sum(1)
        if np.any(lengths < 1) or not np.array_equal(
                v != 0, np.arange(T)[None, :] < lengths[:, None]):
            return _NOT_PACKED
        if self.cfg.max_positions and T > self.cfg.max_positions:
            return _NOT_PACKED
        total = int(lengths.sum())
        rows = int(((total + self._BUCKET - 1) // self._BUCKET) * self._BUCKET)
        key = ("packed", B, rows)
        slot = self._cap.get(key)
        if slot is None and key not in self._cap_seen:
            # Seen once: run it eagerly in this layout, without taking one of the recorded
            # slots. Eagerly matters -- a route race met for the first time inside a
            # recording would be recorded with it, every candidate replayed on every call.
            self._cap_seen.add(key)
            slot = self._cap_make_packed(B, rows, register=False)
            self._cap_write_packed(slot, ids, T, B, lengths)
            out = self._layers(self._packed_input(slot), None, B, packed=slot)
            return wt.gather_rows_at(out, slot["gather"], B * T)
        if slot is None:
            if len(self._cap) >= self._CAP_MAX:
                old_key = next(iter(self._cap))
                self._capture_platform().releaseCapture(self._cap[old_key]["name"])
                del self._cap[old_key]
            slot = self._cap_make_packed(B, rows)
        else:
            self._cap.pop(key)
            self._cap[key] = slot
        stage_start = time.perf_counter()
        self._cap_write_packed(slot, ids, T, B, lengths)
        write_ms = (time.perf_counter() - stage_start) * 1000
        plat = self._capture_platform()
        stage_start = time.perf_counter()
        if slot["recorded"]:
            plat.replay(slot["name"])
        else:
            plat.beginCapture(slot["name"])
            slot["out"] = self._layers(self._packed_input(slot), None, B, packed=slot)
            plat.endCapture()
            slot["recorded"] = True
        submit_ms = (time.perf_counter() - stage_start) * 1000
        # Back to the (B, T) rows the head reads: one gather, as the padded pass's trim was.
        stage_start = time.perf_counter()
        out = wt.gather_rows_at(slot["out"], slot["gather"], B * T)
        self._last_capture_timing = {"write_ms": round(write_ms, 3),
                                     "submit_ms": round(submit_ms, 3),
                                     "trim_ms": round((time.perf_counter() - stage_start) * 1000, 3),
                                     "packed_rows": rows}
        return out

    def _packed_input(self, slot):
        """The pass's input rows: looked up on the device from the staged tokens, or the
        staged rows themselves when the vocabulary is kept on the host."""
        if slot["tok"] is None:
            return slot["x"]
        return wt.gather_rows_at(self._t(self.p + "embeddings.tok_embeddings.weight"),
                                 slot["tok"], slot["rows"])

    def _cap_make_packed(self, B, rows, register=True):
        """A packed shape's buffers. The vocabulary on the device: each row's token is staged
        and the lookup is the pass's first dispatch. On the host: the rows themselves."""
        self._host_embedding_rows(np.zeros((1,), np.int64))   # settles where the table lives
        on_device = self._emb_host is None
        if not on_device and self._emb_host.dtype not in (np.float16, np.float32):
            on_device = True
        slot = {"tok": wt.xp.empty((rows,), np.float32) if on_device else None,
                "x": None if on_device else Tensor(wt._empty((rows, self.cfg.hidden))),
                "seg": wt.xp.empty((2 * B,), np.int32),
                "pos": wt.xp.empty((rows,), np.int32),
                "gather": wt.xp.empty((B * rows,), np.float32),
                "out": None, "recorded": False, "rows": rows, "B": B,
                "name": "encp%d_%d_%d" % (id(self) & 0xffff, B, rows)}
        if register:
            self._cap[("packed", B, rows)] = slot
        return slot

    def _cap_write_packed(self, slot, ids, T, B, lengths):
        """Everything the recorded pass reads, staged by JS in one call: token rows (or
        embedding rows), segments, positions and the gather back to the head's layout."""
        import js
        table = None if slot["tok"] is not None else self._emb_host
        js.gpu.stageDecisionPacked(
            -1 if slot["x"] is None else int(slot["x"].data.buffer.buffer_id),
            -1 if slot["tok"] is None else int(slot["tok"].buffer.buffer_id),
            int(slot["seg"].buffer.buffer_id), int(slot["pos"].buffer.buffer_id),
            int(slot["gather"].buffer.buffer_id),
            np.asarray(ids, dtype=np.int64).view(np.uint8),
            np.asarray(lengths, dtype=np.int64).view(np.uint8),
            None if table is None else table.view(np.uint8),
            "f16" if table is not None and table.dtype == np.float16 else "f32",
            B, T, slot["rows"], self.cfg.hidden, self.cfg.vocab, self.cfg.pad_id or 0,
            B * slot["rows"])

    def _embed_rows(self, ids):
        """The embedding rows for `ids`, as a host array."""
        gathered = self._host_embedding_rows(ids)
        if gathered is not None:
            return gathered
        t = self._embed(ids)
        return np.asarray(t.numpy()) if hasattr(t, "numpy") else np.asarray(t.data)

    def _layers(self, x, masks, B, packed=None):
        """The stack, from embeddings to the final norm. Separate from `_run` because this
        part is the same dispatches every time for a given length -- which is what makes it
        capturable. `packed`: the sequences lie end to end (`_replayed_packed`); attention
        reads the slot's segments and positions instead of masks."""
        x = self._norm(x, self.p + "embeddings.norm")
        L = self.cfg.layers
        if not L:
            return self._norm(x, self.p + "final_norm")

        def attn_norm(i):
            return "%slayers.%d.attn_norm" % (self.p, i)
        # Every residual add is followed by a norm -- the next block's, or the final one -- so
        # each is one `_add_norm`: the sum is the stream, the norm the next block's input.
        xa = self._norm(x, attn_norm(0)) if self._has(attn_norm(0) + ".weight") else x
        for i in range(L):
            kind = self.cfg.layer_types[i]
            x, h = self._add_norm(x, self._attn(xa, i, kind, None if masks is None
                                                else masks[kind], B, packed=packed),
                                  "%slayers.%d.mlp_norm" % (self.p, i))
            m = self._mlp(h, i)
            nxt = attn_norm(i + 1) if i + 1 < L else self.p + "final_norm"
            if i + 1 < L and not self._has(nxt + ".weight"):
                x = x + m
                xa = x
            else:
                x, xa = self._add_norm(x, m, nxt)
        return xa

    def _run(self, ids, T, B, valid):
        got = self._replayed(ids, T, B, valid)
        if got is not None:
            return got
        masks = {k: self._mask(T, valid, k, B) for k in set(self.cfg.layer_types)}
        return self._layers(self._embed(ids), masks, B)

    forward = encode
