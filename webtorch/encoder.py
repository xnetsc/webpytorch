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
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


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
        self.shape_of = {k: tuple(np.shape(v)) for k, v in weights.items()}
        self._src = dict(weights)          # emptied as tensors are built
        self._ten = {}
        self.act = _ACT.get(cfg.act, gelu)
        self._rope = {}
        self._emb_host = None
        missing = [n for n in (prefix + "embeddings.tok_embeddings.weight",
                               prefix + "final_norm.weight") if n not in self.have]
        if missing:
            raise ValueError("this checkpoint has no %s -- it does not look like an encoder "
                             "this engine can build" % ", ".join(missing))

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
            a = _f32(a)
            got = Tensor(np.ascontiguousarray(a.T) if transposed else a)
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
        if src is None or not wt._adam_backend_ready():
            return None
        n_out, n_in = self.shape_of[name]
        if n_out % 64 or n_in % 4 or n_out % 8:
            self._ten[(name, "nof16")] = True
            return None
        packed = wt.pack_f16_weight(np.ascontiguousarray(_f32(src).T))
        if packed is None:
            self._ten[(name, "nof16")] = True
            return None
        t = Tensor(packed)
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
        if not x.requires_grad:
            pk = self._packed(wn)
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
        base = np.zeros((T, T), dtype=np.float32)
        if kind == "sliding_attention" and self.cfg.window:
            idx = np.arange(T)
            base[np.abs(idx[:, None] - idx[None, :]) > self.cfg.window] = -1e9
        if B == 1:
            m = base
            if valid is not None:
                m = base.copy()
                m[:, np.asarray(valid[0] if np.ndim(valid) == 2 else valid) == 0] = -1e9
            return Tensor(m)
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
        return Tensor(out)

    def _attn(self, x, layer, kind, mask, B=1):
        h, hd = self.cfg.heads, self.cfg.head_dim
        T = x.shape[0] // B
        p = "%slayers.%d.attn." % (self.p, layer)
        cos, sin = self._rope_tables(T, kind)
        if self._has(p + "Wqkv.weight"):
            # One projection holding q, k and v back to back. The reference views it as
            # (T, 3, heads, head_dim) and unbinds axis 1, which in memory is exactly three
            # consecutive slices of the flat row.
            qkv = self._lin(x, p + "Wqkv")                     # (T, 3*h*hd)
            # Taken, transposed and rotated in one pass each. Written out, this is a slice
            # and a transpose before the rotation, and both are strided COPIES of the whole
            # tensor -- the backend has no view of a slice of a row -- so three of them per
            # layer cost more than every multiply in the pass put together.
            qq = None if x.requires_grad else wt.qkv_take(qkv, 0, h, hd, T, cos, sin, B)
            if qq is not None:
                q = qq
                k = wt.qkv_take(qkv, 1, h, hd, T, cos, sin, B)
                v = wt.qkv_take(qkv, 2, h, hd, T, B=B)
                return self._out(q, k, v, mask, p, T, h, hd, B)
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
        return self._out(q, k, v, mask, p, T, h, hd, B)

    def _out(self, q, k, v, mask, p, T, h, hd, B=1):
        """Attention over q, k, v laid out as (B*heads, T, head_dim)."""
        scores = bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))
        o = bmm(softmax(scores + mask), v)                     # (B*h, T, hd)
        if B == 1:
            o = o.permute(1, 0, 2).reshape(T, h * hd)
        else:
            o = o.reshape(B, h, T, hd).permute(0, 2, 1, 3).reshape(B * T, h * hd)
        return self._lin(o, p + "Wo" if self._has(p + "Wo.weight") else p + "o_proj")

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
        name = self.p + "embeddings.tok_embeddings.weight"
        if self._emb_host is None:
            rows, dim = self.shape_of[name]
            if rows * dim <= self._EMB_DEVICE_MAX:
                return wt.embedding(self._t(name), ids)
            src = self._src.pop(name, None)
            if src is None:
                return wt.embedding(self._t(name), ids)
            self._emb_host = np.ascontiguousarray(src)       # left at the file's own width
        return Tensor(_f32(self._emb_host[ids]))

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

    def _run(self, ids, T, B, valid):
        x = self._embed(ids)
        x = self._norm(x, self.p + "embeddings.norm")
        masks = {k: self._mask(T, valid, k, B) for k in set(self.cfg.layer_types)}
        for i in range(self.cfg.layers):
            kind = self.cfg.layer_types[i]
            an = "%slayers.%d.attn_norm" % (self.p, i)
            xa = self._norm(x, an) if self._has(an + ".weight") else x
            x = x + self._attn(xa, i, kind, masks[kind], B)
            x = x + self._mlp(self._norm(x, "%slayers.%d.mlp_norm" % (self.p, i)), i)
        return self._norm(x, self.p + "final_norm")

    forward = encode
