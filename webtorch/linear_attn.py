"""Generic linear-attention / state-space (SSM) layers for hybrid decoders.

Recent decoder families (Qwen3.5/Qwen3.8, MiniMax, Mamba-hybrids, …) replace most softmax
attention layers with a **linear attention** layer that carries a fixed-size recurrent state
instead of a growing KV cache. A "hybrid" model interleaves the two — e.g. Qwen3.8-27B has 64
layers: 48 `linear_attention` + 16 `full_attention` (every 4th).

This module implements the **Gated DeltaNet** style linear attention used by that family, in a
config-driven way (head counts / dims / conv width all come from config, no model names):

    state_t = state_{t-1} * decay_t  +  k_t^T v_t          (per value head)
    out_t   = q_t @ state_t                                 (then gated + normed)

plus the short depthwise **causal conv** over q/k/v that these models apply before the
recurrence. Because the state is fixed-size, decoding is O(1) per token in memory — the whole
point of the design.

The layer exposes the same shape contract as the softmax path: `(T, H) -> (T, H)`, with an
explicit `state` object so prefill and step-by-step decode share one implementation.
"""
import math
import numpy as np

from . import _core as wt


class LinearAttentionState:
    """Recurrent state of one linear-attention layer: the (heads, k_dim, v_dim) matrix plus the
    causal-conv ring buffer. Fixed size — it does not grow with sequence length."""

    def __init__(self, n_v_heads, k_dim, v_dim, conv_width, conv_channels):
        self.S = np.zeros((n_v_heads, k_dim, v_dim), np.float32)
        self.conv = np.zeros((conv_width - 1, conv_channels), np.float32) if conv_width > 1 else None
        # A GPU copy exists only while the GPU path is driving. Whichever side was written
        # last is the authoritative one, and asking for the other syncs it -- so prefill on
        # the host and decode on the device can be mixed without either going stale.
        self._gpu = None

    def reset(self):
        self.S[:] = 0.0
        if self.conv is not None:
            self.conv[:] = 0.0
        self._gpu = None

    def gpu(self):
        """State on the device: (S, conv) as GPU arrays, uploading if the host wrote last."""
        if self._gpu is None:
            # Both go straight to kernels, so both stay raw device arrays.
            self._gpu = [wt.xp.asarray(self.S.reshape(-1)),
                         None if self.conv is None else wt.xp.asarray(self.conv.reshape(-1))]
        return self._gpu

    def host(self):
        """State on the host, downloading if the device wrote last."""
        if self._gpu is not None:
            g = self._gpu
            self.S = np.asarray(wt.cp.asnumpy(g[0]), np.float32).reshape(self.S.shape)
            if self.conv is not None and g[1] is not None:
                self.conv = np.asarray(wt.cp.asnumpy(g[1]), np.float32).reshape(self.conv.shape)
            self._gpu = None
        return self


def causal_conv1d(x, weight, bias=None, state=None):
    """Depthwise causal conv1d over a (T, C) sequence. `weight` is (C, W) (per-channel kernel).
    `state` (W-1, C) carries the tail of the previous chunk so prefill and incremental decode
    give identical results; it is updated in place."""
    T, C = x.shape
    W = weight.shape[1]
    if W <= 1:
        y = x * weight[:, 0][None, :]
        return (y + bias[None, :]) if bias is not None else y
    prev = state if state is not None else np.zeros((W - 1, C), np.float32)
    ext = np.concatenate([prev, x], 0)                       # (W-1+T, C)
    out = np.zeros((T, C), np.float32)
    for w in range(W):                                       # depthwise: no channel mixing
        out += ext[w:w + T] * weight[:, w][None, :]
    if bias is not None:
        out += bias[None, :]
    if state is not None:                                    # carry the tail for the next call
        tail = ext[-(W - 1):] if T >= 1 else prev
        state[:] = tail
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float32), -60, 60)))


def _softplus(x):
    x = np.asarray(x, np.float32)
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -60, 20))))


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def gated_delta_step(S, q, k, v, decay, beta):
    """One Gated-DeltaNet recurrence step (per value head), vectorised over heads.

    S     : (Hv, Dk, Dv) recurrent state, updated in place
    q,k   : (Hv, Dk)      query / key for this token (key L2-normalised by the caller)
    v     : (Hv, Dv)      value
    decay : (Hv,)         per-head forget gate in (0,1]
    beta  : (Hv,)         per-head write strength
    -> (Hv, Dv) output
    """
    S *= decay[:, None, None]                                # gated forgetting
    # delta rule: write (v - what the state already predicts for k)
    pred = np.einsum("hd,hdv->hv", k, S)
    delta = (v - pred) * beta[:, None]
    S += k[:, :, None] * delta[:, None, :]
    return np.einsum("hd,hdv->hv", q, S)


class LinearAttention:
    """Config-driven Gated-DeltaNet linear attention.

    cfg keys (all from the model config, no model-specific branches):
      n_k_heads, n_v_heads, k_head_dim, v_head_dim, conv_kernel_dim, hidden
    weights dict (names are supplied by the loader, so any naming scheme works):
      q, k, v      : projections   (callables: (T,H) -> (T, n*dim))
      a, dt        : gate projections producing decay / write strength
      g            : output gate projection (optional)
      o            : output projection
      conv_w/conv_b: depthwise conv weights (optional)
      norm         : output RMSNorm weight (optional)
    """

    def __init__(self, cfg, w, eps=1e-6):
        self.hk = int(cfg.get("n_k_heads") or 0); self.hv = int(cfg.get("n_v_heads") or 0)
        self.dk = int(cfg.get("k_head_dim") or 0); self.dv = int(cfg.get("v_head_dim") or 0)
        self.conv_width = int(cfg.get("conv_kernel_dim", 0) or 0)
        self.H = int(cfg.get("hidden") or 0); self.w = w; self.eps = eps
        self._infer_dims()                                   # fill anything the config omitted
        self.rep = max(1, self.hv // max(1, self.hk))        # key/value head grouping

    def _infer_dims(self):
        """Derive any head/dim the config did not state from the weights themselves, so a file
        that omits the ssm.* metadata still loads. Nothing here is model-specific: the shapes
        come from the tensors the model ships."""
        A = self.w.get("A"); nrm = self.w.get("norm"); cw = self.w.get("conv_w")
        if not self.hv and A is not None:
            self.hv = int(np.asarray(A).size)                # one decay scale per value head
        if not self.dv and nrm is not None:
            n = int(np.asarray(nrm).size)
            self.dv = n if (not self.hv or n != self.hv) else n
        if cw is not None:
            conv_dim = int(np.asarray(cw).shape[0])          # 2*hk*dk + hv*dv channels
            if not self.dk: self.dk = self.dv or 0
            if self.hv and self.dv and self.dk and not self.hk:
                key_total = max(0, conv_dim - self.hv * self.dv)
                self.hk = max(1, key_total // (2 * self.dk))
        if not self.dk: self.dk = self.dv
        if not self.dv: self.dv = self.dk
        if not self.hk: self.hk = self.hv or 1
        if not self.hv: self.hv = self.hk or 1
        if not self.dk or not self.dv:
            raise ValueError("cannot determine linear-attention head dims from config or weights")

    def new_state(self):
        ch = self.hk * self.dk * 2 + self.hv * self.dv
        return LinearAttentionState(self.hv, self.dk, self.dv, max(self.conv_width, 1), ch)

    def _proj(self, name, x):
        """Apply a projection to an (T,H) ndarray -> ndarray. Accepts either a webtorch layer
        (QuantizedLinear / UnquantizedLinear, which take a Tensor) or a plain callable."""
        f = self.w.get(name)
        if f is None:
            return None
        try:
            y = f(wt.Tensor(np.asarray(x, np.float32)))
        except Exception:
            y = f(np.asarray(x, np.float32))
        return np.asarray(y.numpy() if hasattr(y, "numpy") else y, np.float32)

    # ---- GPU decode path ------------------------------------------------------------
    # Same recurrence, but nothing leaves the device. The host path costs five GPU round
    # trips per layer for the projections alone -- measured at 4.6 ms for one that takes
    # 1.6 ms to compute -- and a 27B has 48 of these layers per token. Only the batch-of-one
    # step is done this way: prefill batches its projections already, and its recurrence is
    # sequential either way.

    def _t(self, name, xt):
        f = self.w.get(name)
        return None if f is None else f(xt)

    @staticmethod
    def _softplus_t(t):
        # max(x,0) + log1p(exp(-|x|)) -- the stable form, in terms of ops the Tensor has
        return t.relu() + ((-(t.abs())).exp() + 1.0).log()

    @staticmethod
    def _l2norm_t(t, heads, dim, eps):
        r = t.reshape(heads, dim)
        return r / ((r * r).sum(axis=-1, keepdims=True) + eps).sqrt()

    def _project_gpu(self, xt):
        """Every projection this layer needs, for however many rows `xt` has.

        Separated from the recurrence because it does not participate in it. These are plain
        matrix products over the row axis: batching them costs one pass over each weight
        matrix for the whole prompt, where doing them a row at a time costs one pass PER
        TOKEN. On a 27B with 48 recurrent layers that was the difference between prefill
        being cheaper per token than decode and being several times dearer."""
        nq = self.hk * self.dk
        nv = self.hv * self.dv
        if self.w.get("qkv") is not None:
            qkv = self._t("qkv", xt)
        else:
            qkv = wt.cat([self._t("q", xt)[:, :nq],
                          self._t("k", xt)[:, :nq],
                          self._t("v", xt)[:, :nv]], axis=1)
        return qkv, self._t("beta", xt), self._t("alpha", xt), self._t("g", xt)

    def _recur_gpu(self, qkv, braw, araw, state):
        """The genuinely sequential part, for ONE row of already-projected values.

        Four dispatches carry it -- prepare, the two recurrence passes, and the output norm.
        Doing the same arithmetic with tensor ops is ~50 dispatches, and at 0.3-0.9 ms of
        fixed overhead each that loses to numpy however little data is involved."""
        nq = self.hk * self.dk
        nv = self.hv * self.dv
        zero = self._zero_t()
        g = state.gpu()
        packed = wt._empty((2 * nq + nv + 2 * self.hv,))
        flags = ((1 if self._has_conv else 0) | (2 if self.w.get("dt_bias") is not None else 0)
                 | (4 if self.w.get("A") is not None else 0) | (8 if braw is not None else 0)
                 | (16 if self.w.get("conv_b") is not None else 0))
        # Both kernels hand the state back: in place on a backend that can write it that
        # way, and as a new buffer on one that cannot. Storing what comes back is what makes
        # the second case free -- the alternative is a copy over the whole state per layer.
        packed, cnew = wt.gdn_prepare(
            wt._contig(qkv.data), (zero if braw is None else wt._contig(braw.data)),
            (zero if araw is None else wt._contig(araw.data)),
            g[1] if g[1] is not None else zero, self._konst(), packed,
            self.hk, self.hv, self.dk, self.dv, max(1, self.conv_width), flags)
        if g[1] is not None:
            g[1] = cnew
        out, g[0] = wt.gdn_step(g[0], packed, self.hv, self.dk, self.dv, self.rep)
        if self.w.get("norm") is not None:
            gw = self._norm_t()
            rows = self.hv if self._norm_per_head else 1
            r = wt.rmsnorm(out.reshape(rows, nv // rows), gw, self.eps)
            out = r if r is not None else out                # already a Tensor
        return out.reshape(1, nv)

    def _gate_out(self, y, gp, nv):
        """The output gate and projection, over however many rows `y` has."""
        if gp is not None:
            y = y * wt.silu(gp[:, :nv])
        o = self.w.get("o")
        return y if o is None else o(y)

    def _step_gpu(self, xt, state):
        """One token, everything on the device. `xt` is a (1, H) Tensor -> (1, H) Tensor."""
        nv = self.hv * self.dv
        qkv, braw, araw, gp = self._project_gpu(xt)
        y = self._recur_gpu(qkv.reshape(-1), braw, araw, state)
        return self._gate_out(y, None if gp is None else gp.reshape(1, -1), nv)

    def _zero_t(self):
        return self._cache_t("zero", lambda: wt.xp.zeros((1,), np.float32))

    @property
    def _has_conv(self):
        return self.conv_width > 1 and self.w.get("conv_w") is not None

    def _konst(self):
        """conv_w (W,C) | conv_b (C) | A (hv) | dt_bias (hv), uploaded once."""
        def build():
            nq = self.hk * self.dk
            C = 2 * nq + self.hv * self.dv
            parts = []
            W = max(1, self.conv_width)
            if self._has_conv:
                cw = np.asarray(self.w["conv_w"], np.float32).reshape(C, W)
                parts.append(np.ascontiguousarray(cw.T).reshape(-1))
            else:
                parts.append(np.zeros(W * C, np.float32))
            cb = self.w.get("conv_b")
            parts.append(np.asarray(cb, np.float32).reshape(-1)[:C] if cb is not None
                         else np.zeros(C, np.float32))
            for key in ("A", "dt_bias"):
                v = self.w.get(key)
                parts.append(np.asarray(v, np.float32).reshape(-1)[:self.hv] if v is not None
                             else np.zeros(self.hv, np.float32))
            return wt.xp.asarray(np.concatenate(parts).astype(np.float32))
        return self._cache_t("konst", build)

    def _cache_t(self, key, build):
        c = self.__dict__.setdefault("_tcache", {})
        if key not in c:
            c[key] = build()
        return c[key]

    def _conv_w_t(self):
        return self._cache_t("cw", lambda: wt.Tensor(
            np.ascontiguousarray(np.asarray(self.w["conv_w"], np.float32).T)))

    def _conv_b_t(self):
        return self._cache_t("cb", lambda: wt.Tensor(
            np.asarray(self.w["conv_b"], np.float32).reshape(-1)))

    def _dtb_t(self):
        d = self.w.get("dt_bias")
        return None if d is None else self._cache_t(
            "dtb", lambda: wt.Tensor(np.asarray(d, np.float32).reshape(-1)))

    def _A_t(self):
        a = self.w.get("A")
        return None if a is None else self._cache_t(
            "A", lambda: wt.Tensor(np.asarray(a, np.float32).reshape(-1)))

    def _norm_t(self):
        gw = np.asarray(self.w["norm"], np.float32).reshape(-1)
        self._norm_per_head = gw.size == self.dv
        return self._cache_t("nrm", lambda: wt.Tensor(
            gw.reshape(1, self.dv) if gw.size == self.dv else gw.reshape(1, -1)))

    # Off by default, and measured rather than assumed. Keeping the step on the device is
    # only worth it once it is FUSED: every tensor op here costs 0.3-0.9 ms of fixed
    # dispatch overhead in this runtime -- near enough the same for 256 elements as for
    # 260k, so it is the call, not the data -- and the step in its unfused form is ~50 of
    # them, which came out slower (28.2 ms) than doing the whole thing in numpy (24.8 ms).
    # The recurrence kernels below are verified correct to 4e-7 across steps and stay here
    # for the fused version; flipping this on without fusing makes decode slower.
    _GDN_GPU = True

    def _gpu_step_ok(self):
        """Is every piece this layer uses available on the device?"""
        if not self._GDN_GPU or not (wt._adam_backend_ready() or wt._webgl_ready()):
            return False
        return all(not callable(self.w.get(n)) or hasattr(self.w.get(n), "forward")
                   for n in ("qkv", "q", "k", "v", "beta", "alpha", "g", "o")
                   if self.w.get(n) is not None)

    def forward(self, x, state):
        """x: (T, H) ndarray -> (T, H). Sequential over T (the recurrence is inherently
        sequential); prefill and single-token decode use the same code."""
        T0 = int(getattr(x, "shape", (0,))[0] or 0)
        if T0 >= 1 and self._gpu_step_ok():
            xt = x if isinstance(x, wt.Tensor) else wt.Tensor(np.asarray(x, np.float32))
            if T0 == 1:
                return self._step_gpu(xt.reshape(1, -1), state)
            # A prompt projects ONCE and then recurs per position. The recurrence is
            # sequential -- state t depends on state t-1 -- but the projections are not, and
            # stepping the whole layer per token re-read every projection weight T times.
            # That is why a prompt cost more per token than decoding did, which is backwards.
            nv = self.hv * self.dv
            qkv, braw, araw, gp = self._project_gpu(xt)
            outs = [self._recur_gpu(qkv[t:t + 1].reshape(-1),
                                    None if braw is None else braw[t:t + 1],
                                    None if araw is None else araw[t:t + 1],
                                    state)
                    for t in range(T0)]
            return self._gate_out(wt.cat(outs, axis=0), gp, nv)
        state = state.host()                       # host path owns the state from here
        x = np.asarray(x.numpy() if hasattr(x, "numpy") else x, np.float32)
        T = x.shape[0]
        nq = self.hk * self.dk; nk = nq
        if self.w.get("qkv") is not None:          # single fused [q|k|v] projection
            flat0 = self._proj("qkv", x).reshape(T, -1)
            q = flat0[:, :nq].reshape(T, self.hk, self.dk)
            k = flat0[:, nq:nq + nk].reshape(T, self.hk, self.dk)
            v = flat0[:, nq + nk:nq + nk + self.hv * self.dv].reshape(T, self.hv, self.dv)
        else:                                      # separate projections
            q = self._proj("q", x).reshape(T, self.hk, self.dk)
            k = self._proj("k", x).reshape(T, self.hk, self.dk)
            v = self._proj("v", x).reshape(T, self.hv, self.dv)
        if self.conv_width > 1 and self.w.get("conv_w") is not None:
            flat = np.concatenate([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], 1)
            flat = causal_conv1d(flat, self.w["conv_w"], self.w.get("conv_b"), state.conv)
            flat = _silu(flat)
            q = flat[:, :nq].reshape(T, self.hk, self.dk)
            k = flat[:, nq:nq + nk].reshape(T, self.hk, self.dk)
            v = flat[:, nq + nk:].reshape(T, self.hv, self.dv)
        # queries and keys are L2-normalised, and q carries the 1/sqrt(d) scale
        q = _l2norm(q, self.eps) * (1.0 / math.sqrt(self.dk))
        k = _l2norm(k, self.eps)

        # Write strength: beta = sigmoid(W_beta x)  (its own projection).
        b_raw = self._proj("beta", x)
        beta = (_sigmoid(b_raw) if b_raw is not None
                else np.ones((T, self.hv), np.float32))

        # Forget gate: decay = exp(softplus(W_alpha x + dt_bias) * A), with A <= 0 so the decay
        # stays in (0, 1]. `dt_bias` and `A` are per-head vectors from the model.
        a_raw = self._proj("alpha", x)
        if a_raw is not None:
            a_raw = a_raw.reshape(T, -1)
            dtb = self.w.get("dt_bias")
            if dtb is not None:
                a_raw = a_raw + np.asarray(dtb, np.float32).reshape(1, -1)
            g = _softplus(a_raw)
            A = self.w.get("A")
            if A is not None:
                g = g * np.asarray(A, np.float32).reshape(1, -1)
            decay = np.exp(np.clip(g, -60, 0))
        else:
            decay = np.ones((T, self.hv), np.float32)
        decay = np.broadcast_to(decay.reshape(T, -1)[:, :self.hv], (T, self.hv))
        beta = np.broadcast_to(np.asarray(beta, np.float32).reshape(T, -1)[:, :self.hv], (T, self.hv))

        # Value head v takes key head `v % n_k_heads` -- ggml's gated_delta_net indexes
        # iq1 = iv1 % neq1, so the key heads CYCLE across the value heads rather than each
        # covering a contiguous block. np.repeat gives the block mapping and is wrong here:
        # it pairs every query with the wrong key for all but the first of each group.
        qh = np.tile(q, (1, self.rep, 1)) if self.rep > 1 else q
        kh = np.tile(k, (1, self.rep, 1)) if self.rep > 1 else k
        out = np.empty((T, self.hv, self.dv), np.float32)
        S = state.S
        for t in range(T):                                    # inherently sequential recurrence
            out[t] = gated_delta_step(S, qh[t], kh[t], v[t], decay[t], beta[t])
        if self.w.get("norm") is not None:
            gw = np.asarray(self.w["norm"], np.float32).reshape(-1)
            if gw.size == self.dv:                            # per-head weight -> normalise per head
                out = (out / np.sqrt((out * out).mean(-1, keepdims=True) + self.eps)
                       * gw.reshape(1, 1, self.dv))
            else:                                             # one weight over the flattened row
                flat_y = out.reshape(T, -1)
                out = (flat_y / np.sqrt((flat_y * flat_y).mean(-1, keepdims=True) + self.eps)
                       * gw.reshape(1, -1)).reshape(T, self.hv, self.dv)
        y = out.reshape(T, self.hv * self.dv)
        if self.w.get("g") is not None:                       # swish output gate (z)
            y = y * _silu(np.asarray(self._proj("g", x), np.float32).reshape(T, -1)[:, :y.shape[1]])
        o = self.w.get("o")
        if o is None:
            return y
        r = o(wt.Tensor(y))
        return np.asarray(r.numpy() if hasattr(r, "numpy") else r, np.float32)
