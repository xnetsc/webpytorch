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
import numpy as np

from . import _core as wt


class LinearAttentionState:
    """Recurrent state of one linear-attention layer: the (heads, k_dim, v_dim) matrix plus the
    causal-conv ring buffer. Fixed size — it does not grow with sequence length."""

    def __init__(self, n_v_heads, k_dim, v_dim, conv_width, conv_channels):
        self.S = np.zeros((n_v_heads, k_dim, v_dim), np.float32)
        self.conv = np.zeros((conv_width - 1, conv_channels), np.float32) if conv_width > 1 else None

    def reset(self):
        self.S[:] = 0.0
        if self.conv is not None:
            self.conv[:] = 0.0


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
        self.hk = int(cfg["n_k_heads"]); self.hv = int(cfg["n_v_heads"])
        self.dk = int(cfg["k_head_dim"]); self.dv = int(cfg["v_head_dim"])
        self.conv_width = int(cfg.get("conv_kernel_dim", 0) or 0)
        self.H = int(cfg["hidden"]); self.w = w; self.eps = eps
        self.rep = max(1, self.hv // max(1, self.hk))        # key/value head grouping

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

    def forward(self, x, state):
        """x: (T, H) ndarray -> (T, H). Sequential over T (the recurrence is inherently
        sequential); prefill and single-token decode use the same code."""
        x = np.asarray(x, np.float32)
        T = x.shape[0]
        q = self._proj("q", x).reshape(T, self.hk, self.dk)
        k = self._proj("k", x).reshape(T, self.hk, self.dk)
        v = self._proj("v", x).reshape(T, self.hv, self.dv)
        if self.conv_width > 1 and self.w.get("conv_w") is not None:
            flat = np.concatenate([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], 1)
            flat = causal_conv1d(flat, self.w["conv_w"], self.w.get("conv_b"), state.conv)
            flat = _silu(flat)
            nq = self.hk * self.dk; nk = nq
            q = flat[:, :nq].reshape(T, self.hk, self.dk)
            k = flat[:, nq:nq + nk].reshape(T, self.hk, self.dk)
            v = flat[:, nq + nk:].reshape(T, self.hv, self.dv)
        k = _l2norm(k, self.eps)
        # gates: decay in (0,1] via sigmoid(-softplus) style, write strength via sigmoid
        a = self._proj("a", x); dt = self._proj("dt", x)
        decay = (1.0 / (1.0 + np.exp(-np.clip(a, -60, 60)))) if a is not None else np.ones((T, self.hv), np.float32)
        beta = (1.0 / (1.0 + np.exp(-np.clip(dt, -60, 60)))) if dt is not None else np.ones((T, self.hv), np.float32)
        decay = np.broadcast_to(decay.reshape(T, -1)[:, :self.hv], (T, self.hv))
        beta = np.broadcast_to(beta.reshape(T, -1)[:, :self.hv], (T, self.hv))

        qh = np.repeat(q, self.rep, axis=1) if self.rep > 1 else q      # group k/q heads to v heads
        kh = np.repeat(k, self.rep, axis=1) if self.rep > 1 else k
        out = np.empty((T, self.hv, self.dv), np.float32)
        S = state.S
        for t in range(T):                                    # inherently sequential recurrence
            out[t] = gated_delta_step(S, qh[t], kh[t], v[t], decay[t], beta[t])
        y = out.reshape(T, self.hv * self.dv)
        if self.w.get("norm") is not None:                    # output RMSNorm
            g = np.asarray(self.w["norm"], np.float32)
            y = y / np.sqrt((y * y).mean(-1, keepdims=True) + self.eps) * g
        if self.w.get("g") is not None:                       # swish output gate
            y = y * _silu(self._proj("g", x))
        o = self.w.get("o")
        if o is None:
            return y
        r = o(wt.Tensor(y))
        return np.asarray(r.numpy() if hasattr(r, "numpy") else r, np.float32)
