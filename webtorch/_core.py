"""webtorch — a minimal PyTorch-compatible shim with define-by-run autograd,
backed by WgPy's cupy arrays (GPU via WebGPU/WebGL) inside Pyodide.

Phase-2 vertical slice: enough to build and TRAIN a small MLP and verify that
gradients (computed on the GPU) match numerical finite differences. Conv2d /
attention / more ops come later; the autograd core here is what everything else
builds on.
"""
import numpy as np

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:  # no GPU backend -> fall back to numpy CPU
    xp = np
    GPU = False


def _to_xp(x):
    # WgPy's cupy shim: asarray has no dtype kwarg and no cp.float32 — cast on
    # the numpy side first, then move to the GPU array type.
    if isinstance(x, xp.ndarray):
        return x
    return xp.asarray(np.asarray(x, dtype=np.float32))


def _swap_last2(a):
    # Transpose the last two axes as a VIEW (no copy). Verified on both backends:
    # WgPy matmul reads transposed strides correctly (LHS, RHS, and 3D bmm), so the
    # old `* 1.0` materialization is unnecessary — dropping it removes a GPU->GPU
    # copy per transpose in every Linear/attention backward. Every consumer here
    # either feeds matmul (stride-aware) or is wrapped in _contig before reshape.
    if a.ndim == 2:
        return xp.transpose(a, (1, 0))
    axes = list(range(a.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return xp.transpose(a, axes)


def _ipow(a, k):
    # Integer power via repeated multiplication — WgPy's `**` operator returns
    # wrong values (not just for k==1), so never use it.
    if k == 0:
        return xp.ones_like(a)
    r = a
    for _ in range(k - 1):
        r = r * a
    return r


def _unbroadcast(grad, shape):
    """Reduce `grad` so its shape matches `shape` (reverse of broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape)):
        if shape[i] == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = _to_xp(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = tuple(_children)   # NOT a set — allow __eq__ override (torch shim)
        self._op = _op

    # ---- properties -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def _accum(self, g):
        if self.grad is None:
            self.grad = xp.zeros_like(self.data)
        self.grad = self.grad + g

    # ---- ops --------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(out.grad, other.data.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(other.data * out.grad, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(self.data * out.grad, other.data.shape))
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad @ _swap_last2(other.data), self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(_swap_last2(self.data) @ out.grad, other.data.shape))
        out._backward = _backward
        return out

    __matmul__ = matmul

    def relu(self):
        out = Tensor(xp.maximum(self.data, 0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                mask = (self.data > 0).astype(np.float32)
                self._accum(mask * out.grad)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            out = Tensor(self.data.sum().reshape(()), self.requires_grad, (self,), "sum")

            def _backward():
                if self.requires_grad:
                    self._accum(xp.ones_like(self.data) * out.grad)
            out._backward = _backward
            return out
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims:
                    shp = list(self.data.shape)
                    for a in axes:
                        shp[a % self.data.ndim] = 1
                    g = g.reshape(*shp)
                self._accum(xp.ones_like(self.data) * g)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            return self.sum() * (1.0 / self.data.size)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        n = 1
        for a in axes:
            n *= self.data.shape[a % self.data.ndim]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # resolve a single -1 explicitly (WgPy reshape may not infer it)
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape = tuple(self.data.size // known if s == -1 else s for s in shape)
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accum(out.grad.reshape(*old_shape))
        out._backward = _backward
        return out

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        # materialize (transposed views break WgPy reshape, and permute usually
        # feeds a reshape — e.g. multi-head split/merge)
        out = Tensor(_contig(xp.transpose(self.data, axes)), self.requires_grad, (self,), "permute")
        inv = [0] * len(axes)
        for i, a in enumerate(axes):
            inv[a] = i

        def _backward():
            if self.requires_grad:
                self._accum(_contig(xp.transpose(out.grad, tuple(inv))))
        out._backward = _backward
        return out

    def transpose(self, a, b):
        axes = list(range(self.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.permute(*axes)

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, p):
        assert isinstance(p, int) and p >= 0, "only non-negative integer powers supported"
        out = Tensor(_ipow(self.data, p), self.requires_grad, (self,), f"**{p}")

        def _backward():
            if self.requires_grad:
                self._accum((p * _ipow(self.data, p - 1)) * out.grad)
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data,
                     self.requires_grad or other.requires_grad, (self, other), "/")

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad / other.data, self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(-self.data / (other.data * other.data) * out.grad, other.data.shape))
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def _unary(self, val, grad_fn, op):
        out = Tensor(val, self.requires_grad, (self,), op)

        def _backward():
            if self.requires_grad:
                self._accum(grad_fn(out.grad, out.data))
        out._backward = _backward
        return out

    def exp(self):
        return self._unary(xp.exp(self.data), lambda g, o: o * g, "exp")

    def log(self):
        return self._unary(xp.log(self.data), lambda g, o: g / self.data, "log")

    def sqrt(self):
        return self._unary(xp.sqrt(self.data), lambda g, o: g / (2.0 * o), "sqrt")

    def tanh(self):
        return self._unary(xp.tanh(self.data), lambda g, o: (1.0 - o * o) * g, "tanh")

    def sigmoid(self):
        return self._unary(1.0 / (1.0 + xp.exp(-self.data)), lambda g, o: o * (1.0 - o) * g, "sigmoid")

    def abs(self):
        sign = (self.data > 0).astype(np.float32) - (self.data < 0).astype(np.float32)
        return self._unary(xp.maximum(self.data, -self.data), lambda g, o: sign * g, "abs")

    # ---- autograd ---------------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:      # id-based (not value-eq) so __eq__ can be overridden
                visited.add(id(v))
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)

        self.grad = xp.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def numpy(self):
        return cp.asnumpy(self.data) if GPU else np.asarray(self.data)

    def item(self):
        return float(self.numpy())

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, grad={self.requires_grad})"


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


# GPU concatenate. WgPy's xp.concatenate is HOST-based (asnumpy each input ->
# np.concatenate -> upload), so it is NOT a recorded GPU kernel and a captured
# graph containing it replays with a FROZEN output (breaks decode capture --
# rope's rotate_half and the lm_head both cat). This kernel copies each input
# into its slice of the output on the GPU, so it's capture-safe.
_CAT_WGSL = """@group(0) @binding(0) var<storage,read_write> outp: array<f32>;
@group(0) @binding(1) var<storage,read> src: array<f32>;
struct M { pre:u32, Ni:u32, post:u32, W:u32, off:u32, }
@group(0) @binding(2) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = m.pre * m.Ni * m.post;
  if (i >= total) { return; }
  let q = i % m.post;
  let n = (i / m.post) % m.Ni;
  let p = i / (m.post * m.Ni);
  outp[p * m.W * m.post + (m.off + n) * m.post + q] = src[i];
}
"""
_catk = {"added": False}


def _cat_gpu_data(datas, axis):
    shapes = [d.shape for d in datas]
    W = sum(int(s[axis]) for s in shapes)
    outsh = list(shapes[0]); outsh[axis] = W
    out = _empty(tuple(outsh))
    pre = 1
    for x in outsh[:axis]:
        pre *= int(x)
    post = 1
    for x in outsh[axis + 1:]:
        post *= int(x)
    plat = _adam_kernel["platform"]
    if not _catk["added"]:
        plat.addKernel("cat_copy", {"source": _CAT_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "read-only-storage"]})
        _catk["added"] = True
    off = 0
    for d in datas:
        Ni = int(d.shape[axis])
        total = pre * Ni * post
        meta = _adam_kernel["make_meta"]((int(pre), Ni, int(post), int(W), int(off)), "u4,u4,u4,u4,u4")
        plat.runKernel({"name": "cat_copy",
            "tensors": [out.buffer.buffer_id, _contig(d).buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (total + 63) // 64, "y": 1, "z": 1}})
        off += Ni
    return out


def cat(tensors, axis=0):
    tensors = list(tensors)
    nd = tensors[0].ndim
    if axis < 0:
        axis += nd
    if _adam_backend_ready():
        outdata = _cat_gpu_data([t.data for t in tensors], axis)   # GPU, capture-safe
    else:
        outdata = xp.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(outdata, any(t.requires_grad for t in tensors), tuple(tensors), "cat")
    sizes = [t.shape[axis] for t in tensors]

    def _backward():
        off = 0
        for t, sz in zip(tensors, sizes):
            if t.requires_grad:
                idx = [slice(None)] * nd
                idx[axis] = slice(off, off + sz)
                t._accum(_contig(out.grad[tuple(idx)]))
            off += sz
    out._backward = _backward
    return out


def stack(tensors, axis=0):
    nd = tensors[0].ndim
    if axis < 0:
        axis += nd + 1
    expanded = [t.reshape(*(t.shape[:axis] + (1,) + t.shape[axis:])) for t in tensors]
    return cat(expanded, axis=axis)


_SIN_WGSL = """@group(0) @binding(0) var<storage,read_write> o: array<f32>;
@group(0) @binding(1) var<storage,read> s: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= arrayLength(&s)) { return; }
  o[i] = sin(s[i]);
}
"""
_sink = {"added": False}


def _sin_data(d):
    # WgPy cupy has no xp.sin -> custom elementwise WGSL kernel on WebGPU; xp.sin otherwise.
    if not _adam_backend_ready():
        return xp.sin(d)
    plat = _adam_kernel["platform"]
    if not _sink["added"]:
        plat.addKernel("sin_k", {"source": _SIN_WGSL, "bindingTypes": ["storage", "read-only-storage"]})
        _sink["added"] = True
    dc = _contig(d); out = _empty(dc.shape)
    n = 1
    for s in dc.shape:
        n *= int(s)
    plat.runKernel({"name": "sin_k",
        "tensors": [out.buffer.buffer_id, dc.buffer.buffer_id],
        "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return out


def sin(t):
    return Tensor(_sin_data(t.data if isinstance(t, Tensor) else t))


def argmax(x, axis=-1):
    d = cp.asnumpy(x.data) if GPU else np.asarray(x.data)
    return d.argmax(axis=axis)


# ---- nn -------------------------------------------------------------------
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self):
        seen, out = set(), []
        for v in vars(self).values():
            if isinstance(v, Parameter):
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for it in v:
                    if isinstance(it, Module):
                        out.extend(it.parameters())
                    elif isinstance(it, Parameter):
                        out.append(it)
        # de-dup preserving order
        uniq = []
        for p in out:
            if id(p) not in seen:
                seen.add(id(p)); uniq.append(p)
        return uniq

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class Linear(Module):
    def __init__(self, in_features, out_features):
        # He initialization (uses numpy RNG, then moves to GPU)
        w = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros((out_features,), dtype=np.float32))

    def forward(self, x):
        if x.ndim == 2:
            return x.matmul(self.weight) + self.bias
        # fold leading dims (WgPy matmul is 2D-only): (..., in) -> (prod, in)
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        out = flat.matmul(self.weight) + self.bias
        return out.reshape(*lead, self.weight.shape[1])


class ReLU(Module):
    def forward(self, x):
        return x.relu()


def gelu(x):
    # tanh approximation (as used in GPT). Composed from autograd primitives, so
    # it works on both backends; a fused kernel is a later optimization.
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * c
    return x * (inner.tanh() + 1.0) * 0.5


def silu(x):
    return x * x.sigmoid()


class GELU(Module):
    def forward(self, x):
        return gelu(x)


class SiLU(Module):
    def forward(self, x):
        return silu(x)


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # inverted dropout; mask is a constant per forward (fine for capture only
        # if fixed — for training use eval() or p=0 under capture)
        keep = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(keep)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        KH = KW = kernel_size
        fan_in = in_channels * KH * KW
        w = np.random.randn(out_channels, in_channels, KH, KW).astype(np.float32) * np.sqrt(2.0 / fan_in)
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros((out_channels,), dtype=np.float32))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


class Flatten(Module):
    def forward(self, x):
        n = x.shape[0]
        return x.reshape(n, -1)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.weight = Parameter(np.ones((dim,), dtype=np.float32))
        self.bias = Parameter(np.zeros((dim,), dtype=np.float32))
        self.eps = eps

    def forward(self, x):
        return layernorm(x, self.weight, self.bias, self.eps)


class Embedding(Module):
    def __init__(self, num_embeddings, dim):
        w = np.random.randn(num_embeddings, dim).astype(np.float32) * 0.02
        self.weight = Parameter(w)

    def forward(self, idx):
        return embedding(self.weight, idx)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = Parameter(np.ones((dim,), dtype=np.float32))
        self.eps = eps

    def forward(self, x):
        ms = (x * x).mean(axis=-1, keepdims=True)
        return x / (ms + self.eps).sqrt() * self.weight


class MultiheadAttention(Module):
    """Batched multi-head self-attention with optional causal masking."""
    def __init__(self, dim, n_heads, causal=False):
        assert dim % n_heads == 0
        self.h = n_heads
        self.hd = dim // n_heads
        self.dim = dim
        self.causal = causal
        self.wq = Linear(dim, dim); self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim); self.wo = Linear(dim, dim)

    def forward(self, x):
        B, Tn, D = x.shape
        h, hd = self.h, self.hd

        def split(t):  # (B,T,D) -> (B*h, T, hd)
            return t.reshape(B, Tn, h, hd).permute(0, 2, 1, 3).reshape(B * h, Tn, hd)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        scores = bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))
        if self.causal:
            mask = np.triu(np.full((Tn, Tn), -1e9, dtype=np.float32), 1)
            scores = scores + Tensor(mask)
        o = bmm(softmax(scores), v)                               # (B*h, T, hd)
        o = o.reshape(B, h, Tn, hd).permute(0, 2, 1, 3).reshape(B, Tn, D)
        return self.wo(o)


def _max_lastdim(x):
    """Max over the last axis, with gradient routed to the argmax (ties split)."""
    md = x.data.max(axis=-1, keepdims=True)
    out = Tensor(x.data.max(axis=-1), x.requires_grad, (x,), "max")

    def _backward():
        if x.requires_grad:
            mask = (x.data >= md).astype(np.float32)
            cnt = mask.sum(axis=-1, keepdims=True)
            g = out.grad.reshape(*(list(out.data.shape) + [1]))
            x._accum(mask / cnt * g)
    out._backward = _backward
    return out


class AvgPool2d(Module):
    def __init__(self, kernel_size):
        self.k = kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        return x.reshape(N, C, H // k, k, W // k, k).mean(axis=(3, 5))


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        self.k = kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        xr = x.reshape(N, C, H // k, k, W // k, k).permute(0, 1, 2, 4, 3, 5).reshape(N, C, H // k, W // k, k * k)
        return _max_lastdim(xr)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5):
        self.weight = Parameter(np.ones((num_features,), dtype=np.float32))
        self.bias = Parameter(np.zeros((num_features,), dtype=np.float32))
        self.eps = eps

    def forward(self, x):  # (N,C,H,W); normalize over N,H,W per channel (batch stats)
        mu = x.mean(axis=(0, 2, 3), keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=(0, 2, 3), keepdims=True)
        xhat = xc / (var + self.eps).sqrt()
        return xhat * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        self.g = num_groups
        self.weight = Parameter(np.ones((num_channels,), dtype=np.float32))
        self.bias = Parameter(np.zeros((num_channels,), dtype=np.float32))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        xg = x.reshape(N, self.g, (C // self.g) * H * W)
        mu = xg.mean(axis=-1, keepdims=True)
        xc = xg - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        xhat = (xc / (var + self.eps).sqrt()).reshape(N, C, H, W)
        return xhat * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


def mse_loss(pred, target):
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = pred - target
    return (diff * diff).mean()  # avoid ** (WgPy pow is fragile); mul is verified


def l1_loss(pred, target):
    if not isinstance(target, Tensor):
        target = Tensor(target)
    return (pred - target).abs().mean()


def bce_loss(pred, target):
    """Binary cross-entropy; `pred` is a probability in (0,1)."""
    if not isinstance(target, Tensor):
        target = Tensor(target)
    eps = 1e-7
    return -(target * (pred + eps).log() + (1.0 - target) * (1.0 - pred + eps).log()).mean()


def nll_loss(log_probs, targets):
    """Negative log-likelihood. log_probs: (N, C); targets: numpy int (N,).
    cross_entropy == nll_loss(log_softmax(x))."""
    ld = log_probs.data
    N, C = ld.shape
    onehot = Tensor(xp.asarray(np.eye(C, dtype=np.float32)[np.asarray(targets).astype(np.int64)]))
    return -(onehot * log_probs).sum() * (1.0 / N)


# ---- conv2d (im2col + matmul) --------------------------------------------
def _zeros(shape):
    # GPU-native allocation (no CPU->GPU upload). cp.zeros defaults to float64 and
    # cp.float32 doesn't exist in the shim, so pass numpy's np.float32.
    return xp.zeros(shape, np.float32)


def _empty(shape):
    # GPU-native, uninitialized — for buffers a kernel fully overwrites (Adam temps,
    # softmax/ln outputs). Skips even the zero-fill.
    return xp.empty(shape, np.float32)


# GLSL helper: read element `idx` of a texture laid out row-major (width from the
# texture). Shared by all WebGL kernels. Defined early so module-level kernel
# strings that .replace("FETCH", _GL_FETCH) can use it.
_GL_FETCH = ("float fetch(sampler2D t, int idx) { int tw = textureSize(t, 0).x; "
             "int y = idx / tw; int x = idx - y * tw; return texelFetch(t, ivec2(x, y), 0).r; }")


def _contig(a):
    # materialize a (possibly transposed/strided) array — WgPy reshape/matmul
    # are unreliable on non-contiguous views; `* 1.0` forces a stride-aware kernel.
    return a * 1.0


def _pad2d(x, ph, pw):
    if ph == 0 and pw == 0:
        return x
    N, C, H, W = x.shape
    z = _zeros((N, C, H + 2 * ph, W + 2 * pw))
    z[:, :, ph:ph + H, pw:pw + W] = x
    return z


def _im2col(xpad, KH, KW, s):
    N, C, Hp, Wp = xpad.shape
    OH = (Hp - KH) // s + 1
    OW = (Wp - KW) // s + 1
    cols = _zeros((N, C, KH, KW, OH, OW))
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j, :, :] = xpad[:, :, i:i + s * OH:s, j:j + s * OW:s]
    return cols, OH, OW


def _col2im(dcols, N, C, Hp, Wp, KH, KW, s, OH, OW):
    dx = _zeros((N, C, Hp, Wp))
    for i in range(KH):
        for j in range(KW):
            dx[:, :, i:i + s * OH:s, j:j + s * OW:s] = \
                dx[:, :, i:i + s * OH:s, j:j + s * OW:s] + dcols[:, :, i, j, :, :]
    return dx


# ---- direct convolution kernels (no im2col) --------------------------------
# Meta order everywhere: (N, Cin, Cout, H, W, OH, OW, KH, KW, stride, pad).
_CONV_STRUCT = "struct CMeta { N:u32,Cin:u32,Cout:u32,H:u32,W:u32,OH:u32,OW:u32,KH:u32,KW:u32,stride:u32,pad:u32, }\n@group(0) @binding(BND) var<storage,read> c: CMeta;\n"
_CONV_FWD_WGSL = """@group(0) @binding(0) var<storage,read> xin: array<f32>;
@group(0) @binding(1) var<storage,read> wt: array<f32>;
@group(0) @binding(2) var<storage,read> bs: array<f32>;
@group(0) @binding(3) var<storage,read_write> outp: array<f32>;
""" + _CONV_STRUCT.replace("BND", "4") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.N*c.Cout*c.OH*c.OW) { return; }
  let ow = i % c.OW; var t = i / c.OW; let oh = t % c.OH; t = t / c.OH; let co = t % c.Cout; let n = t / c.Cout;
  var sum = bs[co];
  for (var ci:u32=0u; ci<c.Cin; ci++) { for (var kh:u32=0u; kh<c.KH; kh++) {
    let ih = i32(oh*c.stride+kh) - i32(c.pad); if (ih<0 || ih>=i32(c.H)) { continue; }
    for (var kw:u32=0u; kw<c.KW; kw++) {
      let iw = i32(ow*c.stride+kw) - i32(c.pad); if (iw<0 || iw>=i32(c.W)) { continue; }
      sum = sum + xin[((n*c.Cin+ci)*c.H+u32(ih))*c.W+u32(iw)] * wt[((co*c.Cin+ci)*c.KH+kh)*c.KW+kw];
    }
  } }
  outp[i] = sum;
}
"""
_CONV_DIN_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read> wt: array<f32>;
@group(0) @binding(2) var<storage,read_write> dx: array<f32>;
""" + _CONV_STRUCT.replace("BND", "3") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.N*c.Cin*c.H*c.W) { return; }
  let iw = i % c.W; var t = i / c.W; let ih = t % c.H; t = t / c.H; let ci = t % c.Cin; let n = t / c.Cin;
  var sum: f32 = 0.0;
  for (var co:u32=0u; co<c.Cout; co++) { for (var kh:u32=0u; kh<c.KH; kh++) {
    let a = i32(ih)+i32(c.pad)-i32(kh); if (a<0 || (a%i32(c.stride))!=0) { continue; }
    let oh = a/i32(c.stride); if (oh>=i32(c.OH)) { continue; }
    for (var kw:u32=0u; kw<c.KW; kw++) {
      let b2 = i32(iw)+i32(c.pad)-i32(kw); if (b2<0 || (b2%i32(c.stride))!=0) { continue; }
      let ow = b2/i32(c.stride); if (ow>=i32(c.OW)) { continue; }
      sum = sum + dout[((n*c.Cout+co)*c.OH+u32(oh))*c.OW+u32(ow)] * wt[((co*c.Cin+ci)*c.KH+kh)*c.KW+kw];
    }
  } }
  dx[i] = sum;
}
"""
_CONV_DW_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read> xin: array<f32>;
@group(0) @binding(2) var<storage,read_write> dw: array<f32>;
""" + _CONV_STRUCT.replace("BND", "3") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.Cout*c.Cin*c.KH*c.KW) { return; }
  let kw = i % c.KW; var t = i / c.KW; let kh = t % c.KH; t = t / c.KH; let ci = t % c.Cin; let co = t / c.Cin;
  var sum: f32 = 0.0;
  for (var n:u32=0u; n<c.N; n++) { for (var oh:u32=0u; oh<c.OH; oh++) {
    let ih = i32(oh*c.stride+kh) - i32(c.pad); if (ih<0 || ih>=i32(c.H)) { continue; }
    for (var ow:u32=0u; ow<c.OW; ow++) {
      let iw = i32(ow*c.stride+kw) - i32(c.pad); if (iw<0 || iw>=i32(c.W)) { continue; }
      sum = sum + dout[((n*c.Cout+co)*c.OH+oh)*c.OW+ow] * xin[((n*c.Cin+ci)*c.H+u32(ih))*c.W+u32(iw)];
    }
  } }
  dw[i] = sum;
}
"""
_CONV_DB_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read_write> db: array<f32>;
""" + _CONV_STRUCT.replace("BND", "2") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let co = gid.x; if (co >= c.Cout) { return; }
  var sum: f32 = 0.0;
  for (var n:u32=0u; n<c.N; n++) { for (var oh:u32=0u; oh<c.OH; oh++) { for (var ow:u32=0u; ow<c.OW; ow++) {
    sum = sum + dout[((n*c.Cout+co)*c.OH+oh)*c.OW+ow];
  } } }
  db[co] = sum;
}
"""
_CONV_GL_U = "uniform int _ka_tex_output_texture_w; uniform int N,Cin,Cout,H,W,OH,OW,KH,KW,stride,pad;"
_GL_CONV_FWD = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
    + _CONV_GL_U + "\nuniform sampler2D tex_x, tex_w, tex_b;\nout float fragColor;\nFETCH\n"
    + """void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=N*Cout*OH*OW){fragColor=0.0;return;}
  int ow=i%OW; int t=i/OW; int oh=t%OH; t/=OH; int co=t%Cout; int n=t/Cout;
  float sum=fetch(tex_b,co);
  for(int ci=0;ci<Cin;ci++)for(int kh=0;kh<KH;kh++){int ih=oh*stride+kh-pad; if(ih<0||ih>=H)continue;
    for(int kw=0;kw<KW;kw++){int iw=ow*stride+kw-pad; if(iw<0||iw>=W)continue;
      sum+=fetch(tex_x,((n*Cin+ci)*H+ih)*W+iw)*fetch(tex_w,((co*Cin+ci)*KH+kh)*KW+kw);}}
  fragColor=sum;
}""").replace("FETCH", _GL_FETCH)
_GL_CONV_DIN = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
    + _CONV_GL_U + "\nuniform sampler2D tex_g, tex_w;\nout float fragColor;\nFETCH\n"
    + """void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=N*Cin*H*W){fragColor=0.0;return;}
  int iw=i%W; int t=i/W; int ih=t%H; t/=H; int ci=t%Cin; int n=t/Cin;
  float sum=0.0;
  for(int co=0;co<Cout;co++)for(int kh=0;kh<KH;kh++){int a=ih+pad-kh; if(a<0||a%stride!=0)continue; int oh=a/stride; if(oh>=OH)continue;
    for(int kw=0;kw<KW;kw++){int b2=iw+pad-kw; if(b2<0||b2%stride!=0)continue; int ow=b2/stride; if(ow>=OW)continue;
      sum+=fetch(tex_g,((n*Cout+co)*OH+oh)*OW+ow)*fetch(tex_w,((co*Cin+ci)*KH+kh)*KW+kw);}}
  fragColor=sum;
}""").replace("FETCH", _GL_FETCH)
_GL_CONV_DW = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
    + _CONV_GL_U + "\nuniform sampler2D tex_g, tex_x;\nout float fragColor;\nFETCH\n"
    + """void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=Cout*Cin*KH*KW){fragColor=0.0;return;}
  int kw=i%KW; int t=i/KW; int kh=t%KH; t/=KH; int ci=t%Cin; int co=t/Cin;
  float sum=0.0;
  for(int n=0;n<N;n++)for(int oh=0;oh<OH;oh++){int ih=oh*stride+kh-pad; if(ih<0||ih>=H)continue;
    for(int ow=0;ow<OW;ow++){int iw=ow*stride+kw-pad; if(iw<0||iw>=W)continue;
      sum+=fetch(tex_g,((n*Cout+co)*OH+oh)*OW+ow)*fetch(tex_x,((n*Cin+ci)*H+ih)*W+iw);}}
  fragColor=sum;
}""").replace("FETCH", _GL_FETCH)
_GL_CONV_DB = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
    + _CONV_GL_U + "\nuniform sampler2D tex_g;\nout float fragColor;\nFETCH\n"
    + """void main(){
  int co=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(co>=Cout){fragColor=0.0;return;}
  float sum=0.0;
  for(int n=0;n<N;n++)for(int oh=0;oh<OH;oh++)for(int ow=0;ow<OW;ow++) sum+=fetch(tex_g,((n*Cout+co)*OH+oh)*OW+ow);
  fragColor=sum;
}""").replace("FETCH", _GL_FETCH)
_conv_k = {"added": False, "gl": False}


def _conv_gl_uniforms(dims, out_w):
    u = [{"name": "_ka_tex_output_texture_w", "value": out_w, "type": "int"}]
    for nm, val in zip(["N", "Cin", "Cout", "H", "W", "OH", "OW", "KH", "KW", "stride", "pad"], dims):
        u.append({"name": nm, "value": int(val), "type": "int"})
    return u


def _conv2d_fused(x, weight, bias, stride, pad):
    N, Cin, H, W = x.data.shape
    Cout, _, KH, KW = weight.data.shape
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1
    dims = (N, Cin, Cout, H, W, OH, OW, KH, KW, stride, pad)
    wgpu = _adam_backend_ready()
    if wgpu and not _conv_k["added"]:
        plat = _adam_kernel["platform"]
        plat.addKernel("conv_fwd", {"source": _CONV_FWD_WGSL, "bindingTypes": ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
        plat.addKernel("conv_din", {"source": _CONV_DIN_WGSL, "bindingTypes": ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
        plat.addKernel("conv_dw", {"source": _CONV_DW_WGSL, "bindingTypes": ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
        plat.addKernel("conv_db", {"source": _CONV_DB_WGSL, "bindingTypes": ["read-only-storage", "storage", "read-only-storage"]})
        _conv_k["added"] = True
    if (not wgpu) and not _conv_k["gl"]:
        plat = _copy_kernel["plat"]
        plat.addKernel("conv_fwd", {"source": _GL_CONV_FWD})
        plat.addKernel("conv_din", {"source": _GL_CONV_DIN})
        plat.addKernel("conv_dw", {"source": _GL_CONV_DW})
        plat.addKernel("conv_db", {"source": _GL_CONV_DB})
        _conv_k["gl"] = True

    def wmeta():
        return _adam_kernel["make_meta"](dims, _CONV_META_FMT).buffer_id

    out_data = _empty((N, Cout, OH, OW))
    if wgpu:
        plat = _adam_kernel["platform"]
        plat.runKernel({"name": "conv_fwd", "tensors": [x.data.buffer.buffer_id, weight.data.buffer.buffer_id, bias.data.buffer.buffer_id, out_data.buffer.buffer_id, wmeta()],
                        "workGroups": {"x": (N * Cout * OH * OW + 63) // 64, "y": 1, "z": 1}})
    else:
        plat = _copy_kernel["plat"]
        plat.runKernel({"name": "conv_fwd",
            "inputs": [{"name": "tex_x", "id": x.data.buffer.buffer_id}, {"name": "tex_w", "id": weight.data.buffer.buffer_id}, {"name": "tex_b", "id": bias.data.buffer.buffer_id}],
            "output": out_data.buffer.buffer_id, "uniforms": _conv_gl_uniforms(dims, out_data.buffer.texture_shape.width)})
    out = Tensor(out_data, x.requires_grad or weight.requires_grad or bias.requires_grad, (x, weight, bias), "conv2d")

    def _backward():
        g = _contig(out.grad)
        if x.requires_grad:
            dx = _empty((N, Cin, H, W))
            if wgpu:
                plat = _adam_kernel["platform"]
                plat.runKernel({"name": "conv_din", "tensors": [g.buffer.buffer_id, weight.data.buffer.buffer_id, dx.buffer.buffer_id, wmeta()],
                                "workGroups": {"x": (N * Cin * H * W + 63) // 64, "y": 1, "z": 1}})
            else:
                plat = _copy_kernel["plat"]
                plat.runKernel({"name": "conv_din", "inputs": [{"name": "tex_g", "id": g.buffer.buffer_id}, {"name": "tex_w", "id": weight.data.buffer.buffer_id}],
                                "output": dx.buffer.buffer_id, "uniforms": _conv_gl_uniforms(dims, dx.buffer.texture_shape.width)})
            x._accum(dx)
        if weight.requires_grad:
            dw = _empty((Cout, Cin, KH, KW))
            if wgpu:
                plat = _adam_kernel["platform"]
                plat.runKernel({"name": "conv_dw", "tensors": [g.buffer.buffer_id, x.data.buffer.buffer_id, dw.buffer.buffer_id, wmeta()],
                                "workGroups": {"x": (Cout * Cin * KH * KW + 63) // 64, "y": 1, "z": 1}})
            else:
                plat = _copy_kernel["plat"]
                plat.runKernel({"name": "conv_dw", "inputs": [{"name": "tex_g", "id": g.buffer.buffer_id}, {"name": "tex_x", "id": x.data.buffer.buffer_id}],
                                "output": dw.buffer.buffer_id, "uniforms": _conv_gl_uniforms(dims, dw.buffer.texture_shape.width)})
            weight._accum(dw)
        if bias.requires_grad:
            db = _empty((Cout,))
            if wgpu:
                plat = _adam_kernel["platform"]
                plat.runKernel({"name": "conv_db", "tensors": [g.buffer.buffer_id, db.buffer.buffer_id, wmeta()],
                                "workGroups": {"x": (Cout + 63) // 64, "y": 1, "z": 1}})
            else:
                plat = _copy_kernel["plat"]
                plat.runKernel({"name": "conv_db", "inputs": [{"name": "tex_g", "id": g.buffer.buffer_id}],
                                "output": db.buffer.buffer_id, "uniforms": _conv_gl_uniforms(dims, db.buffer.texture_shape.width)})
            bias._accum(db)
    out._backward = _backward
    return out


_CONV_META_FMT = "u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4"


def conv2d(x, weight, bias, stride=1, padding=0):
    """NCHW conv. weight: (Cout, Cin, KH, KW), bias: (Cout,). Direct-convolution
    kernels (fwd + dInput/dWeight/dBias) when a GPU backend is available; else the
    im2col fallback."""
    if _adam_backend_ready() or _webgl_ready():
        return _conv2d_fused(x, weight, bias, stride, padding)
    s, ph, pw = stride, padding, padding
    N, C, H, W = x.data.shape
    Cout, Cin, KH, KW = weight.data.shape
    assert Cin == C, f"conv2d channel mismatch: {Cin} vs {C}"

    xpad = _pad2d(x.data, ph, pw)
    cols, OH, OW = _im2col(xpad, KH, KW, s)                 # (N,C,KH,KW,OH,OW)
    cols2d = _contig(xp.transpose(cols, (0, 4, 5, 1, 2, 3))).reshape(N * OH * OW, C * KH * KW)
    Wcol = weight.data.reshape(Cout, C * KH * KW)          # (Cout, CKK)
    out2d = cols2d @ _swap_last2(Wcol)                     # (N*OH*OW, Cout)
    out2d = out2d + bias.data
    out4d = _contig(out2d.reshape(N, OH, OW, Cout))
    out4d = _contig(xp.transpose(out4d, (0, 3, 1, 2)))    # (N,Cout,OH,OW)

    out = Tensor(out4d, x.requires_grad or weight.requires_grad or bias.requires_grad,
                 (x, weight, bias), "conv2d")

    def _backward():
        d2d = _contig(xp.transpose(out.grad, (0, 2, 3, 1))).reshape(N * OH * OW, Cout)
        if weight.requires_grad:
            dWcol = _swap_last2(cols2d) @ d2d              # (CKK, Cout)
            weight._accum(_contig(_swap_last2(dWcol)).reshape(Cout, C, KH, KW))
        if bias.requires_grad:
            bias._accum(d2d.sum(axis=0))
        if x.requires_grad:
            dcols2d = d2d @ Wcol                           # (N*OH*OW, CKK)
            dcols = _contig(dcols2d.reshape(N, OH, OW, C, KH, KW))
            dcols = _contig(xp.transpose(dcols, (0, 3, 4, 5, 1, 2)))  # (N,C,KH,KW,OH,OW)
            dxpad = _col2im(dcols, N, C, xpad.shape[2], xpad.shape[3], KH, KW, s, OH, OW)
            dx = dxpad if (ph == 0 and pw == 0) else _contig(dxpad[:, :, ph:ph + H, pw:pw + W])
            x._accum(dx)
    out._backward = _backward
    return out


# ---- Conv1d / ConvTranspose1d (inference forward; used by TTS vocoder/flow) ---
def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """NCL conv. weight: (Cout, Cin/groups, K), bias: (Cout,) or None. Forward-only
    (no autograd) -- built for inference (HiFiGAN / WaveNet dilated convs)."""
    xd = x.data if isinstance(x, Tensor) else x
    wd = weight.data if isinstance(weight, Tensor) else weight
    N, C, L = xd.shape
    O, Cg, K = wd.shape
    if padding:
        xpad = xp.zeros((N, C, L + 2 * padding), xd.dtype)
        xpad[:, :, padding:padding + L] = xd
    else:
        xpad = xd
    Lp = xpad.shape[2]
    eff = (K - 1) * dilation + 1
    Lout = (Lp - eff) // stride + 1
    cols = xp.stack([xpad[:, :, k * dilation: k * dilation + stride * Lout: stride] for k in range(K)], axis=2)
    if groups == 1:
        cols2d = _contig(cols.transpose(0, 1, 2, 3).reshape(N, C * K, Lout))  # (N, C*K, Lout), C outer, K inner
        Wm = wd.reshape(O, C * K)
        out = xp.matmul(Wm, cols2d)                                           # (N,O,Lout)
    else:
        cg = C // groups; og = O // groups; outs = []
        for gi in range(groups):
            cc = _contig(cols[:, gi * cg:(gi + 1) * cg].reshape(N, cg * K, Lout))
            Wm = wd[gi * og:(gi + 1) * og].reshape(og, cg * K)
            outs.append(xp.matmul(Wm, cc))
        out = xp.concatenate(outs, axis=1)
    if bias is not None:
        bd = bias.data if isinstance(bias, Tensor) else bias
        out = out + bd.reshape(1, O, 1)
    return Tensor(_contig(out))


def conv_transpose1d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """NCL transposed conv. weight: (Cin, Cout, K) (torch layout). Forward-only."""
    xd = x.data if isinstance(x, Tensor) else x
    wd = weight.data if isinstance(weight, Tensor) else weight
    N, C, L = xd.shape
    Ci, O, K = wd.shape
    Lout = (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    full = xp.zeros((N, O, Lout + 2 * padding), xd.dtype)
    for k in range(K):
        Wk = wd[:, :, k]                                   # (C,O)
        t = xp.matmul(_contig(_swap_last2(Wk))[None], xd)  # (N,O,L)
        full[:, :, k * dilation: k * dilation + stride * L: stride] += t
    out = full[:, :, padding:padding + Lout] if padding else full
    if bias is not None:
        bd = bias.data if isinstance(bias, Tensor) else bias
        out = out + bd.reshape(1, O, 1)
    return Tensor(_contig(out))


def leaky_relu(x, slope=0.01):
    return x.relu() + (x - x.relu()) * slope


# ---- Conv3d (direct convolution, NCDHW) ------------------------------------
# Meta order: (N,Cin,Cout,D,H,W,OD,OH,OW,KD,KH,KW,stride,pad).
_C3_META_FMT = "u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4"
_C3_STRUCT = ("struct CMeta { N:u32,Cin:u32,Cout:u32,D:u32,H:u32,W:u32,OD:u32,OH:u32,OW:u32,"
              "KD:u32,KH:u32,KW:u32,stride:u32,pad:u32, }\n@group(0) @binding(BND) var<storage,read> c: CMeta;\n")
_C3_FWD_WGSL = """@group(0) @binding(0) var<storage,read> xin: array<f32>;
@group(0) @binding(1) var<storage,read> wt: array<f32>;
@group(0) @binding(2) var<storage,read> bs: array<f32>;
@group(0) @binding(3) var<storage,read_write> outp: array<f32>;
""" + _C3_STRUCT.replace("BND", "4") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.N*c.Cout*c.OD*c.OH*c.OW) { return; }
  let ow=i%c.OW; var t=i/c.OW; let oh=t%c.OH; t=t/c.OH; let od=t%c.OD; t=t/c.OD; let co=t%c.Cout; let n=t/c.Cout;
  var sum = bs[co];
  for (var ci:u32=0u; ci<c.Cin; ci++) { for (var kd:u32=0u; kd<c.KD; kd++) {
    let id = i32(od*c.stride+kd)-i32(c.pad); if (id<0||id>=i32(c.D)) { continue; }
    for (var kh:u32=0u; kh<c.KH; kh++) {
      let ih = i32(oh*c.stride+kh)-i32(c.pad); if (ih<0||ih>=i32(c.H)) { continue; }
      for (var kw:u32=0u; kw<c.KW; kw++) {
        let iw = i32(ow*c.stride+kw)-i32(c.pad); if (iw<0||iw>=i32(c.W)) { continue; }
        sum = sum + xin[(((n*c.Cin+ci)*c.D+u32(id))*c.H+u32(ih))*c.W+u32(iw)] * wt[(((co*c.Cin+ci)*c.KD+kd)*c.KH+kh)*c.KW+kw];
      }
    }
  } }
  outp[i] = sum;
}
"""
_C3_DIN_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read> wt: array<f32>;
@group(0) @binding(2) var<storage,read_write> dx: array<f32>;
""" + _C3_STRUCT.replace("BND", "3") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.N*c.Cin*c.D*c.H*c.W) { return; }
  let iw=i%c.W; var t=i/c.W; let ih=t%c.H; t=t/c.H; let id=t%c.D; t=t/c.D; let ci=t%c.Cin; let n=t/c.Cin;
  var sum: f32 = 0.0;
  for (var co:u32=0u; co<c.Cout; co++) { for (var kd:u32=0u; kd<c.KD; kd++) {
    let ad=i32(id)+i32(c.pad)-i32(kd); if (ad<0||(ad%i32(c.stride))!=0) { continue; } let od=ad/i32(c.stride); if (od>=i32(c.OD)) { continue; }
    for (var kh:u32=0u; kh<c.KH; kh++) {
      let ah=i32(ih)+i32(c.pad)-i32(kh); if (ah<0||(ah%i32(c.stride))!=0) { continue; } let oh=ah/i32(c.stride); if (oh>=i32(c.OH)) { continue; }
      for (var kw:u32=0u; kw<c.KW; kw++) {
        let aw=i32(iw)+i32(c.pad)-i32(kw); if (aw<0||(aw%i32(c.stride))!=0) { continue; } let ow=aw/i32(c.stride); if (ow>=i32(c.OW)) { continue; }
        sum = sum + dout[(((n*c.Cout+co)*c.OD+u32(od))*c.OH+u32(oh))*c.OW+u32(ow)] * wt[(((co*c.Cin+ci)*c.KD+kd)*c.KH+kh)*c.KW+kw];
      }
    }
  } }
  dx[i] = sum;
}
"""
_C3_DW_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read> xin: array<f32>;
@group(0) @binding(2) var<storage,read_write> dw: array<f32>;
""" + _C3_STRUCT.replace("BND", "3") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.Cout*c.Cin*c.KD*c.KH*c.KW) { return; }
  let kw=i%c.KW; var t=i/c.KW; let kh=t%c.KH; t=t/c.KH; let kd=t%c.KD; t=t/c.KD; let ci=t%c.Cin; let co=t/c.Cin;
  var sum: f32 = 0.0;
  for (var n:u32=0u; n<c.N; n++) { for (var od:u32=0u; od<c.OD; od++) {
    let id=i32(od*c.stride+kd)-i32(c.pad); if (id<0||id>=i32(c.D)) { continue; }
    for (var oh:u32=0u; oh<c.OH; oh++) {
      let ih=i32(oh*c.stride+kh)-i32(c.pad); if (ih<0||ih>=i32(c.H)) { continue; }
      for (var ow:u32=0u; ow<c.OW; ow++) {
        let iw=i32(ow*c.stride+kw)-i32(c.pad); if (iw<0||iw>=i32(c.W)) { continue; }
        sum = sum + dout[(((n*c.Cout+co)*c.OD+od)*c.OH+oh)*c.OW+ow] * xin[(((n*c.Cin+ci)*c.D+u32(id))*c.H+u32(ih))*c.W+u32(iw)];
      }
    }
  } }
  dw[i] = sum;
}
"""
_C3_DB_WGSL = """@group(0) @binding(0) var<storage,read> dout: array<f32>;
@group(0) @binding(1) var<storage,read_write> db: array<f32>;
""" + _C3_STRUCT.replace("BND", "2") + """@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let co = gid.x; if (co >= c.Cout) { return; }
  var sum: f32 = 0.0;
  for (var n:u32=0u; n<c.N; n++) { for (var od:u32=0u; od<c.OD; od++) { for (var oh:u32=0u; oh<c.OH; oh++) { for (var ow:u32=0u; ow<c.OW; ow++) {
    sum = sum + dout[(((n*c.Cout+co)*c.OD+od)*c.OH+oh)*c.OW+ow];
  } } } }
  db[co] = sum;
}
"""
_C3_GL_U = "uniform int _ka_tex_output_texture_w; uniform int N,Cin,Cout,D,H,W,OD,OH,OW,KD,KH,KW,stride,pad;"
_GL_C3_FWD = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n" + _C3_GL_U
    + "\nuniform sampler2D tex_x, tex_w, tex_b;\nout float fragColor;\nFETCH\nvoid main(){\n"
    + "int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=N*Cout*OD*OH*OW){fragColor=0.0;return;}\n"
    + "int ow=i%OW; int t=i/OW; int oh=t%OH; t/=OH; int od=t%OD; t/=OD; int co=t%Cout; int n=t/Cout;\n"
    + "float sum=fetch(tex_b,co);\n"
    + "for(int ci=0;ci<Cin;ci++)for(int kd=0;kd<KD;kd++){int id=od*stride+kd-pad; if(id<0||id>=D)continue;\n"
    + " for(int kh=0;kh<KH;kh++){int ih=oh*stride+kh-pad; if(ih<0||ih>=H)continue;\n"
    + "  for(int kw=0;kw<KW;kw++){int iw=ow*stride+kw-pad; if(iw<0||iw>=W)continue;\n"
    + "   sum+=fetch(tex_x,(((n*Cin+ci)*D+id)*H+ih)*W+iw)*fetch(tex_w,(((co*Cin+ci)*KD+kd)*KH+kh)*KW+kw);}}}\n"
    + "fragColor=sum;\n}").replace("FETCH", _GL_FETCH)
_GL_C3_DIN = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n" + _C3_GL_U
    + "\nuniform sampler2D tex_g, tex_w;\nout float fragColor;\nFETCH\nvoid main(){\n"
    + "int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=N*Cin*D*H*W){fragColor=0.0;return;}\n"
    + "int iw=i%W; int t=i/W; int ih=t%H; t/=H; int id=t%D; t/=D; int ci=t%Cin; int n=t/Cin;\n"
    + "float sum=0.0;\n"
    + "for(int co=0;co<Cout;co++)for(int kd=0;kd<KD;kd++){int ad=id+pad-kd; if(ad<0||ad%stride!=0)continue; int od=ad/stride; if(od>=OD)continue;\n"
    + " for(int kh=0;kh<KH;kh++){int ah=ih+pad-kh; if(ah<0||ah%stride!=0)continue; int oh=ah/stride; if(oh>=OH)continue;\n"
    + "  for(int kw=0;kw<KW;kw++){int aw=iw+pad-kw; if(aw<0||aw%stride!=0)continue; int ow=aw/stride; if(ow>=OW)continue;\n"
    + "   sum+=fetch(tex_g,(((n*Cout+co)*OD+od)*OH+oh)*OW+ow)*fetch(tex_w,(((co*Cin+ci)*KD+kd)*KH+kh)*KW+kw);}}}\n"
    + "fragColor=sum;\n}").replace("FETCH", _GL_FETCH)
_GL_C3_DW = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n" + _C3_GL_U
    + "\nuniform sampler2D tex_g, tex_x;\nout float fragColor;\nFETCH\nvoid main(){\n"
    + "int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=Cout*Cin*KD*KH*KW){fragColor=0.0;return;}\n"
    + "int kw=i%KW; int t=i/KW; int kh=t%KH; t/=KH; int kd=t%KD; t/=KD; int ci=t%Cin; int co=t/Cin;\n"
    + "float sum=0.0;\n"
    + "for(int n=0;n<N;n++)for(int od=0;od<OD;od++){int id=od*stride+kd-pad; if(id<0||id>=D)continue;\n"
    + " for(int oh=0;oh<OH;oh++){int ih=oh*stride+kh-pad; if(ih<0||ih>=H)continue;\n"
    + "  for(int ow=0;ow<OW;ow++){int iw=ow*stride+kw-pad; if(iw<0||iw>=W)continue;\n"
    + "   sum+=fetch(tex_g,(((n*Cout+co)*OD+od)*OH+oh)*OW+ow)*fetch(tex_x,(((n*Cin+ci)*D+id)*H+ih)*W+iw);}}}\n"
    + "fragColor=sum;\n}").replace("FETCH", _GL_FETCH)
_GL_C3_DB = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n" + _C3_GL_U
    + "\nuniform sampler2D tex_g;\nout float fragColor;\nFETCH\nvoid main(){\n"
    + "int co=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(co>=Cout){fragColor=0.0;return;}\n"
    + "float sum=0.0;\n"
    + "for(int n=0;n<N;n++)for(int od=0;od<OD;od++)for(int oh=0;oh<OH;oh++)for(int ow=0;ow<OW;ow++) sum+=fetch(tex_g,(((n*Cout+co)*OD+od)*OH+oh)*OW+ow);\n"
    + "fragColor=sum;\n}").replace("FETCH", _GL_FETCH)
_c3_k = {"added": False, "gl": False}


def _c3_uniforms(dims, out_w):
    u = [{"name": "_ka_tex_output_texture_w", "value": out_w, "type": "int"}]
    for nm, val in zip(["N", "Cin", "Cout", "D", "H", "W", "OD", "OH", "OW", "KD", "KH", "KW", "stride", "pad"], dims):
        u.append({"name": nm, "value": int(val), "type": "int"})
    return u


def conv3d(x, weight, bias, stride=1, padding=0):
    """NCDHW conv. weight: (Cout,Cin,KD,KH,KW), bias: (Cout,). Direct-convolution
    kernels on both backends (falls back to no-GPU error otherwise)."""
    N, Cin, D, H, W = x.data.shape
    Cout, _, KD, KH, KW = weight.data.shape
    s, p = stride, padding
    OD = (D + 2 * p - KD) // s + 1; OH = (H + 2 * p - KH) // s + 1; OW = (W + 2 * p - KW) // s + 1
    dims = (N, Cin, Cout, D, H, W, OD, OH, OW, KD, KH, KW, s, p)
    wgpu = _adam_backend_ready()
    assert wgpu or _webgl_ready(), "conv3d needs a GPU backend"
    if wgpu and not _c3_k["added"]:
        plat = _adam_kernel["platform"]
        r3, r2 = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"], ["read-only-storage", "storage", "read-only-storage"]
        plat.addKernel("c3_fwd", {"source": _C3_FWD_WGSL, "bindingTypes": ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
        plat.addKernel("c3_din", {"source": _C3_DIN_WGSL, "bindingTypes": r3})
        plat.addKernel("c3_dw", {"source": _C3_DW_WGSL, "bindingTypes": r3})
        plat.addKernel("c3_db", {"source": _C3_DB_WGSL, "bindingTypes": r2})
        _c3_k["added"] = True
    if (not wgpu) and not _c3_k["gl"]:
        plat = _copy_kernel["plat"]
        plat.addKernel("c3_fwd", {"source": _GL_C3_FWD}); plat.addKernel("c3_din", {"source": _GL_C3_DIN})
        plat.addKernel("c3_dw", {"source": _GL_C3_DW}); plat.addKernel("c3_db", {"source": _GL_C3_DB})
        _c3_k["gl"] = True

    def run(name, ins, out_buf, nthreads):
        if wgpu:
            meta = _adam_kernel["make_meta"](dims, _C3_META_FMT).buffer_id
            _adam_kernel["platform"].runKernel({"name": name, "tensors": [b for b in ins] + [out_buf.buffer.buffer_id, meta],
                "workGroups": {"x": (nthreads + 63) // 64, "y": 1, "z": 1}})
        else:
            names = {"c3_fwd": ["tex_x", "tex_w", "tex_b"], "c3_din": ["tex_g", "tex_w"], "c3_dw": ["tex_g", "tex_x"], "c3_db": ["tex_g"]}[name]
            _copy_kernel["plat"].runKernel({"name": name,
                "inputs": [{"name": nm, "id": bid} for nm, bid in zip(names, ins)],
                "output": out_buf.buffer.buffer_id, "uniforms": _c3_uniforms(dims, out_buf.buffer.texture_shape.width)})

    of = _empty((N, Cout, OD, OH, OW))
    run("c3_fwd", [x.data.buffer.buffer_id, weight.data.buffer.buffer_id, bias.data.buffer.buffer_id], of, N * Cout * OD * OH * OW)
    out = Tensor(of, x.requires_grad or weight.requires_grad or bias.requires_grad, (x, weight, bias), "conv3d")

    def _backward():
        g = _contig(out.grad)
        if x.requires_grad:
            dx = _empty((N, Cin, D, H, W)); run("c3_din", [g.buffer.buffer_id, weight.data.buffer.buffer_id], dx, N * Cin * D * H * W); x._accum(dx)
        if weight.requires_grad:
            dw = _empty((Cout, Cin, KD, KH, KW)); run("c3_dw", [g.buffer.buffer_id, x.data.buffer.buffer_id], dw, Cout * Cin * KD * KH * KW); weight._accum(dw)
        if bias.requires_grad:
            db = _empty((Cout,)); run("c3_db", [g.buffer.buffer_id], db, Cout); bias._accum(db)
    out._backward = _backward
    return out


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        KD = KH = KW = kernel_size
        fan_in = in_channels * KD * KH * KW
        w = np.random.randn(out_channels, in_channels, KD, KH, KW).astype(np.float32) * np.sqrt(2.0 / fan_in)
        self.weight = Parameter(w)
        self.bias = Parameter(np.zeros((out_channels,), dtype=np.float32))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return conv3d(x, self.weight, self.bias, self.stride, self.padding)


# ---- transformer ops ------------------------------------------------------
_bmm_native = [None]  # None=unknown, True=native 3D matmul (patched WebGPU), False=loop


def _bmm_raw(A, B):
    """Batched matmul on raw xp arrays: (Bs,M,K)@(Bs,K,N)->(Bs,M,N).
    Uses WgPy's native batched kernel (one dispatch) when available; else loops."""
    if _bmm_native[0] is not False:
        try:
            r = A @ B                 # patched WebGPU handles 3D@3D in one dispatch
            _bmm_native[0] = True
            return r
        except Exception:
            _bmm_native[0] = False
    # 2D-loop fallback (WebGL). Build the output by CONCATENATING per-batch 2D
    # matmuls rather than slice-assigning into a _zeros buffer: slice-assignment
    # is not graph-capture-safe on WebGL (writes a fresh texture each time), which
    # silently corrupts replayed training. concatenate is a normal kernel and
    # replays correctly.
    Bs, M, K = A.shape
    N = B.shape[2]
    parts = [(_contig(A[i]) @ _contig(B[i])).reshape(1, M, N) for i in range(Bs)]
    return xp.concatenate(parts, axis=0)


def _bt(x):
    # batched transpose of last two axes, materialized contiguous
    return _contig(xp.transpose(x, (0, 2, 1)))


def bmm(a, b):
    """Batched matmul (B,M,K)@(B,K,N)->(B,M,N)."""
    A, Bd = a.data, b.data
    out = Tensor(_bmm_raw(A, Bd), a.requires_grad or b.requires_grad, (a, b), "bmm")

    def _backward():
        g = out.grad
        if a.requires_grad:
            a._accum(_bmm_raw(g, _bt(Bd)))       # (B,M,N)@(B,N,K)
        if b.requires_grad:
            b._accum(_bmm_raw(_bt(A), g))        # (B,K,M)@(B,M,N)
    out._backward = _backward
    return out


def gqa_attention(q, k, v, mask=None, scale=None):
    """Grouped-query attention WITHOUT materializing the KV head expansion.

    q (nh, T, hd); k, v (nkv, S, hd); nh % nkv == 0.
    The naive path gathers k/v up to nh heads (a big fp32 copy -- measured ~1 GB/s
    and the single largest cost in decode). Instead regroup the *queries* by kv
    head: (nh, T, hd) -> (nkv, rep*T, hd), which makes it a plain batched matmul
    against the un-expanded k/v. Zero copies, autograd-clean.
    `mask` broadcasts over (T, S).
    """
    nh, T, hd = q.shape
    nkv, S, _ = k.shape
    rep = nh // nkv
    if scale is None:
        scale = 1.0 / (float(hd) ** 0.5)               # host-side Python float, not a WgPy tensor pow
    qg = q.reshape(nkv, rep * T, hd)
    a = bmm(qg, transpose_last2(k)) * scale            # (nkv, rep*T, S)
    if mask is not None:
        a = (a.reshape(nkv, rep, T, S) + mask).reshape(nkv, rep * T, S)
    a = softmax(a)
    o = bmm(a, v)                                      # (nkv, rep*T, hd)
    return o.reshape(nh, T, hd)                        # head = kv*rep + r


# In-place KV-cache scatter write. WgPy's `cache[:, pos, :] = kcur` (ndarray
# __setitem__) reads back the ENTIRE cache buffer to host, modifies, re-uploads
# -- a full GPU->CPU->GPU round-trip per write (measured: 72 round-trips/token
# dominate decode). This kernel writes the slot in place on the GPU with no
# readback, and takes `pos` from a meta buffer so ONE fixed dispatch works for
# every position -- which is also what makes the decode step graph-capturable.
_KVWRITE_WGSL = """@group(0) @binding(0) var<storage,read_write> cache: array<f32>;
@group(0) @binding(1) var<storage,read> src: array<f32>;
struct M { pos:u32, T:u32, NKV:u32, HD:u32, LMAX:u32, }
@group(0) @binding(2) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = m.NKV * m.T * m.HD;
  if (i >= total) { return; }
  let hd = i % m.HD;
  let t = (i / m.HD) % m.T;
  let kv = i / (m.HD * m.T);
  cache[kv * m.LMAX * m.HD + (m.pos + t) * m.HD + hd] = src[i];
}
"""
_kvw = {"added": False}


def kv_write(cache, src, pos, T, nkv, hd, lmax, ctl=None):
    """cache[:, pos:pos+T, :] = src, in place on GPU (no readback).
    cache: WgPy ndarray (NKV,LMAX,HD); src: (NKV,T,HD). `ctl`, if given, is a
    persistent int32 ndarray [pos,T,nkv,hd,lmax] (for capture) whose content is
    set outside the graph via ctl.buffer.set_data(...) each step."""
    plat = _adam_kernel["platform"]
    if not _kvw["added"]:
        plat.addKernel("kv_write", {"source": _KVWRITE_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "read-only-storage"]})
        _kvw["added"] = True
    if ctl is not None:
        meta_id = ctl.buffer.buffer_id                 # persistent ndarray meta (capture)
    else:
        meta_id = _adam_kernel["make_meta"](
            (int(pos), int(T), int(nkv), int(hd), int(lmax)), "u4,u4,u4,u4,u4").buffer_id
    total = nkv * T * hd
    plat.runKernel({"name": "kv_write",
        "tensors": [cache.buffer.buffer_id, src.buffer.buffer_id, meta_id],
        "workGroups": {"x": (total + 63) // 64, "y": 1, "z": 1}})


class KVCache:
    """Backend-appropriate KV cache for decode, hiding the WebGPU/WebGL split.

    WebGPU: fixed-capacity buffers written in place by the `kv_write` scatter
    kernel (no readback; fixed buffer ids -> graph-capturable). Attends over the
    full LMAX with a position mask.
    WebGL: a fragment shader cannot render into a texture it samples (no
    feedback), so an in-place scatter is impossible. Grow the cache with `cat`
    instead (reads old cache + new k, writes a NEW texture -> no feedback, no
    readback). Correct; not capture-safe (WebGL's documented limit).
    Both paths return identical attention outputs.
    """
    def __init__(self, n_layers, nkv, hd, lmax, scatter=False):
        # scatter=True: WebGPU fixed-capacity + in-place kv_write (capture-ready,
        # but has a non-capture batching regression -- only use under capture).
        # Default: growing cache via `cat` on BOTH backends (proven, no readback,
        # no hang). WebGL can ONLY grow (no in-place scatter -- texture feedback).
        self.gpu = _adam_backend_ready()
        self.scatter = scatter and self.gpu
        self.L = n_layers; self.nkv = nkv; self.hd = hd; self.lmax = lmax
        self._mkey = None; self._mask = None      # mask identical across layers for a (pos,T)
        if self.scatter:
            self.K = [Tensor(_zeros((nkv, lmax, hd))) for _ in range(n_layers)]
            self.V = [Tensor(_zeros((nkv, lmax, hd))) for _ in range(n_layers)]
        else:
            self.K = [None] * n_layers; self.V = [None] * n_layers

    def _gpu_mask(self, pos, T):
        if self._mkey != (pos, T):
            m = np.zeros((T, self.lmax), np.float32)
            for j in range(T):
                m[j, pos + j + 1:] = -1e9
            self._mask = Tensor(m.reshape(1, T, self.lmax)); self._mkey = (pos, T)
        return self._mask

    def attn(self, i, q, k, v, pos, scale=None):
        """Write k,v (nkv,T,hd) at `pos`, then attend q (nh,T,hd). Returns (nh,T,hd)."""
        T = k.shape[1]
        if self.scatter:
            kc, vc = self.K[i], self.V[i]
            kv_write(kc.data, _contig(k).data, pos, T, self.nkv, self.hd, self.lmax)
            kv_write(vc.data, _contig(v).data, pos, T, self.nkv, self.hd, self.lmax)
            return gqa_attention(q, kc, vc, self._gpu_mask(pos, T), scale)
        # growing cache via cat (both backends)
        if self.K[i] is None:
            self.K[i] = Tensor(_contig(k.data)); self.V[i] = Tensor(_contig(v.data))
        else:
            self.K[i] = cat([self.K[i], k], axis=1); self.V[i] = cat([self.V[i], v], axis=1)
        S = self.K[i].shape[1]
        mask = None
        if T > 1:                                     # prefill: causal, aligned to the end
            mm = np.triu(np.full((T, S), -1e9, np.float32), 1 + (S - T))
            mask = Tensor(mm.reshape(1, T, S))
        return gqa_attention(q, self.K[i], self.V[i], mask, scale)


def transpose_last2(x):
    """Autograd transpose of the last two axes (for K^T in attention)."""
    out = Tensor(_swap_last2(x.data), x.requires_grad, (x,), "T")

    def _backward():
        if x.requires_grad:
            x._accum(_swap_last2(out.grad))
    out._backward = _backward
    return out


_SOFTMAX_WGSL = """@group(0) @binding(0)
var<storage,read> inp: array<f32>;
@group(0) @binding(1)
var<storage,read_write> outp: array<f32>;
struct CMeta { rows: u32, width: u32, }
@group(0) @binding(2)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let base = row * cmeta.width;
  var mx: f32 = inp[base];
  for (var j: u32 = 1u; j < cmeta.width; j = j + 1u) {
    let v = inp[base + j];
    if (v > mx) { mx = v; }
  }
  var sm: f32 = 0.0;
  for (var j: u32 = 0u; j < cmeta.width; j = j + 1u) {
    sm = sm + exp(inp[base + j] - mx);
  }
  for (var j: u32 = 0u; j < cmeta.width; j = j + 1u) {
    outp[base + j] = exp(inp[base + j] - mx) / sm;
  }
}
"""
_softmax_kernel = {"added": False}


def _fused_softmax(xd):
    """One kernel does max/exp/sum/div per row over the last axis."""
    plat = _adam_kernel["platform"]
    if not _softmax_kernel["added"]:
        plat.addKernel("fused_softmax", {
            "source": _SOFTMAX_WGSL,
            "bindingTypes": ["read-only-storage", "storage", "read-only-storage"],
        })
        _softmax_kernel["added"] = True
    width = int(xd.shape[-1])
    rows = int(xd.size) // width
    s = _zeros(xd.shape)
    meta = _adam_kernel["make_meta"]((rows, width), "u4,u4")
    plat.runKernel({
        "name": "fused_softmax",
        "tensors": [xd.buffer.buffer_id, s.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1},
    })
    return s


_softmax_gl = {"added": set()}


def _webgl_softmax(xd):
    """Fused softmax over the last axis, one GLSL kernel (keyed by row width)."""
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1])
    rows = int(xd.size) // width
    name = f"softmax_gl_{width}"
    if name not in _softmax_gl["added"]:
        plat.addKernel(name, {"source": f"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
#define WIDTH {width}
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_in;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / WIDTH; int col = idx - row * WIDTH; int base = row * WIDTH;
  float mx = fetch(tex_in, base);
  for (int j = 1; j < WIDTH; j++) {{ float v = fetch(tex_in, base + j); if (v > mx) mx = v; }}
  float sm = 0.0;
  for (int j = 0; j < WIDTH; j++) {{ sm += exp(fetch(tex_in, base + j) - mx); }}
  fragColor = exp(fetch(tex_in, base + col) - mx) / sm;
}}
"""})
        _softmax_gl["added"].add(name)
    s = _zeros(xd.shape)
    plat.runKernel({"name": name,
        "inputs": [{"name": "tex_in", "id": xd.buffer.buffer_id}],
        "output": s.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": s.buffer.texture_shape.width, "type": "int"}]})
    return s


_SM_BWD_WGSL = """@group(0) @binding(0)
var<storage,read> s_buf: array<f32>;
@group(0) @binding(1)
var<storage,read> g_buf: array<f32>;
@group(0) @binding(2)
var<storage,read_write> dx: array<f32>;
struct CMeta { rows: u32, width: u32, }
@group(0) @binding(3)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let base = row * cmeta.width;
  var dot: f32 = 0.0;
  for (var j: u32 = 0u; j < cmeta.width; j = j + 1u) { dot = dot + s_buf[base + j] * g_buf[base + j]; }
  for (var j: u32 = 0u; j < cmeta.width; j = j + 1u) {
    dx[base + j] = s_buf[base + j] * (g_buf[base + j] - dot);
  }
}
"""
_sm_bwd = {"added": False, "gl": set()}


def _softmax_bwd(s, g):
    """Fused softmax backward: dx = s * (g - sum(s*g, last)). One kernel."""
    width = int(s.shape[-1]); rows = int(s.size) // width
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if not _sm_bwd["added"]:
            plat.addKernel("sm_bwd", {"source": _SM_BWD_WGSL,
                "bindingTypes": ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
            _sm_bwd["added"] = True
        dx = _empty(s.shape)
        meta = _adam_kernel["make_meta"]((rows, width), "u4,u4")
        plat.runKernel({"name": "sm_bwd",
            "tensors": [s.buffer.buffer_id, g.buffer.buffer_id, dx.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
        return dx
    # WebGL: fragment shader keyed by width
    plat = _copy_kernel["plat"]
    name = f"sm_bwd_{width}"
    if name not in _sm_bwd["gl"]:
        plat.addKernel(name, {"source": f"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
#define WIDTH {width}
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_s; uniform sampler2D tex_g;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / WIDTH; int col = idx - row * WIDTH; int base = row * WIDTH;
  float dot = 0.0;
  for (int j = 0; j < WIDTH; j++) {{ dot += fetch(tex_s, base + j) * fetch(tex_g, base + j); }}
  fragColor = fetch(tex_s, base + col) * (fetch(tex_g, base + col) - dot);
}}
"""})
        _sm_bwd["gl"].add(name)
    dx = _empty(s.shape)
    plat.runKernel({"name": name,
        "inputs": [{"name": "tex_s", "id": s.buffer.buffer_id}, {"name": "tex_g", "id": g.buffer.buffer_id}],
        "output": dx.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": dx.buffer.texture_shape.width, "type": "int"}]})
    return dx


def softmax(x):
    """Softmax over the last axis."""
    xd = x.data
    fused = _adam_backend_ready() or _webgl_ready()
    if _adam_backend_ready():
        s = _fused_softmax(xd)      # WebGPU fused kernel
    elif _webgl_ready():
        s = _webgl_softmax(xd)      # WebGL fused kernel
    else:
        m = xd.max(axis=-1, keepdims=True)
        e = xp.exp(xd - m)
        s = e / e.sum(axis=-1, keepdims=True)
    out = Tensor(s, x.requires_grad, (x,), "softmax")

    def _backward():
        if x.requires_grad:
            if fused:
                x._accum(_softmax_bwd(s, out.grad))
            else:
                g = out.grad
                dot = (g * s).sum(axis=-1, keepdims=True)
                x._accum(s * (g - dot))
    out._backward = _backward
    return out


# ---- fused layernorm --------------------------------------------------------
# Row stats (mu/var) are recomputed inside each kernel instead of being staged
# through intermediate buffers: on WebGL the bottleneck is DRAW COUNT, so extra
# ALU inside one draw beats extra draws.
_LN_FWD_WGSL = """@group(0) @binding(0)
var<storage,read> xin: array<f32>;
@group(0) @binding(1)
var<storage,read> gam: array<f32>;
@group(0) @binding(2)
var<storage,read> bet: array<f32>;
@group(0) @binding(3)
var<storage,read_write> outp: array<f32>;
struct CMeta { rows: u32, width: u32, eps: f32, }
@group(0) @binding(4)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let W = cmeta.width;
  let base = row * W;
  var mu: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j = j + 1u) { mu = mu + xin[base + j]; }
  mu = mu / f32(W);
  var vr: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j = j + 1u) { let d = xin[base + j] - mu; vr = vr + d * d; }
  let inv = 1.0 / sqrt(vr / f32(W) + cmeta.eps);
  for (var j: u32 = 0u; j < W; j = j + 1u) {
    outp[base + j] = (xin[base + j] - mu) * inv * gam[j] + bet[j];
  }
}
"""
_LN_DX_WGSL = """@group(0) @binding(0)
var<storage,read> xin: array<f32>;
@group(0) @binding(1)
var<storage,read> gout: array<f32>;
@group(0) @binding(2)
var<storage,read> gam: array<f32>;
@group(0) @binding(3)
var<storage,read_write> dx: array<f32>;
struct CMeta { rows: u32, width: u32, eps: f32, }
@group(0) @binding(4)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let W = cmeta.width;
  let base = row * W;
  var mu: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j = j + 1u) { mu = mu + xin[base + j]; }
  mu = mu / f32(W);
  var vr: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j = j + 1u) { let d = xin[base + j] - mu; vr = vr + d * d; }
  let inv = 1.0 / sqrt(vr / f32(W) + cmeta.eps);
  var s1: f32 = 0.0;
  var s2: f32 = 0.0;
  for (var j: u32 = 0u; j < W; j = j + 1u) {
    let gx = gout[base + j] * gam[j];
    s1 = s1 + gx;
    s2 = s2 + gx * (xin[base + j] - mu) * inv;
  }
  for (var j: u32 = 0u; j < W; j = j + 1u) {
    let xh = (xin[base + j] - mu) * inv;
    dx[base + j] = inv * (gout[base + j] * gam[j] - s1 / f32(W) - xh * s2 / f32(W));
  }
}
"""
_LN_DGB_WGSL = """@group(0) @binding(0)
var<storage,read> xin: array<f32>;
@group(0) @binding(1)
var<storage,read> gout: array<f32>;
@group(0) @binding(2)
var<storage,read_write> dgam: array<f32>;
@group(0) @binding(3)
var<storage,read_write> dbet: array<f32>;
struct CMeta { rows: u32, width: u32, eps: f32, }
@group(0) @binding(4)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= cmeta.width) { return; }
  let W = cmeta.width;
  var dg: f32 = 0.0;
  var db: f32 = 0.0;
  for (var r: u32 = 0u; r < cmeta.rows; r = r + 1u) {
    let base = r * W;
    var mu: f32 = 0.0;
    for (var j: u32 = 0u; j < W; j = j + 1u) { mu = mu + xin[base + j]; }
    mu = mu / f32(W);
    var vr: f32 = 0.0;
    for (var j: u32 = 0u; j < W; j = j + 1u) { let dd = xin[base + j] - mu; vr = vr + dd * dd; }
    let inv = 1.0 / sqrt(vr / f32(W) + cmeta.eps);
    let g = gout[base + d];
    dg = dg + g * (xin[base + d] - mu) * inv;
    db = db + g;
  }
  dgam[d] = dg;
  dbet[d] = db;
}
"""
_ln_wgpu = {"added": False}


def _wgpu_ln_meta(rows, width, eps):
    return _adam_kernel["make_meta"]((rows, width, eps), "u4,u4,f4")


def _wgpu_ln_fwd(xd, gd, bd, eps):
    plat = _adam_kernel["platform"]
    if not _ln_wgpu["added"]:
        rw = ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]
        plat.addKernel("ln_fwd", {"source": _LN_FWD_WGSL, "bindingTypes": rw})
        plat.addKernel("ln_dx", {"source": _LN_DX_WGSL, "bindingTypes": rw})
        plat.addKernel("ln_dgb", {"source": _LN_DGB_WGSL, "bindingTypes":
                                  ["read-only-storage", "read-only-storage", "storage", "storage", "read-only-storage"]})
        _ln_wgpu["added"] = True
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    out = _zeros(xd.shape)
    plat.runKernel({"name": "ln_fwd",
        "tensors": [xd.buffer.buffer_id, gd.buffer.buffer_id, bd.buffer.buffer_id,
                    out.buffer.buffer_id, _wgpu_ln_meta(rows, width, eps).buffer_id],
        "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
    return out


def _wgpu_ln_bwd(xd, g, gd, eps):
    plat = _adam_kernel["platform"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    dx = _zeros(xd.shape); dgam = _zeros((width,)); dbet = _zeros((width,))
    meta = _wgpu_ln_meta(rows, width, eps)
    plat.runKernel({"name": "ln_dx",
        "tensors": [xd.buffer.buffer_id, g.buffer.buffer_id, gd.buffer.buffer_id,
                    dx.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
    plat.runKernel({"name": "ln_dgb",
        "tensors": [xd.buffer.buffer_id, g.buffer.buffer_id, dgam.buffer.buffer_id,
                    dbet.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (width + 63) // 64, "y": 1, "z": 1}})
    return dx, dgam, dbet


_ln_gl = {"added": set()}


def _gl_ln_stats(width):
    return f"""
  float mu = 0.0;
  for (int j = 0; j < {width}; j++) {{ mu += fetch(tex_x, base + j); }}
  mu /= float({width});
  float vr = 0.0;
  for (int j = 0; j < {width}; j++) {{ float d2 = fetch(tex_x, base + j) - mu; vr += d2 * d2; }}
  float inv = 1.0 / sqrt(vr / float({width}) + EPS);
"""


def _webgl_ln_kernels(rows, width):
    plat = _copy_kernel["plat"]
    key = (rows, width)
    if key in _ln_gl["added"]:
        return
    head = ("#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n"
            "uniform int _ka_tex_output_texture_w; uniform float EPS;\n")
    plat.addKernel(f"ln_fwd_{width}", {"source": f"""{head}
uniform sampler2D tex_x; uniform sampler2D tex_gamma; uniform sampler2D tex_beta;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / {width}; int col = idx - row * {width}; int base = row * {width};
{_gl_ln_stats(width)}
  fragColor = (fetch(tex_x, base + col) - mu) * inv * fetch(tex_gamma, col) + fetch(tex_beta, col);
}}
"""})
    plat.addKernel(f"ln_dx_{width}", {"source": f"""{head}
uniform sampler2D tex_x; uniform sampler2D tex_g; uniform sampler2D tex_gamma;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / {width}; int col = idx - row * {width}; int base = row * {width};
{_gl_ln_stats(width)}
  float s1 = 0.0; float s2 = 0.0;
  for (int j = 0; j < {width}; j++) {{
    float gx = fetch(tex_g, base + j) * fetch(tex_gamma, j);
    s1 += gx;
    s2 += gx * (fetch(tex_x, base + j) - mu) * inv;
  }}
  float xh = (fetch(tex_x, base + col) - mu) * inv;
  fragColor = inv * (fetch(tex_g, base + col) * fetch(tex_gamma, col) - s1 / float({width}) - xh * s2 / float({width}));
}}
"""})
    plat.addKernel(f"ln_dgamma_{rows}_{width}", {"source": f"""{head}
uniform sampler2D tex_x; uniform sampler2D tex_g;
out float fragColor;
{_GL_FETCH}
void main() {{
  int d = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (d >= {width}) {{ return; }}
  float dg = 0.0;
  for (int r = 0; r < {rows}; r++) {{
    int base = r * {width};
{_gl_ln_stats(width)}
    dg += fetch(tex_g, base + d) * (fetch(tex_x, base + d) - mu) * inv;
  }}
  fragColor = dg;
}}
"""})
    plat.addKernel(f"ln_dbeta_{rows}_{width}", {"source": f"""{head}
uniform sampler2D tex_g;
out float fragColor;
{_GL_FETCH}
void main() {{
  int d = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (d >= {width}) {{ return; }}
  float db = 0.0;
  for (int r = 0; r < {rows}; r++) {{ db += fetch(tex_g, r * {width} + d); }}
  fragColor = db;
}}
"""})
    _ln_gl["added"].add(key)


def _webgl_ln_fwd(xd, gd, bd, eps):
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    _webgl_ln_kernels(rows, width)
    out = _zeros(xd.shape)
    plat.runKernel({"name": f"ln_fwd_{width}",
        "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id},
                   {"name": "tex_gamma", "id": gd.buffer.buffer_id},
                   {"name": "tex_beta", "id": bd.buffer.buffer_id}],
        "output": out.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": out.buffer.texture_shape.width, "type": "int"},
                     {"name": "EPS", "value": eps, "type": "float"}]})
    return out


def _webgl_ln_bwd(xd, g, gd, eps):
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    _webgl_ln_kernels(rows, width)
    dx = _zeros(xd.shape); dgam = _zeros((width,)); dbet = _zeros((width,))
    W = lambda a: a.buffer.texture_shape.width
    plat.runKernel({"name": f"ln_dx_{width}",
        "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id},
                   {"name": "tex_g", "id": g.buffer.buffer_id},
                   {"name": "tex_gamma", "id": gd.buffer.buffer_id}],
        "output": dx.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(dx), "type": "int"},
                     {"name": "EPS", "value": eps, "type": "float"}]})
    plat.runKernel({"name": f"ln_dgamma_{rows}_{width}",
        "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id},
                   {"name": "tex_g", "id": g.buffer.buffer_id}],
        "output": dgam.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(dgam), "type": "int"},
                     {"name": "EPS", "value": eps, "type": "float"}]})
    plat.runKernel({"name": f"ln_dbeta_{rows}_{width}",
        "inputs": [{"name": "tex_g", "id": g.buffer.buffer_id}],
        "output": dbet.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(dbet), "type": "int"}]})
    return dx, dgam, dbet


def layernorm(x, gamma, beta, eps=1e-5):
    """LayerNorm over the last axis. gamma/beta: (D,). Fused on both backends:
    WebGPU 1 fwd + 2 bwd dispatches; WebGL 1 fwd + 3 bwd draws (one output per
    draw). Fallback: plain xp ops."""
    xd = x.data
    D = xd.shape[-1]
    # Fused LN is a WIN only on WebGL (draw-count-bound). On WebGPU, dispatches
    # are batched into one submit (overhead ~0) and the fused per-row/per-column
    # loop kernels have WORSE parallelism than the elementwise ops — measured
    # 0.64ms -> 1.74ms/step, a pessimization. So: fuse on WebGL only.
    fused_gpu = False
    fused_gl = (not _adam_backend_ready()) and _webgl_ready()
    if fused_gpu:
        od = _wgpu_ln_fwd(xd, gamma.data, beta.data, eps)
    elif fused_gl:
        od = _webgl_ln_fwd(xd, gamma.data, beta.data, eps)
    else:
        mu = xd.sum(axis=-1, keepdims=True) * (1.0 / D)
        xc = xd - mu
        var = (xc * xc).sum(axis=-1, keepdims=True) * (1.0 / D)
        inv = 1.0 / xp.sqrt(var + eps)
        xhat = xc * inv
        od = xhat * gamma.data + beta.data
    out = Tensor(od,
                 x.requires_grad or gamma.requires_grad or beta.requires_grad,
                 (x, gamma, beta), "layernorm")

    def _backward():
        g = out.grad
        if fused_gpu or fused_gl:
            bwd = _wgpu_ln_bwd if fused_gpu else _webgl_ln_bwd
            dx, dgam, dbet = bwd(xd, g, gamma.data, eps)
            if gamma.requires_grad:
                gamma._accum(dgam)
            if beta.requires_grad:
                beta._accum(dbet)
            if x.requires_grad:
                x._accum(dx)
        else:
            if gamma.requires_grad:
                gamma._accum((g * xhat).reshape(-1, D).sum(axis=0))
            if beta.requires_grad:
                beta._accum(g.reshape(-1, D).sum(axis=0))
            if x.requires_grad:
                gxhat = g * gamma.data
                s1 = gxhat.sum(axis=-1, keepdims=True)
                s2 = (gxhat * xhat).sum(axis=-1, keepdims=True)
                x._accum(inv * (gxhat - s1 * (1.0 / D) - xhat * s2 * (1.0 / D)))
    out._backward = _backward
    return out


_EMB_FWD_WGSL = """@group(0) @binding(0)
var<storage,read> w: array<f32>;
@group(0) @binding(1)
var<storage,read> idx: array<f32>;
@group(0) @binding(2)
var<storage,read_write> outp: array<f32>;
struct CMeta { M: u32, dim: u32, vocab: u32, }
@group(0) @binding(3)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cmeta.M * cmeta.dim) { return; }
  let m = i / cmeta.dim;
  let d = i - m * cmeta.dim;
  let row = u32(idx[m]);
  outp[i] = w[row * cmeta.dim + d];
}
"""
# backward scatter: one thread per (vocab-row v, d); loop the M tokens and sum g
# where idx[m]==v. No atomics (WebGL-safe); O(vocab*M*dim) but no one-hot buffer.
_EMB_BWD_WGSL = """@group(0) @binding(0)
var<storage,read> gout: array<f32>;
@group(0) @binding(1)
var<storage,read> idx: array<f32>;
@group(0) @binding(2)
var<storage,read_write> dw: array<f32>;
struct CMeta { M: u32, dim: u32, vocab: u32, }
@group(0) @binding(3)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cmeta.vocab * cmeta.dim) { return; }
  let v = i / cmeta.dim;
  let d = i - v * cmeta.dim;
  var acc: f32 = 0.0;
  for (var m: u32 = 0u; m < cmeta.M; m = m + 1u) {
    if (u32(idx[m]) == v) { acc = acc + gout[m * cmeta.dim + d]; }
  }
  dw[i] = acc;
}
"""
_emb_k = {"added": False, "gl": False}
_GL_EMB_FWD = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_w; uniform sampler2D tex_i; uniform int DIM;
out float fragColor;
FETCH
void main() {
  int i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int m = i / DIM; int d = i - m * DIM;
  int row = int(fetch(tex_i, m) + 0.5);
  fragColor = fetch(tex_w, row * DIM + d);
}
""".replace("FETCH", _GL_FETCH)
_GL_EMB_BWD = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_g; uniform sampler2D tex_i;
uniform int DIM; uniform int M;
out float fragColor;
FETCH
void main() {
  int i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int v = i / DIM; int d = i - v * DIM;
  float acc = 0.0;
  for (int m = 0; m < M; m++) {
    if (int(fetch(tex_i, m) + 0.5) == v) { acc += fetch(tex_g, m * DIM + d); }
  }
  fragColor = acc;
}
""".replace("FETCH", _GL_FETCH)


def _emb_kernels():
    if _adam_backend_ready():
        if not _emb_k["added"]:
            plat = _adam_kernel["platform"]
            b = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
            plat.addKernel("emb_fwd", {"source": _EMB_FWD_WGSL, "bindingTypes": b})
            plat.addKernel("emb_bwd", {"source": _EMB_BWD_WGSL, "bindingTypes": b})
            _emb_k["added"] = True
        return "wgpu"
    if not _emb_k["gl"]:
        plat = _copy_kernel["plat"]
        plat.addKernel("emb_fwd", {"source": _GL_EMB_FWD})
        plat.addKernel("emb_bwd", {"source": _GL_EMB_BWD})
        _emb_k["gl"] = True
    return "gl"


def embedding(weight, idx):
    """Gather rows of `weight` (vocab, dim) by integer `idx`. Fused GPU gather
    (O(M*dim)) on both backends; backward is a non-atomic scatter kernel (one
    thread per (vocab-row, dim), loops the M tokens). `idx` uploaded as f32 once
    (pinned under capture)."""
    ish = tuple(np.asarray(idx).shape)
    flat = np.asarray(idx).reshape(-1).astype(np.float32)
    vocab, dim = weight.data.shape
    M = flat.shape[0]
    if not (_adam_backend_ready() or _webgl_ready()):
        onehot = xp.asarray(np.eye(vocab, dtype=np.float32)[flat.astype(np.int64)])
        out = Tensor((onehot @ weight.data).reshape(*(ish + (dim,))), weight.requires_grad, (weight,), "embedding")

        def _bw():
            if weight.requires_grad:
                weight._accum(_swap_last2(onehot) @ _contig(out.grad.reshape(-1, dim)))
        out._backward = _bw
        return out

    mode = _emb_kernels()
    gidx = xp.asarray(flat)
    of = _empty((M, dim))
    if mode == "wgpu":
        plat = _adam_kernel["platform"]
        meta = _adam_kernel["make_meta"]((M, dim, vocab), "u4,u4,u4")
        plat.runKernel({"name": "emb_fwd",
            "tensors": [weight.data.buffer.buffer_id, gidx.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (M * dim + 63) // 64, "y": 1, "z": 1}})
    else:
        plat = _copy_kernel["plat"]
        plat.runKernel({"name": "emb_fwd",
            "inputs": [{"name": "tex_w", "id": weight.data.buffer.buffer_id}, {"name": "tex_i", "id": gidx.buffer.buffer_id}],
            "output": of.buffer.buffer_id,
            "uniforms": [{"name": "_ka_tex_output_texture_w", "value": of.buffer.texture_shape.width, "type": "int"},
                         {"name": "DIM", "value": dim, "type": "int"}]})
    out = Tensor(of.reshape(*(ish + (dim,))), weight.requires_grad, (weight,), "embedding")

    def _backward():
        if weight.requires_grad:
            g = _contig(out.grad.reshape(M, dim))
            dw = _empty((vocab, dim))
            if mode == "wgpu":
                plat = _adam_kernel["platform"]
                meta = _adam_kernel["make_meta"]((M, dim, vocab), "u4,u4,u4")
                plat.runKernel({"name": "emb_bwd",
                    "tensors": [g.buffer.buffer_id, gidx.buffer.buffer_id, dw.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (vocab * dim + 63) // 64, "y": 1, "z": 1}})
            else:
                plat = _copy_kernel["plat"]
                plat.runKernel({"name": "emb_bwd",
                    "inputs": [{"name": "tex_g", "id": g.buffer.buffer_id}, {"name": "tex_i", "id": gidx.buffer.buffer_id}],
                    "output": dw.buffer.buffer_id,
                    "uniforms": [{"name": "_ka_tex_output_texture_w", "value": dw.buffer.texture_shape.width, "type": "int"},
                                 {"name": "DIM", "value": dim, "type": "int"}, {"name": "M", "value": M, "type": "int"}]})
            weight._accum(dw)
    out._backward = _backward
    return out


_CE_FWD_WGSL = """@group(0) @binding(0)
var<storage,read> s_buf: array<f32>;
@group(0) @binding(1)
var<storage,read> tgt: array<f32>;
@group(0) @binding(2)
var<storage,read_write> outp: array<f32>;
struct CMeta { rows: u32, width: u32, }
@group(0) @binding(3)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let t = u32(tgt[row]);
  outp[row] = -log(s_buf[row * cmeta.width + t] + 1e-12);
}
"""
_CE_BWD_WGSL = """@group(0) @binding(0)
var<storage,read> s_buf: array<f32>;
@group(0) @binding(1)
var<storage,read> tgt: array<f32>;
@group(0) @binding(2)
var<storage,read_write> dl: array<f32>;
struct CMeta { rows: u32, width: u32, invn: f32, }
@group(0) @binding(3)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cmeta.rows * cmeta.width) { return; }
  let row = i / cmeta.width;
  let col = i - row * cmeta.width;
  var oh: f32 = 0.0;
  if (col == u32(tgt[row])) { oh = 1.0; }
  dl[i] = (s_buf[i] - oh) * cmeta.invn;
}
"""
_ce_k = {"added": False, "gl": False}
_GL_CE_FWD = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_s; uniform sampler2D tex_t; uniform int CLS;
out float fragColor;
FETCH
void main() {
  int row = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int t = int(fetch(tex_t, row) + 0.5);
  fragColor = -log(fetch(tex_s, row * CLS + t) + 1e-12);
}
""".replace("FETCH", _GL_FETCH)
_GL_CE_BWD = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_s; uniform sampler2D tex_t;
uniform int CLS; uniform float INVN;
out float fragColor;
FETCH
void main() {
  int i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = i / CLS; int col = i - row * CLS;
  float oh = (col == int(fetch(tex_t, row) + 0.5)) ? 1.0 : 0.0;
  fragColor = (fetch(tex_s, i) - oh) * INVN;
}
""".replace("FETCH", _GL_FETCH)


def _ce_fwd(s, tgt, N, Cls):
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if not _ce_k["added"]:
            b3 = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
            plat.addKernel("ce_fwd", {"source": _CE_FWD_WGSL, "bindingTypes": b3})
            plat.addKernel("ce_bwd", {"source": _CE_BWD_WGSL, "bindingTypes": b3})
            _ce_k["added"] = True
        out = _empty((N,))
        meta = _adam_kernel["make_meta"]((N, Cls), "u4,u4")
        plat.runKernel({"name": "ce_fwd",
            "tensors": [s.buffer.buffer_id, tgt.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (N + 63) // 64, "y": 1, "z": 1}})
        return out
    plat = _copy_kernel["plat"]
    if not _ce_k["gl"]:
        plat.addKernel("ce_fwd", {"source": _GL_CE_FWD})
        plat.addKernel("ce_bwd", {"source": _GL_CE_BWD})
        _ce_k["gl"] = True
    out = _empty((N,))
    plat.runKernel({"name": "ce_fwd",
        "inputs": [{"name": "tex_s", "id": s.buffer.buffer_id}, {"name": "tex_t", "id": tgt.buffer.buffer_id}],
        "output": out.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": out.buffer.texture_shape.width, "type": "int"},
                     {"name": "CLS", "value": Cls, "type": "int"}]})
    return out


def _ce_bwd(s, tgt, N, Cls):
    invn = 1.0 / N
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        dl = _empty((N, Cls))
        meta = _adam_kernel["make_meta"]((N, Cls, invn), "u4,u4,f4")
        plat.runKernel({"name": "ce_bwd",
            "tensors": [s.buffer.buffer_id, tgt.buffer.buffer_id, dl.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (N * Cls + 63) // 64, "y": 1, "z": 1}})
        return dl
    plat = _copy_kernel["plat"]
    dl = _empty((N, Cls))
    plat.runKernel({"name": "ce_bwd",
        "inputs": [{"name": "tex_s", "id": s.buffer.buffer_id}, {"name": "tex_t", "id": tgt.buffer.buffer_id}],
        "output": dl.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": dl.buffer.texture_shape.width, "type": "int"},
                     {"name": "CLS", "value": Cls, "type": "int"}, {"name": "INVN", "value": invn, "type": "float"}]})
    return dl


def cross_entropy(logits, targets):
    """Softmax cross-entropy. logits: (N, Cls); targets: numpy int (N,).
    Fused: softmax kernel + per-row NLL kernel + per-element grad kernel. No CPU
    one-hot (the target comparison is done inside the kernel)."""
    xd = logits.data
    N, Cls = xd.shape
    if not (_adam_backend_ready() or _webgl_ready()):
        m = xd.max(axis=-1, keepdims=True)
        e = xp.exp(xd - m)
        s = e / e.sum(axis=-1, keepdims=True)
        onehot = xp.asarray(np.eye(Cls, dtype=np.float32)[np.asarray(targets).astype(np.int64)])
        loss_val = (-(onehot * xp.log(s + 1e-12)).sum()) * (1.0 / N)
        out = Tensor(loss_val.reshape(()), logits.requires_grad, (logits,), "cross_entropy")

        def _bw():
            if logits.requires_grad:
                logits._accum((s - onehot) * (1.0 / N) * out.grad)
        out._backward = _bw
        return out

    tgt = xp.asarray(np.asarray(targets).astype(np.float32))
    s = _fused_softmax(xd) if _adam_backend_ready() else _webgl_softmax(xd)
    perrow = _ce_fwd(s, tgt, N, Cls)
    loss_val = perrow.sum() * (1.0 / N)
    out = Tensor(loss_val.reshape(()), logits.requires_grad, (logits,), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            logits._accum(_ce_bwd(s, tgt, N, Cls) * out.grad)
    out._backward = _backward
    return out


# ---- optim ----------------------------------------------------------------
def clip_grad_norm_(params, max_norm):
    """Clip gradients of `params` in place so their global L2 norm <= max_norm.
    Returns the total norm before clipping (a python float)."""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0
    total = 0.0
    for p in params:
        total += float((p.grad * p.grad).sum())
    total = total ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            p.grad = p.grad * scale
    return total


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.buf = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0.0:
                g = g + self.wd * p.data
            if self.momentum != 0.0:
                if self.buf[i] is None:
                    self.buf[i] = g * 1.0
                else:
                    self.buf[i] = self.momentum * self.buf[i] + g
                g = (g + self.momentum * self.buf[i]) if self.nesterov else self.buf[i]
            p.data = p.data - self.lr * g

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# Fused Adam update as a single WGSL kernel (one dispatch per param tensor)
# instead of ~13 cupy ops. Pyodide Python per-op overhead dominates, so cutting
# op count is the win. WebGPU only; falls back to the slow path elsewhere.
_ADAM_WGSL = """@group(0) @binding(0)
var<storage,read_write> param: array<f32>;
@group(0) @binding(1)
var<storage,read> grad: array<f32>;
@group(0) @binding(2)
var<storage,read_write> m_buf: array<f32>;
@group(0) @binding(3)
var<storage,read_write> v_buf: array<f32>;
struct CMeta { N: u32, lr: f32, b1: f32, b2: f32, eps: f32, bc1: f32, bc2: f32, wd: f32, }
@group(0) @binding(4)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cmeta.N) { return; }
  let g = grad[i];
  let mi = cmeta.b1 * m_buf[i] + (1.0 - cmeta.b1) * g;
  let vi = cmeta.b2 * v_buf[i] + (1.0 - cmeta.b2) * g * g;
  m_buf[i] = mi;
  v_buf[i] = vi;
  let mhat = mi / cmeta.bc1;
  let vhat = vi / cmeta.bc2;
  param[i] = param[i] - cmeta.lr * (mhat / (sqrt(vhat) + cmeta.eps) + cmeta.wd * param[i]);
}
"""

_adam_kernel = {"added": False, "platform": None, "make_meta": None}


def _adam_backend_ready():
    if _adam_kernel["platform"] is not None:
        return True
    if not GPU:
        return False
    try:
        if cp.get_backend_name() != "webgpu":
            return False
        from wgpy_backends.webgpu.platform import get_platform
        from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
        _adam_kernel["platform"] = get_platform()
        _adam_kernel["make_meta"] = create_meta_buffer_from_structure
        return True
    except Exception:
        return False


def _fused_adam(param, grad, m, v, lr, b1, b2, eps, bc1, bc2, wd=0.0):
    plat = _adam_kernel["platform"]
    if not _adam_kernel["added"]:
        plat.addKernel("fused_adam", {
            "source": _ADAM_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "storage", "storage", "read-only-storage"],
        })
        _adam_kernel["added"] = True
    N = int(param.size)
    meta = _adam_kernel["make_meta"]((N, lr, b1, b2, eps, bc1, bc2, wd), "u4,f4,f4,f4,f4,f4,f4,f4")
    plat.runKernel({
        "name": "fused_adam",
        "tensors": [param.buffer.buffer_id, grad.buffer.buffer_id,
                    m.buffer.buffer_id, v.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (N + 63) // 64, "y": 1, "z": 1},
    })


# WebGL in-place writeback: copy `src` into `dst`'s texture via a passthrough
# fragment shader. src != dst textures, so no feedback -> allowed, and it's a
# normal kernel -> graph-capture-safe. This lets WebGL's optimizer update
# persistent m/v/param buffers INSIDE the captured graph (WebGL setitem can't:
# it reallocates the texture, which breaks replay).
_copy_kernel = {"added": False, "plat": None}


def _webgl_ready():
    if _copy_kernel["plat"] is not None:
        return True
    try:
        if not GPU or cp.get_backend_name() != "webgl":
            return False
        from wgpy_backends.webgl.platform import get_platform
        _copy_kernel["plat"] = get_platform()
        return True
    except Exception:
        return False


def _webgl_copy_into(dst, src):
    """dst[:] = src, writing dst's existing texture in place (capture-safe)."""
    plat = _copy_kernel["plat"]
    if not _copy_kernel["added"]:
        plat.addKernel("copy_passthrough", {"source": """#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int _ka_tex_output_texture_w;
uniform sampler2D tex_src;
out float fragColor;
void main() {
    int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
    int tw = textureSize(tex_src, 0).x;
    int y = idx / tw;
    int x = idx - y * tw;
    fragColor = texelFetch(tex_src, ivec2(x, y), 0).r;
}
"""})
        _copy_kernel["added"] = True
    plat.runKernel({
        "name": "copy_passthrough",
        "inputs": [{"name": "tex_src", "id": src.buffer.buffer_id}],
        "output": dst.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w",
                      "value": dst.buffer.texture_shape.width, "type": "int"}],
    })


# Fused WebGL Adam: 3 compute kernels (m, v, param) + 3 copy-backs per param,
# instead of ~17 cupy ops. Assumes capturable (no bias correction). Big win since
# Adam is ~340 of the ~575 kernels/step on WebGL.
_ADAM_M_GLSL = f"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_m; uniform sampler2D tex_g; uniform float B1;
out float fragColor;
{_GL_FETCH}
void main() {{ int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
fragColor = B1 * fetch(tex_m, idx) + (1.0 - B1) * fetch(tex_g, idx); }}
"""
_ADAM_V_GLSL = f"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_v; uniform sampler2D tex_g; uniform float B2;
out float fragColor;
{_GL_FETCH}
void main() {{ int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
float gv = fetch(tex_g, idx); fragColor = B2 * fetch(tex_v, idx) + (1.0 - B2) * gv * gv; }}
"""
_ADAM_P_GLSL = f"""#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_p; uniform sampler2D tex_m; uniform sampler2D tex_v;
uniform float LR; uniform float EPS; uniform float WD;
out float fragColor;
{_GL_FETCH}
void main() {{ int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
float p = fetch(tex_p, idx);
fragColor = p - LR * (fetch(tex_m, idx) / (sqrt(fetch(tex_v, idx)) + EPS) + WD * p); }}
"""
_adam_gl = {"added": False}


def _webgl_adam(p_data, g, m, v, lr, b1, b2, eps, wd=0.0):
    plat = _copy_kernel["plat"]
    if not _adam_gl["added"]:
        plat.addKernel("adam_m", {"source": _ADAM_M_GLSL})
        plat.addKernel("adam_v", {"source": _ADAM_V_GLSL})
        plat.addKernel("adam_p", {"source": _ADAM_P_GLSL})
        _adam_gl["added"] = True
    mtmp = _zeros(m.shape); vtmp = _zeros(v.shape); ptmp = _zeros(p_data.shape)
    W = lambda a: a.buffer.texture_shape.width
    plat.runKernel({"name": "adam_m",
        "inputs": [{"name": "tex_m", "id": m.buffer.buffer_id}, {"name": "tex_g", "id": g.buffer.buffer_id}],
        "output": mtmp.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(mtmp), "type": "int"},
                     {"name": "B1", "value": b1, "type": "float"}]})
    plat.runKernel({"name": "adam_v",
        "inputs": [{"name": "tex_v", "id": v.buffer.buffer_id}, {"name": "tex_g", "id": g.buffer.buffer_id}],
        "output": vtmp.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(vtmp), "type": "int"},
                     {"name": "B2", "value": b2, "type": "float"}]})
    plat.runKernel({"name": "adam_p",
        "inputs": [{"name": "tex_p", "id": p_data.buffer.buffer_id},
                   {"name": "tex_m", "id": mtmp.buffer.buffer_id},
                   {"name": "tex_v", "id": vtmp.buffer.buffer_id}],
        "output": ptmp.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": W(ptmp), "type": "int"},
                     {"name": "LR", "value": lr, "type": "float"},
                     {"name": "EPS", "value": eps, "type": "float"},
                     {"name": "WD", "value": wd, "type": "float"}]})
    _webgl_copy_into(m, mtmp)
    _webgl_copy_into(v, vtmp)
    _webgl_copy_into(p_data, ptmp)


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, capturable=False):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay          # decoupled (AdamW-style)
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0
        self.fused = _adam_backend_ready()
        self.webgl = (not self.fused) and _webgl_ready()
        # capturable: drop the step-dependent bias correction so the update is a
        # STATIC op sequence (identical every step) — required for graph capture,
        # where t can't advance inside a replay. A valid, stable Adam variant.
        self.capturable = capturable

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        if self.capturable:
            bc1 = bc2 = 1.0  # no bias correction -> static update, safe to capture
        else:
            bc1 = 1.0 - self.b1 ** self.t
            bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.m[i] is None:
                self.m[i] = _zeros(g.shape)
                self.v[i] = _zeros(g.shape)
            if self.fused:
                # one kernel updates param, m, v in place
                _fused_adam(p.data, g, self.m[i], self.v[i],
                            self.lr, self.b1, self.b2, self.eps, bc1, bc2, self.wd)
            elif self.webgl and self.capturable:
                # WebGL fused Adam: 3 GLSL compute kernels + 3 copy-backs into the
                # PERSISTENT m/v/param textures. Fully captured, no Python, no CPU
                # writeback. Assumes bc==1 (capturable). ~6 kernels/param vs ~17.
                _webgl_adam(p.data, g, self.m[i], self.v[i],
                            self.lr, self.b1, self.b2, self.eps, self.wd)
            elif self.webgl:
                mtmp = self.b1 * self.m[i] + (1.0 - self.b1) * g
                vtmp = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
                ptmp = p.data - self.lr * ((mtmp * (1.0 / bc1)) / (xp.sqrt(vtmp * (1.0 / bc2)) + self.eps) + self.wd * p.data)
                _webgl_copy_into(self.m[i], mtmp)
                _webgl_copy_into(self.v[i], vtmp)
                _webgl_copy_into(p.data, ptmp)
            else:
                # non-captured fallback (numpy / other): plain in-place update
                self.m[i][...] = self.b1 * self.m[i] + (1.0 - self.b1) * g
                self.v[i][...] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
                mhat = self.m[i] * (1.0 / bc1)
                vhat = self.v[i] * (1.0 / bc2)
                p.data[...] = p.data - self.lr * (mhat / (xp.sqrt(vhat) + self.eps) + self.wd * p.data)



# ---- quantization: GPTQ-compatible group-wise int4/int8 --------------------
# AutoGPTQ tensor layout (per Linear weight W of shape (K, N)):
#   qweight (K/per, N) int32   -- `per`=32/bits values packed along K
#   qzeros  (nG, N/per) int32  -- zero-points packed along N
#   scales  (nG, N) f32,  nG = K/group_size
#   dequant: w[k,n] = scales[k//gs, n] * (q[k,n] - zero[k//gs, n])
# Quantization is PER-TENSOR (independent) -> naturally streamable.
def _gptq_quantize(W, group_size=32, bits=4):
    K, N = W.shape
    per = 32 // bits
    qmax = (1 << bits) - 1
    assert K % group_size == 0 and K % per == 0 and N % per == 0, "dims must divide group/pack size"
    nG = K // group_size
    scales = np.zeros((nG, N), np.float32)
    zeros = np.zeros((nG, N), np.int32)
    q = np.zeros((K, N), np.int32)
    for g in range(nG):
        blk = W[g * group_size:(g + 1) * group_size]
        wmin = blk.min(0); wmax = blk.max(0)
        sc = (wmax - wmin) / qmax
        sc[sc == 0] = 1e-8
        zp = np.clip(np.round(-wmin / sc), 0, qmax).astype(np.int32)
        scales[g] = sc; zeros[g] = zp
        q[g * group_size:(g + 1) * group_size] = np.clip(np.round(blk / sc) + zp, 0, qmax).astype(np.int32)
    # vectorized bit-packing (was a per-element Python loop -> slow for GGUF requant)
    sh_k = (np.arange(per, dtype=np.int32) * bits).reshape(1, per, 1)
    qweight = np.bitwise_or.reduce(q.reshape(K // per, per, N) << sh_k, axis=1).astype(np.int32)
    sh_n = (np.arange(per, dtype=np.int32) * bits).reshape(1, 1, per)
    qzeros = np.bitwise_or.reduce(zeros.reshape(nG, N // per, per) << sh_n, axis=2).astype(np.int32)
    return qweight, qzeros, scales, K, N


# GPTQ dequant-matmul. The naive form (one thread per output, loop k) re-reads
# each packed qweight u32 PER times, and scales/qzeros `gs` times, and every
# thread re-reads the whole x row -- ~32x more traffic than the weights alone.
# This version: hoist scales/qzeros to the group loop, unpack each u32 once,
# stage the x tile in workgroup memory (shared by the 64 threads), and tile 4
# rows of M per thread (vec4) so prefill amortizes the weight reads.
_GPTQ_WGSL = """@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> qweight: array<u32>;
@group(0) @binding(2) var<storage,read> qzeros: array<u32>;
@group(0) @binding(3) var<storage,read> scales: array<f32>;
@group(0) @binding(4) var<storage,read_write> outp: array<f32>;
struct CMeta { M:u32, N:u32, K:u32, gs:u32, }
@group(0) @binding(5) var<storage,read> c: CMeta;
var<workgroup> xs: array<f32, XSSZ>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let m0 = gid.y * 4u;
  let tid = lid.x;
  let Ndp = c.N / PERu;
  let nG = c.K / GSu;
  let kbPerG = GSu / PERu;
  var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var g: u32 = 0u; g < nG; g = g + 1u) {
    let xbase = g * GSu;
    for (var idx: u32 = tid; idx < XSSZu; idx = idx + 64u) {
      let r = idx / GSu;
      let cc = idx - r * GSu;
      let mm = m0 + r;
      var val: f32 = 0.0;
      if (mm < c.M) { val = x[mm * c.K + xbase + cc]; }
      xs[idx] = val;
    }
    workgroupBarrier();
    if (n < c.N) {
      let sc = scales[g * c.N + n];
      let qz = qzeros[g * Ndp + n / PERu];
      let zv = f32((qz >> ((n % PERu) * BITSu)) & MASKu) + ZOFFf;
      var part = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      for (var t: u32 = 0u; t < kbPerG; t = t + 1u) {
        let qw = qweight[(g * kbPerG + t) * c.N + n];
        let ko = t * PERu;
        for (var j: u32 = 0u; j < PERu; j = j + 1u) {
          let qv = f32((qw >> (j * BITSu)) & MASKu) - zv;
          let kk = ko + j;
          let xv = vec4<f32>(xs[kk], xs[GSu + kk], xs[2u * GSu + kk], xs[3u * GSu + kk]);
          part = part + xv * qv;
        }
      }
      acc = acc + sc * part;
    }
    workgroupBarrier();
  }
  if (n < c.N) {
    if (m0 + 0u < c.M) { outp[(m0 + 0u) * c.N + n] = acc.x; }
    if (m0 + 1u < c.M) { outp[(m0 + 1u) * c.N + n] = acc.y; }
    if (m0 + 2u < c.M) { outp[(m0 + 2u) * c.N + n] = acc.z; }
    if (m0 + 3u < c.M) { outp[(m0 + 3u) * c.N + n] = acc.w; }
  }
}
"""
# WebGL has no workgroup memory, but the same hoisting/unpacking removes the
# scales/qzeros/qweight redundancy (x still relies on the texture cache).
_GL_GPTQ = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp isampler2D;
uniform int _ka_tex_output_texture_w; uniform int M, N, K, gs;
uniform sampler2D tex_x, tex_s; uniform isampler2D tex_qw, tex_qz;
out float fragColor;
FETCH
int ifetch(isampler2D t, int idx){ int tw=textureSize(t,0).x; int y=idx/tw; int x=idx-y*tw; return texelFetch(t,ivec2(x,y),0).r; }
void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=M*N){fragColor=0.0;return;}
  int m=i/N; int n=i-m*N; int Ndp=N/PER;
  int nG=K/gs; int kbPerG=gs/PER;
  float sum=0.0;
  for(int g=0; g<nG; g++){
    float sc = fetch(tex_s, g*N+n);
    int qz = ifetch(tex_qz, g*Ndp + n/PER);
    float zv = float((qz>>((n%PER)*BITS))&MASK) + ZOFFf;
    float part = 0.0;
    for(int t=0;t<kbPerG;t++){
      int kb = g*kbPerG + t;
      int qw = ifetch(tex_qw, kb*N + n);
      int kb0 = kb*PER;
      for(int j=0;j<PER;j++){
        float qv = float((qw>>(j*BITS))&MASK) - zv;
        part += fetch(tex_x, m*K + kb0 + j) * qv;
      }
    }
    sum += sc*part;
  }
  fragColor=sum;
}
""".replace("FETCH", _GL_FETCH)
# Decode (M==1) is a GEMV: the tiled kernel above spawns only N threads, each
# serially reducing over K, so it is occupancy/latency-bound (10 GB/s) rather
# than bandwidth-bound. This variant splits the K reduction across KS lanes
# (KS*N threads) and reduces the partial sums in workgroup memory.
_GPTQ_GEMV_WGSL = """@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> qweight: array<u32>;
@group(0) @binding(2) var<storage,read> qzeros: array<u32>;
@group(0) @binding(3) var<storage,read> scales: array<f32>;
@group(0) @binding(4) var<storage,read_write> outp: array<f32>;
struct CMeta { M:u32, N:u32, K:u32, gs:u32, }
@group(0) @binding(5) var<storage,read> c: CMeta;
var<workgroup> xsg: array<f32, KSxGS>;
var<workgroup> psum: array<f32, KSx64>;
@compute @workgroup_size(64, KS)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let lx = lid.x;
  let ly = lid.y;
  let Ndp = c.N / PERu;
  let nG = c.K / GSu;
  let kbPerG = GSu / PERu;
  let steps = (nG + KSu - 1u) / KSu;
  var sum: f32 = 0.0;
  for (var gi: u32 = 0u; gi < steps; gi = gi + 1u) {
    let g = gi * KSu + ly;
    for (var t: u32 = lx; t < GSu; t = t + 64u) {
      var val: f32 = 0.0;
      if (g < nG) { val = x[g * GSu + t]; }
      xsg[ly * GSu + t] = val;
    }
    workgroupBarrier();
    if (g < nG && n < c.N) {
      let sc = scales[g * c.N + n];
      let qz = qzeros[g * Ndp + n / PERu];
      let zv = f32((qz >> ((n % PERu) * BITSu)) & MASKu) + ZOFFf;
      var part: f32 = 0.0;
      for (var t2: u32 = 0u; t2 < kbPerG; t2 = t2 + 1u) {
        let qw = qweight[(g * kbPerG + t2) * c.N + n];
        let ko = t2 * PERu;
        for (var j: u32 = 0u; j < PERu; j = j + 1u) {
          part = part + xsg[ly * GSu + ko + j] * (f32((qw >> (j * BITSu)) & MASKu) - zv);
        }
      }
      sum = sum + sc * part;
    }
    workgroupBarrier();
  }
  psum[ly * 64u + lx] = sum;
  workgroupBarrier();
  if (ly == 0u && n < c.N) {
    var tot: f32 = 0.0;
    for (var r: u32 = 0u; r < KSu; r = r + 1u) { tot = tot + psum[r * 64u + lx]; }
    outp[n] = tot;
  }
}
"""
_GPTQ_KS = 8
_gptq_k = {"wgpu": set(), "gl": set()}


def _gptq_gemv_src(bits, gs, ks=_GPTQ_KS, zoff=0.0):
    per = 32 // bits
    src = _GPTQ_GEMV_WGSL
    # longest/most-specific tokens first
    for k, v in [("ZOFFf", "%.1f" % float(zoff)), ("KSxGS", str(ks * gs)),
                 ("KSx64", str(ks * 64)), ("KSu", f"{ks}u"), ("KS", str(ks)),
                 ("GSu", f"{gs}u"), ("PERu", f"{per}u"),
                 ("BITSu", f"{bits}u"), ("MASKu", f"{(1 << bits) - 1}u")]:
        src = src.replace(k, v)
    return src


def _gptq_src(tmpl, bits, gs=None, zoff=0.0):
    per = 32 // bits
    d = {"ZOFFf": "%.1f" % float(zoff)}
    if gs is not None:                       # tiled dequant-matmul kernel only
        # XSSZu must be substituted before XSSZ; likewise PERu before PER, etc.
        d.update({"XSSZu": f"{4 * gs}u", "XSSZ": str(4 * gs), "GSu": f"{gs}u"})
    d.update({"PERu": f"{per}u", "PER": str(per), "BITSu": f"{bits}u", "BITS": str(bits),
              "MASKu": f"{(1 << bits) - 1}u", "MASK": str((1 << bits) - 1)})
    for k, v in d.items():
        tmpl = tmpl.replace(k, v)
    return tmpl


def _gptq_matmul(xf, qweight, qzeros, scales, K, N, gs, bits, zoff=0.0):
    M = int(xf.shape[0])
    gemv = (M == 1)                                   # decode path: split-K GEMV
    zt = 1 if zoff else 0                             # AutoGPTQ stores zero-1
    key = (bits, gs, gemv, zt)
    name = f"gptq{'v' if gemv else ''}{bits}_g{gs}_z{zt}"
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if key not in _gptq_k["wgpu"]:
            src = (_gptq_gemv_src(bits, gs, zoff=zoff) if gemv
                   else _gptq_src(_GPTQ_WGSL, bits, gs, zoff=zoff))
            plat.addKernel(name, {"source": src,
                "bindingTypes": ["read-only-storage"] * 4 + ["storage", "read-only-storage"]})
            _gptq_k["wgpu"].add(key)
        of = _empty((M, N))
        meta = _adam_kernel["make_meta"]((M, N, K, gs), "u4,u4,u4,u4")
        plat.runKernel({"name": name,
            "tensors": [xf.buffer.buffer_id, qweight.buffer.buffer_id, qzeros.buffer.buffer_id, scales.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (N + 63) // 64, "y": 1 if gemv else (M + 3) // 4, "z": 1}})
        return of
    _webgl_ready()
    plat = _copy_kernel["plat"]
    if key not in _gptq_k["gl"]:
        plat.addKernel(name, {"source": _gptq_src(_GL_GPTQ, bits, gs, zoff=zoff)})
        _gptq_k["gl"].add(key)
    of = _empty((M, N))
    plat.runKernel({"name": name,
        "inputs": [{"name": "tex_x", "id": xf.buffer.buffer_id}, {"name": "tex_s", "id": scales.buffer.buffer_id},
                   {"name": "tex_qw", "id": qweight.buffer.buffer_id}, {"name": "tex_qz", "id": qzeros.buffer.buffer_id}],
        "output": of.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": of.buffer.texture_shape.width, "type": "int"},
                     {"name": "M", "value": M, "type": "int"}, {"name": "N", "value": N, "type": "int"},
                     {"name": "K", "value": K, "type": "int"}, {"name": "gs", "value": gs, "type": "int"}]})
    return of


class QuantizedLinear(Module):
    """Inference-only GPTQ-format weight-quantized Linear (group-wise int4/int8)."""
    def __init__(self, qweight, qzeros, scales, bias, Kt, Nt, Kp, Np, gs, bits, zero_offset=0.0):
        self.qweight = xp.asarray(qweight)     # int32 GPU
        self.qzeros = xp.asarray(qzeros)       # int32 GPU
        self.scales = xp.asarray(scales)       # f32 GPU
        self.bias = xp.asarray(bias.astype(np.float32))
        self.Kt = Kt; self.Nt = Nt; self.Kp = Kp; self.Np = Np; self.gs = gs; self.bits = bits
        self.zero_offset = float(zero_offset)  # AutoGPTQ stores (zero-1) -> use 1.0

    @staticmethod
    def from_autogptq(qweight, qzeros, scales, bias, gs, bits):
        """Load real AutoGPTQ int4/int8 tensors directly (desc_act=false).
        qweight (K/per,N) int32, qzeros (K/gs,N/per) int32, scales (K/gs,N),
        bias (N,) or None. Layout matches ours; zero-point uses the +1 convention."""
        per = 32 // bits
        K = int(qweight.shape[0]) * per
        N = int(qweight.shape[1])
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return QuantizedLinear(np.asarray(qweight, np.int32), np.asarray(qzeros, np.int32),
                               np.asarray(scales, np.float32), b, K, N, K, N, gs, bits,
                               zero_offset=1.0)

    @staticmethod
    def from_linear(lin, group_size=32, bits=4):
        W = cp.asnumpy(lin.weight.data) if GPU else np.asarray(lin.weight.data)   # (K, N)
        b = cp.asnumpy(lin.bias.data) if GPU else np.asarray(lin.bias.data)
        K, N = W.shape
        per = 32 // bits
        kmul = group_size if group_size % per == 0 else group_size * per
        Kp = K + (-K) % kmul            # pad contraction to divide group_size & per
        Np = N + (-N) % per             # pad output to divide pack factor
        if Kp != K or Np != N:
            W = np.pad(W, ((0, Kp - K), (0, Np - N)))
        qw, qz, sc, _, _ = _gptq_quantize(W, group_size, bits)
        return QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, group_size, bits)

    def forward(self, x):
        xd = x.data
        lead = xd.shape[:-1]
        xf = _contig(xd.reshape(-1, self.Kt))
        if self.Kp != self.Kt:                      # pad activation to padded K
            xp_ = _zeros((int(xf.shape[0]), self.Kp)); xp_[:, :self.Kt] = xf; xf = xp_
        of = _gptq_matmul(xf, self.qweight, self.qzeros, self.scales, self.Kp, self.Np,
                          self.gs, self.bits, zoff=self.zero_offset)
        if self.Np != self.Nt:
            of = _contig(of[:, :self.Nt])
        return Tensor((of + self.bias).reshape(*lead, self.Nt))   # inference-only

    def nbytes(self):
        return int(self.qweight.size * 4 + self.qzeros.size * 4 + self.scales.size * 4 + self.bias.size * 4)


def quantize_model(module, group_size=32, bits=4):
    """Replace each Linear -> QuantizedLinear and each Embedding ->
    QuantizedEmbedding, one tensor at a time (streaming-friendly)."""
    for name, val in list(vars(module).items()):
        if isinstance(val, Linear):
            setattr(module, name, QuantizedLinear.from_linear(val, group_size, bits))
        elif isinstance(val, Embedding):
            setattr(module, name, QuantizedEmbedding.from_embedding(val, group_size, bits))
        elif isinstance(val, Conv2d):
            setattr(module, name, QuantizedConv2d.from_conv2d(val, group_size, bits))
        elif isinstance(val, Conv3d):
            setattr(module, name, QuantizedConv3d.from_conv3d(val, group_size, bits))
        elif isinstance(val, Module):
            quantize_model(val, group_size, bits)
        elif isinstance(val, (list, tuple)):
            for it in val:
                if isinstance(it, Module):
                    quantize_model(it, group_size, bits)
    return module


# ---- quantized embedding (group-wise int4/int8, gather + dequant) ----------
def _quantize_emb(W, group_size, bits):
    """W: (vocab, dim). Per-row group-wise along dim. Returns qweight (vocab,
    dim/per) int32, zeros (vocab, nG) f32, scales (vocab, nG) f32."""
    vocab, dim = W.shape
    per = 32 // bits
    qmax = (1 << bits) - 1
    assert dim % group_size == 0 and dim % per == 0
    nG = dim // group_size
    scales = np.zeros((vocab, nG), np.float32)
    zeros = np.zeros((vocab, nG), np.float32)
    q = np.zeros((vocab, dim), np.int32)
    for g in range(nG):
        blk = W[:, g * group_size:(g + 1) * group_size]      # (vocab, gs)
        wmin = blk.min(1); wmax = blk.max(1)
        sc = (wmax - wmin) / qmax; sc[sc == 0] = 1e-8
        zp = np.clip(np.round(-wmin / sc), 0, qmax)
        scales[:, g] = sc; zeros[:, g] = zp
        q[:, g * group_size:(g + 1) * group_size] = np.clip(np.round(blk / sc[:, None]) + zp[:, None], 0, qmax).astype(np.int32)
    qweight = np.zeros((vocab, dim // per), np.int32)
    for d in range(dim):
        qweight[:, d // per] |= (q[:, d] << ((d % per) * bits))
    return qweight, zeros, scales, vocab, dim


_QEMB_WGSL = """@group(0) @binding(0) var<storage,read> idx: array<f32>;
@group(0) @binding(1) var<storage,read> qweight: array<u32>;
@group(0) @binding(2) var<storage,read> zeros: array<f32>;
@group(0) @binding(3) var<storage,read> scales: array<f32>;
@group(0) @binding(4) var<storage,read_write> outp: array<f32>;
struct CMeta { M:u32, dim:u32, gs:u32, }
@group(0) @binding(5) var<storage,read> c: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.M*c.dim) { return; }
  let m = i / c.dim; let d = i - m*c.dim;
  let v = u32(idx[m]); let nG = c.dim / c.gs; let g = d / c.gs; let prw = c.dim / PERu;
  let qw = qweight[v*prw + d/PERu];
  let qv = (qw >> ((d%PERu)*BITSu)) & MASKu;
  outp[i] = scales[v*nG + g] * (f32(qv) - zeros[v*nG + g]);
}
"""
_GL_QEMB = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp isampler2D;
uniform int _ka_tex_output_texture_w; uniform int M, dim, gs;
uniform sampler2D tex_idx, tex_z, tex_s; uniform isampler2D tex_qw;
out float fragColor;
FETCH
int ifetch(isampler2D t, int idx){ int tw=textureSize(t,0).x; int y=idx/tw; int x=idx-y*tw; return texelFetch(t,ivec2(x,y),0).r; }
void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=M*dim){fragColor=0.0;return;}
  int m=i/dim; int d=i-m*dim;
  int v=int(fetch(tex_idx,m)+0.5); int nG=dim/gs; int g=d/gs; int prw=dim/PER;
  int qw=ifetch(tex_qw, v*prw + d/PER);
  int qv=(qw>>((d%PER)*BITS))&MASK;
  fragColor = fetch(tex_s, v*nG+g) * (float(qv) - fetch(tex_z, v*nG+g));
}
""".replace("FETCH", _GL_FETCH)
_qemb_k = {"wgpu": set(), "gl": set()}


def _qemb_gather(gidx, qweight, zeros, scales, M, dim, gs, bits):
    name = f"qemb{bits}"
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if bits not in _qemb_k["wgpu"]:
            plat.addKernel(name, {"source": _gptq_src(_QEMB_WGSL, bits),
                "bindingTypes": ["read-only-storage"] * 4 + ["storage", "read-only-storage"]})
            _qemb_k["wgpu"].add(bits)
        of = _empty((M, dim))
        meta = _adam_kernel["make_meta"]((M, dim, gs), "u4,u4,u4")
        plat.runKernel({"name": name,
            "tensors": [gidx.buffer.buffer_id, qweight.buffer.buffer_id, zeros.buffer.buffer_id, scales.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (M * dim + 63) // 64, "y": 1, "z": 1}})
        return of
    _webgl_ready()
    plat = _copy_kernel["plat"]
    if bits not in _qemb_k["gl"]:
        plat.addKernel(name, {"source": _gptq_src(_GL_QEMB, bits)})
        _qemb_k["gl"].add(bits)
    of = _empty((M, dim))
    plat.runKernel({"name": name,
        "inputs": [{"name": "tex_idx", "id": gidx.buffer.buffer_id}, {"name": "tex_z", "id": zeros.buffer.buffer_id},
                   {"name": "tex_s", "id": scales.buffer.buffer_id}, {"name": "tex_qw", "id": qweight.buffer.buffer_id}],
        "output": of.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": of.buffer.texture_shape.width, "type": "int"},
                     {"name": "M", "value": M, "type": "int"}, {"name": "dim", "value": dim, "type": "int"}, {"name": "gs", "value": gs, "type": "int"}]})
    return of


class QuantizedEmbedding(Module):
    """Inference-only group-wise int4/int8 quantized embedding (gather+dequant)."""
    def __init__(self, qweight, zeros, scales, vocab, dim, dim_pad, gs, bits):
        self.qweight = xp.asarray(qweight); self.zeros = xp.asarray(zeros); self.scales = xp.asarray(scales)
        self.vocab = vocab; self.dim = dim; self.dim_pad = dim_pad; self.gs = gs; self.bits = bits

    @staticmethod
    def from_embedding(emb, group_size=32, bits=4):
        W = cp.asnumpy(emb.weight.data) if GPU else np.asarray(emb.weight.data)
        vocab, dim = W.shape
        pad = (-dim) % group_size                 # pad dim to divide group_size (& pack factor)
        if pad:
            W = np.pad(W, ((0, 0), (0, pad)))
        qw, zr, sc, _, dim_pad = _quantize_emb(W, group_size, bits)
        return QuantizedEmbedding(qw, zr, sc, vocab, dim, dim_pad, group_size, bits)

    def forward(self, idx):
        ish = tuple(np.asarray(idx).shape)
        flat = np.asarray(idx).reshape(-1).astype(np.float32)
        gidx = xp.asarray(flat)
        M = int(flat.shape[0])
        of = _qemb_gather(gidx, self.qweight, self.zeros, self.scales, M, self.dim_pad, self.gs, self.bits)
        if self.dim_pad != self.dim:
            of = _contig(of[:, :self.dim])
        return Tensor(of.reshape(*(ish + (self.dim,))))

    def nbytes(self):
        return int(self.qweight.size * 4 + self.zeros.size * 4 + self.scales.size * 4)


# ---- quantized conv2d (dequant packed weight -> reuse conv2d kernel) --------
class QuantizedConv2d(Module):
    """Inference-only weight-quantized Conv2d. Stores the weight as group-wise
    int4/int8 (4x-8x smaller). forward dequantizes to a transient fp32 weight
    (conv weights are small) and runs the verified conv2d kernel."""
    def __init__(self, qweight, zeros, scales, bias, shape, CinKK, dim_pad, gs, bits, stride, padding):
        self.qweight = xp.asarray(qweight); self.zeros = xp.asarray(zeros); self.scales = xp.asarray(scales)
        self.bias = xp.asarray(bias.astype(np.float32))
        self.Cout, self.Cin, self.KH, self.KW = shape
        self.CinKK = CinKK; self.dim_pad = dim_pad; self.gs = gs; self.bits = bits
        self.stride = stride; self.padding = padding

    @staticmethod
    def from_conv2d(conv, group_size=32, bits=4):
        W = cp.asnumpy(conv.weight.data) if GPU else np.asarray(conv.weight.data)   # (Cout,Cin,KH,KW)
        Cout, Cin, KH, KW = W.shape
        CinKK = Cin * KH * KW
        W2 = W.reshape(Cout, CinKK)
        pad = (-CinKK) % group_size
        if pad:
            W2 = np.pad(W2, ((0, 0), (0, pad)))
        qw, zr, sc, _, dim_pad = _quantize_emb(W2, group_size, bits)   # per-row group-wise
        b = cp.asnumpy(conv.bias.data) if GPU else np.asarray(conv.bias.data)
        return QuantizedConv2d(qw, zr, sc, b, (Cout, Cin, KH, KW), CinKK, dim_pad, group_size, bits, conv.stride, conv.padding)

    def forward(self, x):
        idx = xp.asarray(np.arange(self.Cout, dtype=np.float32))
        Wfp = _qemb_gather(idx, self.qweight, self.zeros, self.scales, self.Cout, self.dim_pad, self.gs, self.bits)
        Wt = Tensor(_contig(Wfp[:, :self.CinKK]).reshape(self.Cout, self.Cin, self.KH, self.KW))
        return conv2d(x, Wt, Tensor(self.bias), self.stride, self.padding)

    def nbytes(self):
        return int(self.qweight.size * 4 + self.zeros.size * 4 + self.scales.size * 4 + self.bias.size * 4)


# ---- QLoRA: dequantize frozen weight (differentiable wrt input) + LoRA ------
_DQF_WGSL = """@group(0) @binding(0) var<storage,read> qweight: array<u32>;
@group(0) @binding(1) var<storage,read> qzeros: array<u32>;
@group(0) @binding(2) var<storage,read> scales: array<f32>;
@group(0) @binding(3) var<storage,read_write> outp: array<f32>;
struct CMeta { Kp:u32, Np:u32, gs:u32, }
@group(0) @binding(4) var<storage,read> c: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= c.Kp*c.Np) { return; }
  let k = i / c.Np; let n = i - k*c.Np; let g = k / c.gs; let Npp = c.Np / PERu;
  let qw = qweight[(k/PERu)*c.Np + n];
  let qv = (qw >> ((k%PERu)*BITSu)) & MASKu;
  let qz = qzeros[g*Npp + n/PERu];
  let zv = (qz >> ((n%PERu)*BITSu)) & MASKu;
  outp[i] = scales[g*c.Np + n] * (f32(qv) - f32(zv));
}
"""
_GL_DQF = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp isampler2D;
uniform int _ka_tex_output_texture_w; uniform int Kp, Np, gs;
uniform sampler2D tex_s; uniform isampler2D tex_qw, tex_qz;
out float fragColor;
FETCH
int ifetch(isampler2D t, int idx){ int tw=textureSize(t,0).x; int y=idx/tw; int x=idx-y*tw; return texelFetch(t,ivec2(x,y),0).r; }
void main(){
  int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w; if(i>=Kp*Np){fragColor=0.0;return;}
  int k=i/Np; int n=i-k*Np; int g=k/gs; int Npp=Np/PER;
  int qw=ifetch(tex_qw,(k/PER)*Np+n); int qv=(qw>>((k%PER)*BITS))&MASK;
  int qz=ifetch(tex_qz,g*Npp+n/PER); int zv=(qz>>((n%PER)*BITS))&MASK;
  fragColor = fetch(tex_s,g*Np+n)*(float(qv)-float(zv));
}
""".replace("FETCH", _GL_FETCH)
_dqf_k = {"wgpu": set(), "gl": set()}


def _dequant_full(qweight, qzeros, scales, Kp, Np, gs, bits):
    name = f"dqf{bits}"
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if bits not in _dqf_k["wgpu"]:
            plat.addKernel(name, {"source": _gptq_src(_DQF_WGSL, bits),
                "bindingTypes": ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
            _dqf_k["wgpu"].add(bits)
        of = _empty((Kp, Np))
        meta = _adam_kernel["make_meta"]((Kp, Np, gs), "u4,u4,u4")
        plat.runKernel({"name": name, "tensors": [qweight.buffer.buffer_id, qzeros.buffer.buffer_id, scales.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id],
                        "workGroups": {"x": (Kp * Np + 63) // 64, "y": 1, "z": 1}})
        return of
    _webgl_ready()
    plat = _copy_kernel["plat"]
    if bits not in _dqf_k["gl"]:
        plat.addKernel(name, {"source": _gptq_src(_GL_DQF, bits)})
        _dqf_k["gl"].add(bits)
    of = _empty((Kp, Np))
    plat.runKernel({"name": name,
        "inputs": [{"name": "tex_s", "id": scales.buffer.buffer_id}, {"name": "tex_qw", "id": qweight.buffer.buffer_id}, {"name": "tex_qz", "id": qzeros.buffer.buffer_id}],
        "output": of.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": of.buffer.texture_shape.width, "type": "int"},
                     {"name": "Kp", "value": Kp, "type": "int"}, {"name": "Np", "value": Np, "type": "int"}, {"name": "gs", "value": gs, "type": "int"}]})
    return of


def _qlin_dequant_weight(ql):
    """Dequantize a QuantizedLinear's frozen weight to a (Kt, Nt) fp32 Tensor
    (constant; gradient flows to the input, not the weight)."""
    full = _dequant_full(ql.qweight, ql.qzeros, ql.scales, ql.Kp, ql.Np, ql.gs, ql.bits)
    if ql.Kp != ql.Kt or ql.Np != ql.Nt:
        full = _contig(full[:ql.Kt, :ql.Nt])
    return Tensor(full)   # no grad


class LoRALinear(Module):
    """QLoRA adapter over a frozen QuantizedLinear: y = x @ dequant(Wq) + bias +
    (x @ A) @ B * (alpha/rank). Only A, B are trainable."""
    def __init__(self, qlinear, rank=8, alpha=16):
        self.q = qlinear
        self.A = Parameter(np.random.randn(qlinear.Kt, rank).astype(np.float32) * 0.01)
        self.B = Parameter(np.zeros((rank, qlinear.Nt), dtype=np.float32))   # init 0 => starts == quantized
        self.scaling = alpha / rank
        self.bias_t = Tensor(qlinear.bias)   # frozen

    def forward(self, x):
        Wfp = _qlin_dequant_weight(self.q)         # (Kt, Nt) frozen, differentiable wrt x
        lead = x.shape[:-1]
        xf = x.reshape(-1, self.q.Kt)              # fold to 2D (WgPy matmul is 2D)
        base = xf.matmul(Wfp) + self.bias_t
        lora = xf.matmul(self.A).matmul(self.B) * self.scaling
        return (base + lora).reshape(*lead, self.q.Nt)


def add_lora(module, rank=8, alpha=16):
    """Wrap every QuantizedLinear with a trainable LoRA adapter."""
    for name, val in list(vars(module).items()):
        if isinstance(val, QuantizedLinear):
            setattr(module, name, LoRALinear(val, rank, alpha))
        elif isinstance(val, Module):
            add_lora(val, rank, alpha)
        elif isinstance(val, (list, tuple)):
            for it in val:
                if isinstance(it, Module):
                    add_lora(it, rank, alpha)
    return module


# ---- quantized conv3d (dequant packed weight -> reuse conv3d kernel) --------
class QuantizedConv3d(Module):
    """Inference-only weight-quantized Conv3d (group-wise int4/int8)."""
    def __init__(self, qweight, zeros, scales, bias, shape, CinK, dim_pad, gs, bits, stride, padding):
        self.qweight = xp.asarray(qweight); self.zeros = xp.asarray(zeros); self.scales = xp.asarray(scales)
        self.bias = xp.asarray(bias.astype(np.float32))
        self.Cout, self.Cin, self.KD, self.KH, self.KW = shape
        self.CinK = CinK; self.dim_pad = dim_pad; self.gs = gs; self.bits = bits
        self.stride = stride; self.padding = padding

    @staticmethod
    def from_conv3d(conv, group_size=32, bits=4):
        W = cp.asnumpy(conv.weight.data) if GPU else np.asarray(conv.weight.data)   # (Cout,Cin,KD,KH,KW)
        Cout, Cin, KD, KH, KW = W.shape
        CinK = Cin * KD * KH * KW
        W2 = W.reshape(Cout, CinK)
        pad = (-CinK) % group_size
        if pad:
            W2 = np.pad(W2, ((0, 0), (0, pad)))
        qw, zr, sc, _, dim_pad = _quantize_emb(W2, group_size, bits)
        b = cp.asnumpy(conv.bias.data) if GPU else np.asarray(conv.bias.data)
        return QuantizedConv3d(qw, zr, sc, b, (Cout, Cin, KD, KH, KW), CinK, dim_pad, group_size, bits, conv.stride, conv.padding)

    def forward(self, x):
        idx = xp.asarray(np.arange(self.Cout, dtype=np.float32))
        Wfp = _qemb_gather(idx, self.qweight, self.zeros, self.scales, self.Cout, self.dim_pad, self.gs, self.bits)
        Wt = Tensor(_contig(Wfp[:, :self.CinK]).reshape(self.Cout, self.Cin, self.KD, self.KH, self.KW))
        return conv3d(x, Wt, Tensor(self.bias), self.stride, self.padding)

    def nbytes(self):
        return int(self.qweight.size * 4 + self.zeros.size * 4 + self.scales.size * 4 + self.bias.size * 4)


# ---- Llama-family building blocks (RoPE, GQA, SwiGLU) -----------------------
def rope_tables(Tn, hd, theta=10000.0, offset=0):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(offset, offset + Tn), inv)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1).astype(np.float32)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1).astype(np.float32)
    return cos, sin


def _slice_last(x, start, end):
    """Autograd slice of the last axis: x[..., start:end]."""
    out = Tensor(_contig(x.data[..., start:end]), x.requires_grad, (x,), "slice")

    def _backward():
        if x.requires_grad:
            g = xp.zeros_like(x.data)
            g[..., start:end] = out.grad
            x._accum(g)
    out._backward = _backward
    return out


def apply_rope(t, cos, sin):
    """Rotary position embedding on the last axis (Llama 'rotate_half' convention).
    t: (..., T, hd); cos/sin: (T, hd) Tensors. Autograd-correct."""
    hd = t.shape[-1]
    h = hd // 2
    rot = cat([-_slice_last(t, h, hd), _slice_last(t, 0, h)], axis=-1)
    return t * cos + rot * sin


class SwiGLU(Module):
    def __init__(self, dim, ffn):
        self.gate = Linear(dim, ffn); self.up = Linear(dim, ffn); self.down = Linear(ffn, dim)
        for m in (self.gate, self.up, self.down):
            m.bias.data = xp.zeros(m.bias.data.shape, np.float32)   # Llama MLPs are bias-free

    def forward(self, x):
        return self.down(silu(self.gate(x)) * self.up(x))


class LlamaAttention(Module):
    """Grouped-query attention with rotary embeddings (bias-free, causal)."""
    def __init__(self, dim, n_heads, n_kv, theta=10000.0):
        self.H = n_heads; self.KV = n_kv; self.hd = dim // n_heads; self.dim = dim; self.theta = theta
        self.wq = Linear(dim, n_heads * self.hd); self.wk = Linear(dim, n_kv * self.hd)
        self.wv = Linear(dim, n_kv * self.hd); self.wo = Linear(n_heads * self.hd, dim)
        for m in (self.wq, self.wk, self.wv, self.wo):
            m.bias.data = xp.zeros(m.bias.data.shape, np.float32)

    def forward(self, x):
        B, Tn, D = x.shape
        H, KV, hd = self.H, self.KV, self.hd
        cos, sin = rope_tables(Tn, hd, self.theta)
        ct, st = Tensor(cos), Tensor(sin)

        def heads(t, nh):
            return t.reshape(B, Tn, nh, hd).permute(0, 2, 1, 3).reshape(B * nh, Tn, hd)
        q = apply_rope(heads(self.wq(x), H), ct, st)
        k = apply_rope(heads(self.wk(x), KV), ct, st)
        v = heads(self.wv(x), KV)
        rep = H // KV
        if rep > 1:      # GQA expand (data-level; inference)
            kd = k.data.reshape(B, KV, Tn, hd); vd = v.data.reshape(B, KV, Tn, hd)
            kd = xp.concatenate([kd[:, i:i + 1] for i in range(KV) for _ in range(rep)], axis=1).reshape(B * H, Tn, hd)
            vd = xp.concatenate([vd[:, i:i + 1] for i in range(KV) for _ in range(rep)], axis=1).reshape(B * H, Tn, hd)
            k = Tensor(_contig(kd)); v = Tensor(_contig(vd))
        mask = np.triu(np.full((Tn, Tn), -1e9, np.float32), 1)
        scores = bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5)) + Tensor(mask)
        o = bmm(softmax(scores), v).reshape(B, H, Tn, hd).permute(0, 2, 1, 3).reshape(B, Tn, D)
        return self.wo(o)


class LlamaBlock(Module):
    def __init__(self, dim, n_heads, n_kv, ffn, eps=1e-5, theta=10000.0):
        self.an = RMSNorm(dim, eps); self.attn = LlamaAttention(dim, n_heads, n_kv, theta)
        self.fn = RMSNorm(dim, eps); self.mlp = SwiGLU(dim, ffn)

    def forward(self, x):
        x = x + self.attn(self.an(x))
        return x + self.mlp(self.fn(x))
