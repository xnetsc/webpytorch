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

    def __getitem__(self, idx):
        """Slice a tensor. Inference-only: the result is detached and materialized, since a
        strided view is not something the GPU kernels can be handed."""
        return Tensor(_contig(self.data[idx]))

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
    # WARNING: host-backed in this WgPy build (construct.zeros -> np.zeros -> staging
    # upload), so it costs RAM twice over. Fine for small buffers; for anything seq²-sized
    # (attention scores) use _empty + a kernel that fully overwrites the output.
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
    # Materialize a (possibly transposed/strided) array — WgPy reshape/matmul are unreliable
    # on non-contiguous views, and the kernels here index the buffer linearly, so a view that
    # does not start at offset 0 or does not fill its buffer would read the wrong elements.
    # `* 1.0` forces a stride-aware kernel that produces one that does.
    #
    # An array already satisfying both is returned unchanged. Copying it is pure bandwidth,
    # and the KV cache is bound this way once per attention layer per token: at a 4096-token
    # context that copy alone moved ~940 MB per token — more than everything else in the step
    # put together, and the reason a larger context slowed decode down even when the
    # conversation was short.
    f = getattr(a, "flags", None)
    if f is not None and getattr(f, "c_contiguous_full", False):
        return a
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


# Decode attention, fused. The general path transposes the whole KV cache every token --
# every slot, including the ones not written yet -- then runs a batched matmul, a scale, a
# mask add, a multi-kernel softmax and a second matmul: about ten dispatches per layer, and
# measured the largest single item in a decode step. With one query position the whole thing
# is a single workgroup per head: score against each cached key, soft-max in workgroup
# memory, then accumulate the values. Sizes here are small (LMAX scores, HD lanes), so the
# reduction cost is dominated by what it replaces.
# Fused single-position attention, online-softmax (Flash-Attention style).
#
# Two things the general path cannot do, and both are why decode was slow:
#
#  * It attends over `valid` positions, not over the whole cache. A KV cache is one fixed
#    buffer sized for the context, and a matmul against it costs the WHOLE buffer on every
#    step no matter how little is filled -- so a model loaded with room for 32k tokens
#    decodes at 32k speed while answering its first question. llama.cpp scans n_kv, and so
#    does this. `valid` arrives in the meta buffer, which keeps ONE captured dispatch
#    correct for every step: the shape never changes, only a number the shader reads.
#  * Softmax runs blockwise with a running max and sum, so nothing is sized by the context
#    length. The previous kernel held every score in workgroup memory, which capped it at
#    1024 positions; here workgroup memory is 3 * 128 floats regardless.
#
# One workgroup per query head, 128 lanes. Per block of 128 positions: each lane scores one
# position, the block is reduced for its max and sum, and the accumulator is rescaled by
# exp(m_old - m_new) before the block is added -- the standard stable online update. `m_run`
# and `l_run` are per-lane but every lane derives them from the same reduced values, so they
# agree without needing to be shared.
_GQA_DECODE_WGSL = """@group(0) @binding(0)
var<storage,read_write> outp: array<f32>;
@group(0) @binding(1)
var<storage,read> q: array<f32>;
@group(0) @binding(2)
var<storage,read> kc: array<f32>;
@group(0) @binding(3)
var<storage,read> vc: array<f32>;
struct GMeta { nh: u32, nkv: u32, hd: u32, lmax: u32, valid: u32, use_ctl: u32, scale: f32, }
@group(0) @binding(4)
var<storage,read> gm: GMeta;
// The step control block the decode loop already rewrites each token: ctl[0] is the position
// just written. Reading the length from HERE rather than from GMeta is what lets one
// captured dispatch serve every step -- a capture replays fixed commands, so a value baked
// into a meta buffer at capture time would freeze the scan length at the first token's.
@group(0) @binding(5)
var<storage,read> ctl: array<i32>;
var<workgroup> sc: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> acc: array<f32, 256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let rep = gm.nh / gm.nkv;
  let kv = h / rep;
  let qo = h * gm.hd;
  let ko = kv * gm.lmax * gm.hd;
  var n: u32 = gm.valid;
  if (gm.use_ctl == 1u) { n = u32(max(ctl[0], 0)) + 1u; }
  n = clamp(n, 1u, gm.lmax);

  // One lane per head dimension where it fits, striding when hd exceeds the workgroup.
  for (var d: u32 = t; d < gm.hd; d = d + 128u) { acc[d] = 0.0; }
  var m_run: f32 = -1e30;
  var l_run: f32 = 0.0;
  workgroupBarrier();

  // Uniform loop bound: `n` comes from the meta buffer, so every lane runs the same number
  // of iterations and the barriers inside stay uniform.
  var base: u32 = 0u;
  loop {
    if (base >= n) { break; }
    let s = base + t;
    var d: f32 = -1e30;
    if (s < n) {
      var dd: f32 = 0.0;
      let kb = ko + s * gm.hd;
      for (var i: u32 = 0u; i < gm.hd; i = i + 1u) {
        dd = dd + q[qo + i] * kc[kb + i];
      }
      d = dd * gm.scale;
    }
    red[t] = d;
    workgroupBarrier();
    var r: u32 = 64u;
    loop {
      if (r == 0u) { break; }
      if (t < r) { red[t] = max(red[t], red[t + r]); }
      workgroupBarrier();
      r = r / 2u;
    }
    let m_new = max(m_run, red[0]);
    workgroupBarrier();

    var e: f32 = 0.0;
    if (s < n) { e = exp(d - m_new); }
    sc[t] = e;
    red[t] = e;
    workgroupBarrier();
    r = 64u;
    loop {
      if (r == 0u) { break; }
      if (t < r) { red[t] = red[t] + red[t + r]; }
      workgroupBarrier();
      r = r / 2u;
    }
    let corr = exp(m_run - m_new);
    l_run = l_run * corr + red[0];
    m_run = m_new;
    workgroupBarrier();

    // rescale what is already accumulated, then add this block's weighted values
    let cnt = min(128u, n - base);
    for (var d: u32 = t; d < gm.hd; d = d + 128u) {
      var o: f32 = acc[d] * corr;
      for (var pp: u32 = 0u; pp < cnt; pp = pp + 1u) {
        o = o + sc[pp] * vc[ko + (base + pp) * gm.hd + d];
      }
      acc[d] = o;
    }
    workgroupBarrier();
    base = base + 128u;
  }

  for (var d: u32 = t; d < gm.hd; d = d + 128u) { outp[qo + d] = acc[d] / l_run; }
}
"""
_gqa_k = {"added": False}
_GQA_FUSED = True      # A/B switch for the fused decode attention


def gqa_decode(q, kc, vc, mask, scale, valid=None, ctl=None):
    """Single-position grouped-query attention in one dispatch.

    `q` (nh, 1, hd); `kc`/`vc` (nkv, lmax, hd). `valid` is how many cache positions actually
    hold a token -- the kernel reads no further, which is what keeps decode speed tied to the
    conversation rather than to the context the model was loaded with. `mask` is accepted for
    signature compatibility and unused: with `valid` there is nothing to mask, since every
    position scanned is one that was written.

    Returns None when the backend or the shapes fall outside what the kernel covers, so
    callers keep the general path.
    """
    if not _adam_backend_ready():
        return None
    qd = q.data if isinstance(q, Tensor) else q
    kd = kc.data if isinstance(kc, Tensor) else kc
    vd = vc.data if isinstance(vc, Tensor) else vc
    nh, T, hd = (int(v) for v in qd.shape)
    nkv, lmax, hd2 = (int(v) for v in kd.shape)
    # `acc` is sized for hd <= 256, which covers every head dimension in use (128 and 256
    # are the common ones); anything larger keeps the general path.
    if T != 1 or hd != hd2 or hd > 256 or nh % nkv:
        return None
    n = lmax if valid is None else max(1, min(int(valid), lmax))
    plat = _adam_kernel["platform"]
    if not _gqa_k["added"]:
        plat.addKernel("gqa_decode", {"source": _GQA_DECODE_WGSL,
                                      "bindingTypes": ["storage"]
                                      + ["read-only-storage"] * 5})
        _gqa_k["added"] = True
    # Bind through named locals. Inlining `_contig(...)` into the list drops the only
    # reference to each temporary as soon as its id is read, so its GPU buffer can be
    # recycled for the next one -- two bindings then silently share a buffer.
    qc = _contig(qd); kcc = _contig(kd); vcc = _contig(vd)
    # Allocate flat: the kernel indexes linearly, and a 2-D allocation is not guaranteed
    # to be an unpadded row-major buffer.
    of = _empty((nh * hd,))
    meta = _adam_kernel["make_meta"]((nh, nkv, hd, lmax, n,
                                      1 if ctl is not None else 0, float(scale)),
                                     "u4,u4,u4,u4,u4,u4,f4")
    # binding 5 must always be bound; without a control buffer it points at the meta buffer
    # and `use_ctl` tells the shader to ignore it.
    cb = ctl.buffer if ctl is not None else meta
    plat.runKernel({"name": "gqa_decode",
                    "tensors": [of.buffer.buffer_id, qc.buffer.buffer_id,
                                kcc.buffer.buffer_id, vcc.buffer.buffer_id,
                                meta.buffer_id, cb.buffer_id],
                    "workGroups": {"x": nh, "y": 1, "z": 1}})
    return Tensor(of.reshape(nh, 1, hd))


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
    # GPU-native: the attention score matrix is seq²-sized, and _zeros would allocate it on
    # the HOST first then stage it up — that staging copy is what OOMs long prompts. The
    # kernel below writes every output element, so an uninitialized buffer is exact.
    s = _empty(xd.shape)
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


# A decode step is dominated by kernel launches, not bandwidth: on this stack every
# dispatch costs ~21us whatever its size, and RMS norm as an expression is six of them
# (square, mean, add eps, sqrt, divide, scale). Two per layer across 28 layers is most of
# the step. Fused, it is one launch. eps travels as its bit pattern because the meta
# buffer carries u32 words.
_RMS_WGSL = """@group(0) @binding(0)
var<storage,read> x: array<f32>;
@group(0) @binding(1)
var<storage,read> w: array<f32>;
@group(0) @binding(2)
var<storage,read_write> outp: array<f32>;
struct RMeta { T: u32, H: u32, epsbits: u32, }
@group(0) @binding(3)
var<storage,read> rm: RMeta;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= rm.T) { return; }
  let t = lid.x;
  let base = row * rm.H;
  var s: f32 = 0.0;
  var i: u32 = t;
  loop {
    if (i >= rm.H) { break; }
    let v = x[base + i];
    s = s + v * v;
    i = i + 256u;
  }
  red[t] = s;
  workgroupBarrier();
  var k: u32 = 128u;
  loop {
    if (k == 0u) { break; }
    if (t < k) { red[t] = red[t] + red[t + k]; }
    workgroupBarrier();
    k = k / 2u;
  }
  let scale = inverseSqrt(red[0] / f32(rm.H) + bitcast<f32>(rm.epsbits));
  var j: u32 = t;
  loop {
    if (j >= rm.H) { break; }
    outp[base + j] = x[base + j] * scale * w[j];
    j = j + 256u;
  }
}
"""
_rms_k = {"added": False}
_RMS_FUSED = True      # A/B switch for the fused path
_ROPE_FUSED = True     # A/B switch for the fused rope


def rmsnorm(x, w, eps):
    """Fused RMS norm: `x * rsqrt(mean(x^2) + eps) * w`, one dispatch instead of six.

    Returns None when there is no WebGPU backend, so callers keep their own expression.
    Inference-only: no autograd node is built, which is why the graph path stays intact.
    """
    if not _adam_backend_ready():
        return None
    xd = x.data if isinstance(x, Tensor) else x
    wd = w.data if isinstance(w, Tensor) else w
    shape = tuple(xd.shape)
    H = int(shape[-1])
    T = 1
    for d in shape[:-1]:
        T *= int(d)
    plat = _adam_kernel["platform"]
    if not _rms_k["added"]:
        plat.addKernel("rmsnorm", {"source": _RMS_WGSL,
                                   "bindingTypes": ["read-only-storage", "read-only-storage",
                                                    "storage", "read-only-storage"]})
        _rms_k["added"] = True
    xd = _contig(xd)
    of = _empty((T, H))
    ebits = int(np.float32(eps).view(np.uint32))
    meta = _adam_kernel["make_meta"]((T, H, ebits), "u4,u4,u4")
    plat.runKernel({"name": "rmsnorm",
                    "tensors": [xd.buffer.buffer_id, _contig(wd).buffer.buffer_id,
                                of.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": T, "y": 1, "z": 1}})
    return Tensor(of.reshape(*shape))


# Rope written as `x*cos + rotate_half(x)*sin` is about eight dispatches per tensor: two or
# three slices, a negation, a concat, then two multiplies and an add. Twice per layer across
# a deep model that is the largest single block of launches in a decode step. Fused, it is
# one. Decode only: there cos/sin are a single row indexed by position within the head, so
# the mapping is just `i % HD`.
_ROPE_WGSL = """@group(0) @binding(0)
var<storage,read> x: array<f32>;
@group(0) @binding(1)
var<storage,read> cosb: array<f32>;
@group(0) @binding(2)
var<storage,read> sinb: array<f32>;
@group(0) @binding(3)
var<storage,read_write> outp: array<f32>;
struct PMeta { n: u32, HD: u32, rd: u32, }
@group(0) @binding(4)
var<storage,read> pm: PMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= pm.n) { return; }
  let d = i % pm.HD;
  let half = pm.rd / 2u;
  var rot: f32;
  if (d < half) {
    rot = -x[i + half];
  } else if (d < pm.rd) {
    rot = x[i - half];
  } else {
    rot = x[i];                      // pass-through tail: sin is 0 here, value is inert
  }
  outp[i] = x[i] * cosb[d] + rot * sinb[d];
}
"""
_rope_k = {"added": False}


def rope_decode(x, cos, sin, HD, rd):
    """Fused rotary embedding for a single position: `x*cos + rotate_half(x)*sin`.

    Returns None without a WebGPU backend so callers keep their expression. `cos`/`sin` are
    one row of length HD. `rd` is the rotated prefix for partial rope; the tail passes
    through unchanged, matching the unfused form.
    """
    if not _adam_backend_ready():
        return None
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    shape = tuple(xd.shape)
    n = 1
    for d in shape:
        n *= int(d)
    plat = _adam_kernel["platform"]
    if not _rope_k["added"]:
        plat.addKernel("rope", {"source": _ROPE_WGSL,
                                "bindingTypes": ["read-only-storage"] * 3 + ["storage",
                                                                             "read-only-storage"]})
        _rope_k["added"] = True
    cd = _contig(cos.data if isinstance(cos, Tensor) else cos)
    sd = _contig(sin.data if isinstance(sin, Tensor) else sin)
    of = _empty((n,))
    meta = _adam_kernel["make_meta"]((n, int(HD), int(rd)), "u4,u4,u4")
    plat.runKernel({"name": "rope",
                    "tensors": [xd.buffer.buffer_id, cd.buffer.buffer_id, sd.buffer.buffer_id,
                                of.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return Tensor(of.reshape(*shape))


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
        # Gather the rows directly. Selecting them with a one-hot matmul costs O(M*vocab)
        # and builds a (vocab, vocab) identity first -- at a 152k vocab that is terabytes,
        # so this path could never run a real model. The backward is the matching
        # scatter-add, which also handles a token repeated in the batch.
        ids = flat.astype(np.int64)
        out = Tensor(weight.data[ids].reshape(*(ish + (dim,))),
                     weight.requires_grad, (weight,), "embedding")

        def _bw():
            if weight.requires_grad:
                dw = np.zeros_like(weight.data)
                np.add.at(dw, ids, _contig(out.grad.reshape(-1, dim)))
                weight._accum(dw)
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
# ---- native ggml weights: matmul that reads GGUF's own encodings --------------------
# Converting a GGUF into the kernel's packed format is what makes a load slow: on a 27B it
# is 23 minutes, of which reading the 12.2 GB is 13 seconds and uploading it 19. All the
# rest is the host dequantizing ggml blocks to fp32 and requantizing them -- work that also
# quantizes twice, refitting ggml's scheme onto this one and losing a little accuracy.
#
# So do what llama.cpp does and decode inside the matmul. The packed bytes go to the GPU
# exactly as they sit in the file, and a shader unpacks each block on the fly.
#
# One framework serves every ggml type: a thread owns one output column `n` and four rows of
# the batch, walks that weight row's blocks, and hands each decoded value to ACC(). Only the
# decode differs per type, so a new format is a fragment, not a kernel. ACC and the
# accumulator live in `private` storage because a WGSL function cannot take a pointer to a
# local array.
_GGML_BIND = """@group(0) @binding(0)
var<storage,read> x: array<f32>;
@group(0) @binding(1)
var<storage,read> w: array<u32>;
@group(0) @binding(2)
var<storage,read_write> outp: array<f32>;
struct GM { M: u32, N: u32, K: u32, rowb: u32, }
@group(0) @binding(3)
var<storage,read> gm: GM;
// Byte addressing, not word: most ggml blocks are not a multiple of four bytes (Q8_0 is 34,
// Q3_K and IQ3_S 110, MXFP4 17), so a u32 index would drift out of alignment after the
// first block. These mirror the reference dequantizer's own byte offsets exactly.
var<private> nrow: u32;
fn W(wo: u32) -> u32 { return w[wo * gm.N + nrow]; }
fn B(o: u32) -> u32 { return (W(o >> 2u) >> ((o & 3u) * 8u)) & 255u; }
fn I8(o: u32) -> f32 { return f32(i32(B(o) << 24u) >> 24u); }
fn U16(o: u32) -> u32 { return B(o) | (B(o + 1u) << 8u); }
fn U32(o: u32) -> u32 { return U16(o) | (U16(o + 2u) << 16u); }
fn F16(o: u32) -> f32 { return HF(U16(o)); }
fn HF(h: u32) -> f32 {
  let m = h & 1023u;
  let e = (h >> 10u) & 31u;
  var v: f32;
  if (e == 0u) { v = f32(m) * 5.9604644775390625e-8; }
  else if (e == 31u) { v = 65504.0; }
  else { v = exp2(f32(i32(e) - 15)) * (1.0 + f32(m) * 0.0009765625); }
  if ((h & 32768u) != 0u) { return -v; }
  return v;
}
"""

# GEMM (prefill): one thread owns an output column and four rows of the batch, and reads
# activations straight from global memory -- with a batch to amortize, there is enough work
# in flight to hide that.
_GGML_GEMM_PRE = """
var<workgroup> psum: array<f32, KSGx256>;
var<private> a0: f32;
var<private> a1: f32;
var<private> a2: f32;
var<private> a3: f32;
var<private> mb: u32;
var<private> mn: u32;
// Separate scalars, not array<f32,4>. A private array indexed by a loop variable is not
// guaranteed to live in registers, and here it did not: the batched matmul cost time
// strictly proportional to the batch (12.3 / 35.5 / 94.2 ms at M = 8 / 24 / 64) because
// every accumulate went to memory. Widening the rows per thread made it worse, which is
// what ruled out the decode being the expensive part. gemv was always fast for the same
// reason in reverse -- it accumulates into one scalar.
fn ACC(k: u32, v: f32) {
  let b = mb * gm.K + k;
  a0 = a0 + x[b] * v;
  if (mn > 1u) { a1 = a1 + x[b + gm.K] * v; }
  if (mn > 2u) { a2 = a2 + x[b + 2u * gm.K] * v; }
  if (mn > 3u) { a3 = a3 + x[b + 3u * gm.K] * v; }
}
fn ACC4(k: u32, v: vec4<f32>) {
  let b = mb * gm.K + k;
  a0 = a0 + dot(vec4<f32>(x[b], x[b + 1u], x[b + 2u], x[b + 3u]), v);
  if (mn > 1u) { let c = b + gm.K;
    a1 = a1 + dot(vec4<f32>(x[c], x[c + 1u], x[c + 2u], x[c + 3u]), v); }
  if (mn > 2u) { let c = b + 2u * gm.K;
    a2 = a2 + dot(vec4<f32>(x[c], x[c + 1u], x[c + 2u], x[c + 3u]), v); }
  if (mn > 3u) { let c = b + 3u * gm.K;
    a3 = a3 + dot(vec4<f32>(x[c], x[c + 1u], x[c + 2u], x[c + 3u]), v); }
}
"""

_GGML_GEMM_MAIN = """
@compute @workgroup_size(64, KSGu)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let lx = lid.x;
  let ly = lid.y;
  mb = gid.z * 4u;
  mn = select(0u, min(4u, gm.M - mb), mb < gm.M);
  nrow = n;
  a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0;
  let base = 0u;
  let nb = gm.K / BLKVALS;
  let livec = (n < gm.N && mn > 0u);
  if (livec) {
    for (var b: u32 = ly; b < nb; b = b + KSGu) {
"""

_GGML_GEMM_TAIL = """
    }
  }
  let pb = (ly * 64u + lx) * 4u;
  psum[pb] = a0; psum[pb + 1u] = a1; psum[pb + 2u] = a2; psum[pb + 3u] = a3;
  workgroupBarrier();
  if (ly == 0u && livec) {
    for (var r: u32 = 0u; r < mn; r = r + 1u) {
      var tot: f32 = 0.0;
      for (var i: u32 = 0u; i < KSGu; i = i + 1u) {
        tot = tot + psum[(i * 64u + lx) * 4u + r];
      }
      outp[(mb + r) * gm.N + n] = tot;
    }
  }
}
"""

_GGML_KSG = 2               # split-K rows on the batched path

# GEMV (decode, batch of one): the shape where the naive kernel loses. Two things fix it,
# both of which ggml blocks happen to suit. Blocks are independent, so KS rows of threads
# can each take every KS-th block and the partial sums are added at the end -- KS times the
# parallelism on a matmul that is otherwise one long serial walk per output column. And all
# 64 threads in a row want the SAME activations, so a block's worth is staged in workgroup
# memory once instead of being re-read from global memory 64 times.
_GGML_GEMV_PRE = """
var<workgroup> xs: array<f32, KSxBLKxR>;
var<workgroup> psum: array<f32, PSUMSZ>;
var<private> accs: array<f32, ORW>;
var<private> acc0: f32;
ACCDECL1
var<private> xoff: u32;
fn ACC(k: u32, v: f32) {
  let i = xoff + (k & MASKBLK);
  acc0 = acc0 + xs[i] * v;
ACCBODY1
}
fn ACC4(k: u32, v: vec4<f32>) {
  let i = xoff + (k & MASKBLK);
  acc0 = acc0 + dot(vec4<f32>(xs[i], xs[i + 1u], xs[i + 2u], xs[i + 3u]), v);
ACC4BODY1
}
"""

_GGML_GEMV_MAIN = """
@compute @workgroup_size(WGX, KSu)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let rowbase = wid.x * (WGXu * ORWu);
  let lx = lid.x;
  let ly = lid.y;
  xoff = ly * BLKVALS;
  acc0 = 0.0;
ACCINIT1
  for (var q: u32 = 0u; q < ORWu; q = q + 1u) { accs[q] = 0.0; }
  let base = 0u;
  let nb = gm.K / BLKVALS;
  // A fixed step count, not `b < nb` per row: every thread must reach the same barriers.
  let steps = (nb + KSu - 1u) / KSu;
  for (var st: u32 = 0u; st < steps; st = st + 1u) {
    let b = st * KSu + ly;
    for (var t: u32 = lx; t < BLKVALS; t = t + WGXu) {
      var xv: f32 = 0.0;
      if (b < nb) { xv = x[b * BLKVALS + t]; }
      xs[xoff + t] = xv;
XLOAD1
    }
    workgroupBarrier();
    for (var orw: u32 = 0u; orw < ORWu; orw = orw + 1u) {
    let nn = rowbase + orw * WGXu + lx;
    nrow = nn;
    acc0 = accs[orw];
    if (b < nb && nn < gm.N) {
"""

_GGML_GEMV_TAIL = """
    }
    accs[orw] = acc0;
    }
    workgroupBarrier();
  }
  for (var q: u32 = 0u; q < ORWu; q = q + 1u) {
    psum[q * KSxWGX + ly * WGXu + lx] = accs[q];
  }
PSUM1
  workgroupBarrier();
  if (ly == 0u) {
    for (var q: u32 = 0u; q < ORWu; q = q + 1u) {
      let nn = rowbase + q * WGXu + lx;
      if (nn < gm.N) {
        var tot: f32 = 0.0;
        for (var i: u32 = 0u; i < KSu; i = i + 1u) {
          tot = tot + psum[q * KSxWGX + i * WGXu + lx];
        }
        outp[nn] = tot;
      }
    }
    // The second row keeps its own fixed psum half and its own bounds check. Moving the
    // `n < gm.N` guard into the loop above left this write unguarded, and every lane past
    // N wrote into the row after it -- which is most of them whenever N is small.
    if (n < gm.N) {
OUT1
    }
  }
}
"""

_GGML_KS = 4                 # split-K rows per workgroup on the decode path
# Output rows per workgroup on the decode path. WGX * _GGML_KS is the workgroup size, so
# these trade against each other at a fixed 256 threads -- and the trade matters, because
# the dispatch is N / WGX workgroups. At 64 a 1024-wide projection filled only 16 of them,
# far too few to occupy the GPU, and the measured cost per matmul was almost independent of
# how many bytes it read (gate, N=3072, ran FASTER than q, N=2048, despite being larger).
# Halving the rows doubles the workgroups and splits K further to keep the threads busy.
_GGML_WGX = 64
# Output rows each lane accumulates. The activations for a block are loaded into workgroup
# memory once and then multiplied against every row the lane owns, so this divides how many
# times the activation vector is re-read across the dispatch -- and that repetition is not
# small: at N=17408 with WGX=64 the 5120-float input was pulled in 272 times over, gigabytes
# per token of pure duplication. Unlike WGX it is not bounded by the 256-thread workgroup
# limit, and it raises instruction-level parallelism at the same time.
#
# It is nonetheless 1, because the saving does not survive contact with the hardware: raising
# it divides the workgroup count by the same factor, and on this GPU the lost occupancy costs
# more than the duplicated reads. Measured on the 27B (64 layers, 12 quant types), whole
# captured decode step: ORW=1 187.9ms, ORW=2 194.4ms, ORW=4 214.8ms. Per-matmul microbenchmarks
# agree -- ORW=1 wins on 22 of the 28 distinct (type, N, K) shapes in that model, and where 2
# wins it is by a few percent, against a 67% win on exactly one shape (IQ2_S at N=5120).
# Outputs are identical across values (checked shape by shape, and on the full logit vector
# from a clean state), so this is purely a performance knob -- worth revisiting on a GPU with
# different occupancy characteristics.
_GGML_ORW = 1

# The IQ4_NL/IQ4_XS codebook. Indexing a `const` array dynamically is legal WGSL but not
# uniformly implemented, so the sixteen signed bytes ride in four u32s, sign-extended on read.
_KV_FN = """
fn kv(i: u32) -> f32 {
  // Branchless: this is called once per decoded value, and an if-chain here showed up as
  // the limit on IQ4_XS once its quants were being read a word at a time.
  let lo = select(0xBFAD9881u, 0xF6EADDCFu, (i & 4u) != 0u);
  let hi = select(0x26190D01u, 0x71594535u, (i & 4u) != 0u);
  let p = select(lo, hi, (i & 8u) != 0u);
  return f32(i32(((p >> (8u * (i & 3u))) & 255u) << 24u) >> 24u);
}
"""

# get_scale_min_k4: Q4_K and Q5_K pack eight 6-bit scales and eight 6-bit mins into 12 bytes.
_K4SC_FN = """
fn k4sc(so: u32, j: u32) -> vec2<f32> {
  if (j < 4u) { return vec2<f32>(f32(B(so + j) & 63u), f32(B(so + j + 4u) & 63u)); }
  let a = B(so + j + 4u);
  return vec2<f32>(f32((a & 15u) | ((B(so + j - 4u) >> 6u) << 4u)),
                   f32((a >> 4u) | ((B(so + j) >> 6u) << 4u)));
}
"""

# Q8_0: 32 values / 34 bytes -- f16 d + int8[32].
_Q8_0_DEC = """
    let o = base + b * 34u;
    let d = F16(o);
    let kb = b * 32u;
    for (var l: u32 = 0u; l < 32u; l = l + 1u) { ACC(kb + l, d * I8(o + 2u + l)); }
"""

# IQ4_NL: 32 values / 18 bytes -- f16 d + 16 bytes of paired codebook indices.
_IQ4NL_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let by = B(o + 2u + j);
      ACC(kb + j, d * kv(by & 15u));
      ACC(kb + 16u + j, d * kv(by >> 4u));
    }
"""

# IQ4_XS: 256 values / 136 bytes -- f16 d | u16 scales_h | u8 scales_l[4] | u8 qs[128].
# Eight 32-value sub-blocks, each with a 6-bit scale split across scales_l and scales_h; low
# nibbles fill a sub-block's first 16 slots, high nibbles the next 16.
_IQ4XS_DEC = """
    let o = base + b * 136u;
    let d = F16(o);
    let sh = U16(o + 2u);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let ls = ((B(o + 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4u);
      let dl = d * f32(i32(ls) - 32);
      let qwb = o + 8u + ib * 16u;
      let k0 = kb + ib * 32u;
      // The four quants in a word land on four CONSECUTIVE activations, so each half of the
      // byte pair is one dot product rather than four scalar accumulates -- same arithmetic,
      // a quarter of the workgroup-memory reads.
      //
      // Worth 1.5x when the weight is already in cache, and NOTHING in a real decode step:
      // 64 layers of distinct weights stream past, the ALU hides entirely behind memory
      // latency, and an MLP-only capture measured 113.7ms before and 113.2ms after. Kept
      // because it is strictly less work and may matter where the working set does fit,
      // but do not expect it to move a large model.
      for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
        let w4 = W((qwb >> 2u) + jw);
        let j = jw * 4u;
        let c0 = w4 & 255u; let c1 = (w4 >> 8u) & 255u;
        let c2 = (w4 >> 16u) & 255u; let c3 = (w4 >> 24u) & 255u;
        ACC4(k0 + j, vec4<f32>(dl * kv(c0 & 15u), dl * kv(c1 & 15u),
                               dl * kv(c2 & 15u), dl * kv(c3 & 15u)));
        ACC4(k0 + 16u + j, vec4<f32>(dl * kv(c0 >> 4u), dl * kv(c1 >> 4u),
                                     dl * kv(c2 >> 4u), dl * kv(c3 >> 4u)));
      }
    }
"""

# Q4_K: 256 values / 144 bytes -- f16 d | f16 dmin | scales[12] | qs[128].
_Q4K_DEC = """
    let o = base + b * 144u;
    let d = F16(o); let dmin = F16(o + 2u);
    let so = o + 4u; let qo = o + 16u;
    let kb = b * 256u;
    for (var g: u32 = 0u; g < 4u; g = g + 1u) {
      let i0 = 2u * g;
      let s1 = k4sc(so, i0); let s2 = k4sc(so, i0 + 1u);
      let d1 = d * s1.x; let m1 = dmin * s1.y;
      let d2 = d * s2.x; let m2 = dmin * s2.y;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let q = B(qo + g * 32u + l);
        ACC(kb + i0 * 32u + l, d1 * f32(q & 15u) - m1);
        ACC(kb + (i0 + 1u) * 32u + l, d2 * f32(q >> 4u) - m2);
      }
    }
"""

# Q5_K: 256 values / 176 bytes -- f16 d | f16 dmin | scales[12] | qh[32] | qs[128].
# qh carries each value's fifth bit, one bit per sub-block index.
_Q5K_DEC = """
    let o = base + b * 176u;
    let d = F16(o); let dmin = F16(o + 2u);
    let so = o + 4u; let ho = o + 16u; let qo = o + 48u;
    let kb = b * 256u;
    for (var g: u32 = 0u; g < 4u; g = g + 1u) {
      let i0 = 2u * g;
      let s1 = k4sc(so, i0); let s2 = k4sc(so, i0 + 1u);
      let d1 = d * s1.x; let m1 = dmin * s1.y;
      let d2 = d * s2.x; let m2 = dmin * s2.y;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let q = B(qo + g * 32u + l);
        let h = B(ho + l);
        let lo = f32(q & 15u) + select(0.0, 16.0, (h & (1u << i0)) != 0u);
        let hi = f32(q >> 4u) + select(0.0, 16.0, (h & (1u << (i0 + 1u))) != 0u);
        ACC(kb + i0 * 32u + l, d1 * lo - m1);
        ACC(kb + (i0 + 1u) * 32u + l, d2 * hi - m2);
      }
    }
"""

# Q6_K: 256 values / 210 bytes -- ql[128] | qh[64] | int8 scales[16] | f16 d.
_Q6K_DEC = """
    let o = base + b * 210u;
    let d = F16(o + 208u);
    let kb = b * 256u;
    for (var half: u32 = 0u; half < 2u; half = half + 1u) {
      let lo = o + half * 64u;
      let ho = o + 128u + half * 32u;
      let so = o + 192u + half * 8u;
      let k0 = kb + half * 128u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let ii = l >> 4u;
        let a = B(lo + l); let c = B(lo + l + 32u); let h = B(ho + l);
        ACC(k0 + l,       d * I8(so + ii)      * (f32((a & 15u) | (((h >> 0u) & 3u) << 4u)) - 32.0));
        ACC(k0 + l + 32u, d * I8(so + ii + 2u) * (f32((c & 15u) | (((h >> 2u) & 3u) << 4u)) - 32.0));
        ACC(k0 + l + 64u, d * I8(so + ii + 4u) * (f32((a >> 4u)  | (((h >> 4u) & 3u) << 4u)) - 32.0));
        ACC(k0 + l + 96u, d * I8(so + ii + 6u) * (f32((c >> 4u)  | (((h >> 6u) & 3u) << 4u)) - 32.0));
      }
    }
"""

# Q3_K: 256 values / 110 bytes -- hmask[32] | qs[64] | scales[12] | f16 d.
# The sixteen 6-bit scales are split across three u32s and reassembled four at a time;
# hmask supplies a per-value bit that shifts the 2-bit quant from [-4,-1] to [0,3].
_Q3K_DEC = """
    let o = base + b * 110u;
    let d = F16(o + 108u);
    let a0 = U32(o + 96u); let a1 = U32(o + 100u); let a2 = U32(o + 104u);
    let kb = b * 256u;
    var k: u32 = 0u; var is: u32 = 0u;
    for (var blk2: u32 = 0u; blk2 < 2u; blk2 = blk2 + 1u) {
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let shift = 2u * j;
        let mbit = 1u << (blk2 * 4u + j);
        for (var half: u32 = 0u; half < 2u; half = half + 1u) {
          var v: u32;
          let wsel = is >> 2u;
          if (wsel == 0u) { v = (a0 & 0x0F0F0F0Fu) | (((a2 >> 0u) & 0x03030303u) << 4u); }
          else if (wsel == 1u) { v = (a1 & 0x0F0F0F0Fu) | (((a2 >> 2u) & 0x03030303u) << 4u); }
          else if (wsel == 2u) { v = ((a0 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 4u) & 0x03030303u) << 4u); }
          else { v = ((a1 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 6u) & 0x03030303u) << 4u); }
          let sc = f32(i32(((v >> (8u * (is & 3u))) & 255u) << 24u) >> 24u) - 32.0;
          let dl = d * sc;
          let qo = o + 32u + blk2 * 32u + half * 16u;
          let mo = o + half * 16u;
          for (var l: u32 = 0u; l < 16u; l = l + 1u) {
            let q = f32((B(qo + l) >> shift) & 3u);
            ACC(kb + k + l, dl * (q - select(4.0, 0.0, (B(mo + l) & mbit) != 0u)));
          }
          k = k + 16u; is = is + 1u;
        }
      }
    }
"""

# Q2_K: 256 values / 84 bytes -- scales[16] (4-bit scale + 4-bit min) | qs[64] | f16 d | f16 dmin.
_Q2K_DEC = """
    let o = base + b * 84u;
    let d = F16(o + 80u); let dmin = F16(o + 82u);
    let kb = b * 256u;
    var k: u32 = 0u; var is: u32 = 0u;
    for (var blk2: u32 = 0u; blk2 < 2u; blk2 = blk2 + 1u) {
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let shift = 2u * j;
        for (var half: u32 = 0u; half < 2u; half = half + 1u) {
          let sc = B(o + is);
          let dl = d * f32(sc & 15u); let ml = dmin * f32(sc >> 4u);
          let qo = o + 16u + blk2 * 32u + half * 16u;
          for (var l: u32 = 0u; l < 16u; l = l + 1u) {
            ACC(kb + k + l, dl * f32((B(qo + l) >> shift) & 3u) - ml);
          }
          k = k + 16u; is = is + 1u;
        }
      }
    }
"""

# The i-quants store INDICES into ggml's codebook grids rather than values, so the shader
# needs the grids too. They are 1-16 KB -- too big to inline as WGSL constants, and dynamic
# indexing of a `const` array is unevenly supported anyway -- so each type gets one extra
# storage binding: ksigns in the first 128 bytes, the grid from byte 128 on.
_GRID_FN = """
@group(0) @binding(4)
var<storage,read> gr: array<u32>;
fn GB(o: u32) -> u32 { return (gr[o >> 2u] >> ((o & 3u) * 8u)) & 255u; }
fn G4(idx: u32) -> u32 { return gr[32u + idx]; }        // one 4-byte entry, one read
fn G4V(idx: u32) -> vec4<f32> { return unpack4x8unorm(gr[32u + idx]) * 255.0; }
fn SGN4(mask: u32, j0: u32) -> vec4<f32> {
  return vec4<f32>(SGN(mask, j0), SGN(mask, j0 + 1u), SGN(mask, j0 + 2u), SGN(mask, j0 + 3u));
}
fn BY(w: u32, q: u32) -> f32 { return f32((w >> (8u * q)) & 255u); }
fn GI8(o: u32) -> f32 { return f32(i32(GB(o) << 24u) >> 24u); }
fn SGN(mask: u32, j: u32) -> f32 { return select(1.0, -1.0, (mask & (1u << j)) != 0u); }
"""

# IQ2_XXS: 256 values / 66 bytes -- f16 d | 8 sub-blocks of 8 bytes. Each sub-block holds
# four grid indices in its low four bytes; the high u32 carries four 7-bit sign codes and,
# in its top nibble, the sub-block scale.
_IQ2XXS_DEC = """
    let o = base + b * 66u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let ao = o + 2u + ib * 8u;
      let a1 = U32(ao + 4u);
      let db = d * (0.5 + f32(a1 >> 28u)) * 0.25;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let gx = B(ao + l) * 2u;
        let ga = G4(gx); let gb = G4(gx + 1u);
        let sm = GB((a1 >> (7u * l)) & 127u);
        let k0 = kb + ib * 32u + l * 8u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          ACC(k0 + j, db * BY(ga, j) * SGN(sm, j));
          ACC(k0 + 4u + j, db * BY(gb, j) * SGN(sm, 4u + j));
        }
      }
    }
"""

# IQ2_XS: 256 values / 74 bytes -- f16 d | u16 qs[32] | u8 scales[8]. Each u16 is a 9-bit
# grid index plus a 7-bit sign code; the two nibbles of a scale byte cover l=0,1 and l=2,3.
_IQ2XS_DEC = """
    let o = base + b * 74u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let sc = B(o + 66u + ib);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let q = U16(o + 2u + ib * 8u + l * 2u);
        let nib = select(sc & 15u, sc >> 4u, l >= 2u);
        let db = d * (0.5 + f32(nib)) * 0.25;
        let gx = (q & 511u) * 2u;
        let ga = G4(gx); let gb = G4(gx + 1u);
        let sm = GB(q >> 9u);
        let k0 = kb + ib * 32u + l * 8u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          ACC(k0 + j, db * BY(ga, j) * SGN(sm, j));
          ACC(k0 + 4u + j, db * BY(gb, j) * SGN(sm, 4u + j));
        }
      }
    }
"""

# IQ2_S: 256 values / 82 bytes -- f16 d | qs[32] | signs[32] | qh[8] | scales[8]. The grid
# index is 8 bits from qs plus 2 from qh, and the sign byte is used directly as a mask
# rather than as an index into ksigns.
_IQ2S_DEC = """
    let o = base + b * 82u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let sc = B(o + 74u + ib);
      let qh = B(o + 66u + ib);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let nib = select(sc & 15u, sc >> 4u, l >= 2u);
        let db = d * (0.5 + f32(nib)) * 0.25;
        let gx = (B(o + 2u + ib * 4u + l) | ((qh << (8u - 2u * l)) & 768u)) * 2u;
        let ga = G4(gx); let gb = G4(gx + 1u);
        let sm = B(o + 34u + ib * 4u + l);
        let k0 = kb + ib * 32u + l * 8u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          ACC(k0 + j, db * BY(ga, j) * SGN(sm, j));
          ACC(k0 + 4u + j, db * BY(gb, j) * SGN(sm, 4u + j));
        }
      }
    }
"""

# IQ3_XXS: 256 values / 98 bytes -- f16 d | qs[64] | u32 aux[8]. Eight grid indices per
# sub-block, four bytes each, so the 32 decoded values regroup into four sign-groups of 8.
_IQ3XXS_DEC = """
    let o = base + b * 98u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let a1 = U32(o + 66u + ib * 4u);
      let db = d * (0.5 + f32(a1 >> 28u)) * 0.5;
      let k0 = kb + ib * 32u;
      for (var p: u32 = 0u; p < 8u; p = p + 1u) {
        let f0 = p * 4u;
        let sm = GB((a1 >> (7u * (f0 >> 3u))) & 127u);
        ACC4(k0 + f0, G4V(B(o + 2u + ib * 8u + p)) * SGN4(sm, f0 & 7u) * db);
      }
    }
"""

# IQ3_S: 256 values / 110 bytes -- f16 d | qs[64] | qh[8] | signs[32] | scales[4]. qh adds a
# ninth bit to each grid index (shift 8-p for both even and odd slots), and like IQ2_S the
# sign byte is a mask, not a ksigns index.
_IQ3S_DEC = """
    let o = base + b * 110u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let scb = B(o + 106u + (ib >> 1u));
      let nib = select(scb & 15u, scb >> 4u, (ib & 1u) != 0u);
      let db = d * (1.0 + 2.0 * f32(nib));
      let qh = B(o + 66u + ib);
      let k0 = kb + ib * 32u;
      // Two consecutive p share one sign byte (f0 is a multiple of four, and f0 >> 3 is
      // p >> 1), so read it once for the pair instead of once per value. Same caveat as
      // IQ4_XS above: 1.5x on a cached weight, no change in a real step.
      let sb = o + 74u + ib * 4u;
      let qb = o + 2u + ib * 8u;
      for (var pg: u32 = 0u; pg < 4u; pg = pg + 1u) {
        let sm = B(sb + pg);
        let p0 = pg * 2u;
        let p1 = p0 + 1u;
        ACC4(k0 + p0 * 4u, G4V(B(qb + p0) | ((qh << (8u - p0)) & 256u))
                           * SGN4(sm, 0u) * db);
        ACC4(k0 + p1 * 4u, G4V(B(qb + p1) | ((qh << (8u - p1)) & 256u))
                           * SGN4(sm, 4u) * db);
      }
    }
"""

# IQ1_S: 256 values / 50 bytes -- f16 d | qs[32] | u16 qh[8]. qh carries the sub-block scale,
# a shared +/-0.125 offset, and three extra bits for each of the four grid indices. The grid
# entries are signed bytes here, unlike the IQ2/IQ3 grids.
_IQ1S_DEC = """
    let o = base + b * 50u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let qh = U16(o + 34u + ib * 2u);
      let dl = d * (2.0 * f32((qh >> 12u) & 7u) + 1.0);
      let delta = select(0.125, -0.125, (qh & 32768u) != 0u);
      let k0 = kb + ib * 32u;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let gi = 128u + (B(o + 2u + ib * 4u + l) | (((qh >> (3u * l)) & 7u) << 8u)) * 8u;
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
          ACC(k0 + l * 8u + j, dl * (GI8(gi + j) + delta));
        }
      }
    }
"""

# IQ1_M: 256 values / 56 bytes -- qs[32] | qh[16] | u16 scales[4]. There is no d field: the
# block scale is assembled from the four scale words' spare nibbles. Each sub-block has two
# 3-bit half-scales, and each pair of grid indices takes its extra bits and sign offset from
# one qh byte.
_IQ1M_DEC = """
    let o = base + b * 56u;
    let s0 = U16(o + 48u); let s1 = U16(o + 50u);
    let s2 = U16(o + 52u); let s3 = U16(o + 54u);
    let d = HF((s0 >> 12u) | ((s1 >> 8u) & 240u) | ((s2 >> 4u) & 3840u) | (s3 & 61440u));
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      var sw: u32 = s0;
      if (ib >= 6u) { sw = s3; } else if (ib >= 4u) { sw = s2; } else if (ib >= 2u) { sw = s1; }
      let sh = 6u * (ib & 1u);
      let dl1 = d * (2.0 * f32((sw >> sh) & 7u) + 1.0);
      let dl2 = d * (2.0 * f32((sw >> (sh + 3u)) & 7u) + 1.0);
      let k0 = kb + ib * 32u;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let qhb = B(o + 32u + ib * 2u + (l >> 1u));
        let gi = 128u + (B(o + ib * 4u + l) | ((qhb << (8u - 4u * (l & 1u))) & 1792u)) * 8u;
        let dbit = select(8u, 128u, (l & 1u) != 0u);
        let delta = select(0.125, -0.125, (qhb & dbit) != 0u);
        let dl = select(dl1, dl2, l >= 2u);
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
          ACC(k0 + l * 8u + j, dl * (GI8(gi + j) + delta));
        }
      }
    }
"""

# Q4_0: 32 values / 18 bytes -- f16 d + 16 nibble pairs, quants centred on 8.
_Q4_0_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 2u + j);
      ACC(kb + j, d * (f32(q & 15u) - 8.0));
      ACC(kb + 16u + j, d * (f32(q >> 4u) - 8.0));
    }
"""

# Q4_1: 32 values / 20 bytes -- f16 d, f16 min, 16 nibble pairs.
_Q4_1_DEC = """
    let o = base + b * 20u;
    let d = F16(o); let mn = F16(o + 2u);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 4u + j);
      ACC(kb + j, d * f32(q & 15u) + mn);
      ACC(kb + 16u + j, d * f32(q >> 4u) + mn);
    }
"""

# Q5_0: 32 values / 22 bytes -- f16 d, u32 qh holding each value's fifth bit, 16 nibble pairs.
_Q5_0_DEC = """
    let o = base + b * 22u;
    let d = F16(o);
    let qh = U32(o + 2u);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 6u + j);
      ACC(kb + j, d * (f32((q & 15u) | (((qh >> j) << 4u) & 16u)) - 16.0));
      ACC(kb + 16u + j, d * (f32((q >> 4u) | ((qh >> (j + 12u)) & 16u)) - 16.0));
    }
"""

# Q5_1: 32 values / 24 bytes -- f16 d, f16 min, u32 qh, 16 nibble pairs.
_Q5_1_DEC = """
    let o = base + b * 24u;
    let d = F16(o); let mn = F16(o + 2u);
    let qh = U32(o + 4u);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 8u + j);
      ACC(kb + j, d * f32((q & 15u) | (((qh >> j) << 4u) & 16u)) + mn);
      ACC(kb + 16u + j, d * f32((q >> 4u) | ((qh >> (j + 12u)) & 16u)) + mn);
    }
"""

# F16 / F32: not quantized at all, but going through the same kernel keeps an unquantized
# tensor on the no-conversion path instead of sending it back through the fp32 expansion.
_F16_DEC = """
    ACC(b, F16(base + b * 2u));
"""

_F32_DEC = """
    ACC(b, bitcast<f32>(U32(base + b * 4u)));
"""

# E2M1 values (doubled) -- ggml's kvalues_fp4, shared by MXFP4 and NVFP4 -- plus the two
# scale encodings those formats use. Packed into u32s for the same reason as the IQ4 table.
_FP4_FN = """
fn fp4(i: u32) -> f32 {
  let lo = select(0x03020100u, 0x0C080604u, (i & 4u) != 0u);
  let hi = select(0xFDFEFF00u, 0xF4F8FAFCu, (i & 4u) != 0u);
  let p = select(lo, hi, (i & 8u) != 0u);
  return f32(i32(((p >> (8u * (i & 3u))) & 255u) << 24u) >> 24u);
}
fn e8m0h(e: u32) -> f32 {
  if (e < 2u) { return bitcast<f32>(0x00200000u << e); }
  return bitcast<f32>((e - 1u) << 23u);
}
fn ue4m3(v: u32) -> f32 {
  if (v == 0u || v == 127u) { return 0.0; }
  let e = (v >> 3u) & 15u;
  let m = f32(v & 7u);
  if (e == 0u) { return m * exp2(-9.0) * 0.5; }
  return (1.0 + m * 0.125) * exp2(f32(i32(e) - 7)) * 0.5;
}
"""

_POW3_FN = """
fn pow3(n: u32) -> u32 {
  var p: u32 = 1u;
  for (var i: u32 = 0u; i < n; i = i + 1u) { p = p * 3u; }
  return p;
}
"""

# BF16: the top 16 bits of an fp32, so widening is a shift.
_BF16_DEC = """
    ACC(b, bitcast<f32>(U16(base + b * 2u) << 16u));
"""

# TQ1_0: 256 values / 54 bytes -- qs[48] | qh[4] | f16 d. Ternary, five values packed per
# byte: multiply the byte by a power of three (mod 256) and the top of byte*3 is the trit.
_TQ1_0_DEC = """
    let o = base + b * 54u;
    let d = F16(o + 52u);
    let kb = b * 256u;
    var k: u32 = 0u;
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
      let jo = g * 32u;
      let cnt = select(32u, 16u, g == 1u);
      for (var p: u32 = 0u; p < 5u; p = p + 1u) {
        let p3 = pow3(p);
        for (var m: u32 = 0u; m < cnt; m = m + 1u) {
          let q = (B(o + jo + m) * p3) & 255u;
          ACC(kb + k + m, (f32((q * 3u) >> 8u) - 1.0) * d);
        }
        k = k + cnt;
      }
    }
    for (var p: u32 = 0u; p < 4u; p = p + 1u) {      // the four qh bytes, four values each
      let p3 = pow3(p);
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let q = (B(o + 48u + j) * p3) & 255u;
        ACC(kb + k + j, (f32((q * 3u) >> 8u) - 1.0) * d);
      }
      k = k + 4u;
    }
"""

# TQ2_0: 256 values / 66 bytes -- qs[64] | f16 d. Two bits per value, one bit-plane at a
# time across each 32-byte group.
_TQ2_0_DEC = """
    let o = base + b * 66u;
    let d = F16(o + 64u);
    let kb = b * 256u;
    var k: u32 = 0u;
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
      let jo = g * 32u;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        for (var m: u32 = 0u; m < 32u; m = m + 1u) {
          ACC(kb + k + m, (f32((B(o + jo + m) >> (l * 2u)) & 3u) - 1.0) * d);
        }
        k = k + 32u;
      }
    }
"""

# MXFP4: 32 values / 17 bytes -- one E8M0 exponent byte then 16 nibble pairs.
_MXFP4_DEC = """
    let o = base + b * 17u;
    let d = e8m0h(B(o));
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 1u + j);
      ACC(kb + j, fp4(q & 15u) * d);
      ACC(kb + 16u + j, fp4(q >> 4u) * d);
    }
"""

# NVFP4: 64 values / 36 bytes -- four UE4M3 scales, one per 16-value sub-block, then 32
# nibble pairs.
_NVFP4_DEC = """
    let o = base + b * 36u;
    let kb = b * 64u;
    for (var s: u32 = 0u; s < 4u; s = s + 1u) {
      let d = ue4m3(B(o + s));
      let k0 = kb + s * 16u;
      for (var j: u32 = 0u; j < 8u; j = j + 1u) {
        let q = B(o + 4u + s * 8u + j);
        ACC(k0 + j, fp4(q & 15u) * d);
        ACC(k0 + 8u + j, fp4(q >> 4u) * d);
      }
    }
"""

# Q1_0: 128 values / 18 bytes -- f16 d and one bit per value, +d or -d.
_Q1_0_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 128u;
    for (var j: u32 = 0u; j < 128u; j = j + 1u) {
      ACC(kb + j, select(-d, d, ((B(o + 2u + (j >> 3u)) >> (j & 7u)) & 1u) != 0u));
    }
"""

# Q2_0: 64 values / 18 bytes -- f16 d and two bits per value, 00=-1 01=0 10=+1 11=+2.
_Q2_0_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 64u;
    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
      ACC(kb + j, (f32((B(o + 2u + (j >> 2u)) >> ((j & 3u) * 2u)) & 3u) - 1.0) * d);
    }
"""

# name -> (decode fragment, helper functions, values per block, bytes per block, codebook)
_GGML_TYPES = {
    "F32":     (_F32_DEC,     "",                   1,   4, None),
    "F16":     (_F16_DEC,     "",                   1,   2, None),
    "BF16":    (_BF16_DEC,    "",                   1,   2, None),
    "TQ1_0":   (_TQ1_0_DEC,   _POW3_FN,           256,  54, None),
    "TQ2_0":   (_TQ2_0_DEC,   "",                 256,  66, None),
    "MXFP4":   (_MXFP4_DEC,   _FP4_FN,             32,  17, None),
    "NVFP4":   (_NVFP4_DEC,   _FP4_FN,             64,  36, None),
    "Q1_0":    (_Q1_0_DEC,    "",                 128,  18, None),
    "Q2_0":    (_Q2_0_DEC,    "",                  64,  18, None),
    "Q4_0":    (_Q4_0_DEC,    "",                  32,  18, None),
    "Q4_1":    (_Q4_1_DEC,    "",                  32,  20, None),
    "Q5_0":    (_Q5_0_DEC,    "",                  32,  22, None),
    "Q5_1":    (_Q5_1_DEC,    "",                  32,  24, None),
    "Q8_0":    (_Q8_0_DEC,    "",                  32,  34, None),
    "IQ4_NL":  (_IQ4NL_DEC,   _KV_FN,              32,  18, None),
    "IQ4_XS":  (_IQ4XS_DEC,   _KV_FN,             256, 136, None),
    "Q4_K":    (_Q4K_DEC,     _K4SC_FN,           256, 144, None),
    "Q5_K":    (_Q5K_DEC,     _K4SC_FN,           256, 176, None),
    "Q6_K":    (_Q6K_DEC,     "",                 256, 210, None),
    "Q3_K":    (_Q3K_DEC,     "",                 256, 110, None),
    "Q2_K":    (_Q2K_DEC,     "",                 256,  84, None),
    "IQ2_XXS": (_IQ2XXS_DEC,  _GRID_FN,           256,  66, "IQ2XXS_GRID_U8"),
    "IQ2_XS":  (_IQ2XS_DEC,   _GRID_FN,           256,  74, "IQ2XS_GRID_U8"),
    "IQ2_S":   (_IQ2S_DEC,    _GRID_FN,           256,  82, "IQ2S_GRID_U8"),
    "IQ3_XXS": (_IQ3XXS_DEC,  _GRID_FN,           256,  98, "IQ3XXS_GRID_U8"),
    "IQ3_S":   (_IQ3S_DEC,    _GRID_FN,           256, 110, "IQ3S_GRID_U8"),
    "IQ1_S":   (_IQ1S_DEC,    _GRID_FN,           256,  50, "IQ1S_GRID_I8"),
    "IQ1_M":   (_IQ1M_DEC,    _GRID_FN,           256,  56, "IQ1S_GRID_I8"),
}
_ggml_grids = {}
_ggml_k = {"added": set()}
_NATIVE_GGUF = True          # default on; the loader's `weights=` overrides


def ggml_native_supported(type_name):
    """Can this ggml type be multiplied straight out of the file, with no conversion?"""
    return bool(_NATIVE_GGUF) and type_name in _GGML_TYPES and _adam_backend_ready()


def _orw_for(mode=1):
    """Output rows per lane for this path. Only the single-token decode kernel uses more
    than one; the two-row variant addresses psum with its own fixed layout."""
    return _GGML_ORW if (mode == 1 and _GGML_ORW > 1) else 1


def _gemv_groups(N, mode=1):
    """Workgroups needed to cover N output rows on the decode path."""
    per = _GGML_WGX * _orw_for(mode)
    return (int(N) + per - 1) // per


def _ggml_src(type_name, mode):
    """`mode`: 1 or 2 for the decode kernel with that many rows, 0 for the batched one."""
    dec, helpers, vals, _, _ = _GGML_TYPES[type_name]
    pre, main, tail = ((_GGML_GEMV_PRE, _GGML_GEMV_MAIN, _GGML_GEMV_TAIL) if mode
                       else (_GGML_GEMM_PRE, _GGML_GEMM_MAIN, _GGML_GEMM_TAIL))
    # helpers are functions, so they go BEFORE main -- WGSL has no nested functions.
    src = _GGML_BIND + helpers + pre + main + dec + tail
    rows = max(1, mode)
    two = rows == 2
    # Multi-row accumulation is for the single-token decode path, which is the hot one. The
    # two-token variant addresses psum with its own fixed layout, so it stays at one row.
    orw = _orw_for(mode)
    xrow = _GGML_KS * vals                      # where the second row's activations start
    subs = [("ACCDECL1", "var<private> acc1: f32;" if two else ""),
            ("ACCBODY1", "  acc1 = acc1 + xs[i + %du] * v;" % xrow if two else ""),
            ("ACC4BODY1", ("  acc1 = acc1 + dot(vec4<f32>(xs[i + %du], xs[i + %du], "
                           "xs[i + %du], xs[i + %du]), v);"
                           % (xrow, xrow + 1, xrow + 2, xrow + 3)) if two else ""),
            ("ACCINIT1", "  acc1 = 0.0;" if two else ""),
            ("XLOAD1", ("      var xv1: f32 = 0.0;\n"
                        "      if (b < nb) { xv1 = x[gm.K + b * BLKVALS + t]; }\n"
                        "      xs[%du + xoff + t] = xv1;" % xrow) if two else ""),
            ("PSUM1", "  psum[%du + ly * WGXu + lx] = acc1;" % (_GGML_KS * _GGML_WGX)
             if two else ""),
            ("OUT1", ("    var t1: f32 = 0.0;\n"
                      "    for (var i: u32 = 0u; i < KSu; i = i + 1u) "
                      "{ t1 = t1 + psum[%du + i * WGXu + lx]; }\n"
                      "    outp[gm.N + n] = t1;" % (_GGML_KS * _GGML_WGX)) if two else ""),
            ("KSxBLKxR", str(_GGML_KS * vals * rows)),
            ("PSUMSZ", str(_GGML_KS * _GGML_WGX * rows * orw)),
            ("KSxBLK", str(_GGML_KS * vals)), ("KSxWGX", str(_GGML_KS * _GGML_WGX)),
            ("KSGx256", str(_GGML_KSG * 256)), ("KSGu", "%uu" % _GGML_KSG),
            ("KSu", "%uu" % _GGML_KS), ("MASKBLK", "%uu" % (vals - 1)),
            ("BLKVALS", "%uu" % vals),
            ("WGXu", "%uu" % _GGML_WGX), ("WGX", str(_GGML_WGX)),
            ("ORWu", "%uu" % orw), ("ORW", str(orw))]
    for k, v in subs:
        src = src.replace(k, v)
    return src


def _ggml_selfcheck(type_name, mode):
    """Multiply one random block and compare against the reference dequantizer.

    A WGSL compile error surfaces as a console warning and a buffer full of zeros, not as an
    exception -- which reads exactly like a working kernel on an all-zero weight, and has
    twice sent me chasing a numerical bug that was a syntax error. One block per type, once
    per session, turns both failure modes into something that raises."""
    from . import ggufload as G
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    # Random bytes are a legal block for every type, and exercise every scale and codebook
    # index -- but a random f16 scale is Inf or NaN one time in 32, and the f16 fields sit at
    # a different offset in each format. Rather than teach the table where they are, redraw
    # until the reference comes out finite.
    for seed in range(32):
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 256, (2, blk), dtype=np.uint8).tobytes()
        ref = np.asarray(G.dequant(G.GGML_IDS[type_name], raw, 2 * vals),
                         np.float32).reshape(2, vals)
        if np.all(np.isfinite(ref)) and float(np.abs(ref).max()) < 1e4:
            break
    else:
        raise RuntimeError("could not draw a finite %s block to self-check against" % type_name)
    M = mode if mode else 3
    x = rng.standard_normal((M, vals)).astype(np.float32)
    raw = raw + b"\x00" * ((-len(raw)) % 4)      # a block is not always a whole number of u32
    pk = ggml_transpose(xp.asarray(np.frombuffer(raw, np.int32)), 2, blk)
    got = np.asarray(cp.asnumpy(_ggml_run(xf=xp.asarray(x), packed=pk,
                                          type_name=type_name, K=vals, N=2))).reshape(M, 2)
    want = x @ ref.T
    err = float(np.abs(got - want).max() / (np.abs(want).max() + 1e-30))
    if not (err < 1e-4):
        raise RuntimeError("native ggml %s kernel for %s is wrong (rel err %.3g) -- check "
                           "the console for a WGSL compile error"
                           % ({0: "gemm", 1: "gemv", 2: "gemv2"}.get(mode, "mode%d" % mode),
                              type_name, err))


def ggml_matmul(xf, packed, type_name, K, N):
    """xf(M,K) @ packed(N,K).T -> (M,N), decoding ggml blocks in the shader.

    `packed` must be in the transposed (word, row) layout that `ggml_transpose` produces --
    GGMLLinear does that once at upload."""
    # A dedicated two-row kernel, not the batched one: verifying a speculative draft is a
    # batch of two, and it only pays if the second row rides along with the first.
    m = int(xf.shape[0])
    mode = m if m <= 2 else 0
    key = (type_name, mode)
    if key not in _ggml_k["added"]:
        _ggml_add(type_name, mode)
        _ggml_k["added"].add(key)           # set before the check: it calls back in here
        _ggml_selfcheck(type_name, mode)
    return _ggml_run(xf, packed, type_name, K, N)


def _ggml_grid(type_name):
    """The codebook buffer for an i-quant type: ksigns[128] then the grid. Built once."""
    tab = _GGML_TYPES[type_name][4]
    if tab is None:
        return None
    if type_name not in _ggml_grids:
        from . import iqtables as T
        g = np.ascontiguousarray(getattr(T, tab)).view(np.uint8).reshape(-1)
        buf = np.zeros(128 + g.size + (-(128 + g.size)) % 4, np.uint8)
        buf[:128] = np.asarray(T.KSIGNS_IQ2XS, np.uint8)
        buf[128:128 + g.size] = g
        _ggml_grids[type_name] = xp.asarray(buf.view(np.int32))
    return _ggml_grids[type_name]


def _ggml_add(type_name, mode):
    plat = _adam_kernel["platform"]
    binds = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
    if _GGML_TYPES[type_name][4] is not None:
        binds.append("read-only-storage")
    plat.addKernel(_ggml_name(type_name, mode),
                   {"source": _ggml_src(type_name, mode), "bindingTypes": binds})


def _ggml_name(type_name, mode, orw=None):
    # ORW is a compile-time constant in the shader, so a kernel is identified by it too --
    # otherwise a second value would silently reuse the first one's pipeline.
    o = _orw_for(mode) if orw is None else orw
    return "ggml%s_%s%s" % (("v", "v2", "")[mode], type_name.lower(),
                            "" if o <= 1 else "_r%d" % o)


def _ggml_run(xf, packed, type_name, K, N):
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    M = int(xf.shape[0])
    name = _ggml_name(type_name, M if M <= 2 else 0)
    plat = _adam_kernel["platform"]
    of = _empty((M, N))
    meta = _adam_kernel["make_meta"]((M, N, K, (K // vals) * blk), "u4,u4,u4,u4")
    bufs = [xf.buffer.buffer_id, packed.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id]
    grid = _ggml_grid(type_name)
    if grid is not None:
        bufs.append(grid.buffer.buffer_id)
    plat.runKernel({"name": name, "tensors": bufs,
                    "workGroups": {"x": (_gemv_groups(N) if M <= 2
                                         else (N + 63) // 64), "y": 1,
                                   "z": 1 if M <= 2 else (M + 3) // 4}})
    return of


def _gptq_quantize(W, group_size=32, bits=4, from_out_in=False, block=2048):
    """Quantize a weight to packed int`bits` with per-group scales and zero points.

    Columns are independent, so this walks them in blocks: a whole-tensor int32 staging
    array is 356 MB on a 27B feed-forward weight and the packing shift doubles it, which no
    32-bit heap will give. Blocked, the peak is set by `block`, not by the tensor.

    `from_out_in=True` means `W` is the (out, in) layout the loaders hold, and the (in, out)
    the packing wants is produced one block at a time -- so the full transposed copy, another
    356 MB, never exists either.
    """
    K, N = (W.shape[1], W.shape[0]) if from_out_in else W.shape
    per = 32 // bits
    qmax = (1 << bits) - 1
    assert K % group_size == 0 and K % per == 0 and N % per == 0, "dims must divide group/pack size"
    nG = K // group_size
    scales = np.zeros((nG, N), np.float32)
    zeros = np.zeros((nG, N), np.int32)
    qweight = np.empty((K // per, N), np.int32)
    for n0 in range(0, N, block):
        n1 = min(N, n0 + block)
        Wb = np.ascontiguousarray(W[n0:n1].T) if from_out_in else W[:, n0:n1]
        qb = np.empty((K, n1 - n0), np.int32)
        for g in range(nG):
            blk = Wb[g * group_size:(g + 1) * group_size]
            wmin = blk.min(0); wmax = blk.max(0)
            sc = (wmax - wmin) / qmax
            sc[sc == 0] = 1e-8
            zp = np.clip(np.round(-wmin / sc), 0, qmax).astype(np.int32)
            scales[g, n0:n1] = sc; zeros[g, n0:n1] = zp
            qb[g * group_size:(g + 1) * group_size] = np.clip(np.round(blk / sc) + zp,
                                                             0, qmax).astype(np.int32)
        # Pack `per` rows into each u32, accumulating in place rather than building a
        # (K/per, per, N) shifted copy.
        qv = qb.reshape(K // per, per, n1 - n0)
        acc = np.zeros((K // per, n1 - n0), np.int32)
        for j in range(per):
            acc |= qv[:, j, :] << np.int32(j * bits)
        qweight[:, n0:n1] = acc
        del Wb, qb, qv, acc
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
    if not GPU:
        return _gptq_matmul_np(xf, qweight, qzeros, scales, K, N, gs, bits, zoff)
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


def _gptq_matmul_np(xf, qweight, qzeros, scales, K, N, gs, bits, zoff=0.0, block=2048):
    """CPU/numpy fallback for the int4/int8 matmul (no WebGPU/WebGL backend).

    The weights stay PACKED in memory and are unpacked one column block at a time, so peak
    memory is the packed model plus one small block — never the full fp32 weight. That is what
    lets a multi-billion-parameter 4-bit model run in a fraction of its fp32 footprint."""
    x = np.asarray(xf, np.float32)
    qw = np.asarray(qweight); qz = np.asarray(qzeros); sc = np.asarray(scales, np.float32)
    per = 32 // bits; qmax = (1 << bits) - 1
    Kp = int(qw.shape[0]) * per
    g = np.arange(Kp) // gs                       # group index per contracted row
    out = np.empty((int(x.shape[0]), N), np.float32)
    for c0 in range(0, N, block):                 # column blocks bound the temporary
        c1 = min(N, c0 + block)
        qwb = qw[:, c0:c1]
        q = np.empty((Kp, c1 - c0), np.int32)
        for r in range(per):
            q[r::per] = (qwb >> (bits * r)) & qmax
        # zeros are packed along N (per column), unpack the columns this block needs
        z_full = np.empty((int(qz.shape[0]), N), np.int32)
        for r in range(per):
            z_full[:, r::per] = (qz >> (bits * r)) & qmax
        z = z_full[:, c0:c1]; del z_full
        w = (sc[g, c0:c1] * (q - (z[g] + zoff))).astype(np.float32)
        out[:, c0:c1] = x @ w
        del q, z, w
    return out


# ---- Gated DeltaNet recurrence on the GPU -----------------------------------------
# The recurrent state S is (heads, Dk, Dv) and every decode step reads it, writes it, and
# reads it again. Done on the host that is three passes over a few megabytes plus a round
# trip for each of the layer's projections -- on a 27B, 48 such layers dominate a token.
#
# Kept on the GPU it is two dispatches, because the output can be written from the OLD
# state. Substituting the update into the read gives
#
#   out = decay * (q . S_old) + delta * (q . k)
#
# so the (head, v) pass computes both contractions it needs from S_old, and the (head, k, v)
# pass updates S independently. Neither has to wait for the other's result.
_GDN_STEP_WGSL = """@group(0) @binding(0)
var<storage,read> S: array<f32>;
@group(0) @binding(1)
var<storage,read> qkv: array<f32>;
@group(0) @binding(2)
var<storage,read_write> od: array<f32>;
struct GD { hv: u32, dk: u32, dv: u32, rep: u32, }
@group(0) @binding(3)
var<storage,read> gd: GD;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = gd.hv * gd.dv;
  if (i >= n) { return; }
  let h = i / gd.dv;
  let vi = i % gd.dv;
  // qkv packs q | k | v | decay | beta for this token. q and k are stored per KEY head and
  // the key heads CYCLE across the value heads (ggml: iq1 = iv1 % n_q_heads), so this is a
  // modulo, not a divide -- the block mapping pairs each query with the wrong key.
  let hk = gd.hv / gd.rep;
  let nq = hk * gd.dk;
  let qo = (h % hk) * gd.dk;
  let ko = nq + qo;
  let vo = 2u * nq + h * gd.dv;
  let dcy = qkv[2u * nq + gd.hv * gd.dv + h];
  let bta = qkv[2u * nq + gd.hv * gd.dv + gd.hv + h];
  let sbase = h * gd.dk * gd.dv + vi;
  var pred: f32 = 0.0;
  var qs: f32 = 0.0;
  var qk: f32 = 0.0;
  for (var d: u32 = 0u; d < gd.dk; d = d + 1u) {
    let sv = S[sbase + d * gd.dv];
    let kd = qkv[ko + d];
    let qd = qkv[qo + d];
    pred = pred + kd * sv;
    qs = qs + qd * sv;
    qk = qk + qd * kd;
  }
  let delta = (qkv[vo + vi] - dcy * pred) * bta;
  od[i] = dcy * qs + delta * qk;          // output, from the old state
  od[n + i] = delta;                      // handed to the update pass
}
"""

# S = S * decay + outer(k, delta), one thread per state element.
_GDN_UPD_WGSL = """@group(0) @binding(0)
var<storage,read_write> S: array<f32>;
@group(0) @binding(1)
var<storage,read> qkv: array<f32>;
@group(0) @binding(2)
var<storage,read> od: array<f32>;
struct GD { hv: u32, dk: u32, dv: u32, rep: u32, }
@group(0) @binding(3)
var<storage,read> gd: GD;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= gd.hv * gd.dk * gd.dv) { return; }
  let h = i / (gd.dk * gd.dv);
  let rem = i % (gd.dk * gd.dv);
  let d = rem / gd.dv;
  let vi = rem % gd.dv;
  let hk = gd.hv / gd.rep;
  let nq = hk * gd.dk;
  let dcy = qkv[2u * nq + gd.hv * gd.dv + h];
  S[i] = S[i] * dcy + qkv[nq + (h % hk) * gd.dk + d] * od[gd.hv * gd.dv + h * gd.dv + vi];
}
"""

# Everything between the projections and the recurrence, in one dispatch: the causal conv,
# its SiLU, the L2 norm on q and k, and the decay/write gates. Done with tensor ops these
# are ~40 separate calls, and a call costs 0.3-0.9 ms here regardless of how little data it
# touches, which is why the unfused version lost to numpy.
#
# One workgroup per head (plus one for the gates). Threads within it hold a head's channels,
# so the L2 norm's sum is a workgroup reduction rather than another dispatch.
_GDN_PRE_WGSL = """@group(0) @binding(0)
var<storage,read> qkv: array<f32>;
@group(0) @binding(1)
var<storage,read> braw: array<f32>;
@group(0) @binding(2)
var<storage,read> araw: array<f32>;
@group(0) @binding(3)
var<storage,read_write> cst: array<f32>;
@group(0) @binding(4)
var<storage,read> konst: array<f32>;
@group(0) @binding(5)
var<storage,read_write> outp: array<f32>;
struct GP { hk: u32, hv: u32, dk: u32, dv: u32, W: u32, flags: u32, }
@group(0) @binding(6)
var<storage,read> gp: GP;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;
  let t = lid.x;
  let nq = gp.hk * gp.dk;
  let nv = gp.hv * gp.dv;
  let C = 2u * nq + nv;
  let heads = 2u * gp.hk + gp.hv;
  if (g >= heads) {
    // gate workgroup: decay = exp(min(softplus(a + dt_bias) * A, 0)), beta = sigmoid(b)
    if (t < gp.hv) {
      let ko = gp.W * C + C;                       // conv_w, conv_b, then A, dt_bias
      var a = araw[t];
      if ((gp.flags & 2u) != 0u) { a = a + konst[ko + gp.hv + t]; }
      let sp = max(a, 0.0) + log(1.0 + exp(-abs(a)));
      var d = sp;
      if ((gp.flags & 4u) != 0u) { d = sp * konst[ko + t]; }
      outp[2u * nq + nv + t] = exp(min(d, 0.0));
      var bv: f32 = 1.0;
      if ((gp.flags & 8u) != 0u) { bv = 1.0 / (1.0 + exp(-braw[t])); }
      outp[2u * nq + nv + gp.hv + t] = bv;
    }
    return;
  }
  // a q, k or v head: its channels are [c0, c0 + dim)
  var c0: u32; var dim: u32; var isqk: bool;
  if (g < 2u * gp.hk) { c0 = g * gp.dk; dim = gp.dk; isqk = true; }
  else { c0 = 2u * nq + (g - 2u * gp.hk) * gp.dv; dim = gp.dv; isqk = false; }
  var val: f32 = 0.0;
  if (t < dim) {
    let c = c0 + t;
    let raw = qkv[c];
    if ((gp.flags & 1u) != 0u) {                   // causal depthwise conv over W taps
      var acc: f32 = 0.0;
      for (var j: u32 = 0u; j + 1u < gp.W; j = j + 1u) {
        acc = acc + cst[j * C + c] * konst[j * C + c];
      }
      acc = acc + raw * konst[(gp.W - 1u) * C + c];
      if ((gp.flags & 16u) != 0u) { acc = acc + konst[gp.W * C + c]; }
      val = acc / (1.0 + exp(-acc));               // SiLU
      // the ring buffer holds INPUTS, so it shifts in `raw`, not the conv output
      for (var j: u32 = 0u; j + 2u < gp.W; j = j + 1u) {
        cst[j * C + c] = cst[(j + 1u) * C + c];
      }
      if (gp.W > 1u) { cst[(gp.W - 2u) * C + c] = raw; }
    } else {
      val = raw;
    }
  }
  if (isqk) {
    red[t] = val * val;
    workgroupBarrier();
    for (var s: u32 = 64u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] + 1e-6);
    if (t < dim) {
      var v = val * inv;
      if (g < gp.hk) { v = v * inverseSqrt(f32(gp.dk)); }   // q also carries 1/sqrt(dk)
      outp[c0 + t] = v;
    }
  } else if (t < dim) {
    outp[c0 + t] = val;
  }
}
"""

_gdnp_k = {"added": False}


def gdn_prepare(qkv, braw, araw, cst, konst, out, hk, hv, dk, dv, W, flags):
    """Conv + SiLU + L2 norm + gates, writing the packed q|k|v|decay|beta the step wants.

    `cst` (the conv ring buffer) is updated in place. `flags` bits: 1 conv, 2 dt_bias,
    4 A, 8 beta projection, 16 conv bias."""
    plat = _adam_kernel["platform"]
    if not _gdnp_k["added"]:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel("gdn_pre", {"source": _GDN_PRE_WGSL,
                                   "bindingTypes": [ro, ro, ro, rw, ro, rw, ro]})
        _gdnp_k["added"] = True
    meta = _adam_kernel["make_meta"]((hk, hv, dk, dv, W, flags), "u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "gdn_pre",
                    "tensors": [qkv.buffer.buffer_id, braw.buffer.buffer_id,
                                araw.buffer.buffer_id, cst.buffer.buffer_id,
                                konst.buffer.buffer_id, out.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": 2 * hk + hv + 1, "y": 1, "z": 1}})
    return out


# Weights are stored one row after another, so in a matmul the threads of a workgroup --
# each owning an output row -- read addresses a whole row apart. Measured with the decode
# happening: 59 GB/s in that layout against 90+ for the same traffic read contiguously.
# Transposing to (word, row) at upload makes neighbouring threads read neighbouring words.
# Rows are padded to a word boundary first, since a block size is not always a multiple of
# four and a row would otherwise start mid-word.
_TRANSPOSE_WGSL = """@group(0) @binding(0)
var<storage,read> src: array<u32>;
@group(0) @binding(1)
var<storage,read_write> dst: array<u32>;
struct TM { n: u32, words: u32, rowb: u32, total: u32, gx: u32, pad: u32, }
@group(0) @binding(2)
var<storage,read> tm: TM;
@group(0) @binding(3)
var<storage,read_write> flag: array<f32>;
fn sb(o: u32) -> u32 { return (src[o >> 2u] >> ((o & 3u) * 8u)) & 255u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Two-dimensional: a big head needs ~900k workgroups and a dimension caps at 65535.
  let i = gid.y * tm.gx * 64u + gid.x;
  if (i >= tm.total) { return; }
  let wo = i / tm.n;
  let row = i - wo * tm.n;
  let b = row * tm.rowb + wo * 4u;          // byte offset of this word within the row
  var v: u32 = 0u;
  if (b + 3u < (row + 1u) * tm.rowb) {
    v = sb(b) | (sb(b + 1u) << 8u) | (sb(b + 2u) << 16u) | (sb(b + 3u) << 24u);
  } else {
    var k: u32 = 0u;                        // tail of the row: whatever bytes remain
    loop {
      if (k >= 4u || b + k >= (row + 1u) * tm.rowb) { break; }
      v = v | (sb(b + k) << (8u * k));
      k = k + 1u;
    }
  }
  dst[i] = v;
  if (i == 0u) { flag[0] = 1.0; }        // four bytes to read back, instead of the tensor
}
"""

_tr_k = {"added": False}


def ggml_transpose(src, n, rowb):
    """(n, rowb bytes) -> (words, n) u32, so a matmul's threads read adjacent words."""
    words = (rowb + 3) // 4
    plat = _adam_kernel["platform"]
    if not _tr_k["added"]:
        plat.addKernel("ggml_tr", {"source": _TRANSPOSE_WGSL,
                                   "bindingTypes": ["read-only-storage", "storage",
                                                    "read-only-storage", "storage"]})
        _tr_k["added"] = True
    total = words * n
    dst = _empty((total,))
    groups = (total + 63) // 64
    gx = min(groups, 32768)                     # a dispatch dimension caps at 65535
    gy = (groups + gx - 1) // gx
    meta = _adam_kernel["make_meta"]((n, words, rowb, total, gx, 0), "u4,u4,u4,u4,u4,u4")
    flag = _empty((1,))
    plat.runKernel({"name": "ggml_tr",
                    "tensors": [src.buffer.buffer_id, dst.buffer.buffer_id, meta.buffer_id,
                                flag.buffer.buffer_id],
                    "workGroups": {"x": gx, "y": gy, "z": 1}})
    # The dispatch is queued, not done. The caller drops the source right after this, and
    # freeing a buffer a pending command still reads leaves the destination zeroed -- which
    # showed up as a loaded model of all-zero weights while the same tensor transposed on
    # its own was fine, because using it immediately forced the queue to drain.
    #
    # Read the flag, not the tensor: a read-back is sized to the whole buffer, and asking
    # for a 210 MB head block back exceeds what the backend will stage ("buffer size
    # insufficient"). Four bytes drain the queue just as well.
    cp.asnumpy(flag)
    return dst


_gdn_k = {"added": False}


def gdn_step(S, qkv, hv, dk, dv, rep=1):
    """One Gated-DeltaNet step, entirely on the GPU.

    `S` is the (hv, dk, dv) state, updated in place; `qkv` packs q | k | v | decay | beta
    for this token, with q and k stored per KEY head (`rep` value heads share each).
    Returns the (hv*dv,) output."""
    plat = _adam_kernel["platform"]
    if not _gdn_k["added"]:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel("gdn_step", {"source": _GDN_STEP_WGSL, "bindingTypes": [ro, ro, rw, ro]})
        plat.addKernel("gdn_upd", {"source": _GDN_UPD_WGSL, "bindingTypes": [rw, ro, ro, ro]})
        _gdn_k["added"] = True
    n = hv * dv
    od = _empty((2 * n,))                       # [output | delta]
    meta = _adam_kernel["make_meta"]((hv, dk, dv, max(1, rep)), "u4,u4,u4,u4")
    plat.runKernel({"name": "gdn_step",
                    "tensors": [S.buffer.buffer_id, qkv.buffer.buffer_id,
                                od.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    tot = hv * dk * dv
    plat.runKernel({"name": "gdn_upd",
                    "tensors": [S.buffer.buffer_id, qkv.buffer.buffer_id,
                                od.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (tot + 63) // 64, "y": 1, "z": 1}})
    return Tensor(od[:n])


class GGMLLinear(Module):
    """Inference-only Linear whose weight stays in the encoding the GGUF shipped it in.

    Nothing is dequantized or requantized at load: the file's bytes go to the GPU as they
    are and `ggml_matmul` unpacks each block while it multiplies. That removes the whole
    conversion pass -- the bulk of a load -- and the second rounding it imposed."""

    def __init__(self, raw, type_name, K, N, bias=None):
        b = np.frombuffer(raw, np.uint8)
        pad = (-b.size) % 4
        if pad:
            b = np.concatenate([b, np.zeros(pad, np.uint8)])
        up = xp.asarray(b.view(np.int32))
        # Transposed on the way in, once, so every later matmul reads it coalesced. Done on
        # the device: the host copy is already the largest thing in flight during a load.
        vals, blk = _GGML_TYPES[type_name][2], _GGML_TYPES[type_name][3]
        self.packed = ggml_transpose(up, int(N), (int(K) // vals) * blk)
        del up
        self.type_name = type_name
        self.Kt = int(K); self.Nt = int(N)
        self.bias = None if bias is None else xp.asarray(np.asarray(bias, np.float32))

    def forward(self, x):
        xd = x.data
        lead = xd.shape[:-1]
        of = ggml_matmul(_contig(xd.reshape(-1, self.Kt)), self.packed,
                         self.type_name, self.Kt, self.Nt)
        if self.bias is not None:
            of = of + self.bias
        return Tensor(of.reshape(*lead, self.Nt))                 # inference-only


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


class UnquantizedLinear(Module):
    """Inference-only UNquantized Linear: `y = x @ W.T + b`. Weights come from an fp16/bf16
    model and are computed in fp32 (the WebGPU/WebGL backend is fp32). Exposes the same
    `__call__(x) -> Tensor` interface as `QuantizedLinear`, so the LLM engine treats int4 /
    int8 / fp16 layers identically (capture-replay decode works the same)."""
    def __init__(self, weight, bias=None):
        W = np.asarray(weight)                                   # (Nt=out, Kt=in)
        self.Nt, self.Kt = int(W.shape[0]), int(W.shape[1])
        self.Wt = xp.asarray(np.ascontiguousarray(W.T.astype(np.float32)))   # (Kt, Nt) for x@Wt
        self.bias = xp.asarray(np.zeros((self.Nt,), np.float32) if bias is None
                               else np.asarray(bias, np.float32))

    def forward(self, x):
        xd = x.data; lead = xd.shape[:-1]
        xf = _contig(xd.reshape(-1, self.Kt))
        of = xf @ self.Wt
        return Tensor((of + self.bias).reshape(*lead, self.Nt))

    def nbytes(self):
        return int(self.Wt.size * 4 + self.bias.size * 4)


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
