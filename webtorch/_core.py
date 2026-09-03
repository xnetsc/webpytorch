"""webtorch — a minimal PyTorch-compatible shim with define-by-run autograd,
backed by WgPy's cupy arrays (GPU via WebGPU/WebGL) inside Pyodide.

Phase-2 vertical slice: enough to build and TRAIN a small MLP and verify that
gradients (computed on the GPU) match numerical finite differences. Conv2d /
attention / more ops come later; the autograd core here is what everything else
builds on.
"""
import re
import numpy as np

# Why the GPU is not being used, when it is not. Every step that can fail writes its reason
# here instead of discarding it, because "it fell back to the CPU" is a symptom and the
# reason is what anyone can act on -- and the difference is roughly three hundred times, so
# somebody always ends up asking.
_backend_why = {"gpu_import": None, "backend_name": None, "platform": None}

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception as _e:  # no GPU backend -> fall back to numpy CPU
    xp = np
    GPU = False
    _backend_why["gpu_import"] = "%s: %s" % (type(_e).__name__, _e)


def backend_reason():
    """What is stopping the GPU path, as a sentence, or None when nothing is.

    Reading this is the supported way to find out why a machine that should be fast is not.
    """
    if _adam_kernel.get("platform") is not None:
        return None
    if _backend_why["gpu_import"]:
        return ("the GPU array backend could not be imported (" + _backend_why["gpu_import"]
                + ") -- in a browser this is normally a page that is not cross-origin "
                  "isolated, so SharedArrayBuffer is unavailable")
    if _backend_why["backend_name"]:
        return "the array backend in use is '%s', not 'webgpu'" % _backend_why["backend_name"]
    if _backend_why["platform"]:
        return "the WebGPU platform failed to start (" + _backend_why["platform"] + ")"
    return "the GPU backend has not been initialised yet"


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

    def _setback(self, fn):
        """Attach a backward closure, but only where one can ever run.

        Every such closure reads `out.grad`, so it refers to the tensor it is attached to:
        attaching one puts that tensor in a reference CYCLE, which refcounting can never
        free and only a full collection can find. In inference nothing requires grad and the
        closure is dead weight -- but it kept every intermediate alive to the end of a
        prefill. Measured before this: sixty tensors' worth of arithmetic left 1188 objects
        in cycles, 96 of them Tensors with their buffers, and a prompt held ~10GB of
        intermediates that a collection then handed straight back.

        Skipping it is what makes them die the moment they go out of scope, which is what
        was supposed to happen all along.
        """
        if self.requires_grad:
            self._backward = fn

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
        out._setback(_backward)
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
        out._setback(_backward)
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "@")

        def _backward():
            if self.requires_grad:
                self._accum(_unbroadcast(out.grad @ _swap_last2(other.data), self.data.shape))
            if other.requires_grad:
                other._accum(_unbroadcast(_swap_last2(self.data) @ out.grad, other.data.shape))
        out._setback(_backward)
        return out

    __matmul__ = matmul

    def relu(self):
        out = Tensor(xp.maximum(self.data, 0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                mask = (self.data > 0).astype(np.float32)
                self._accum(mask * out.grad)
        out._setback(_backward)
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
            out._setback(_backward)
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
        out._setback(_backward)
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
        out._setback(_backward)
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
        out._setback(_backward)
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
        out._setback(_backward)
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
        out._setback(_backward)
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
        out._setback(_backward)
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
    elif _webgl_ready():
        outdata = _webgl_cat([t.data for t in tensors], axis)
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
    out._setback(_backward)
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
    """x * sigmoid(x).

    Fused where the backend has it and the value carries no autograd node. Written out it
    is five dispatches -- negate, exp, add, divide, multiply -- each reading and writing the
    whole tensor, and the recurrent layers call it once per layer per token.
    """
    if (isinstance(x, Tensor) and not x.requires_grad
            and _webgl_ready() and not _adam_backend_ready()):
        r = _webgl_silu(x)
        if r is not None:
            return r
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
    out._setback(_backward)
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


def _onehot(targets, N, C):
    """(N, C) float32 with a single 1 per row, WITHOUT building an identity matrix.

    `np.eye(C)[targets]` reads well and allocates C x C to select N rows of it. C here is the
    number of classes, which for a language model is the vocabulary: 151936 classes is 92 TB
    before a single row is taken. This is the same array the identity would have produced.
    """
    oh = np.zeros((int(N), int(C)), np.float32)
    oh[np.arange(int(N)), np.asarray(targets).astype(np.int64).reshape(-1)] = 1.0
    return oh


def nll_loss(log_probs, targets):
    """Negative log-likelihood. log_probs: (N, C); targets: numpy int (N,).
    cross_entropy == nll_loss(log_softmax(x))."""
    ld = log_probs.data
    N, C = ld.shape
    onehot = Tensor(xp.asarray(_onehot(targets, N, C)))
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


_MOE_ROUTE_WGSL = """@group(0) @binding(0)
var<storage,read> lg: array<f32>;
@group(0) @binding(1)
var<storage,read_write> eidx: array<i32>;
@group(0) @binding(2)
var<storage,read_write> ew: array<f32>;
struct RM { ne: u32, k: u32, norm: u32, pad: u32, }
@group(0) @binding(3)
var<storage,read> rm: RM;
var<workgroup> v: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
var<workgroup> ridx: array<u32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  // stage the router's scores; anything past ne is -inf so it never wins a pass
  for (var e: u32 = t; e < 512u; e = e + 128u) {
    v[e] = select(-1e30, lg[e], e < rm.ne);
  }
  workgroupBarrier();
  // k passes of argmax, each taking the winner out of the running
  for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
    var best: f32 = -1e30;
    var bi: u32 = 0u;
    for (var e: u32 = t; e < rm.ne; e = e + 128u) {
      if (v[e] > best) { best = v[e]; bi = e; }
    }
    red[t] = best; ridx[t] = bi;
    workgroupBarrier();
    var r: u32 = 64u;
    loop {
      if (r == 0u) { break; }
      if (t < r) {
        if (red[t + r] > red[t]) { red[t] = red[t + r]; ridx[t] = ridx[t + r]; }
      }
      workgroupBarrier();
      r = r / 2u;
    }
    if (t == 0u) {
      eidx[s] = i32(ridx[0]);
      ew[s] = red[0];
      v[ridx[0]] = -1e30;
    }
    workgroupBarrier();
  }
  // softmax over ALL experts, then renormalise across the chosen ones (the Qwen convention);
  // with norm == 0 the raw softmax weights are kept.
  if (t == 0u) {
    var mx: f32 = -1e30;
    for (var e: u32 = 0u; e < rm.ne; e = e + 1u) { mx = max(mx, lg[e]); }
    var den: f32 = 0.0;
    for (var e: u32 = 0u; e < rm.ne; e = e + 1u) { den = den + exp(lg[e] - mx); }
    var tot: f32 = 0.0;
    for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
      let p = exp(ew[s] - mx) / den;
      ew[s] = p; tot = tot + p;
    }
    if (rm.norm == 1u && tot > 0.0) {
      for (var s: u32 = 0u; s < rm.k; s = s + 1u) { ew[s] = ew[s] / tot; }
    }
  }
}
"""
_moe_r = {"added": False}


def moe_route(logits, eidx, ew, ne, k, norm=True):
    """Router scores -> chosen experts and their weights, entirely on the device.

    Doing this on the host means reading the router's output back once per MoE layer, which
    at 48 layers is most of a decode step -- and it is a tiny amount of data, so the cost is
    all round-trip. Keeping it here also keeps the step capturable: the indices land in a
    buffer the expert matmuls already read, and no command depends on their value."""
    if _webgl_ready() and not _adam_backend_ready():
        return _webgl_moe_route(logits, eidx, ew, ne, k, norm)
    plat = _adam_kernel["platform"]
    if not _moe_r["added"]:
        plat.addKernel("moe_route", {"source": _MOE_ROUTE_WGSL,
                                     "bindingTypes": ["read-only-storage", "storage",
                                                      "storage", "read-only-storage"]})
        _moe_r["added"] = True
    meta = _adam_kernel["make_meta"]((int(ne), int(k), 1 if norm else 0, 0), "u4,u4,u4,u4")
    plat.runKernel({"name": "moe_route",
                    "tensors": [logits.buffer.buffer_id, eidx.buffer.buffer_id,
                                ew.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": 1, "y": 1, "z": 1}})


def _empty_i32(shape):
    """A small int32 buffer the host rewrites between dispatches -- a step's control block,
    or the expert indices a MoE layer routes to. Persistent, so a capture can bind it once
    and see the new contents on every replay."""
    return xp.empty(shape, np.int32)


# GLSL helper: read element `idx` of a texture laid out row-major (width from the
# texture). Shared by all WebGL kernels. Defined early so module-level kernel
# strings that .replace("FETCH", _GL_FETCH) can use it.
_GL_FETCH = ("float fetch(sampler2D t, int idx) { int tw = textureSize(t, 0).x; "
             "int y = idx / tw; int x = idx - y * tw; return texelFetch(t, ivec2(x, y), 0).r; }")


def _laid_out_contiguously(a):
    """Is `a` laid out byte for byte like a C-contiguous array filling its own buffer?

    `flags.c_contiguous` compares strides axis by axis, so it says no to an axis of length 1
    that a transpose moved -- even though an axis holding no elements cannot move any. Decode
    transposes exactly that shape on every attention layer: k and v come out (1, heads, dim)
    and are wanted (heads, 1, dim), which is the same bytes in the same order.

    Skipping those axes is not a relaxation of what the caller needs. The requirement is that
    a kernel indexing the buffer linearly reads the right elements, and that holds exactly
    when the strides of the axes that DO hold elements are the C-contiguous ones, the view
    starts at 0, and it covers the whole buffer -- which is what this checks."""
    st = getattr(a, "strides", None)
    if st is None or getattr(a, "offset", 0) != 0:
        return False
    buf = getattr(a, "buffer", None)
    if buf is None or int(a.size) != int(buf.size):
        return False
    exp = a.itemsize
    for d, s in zip(reversed(a.shape), reversed(st)):
        if d == 1:
            continue                      # no elements on this axis; its stride is unused
        if s != exp:
            return False
        exp *= d
    return True


def _contig(a):
    # Materialize a (possibly transposed/strided) array — WgPy reshape/matmul are unreliable
    # on non-contiguous views, and the kernels here index the buffer linearly, so a view that
    # does not start at offset 0 or does not fill its buffer would read the wrong elements.
    # Multiplying by one forces a stride-aware kernel that produces one that does.
    #
    # An array already satisfying both is returned unchanged. Copying it is pure bandwidth,
    # and the KV cache is bound this way once per attention layer per token: at a 4096-token
    # context that copy alone moved ~940 MB per token — more than everything else in the step
    # put together, and the reason a larger context slowed decode down even when the
    # conversation was short.
    #
    # The stride check catches what the flag misses -- a transposed axis of length 1 -- which
    # on a 28-layer decode step was 168 copies of 4 KB, each costing two dispatches, to
    # produce buffers identical to what it read.
    # A Tensor has no `flags`, so before this it failed both checks and was copied every
    # time -- silently, since the copy is correct, just wasted. Decode binds k and v this way
    # on every attention layer.
    if isinstance(a, Tensor):
        d = _contig(a.data)
        if d is a.data:
            return a
        out = Tensor(d, a.requires_grad, (a,), "contig")

        def _backward():
            if a.requires_grad:
                a._accum(out.grad)          # same elements in the same order; layout only
        out._setback(_backward)
        return out
    f = getattr(a, "flags", None)
    if (f is not None and getattr(f, "c_contiguous_full", False)) or _laid_out_contiguously(a):
        return a
    # np.float32(1.0), not 1.0: a Python float is float64 to the ufunc, so it inserts an
    # astype over the whole array before the multiply and the copy costs two dispatches
    # instead of one.
    return a * np.float32(1.0)


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
    out._setback(_backward)
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
    out._setback(_backward)
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
    out._setback(_backward)
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
    out._setback(_backward)
    return out


def gqa_attention(q, k, v, mask=None, scale=None, causal_start=None):
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
    if causal_start is not None and _adam_backend_ready() and not q.requires_grad:
        # Nothing seq-squared is built at all -- see `_FLASH_WGSL`. The fused causal softmax
        # below is the previous step of the same argument and stays as the fallback for a
        # shape whose tiles do not fit workgroup memory.
        if 2 * _FLASH_BQ * hd + _FLASH_BK * hd + _FLASH_BQ * _FLASH_BK <= 8192:
            return flash_attention(q, k, v, start=causal_start, scale=scale)
        a = Tensor(_fused_causal_softmax(bmm(qg, transpose_last2(k)).data,
                                         T, causal_start, scale))
    else:
        a = bmm(qg, transpose_last2(k)) * scale        # (nkv, rep*T, S)
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
# Split-sequence decode attention: the same answer as the kernel above, but spread over
# `SPLIT` times as many workgroups.
#
# The kernel above dispatches ONE workgroup per attention head -- 16 of them on a 0.6B, 2048
# threads for the whole GPU -- and every one of them walks the whole cache alone. Measured on
# the captured decode step: 7.07ms fixed plus 8.3us per context token, which for the 229KB
# each context token costs across 28 layers is 27.6 GB/s, an order of magnitude under what
# the machine can do. Nothing is compute-bound here; the device is simply idle.
#
# So each head's scan is cut into SPLIT chunks that run at once, each producing a PARTIAL
# softmax -- its own running max, its own sum, its own weighted values -- and a second, tiny
# pass merges them. Merging is exact, not an approximation: the max/sum/accumulator triple is
# what the online softmax already carries between blocks inside one workgroup, and combining
# two of them is the same rescale it already does.
#
# The chunk bounds come from `n` at RUN time while SPLIT is fixed at compile time, because a
# captured graph replays fixed dispatch dimensions -- a chunk that lands past the end of a
# short conversation contributes nothing and says so with l = 0.
_GQA_SPLIT = 16

_GQA_SPLIT_WGSL = """@group(0) @binding(0)
var<storage,read_write> part: array<f32>;      // (nh*SPLIT) x (hd + 2): acc, then m, l
@group(0) @binding(1)
var<storage,read> q: array<f32>;
@group(0) @binding(2)
var<storage,read> kc: array<f32>;
@group(0) @binding(3)
var<storage,read> vc: array<f32>;
struct GMeta { nh: u32, nkv: u32, hd: u32, lmax: u32, valid: u32, use_ctl: u32, scale: f32, }
@group(0) @binding(4)
var<storage,read> gm: GMeta;
@group(0) @binding(5)
var<storage,read> ctl: array<i32>;
var<workgroup> sc: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> acc: array<f32, 256>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x / SPLITu;
  let ch = wid.x % SPLITu;
  let t = lid.x;
  let rep = gm.nh / gm.nkv;
  let kv = h / rep;
  let qo = h * gm.hd;
  let ko = kv * gm.lmax * gm.hd;
  var n: u32 = gm.valid;
  if (gm.use_ctl == 1u) { n = u32(max(ctl[0], 0)) + 1u; }
  n = clamp(n, 1u, gm.lmax);
  // Even split, rounded up, so the last chunk is the short one and every chunk index is
  // computed the same way whatever `n` turns out to be.
  let per = (n + SPLITu - 1u) / SPLITu;
  let lo = ch * per;
  let hi = min(n, lo + per);
  let po = (h * SPLITu + ch) * (gm.hd + 2u);

  for (var d: u32 = t; d < gm.hd; d = d + 128u) { acc[d] = 0.0; }
  var m_run: f32 = -1e30;
  var l_run: f32 = 0.0;
  workgroupBarrier();

  // An empty chunk still has to write its slot -- the merge reads every one of them.
  if (lo >= hi) {
    for (var d: u32 = t; d < gm.hd; d = d + 128u) { part[po + d] = 0.0; }
    if (t == 0u) { part[po + gm.hd] = -1e30; part[po + gm.hd + 1u] = 0.0; }
    return;
  }

  var base: u32 = lo;
  loop {
    if (base >= hi) { break; }
    let s = base + t;
    var d0: f32 = -1e30;
    if (s < hi) {
      var dd: f32 = 0.0;
      let kb = ko + s * gm.hd;
      for (var i: u32 = 0u; i < gm.hd; i = i + 1u) {
        dd = dd + q[qo + i] * kc[kb + i];
      }
      d0 = dd * gm.scale;
    }
    red[t] = d0;
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
    if (s < hi) { e = exp(d0 - m_new); }
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

    let cnt = min(128u, hi - base);
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

  // Unnormalised: the merge divides once, by the total across chunks.
  for (var d: u32 = t; d < gm.hd; d = d + 128u) { part[po + d] = acc[d]; }
  if (t == 0u) { part[po + gm.hd] = m_run; part[po + gm.hd + 1u] = l_run; }
}
"""

# The merge. One workgroup per head, one lane per head dimension: read SPLIT partial
# softmaxes and fold them into one, which is the same rescale-and-add the split kernel does
# between its own blocks. Chunks that covered nothing carry l = 0 and drop out of the sum.
_GQA_MERGE_WGSL = """@group(0) @binding(0)
var<storage,read_write> outp: array<f32>;
@group(0) @binding(1)
var<storage,read> part: array<f32>;
struct GMeta { nh: u32, nkv: u32, hd: u32, lmax: u32, valid: u32, use_ctl: u32, scale: f32, }
@group(0) @binding(2)
var<storage,read> gm: GMeta;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  var m_all: f32 = -1e30;
  for (var c: u32 = 0u; c < SPLITu; c = c + 1u) {
    let po = (h * SPLITu + c) * (gm.hd + 2u);
    if (part[po + gm.hd + 1u] > 0.0) { m_all = max(m_all, part[po + gm.hd]); }
  }
  var l_all: f32 = 0.0;
  for (var c: u32 = 0u; c < SPLITu; c = c + 1u) {
    let po = (h * SPLITu + c) * (gm.hd + 2u);
    let l = part[po + gm.hd + 1u];
    if (l > 0.0) { l_all = l_all + l * exp(part[po + gm.hd] - m_all); }
  }
  for (var d: u32 = t; d < gm.hd; d = d + 128u) {
    var o: f32 = 0.0;
    for (var c: u32 = 0u; c < SPLITu; c = c + 1u) {
      let po = (h * SPLITu + c) * (gm.hd + 2u);
      if (part[po + gm.hd + 1u] > 0.0) {
        o = o + part[po + d] * exp(part[po + gm.hd] - m_all);
      }
    }
    outp[h * gm.hd + d] = o / l_all;
  }
}
"""


_gqa_k = {"added": False}
_GQA_FUSED = True      # A/B switch for the fused decode attention
_GQA_SPLIT_ON = True   # A/B switch for the split-sequence decode attention


# Choosing the split factor by measuring it, because it cannot be chosen by reasoning.
#
# How many chunks fills a GPU best depends on the GPU, and on how long the conversation is,
# and the two disagree: measured on one machine with a 0.6B, medians of interleaved replays,
#
#     split      1      4      8     16     32     64
#     n=256   10.14   9.05   8.76   9.14   9.61  10.99
#     n=2048  24.20  13.29  12.95  12.38  12.57  12.75
#
# -- 8 wins short, 16 wins long, 1 and 64 lose everywhere. A constant compiled in is a guess
# about somebody else's machine; this asks the machine in front of it. The same is true of
# every other shape constant here (`_GGML_WGX`, `_SMALL_N`, `_GGML_KS`), and one of them has
# already been caught being wrong by 37% on shapes it was never measured on.
#
# Timed the only way these separate at all: candidates are captured once each and their
# replays INTERLEAVED, compared by median. Run back to back instead, the same configuration
# measured 7.61ms and 4.49ms on this machine -- enough drift to invert any ranking.
# ---- picking shape constants by measuring them ----------------------------------------
#
# Workgroup widths, split factors and the thresholds that choose between thread shapes are
# properties of the MACHINE and of the shapes a given model uses -- not of the algorithm.
# A number compiled in here is a guess about somebody else's GPU, and the guesses have been
# caught wrong: the decode matmul's narrow/wide threshold, measured on shapes it was never
# measured on, picks the slower kernel by 37%.
#
# So they are run instead. Two rules, both learned the hard way:
#
#  - INTERLEAVE and take medians. Timed back to back, one unchanged configuration measured
#    7.61ms and then 4.49ms on this stack -- drift enough to invert any ranking, which is
#    how a guess gets confirmed by accident.
#  - GATE ON CORRECTNESS FIRST. A WGSL kernel that fails to compile returns zeros without
#    raising, and a kernel that does nothing is very fast. A candidate that cannot be shown
#    to compute the right answer is dropped before it is ever timed.
_TUNED = {}


def tune(key, candidates, apply, bench, check=None, rounds=5, default=None):
    """The best of `candidates` on this device, remembered under `key`.

    `apply(v)` installs a candidate, `bench()` runs the work once and returns only when the
    GPU has, `check(v)` (optional) returns True if the candidate is correct.
    """
    if key in _TUNED:
        return _TUNED[key]
    import time as _t
    ok = []
    for v in candidates:
        try:
            apply(v)
            if check is not None and not check(v):
                continue
            bench()
            ok.append(v)
        except Exception:
            continue
    if not ok:
        _TUNED[key] = default
        return default
    times = {v: [] for v in ok}
    for _ in range(rounds):
        for v in ok:
            apply(v)
            t0 = _t.perf_counter()
            bench()
            times[v].append(_t.perf_counter() - t0)
    best, best_ms = None, None
    for v, xs in times.items():
        xs.sort()
        med = xs[len(xs) // 2]
        if best_ms is None or med < best_ms:
            best, best_ms = v, med
    _TUNED[key] = best
    return best


# Which thread shape the decode matmul should use for one (format, N, K), decided by running
# all three rather than by comparing N against a constant. The kernels already exist -- this
# only stops `_SMALL_N` being the thing that chooses between them.
def _ggml_shape_for(type_name, N, K, packed):
    """The thread shape the decode matmul should use for this (format, N, K), decided by
    running all three on this device with THIS model's own weights.

    The kernels already exist; this only stops a compiled-in threshold being what chooses
    between them. Each candidate is self-checked before it is timed -- a shader that failed
    to compile returns zeros without raising, and doing nothing is fast.
    """
    vals = _GGML_TYPES[type_name][2]
    fallback = _shape_kind(N, K, vals)
    key = ("ggml_shape", type_name, int(N), int(K))
    if key in _TUNED:
        return _TUNED[key]
    if _adam_kernel.get("platform") is None:
        return fallback
    nb = max(1, int(K) // max(1, vals))
    xd = _contig(Tensor(np.zeros((1, int(K)), np.float32)).data)
    state = {"kind": fallback}

    def apply(kind):
        state["kind"] = kind
        k = (type_name, 1, kind, False)
        if k not in _ggml_k["added"]:
            _ggml_add(type_name, 1, kind, False)
            _ggml_k["added"].add(k)

    def check(kind):
        try:
            _selfcheck_one(type_name, 1, kind, False, int(N), nb)
            return True
        except Exception:
            return False

    # Many dispatches per sync. A readback costs 1-2ms on this stack and one decode matmul
    # costs tens of microseconds, so timing them one at a time measures the readback and
    # ranks the candidates by noise -- which is exactly what happened: the first version of
    # this picked a mix of shapes that took a 0.6B from 104 tok/s to 64.
    REP = 24

    def bench():
        o = None
        for _ in range(REP):
            o = _ggml_run(xd, packed, type_name, int(K), int(N), small=state["kind"])
        o.get()

    return tune(key, ("narrow", "shortk", None), apply, bench, check=check,
                default=fallback)


_GQA_TUNED = {}

def gqa_tune(nh, nkv, hd, n, candidates=(4, 8, 16, 32), rounds=5):
    """Pick the split factor for this device at this context length, by running them.

    Cheap enough to do at load: it times the attention kernel alone, not a whole step. The
    answer is remembered per (shape, context bucket) -- `n` is bucketed by powers of four,
    because the ranking moves with the order of magnitude of the scan and not with a token.
    """
    global _GQA_SPLIT, _GQA_SPLIT_ON
    import time as _t
    bucket = 1 << (max(0, int(n)).bit_length() // 2 * 2)
    key = (int(nh), int(nkv), int(hd), int(bucket))
    if key in _GQA_TUNED:
        return _GQA_TUNED[key]
    was, was_on = _GQA_SPLIT, _GQA_SPLIT_ON
    # Sized to the scan being tuned for, not to the cache's capacity: a full-capacity pair
    # is 67MB on an 8k context, which is a lot of allocation to answer a question about
    # thread shape.
    lmax = max(64, int(n))
    q = Tensor(np.zeros((nh, 1, hd), np.float32))
    kc = Tensor(_empty((nkv, lmax, hd)))
    vc = Tensor(_empty((nkv, lmax, hd)))
    mask = Tensor(np.zeros((1, 1, lmax), np.float32))
    try:
        outs = {}
        for sp in candidates:
            _set_split(sp)
            outs[sp] = gqa_decode(q, kc, vc, mask, 1.0, valid=n)
        for o in outs.values():
            o.numpy()                                   # warm and settle
        best, best_ms = was, None
        times = {sp: [] for sp in candidates}
        for _ in range(rounds):
            for sp in candidates:
                _set_split(sp)
                t0 = _t.perf_counter()
                gqa_decode(q, kc, vc, mask, 1.0, valid=n).numpy()
                times[sp].append(_t.perf_counter() - t0)
        for sp, xs in times.items():
            xs.sort()
            med = xs[len(xs) // 2]
            if best_ms is None or med < best_ms:
                best, best_ms = sp, med
        _GQA_TUNED[key] = best
        return best
    except Exception:
        return was
    finally:
        _GQA_SPLIT, _GQA_SPLIT_ON = was, was_on


def _set_split(sp):
    global _GQA_SPLIT, _GQA_SPLIT_ON
    _GQA_SPLIT = int(sp)
    _GQA_SPLIT_ON = int(sp) > 1


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
    if not (_adam_backend_ready() or _webgl_ready()):
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
    # Defaulted here, above the backend split: `gqa_attention` has always filled this in for
    # its callers, and routing the single-position case here instead handed `None` straight
    # to a `float()` on the WebGPU side.
    if scale is None:
        scale = 1.0 / (float(hd) ** 0.5)
    if _webgl_ready() and not _adam_backend_ready():
        # No `ctl`: reading the scan length from a control buffer is what keeps ONE captured
        # dispatch correct across steps, and WebGL does not capture here. `n` is passed as a
        # uniform, which is the same number by a shorter route.
        return Tensor(_webgl_gqa_decode(_contig(qd), _contig(kd), _contig(vd),
                                        nh, nkv, hd, n, scale))
    plat = _adam_kernel["platform"]
    if not _gqa_k["added"]:
        plat.addKernel("gqa_decode", {"source": _GQA_DECODE_WGSL,
                                      "bindingTypes": ["storage"]
                                      + ["read-only-storage"] * 5})
        _gqa_k["added"] = True
    # The split factor is baked into the shader, so each value is its OWN kernel -- named
    # for it, so several can exist at once and be compared on the device that will run them.
    if _GQA_SPLIT not in _gqa_k:
        sub = lambda src: src.replace("SPLITu", "%du" % _GQA_SPLIT)
        plat.addKernel("gqa_split_%d" % _GQA_SPLIT,
                       {"source": sub(_GQA_SPLIT_WGSL),
                        "bindingTypes": ["storage"] + ["read-only-storage"] * 5})
        plat.addKernel("gqa_merge_%d" % _GQA_SPLIT,
                       {"source": sub(_GQA_MERGE_WGSL),
                        "bindingTypes": ["storage", "read-only-storage",
                                         "read-only-storage"]})
        _gqa_k[_GQA_SPLIT] = True
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
    if _GQA_SPLIT_ON:
        # Two dispatches instead of one, and SPLIT times the workgroups in the first. Both
        # shapes are fixed, so the pair captures and replays exactly like the single kernel.
        part = _empty((nh * _GQA_SPLIT * (hd + 2),))
        plat.runKernel({"name": "gqa_split_%d" % _GQA_SPLIT,
                        "tensors": [part.buffer.buffer_id, qc.buffer.buffer_id,
                                    kcc.buffer.buffer_id, vcc.buffer.buffer_id,
                                    meta.buffer_id, cb.buffer_id],
                        "workGroups": {"x": nh * _GQA_SPLIT, "y": 1, "z": 1}})
        plat.runKernel({"name": "gqa_merge_%d" % _GQA_SPLIT,
                        "tensors": [of.buffer.buffer_id, part.buffer.buffer_id,
                                    meta.buffer_id],
                        "workGroups": {"x": nh, "y": 1, "z": 1}})
        return Tensor(of.reshape(nh, 1, hd))
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
    set outside the graph via ctl.buffer.set_data(...) each step.

    Returns the cache to use next: `cache` itself where the backend writes it in place, and
    a NEW buffer on WebGL, where a fragment shader cannot render into a texture it samples.
    The caller stores what comes back -- the same contract the recurrent state uses."""
    if _webgl_ready() and not _adam_backend_ready():
        return _webgl_kv_write(cache, src, pos, T, nkv, hd, lmax)
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
    return cache


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

    def length(self):
        """Positions currently held. A growing cache is as long as it grew; a scatter cache
        has fixed capacity and its live length is the caller's `pos`, not a property of the
        buffers, so it reports None."""
        if self.scatter:
            return None
        k = self.K[0]
        return 0 if k is None else int(k.shape[1])

    def truncate(self, n):
        """Drop everything after position `n`, keeping the first `n`.

        What makes a growing cache reusable across turns: turn N's prompt shares a prefix
        with turn N-1's, and the rows past that prefix -- the last reply, the markup that
        closed it -- have to go before the new tail is appended. Slicing is the whole
        operation; the mask this cache builds is aligned to its end, so a shorter cache is
        simply a cache at an earlier position."""
        if self.scatter:
            return
        for i in range(self.L):
            if self.K[i] is None:
                continue
            if int(self.K[i].shape[1]) <= n:
                continue
            if n <= 0:
                self.K[i] = None; self.V[i] = None
            else:
                self.K[i] = Tensor(_contig(self.K[i].data[:, :n, :]))
                self.V[i] = Tensor(_contig(self.V[i].data[:, :n, :]))
        self._mkey = None                       # the mask is keyed on (pos, T); both changed

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
            self.K[i] = Tensor(kv_write(self.K[i].data, _contig(k).data, pos, T,
                                        self.nkv, self.hd, self.lmax))
            self.V[i] = Tensor(kv_write(self.V[i].data, _contig(v).data, pos, T,
                                        self.nkv, self.hd, self.lmax))
            kc, vc = self.K[i], self.V[i]
            return gqa_attention(q, kc, vc, self._gpu_mask(pos, T), scale)
        # growing cache via cat (both backends)
        if self.K[i] is None:
            self.K[i] = Tensor(_contig(k.data)); self.V[i] = Tensor(_contig(v.data))
        else:
            self.K[i] = cat([self.K[i], k], axis=1); self.V[i] = cat([self.V[i], v], axis=1)
        S = self.K[i].shape[1]
        if T == 1:
            # One query position is the decode step, and the fused kernel covers it: two
            # dispatches against the ten the expression form costs. It reads the cache as it
            # stands, so `valid` is simply its length -- there are no unwritten slots in a
            # cache that grew to fit.
            o = gqa_decode(q, self.K[i], self.V[i], None, scale, valid=S)
            if o is not None:
                return o
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
    out._setback(_backward)
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
# Prefill attention with the score matrix never written down.
#
# The path this replaces materialises it: at a 1536-token prompt the scores are 151MB PER
# LAYER, written by one batched matmul, read by a softmax, read again by a second batched
# matmul. Measured per layer at that length: 133.5ms to write them, 9.7ms to soft-max them,
# 69.7ms to read them back -- 213ms a layer, 6.0s across 28, and the two matmuls run at 72
# and 139 GFLOPS because they are moving a buffer, not doing arithmetic.
#
# Here a workgroup owns BQ queries and walks the keys in tiles of BK, keeping the running
# (max, sum, accumulator) of the online softmax in workgroup memory. Scores exist only
# inside the tile loop. Nothing seq-squared is allocated, read or written at all -- the
# traffic becomes K and V once each instead of the score matrix three times.
#
# The same online-softmax merge as the split decode kernel, and exact for the same reason:
# rescaling a partial sum by exp(m_old - m_new) is what the algorithm already does between
# blocks. Causality is a comparison on indices, so tiles entirely past the diagonal are
# skipped rather than computed and discarded.
_FLASH_WGSL = """@group(0) @binding(0)
var<storage,read_write> outp: array<f32>;
@group(0) @binding(1)
var<storage,read> qg: array<f32>;
@group(0) @binding(2)
var<storage,read> kc: array<f32>;
@group(0) @binding(3)
var<storage,read> vc: array<f32>;
struct FMeta { nkv: u32, rep: u32, T: u32, S: u32, hd: u32, start: u32, scale: f32, }
@group(0) @binding(4)
var<storage,read> fm: FMeta;
var<workgroup> qs: array<f32, BQxHD>;
var<workgroup> kvs: array<f32, BKxHD>;
var<workgroup> sc: array<f32, BQxBK>;
var<workgroup> acc: array<f32, BQxHD>;
var<workgroup> mrun: array<f32, BQu>;
var<workgroup> lrun: array<f32, BQu>;
var<workgroup> crun: array<f32, BQu>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let tiles = (fm.T + BQu - 1u) / BQu;
  let b = wid.x / tiles;                 // which (kv, rep) row block
  let tq = wid.x % tiles;                // which query tile inside it
  let kv = b / fm.rep;
  let i0 = tq * BQu;                     // first query index of this tile
  let qbase = (b * fm.T + i0) * fm.hd;
  let kvbase = kv * fm.S * fm.hd;

  // Queries and the accumulator stay resident for the whole scan; the scores never leave.
  for (var x: u32 = t; x < BQu * fm.hd; x = x + 128u) {
    let r = x / fm.hd;
    qs[x] = select(0.0, qg[qbase + x], i0 + r < fm.T);
    acc[x] = 0.0;
  }
  if (t < BQu) { mrun[t] = -1e30; lrun[t] = 0.0; crun[t] = 1.0; }
  workgroupBarrier();

  // Only keys the LAST query of this tile can see are worth visiting: everything past the
  // diagonal is skipped rather than computed and thrown away.
  let hi = min(fm.S, fm.start + min(i0 + BQu - 1u, fm.T - 1u) + 1u);
  var j0: u32 = 0u;
  loop {
    if (j0 >= hi) { break; }
    let jn = min(BKu, hi - j0);
    for (var x: u32 = t; x < BKu * fm.hd; x = x + 128u) {
      let j = x / fm.hd;
      kvs[x] = select(0.0, kc[kvbase + (j0 + j) * fm.hd + (x % fm.hd)], j < jn);
    }
    workgroupBarrier();

    for (var x: u32 = t; x < BQu * BKu; x = x + 128u) {
      let r = x / BKu;
      let j = x % BKu;
      var d0: f32 = -1e30;
      if (j < jn && i0 + r < fm.T && j0 + j <= fm.start + i0 + r) {
        var dd: f32 = 0.0;
        for (var d: u32 = 0u; d < fm.hd; d = d + 1u) {
          dd = dd + qs[r * fm.hd + d] * kvs[j * fm.hd + d];
        }
        d0 = dd * fm.scale;
      }
      sc[x] = d0;
    }
    workgroupBarrier();

    // One lane per query row folds this tile into that row's running softmax, and leaves
    // behind the factor the accumulator has to be rescaled by.
    if (t < BQu) {
      var m_new: f32 = mrun[t];
      for (var j: u32 = 0u; j < BKu; j = j + 1u) {
        m_new = max(m_new, sc[t * BKu + j]);
      }
      var ssum: f32 = 0.0;
      for (var j: u32 = 0u; j < BKu; j = j + 1u) {
        let v = sc[t * BKu + j];
        let e = select(0.0, exp(v - m_new), v > -1e29);
        sc[t * BKu + j] = e;
        ssum = ssum + e;
      }
      let corr = select(0.0, exp(mrun[t] - m_new), mrun[t] > -1e29);
      crun[t] = corr;
      lrun[t] = lrun[t] * corr + ssum;
      mrun[t] = m_new;
    }
    workgroupBarrier();

    // V reuses the staging K is done with; the scores for this tile are already exp'd.
    for (var x: u32 = t; x < BKu * fm.hd; x = x + 128u) {
      let j = x / fm.hd;
      kvs[x] = select(0.0, vc[kvbase + (j0 + j) * fm.hd + (x % fm.hd)], j < jn);
    }
    workgroupBarrier();

    for (var x: u32 = t; x < BQu * fm.hd; x = x + 128u) {
      let r = x / fm.hd;
      let d = x % fm.hd;
      var o: f32 = acc[x] * crun[r];
      for (var j: u32 = 0u; j < BKu; j = j + 1u) {
        o = o + sc[r * BKu + j] * kvs[j * fm.hd + d];
      }
      acc[x] = o;
    }
    workgroupBarrier();
    j0 = j0 + BKu;
  }

  for (var x: u32 = t; x < BQu * fm.hd; x = x + 128u) {
    let r = x / fm.hd;
    if (i0 + r < fm.T) {
      outp[qbase + x] = acc[x] / max(lrun[r], 1e-30);
    }
  }
}
"""


_flash_k = {}
# Tile shape. BQ*hd + BK*hd + BQ*BK + BQ*hd floats of workgroup memory must fit the 32KB
# limit: at hd=128 that is (8+32+8)*128 + 8*32 = 6400 floats = 25.6KB. Both are candidates
# for `tune` once there is a second machine to disagree about them.
_FLASH_BQ = 16
_FLASH_BK = 8


def flash_tune(nh, nkv, hd, T=256):
    """Pick the tile shape on this device. The candidates that fit 32KB of workgroup memory
    differ by more than 2x on one machine -- (8,32) is the slowest of them and was the value
    guessed first -- so this is measured, not chosen.

    Tuned once at a short sequence and reused for all of them: the ranking is about how many
    queries one loaded K tile serves, which does not change with length. Measured, the order
    is identical at 256, 512 and 1536 tokens.
    """
    global _FLASH_BQ, _FLASH_BK
    key = ("flash_tile", int(nh), int(nkv), int(hd))
    if key in _TUNED:
        _FLASH_BQ, _FLASH_BK = _TUNED[key]
        return _TUNED[key]
    if _adam_kernel.get("platform") is None:
        return (_FLASH_BQ, _FLASH_BK)
    cand = [(bq, bk) for bq, bk in ((16, 8), (8, 16), (16, 16), (24, 8), (8, 32))
            if 2 * bq * int(hd) + bk * int(hd) + bq * bk <= 8192]
    q = Tensor(np.zeros((nh, T, hd), np.float32))
    k = Tensor(_empty((nkv, T, hd)))
    v = Tensor(_empty((nkv, T, hd)))
    was = (_FLASH_BQ, _FLASH_BK)

    def apply(p):
        global _FLASH_BQ, _FLASH_BK
        _FLASH_BQ, _FLASH_BK = p

    def bench():
        _contig(flash_attention(q, k, v, start=0, scale=1.0).data[:1, :1, :1]).get()

    best = tune(key, cand, apply, bench, default=was)
    _FLASH_BQ, _FLASH_BK = best
    return best


def flash_attention(q, k, v, start=0, scale=None):
    """Causal grouped-query attention that never writes the score matrix.

    q (nh, T, hd); k, v (nkv, S, hd). Query i attends to keys 0..start+i, which is what a
    continued prefill needs -- the cache already holds `start` positions before these.
    """
    nh, T, hd = q.shape
    nkv, S, _ = k.shape
    rep = nh // nkv
    if scale is None:
        scale = 1.0 / (float(hd) ** 0.5)
    plat = _adam_kernel["platform"]
    key = (_FLASH_BQ, _FLASH_BK, int(hd))
    if key not in _flash_k:
        src = (_FLASH_WGSL.replace("BQxHD", str(_FLASH_BQ * int(hd)))
                          .replace("BKxHD", str(_FLASH_BK * int(hd)))
                          .replace("BQxBK", str(_FLASH_BQ * _FLASH_BK))
                          .replace("BQu", "%du" % _FLASH_BQ)
                          .replace("BKu", "%du" % _FLASH_BK))
        plat.addKernel("flash_%d_%d_%d" % key,
                       {"source": src,
                        "bindingTypes": ["storage"] + ["read-only-storage"] * 4})
        _flash_k[key] = True
    qg = _contig(q.reshape(nkv, rep * T, hd).data)
    kc = _contig(k.data)
    vc = _contig(v.data)
    of = _empty((nkv * rep * T * hd,))
    meta = _adam_kernel["make_meta"]((nkv, rep, T, S, hd, int(start), float(scale)),
                                     "u4,u4,u4,u4,u4,u4,f4")
    tiles = (T + _FLASH_BQ - 1) // _FLASH_BQ
    plat.runKernel({"name": "flash_%d_%d_%d" % key,
                    "tensors": [of.buffer.buffer_id, qg.buffer.buffer_id,
                                kc.buffer.buffer_id, vc.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": nkv * rep * tiles, "y": 1, "z": 1}})
    return Tensor(of.reshape(nh, T, hd))


# Scale, causal mask and softmax in ONE pass over the score matrix.
#
# The prefill path used to do them as four separate ones -- `a * scale`, `a + mask`, then the
# fused softmax -- and the thing being traversed is seq-squared: at a 1536-token prompt the
# scores are 151MB per layer, so each extra traversal is 300MB of reading and writing, 28
# times over. Measured, prefill attention ran at about 7 GB/s for that reason, and at 1536
# tokens it was 4.7s of an 8.2s prefill.
#
# The mask does not exist here at all. A causal mask is pure structure -- column j is allowed
# for query i exactly when j <= start + i -- so it is a comparison on indices, not 9.4MB of
# -1e9 built on the HOST with np.triu and uploaded. And knowing where the row ends means the
# loops stop there: the masked half was still being read and exponentiated to produce zeros.
_SOFTMAX_CAUSAL_WGSL = """@group(0) @binding(0)
var<storage,read> inp: array<f32>;
@group(0) @binding(1)
var<storage,read_write> outp: array<f32>;
struct CMeta { rows: u32, width: u32, T: u32, start: u32, scale: f32, }
@group(0) @binding(2)
var<storage,read> cmeta: CMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= cmeta.rows) { return; }
  let base = row * cmeta.width;
  // Rows run (kv, rep, T) so the query index is the row within its T block.
  let i = row % cmeta.T;
  let lim = min(cmeta.width, cmeta.start + i + 1u);
  var mx: f32 = inp[base] * cmeta.scale;
  for (var j: u32 = 1u; j < lim; j = j + 1u) {
    let v = inp[base + j] * cmeta.scale;
    if (v > mx) { mx = v; }
  }
  var sm: f32 = 0.0;
  for (var j: u32 = 0u; j < lim; j = j + 1u) {
    sm = sm + exp(inp[base + j] * cmeta.scale - mx);
  }
  for (var j: u32 = 0u; j < lim; j = j + 1u) {
    outp[base + j] = exp(inp[base + j] * cmeta.scale - mx) / sm;
  }
  // Everything past the diagonal is exactly zero, and has to be written: the buffer is
  // reused and the matmul that follows reads the whole row.
  for (var j: u32 = lim; j < cmeta.width; j = j + 1u) { outp[base + j] = 0.0; }
}
"""
_softmax_causal_k = {"added": False}


def _fused_causal_softmax(xd, T, start, scale):
    """scale -> causal mask -> softmax, in one pass. `xd` is (..., rows, width) with the
    query index running fastest over `T` inside each block."""
    plat = _adam_kernel["platform"]
    if not _softmax_causal_k["added"]:
        plat.addKernel("softmax_causal", {
            "source": _SOFTMAX_CAUSAL_WGSL,
            "bindingTypes": ["read-only-storage", "storage", "read-only-storage"]})
        _softmax_causal_k["added"] = True
    width = int(xd.shape[-1])
    rows = int(xd.size) // width
    of = _empty(xd.shape)
    meta = _adam_kernel["make_meta"]((rows, width, int(T), int(start), float(scale)),
                                     "u4,u4,u4,u4,f4")
    plat.runKernel({"name": "softmax_causal",
                    "tensors": [_contig(xd).buffer.buffer_id, of.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
    return of


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
    out._setback(_backward)
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
    out._setback(_backward)
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


_SWIGLU_WGSL = """@group(0) @binding(0)
var<storage,read> g: array<f32>;
@group(0) @binding(1)
var<storage,read> u: array<f32>;
@group(0) @binding(2)
var<storage,read_write> outp: array<f32>;
@group(0) @binding(3)
var<storage,read> sw: SW;
struct SW { half: u32, gstride: u32, ustride: u32, uoff: u32, }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= sw.half) { return; }
  let r = gid.y;
  let x = g[r * sw.gstride + c];
  // silu(x) * y, written as the division rather than x * sigmoid(x): same value, and the
  // reciprocal is one instruction where the graph form was neg, exp, add, div and mul --
  // five dispatches, each reading and writing the whole tensor.
  outp[r * sw.half + c] = (x / (1.0 + exp(-x))) * u[r * sw.ustride + sw.uoff + c];
}
"""

_swiglu_k = {"added": False}


def swiglu(gate, up=None):
    """silu(gate) * up, in one dispatch.

    With `up` omitted, `gate` holds both halves along its last axis -- the layout a fused
    gate/up projection produces -- and the two halves are read in place, so the slices that
    would otherwise be materialised never exist.

    This is the whole activation of a SwiGLU MLP, which is every Llama-family model, dense or
    routed. The graph form costs six dispatches (neg, exp, add, div for the sigmoid, then two
    multiplies) plus a copy per slice, each of them a full pass over the tensor; a routed 30B
    spent 240 elementwise multiplies and 192 sigmoid fragments a token on it.

    Returns None when there is no GPU backend, so callers keep their own expression.
    Inference-only: no autograd node is built.
    """
    if not (_adam_backend_ready() or _webgl_ready()):
        return None
    gd = gate.data if isinstance(gate, Tensor) else gate
    shape = tuple(gd.shape)
    rows = 1
    for d in shape[:-1]:
        rows *= int(d)
    if up is None:
        w = int(shape[-1])
        if w % 2:
            return None
        half, gstride, ustride, uoff = w // 2, w, w, w // 2
        ud = gd
    else:
        ud = up.data if isinstance(up, Tensor) else up
        if tuple(ud.shape) != shape:
            return None
        half = gstride = ustride = int(shape[-1])
        uoff = 0
    gd = _contig(gd)
    ud = gd if up is None else _contig(ud)
    if _webgl_ready() and not _adam_backend_ready():
        of = _webgl_swiglu(gd, ud, rows, half, gstride, ustride, uoff)
        return Tensor(of.reshape(*(shape[:-1] + (half,))))
    plat = _adam_kernel["platform"]
    if not _swiglu_k["added"]:
        plat.addKernel("swiglu", {"source": _SWIGLU_WGSL,
                                  "bindingTypes": ["read-only-storage", "read-only-storage",
                                                   "storage", "read-only-storage"]})
        _swiglu_k["added"] = True
    of = _empty((rows, half))
    meta = _adam_kernel["make_meta"]((half, gstride, ustride, uoff), "u4,u4,u4,u4")
    plat.runKernel({"name": "swiglu",
                    "tensors": [gd.buffer.buffer_id, ud.buffer.buffer_id,
                                of.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (half + 63) // 64, "y": rows, "z": 1}})
    return Tensor(of.reshape(*(shape[:-1] + (half,))))


def rmsnorm(x, w, eps):
    """Fused RMS norm: `x * rsqrt(mean(x^2) + eps) * w`, one dispatch instead of six.

    Returns None when there is no GPU backend, so callers keep their own expression.
    Inference-only: no autograd node is built, which is why the graph path stays intact.
    """
    if not (_adam_backend_ready() or _webgl_ready()):
        return None
    xd = x.data if isinstance(x, Tensor) else x
    wd = w.data if isinstance(w, Tensor) else w
    shape = tuple(xd.shape)
    H = int(shape[-1])
    T = 1
    for d in shape[:-1]:
        T *= int(d)
    if _webgl_ready() and not _adam_backend_ready():
        of = _webgl_rmsnorm(_contig(xd), _contig(wd), T, H, eps)
        return Tensor(of.reshape(*shape))
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
struct PMeta { n: u32, HD: u32, rd: u32, T: u32, }
@group(0) @binding(4)
var<storage,read> pm: PMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= pm.n) { return; }
  let d = i % pm.HD;
  // cos/sin are (T, HD) against an x of (heads, T, HD), so the row is the middle axis.
  // At T = 1 this is the single-position form the kernel started as.
  let trow = (i / pm.HD) % pm.T;
  let ci = trow * pm.HD + d;
  let half = pm.rd / 2u;
  var rot: f32;
  if (d < half) {
    rot = -x[i + half];
  } else if (d < pm.rd) {
    rot = x[i - half];
  } else {
    rot = x[i];                      // pass-through tail: sin is 0 here, value is inert
  }
  outp[i] = x[i] * cosb[ci] + rot * sinb[ci];
}
"""
_rope_k = {"added": False}


def rope_decode(x, cos, sin, HD, rd, T=1):
    """Fused rotary embedding: `x*cos + rotate_half(x)*sin`, in one dispatch.

    Returns None without a GPU backend so callers keep their expression. `cos`/`sin` hold
    `T` rows of length HD, against an `x` of (heads, T, HD). `rd` is the rotated prefix for
    partial rope; the tail passes through unchanged, matching the unfused form.

    `T` above 1 is prefill, and it is worth having: the expression this replaces costs eight
    dispatches per tensor per layer whatever T is, so prefill pays the same launch overhead
    decode does, on top of doing more work.
    """
    if not (_adam_backend_ready() or _webgl_ready()):
        return None
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    shape = tuple(xd.shape)
    n = 1
    for d in shape:
        n *= int(d)
    if _webgl_ready() and not _adam_backend_ready():
        cd = _contig(cos.data if isinstance(cos, Tensor) else cos)
        sd = _contig(sin.data if isinstance(sin, Tensor) else sin)
        return Tensor(_webgl_rope(xd, cd, sd, n, HD, rd, T).reshape(*shape))
    plat = _adam_kernel["platform"]
    if not _rope_k["added"]:
        plat.addKernel("rope", {"source": _ROPE_WGSL,
                                "bindingTypes": ["read-only-storage"] * 3 + ["storage",
                                                                             "read-only-storage"]})
        _rope_k["added"] = True
    cd = _contig(cos.data if isinstance(cos, Tensor) else cos)
    sd = _contig(sin.data if isinstance(sin, Tensor) else sin)
    of = _empty((n,))
    meta = _adam_kernel["make_meta"]((n, int(HD), int(rd), int(T)), "u4,u4,u4,u4")
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
        out._setback(_bw)
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
    out._setback(_backward)
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
        onehot = xp.asarray(_onehot(targets, N, Cls))
        loss_val = (-(onehot * xp.log(s + 1e-12)).sum()) * (1.0 / N)
        out = Tensor(loss_val.reshape(()), logits.requires_grad, (logits,), "cross_entropy")

        def _bw():
            if logits.requires_grad:
                logits._accum((s - onehot) * (1.0 / N) * out.grad)
        out._setback(_bw)
        return out

    tgt = xp.asarray(np.asarray(targets).astype(np.float32))
    s = _fused_softmax(xd) if _adam_backend_ready() else _webgl_softmax(xd)
    perrow = _ce_fwd(s, tgt, N, Cls)
    loss_val = perrow.sum() * (1.0 / N)
    out = Tensor(loss_val.reshape(()), logits.requires_grad, (logits,), "cross_entropy")

    def _backward():
        if logits.requires_grad:
            logits._accum(_ce_bwd(s, tgt, N, Cls) * out.grad)
    out._setback(_backward)
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
        name = cp.get_backend_name()
        if name != "webgpu":
            _backend_why["backend_name"] = str(name)
            return False
        from wgpy_backends.webgpu.platform import get_platform
        from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
        _adam_kernel["platform"] = get_platform()
        _adam_kernel["make_meta"] = create_meta_buffer_from_structure
        return True
    except Exception as e:
        # Kept, not swallowed. This except used to return False and lose the reason, which
        # is why a report of 0.5 tok/s on a machine that should manage hundreds could not be
        # diagnosed from anything the page knew.
        _backend_why["platform"] = "%s: %s" % (type(e).__name__, e)
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


def _gpu_release_memory():
    """Give the device back every byte a released model was still holding.

    Called from the release path AFTER the heavy attributes were nulled, i.e. the
    buffer finalizers have already run. Two classes of memory never come back on
    their own: buffers pinned by a captured decode graph (their finalizer drops
    them instead of pooling, and JS refuses their disposeBuffer while pinned), and
    the reuse pool itself, which only pays off when the NEXT model wants the same
    shapes. Left alone, a released model keeps its whole GPU footprint, and the
    next model allocates on top of it until the device dies (seen for real:
    releasing a 0.6B, then loading a 30B-A3B loses the device mid-load)."""
    try:
        if _adam_backend_ready():
            from wgpy_backends.webgpu import webgpu_buffer as _wb
            plat = _adam_kernel["platform"]
        elif _webgl_ready():
            from wgpy_backends.webgl import webgl_buffer as _wb
            plat = _copy_kernel["plat"]
        else:
            return
    except Exception:
        return
    try:
        import gc
        gc.collect()   # buffers caught in reference cycles only reach the pools once collected
        # Order matters, and the worker->main channel is FIFO: resetCaptures must
        # clear the JS-side pin set before the disposeBuffer messages below arrive,
        # or every pinned buffer is refused and never freed.
        plat.resetCaptures()
        _wb.release_capture_buffers()
        _wb.release_pooled_buffers()
    except Exception:
        pass


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
struct GM { M: u32, N: u32, K: u32, rowb: u32, estride: u32, eslot: u32, xper: u32, pad: u32, }
@group(0) @binding(3)
var<storage,read> gm: GM;
// Byte addressing, not word: most ggml blocks are not a multiple of four bytes (Q8_0 is 34,
// Q3_K and IQ3_S 110, MXFP4 17), so a u32 index would drift out of alignment after the
// first block. These mirror the reference dequantizer's own byte offsets exactly.
var<private> nrow: u32;
MOEVARS
fn W(wo: u32) -> u32 { return w[WOFSwo * gm.N + nrow]; }
fn B(o: u32) -> u32 { return (W(o >> 2u) >> ((o & 3u) * 8u)) & 255u; }
fn I8(o: u32) -> f32 { return f32(i32(B(o) << 24u) >> 24u); }
// Four consecutive bytes at byte offset `o`, packed little-endian, in ONE load when `o` is
// word-aligned and two when it is not. `B` is a whole load per byte, so anything reading a
// field a byte at a time paid four loads for four bytes that are almost always in the same
// word -- and a decode fragment does that thousands of times per block. Blocks whose size is
// not a multiple of four (Q3_K and IQ3_S are 110 bytes, Q6_K 210) put `o` at any alignment,
// which is why the aligned fast path cannot simply be assumed.
//
// The second load stays inside the row: `W` strides by gm.N, so wo+1 is the next word of
// THIS row, not the next row. Past the last word of a row it reads whatever follows, but
// every field this is used for has its bytes inside the block, so those bits are masked off.
fn B4(o: u32) -> u32 {
  let wo = o >> 2u;
  let sh = (o & 3u) * 8u;
  let lo = W(wo);
  if (sh == 0u) { return lo; }
  return (lo >> sh) | (W(wo + 1u) << (32u - sh));
}
fn U16(o: u32) -> u32 { return B4(o) & 65535u; }
fn U32(o: u32) -> u32 { return B4(o); }
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
// The block's activations, staged once for the whole workgroup: KSG blocks in flight, four
// rows each. Every one of the 64 threads in x holds a different OUTPUT COLUMN and therefore
// reads the same activations, so without this each of them fetches them again from global
// memory. Counted as traffic that is not close: 16*N*K bytes of activation against N*K*0.33
// of weight for a 2-bit format -- about fifty times more spent on re-reading the same rows
// than on the weights the pass exists to read. It is why the batched path ran at a tenth of
// the decode path's bandwidth, and why widening the rows made it worse rather than better.
var<workgroup> xs4: array<f32, XS4SZ>;
var<private> a0: f32;
var<private> a1: f32;
var<private> a2: f32;
var<private> a3: f32;
var<private> mb: u32;
var<private> mn: u32;
var<private> kbase: u32;
var<private> xsoff: u32;
var<private> live: bool;
// Separate scalars, not array<f32,4>. A private array indexed by a loop variable is not
// guaranteed to live in registers, and here it did not: the batched matmul cost time
// strictly proportional to the batch (12.3 / 35.5 / 94.2 ms at M = 8 / 24 / 64) because
// every accumulate went to memory. gemv was always fast for the same reason in reverse --
// it accumulates into one scalar.
fn ACC(k: u32, v: f32) {
  if (!live) { return; }
  let s = xsoff + (k - kbase);
  a0 = a0 + xs4[s] * v;
  if (mn > 1u) { a1 = a1 + xs4[s + BLKVALS] * v; }
  if (mn > 2u) { a2 = a2 + xs4[s + 2u * BLKVALS] * v; }
  if (mn > 3u) { a3 = a3 + xs4[s + 3u * BLKVALS] * v; }
}
fn ACC4(k: u32, v: vec4<f32>) {
  if (!live) { return; }
  let s = xsoff + (k - kbase);
  a0 = a0 + dot(vec4<f32>(xs4[s], xs4[s + 1u], xs4[s + 2u], xs4[s + 3u]), v);
  if (mn > 1u) { let c = s + BLKVALS;
    a1 = a1 + dot(vec4<f32>(xs4[c], xs4[c + 1u], xs4[c + 2u], xs4[c + 3u]), v); }
  if (mn > 2u) { let c = s + 2u * BLKVALS;
    a2 = a2 + dot(vec4<f32>(xs4[c], xs4[c + 1u], xs4[c + 2u], xs4[c + 3u]), v); }
  if (mn > 3u) { let c = s + 3u * BLKVALS;
    a3 = a3 + dot(vec4<f32>(xs4[c], xs4[c + 1u], xs4[c + 2u], xs4[c + 3u]), v); }
}
"""

_GGML_GEMM_MAIN = """
@compute @workgroup_size(64, KSGu)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let lx = lid.x;
  let ly = lid.y;
WOFFINIT
GEMMINIT
  // From `workgroup_id`, not from the private `mb`/`mn` below, and not from
  // `global_invocation_id`: the loop that follows contains barriers, so its condition has to
  // be UNIFORM, and uniformity here is decided syntactically. A module-scope private is
  // treated as possibly non-uniform however it was assigned -- the compiler rejected exactly
  // that -- while `workgroup_id` is uniform by definition. The z dimension of the workgroup
  // size is 1, so this is the same number the private one holds.
  // Each lane row owns four output rows of its own. From `workgroup_id`, so the loop below
  // -- which contains barriers -- has a bound every lane agrees on; the per-lane part is the
  // row group, which only selects data.
  let mgroup = wgid.z * (4u * KSGu);
  let mbu = mgroup + ly * 4u;
  let anyrow = mgroup < gm.M;
  mb = mbu;
  mn = select(0u, min(4u, gm.M - mbu), mbu < gm.M);
  nrow = n;
  a0 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0;
  xsoff = ly * 4u * BLKVALS;
  live = (n < gm.N);
  let base = 0u;
  let nb = gm.K / BLKVALS;
  let livec = (n < gm.N && mn > 0u);
  // `mn` comes from gid.z, so this is uniform across the workgroup and the barriers below
  // are reached by every lane. The per-thread part of "is this lane live" is `live`, which
  // gates the accumulate rather than the control flow -- a barrier inside a branch that
  // only some lanes take is undefined behaviour, not a slow path.
  if (anyrow) {
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
      kbase = b * BLKVALS;
      for (var t: u32 = lx; t < BLKVALS; t = t + 64u) {
        let sx = mbu * gm.K + kbase + t;
        xs4[xsoff + t] = select(0.0, x[sx], mn > 0u);
        xs4[xsoff + BLKVALS + t] = select(0.0, x[sx + gm.K], mn > 1u);
        xs4[xsoff + 2u * BLKVALS + t] = select(0.0, x[sx + 2u * gm.K], mn > 2u);
        xs4[xsoff + 3u * BLKVALS + t] = select(0.0, x[sx + 3u * gm.K], mn > 3u);
      }
      workgroupBarrier();
"""

_GGML_GEMM_TAIL = """
      workgroupBarrier();
    }
  }
  if (live) {
    if (mn > 0u) { outp[mbu * gm.N + n] = a0; }
    if (mn > 1u) { outp[(mbu + 1u) * gm.N + n] = a1; }
    if (mn > 2u) { outp[(mbu + 2u) * gm.N + n] = a2; }
    if (mn > 3u) { outp[(mbu + 3u) * gm.N + n] = a3; }
  }
}
"""

# The batched path splits the ROW dimension across `lid.y`, not K.
#
# Splitting K was what it did, and it is incompatible with staging the activations: the loop
# bound becomes `lid.y`, each lane walks a different number of blocks, and a barrier inside a
# loop whose trip count differs per lane is rejected by the compiler outright. Splitting rows
# instead makes the block loop identical for every lane -- so the barrier is legal -- and
# costs nothing, because each lane then owns four output rows outright and the cross-lane
# reduction the split-K version needed disappears with it.
#
# It does NOT pay to widen it past one, and the measurement says why. Four lane rows cover
# sixteen output rows per pass instead of four, which looked like a fourfold cut in weight
# traffic and delivered nothing (mlp 66.5s at one lane row, 70.1s at four). The reason is
# that `dec` runs per THREAD: every lane row decodes the same weight block again, so wider
# lanes buy more rows and pay for them in repeated decode. Sharing the DECODED block through
# workgroup memory is what would actually cut it, and that is a different kernel.
# Row groups (of 4 rows each) per workgroup on the batched path. A workgroup covers 4*KSG
# activation rows, so the weights it reads serve that many -- which divides the weight
# traffic by 4*KSG and looks like the obvious lever for prefill.
#
# It is not one, and the measurement says why. Interleaved medians, 0.6B, all 28 layers:
#
#     T       KSG=1     KSG=2     KSG=4
#     512    1180.5    1175.6    1165.8   ms
#     1536   3421.2    3374.8    3405.2   ms
#
# 1.3% apart, which is noise. The arithmetic is invariant to KSG (2*M*N*K either way) and it
# comes out at 395 / 401 / 397 GFLOPS, while the weight traffic the change actually removes
# would have been 29.5 / 15.0 / 7.4 GB/s. Constant compute, collapsing traffic, unchanged
# time: this kernel is bound by the arithmetic, not by re-reading weights, so nothing that
# only moves traffic can help it. (For scale, the generic fp32 batched matmul next door runs
# at 72-139 GFLOPS; this one is already the fast path.)
#
# So KSG is deliberately NOT a tuned knob -- measuring it on every load would cost time to
# rediscover a number that does not matter.
_GGML_KSG = 1

# GEMV (decode, batch of one): the shape where the naive kernel loses. Two things fix it,
# both of which ggml blocks happen to suit. Blocks are independent, so KS rows of threads
# can each take every KS-th block and the partial sums are added at the end -- KS times the
# parallelism on a matmul that is otherwise one long serial walk per output column. And all
# 64 threads in a row want the SAME activations, so a block's worth is staged in workgroup
# memory once instead of being re-read from global memory 64 times.
# Bound only by the MoE variant of a kernel. `eidx[slot]` is the expert the router chose
# for that slot on this token; the host rewrites this small buffer each step, and the captured
# command list never changes. This mirrors what llama.cpp does with ggml_mul_mat_id: one
# stacked expert tensor plus an index, rather than a separate dispatch per expert.
_MOE_VARS = """// Base word offset into `w`: all experts of a projection live in ONE buffer and this
// selects the slice, so the dispatch is the same command whichever expert the router
// picked -- which is what lets a MoE step be captured at all.
var<private> woff: u32;
// Which routed slot this invocation serves; also selects the output row.
var<private> oslot: u32;
// Where this slot's input starts. A routed layer's first two projections share one
// input row; the third takes the row that projection produced for the same slot.
var<private> xbase: u32;"""


_MOE_BIND = """
@group(0) @binding(4)
var<storage,read> eidx: array<u32>;
"""

# The activation window comes in two shapes and the format picks one.
#
# `ACC4` reads four consecutive activations for every four weights it decodes. As four scalar
# indices that is four workgroup-memory accesses where one vec4 access would do, and it is the
# second-largest cost in the decode path after the decode arithmetic -- isolated by running
# the kernel with the same reads and the same number of ACC4 calls but no decode maths, which
# came out at 97.2 GB/s against 121.1 for the reads alone on IQ3_S. Making the window a vec4
# array is worth 17-21% on every format that decodes four at a time.
#
# Reading the bytes runs at 106-121 GB/s for EVERY format, so there is no memory problem to
# find here -- and GB/s is the wrong way to compare two formats anyway, because the work is
# per value and a 2-bit format packs more values into each byte. In values per second the
# formats are within 15% of each other:
#   IQ4_XS 1.88 vals/byte  116.8 GB/s  220 G/s     IQ2_S   3.12  74.6  233 G/s
#   IQ3_S  2.33            85.4        199         IQ2_XS  3.46  66.5  230
#   IQ3_XXS 2.61           86.8        227
# IQ2_XS at 66.5 GB/s is decoding MORE values per second than IQ4_XS at 116.8. Asking it to
# reach 100 GB/s is asking for 346 G values/s, half again what anything here achieves.
#
# It is worth MINUS 30% on the ones that do not. Twenty formats accumulate a value at a time,
# and a scalar read from a vec4 window is a dynamic component index -- Q4_K measured 95.8 ->
# 68.2 GB/s that way. So the window type follows the fragment: four-at-a-time formats get
# vec4, the rest keep the float array they were already fast with. (It is also what keeps
# F16/F32/BF16 working at all: their "block" is a single value, so there is no vec4 to fill.)
_GGML_GEMV_PRE_F32 = """
var<workgroup> xs: array<f32, KSxBLKxR>;
XSCOMMON
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

_GGML_GEMV_PRE_V4 = """
var<workgroup> xs: array<vec4<f32>, XSVEC4N>;
XSCOMMON
fn ACC(k: u32, v: f32) {
  let i = xoff + (k & MASKBLK);
  acc0 = acc0 + xs[i >> 2u][i & 3u] * v;
ACCBODY1
}
fn ACC4(k: u32, v: vec4<f32>) {
  // Every ACC4 call site indexes a multiple of four -- checked across all eight formats that
  // use it, and `MASKBLK` and `xoff` both preserve it -- so the vec4 index is just i >> 2.
  let i = xoff + (k & MASKBLK);
  acc0 = acc0 + dot(xs[i >> 2u], v);
ACC4BODY1
}
"""

_XS_COMMON = """var<workgroup> psum: array<f32, PSUMSZ>;
var<private> accs: array<f32, ORW>;
var<private> acc0: f32;
ACCDECL1
var<private> xoff: u32;"""

# The fill matches how the window is read: one float per thread per pass, or one vec4.
_XS_FILL_F32 = """    for (var t: u32 = lx; t < BLKVALS; t = t + WGXu) {
      var xv: f32 = 0.0;
      if (b < nb) { xv = x[XBASb * BLKVALS + t]; }
      xs[xoff + t] = xv;
XLOAD1
    }"""

_XS_FILL_V4 = """    for (var t: u32 = lx * 4u; t < BLKVALS; t = t + WGXu * 4u) {
      var xv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      if (b < nb) {
        let sx = XBASb * BLKVALS + t;
        xv = vec4<f32>(x[sx], x[sx + 1u], x[sx + 2u], x[sx + 3u]);
      }
      xs[(xoff + t) >> 2u] = xv;
XLOAD1
    }"""


_GGML_GEMV_MAIN = """
@compute @workgroup_size(WGX, KSu)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = gid.x;
  let rowbase = wid.x * (WGXu * ORWu);
  let lx = lid.x;
  let ly = lid.y;
WOFFINIT
  xoff = ly * BLKVALS;
HELPERINIT
  acc0 = 0.0;
ACCINIT1
  for (var q: u32 = 0u; q < ORWu; q = q + 1u) { accs[q] = 0.0; }
  let base = 0u;
  let nb = gm.K / BLKVALS;
  // A fixed step count, not `b < nb` per row: every thread must reach the same barriers.
  let steps = (nb + KSu - 1u) / KSu;
  for (var st: u32 = 0u; st < steps; st = st + 1u) {
    let b = st * KSu + ly;
XSFILL
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
        outp[OSLTnn] = tot;
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
// The IQ4 codebook, staged in workgroup memory once per workgroup and then read.
//
// Computing it per value -- three selects, a shift, a mask and a sign-extend, packed into
// four u32s because a dynamically indexed `const` array is not uniformly implemented -- is
// about ten instructions, and it runs once for every quant in the tensor. Sixteen floats of
// workgroup memory turn that into one load. Streaming all 80 IQ4_XS tensors in a 27B:
// 63.3 -> 86.0 GB/s, against 100 GB/s for reading the same bytes and decoding nothing.
var<workgroup> kvtab: array<f32, 16>;
fn kvfill(t: u32) {
  if (t < 16u) {
    let lo = select(0xBFAD9881u, 0xF6EADDCFu, (t & 4u) != 0u);
    let hi = select(0x26190D01u, 0x71594535u, (t & 4u) != 0u);
    let p = select(lo, hi, (t & 8u) != 0u);
    kvtab[t] = f32(i32(((p >> (8u * (t & 3u))) & 255u) << 24u) >> 24u);
  }
}
fn kv(i: u32) -> f32 { return kvtab[i]; }
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
      // Four values at a time. Every offset here is 4-aligned (the block is 176 bytes and
      // both sub-arrays start on a word), so the quants and the high bits each come from ONE
      // word read instead of four byte extractions, and the results land on four consecutive
      // activations -- one dot product each.
      let hb0 = 1u << i0;
      let hb1 = 1u << (i0 + 1u);
      for (var lw: u32 = 0u; lw < 8u; lw = lw + 1u) {
        let l = lw * 4u;
        let qw = W((qo + g * 32u + l) >> 2u);
        let hw = W((ho + l) >> 2u);
        let a0 = qw & 255u; let a1 = (qw >> 8u) & 255u;
        let a2 = (qw >> 16u) & 255u; let a3 = (qw >> 24u) & 255u;
        let e0 = hw & 255u; let e1 = (hw >> 8u) & 255u;
        let e2 = (hw >> 16u) & 255u; let e3 = (hw >> 24u) & 255u;
        let vlo = vec4<f32>(
          f32(a0 & 15u) + select(0.0, 16.0, (e0 & hb0) != 0u),
          f32(a1 & 15u) + select(0.0, 16.0, (e1 & hb0) != 0u),
          f32(a2 & 15u) + select(0.0, 16.0, (e2 & hb0) != 0u),
          f32(a3 & 15u) + select(0.0, 16.0, (e3 & hb0) != 0u));
        let vhi = vec4<f32>(
          f32(a0 >> 4u) + select(0.0, 16.0, (e0 & hb1) != 0u),
          f32(a1 >> 4u) + select(0.0, 16.0, (e1 & hb1) != 0u),
          f32(a2 >> 4u) + select(0.0, 16.0, (e2 & hb1) != 0u),
          f32(a3 >> 4u) + select(0.0, 16.0, (e3 & hb1) != 0u));
        ACC4(kb + i0 * 32u + l, vlo * d1 - vec4<f32>(m1, m1, m1, m1));
        ACC4(kb + (i0 + 1u) * 32u + l, vhi * d2 - vec4<f32>(m2, m2, m2, m2));
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

_Q3K_HELP = """
// One 16-byte group of quants against one 16-byte group of hmask, for a given 2-bit lane
// and hmask bit. Both arrive as four bytes packed in a u32, so the whole group is register
// work: `(q >> (8*i)) >> shift` folds into a single shift, and `mbit` is under 256 so the
// per-byte mask needs no truncation.
fn Q3V(q: u32, m: u32, shift: u32, mbit: u32) -> vec4<f32> {
  return vec4<f32>(
    f32((q >> shift) & 3u)         - select(4.0, 0.0, (m & mbit) != 0u),
    f32((q >> (shift + 8u)) & 3u)  - select(4.0, 0.0, ((m >> 8u) & mbit) != 0u),
    f32((q >> (shift + 16u)) & 3u) - select(4.0, 0.0, ((m >> 16u) & mbit) != 0u),
    f32((q >> (shift + 24u)) & 3u) - select(4.0, 0.0, ((m >> 24u) & mbit) != 0u));
}
"""

# Q3_K: 256 values / 110 bytes -- hmask[32] | qs[64] | scales[12] | f16 d.
# The sixteen 6-bit scales are split across three u32s and reassembled four at a time;
# hmask supplies a per-value bit that shifts the 2-bit quant from [-4,-1] to [0,3].
#
# The loops are ordered by what each field depends on, not by the output order. hmask is
# selected by `half` alone and qs by (half, blk2), while the original nesting put `j`
# outside them and so re-read hmask eight times and qs four times per block -- through `B`,
# which is a whole load per byte. That came to 512 loads to read a 28-word block. Ordering
# the loops by dependency and reading four bytes at a time brings it under 60.
#
# `is` and `k` were carried across the loops, which is what forced the original order; they
# are derived from the indices now, so the reorder is pure code motion and the value written
# to any k is unchanged.
_Q3K_DEC = """
    let o = base + b * 110u;
    let d = F16(o + 108u);
    let a0 = U32(o + 96u); let a1 = U32(o + 100u); let a2 = U32(o + 104u);
    let kb = b * 256u;
    for (var half: u32 = 0u; half < 2u; half = half + 1u) {
      let mo = o + half * 16u;
      let m0 = B4(mo); let m1 = B4(mo + 4u); let m2 = B4(mo + 8u); let m3 = B4(mo + 12u);
      for (var blk2: u32 = 0u; blk2 < 2u; blk2 = blk2 + 1u) {
        let qo = o + 32u + blk2 * 32u + half * 16u;
        let q0 = B4(qo); let q1 = B4(qo + 4u); let q2 = B4(qo + 8u); let q3 = B4(qo + 12u);
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          let shift = 2u * j;
          let mbit = 1u << (blk2 * 4u + j);
          let is = (blk2 * 4u + j) * 2u + half;
          var v: u32;
          let wsel = is >> 2u;
          if (wsel == 0u) { v = (a0 & 0x0F0F0F0Fu) | (((a2 >> 0u) & 0x03030303u) << 4u); }
          else if (wsel == 1u) { v = (a1 & 0x0F0F0F0Fu) | (((a2 >> 2u) & 0x03030303u) << 4u); }
          else if (wsel == 2u) { v = ((a0 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 4u) & 0x03030303u) << 4u); }
          else { v = ((a1 >> 4u) & 0x0F0F0F0Fu) | (((a2 >> 6u) & 0x03030303u) << 4u); }
          let sc = f32(i32(((v >> (8u * (is & 3u))) & 255u) << 24u) >> 24u) - 32.0;
          let dl = d * sc;
          let k = is * 16u;
          ACC4(kb + k +  0u, Q3V(q0, m0, shift, mbit) * dl);
          ACC4(kb + k +  4u, Q3V(q1, m1, shift, mbit) * dl);
          ACC4(kb + k +  8u, Q3V(q2, m2, shift, mbit) * dl);
          ACC4(kb + k + 12u, Q3V(q3, m3, shift, mbit) * dl);
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
          // Four at a time: the offsets are 4-aligned (84-byte block, quants start at 16),
          // so one word read replaces four byte extractions and the four results are one
          // dot product.
          for (var lw: u32 = 0u; lw < 4u; lw = lw + 1u) {
            let l = lw * 4u;
            let qw = W((qo + l) >> 2u);
            let v = vec4<f32>(f32((qw >> shift) & 3u),
                              f32((qw >> (8u + shift)) & 3u),
                              f32((qw >> (16u + shift)) & 3u),
                              f32((qw >> (24u + shift)) & 3u));
            ACC4(kb + k + l, v * dl - vec4<f32>(ml, ml, ml, ml));
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
# Workgroup memory a staged codebook may use -- 0, so none of them are staged, because it
# does not pay. The reasoning was sound and the precedent was right there: staging IQ4_XS's
# codebook took it from 63.3 to 86.0 GB/s. But that replaced about ten ALU instructions per
# value with one load, whereas this replaces a global load with a workgroup load, and the
# grids are small enough to sit in L1 already.
#
# Measured on a 27B, 64 distinct weights per format so cache cannot carry it, global read
# against staged:
#   IQ3_S    71.3 -> 71.6 GB/s        IQ2_XS   56.4 -> 58.7
#   IQ3_XXS  71.1 -> 75.6             IQ2_S    61.8 -> 63.2
#   whole captured step             150.79 -> 150.05ms   (0.5%, noise)
#
# Against that: 1 to 8 KB of workgroup memory per kernel. The narrow shape already spends
# 17408 bytes, so staging IQ2_S's 8 KB grid would put the requirement at 25.7 KB -- against
# a WebGPU guaranteed minimum of 16384. Not worth narrowing what the code runs on for half
# a percent. Raise this to 8704 to stage everything but IQ1_S, or to 2176 for the small
# grids only, if a GPU with a weaker L1 turns up.
_GRID_WG_BUDGET = 0


def _grid_u32(type_name):
    """u32 length of the codebook buffer for this format, or 0 if it has none.

    Mirrors `_ggml_grid`'s layout exactly -- ksigns[128] then the grid -- because the shader
    stages the whole buffer and indexes it with the same offsets."""
    tab = _GGML_TYPES[type_name][4]
    if tab is None:
        return 0
    from . import iqtables as T
    g = np.ascontiguousarray(getattr(T, tab)).view(np.uint8).reshape(-1)
    return (128 + g.size + (-(128 + g.size)) % 4) // 4


# Staged codebook. The i-quants read one grid entry per four values straight out of the
# storage buffer, which is a global load on top of the loads for the weight itself -- and
# IQ4_XS shows what that costs: staging ITS codebook (all sixteen entries of it) in
# workgroup memory took it from 63.3 to 86.0 GB/s. These grids are 1 KB to 8 KB rather than
# 64 bytes, but they are read just as often, and a workgroup fills one in a few instructions
# per thread before the barrier that was already there.
_GRID_STAGE = """
var<workgroup> gtab: array<u32, NGRIDu>;
fn kvfill(t: u32) {
  // Strided by 64 because a workgroup is at least that wide on every path here, and the
  // thread count is not visible from inside this function. Threads past 64 do nothing;
  // the fill is a handful of loads either way.
  if (t < 64u) {
    for (var q: u32 = t; q < NGRIDu; q = q + 64u) { gtab[q] = gr[q]; }
  }
}
"""


_GRID_FN = """
@group(0) @binding(GBIND)
var<storage,read> gr: array<u32>;
GRIDSTAGE
fn GB(o: u32) -> u32 { return (GSRC[o >> 2u] >> ((o & 3u) * 8u)) & 255u; }
fn G4(idx: u32) -> u32 { return GSRC[32u + idx]; }      // one 4-byte entry, one read
fn G4V(idx: u32) -> vec4<f32> { return unpack4x8unorm(GSRC[32u + idx]) * 255.0; }
// These two are the i-quant decode arithmetic, and they are the last 13-18% between these
// formats and their own read rate. Two attempts to cut them, both correct and both SLOWER,
// so leave them alone unless you have a measurement that says otherwise:
//
//   * signs by xor into the float's sign bit, and the 255 folded into the scale, replacing
//     a select and two multiplies per value with a vec4 shift and a vec4 xor:
//     IQ3_S 71 -> 66 GB/s, IQ2_S 62 -> 56, whole 27B step 149.9 -> 154.3ms.
//   * signs from a 16-entry workgroup table indexed by the nibble, one vec4 load instead of
//     four shift-mask-select groups -- the trick that makes IQ4_XS nearly free:
//     IQ3_XXS 86.6 -> 69, IQ2_XS 66.4 -> 53.7, whole step 134.5 -> 143.9ms.
//
// The pattern in both: `select` plus a multiply is a predicated fma here, about as cheap as
// an instruction gets, while a bitcast round-trip breaks the float pipeline and a workgroup
// lookup pays latency and bank conflicts on a data-dependent index. IQ4_XS's table wins
// because what it replaces is ten instructions of bit-fiddling, not four selects.
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
    // Whole words, not a load per byte. IQ4_XS runs at 110 GB/s on this hardware because
    // its block is a multiple of four bytes and it reads quants with a plain word load; the
    // i-quants below are 74, 82, 98 and 110 bytes, so their offsets land at any alignment and
    // they went through `B`, which is a whole global load for each byte. That is what put
    // them at a third to a half of IQ4_XS's rate. `B4` funnels the two words when it has to,
    // and anything constant for the block is hoisted out of the loop that was re-reading it.
    let sc0 = B4(o + 66u); let sc1 = B4(o + 70u);
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let sc = (select(sc0, sc1, ib >= 4u) >> (8u * (ib & 3u))) & 255u;
      let qb = o + 2u + ib * 8u;
      let qw0 = B4(qb); let qw1 = B4(qb + 4u);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let q = (select(qw0, qw1, l >= 2u) >> (16u * (l & 1u))) & 65535u;
        let nib = select(sc & 15u, sc >> 4u, l >= 2u);
        let db = d * (0.5 + f32(nib)) * 0.25;
        let gx = (q & 511u) * 2u;
        let sm = GB(q >> 9u);
        let k0 = kb + ib * 32u + l * 8u;
        ACC4(k0, G4V(gx) * SGN4(sm, 0u) * db);
        ACC4(k0 + 4u, G4V(gx + 1u) * SGN4(sm, 4u) * db);
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
    // Whole words, not a load per byte. IQ4_XS runs at 110 GB/s on this hardware because
    // its block is a multiple of four bytes and it reads quants with a plain word load; the
    // i-quants below are 74, 82, 98 and 110 bytes, so their offsets land at any alignment and
    // they went through `B`, which is a whole global load for each byte. That is what put
    // them at a third to a half of IQ4_XS's rate. `B4` funnels the two words when it has to,
    // and anything constant for the block is hoisted out of the loop that was re-reading it.
    let sc0 = B4(o + 74u); let sc1 = B4(o + 78u);
    let qh0 = B4(o + 66u); let qh1 = B4(o + 70u);
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let sc = (select(sc0, sc1, ib >= 4u) >> (8u * (ib & 3u))) & 255u;
      let qh = (select(qh0, qh1, ib >= 4u) >> (8u * (ib & 3u))) & 255u;
      let qw = B4(o + 2u + ib * 4u);
      let sw = B4(o + 34u + ib * 4u);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let nib = select(sc & 15u, sc >> 4u, l >= 2u);
        let db = d * (0.5 + f32(nib)) * 0.25;
        let gx = (((qw >> (8u * l)) & 255u) | ((qh << (8u - 2u * l)) & 768u)) * 2u;
        let sm = (sw >> (8u * l)) & 255u;
        let k0 = kb + ib * 32u + l * 8u;
        // G4V unpacks the codebook entry's four bytes in one instruction, and each half
        // lands on four consecutive activations -- so this is two dot products, not eight
        // scalar accumulates with a byte extraction and a sign select apiece.
        ACC4(k0, G4V(gx) * SGN4(sm, 0u) * db);
        ACC4(k0 + 4u, G4V(gx + 1u) * SGN4(sm, 4u) * db);
      }
    }
"""

# IQ3_XXS: 256 values / 98 bytes -- f16 d | qs[64] | u32 aux[8]. Eight grid indices per
# sub-block, four bytes each, so the 32 decoded values regroup into four sign-groups of 8.
_IQ3XXS_DEC = """
    let o = base + b * 98u;
    let d = F16(o);
    let kb = b * 256u;
    // Whole words, not a load per byte. IQ4_XS runs at 110 GB/s on this hardware because
    // its block is a multiple of four bytes and it reads quants with a plain word load; the
    // i-quants below are 74, 82, 98 and 110 bytes, so their offsets land at any alignment and
    // they went through `B`, which is a whole global load for each byte. That is what put
    // them at a third to a half of IQ4_XS's rate. `B4` funnels the two words when it has to,
    // and anything constant for the block is hoisted out of the loop that was re-reading it.
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let a1 = U32(o + 66u + ib * 4u);
      let db = d * (0.5 + f32(a1 >> 28u)) * 0.5;
      let k0 = kb + ib * 32u;
      let qb = o + 2u + ib * 8u;
      let qw0 = B4(qb); let qw1 = B4(qb + 4u);
      for (var p: u32 = 0u; p < 8u; p = p + 1u) {
        let f0 = p * 4u;
        let sm = GB((a1 >> (7u * (f0 >> 3u))) & 127u);
        let qv = (select(qw0, qw1, p >= 4u) >> (8u * (p & 3u))) & 255u;
        ACC4(k0 + f0, G4V(qv) * SGN4(sm, f0 & 7u) * db);
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
    // Whole words, not a load per byte. IQ4_XS runs at 110 GB/s on this hardware because
    // its block is a multiple of four bytes and it reads quants with a plain word load; the
    // i-quants below are 74, 82, 98 and 110 bytes, so their offsets land at any alignment and
    // they went through `B`, which is a whole global load for each byte. That is what put
    // them at a third to a half of IQ4_XS's rate. `B4` funnels the two words when it has to,
    // and anything constant for the block is hoisted out of the loop that was re-reading it.
    let scw = B4(o + 106u);
    let qh0 = B4(o + 66u); let qh1 = B4(o + 70u);
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let scb = (scw >> (8u * (ib >> 1u))) & 255u;
      let nib = select(scb & 15u, scb >> 4u, (ib & 1u) != 0u);
      let db = d * (1.0 + 2.0 * f32(nib));
      let qh = (select(qh0, qh1, ib >= 4u) >> (8u * (ib & 3u))) & 255u;
      let k0 = kb + ib * 32u;
      // Two consecutive p share one sign byte (f0 is a multiple of four, and f0 >> 3 is
      // p >> 1), so read it once for the pair instead of once per value.
      let smw = B4(o + 74u + ib * 4u);
      let qb = o + 2u + ib * 8u;
      let qw0 = B4(qb); let qw1 = B4(qb + 4u);
      for (var pg: u32 = 0u; pg < 4u; pg = pg + 1u) {
        let sm = (smw >> (8u * pg)) & 255u;
        let p0 = pg * 2u;
        let p1 = p0 + 1u;
        let qa = (select(qw0, qw1, p0 >= 4u) >> (8u * (p0 & 3u))) & 255u;
        let qc = (select(qw0, qw1, p1 >= 4u) >> (8u * (p1 & 3u))) & 255u;
        ACC4(k0 + p0 * 4u, G4V(qa | ((qh << (8u - p0)) & 256u)) * SGN4(sm, 0u) * db);
        ACC4(k0 + p1 * 4u, G4V(qc | ((qh << (8u - p1)) & 256u)) * SGN4(sm, 4u) * db);
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
    "Q3_K":    (_Q3K_DEC,     _Q3K_HELP,          256, 110, None),
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
    return (bool(_NATIVE_GGUF) and type_name in _GGML_TYPES
            and (_adam_backend_ready() or _webgl_ready()))


# Below this many output rows, the decode kernel cannot fill the GPU: the dispatch is
# N / WGX workgroups, so a 48-row projection gets exactly ONE, and the machine idles through
# it. Such projections are small (a gate or a decay term is a few hundred KB), yet a 27B's
# 48 beta and alpha projections took 14.8ms a step -- longer than the 1.28GB of fused QKV
# beside them, at 1.4 GB/s. Narrow shapes get the parallelism moved off the row axis and onto
# K instead: same threads, more workgroups, each doing less of the reduction.
_SMALL_N = 512


def _small_cfg(vals):
    """(WGX, KS) for a narrow output. `xs` holds KS * vals floats of workgroup memory, so the
    split is bounded by the block size of the quantization -- 32 values per block allows a
    32-way split, 256 values allows 16."""
    ks = max(_GGML_KS, min(32, 4096 // max(1, vals)))
    return max(1, 256 // ks), ks


# Blocks per row at or below which the reduction would not be split at all -- OFF, at 0,
# because it measures slower. A short row is every routed expert (K is the hidden size,
# 2048, eight blocks, against a dense layer's 5120 or 17408), and dropping the split there
# looks like it must pay: the split exists to give a lane something to do, and at eight
# blocks each lane has almost nothing while still paying the workgroup staging, the barrier
# and the psum reduction.
#
# It does not, and worse: ANY WGX ABOVE 64 IS WRONG. 128x2, 128x4 and 256x1 all fail the
# self-check at N=576 K=768 with a relative error around 0.92 -- not 1.0, so they are not
# failing to compile, they are computing the wrong answer. Something in the psum layout or
# the row indexing assumes the 64-wide workgroup; nobody has needed a wider one, so it has
# never been fixed. Do not set _SHORT_K_CFG's first element above 64 without fixing that
# first, and do not trust a timing from a shape until the self-check has passed it.
#
# That is also the real story behind a 31% "win" this constant once appeared to produce:
# 256x1 was not fast, it was wrong. An earlier note here blamed the measurement. The
# measurement was fine; the kernel was not, and the self-check could not say so because it
# only ever built the narrow shape.
#
# Of the shapes that ARE correct, none beats the default. Screened on the routed experts of
# a 30B, all 48 layers in one capture: default 10.97ms, 64x4 10.92, 64x2 10.88, 64x8 10.88,
# 32x4 11.01, 32x8 11.06. That spread is noise, and 10.9ms for 0.768 GB is 70.5 GB/s --
# exactly what the same Q3_K kernel reaches on the dense 27B, so there is nothing shape-
# dependent left to find here.
#
# Set to 8 to try again on other hardware; 8 clears a routed expert and leaves a 5120-wide
# layer (20 blocks) alone.
_SHORT_K_BLOCKS = 0


def _cfg_for(kind, vals):
    """The (WGX, KS) a kernel variant was built with. `kind` is what `_shape_kind` returned."""
    if kind == "narrow":
        return _small_cfg(vals)
    if kind == "shortk":
        return _SHORT_K_CFG
    return None


_AUTO = object()      # "work it out"; None means the default shape, explicitly


def _shape_kind(N, K, vals):
    """Which thread shape this matmul wants: 'narrow', 'shortk', or None for the default.

    The two are opposites and the order matters. A narrow output cannot fill the machine
    along the row axis at all, so parallelism has to move onto K. A short reduction has rows
    to spare and nothing to gain from splitting K, so the split comes off entirely. A matmul
    that is both narrow and short is narrow first: no rows is the harder problem."""
    if int(N) <= _SMALL_N:
        return "narrow"
    if (int(K) // max(1, int(vals))) <= _SHORT_K_BLOCKS:
        return "shortk"
    return None


# The shape `_SHORT_K_BLOCKS` would select if it were on: every one of the workgroup's 256
# threads takes its own output row and walks the whole of K, so there is no split to reduce
# afterwards. It is the opposite of the narrow-output shape, which splits as far as the
# workgroup memory allows (16 ways at a 256-value block) because there the row axis is what
# cannot fill the machine. It is also WRONG -- see _SHORT_K_BLOCKS: every WGX above 64 fails
# the self-check. Left as the documented shape only because that is where the investigation
# ended; a correct wide-workgroup variant would have to fix the psum layout first.
_SHORT_K_CFG = (256, 1)


def _orw_for(mode=1):
    """Output rows per lane for this path. Only the single-token decode kernel uses more
    than one; the two-row variant addresses psum with its own fixed layout."""
    return _GGML_ORW if (mode == 1 and _GGML_ORW > 1) else 1


def _gemv_groups(N, mode=1, vals=None, kind=None):
    """Workgroups needed to cover N output rows on the decode path. `kind` is the thread
    shape the kernel was built with, from `_shape_kind`."""
    cfg = _cfg_for(kind, vals) if (kind and vals is not None and mode == 1) else None
    wgx = cfg[0] if cfg else _GGML_WGX
    per = wgx * _orw_for(mode)
    return (int(N) + per - 1) // per


def _ggml_src(type_name, mode, cfg=None, moe=False):
    """`mode`: 1 or 2 for the decode kernel with that many rows, 0 for the batched one.
    `cfg` overrides (WGX, KS) for a narrow output. `moe` selects the variant that reads its
    expert from an index buffer instead of being bound to one expert's weights."""
    dec, helpers, vals, _, _ = _GGML_TYPES[type_name]
    # Whether this format's codebook gets staged in workgroup memory. Decided once, because
    # three substitutions below have to agree about it -- and the one that nearly got away is
    # the fill call: it is emitted on a test for `fn kvfill`, which lives in the text this
    # flag SUBSTITUTES IN, so testing the raw helpers string leaves the table unfilled and
    # every value zero. The self-check caught exactly that, at N=576 on IQ2_XS.
    _ng = _grid_u32(type_name)
    stage_grid = 0 < _ng * 4 <= _GRID_WG_BUDGET
    _GGML_WGX_L, _GGML_KS_L = cfg if cfg else (_GGML_WGX, _GGML_KS)
    # Four-at-a-time formats get the vec4 activation window; a fragment that accumulates a
    # value at a time is faster with the float one, and a single-value "block" (F16 and
    # friends) cannot use vec4 at all.
    vec_win = (vals % 4 == 0 and "ACC4(" in dec
               and "ACC(" not in dec.replace("ACC4(", ""))
    pre, main, tail = (((_GGML_GEMV_PRE_V4 if vec_win else _GGML_GEMV_PRE_F32),
                        _GGML_GEMV_MAIN, _GGML_GEMV_TAIL) if mode
                       else (_GGML_GEMM_PRE, _GGML_GEMM_MAIN, _GGML_GEMM_TAIL))
    # helpers are functions, so they go BEFORE main -- WGSL has no nested functions.
    src = _GGML_BIND + (_MOE_BIND if moe else "") + helpers + pre + main + dec + tail
    rows = max(1, mode)
    two = rows == 2
    # Multi-row accumulation is for the single-token decode path, which is the hot one. The
    # two-token variant addresses psum with its own fixed layout, so it stays at one row.
    orw = _orw_for(mode)
    xrow = _GGML_KS_L * vals                      # where the second row's activations start
    # These two go FIRST: what they expand to contains other placeholders -- XSFILL brings
    # in XLOAD1 and XBAS, XSCOMMON brings in ACCDECL1 -- and a placeholder that arrives after
    # its own substitution has run is left in the source, which compiles to nothing and reads
    # as a numerically broken kernel.
    subs = [("XSFILL", _XS_FILL_V4 if vec_win else _XS_FILL_F32),
            ("XSCOMMON", _XS_COMMON),
            ("ACCDECL1", "var<private> acc1: f32;" if two else ""),
            ("ACCBODY1", (("  let j1 = i + %du;\n"
                           "  acc1 = acc1 + xs[j1 >> 2u][j1 & 3u] * v;" % xrow) if vec_win
                          else "  acc1 = acc1 + xs[i + %du] * v;" % xrow)
             if two else ""),
            ("ACC4BODY1", ("  acc1 = acc1 + dot(xs[(i + %du) >> 2u], v);" % xrow if vec_win
                           else ("  acc1 = acc1 + dot(vec4<f32>(xs[i + %du], xs[i + %du], "
                                 "xs[i + %du], xs[i + %du]), v);"
                                 % (xrow, xrow + 1, xrow + 2, xrow + 3)))
             if two else ""),
            ("ACCINIT1", "  acc1 = 0.0;" if two else ""),
            ("XLOAD1", (("      var xv1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);\n"
                         "      if (b < nb) { let s1 = gm.K + b * BLKVALS + t;\n"
                         "        xv1 = vec4<f32>(x[s1], x[s1 + 1u], x[s1 + 2u], "
                         "x[s1 + 3u]); }\n"
                         "      xs[(%du + xoff + t) >> 2u] = xv1;" % xrow) if vec_win
                        else ("      var xv1: f32 = 0.0;\n"
                              "      if (b < nb) { xv1 = x[gm.K + b * BLKVALS + t]; }\n"
                              "      xs[%du + xoff + t] = xv1;" % xrow)) if two else ""),
            # Its own key rather than a expression on KSxBLKxR, so neither is a prefix of
            # the other and the substitution order cannot matter.
            ("XSVEC4N", str(_GGML_KS_L * vals * rows // 4)),
            ("PSUM1", "  psum[%du + ly * WGXu + lx] = acc1;" % (_GGML_KS_L * _GGML_WGX_L)
             if two else ""),
            ("OUT1", ("    var t1: f32 = 0.0;\n"
                      "    for (var i: u32 = 0u; i < KSu; i = i + 1u) "
                      "{ t1 = t1 + psum[%du + i * WGXu + lx]; }\n"
                      "    outp[gm.N + n] = t1;" % (_GGML_KS_L * _GGML_WGX_L)) if two else ""),
            ("KSxBLKxR", str(_GGML_KS_L * vals * rows)),
            ("PSUMSZ", str(_GGML_KS_L * _GGML_WGX_L * rows * orw)),
            ("KSxBLK", str(_GGML_KS_L * vals)), ("KSxWGX", str(_GGML_KS_L * _GGML_WGX_L)),
            ("KSGx256", str(_GGML_KSG * 256)), ("KSGu", "%uu" % _GGML_KSG),
            # Four rows of one block, for each block in flight. 8KB at the widest format,
            # which every device allows; a workgroup allocation that does not fit fails to
            # COMPILE, and a failed compile here is silent -- the kernel just writes zeros.
            ("XS4SZ", str(_GGML_KSG * 4 * vals)),
            # Literally `0u`, not `ly`: a workgroup of height one makes them equal, but the
            # uniformity analysis is syntactic and `lid.y` is non-uniform whatever the shape.
            ("BSTART", "0u" if _GGML_KSG == 1 else "ly"),
            ("KSu", "%uu" % _GGML_KS_L), ("MASKBLK", "%uu" % (vals - 1)),
            ("BLKVALS", "%uu" % vals),
            ("WGXu", "%uu" % _GGML_WGX_L), ("WGX", str(_GGML_WGX_L)),
            ("ORWu", "%uu" % orw), ("ORW", str(orw)),
            # Helpers that stage a table in workgroup memory fill it here, before the first
            # barrier of the block loop, so every lane sees it.
            # Helpers that stage a table in workgroup memory fill it here, before the first
            # barrier of the block loop, so every lane sees it. BOTH paths need it -- the
            # batched kernel has its own entry point and its own workgroup shape, and leaving
            # it out left the table zeroed there (the self-check caught exactly that).
            # Routing offsets, and ONLY in the routed kernel: a dense weight gets no `woff`
            # at all rather than `woff = 0`. This is for clarity, not speed -- `W` is the
            # innermost function in the kernel, called once per four bytes of every weight,
            # so carrying a dead add through it looked like it had to cost something, and it
            # does not: the dense 27B measured 162.6ms with the term and 162.1ms without,
            # i.e. nothing. The compiler folds it after all. Kept because a kernel that
            # cannot express routing is easier to reason about than one where routing is
            # always present and always disabled.
            ("MOEVARS", _MOE_VARS if moe else ""),
            ("WOFS", "woff + " if moe else ""),
            ("XBAS", "xbase + " if moe else ""),
            ("OSLT", "oslot * gm.N + " if moe else ""),
            # One dispatch covers every routed slot: z indexes the slot, so a MoE
            # projection is a single command with k times the work rather than k commands
            # each too small to fill the machine.
            ("WOFFINIT", ("  oslot = gid.z;\n  woff = eidx[oslot] * gm.estride;\n"
                          "  xbase = select(0u, oslot * gm.K, gm.xper == 1u);"
                          if (moe and mode == 1) else
                          ("  oslot = 0u;\n  woff = eidx[gm.eslot] * gm.estride;\n"
                           "  xbase = 0u;" if moe else ""))),
            # The codebook sits after the expert index when there is one.
            ("GBIND", "5" if moe else "4"),
            # Stage the codebook in workgroup memory when it fits, and read it from there.
            # A grid too large for the budget keeps the global read -- correct either way,
            # and the substitution is what makes that a one-line difference.
            ("GRIDSTAGE", _GRID_STAGE.replace("NGRIDu", "%du" % _ng) if stage_grid else ""),
            ("GSRC", "gtab" if stage_grid else "gr"),
            ("HELPERINIT", ("  kvfill(lx + ly * %du);\n  workgroupBarrier();" % _GGML_WGX_L)
             if ("fn kvfill" in helpers or stage_grid) else ""),
            ("GEMMINIT", "  kvfill(lx + ly * 64u);\n  workgroupBarrier();"
             if ("fn kvfill" in helpers or stage_grid) else "")]
    for k, v in subs:
        src = src.replace(k, v)
    return src


def _selfcheck_shape(kind, vals):
    """An (N, blocks-per-row) that provably lands on `kind`, or None if it cannot.

    The self-check used a single shape -- two output rows, three blocks -- for every variant
    it tested. Two rows is below `_SMALL_N`, so `_shape_kind` called every one of them
    'narrow' and the other thread shapes were never run by it at all. A shape that is never
    checked is a shape that can be silently broken, and one was: a short-K variant produced
    nothing but zeros and read as a 31% speedup, because a kernel that does nothing is fast
    and a self-check that never builds it has nothing to say.

    N is deliberately not a multiple of the group width, so the last group is partial and the
    `n < gm.N` guard is exercised rather than assumed."""
    n_wide = _SMALL_N + 64
    if kind == "narrow":
        n, nb = 2, 3
    elif kind == "shortk":
        n, nb = n_wide, 3
    else:
        n, nb = n_wide, _SHORT_K_BLOCKS + 3
    # A kind can be unreachable -- 'shortk' is, whenever its threshold is off -- and there is
    # nothing to check in that case.
    return (n, nb) if _shape_kind(n, nb * vals, vals) == kind else None


def _ggml_selfcheck(type_name, mode, small=_AUTO, moe=False):
    """Multiply a few random blocks and compare against the reference dequantizer.

    A WGSL compile error surfaces as a console warning and a buffer full of zeros, not as an
    exception -- which reads exactly like a working kernel on an all-zero weight, and has
    twice sent me chasing a numerical bug that was a syntax error. A couple of blocks per
    type, once per session, turns both failure modes into something that raises.

    With `small` left at `_AUTO` this checks EVERY thread shape the decode path can pick,
    not just the one some particular matmul happens to want."""
    # WebGL has one kernel per format, not one per thread shape: without workgroup memory
    # there is nothing for a shape to trade off, so the sweep below has nothing to sweep.
    if _webgl_ready() and not _adam_backend_ready():
        vals = _GGML_TYPES[type_name][2]
        _selfcheck_one(type_name, mode, None, moe, *(_selfcheck_shape(None, vals)
                                                     or (_SMALL_N + 64, 3)))
        return
    if small is _AUTO and mode == 1:
        vals = _GGML_TYPES[type_name][2]
        for kind in ("narrow", "shortk", None):
            shape = _selfcheck_shape(kind, vals)
            if shape is not None:
                _selfcheck_one(type_name, mode, kind, moe, *shape)
        return
    if small is _AUTO:
        small = None
    vals = _GGML_TYPES[type_name][2]
    shape = _selfcheck_shape(small, vals) or (_SMALL_N + 64, 3)
    _selfcheck_one(type_name, mode, small, moe, *shape)


def _selfcheck_one(type_name, mode, small, moe, N, NB):
    """One (thread shape, N, blocks) against the reference. Raises on a mismatch."""
    from . import ggufload as G
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    # THREE blocks per row at least, not one. A block is not always a whole number of words
    # -- Q3_K is 110 bytes -- so a decode fragment that reads a word directly is only correct
    # on blocks whose byte offset happens to land on a word boundary. With one block per row
    # the offset is always zero and such a bug is invisible; it cost half the columns of
    # every Q3_K tensor, which on a model whose experts are all Q3_K is the whole model.
    K = NB * vals
    # Random bytes are a legal block for every type and exercise every scale and codebook
    # index -- but an f16 field is Inf or NaN whenever its exponent is all ones, and over
    # hundreds of rows that is a certainty rather than a risk. The exponent's top bit lives
    # in bit 6 of a byte whatever the field's offset, so clearing it rules the f16 case out
    # without needing to know where each format keeps its scales.
    #
    # That is not enough for every format: MXFP4's scale is a bare e8m0 exponent, so bit 6
    # clear still allows 2^64, and at 576 rows one such row always turns up. Clearing more
    # bits fixes it but costs coverage of the quantized fields, so the masks are tried in
    # order and the loosest one that yields a finite block wins -- full coverage for the
    # formats that can take it, and a checkable block for the ones that cannot.
    for mask in (0xBF, 0x3F, 0x0F):
        for seed in range(32):
            rng = np.random.default_rng(seed)
            raw = (rng.integers(0, 256, (N, NB * blk), dtype=np.uint8) & mask).tobytes()
            ref = np.asarray(G.dequant(G.GGML_IDS[type_name], raw, N * K),
                             np.float32).reshape(N, K)
            if np.all(np.isfinite(ref)) and float(np.abs(ref).max()) < 1e4:
                break
        else:
            continue
        break
    else:
        raise RuntimeError("could not draw a finite %s block to self-check against" % type_name)
    M = mode if mode else 3
    x = rng.standard_normal((M, K)).astype(np.float32)
    raw = raw + b"\x00" * ((-len(raw)) % 4)     # a block is not always a whole number of u32
    pk = ggml_transpose(xp.asarray(np.frombuffer(raw, np.int32)), N, NB * blk)
    eidx = eslot = None
    estride = 0
    if moe:
        # Stack the same weight twice and ask for the SECOND copy through a non-zero slot, so
        # a wrong stride or a slot that is ignored both show up as a mismatch rather than
        # accidentally reading the right bytes.
        estride = int(pk.size)                   # one expert's words, BEFORE stacking
        pk = xp.asarray(np.concatenate([cp.asnumpy(pk), cp.asnumpy(pk)]))
        eidx = xp.asarray(np.array([0, 1], np.int32))
        eslot = 1
    # Build the variant if nobody has. `ggml_matmul` calls this right after adding one, so
    # this only fires for a standalone call -- but a self-check you cannot run on its own is
    # not much of a self-check, and without it a sweep of every format reports every one of
    # them broken (the kernel is missing, so nothing runs and the output stays zero).
    if _webgl_ready() and not _adam_backend_ready():
        moedec = moe and M <= 2
        if (type_name, moe, moedec) not in _ggml_gl["added"]:
            _ggml_add_gl(type_name, moe, moedec)
        out_t = _ggml_run_gl(xp.asarray(x), pk, type_name, K, N, eidx=eidx,
                             eslot=(eslot or 0), estride=estride)
    else:
        key = (type_name, mode, small, moe, _GGML_KSG if mode == 0 else 0)
        if key not in _ggml_k["added"]:
            _ggml_add(type_name, mode, small, moe)
            _ggml_k["added"].add(key)
        out_t = _ggml_run(xf=xp.asarray(x), packed=pk, type_name=type_name,
                          K=K, N=N, small=small, eidx=eidx,
                          eslot=(eslot or 0), estride=estride)
    raw_out = np.asarray(cp.asnumpy(out_t))
    if moe and mode == 1:
        # One row per routed slot. Take the SECOND -- it must have read the second copy of
        # the weight, so a stride that is ignored or wrong shows up here.
        got = raw_out.reshape(-1, M, N)[1]
    else:
        got = raw_out.reshape(M, N)
    want = x @ ref.T
    err = float(np.abs(got - want).max() / (np.abs(want).max() + 1e-30))
    if not (err < 1e-4):
        raise RuntimeError("native ggml %s kernel for %s at N=%d K=%d is wrong (rel err %.3g)"
                           " -- check the console for a shader compile error"
                           % ({0: "gemm", 1: "gemv", 2: "gemv2"}.get(mode, "mode%d" % mode)
                              + ("(%s)" % small if small else "") + ("(moe)" if moe else ""),
                              type_name, N, K, err))


def ggml_matmul(xf, packed, type_name, K, N, eidx=None, eslot=0, estride=0,
                xper=False, bias=None):
    """xf(M,K) @ packed(N,K).T -> (M,N), decoding ggml blocks in the shader.

    `packed` must be in the transposed (word, row) layout that `ggml_transpose` produces --
    GGMLLinear does that once at upload.

    With `eidx`, `packed` holds SEVERAL weights of that shape end to end and the shader picks
    one at run time: `eidx[eslot]` is its index and `estride` its size in words. That is how a
    sparse-MoE projection runs without the choice of expert being baked into the command."""
    # A dedicated two-row kernel, not the batched one: verifying a speculative draft is a
    # batch of two, and it only pays if the second row rides along with the first.
    _gpu_stat_push()
    m = 1 if (eidx is not None and xper) else int(xf.shape[0])
    # The batched kernel is where prefill goes, and it is BAD: it reads the weight once and
    # still costs what reading it once per row costs, because it accumulates through memory
    # rather than registers. On a 2048x11008 Q4_K weight at M=256, one batched call is
    # 42.8ms; the same work as 128 calls to the two-row decode kernel -- 128 weight reads
    # instead of one -- is 21.4ms, and as 256 single-row calls it is 42.4ms.
    #
    # Routing batches through pairs was tried on the strength of that and is NOT here,
    # because it made real prefill SLOWER: 12755ms against 11121ms at T=512 on a 3B. The
    # microbenchmark left out what the routing actually costs -- a slice and a copy per
    # pair, then a 256-way concatenate per matmul -- and that is more than the kernel saves.
    # (Output was exact, max abs difference 0.0 at M=3, 7 and 16, so this is purely about
    # speed.)
    #
    # Prefill is 61% MLP and runs at about 0.6 GB/s against the decode path's 100+. The fix
    # is a GEMM that tiles K into workgroup memory and keeps its accumulators in registers,
    # not a different way to call the one that exists.
    if _webgl_ready() and not _adam_backend_ready():
        moe = eidx is not None
        moedec = moe and m <= 2
        hb = bias is not None
        if (type_name, moe, moedec, hb) not in _ggml_gl["added"]:
            _ggml_add_gl(type_name, moe, moedec, hb)
            if not hb:                      # the bias variant differs only in a final fetch
                _ggml_selfcheck(type_name, m if m <= 2 else 0, None, moe)
        return _ggml_run_gl(xf, packed, type_name, K, N, eidx=eidx, eslot=eslot,
                            estride=estride, xper=xper, bias=bias)
    mode = m if m <= 2 else 0
    small = _ggml_shape_for(type_name, N, K, packed) if mode == 1 else None
    moe = eidx is not None
    key = (type_name, mode, small, moe, _GGML_KSG if mode == 0 else 0)
    if key not in _ggml_k["added"]:
        _ggml_add(type_name, mode, small, moe)
        _ggml_k["added"].add(key)           # set before the check: it calls back in here
        _ggml_selfcheck(type_name, mode, small, moe)
    of = _ggml_run(xf, packed, type_name, K, N, small=small,
                   eidx=eidx, eslot=eslot, estride=estride, xper=xper)
    return of if bias is None else of + bias


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


def _ggml_add(type_name, mode, small=None, moe=False):
    plat = _adam_kernel["platform"]
    binds = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
    if moe:
        binds.append("read-only-storage")               # the expert index, before the grid
    if _GGML_TYPES[type_name][4] is not None:
        binds.append("read-only-storage")
    cfg = _cfg_for(small, _GGML_TYPES[type_name][2])
    plat.addKernel(_ggml_name(type_name, mode, small=small, moe=moe),
                   {"source": _ggml_src(type_name, mode, cfg, moe=moe), "bindingTypes": binds})


def _ggml_name(type_name, mode, orw=None, small=None, moe=False):
    # ORW and the thread shape are compile-time constants in the shader, so a kernel is
    # identified by them too -- otherwise a second variant would silently reuse the first
    # one's pipeline.
    o = _orw_for(mode) if orw is None else orw
    # KSG is a compile-time constant of the BATCHED kernel -- it sets the workgroup's second
    # dimension and the size of the staged activations -- so it belongs in the name for the
    # same reason ORW does. Without it a second value silently reuses the first's pipeline,
    # and every measurement of the second is a measurement of the first.
    return "ggml%s_%s%s%s%s%s" % (("v", "v2", "")[mode], type_name.lower(),
                                  "" if o <= 1 else "_r%d" % o,
                                  "_%s" % small if small else "",
                                  "_e" if moe else "",
                                  "" if (mode != 0 or _GGML_KSG == 1)
                                  else "_g%d" % _GGML_KSG)


# ==== the small fused decode kernels, on WebGL ==========================================
#
# These are what the quantized matmul is NOT: pure elementwise or one-row-reduce work, a few
# hundred KB a piece. They matter anyway, and on this backend they matter more than on the
# other one. A decode step through the unfused expressions costs about 2700 GPU dispatches
# on a 3B, and a dispatch here is worth roughly 50us of host time even when everything it
# touches is already resident -- so the step is bounded by how many there are, not by how
# much memory they move. Each fusion below removes five to eight of them per layer.

_gl_kernels = set()

# One float per texel, addressed linearly; `textureSize` recovers the row width, which the
# platform picks rather than the caller.
_GL_FETCH2 = """float %(fn)s(int i) { ivec2 s = textureSize(%(tex)s, 0);
  if (s.y == 1) { return texelFetch(%(tex)s, ivec2(i, 0), 0).r; }
  int y = i / s.x;
  return texelFetch(%(tex)s, ivec2(i - y * s.x, y), 0).r; }"""


def _gl_head(samplers, ints=(), floats=()):
    return ("#version 300 es\nprecision highp float; precision highp int;\n"
            "precision highp sampler2D;\nuniform int _ka_tex_output_texture_w;\n"
            + "".join("uniform sampler2D %s;\n" % t for t, _ in samplers)
            + "".join("uniform int %s;\n" % k for k in ints)
            + "".join("uniform float %s;\n" % k for k in floats)
            + "out float fragColor;\n"
            + "\n".join(_GL_FETCH2 % {"fn": f, "tex": t} for t, f in samplers)
            + "\nint _idx() { return int(gl_FragCoord.x) + int(gl_FragCoord.y)"
              " * _ka_tex_output_texture_w; }\n")


def _gl_run(name, source, inputs, out, uniforms):
    """Add (once) and dispatch one GLSL kernel. `out` is the destination buffer."""
    plat = _copy_kernel["plat"]
    if name not in _gl_kernels:
        plat.addKernel(name, {"source": source})
        _gl_kernels.add(name)
    u = [{"name": "_ka_tex_output_texture_w",
          "value": int(out.buffer.texture_shape.width), "type": "int"}]
    for k, v in uniforms:
        u.append({"name": k, "value": v,
                  "type": "float" if isinstance(v, float) else "int"})
    plat.runKernel({"name": name,
                    "inputs": [{"name": n, "id": b.buffer.buffer_id} for n, b in inputs],
                    "output": out.buffer.buffer_id, "uniforms": u})
    return out


_SWIGLU_GLSL = _gl_head([("tex_g", "Gf"), ("tex_u", "Uf")],
                        ("u_half", "u_gstride", "u_ustride", "u_uoff", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int r = i / u_half; int c = i - r * u_half;
  float x = Gf(r * u_gstride + c);
  // silu(x) * y as the division rather than x * sigmoid(x), for the same reason as the
  // WGSL: one reciprocal against neg, exp, add, div and mul as five separate passes.
  fragColor = (x / (1.0 + exp(-x))) * Uf(r * u_ustride + u_uoff + c);
}
"""

# Two passes, not one. A fragment owns one output element, so a single-pass form would make
# every element of a row re-sum the whole row -- fine at T=1 and quadratic at prefill.
_RMS_SUM_GLSL = _gl_head([("tex_x", "Xf")], ("u_T", "u_H")) + """
void main() {
  int r = _idx();
  if (r >= u_T) { fragColor = 0.0; return; }
  int base = r * u_H;
  float s = 0.0;
  for (int i = 0; i < u_H; i = i + 1) { float v = Xf(base + i); s += v * v; }
  fragColor = s;
}
"""
_RMS_APPLY_GLSL = _gl_head([("tex_x", "Xf"), ("tex_w", "Wf"), ("tex_s", "Sf")],
                           ("u_H", "u_n"), ("u_eps",)) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int r = i / u_H; int c = i - r * u_H;
  fragColor = Xf(i) * inversesqrt(Sf(r) / float(u_H) + u_eps) * Wf(c);
}
"""

_ROPE_GLSL = _gl_head([("tex_x", "Xf"), ("tex_c", "Cf"), ("tex_s", "Sf")],
                      ("u_n", "u_HD", "u_rd", "u_T")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int d = i % u_HD;
  int ci = ((i / u_HD) % u_T) * u_HD + d;      // cos/sin are (T, HD); x is (heads, T, HD)
  int h = u_rd / 2;
  float rot;
  if (d < h) { rot = -Xf(i + h); }
  else if (d < u_rd) { rot = Xf(i - h); }
  else { rot = Xf(i); }              // pass-through tail: sin is 0 here, the value is inert
  fragColor = Xf(i) * Cf(ci) + rot * Sf(ci);
}
"""


# Concatenate, as a gather. The WGSL form runs one dispatch per input, each writing its own
# slice of a shared output -- which is exactly what a fragment shader cannot do: an
# invocation writes its own fragment, and the platform renders the whole texture, so the
# second input's pass would clear the first one's rows. Written the other way round, with
# every input bound at once and each fragment choosing where its value comes from, it is one
# pass and no sub-rectangle write.
#
# This is not a nicety. The growing KV cache appends through `cat` twice a layer, and without
# a device kernel `xp.concatenate` takes the round trip: read the cache back to the host,
# concatenate there, upload it again. That was 144 read-backs per token on a 36-layer model
# -- and a read-back drains the whole command queue -- which came to 87% of a decode step.
_CAT2_GLSL = _gl_head([("tex_a", "Af"), ("tex_b", "Bf")],
                      ("u_pre", "u_n1", "u_n2", "u_post", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int W = u_n1 + u_n2;
  int q = i / u_post;
  int t = i - q * u_post;              // index within the trailing block
  int w = q % W;                       // position along the concatenated axis
  int p = q / W;                       // index over the leading axes
  fragColor = (w < u_n1) ? Af((p * u_n1 + w) * u_post + t)
                         : Bf((p * u_n2 + (w - u_n1)) * u_post + t);
}
"""


def _webgl_cat2(a, b, axis):
    """`a` and `b` concatenated along `axis`, in one pass."""
    sh = list(a.shape)
    n1, n2 = int(a.shape[axis]), int(b.shape[axis])
    sh[axis] = n1 + n2
    pre = 1
    for d in sh[:axis]:
        pre *= int(d)
    post = 1
    for d in sh[axis + 1:]:
        post *= int(d)
    of = _empty(tuple(sh))
    _gl_run("cat2_gl", _CAT2_GLSL, [("tex_a", _contig(a)), ("tex_b", _contig(b))], of,
            [("u_pre", pre), ("u_n1", n1), ("u_n2", n2), ("u_post", post),
             ("u_n", pre * (n1 + n2) * post)])
    return of


def _webgl_cat(datas, axis):
    """Left-fold of the two-input pass. The decode path always passes two; the folded form
    keeps the general case working rather than falling back to the host round trip."""
    out = datas[0]
    for d in datas[1:]:
        out = _webgl_cat2(out, d, axis)
    return out


# Gated DeltaNet, on WebGL.
#
# The hybrid models put a recurrent layer between the attention ones, and without these the
# layer runs on the HOST: the activations come off the device and the whole block is done in
# numpy. On a 64-layer hybrid that is six read-backs per layer per token -- 347 a token on a
# 27B -- and a read-back drains the command queue, so it costs far more than the arithmetic
# it avoids.
#
# Two things differ from the WGSL. There is no workgroup reduction, so a fragment that needs
# a head's L2 norm recomputes the head rather than sharing one; the head is `dk` wide and the
# fragments run at once, which is the same trade the one-pass rmsnorm makes. And the state
# and the conv ring buffer are updated OUT of place and copied back: a fragment shader cannot
# read the texture it is writing, so `S[i] = S[i] * decay + ...` has to become a new buffer
# and a copy. That copy is one pass over a few megabytes, against the read-back it replaces.

_GDN_PRE_GL = _gl_head([("tex_qkv", "Qf"), ("tex_b", "Bf"), ("tex_a", "Af"),
                        ("tex_cst", "Cf"), ("tex_k", "Kf")],
                       ("u_hk", "u_hv", "u_dk", "u_dv", "u_W", "u_flags", "u_n")) + """
int C_;
float convval(int c) {
  float raw = Qf(c);
  if ((u_flags & 1) == 0) { return raw; }
  float acc = 0.0;
  for (int j = 0; j + 1 < u_W; j = j + 1) { acc += Cf(j * C_ + c) * Kf(j * C_ + c); }
  acc += raw * Kf((u_W - 1) * C_ + c);
  if ((u_flags & 16) != 0) { acc += Kf(u_W * C_ + c); }
  return acc / (1.0 + exp(-acc));            // SiLU
}
void main() {
  int i = _idx();
  int nq = u_hk * u_dk; int nv = u_hv * u_dv;
  C_ = 2 * nq + nv;
  if (i >= u_n) { fragColor = 0.0; return; }
  if (i >= C_) {                              // the decay and beta gates
    int t = i - C_;
    int ko = u_W * C_ + C_;                   // conv_w, conv_b, then A, dt_bias
    if (t < u_hv) {
      float a = Af(t);
      if ((u_flags & 2) != 0) { a += Kf(ko + u_hv + t); }
      float sp = max(a, 0.0) + log(1.0 + exp(-abs(a)));
      float d = ((u_flags & 4) != 0) ? sp * Kf(ko + t) : sp;
      fragColor = exp(min(d, 0.0));
    } else {
      int h = t - u_hv;
      fragColor = ((u_flags & 8) != 0) ? 1.0 / (1.0 + exp(-Bf(h))) : 1.0;
    }
    return;
  }
  float val = convval(i);
  if (i < 2 * nq) {                           // a q or k channel: L2-normalised per head
    int g = i / u_dk;
    int c0 = g * u_dk;
    float s = 0.0;
    for (int t = 0; t < u_dk; t = t + 1) { float v = convval(c0 + t); s += v * v; }
    float v = val * inversesqrt(s + 1e-6);
    if (g < u_hk) { v *= inversesqrt(float(u_dk)); }   // q also carries 1/sqrt(dk)
    fragColor = v;
  } else {
    fragColor = val;
  }
}
"""

# The ring buffer holds INPUTS, so it shifts in the raw projection rather than the conv
# output. Out of place, then copied back.
_GDN_CST_GL = _gl_head([("tex_qkv", "Qf"), ("tex_cst", "Cf")], ("u_C", "u_W", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int j = i / u_C; int c = i - j * u_C;
  fragColor = (j + 2 < u_W) ? Cf((j + 1) * u_C + c) : Qf(c);
}
"""

_GDN_STEP_GL = _gl_head([("tex_S", "Sf"), ("tex_qkv", "Qf")],
                        ("u_hv", "u_dk", "u_dv", "u_rep", "u_n")) + """
void main() {
  int i = _idx();
  int n = u_hv * u_dv;
  if (i >= 2 * n) { fragColor = 0.0; return; }
  int e = (i < n) ? i : (i - n);
  int h = e / u_dv; int vi = e - h * u_dv;
  // q and k are stored per KEY head and the key heads CYCLE across the value heads
  // (ggml: iq1 = iv1 % n_q_heads), so this is a modulo rather than a divide.
  int hk = u_hv / u_rep;
  int nq = hk * u_dk;
  int qo = (h % hk) * u_dk;
  int ko = nq + qo;
  int sbase = h * u_dk * u_dv + vi;
  float pred = 0.0; float qs = 0.0; float qk = 0.0;
  for (int d = 0; d < u_dk; d = d + 1) {
    float sv = Sf(sbase + d * u_dv);
    float kd = Qf(ko + d);
    float qd = Qf(qo + d);
    pred += kd * sv; qs += qd * sv; qk += qd * kd;
  }
  float dcy = Qf(2 * nq + n + h);
  float bta = Qf(2 * nq + n + u_hv + h);
  float delta = (Qf(2 * nq + h * u_dv + vi) - dcy * pred) * bta;
  fragColor = (i < n) ? (dcy * qs + delta * qk) : delta;
}
"""

_GDN_UPD_GL = _gl_head([("tex_S", "Sf"), ("tex_qkv", "Qf"), ("tex_od", "Of")],
                       ("u_hv", "u_dk", "u_dv", "u_rep", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int h = i / (u_dk * u_dv); int rem = i - h * (u_dk * u_dv);
  int d = rem / u_dv; int vi = rem - d * u_dv;
  int hk = u_hv / u_rep;
  int nq = hk * u_dk;
  float dcy = Qf(2 * nq + u_hv * u_dv + h);
  fragColor = Sf(i) * dcy + Qf(nq + (h % hk) * u_dk + d) * Of(u_hv * u_dv + h * u_dv + vi);
}
"""


def _webgl_gdn_prepare(qkv, braw, araw, cst, konst, out, hk, hv, dk, dv, W, flags):
    nq, nv = hk * dk, hv * dv
    C = 2 * nq + nv
    _gl_run("gdn_pre_gl", _GDN_PRE_GL,
            [("tex_qkv", qkv), ("tex_b", braw), ("tex_a", araw),
             ("tex_cst", cst), ("tex_k", konst)], out,
            [("u_hk", hk), ("u_hv", hv), ("u_dk", dk), ("u_dv", dv),
             ("u_W", W), ("u_flags", flags), ("u_n", C + 2 * hv)])
    if (flags & 1) and W > 1:
        cst = _gl_run("gdn_cst_gl", _GDN_CST_GL, [("tex_qkv", qkv), ("tex_cst", cst)],
                      _empty((int(cst.size),)),
                      [("u_C", C), ("u_W", W), ("u_n", int(cst.size))])
    return out, cst


def _webgl_gdn_step(S, qkv, hv, dk, dv, rep):
    n = hv * dv
    od = _empty((2 * n,))
    _gl_run("gdn_step_gl", _GDN_STEP_GL, [("tex_S", S), ("tex_qkv", qkv)], od,
            [("u_hv", hv), ("u_dk", dk), ("u_dv", dv), ("u_rep", max(1, rep)), ("u_n", n)])
    tot = hv * dk * dv
    S_next = _gl_run("gdn_upd_gl", _GDN_UPD_GL,
                     [("tex_S", S), ("tex_qkv", qkv), ("tex_od", od)], _empty((tot,)),
                     [("u_hv", hv), ("u_dk", dk), ("u_dv", dv),
                      ("u_rep", max(1, rep)), ("u_n", tot)])
    return Tensor(od[:n]), S_next


_SILU_GL = _gl_head([("tex_x", "Xf")], ("u_n",)) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  float v = Xf(i);
  fragColor = v / (1.0 + exp(-v));
}
"""


def _webgl_silu(x):
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    n = int(xd.size)
    of = _empty(tuple(xd.shape))
    _gl_run("silu_gl", _SILU_GL, [("tex_x", xd)], of, [("u_n", n)])
    return Tensor(of)


# Router scores -> the chosen experts and their weights.
#
# Two dispatches rather than one, because a fragment writes one value and the indices are
# int32 while the weights are float. Each fragment finds its own rank independently: fragment
# s runs s+1 argmax passes over the scores, skipping what the earlier passes took. That is
# O(k^2 * ne) against the workgroup version's O(k * ne), and at k=8 over 128 experts it is a
# few thousand comparisons in a kernel that runs once per layer -- against a read-back of the
# router's scores, which is what the host path costs and which drains the command queue.
_MOE_TOPK = 32           # fragments hold their picks in a local array, so this is a bound

_MOE_IDX_GL = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w;
uniform sampler2D tex_lg;
uniform int u_ne; uniform int u_k;
out int fragColor;
float Lf(int i) { ivec2 t = textureSize(tex_lg, 0);
  if (t.y == 1) { return texelFetch(tex_lg, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_lg, ivec2(i - y * t.x, y), 0).r; }
void main() {
  int s = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (s >= u_k) { fragColor = 0; return; }
  int chosen[%d];
  for (int t = 0; t <= s; t = t + 1) {
    float best = -1e30; int bi = 0;
    for (int e = 0; e < u_ne; e = e + 1) {
      bool taken = false;
      for (int j = 0; j < t; j = j + 1) { if (chosen[j] == e) { taken = true; } }
      float v = Lf(e);
      if (!taken && v > best) { best = v; bi = e; }
    }
    chosen[t] = bi;
  }
  fragColor = chosen[s];
}
""" % _MOE_TOPK

_MOE_W_GL = """#version 300 es
precision highp float; precision highp int;
precision highp sampler2D; precision highp isampler2D;
uniform int _ka_tex_output_texture_w;
uniform sampler2D tex_lg;
uniform isampler2D tex_idx;
uniform int u_ne; uniform int u_k; uniform int u_norm;
out float fragColor;
float Lf(int i) { ivec2 t = textureSize(tex_lg, 0);
  if (t.y == 1) { return texelFetch(tex_lg, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_lg, ivec2(i - y * t.x, y), 0).r; }
int If(int i) { ivec2 t = textureSize(tex_idx, 0);
  if (t.y == 1) { return texelFetch(tex_idx, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_idx, ivec2(i - y * t.x, y), 0).r; }
void main() {
  int s = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (s >= u_k) { fragColor = 0.0; return; }
  // softmax over ALL experts, then renormalised across the chosen ones (the Qwen
  // convention); with u_norm == 0 the raw softmax weights are kept.
  float mx = -1e30;
  for (int e = 0; e < u_ne; e = e + 1) { mx = max(mx, Lf(e)); }
  float den = 0.0;
  for (int e = 0; e < u_ne; e = e + 1) { den += exp(Lf(e) - mx); }
  float p = exp(Lf(If(s)) - mx) / den;
  if (u_norm == 1) {
    float tot = 0.0;
    for (int j = 0; j < u_k; j = j + 1) { tot += exp(Lf(If(j)) - mx) / den; }
    if (tot > 0.0) { p = p / tot; }
  }
  fragColor = p;
}
"""


def _webgl_moe_route(logits, eidx, ew, ne, k, norm):
    # A fragment holds its picks in a fixed local array, so a larger k would index past it
    # and route to whatever was in that slot -- a wrong expert, silently. Loud instead.
    if int(k) > _MOE_TOPK:
        raise RuntimeError("WebGL MoE routing is built for up to %d experts per token, "
                           "not %d -- raise _MOE_TOPK" % (_MOE_TOPK, int(k)))
    _gl_run("moe_idx_gl", _MOE_IDX_GL, [("tex_lg", logits)], eidx,
            [("u_ne", int(ne)), ("u_k", int(k))])
    _gl_run("moe_w_gl", _MOE_W_GL, [("tex_lg", logits), ("tex_idx", eidx)], ew,
            [("u_ne", int(ne)), ("u_k", int(k)), ("u_norm", 1 if norm else 0)])


# Fused single-position attention, on WebGL.
#
# The general path is about ten dispatches per layer -- transpose the cache, a batched
# matmul, a scale, a mask add, a multi-pass softmax, a second matmul, two reshapes -- and on
# this backend a dispatch is not free. Two passes replace all of it, and neither needs
# workgroup memory:
#
#   scores: one fragment per (head, position), each a dot product over `hd`
#   output: one fragment per (head, dim), walking the positions once with the running
#           max-and-sum of the online (Flash-Attention style) softmax
#
# The second pass never materialises the probabilities, so nothing is sized by the context
# length. Splitting it this way rather than doing everything in the output pass matters: the
# scores would otherwise be recomputed once per output dimension, which is `hd` times over.
_GQA_SCORE_GL = _gl_head([("tex_q", "Qf"), ("tex_k", "Kf")],
                         ("u_nh", "u_nkv", "u_hd", "u_S", "u_n"), ("u_scale",)) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int h = i / u_S; int s = i - h * u_S;
  int kvh = h / (u_nh / u_nkv);
  int qo = h * u_hd;
  int ko = (kvh * u_S + s) * u_hd;
  float acc = 0.0;
  for (int d = 0; d < u_hd; d = d + 1) { acc += Qf(qo + d) * Kf(ko + d); }
  fragColor = acc * u_scale;
}
"""

_GQA_OUT_GL = _gl_head([("tex_sc", "Sf"), ("tex_v", "Vf")],
                       ("u_nh", "u_nkv", "u_hd", "u_S", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int h = i / u_hd; int d = i - h * u_hd;
  int kvh = h / (u_nh / u_nkv);
  int so = h * u_S;
  int vo = kvh * u_S * u_hd + d;
  // Running max and sum, rescaling what is already accumulated when the max moves. The
  // first step has m at -inf, so its rescale factor is zero and the empty accumulator is
  // discarded rather than needing a special case.
  float m = -1e30; float l = 0.0; float acc = 0.0;
  for (int s = 0; s < u_S; s = s + 1) {
    float x = Sf(so + s);
    float mn = max(m, x);
    float w = exp(x - mn);
    float r = exp(m - mn);
    l = l * r + w;
    acc = acc * r + w * Vf(vo + s * u_hd);
    m = mn;
  }
  fragColor = acc / l;
}
"""


def _webgl_gqa_decode(qd, kd, vd, nh, nkv, hd, S, scale):
    sc = _empty((nh, S))
    _gl_run("gqa_score_gl", _GQA_SCORE_GL, [("tex_q", qd), ("tex_k", kd)], sc,
            [("u_nh", nh), ("u_nkv", nkv), ("u_hd", hd), ("u_S", S),
             ("u_n", nh * S), ("u_scale", float(scale))])
    of = _empty((nh, 1, hd))
    _gl_run("gqa_out_gl", _GQA_OUT_GL, [("tex_sc", sc), ("tex_v", vd)], of,
            [("u_nh", nh), ("u_nkv", nkv), ("u_hd", hd), ("u_S", S), ("u_n", nh * hd)])
    return of


# The KV scatter, on WebGL. There is no in-place form -- a fragment shader cannot render
# into a texture it samples -- so this reads the old cache and writes a new one, and returns
# it for the caller to keep. That is a pass over the WHOLE cache per token, against the
# growing `cat` path's pass over the positions actually in use, so it is the more expensive
# of the two until the context is about half of LMAX. It exists because the fixed-capacity
# cache is a real mode with a real caller, not because it should be the default here; the
# default stays the growing cache (see `KVCache`).
_KVWRITE_GL = _gl_head([("tex_dst", "Df"), ("tex_src", "Sf")],
                       ("u_pos", "u_T", "u_nkv", "u_hd", "u_lmax", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int hv = i / (u_lmax * u_hd); int rem = i - hv * (u_lmax * u_hd);
  int p = rem / u_hd; int d = rem - p * u_hd;
  if (p >= u_pos && p < u_pos + u_T) {
    fragColor = Sf((hv * u_T + (p - u_pos)) * u_hd + d);
  } else {
    fragColor = Df(i);
  }
}
"""


def _webgl_kv_write(cache, src, pos, T, nkv, hd, lmax):
    of = _empty(tuple(cache.shape))
    return _gl_run("kv_write_gl", _KVWRITE_GL,
                   [("tex_dst", cache), ("tex_src", _contig(src))], of,
                   [("u_pos", int(pos)), ("u_T", int(T)), ("u_nkv", int(nkv)),
                    ("u_hd", int(hd)), ("u_lmax", int(lmax)),
                    ("u_n", int(nkv) * int(lmax) * int(hd))])


def _webgl_swiglu(gd, ud, rows, half, gstride, ustride, uoff):
    of = _empty((rows, half))
    return _gl_run("swiglu_gl", _SWIGLU_GLSL, [("tex_g", gd), ("tex_u", ud)], of,
                   [("u_half", half), ("u_gstride", gstride), ("u_ustride", ustride),
                    ("u_uoff", uoff), ("u_n", rows * half)])


# The two-pass form has a problem at decode: pass one is one fragment per ROW, so at T = 1
# the whole reduction is a single invocation walking H dependent fetches while the rest of
# the GPU idles. The one-pass form has every output element re-derive the sum -- H times the
# arithmetic -- but spread across H fragments that run at once, and it is one launch instead
# of two. That trade inverts as T grows (the redundant work is O(T*H^2)), so the row count
# picks between them.
_RMS_ONE_GLSL = _gl_head([("tex_x", "Xf"), ("tex_w", "Wf")],
                         ("u_H", "u_n"), ("u_eps",)) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int r = i / u_H; int c = i - r * u_H;
  int base = r * u_H;
  float s = 0.0;
  for (int j = 0; j < u_H; j = j + 1) { float v = Xf(base + j); s += v * v; }
  fragColor = Xf(i) * inversesqrt(s / float(u_H) + u_eps) * Wf(c);
}
"""
_RMS_ONE_PASS_ROWS = 4     # measured: see the note above


def _webgl_rmsnorm(xd, wd, T, H, eps):
    if T <= _RMS_ONE_PASS_ROWS:
        of = _empty((T, H))
        return _gl_run("rms_one_gl", _RMS_ONE_GLSL, [("tex_x", xd), ("tex_w", wd)], of,
                       [("u_H", H), ("u_n", T * H), ("u_eps", float(eps))])
    ss = _empty((T,))
    _gl_run("rms_sum_gl", _RMS_SUM_GLSL, [("tex_x", xd)], ss, [("u_T", T), ("u_H", H)])
    of = _empty((T, H))
    return _gl_run("rms_apply_gl", _RMS_APPLY_GLSL,
                   [("tex_x", xd), ("tex_w", wd), ("tex_s", ss)], of,
                   [("u_H", H), ("u_n", T * H), ("u_eps", float(eps))])


def _webgl_rope(xd, cd, sd, n, HD, rd, T):
    of = _empty((n,))
    return _gl_run("rope_gl", _ROPE_GLSL,
                   [("tex_x", xd), ("tex_c", cd), ("tex_s", sd)], of,
                   [("u_n", n), ("u_HD", int(HD)), ("u_rd", int(rd)), ("u_T", int(T))])


# ==== the same quantized matmul, on WebGL ===============================================
#
# WebGL2 has no compute stage: every "kernel" is a fragment shader, one invocation per
# output element, with no shared memory between invocations and no way to write anywhere
# but its own fragment. The WGSL decode path is built around exactly the two things that
# removes -- a workgroup that stages the activation window in shared memory, and a split-K
# reduction across the threads of that workgroup -- so the shape here is different on
# purpose rather than a port that gave up.
#
# What survives unchanged is the part that matters: the weight layout. `ggml_transpose`
# lays a tensor out as (words, N), so the threads of a WGSL workgroup read adjacent words;
# adjacent FRAGMENTS read those same adjacent words, so the layout that makes the WebGPU
# kernel coalesce makes this one coalesce too, with nothing to change.
#
# What is different:
#   * one fragment owns one whole output row and runs the entire K loop itself. There is no
#     split-K because there is nowhere to reduce it. The parallelism comes from N instead,
#     which for a projection is 1024-5120 fragments -- enough to fill the machine.
#   * activations are read from a texture per value instead of from a staged window. Every
#     fragment reads the same ones in the same order, which is the case a texture cache is
#     built for.
#   * the three WGSL variants (one row, two rows, batched) collapse into one shader. They
#     exist there to divide work between the threads of a workgroup; here the output row
#     just falls out of the fragment index.
#
# The decode arithmetic itself is NOT duplicated -- `_wgsl2glsl` translates the same decoder
# bodies. Two hand-written copies would diverge at the first format added, and the copy
# nobody ran would be the broken one.

_GL_HEAD = """#version 300 es
precision highp float; precision highp int;
precision highp sampler2D; precision highp isampler2D;
uniform int _ka_tex_output_texture_w;
uniform sampler2D tex_x;
uniform isampler2D tex_w;
BIASUNIFORM
uniform int u_M; uniform int u_N; uniform int u_K; uniform int u_rowb;
uniform int u_estride; uniform int u_eslot; uniform int u_xper;
EXTRAUNIFORMS
out float fragColor;

struct GM { uint M; uint N; uint K; uint rowb; uint estride; uint eslot; uint xper; uint pad; };
GM gm;
uint nrow; uint woff; uint xrow; float acc0;
int _xw; int _ww; int _gw; int _ew; int _xh;

// The activation is a (1, K) row, so its texture is K wide and one tall for any hidden size
// up to the 16384 texel limit -- and then the row index is always zero, and the divide that
// computes it is dead. It is not free: this is the innermost read in the kernel, run once
// per weight VALUE rather than once per weight word, and a GPU has no integer divide.
// Measured over a 3B's weights, dropping it took the whole matmul sweep from 32.0 to
// 34.9 GB/s. The test is on the texture rather than on an assumption about K, and every
// fragment agrees on it, so it resolves once rather than per lane.
//
// A LOT more was tried here, because this read is where the fragment form loses to the
// compute one: a fragment owns an output row and reads the entire activation itself, so the
// eight kilobytes of activation cost 2.9 billion texture ops over a 3B's weights -- 44ms
// against 25ms to read all 1.67 GB of the weights. Packing four activations to an RGBA texel
// and keeping the last texel in the fragment cuts that count fourfold and is MUCH SLOWER:
// 16.3 GB/s against 34.9 for the matmul sweep with the packing free (done once, outside the
// timing), and 6.0 with the repack where it would really be. The branch is uniform, so this
// is not divergence -- a texelFetch is simply cheaper than a compare and a dynamically
// indexed vec4, and the fetches were already overlapping with the weight reads. Do not
// retry it without a measurement that says otherwise.
float Xf(int i) {
  if (_xh == 1) { return texelFetch(tex_x, ivec2(i, 0), 0).r; }
  int y = i / _xw; return texelFetch(tex_x, ivec2(i - y * _xw, y), 0).r;
}
// `woff` is a FLAT word offset, not a row-relative one: the WGSL this mirrors expands to
// `w[woff + wo * N + nrow]`, and `estride` is a whole expert's words. Bracketing it as
// `(woff + wo) * N` instead agrees for expert 0 and reads off the end of the buffer for
// every other one -- which is zeros, so slot 0 was exact and every other slot was empty.
uint  W(uint wo) { int i = int(woff) + int(wo) * int(gm.N) + int(nrow);
                   int y = i / _ww; return uint(texelFetch(tex_w, ivec2(i - y * _ww, y), 0).r); }
uint  B(uint o) { return (W(o >> 2u) >> ((o & 3u) * 8u)) & 255u; }
float I8(uint o) { uint v = B(o); return float(v) - ((v >= 128u) ? 256.0 : 0.0); }
// Four consecutive bytes in one fetch when the offset is word-aligned and two when it is
// not -- the same reason as the WGSL `B4`: most ggml blocks are not a multiple of four
// bytes, so a field read a byte at a time costs four fetches for four bytes that share a
// word, thousands of times per block.
uint  B4(uint o) { uint wo = o >> 2u; uint sh = (o & 3u) * 8u; uint lo = W(wo);
                   if (sh == 0u) { return lo; }
                   return (lo >> sh) | (W(wo + 1u) << (32u - sh)); }
uint  U16(uint o) { return B4(o) & 65535u; }
uint  U32(uint o) { return B4(o); }
float HF(uint h) {
  uint m = h & 1023u; uint e = (h >> 10u) & 31u; float v;
  if (e == 0u) { v = float(m) * 5.9604644775390625e-8; }
  else if (e == 31u) { v = 65504.0; }
  else { v = exp2(float(int(e) - 15)) * (1.0 + float(m) * 0.0009765625); }
  return ((h & 32768u) != 0u) ? -v : v;
}
float F16(uint o) { return HF(U16(o)); }
vec4 unpack4x8unorm(uint v) {
  return vec4(float(v & 255u), float((v >> 8u) & 255u),
              float((v >> 16u) & 255u), float((v >> 24u) & 255u)) * 0.00392156862745098;
}
void ACC(uint k, float v) { acc0 += Xf(int(xrow + k)) * v; }
void ACC4(uint k, vec4 v) { int i = int(xrow + k);
  acc0 += dot(vec4(Xf(i), Xf(i + 1), Xf(i + 2), Xf(i + 3)), v); }
GRIDFETCH
EIDXFETCH
"""

# The IQ4 codebook. On WebGPU it is staged in workgroup memory because computing it per
# value costs about ten instructions; here there is no workgroup memory, and a `const`
# array indexed dynamically is legal in GLSL ES 3.00 (the ES 2.0 restriction that made the
# WGSL side pack it into words does not apply), so it is just a lookup.
_KV_GLSL = """const float KVTAB[16] = float[16](
  -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
     1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0);
float kv(uint i) { return KVTAB[i]; }
"""

_GL_MAIN = """
void main() {
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  gm.M = uint(u_M); gm.N = uint(u_N); gm.K = uint(u_K); gm.rowb = uint(u_rowb);
  gm.estride = uint(u_estride); gm.eslot = uint(u_eslot); gm.xper = uint(u_xper);
  gm.pad = 0u;
  _xw = textureSize(tex_x, 0).x; _xh = textureSize(tex_x, 0).y;
  _ww = textureSize(tex_w, 0).x;
  GWINIT
  int r = idx / u_N;
  nrow = uint(idx - r * u_N);
  if (r >= u_ROWS) { fragColor = 0.0; return; }
  ROWINIT
  acc0 = 0.0;
  uint base = 0u;
  uint nb = gm.K / BLKVALS;
  for (uint b = 0u; b < nb; b = b + 1u) {
DECODE
  }
  fragColor = acc0 BIASADD;
}
"""

# `gr` is the i-quant codebook and `eidx` the routed expert; both are int32 textures.
_GL_GRIDFETCH = """uniform isampler2D tex_gr;
uint Gf(int i) { int y = i / _gw; return uint(texelFetch(tex_gr, ivec2(i - y * _gw, y), 0).r); }
"""
_GL_EIDXFETCH = """uniform isampler2D tex_e;
int Ef(int i) { int y = i / _ew; return texelFetch(tex_e, ivec2(i - y * _ew, y), 0).r; }
"""


def _ggml_src_gl(type_name, moe, moedec, bias=False):
    """GLSL ES 3.00 for one (format, routing, bias) combination.

    The bias is a compile-time variant rather than a second dispatch. It is one fetch at the
    end of a fragment that has already read a whole row of the weight, and it removes three
    launches per layer on the backend where a launch is worth the most -- a 36-layer model
    was spending more than a hundred of them a token on `out = out + bias`."""
    from . import _wgsl2glsl as w2g
    dec, helpers, vals, _, _ = _GGML_TYPES[type_name]
    ng = _grid_u32(type_name)
    # Substitutions that inject WGSL text run BEFORE translation, for the same reason they
    # run first on the WebGPU side: what they expand to contains further placeholders.
    def prep(t):
        return (t.replace("GRIDSTAGE", "").replace("GSRC", "gr")
                 .replace("BLKVALS", "%uu" % vals).replace("MASKBLK", "%uu" % (vals - 1))
                 .replace("WOFS", "").replace("XBAS", "").replace("OSLT", "")
                 .replace("GBIND", "0"))
    # The two helpers that ARE about workgroup memory rather than about decoding get a GLSL
    # form here. That is the line: the arithmetic is shared, the staging strategy cannot be.
    h = helpers
    kv_gl = ""
    if "var<workgroup> kvtab" in h:
        h = re.sub(r"//[^\n]*\n(?=.*?var<workgroup> kvtab)|var<workgroup> kvtab.*?"
                   r"fn kv\(i: u32\) -> f32 \{ return kvtab\[i\]; \}", "", h, flags=re.S)
        kv_gl = _KV_GLSL
    ty = w2g._Types()
    ty.fn.update({"W": "uint", "B": "uint", "B4": "uint", "I8": "float", "U16": "uint",
                  "U32": "uint", "F16": "float", "HF": "float", "ACC": "void",
                  "ACC4": "void", "kv": "float", "Gf": "uint", "Ef": "int"})
    ty.var.update({"base": "uint", "b": "uint", "nb": "uint", "nrow": "uint",
                   "acc0": "float", "gm": "GM", "xrow": "uint", "woff": "uint"})
    bufs = {"gr": "Gf", "eidx": "Ef"}
    glsl_h = w2g.translate(prep(h), ty, buffers=bufs) if h.strip() else ""
    glsl_d = w2g.translate(prep(dec), ty, buffers=bufs)
    head = (_GL_HEAD
            .replace("BIASUNIFORM", "uniform sampler2D tex_bias;" if bias else "")
            .replace("GRIDFETCH", _GL_GRIDFETCH if ng else "")
            .replace("EIDXFETCH", _GL_EIDXFETCH if moe else "")
            .replace("EXTRAUNIFORMS", "uniform int u_ROWS;"))
    main = (_GL_MAIN
            .replace("GWINIT", ("_gw = textureSize(tex_gr, 0).x;" if ng else "")
                     + ("\n  _ew = textureSize(tex_e, 0).x;" if moe else ""))
            .replace("ROWINIT", _gl_rowinit(moe, moedec))
            .replace("BLKVALS", "%uu" % vals)
            .replace("BIASADD", " + Bf(int(nrow))" if bias else "")
            .replace("DECODE", glsl_d))
    if bias:
        head += ("float Bf(int i) { ivec2 s = textureSize(tex_bias, 0); int y = i / s.x;\n"
                 "  return texelFetch(tex_bias, ivec2(i - y * s.x, y), 0).r; }\n")
    return head + kv_gl + glsl_h + main


def _gl_rowinit(moe, moedec):
    """Which expert this fragment reads and which activation row it multiplies.

    Three cases, and they are the same three the WGSL kernel has -- it just gets them from
    `gid.z` and a workgroup id instead of from the fragment index."""
    if moedec:
        # decode: the output row IS the routed slot, and each slot has its own expert
        return ("  woff = uint(Ef(r)) * gm.estride;\n"
                "  xrow = (gm.xper == 1u) ? uint(r) * gm.K : 0u;")
    if moe:
        # batched: one expert for the whole dispatch, output rows are batch rows
        return ("  woff = uint(Ef(int(gm.eslot))) * gm.estride;\n"
                "  xrow = uint(r) * gm.K;")
    return "  woff = 0u;\n  xrow = uint(r) * gm.K;"


_ggml_gl = {"added": set()}


def _ggml_name_gl(type_name, moe, moedec, bias=False):
    return "ggml_gl_%s_%s%s" % (type_name.lower().replace("-", "_"),
                                "md" if moedec else ("mb" if moe else "d"),
                                "_b" if bias else "")


def _ggml_add_gl(type_name, moe=False, moedec=False, bias=False):
    key = (type_name, moe, moedec, bias)
    if key in _ggml_gl["added"]:
        return
    plat = _copy_kernel["plat"]
    plat.addKernel(_ggml_name_gl(type_name, moe, moedec, bias),
                   {"source": _ggml_src_gl(type_name, moe, moedec, bias)})
    _ggml_gl["added"].add(key)


def _ggml_run_gl(xf, packed, type_name, K, N, eidx=None, eslot=0, estride=0, xper=False,
                 bias=None):
    """The WebGL dispatch. One fragment per output element; no workgroup shape to choose."""
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    moe = eidx is not None
    M = 1 if (moe and xper) else int(xf.shape[0])
    moedec = moe and M <= 2
    slots = int(eidx.size) if moedec else 1
    rows = slots * M
    _ggml_add_gl(type_name, moe, moedec, bias is not None)
    of = _empty((rows, N))
    grid = _ggml_grid(type_name)
    inputs = [{"name": "tex_x", "id": _contig(xf).buffer.buffer_id},
              {"name": "tex_w", "id": packed.buffer.buffer_id}]
    if moe:
        inputs.append({"name": "tex_e", "id": eidx.buffer.buffer_id})
    if grid is not None:
        inputs.append({"name": "tex_gr", "id": grid.buffer.buffer_id})
    if bias is not None:
        inputs.append({"name": "tex_bias", "id": _contig(bias).buffer.buffer_id})
    U = lambda n, v: {"name": n, "value": int(v), "type": "int"}
    plat = _copy_kernel["plat"]
    plat.runKernel({"name": _ggml_name_gl(type_name, moe, moedec, bias is not None),
                    "inputs": inputs, "output": of.buffer.buffer_id,
                    "uniforms": [U("_ka_tex_output_texture_w", of.buffer.texture_shape.width),
                                 U("u_M", M), U("u_N", N), U("u_K", K),
                                 U("u_rowb", (K // vals) * blk), U("u_estride", estride),
                                 U("u_eslot", eslot), U("u_xper", 1 if xper else 0),
                                 U("u_ROWS", rows)]})
    return of


def _ggml_run(xf, packed, type_name, K, N, small=_AUTO, eidx=None, eslot=0,
              estride=0, xper=False):
    """`eidx`/`eslot`/`estride` select one expert out of a stacked MoE weight at run time:
    the shader reads `eidx[eslot]` and offsets into `packed` by `estride` words. Passing them
    is what keeps the command identical from token to token, so the step stays capturable."""
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    M = 1 if (eidx is not None and xper) else int(xf.shape[0])
    mode = M if M <= 2 else 0
    if small is _AUTO:
        small = _shape_kind(N, K, vals) if mode == 1 else None
    moe = eidx is not None
    # Asking for a variant nobody built is the same silent failure the self-check exists to
    # catch: the platform does not know the name, runs nothing, and leaves the output buffer
    # zeroed -- which reads as a numerically wrong kernel. It cost a full sweep reported as
    # "168 of 168 formats broken" while the model beside it generated perfectly.
    if (type_name, mode, small, moe,
            _GGML_KSG if mode == 0 else 0) not in _ggml_k["added"]:
        raise RuntimeError("ggml kernel variant %r was never built -- go through ggml_matmul, "
                           "or pass the same `small` it derives (_AUTO works)"
                           % ((type_name, mode, small, moe),))
    # A MoE decode runs every routed slot in one dispatch, z indexing the slot, and returns
    # one row per slot for the caller to weight and sum. Batched prefill keeps z for its own
    # row blocking and takes a slot at a time.
    slots = int(eidx.size) if (moe and mode == 1) else 1
    name = _ggml_name(type_name, mode, small=small, moe=moe)
    plat = _adam_kernel["platform"]
    of = _empty((slots * M, N))
    meta = _adam_kernel["make_meta"]((M, N, K, (K // vals) * blk, estride, eslot,
                                      1 if xper else 0, 0), "u4,u4,u4,u4,u4,u4,u4,u4")
    bufs = [xf.buffer.buffer_id, packed.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id]
    if moe:
        bufs.append(eidx.buffer.buffer_id)
    grid = _ggml_grid(type_name)
    if grid is not None:
        bufs.append(grid.buffer.buffer_id)
    plat.runKernel({"name": name, "tensors": bufs,
                    "workGroups": {"x": ((_gemv_groups(N, mode, vals, small)
                                          if M <= 2 else (N + 63) // 64)), "y": 1,
                                   "z": slots if M <= 2 else (M + 4 * _GGML_KSG - 1)
                                                              // (4 * _GGML_KSG)}})
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
    # The zero points do not depend on the column block, so they are unpacked ONCE. Rebuilding
    # this (groups, N) array inside the loop re-did the same work for every block and held a
    # second copy of it while doing so.
    z_all = np.empty((int(qz.shape[0]), N), np.int32)
    for r in range(per):
        z_all[:, r::per] = (qz >> (bits * r)) & qmax
    zo = np.float32(zoff)
    for c0 in range(0, N, block):                 # column blocks bound the temporary
        c1 = min(N, c0 + block)
        qwb = qw[:, c0:c1]
        q = np.empty((Kp, c1 - c0), np.int32)
        for r in range(per):
            q[r::per] = (qwb >> (bits * r)) & qmax
        # float32 the whole way. `q - (z + zoff)` with a Python float for `zoff` promoted the
        # difference to float64, so the scaled weight was built at eight bytes a value and
        # then thrown away at four: a (1024, 2048) block asked for 16 MiB it did not need,
        # which is precisely the allocation that failed on a real machine.
        q -= z_all[g, c0:c1]                      # int32 - int32, in place
        w = q.astype(np.float32)
        if zoff:
            w -= zo
        w *= sc[g, c0:c1]
        out[:, c0:c1] = x @ w
        del q, w
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
# The recurrence's read pass: each (head, value-dim) pair reduces over the key dimension.
#
# One thread per pair leaves 6144 of them for a 48-head layer -- 96 workgroups, each thread
# walking 128 state elements in series with a 512-byte stride. That is far too little
# parallelism to hide the latency: it read 3MB in 0.61ms, about 5 GB/s on a machine that
# streams at 100. Splitting the key loop across a few lanes and reducing at the end fixes it
# without touching the arithmetic: 5.4x faster, and identical output (max rel err 1.4e-07).
# Two lanes already saturate it; four leaves headroom for models with a larger key dim.
_GDN_SPLIT = 4

_GDN_STEP_WGSL = """@group(0) @binding(0)
var<storage,read> S: array<f32>;
@group(0) @binding(1)
var<storage,read> qkv: array<f32>;
@group(0) @binding(2)
var<storage,read_write> od: array<f32>;
struct GD { hv: u32, dk: u32, dv: u32, rep: u32, }
@group(0) @binding(3)
var<storage,read> gd: GD;
var<workgroup> rp: array<f32, SPSZ>;
var<workgroup> rq: array<f32, SPSZ>;
var<workgroup> rk: array<f32, SPSZ>;
@compute @workgroup_size(64, SPN)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let lx = lid.x;
  let ly = lid.y;
  let i = wid.x * 64u + lx;
  let n = gd.hv * gd.dv;
  let h = i / gd.dv;
  let vi = i % gd.dv;
  // qkv packs q | k | v | decay | beta for this token. q and k are stored per KEY head and
  // the key heads CYCLE across the value heads (ggml: iq1 = iv1 % n_q_heads), so this is a
  // modulo, not a divide -- the block mapping pairs each query with the wrong key.
  let hk = gd.hv / gd.rep;
  let nq = hk * gd.dk;
  let qo = (h % hk) * gd.dk;
  let ko = nq + qo;
  let sbase = h * gd.dk * gd.dv + vi;
  var pred: f32 = 0.0;
  var qs: f32 = 0.0;
  var qk: f32 = 0.0;
  // No early return: the barrier below has to be reached by every lane.
  if (i < n) {
    for (var d: u32 = ly; d < gd.dk; d = d + SPu) {
      let sv = S[sbase + d * gd.dv];
      let kd = qkv[ko + d];
      let qd = qkv[qo + d];
      pred = pred + kd * sv;
      qs = qs + qd * sv;
      qk = qk + qd * kd;
    }
  }
  let sl = ly * 64u + lx;
  rp[sl] = pred; rq[sl] = qs; rk[sl] = qk;
  workgroupBarrier();
  if (ly == 0u && i < n) {
    var p: f32 = 0.0; var q: f32 = 0.0; var k: f32 = 0.0;
    for (var t: u32 = 0u; t < SPu; t = t + 1u) {
      p = p + rp[t * 64u + lx]; q = q + rq[t * 64u + lx]; k = k + rk[t * 64u + lx];
    }
    let dcy = qkv[2u * nq + gd.hv * gd.dv + h];
    let bta = qkv[2u * nq + gd.hv * gd.dv + gd.hv + h];
    let vo = 2u * nq + h * gd.dv;
    let delta = (qkv[vo + vi] - dcy * p) * bta;
    od[i] = dcy * q + delta * k;          // output, from the old state
    od[n + i] = delta;                    // handed to the update pass
  }
}
""".replace("SPSZ", str(64 * _GDN_SPLIT)).replace("SPu", "%du" % _GDN_SPLIT) \
   .replace("SPN", str(_GDN_SPLIT))

# The whole recurrence for a whole prompt, in ONE dispatch.
#
# The per-token pair above costs two dispatches per token per layer. On a 65-layer hybrid
# with 48 recurrent layers that is 2 x 48 x T: a 1100-token prompt spent 635 seconds before
# its first token, and the arithmetic was never the problem -- 53,000 dispatches at the
# fixed cost of a dispatch were.
#
# What makes one dispatch possible is that the state does not have to move. A workgroup owns
# one (head, value-channel) pair, so it owns exactly the dk state elements S[h, :, vi]; those
# live in registers for the whole scan, and the update that needed a second dispatch is just
# those threads writing their own registers. T becomes a loop inside the kernel.
#
# The recurrence stays strictly sequential -- every workgroup walks t in order, and state t
# is used before state t+1 is written -- so this is the same computation, not an
# approximation of it.
_GDN_SCAN_WGSL = """@group(0) @binding(0)
var<storage,read_write> S: array<f32>;
@group(0) @binding(1)
var<storage,read> qkv: array<f32>;
@group(0) @binding(2)
var<storage,read_write> od: array<f32>;
struct GD { hv: u32, dk: u32, dv: u32, rep: u32, T: u32, row: u32, }
@group(0) @binding(3)
var<storage,read> gd: GD;
var<workgroup> rp: array<f32, 64>;
var<workgroup> rq: array<f32, 64>;
var<workgroup> rk: array<f32, 64>;
var<workgroup> sh_delta: f32;
var<workgroup> sh_dcy: f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let lx = lid.x;
  let i = wid.x + wid.z * nwg.x;            // folded dispatch: see the platform's runKernel
  let n = gd.hv * gd.dv;
  if (i >= n) { return; }                   // whole workgroup leaves together
  let h = i / gd.dv;
  let vi = i % gd.dv;
  // q and k are stored per KEY head and the key heads CYCLE across value heads
  // (ggml: iq1 = iv1 % n_q_heads), so this is a modulo, not a divide.
  let hk = gd.hv / gd.rep;
  let nq = hk * gd.dk;
  let qo = (h % hk) * gd.dk;
  let ko = nq + qo;
  let sbase = h * gd.dk * gd.dv + vi;

  // This workgroup's slice of the state, held for the whole scan. 8 slots covers dk <= 512;
  // beyond that the kernel would silently drop terms, so the caller checks before using it.
  var sv: array<f32, 8>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    let d = lx + j * 64u;
    if (d < gd.dk) { sv[j] = S[sbase + d * gd.dv]; } else { sv[j] = 0.0; }
  }

  for (var t: u32 = 0u; t < gd.T; t = t + 1u) {
    let base = t * gd.row;
    var pred: f32 = 0.0;
    var qs: f32 = 0.0;
    var qk: f32 = 0.0;
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let d = lx + j * 64u;
      if (d < gd.dk) {
        let kd = qkv[base + ko + d];
        let qd = qkv[base + qo + d];
        pred = pred + kd * sv[j];
        qs = qs + qd * sv[j];
        qk = qk + qd * kd;
      }
    }
    rp[lx] = pred; rq[lx] = qs; rk[lx] = qk;
    workgroupBarrier();
    // Tree reduction rather than one lane adding 64 numbers: the lane doing that is the
    // whole workgroup's critical path, and this runs T times.
    for (var off: u32 = 32u; off > 0u; off = off >> 1u) {
      if (lx < off) {
        rp[lx] = rp[lx] + rp[lx + off];
        rq[lx] = rq[lx] + rq[lx + off];
        rk[lx] = rk[lx] + rk[lx + off];
      }
      workgroupBarrier();
    }
    if (lx == 0u) {
      let dcy = qkv[base + 2u * nq + n + h];
      let bta = qkv[base + 2u * nq + n + gd.hv + h];
      let vv  = qkv[base + 2u * nq + h * gd.dv + vi];
      let delta = (vv - dcy * rp[0]) * bta;
      od[t * n + i] = dcy * rq[0] + delta * rk[0];
      sh_delta = delta;
      sh_dcy = dcy;
    }
    workgroupBarrier();
    // S = S * decay + outer(k, delta) -- the second dispatch of the per-token pair, done
    // here by the threads that already hold the state.
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let d = lx + j * 64u;
      if (d < gd.dk) { sv[j] = sv[j] * sh_dcy + qkv[base + ko + d] * sh_delta; }
    }
    workgroupBarrier();
  }

  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    let d = lx + j * 64u;
    if (d < gd.dk) { S[sbase + d * gd.dv] = sv[j]; }
  }
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
struct GP { hk: u32, hv: u32, dk: u32, dv: u32, W: u32, flags: u32, base: u32, }
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
      outp[gp.base + 2u * nq + nv + t] = exp(min(d, 0.0));
      var bv: f32 = 1.0;
      if ((gp.flags & 8u) != 0u) { bv = 1.0 / (1.0 + exp(-braw[t])); }
      outp[gp.base + 2u * nq + nv + gp.hv + t] = bv;
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
      outp[gp.base + c0 + t] = v;
    }
  } else if (t < dim) {
    outp[gp.base + c0 + t] = val;
  }
}
"""

# The same preparation for a whole prompt, dispatched over (head, token) instead of once
# per token.
#
# The ring buffer looked like a dependency across tokens and is not one. The convolution is
# causal over W taps, so token t's window is inputs t-W+1..t: inside the prompt those rows
# are already in `qkv`, and only the first W-1 tokens reach back into the incoming state. W
# is 4 here. Nothing has to be shifted while the prompt is being prepared -- the ring is
# rewritten once at the end from the prompt's own last W-1 rows.
#
# This is what is left after the recurrence was folded into one dispatch: 48 layers times T
# preparations was the whole remaining cost of a long prompt.
_GDN_PREB_WGSL = """@group(0) @binding(0)
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
struct GP { hk: u32, hv: u32, dk: u32, dv: u32, W: u32, flags: u32, T: u32, row: u32, }
@group(0) @binding(6)
var<storage,read> gp: GP;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let g = wg.x;
  let tk = wg.y;                                   // which token of the prompt
  let t = lid.x;
  let nq = gp.hk * gp.dk;
  let nv = gp.hv * gp.dv;
  let C = 2u * nq + nv;
  let heads = 2u * gp.hk + gp.hv;
  let ob = tk * gp.row;
  if (g >= heads) {
    if (t < gp.hv) {
      let ko = gp.W * C + C;
      var a = araw[tk * gp.hv + t];
      if ((gp.flags & 2u) != 0u) { a = a + konst[ko + gp.hv + t]; }
      let sp = max(a, 0.0) + log(1.0 + exp(-abs(a)));
      var d = sp;
      if ((gp.flags & 4u) != 0u) { d = sp * konst[ko + t]; }
      outp[ob + 2u * nq + nv + t] = exp(min(d, 0.0));
      var bv: f32 = 1.0;
      if ((gp.flags & 8u) != 0u) { bv = 1.0 / (1.0 + exp(-braw[tk * gp.hv + t])); }
      outp[ob + 2u * nq + nv + gp.hv + t] = bv;
    }
    return;
  }
  var c0: u32; var dim: u32; var isqk: bool;
  if (g < 2u * gp.hk) { c0 = g * gp.dk; dim = gp.dk; isqk = true; }
  else { c0 = 2u * nq + (g - 2u * gp.hk) * gp.dv; dim = gp.dv; isqk = false; }
  var val: f32 = 0.0;
  if (t < dim) {
    let c = c0 + t;
    if ((gp.flags & 1u) != 0u) {
      var acc: f32 = 0.0;
      for (var j: u32 = 0u; j < gp.W; j = j + 1u) {
        // input at t - (W-1) + j: a row of this prompt when that is >= 0, and otherwise the
        // incoming ring, whose entry j+tk is the same position (cst[0] is the oldest).
        var x: f32;
        if (tk + j + 1u >= gp.W) {
          x = qkv[(tk + j + 1u - gp.W) * C + c];
        } else {
          x = cst[(j + tk) * C + c];
        }
        acc = acc + x * konst[j * C + c];
      }
      if ((gp.flags & 16u) != 0u) { acc = acc + konst[gp.W * C + c]; }
      val = acc / (1.0 + exp(-acc));               // SiLU
    } else {
      val = qkv[tk * C + c];
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
      if (g < gp.hk) { v = v * inverseSqrt(f32(gp.dk)); }
      outp[ob + c0 + t] = v;
    }
  } else if (t < dim) {
    outp[ob + c0 + t] = val;
  }
}
"""

# The ring, rewritten once from the prompt's own last W-1 input rows. Separate because it
# must not run until every token has read the OLD ring.
_GDN_RING_WGSL = """@group(0) @binding(0)
var<storage,read_write> cst: array<f32>;
@group(0) @binding(1)
var<storage,read> qkv: array<f32>;
struct GR { C: u32, W: u32, T: u32, }
@group(0) @binding(2)
var<storage,read> gr: GR;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.z * nwg.x * 64u;
  let n = (gr.W - 1u) * gr.C;
  if (i >= n) { return; }
  let j = i / gr.C;
  let c = i % gr.C;
  cst[i] = qkv[(gr.T + j + 1u - gr.W) * gr.C + c];
}
"""

_gdnpb_k = {"added": False}


def gdn_prepare_batch(qkv, braw, araw, cst, konst, out, hk, hv, dk, dv, W, flags, T, row):
    """`gdn_prepare` for T tokens at once. Returns (out, cst) or None if unavailable.

    T must be at least W-1: a shorter prompt's new ring would have to be part old and part
    new, and stepping such a prompt costs nothing worth the branch.
    """
    if _webgl_ready() and not _adam_backend_ready():
        return None
    if T < max(1, W) - 1 or T < 1:
        return None
    plat = _adam_kernel["platform"]
    if not _gdnpb_k["added"]:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel("gdn_preb", {"source": _GDN_PREB_WGSL,
                                    "bindingTypes": [ro, ro, ro, rw, ro, rw, ro]})
        plat.addKernel("gdn_ring", {"source": _GDN_RING_WGSL,
                                    "bindingTypes": [rw, ro, ro]})
        _gdnpb_k["added"] = True
    nq = hk * dk
    C = 2 * nq + hv * dv
    meta = _adam_kernel["make_meta"]((hk, hv, dk, dv, W, flags, T, row),
                                     "u4,u4,u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "gdn_preb",
                    "tensors": [qkv.buffer.buffer_id, braw.buffer.buffer_id,
                                araw.buffer.buffer_id, cst.buffer.buffer_id,
                                konst.buffer.buffer_id, out.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": 2 * hk + hv + 1, "y": T, "z": 1}})
    if (flags & 1) and W > 1:
        rmeta = _adam_kernel["make_meta"]((C, W, T), "u4,u4,u4")
        n = (W - 1) * C
        plat.runKernel({"name": "gdn_ring",
                        "tensors": [cst.buffer.buffer_id, qkv.buffer.buffer_id,
                                    rmeta.buffer_id],
                        "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return out, cst


_gdnp_k = {"added": False}


def gdn_prepare(qkv, braw, araw, cst, konst, out, hk, hv, dk, dv, W, flags, base=0):
    """Conv + SiLU + L2 norm + gates, writing the packed q|k|v|decay|beta the step wants.

    Returns `(out, cst_next)`. `cst_next` is `cst` itself where the backend updates the
    ring buffer in place, and a NEW buffer where it cannot -- a fragment shader may not read
    the texture it writes, so WebGL shifts the ring into a fresh one. The caller stores what
    comes back rather than assuming which happened; copying it into the original instead
    costs a full pass over the buffer per layer per token.

    `flags` bits: 1 conv, 2 dt_bias, 4 A, 8 beta projection, 16 conv bias."""
    if _webgl_ready() and not _adam_backend_ready():
        # The WebGL kernel writes from index 0 and has no offset. Callers ask `gdn_scan_ok`
        # before packing a prompt into one buffer, so this is a contract violation rather
        # than a case to handle quietly -- writing row 7 over row 0 would produce fluent,
        # wrong output.
        if base:
            raise NotImplementedError("gdn_prepare: the WebGL path has no output offset")
        return _webgl_gdn_prepare(qkv, braw, araw, cst, konst, out, hk, hv, dk, dv, W, flags)
    plat = _adam_kernel["platform"]
    if not _gdnp_k["added"]:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel("gdn_pre", {"source": _GDN_PRE_WGSL,
                                   "bindingTypes": [ro, ro, ro, rw, ro, rw, ro]})
        _gdnp_k["added"] = True
    meta = _adam_kernel["make_meta"]((hk, hv, dk, dv, W, flags, int(base)),
                                     "u4,u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "gdn_pre",
                    "tensors": [qkv.buffer.buffer_id, braw.buffer.buffer_id,
                                araw.buffer.buffer_id, cst.buffer.buffer_id,
                                konst.buffer.buffer_id, out.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": 2 * hk + hv + 1, "y": 1, "z": 1}})
    return out, cst


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
struct TM { n: u32, words: u32, rowb: u32, total: u32, gx: u32, dstoff: u32, }
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
  dst[tm.dstoff + i] = v;
  if (i == 0u) { flag[0] = 1.0; }        // four bytes to read back, instead of the tensor
}
"""

_tr_k = {"added": False}


_TRANSPOSE_STACK_GLSL = """#version 300 es
precision highp float; precision highp int; precision highp isampler2D;
uniform int _ka_tex_output_texture_w;
uniform isampler2D tex_src;
uniform int u_n; uniform int u_words; uniform int u_rowb; uniform int u_perw; uniform int u_total;
out int fragColor;
int _sw;
uint sw_(int i) { int y = i / _sw; return uint(texelFetch(tex_src, ivec2(i - y * _sw, y), 0).r); }
uint sb(uint o) { return (sw_(int(o >> 2u)) >> ((o & 3u) * 8u)) & 255u; }
void main() {
  int i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (i >= u_total) { fragColor = 0; return; }
  _sw = textureSize(tex_src, 0).x;
  int per = u_words * u_n;
  int e = i / per; int rem = i - e * per;
  int wo = rem / u_n; int row = rem - wo * u_n;
  uint b = uint(e) * uint(u_perw) * 4u + uint(row) * uint(u_rowb) + uint(wo) * 4u;
  uint end = uint(e) * uint(u_perw) * 4u + uint(row + 1) * uint(u_rowb);
  uint v = 0u;
  if (b + 3u < end) {
    v = sb(b) | (sb(b + 1u) << 8u) | (sb(b + 2u) << 16u) | (sb(b + 3u) << 24u);
  } else {
    for (uint k = 0u; k < 4u; k = k + 1u) {
      if (b + k >= end) { break; }
      v = v | (sb(b + k) << (8u * k));
    }
  }
  fragColor = int(v);
}
"""
_tr_gls = {"added": False}


def _ggml_transpose_gl_stack(src, n, rowb, ne, perw):
    """Every expert of a stacked MoE weight transposed in ONE pass.

    A fragment writes its own fragment and the platform renders the whole texture, so the
    per-expert form the WebGPU path uses -- transpose expert e into its slice of a shared
    destination -- would clear the experts already written. Adding the expert to the index
    arithmetic instead makes it a single pass, and the whole layer's expert bytes are in host
    memory at once anyway: the loader fetches them in one range request and slices.

    `perw` is the padded words per expert in the SOURCE; `rowb` its bytes per row."""
    words = (rowb + 3) // 4
    total = words * n * ne
    plat = _copy_kernel["plat"]
    if not _tr_gls["added"]:
        plat.addKernel("ggml_tr_stack_gl", {"source": _TRANSPOSE_STACK_GLSL})
        _tr_gls["added"] = True
    dst = _empty_i32((total,))
    U = lambda k, v: {"name": k, "value": int(v), "type": "int"}
    plat.runKernel({"name": "ggml_tr_stack_gl",
                    "inputs": [{"name": "tex_src", "id": src.buffer.buffer_id}],
                    "output": dst.buffer.buffer_id,
                    "uniforms": [U("_ka_tex_output_texture_w", dst.buffer.texture_shape.width),
                                 U("u_n", n), U("u_words", words), U("u_rowb", rowb),
                                 U("u_perw", perw), U("u_total", total)]})
    return dst


_TRANSPOSE_GLSL = """#version 300 es
precision highp float; precision highp int; precision highp isampler2D;
uniform int _ka_tex_output_texture_w;
uniform isampler2D tex_src;
uniform int u_n; uniform int u_words; uniform int u_rowb; uniform int u_total;
out int fragColor;
int _sw;
uint sw_(int i) { int y = i / _sw; return uint(texelFetch(tex_src, ivec2(i - y * _sw, y), 0).r); }
uint sb(uint o) { return (sw_(int(o >> 2u)) >> ((o & 3u) * 8u)) & 255u; }
void main() {
  int i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (i >= u_total) { fragColor = 0; return; }
  _sw = textureSize(tex_src, 0).x;
  uint wo = uint(i) / uint(u_n);
  uint row = uint(i) - wo * uint(u_n);
  uint b = row * uint(u_rowb) + wo * 4u;
  uint end = (row + 1u) * uint(u_rowb);
  uint v = 0u;
  if (b + 3u < end) {
    v = sb(b) | (sb(b + 1u) << 8u) | (sb(b + 2u) << 16u) | (sb(b + 3u) << 24u);
  } else {
    for (uint k = 0u; k < 4u; k = k + 1u) {
      if (b + k >= end) { break; }
      v = v | (sb(b + k) << (8u * k));
    }
  }
  fragColor = int(v);
}
"""
_tr_gl = {"added": False}


def _ggml_transpose_gl(src, n, rowb):
    """(n, rowb bytes) -> (words, n) int32, as a fragment pass.

    Only the whole-buffer form. Writing into a slice of a larger destination -- how a stack
    of MoE experts is assembled on WebGPU -- has no fragment-shader equivalent: an
    invocation writes its own fragment and nothing else, and the platform renders the whole
    texture rather than a sub-rectangle, so the experts already written would be cleared.
    WebGL therefore keeps one buffer per expert (see `GGMLMoELinear`), which costs nothing:
    stacking exists to keep a captured command list identical from token to token, and a
    non-stacked MoE is not capturable on either backend anyway."""
    words = (rowb + 3) // 4
    total = words * n
    plat = _copy_kernel["plat"]
    if not _tr_gl["added"]:
        plat.addKernel("ggml_tr_gl", {"source": _TRANSPOSE_GLSL})
        _tr_gl["added"] = True
    dst = _empty_i32((total,))
    U = lambda k, v: {"name": k, "value": int(v), "type": "int"}
    plat.runKernel({"name": "ggml_tr_gl",
                    "inputs": [{"name": "tex_src", "id": src.buffer.buffer_id}],
                    "output": dst.buffer.buffer_id,
                    "uniforms": [U("_ka_tex_output_texture_w", dst.buffer.texture_shape.width),
                                 U("u_n", n), U("u_words", words), U("u_rowb", rowb),
                                 U("u_total", total)]})
    return dst


def ggml_transpose(src, n, rowb, dst=None, dstoff=0):
    """(n, rowb bytes) -> (words, n) u32, so a matmul's threads read adjacent words.

    `dst`/`dstoff` write into an existing buffer instead of a fresh one, which is how a
    stack of MoE experts is assembled: each is transposed straight into its slice. Assigning
    into a slice from the host instead would read the whole destination back -- and at a
    hundred-odd megabytes the backend refuses to stage it."""
    if _webgl_ready() and not _adam_backend_ready():
        if dst is not None:
            raise RuntimeError("ggml_transpose: WebGL has no sliced write -- keep experts "
                               "in their own buffers on this backend")
        return _ggml_transpose_gl(src, n, rowb)
    words = (rowb + 3) // 4
    plat = _adam_kernel["platform"]
    if not _tr_k["added"]:
        plat.addKernel("ggml_tr", {"source": _TRANSPOSE_WGSL,
                                   "bindingTypes": ["read-only-storage", "storage",
                                                    "read-only-storage", "storage"]})
        _tr_k["added"] = True
    total = words * n
    if dst is None:
        dst = _empty((total,))
        dstoff = 0
    groups = (total + 63) // 64
    gx = min(groups, 32768)                     # a dispatch dimension caps at 65535
    gy = (groups + gx - 1) // gx
    meta = _adam_kernel["make_meta"]((n, words, rowb, total, gx, int(dstoff)),
                                     "u4,u4,u4,u4,u4,u4")
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
    Returns `(output, S_next)` -- see `gdn_prepare` for why the state comes back."""
    if _webgl_ready() and not _adam_backend_ready():
        return _webgl_gdn_step(S, qkv, hv, dk, dv, rep)
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
    return Tensor(od[:n]), S


def gdn_scan_ok(dk):
    """Can this backend run the whole recurrence in one dispatch for this head width?

    Asked BEFORE a prompt is packed, because the packing differs: the scan wants one buffer
    with a row per token, and the fallback wants a buffer per token.
    """
    if dk > 8 * 64:                       # the workgroup holds 8 registers of state per lane
        return False
    return not (_webgl_ready() and not _adam_backend_ready())


def gdn_scan(S, qkv, T, hv, dk, dv, rep=1):
    """The whole Gated-DeltaNet recurrence for T tokens, in one dispatch.

    `qkv` holds T rows of the same packing `gdn_step` takes, laid out contiguously.
    Returns (outputs (T, hv*dv), S) with the state advanced past the last row.

    Returns None when it cannot be used, and the caller falls back to stepping: the state
    slice a workgroup holds is 8 registers deep, so dk above 512 would silently drop terms,
    and a backend without this kernel has nothing to run.
    """
    if not gdn_scan_ok(dk):
        return None
    plat = _adam_kernel["platform"]
    if not _gdn_scan_k["added"]:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel("gdn_scan", {"source": _GDN_SCAN_WGSL,
                                    "bindingTypes": [rw, ro, rw, ro]})
        _gdn_scan_k["added"] = True
    n = hv * dv
    nq = (hv // max(1, rep)) * dk
    row = 2 * nq + n + 2 * hv
    out = _empty((T * n,))
    meta = _adam_kernel["make_meta"]((hv, dk, dv, max(1, rep), T, row),
                                     "u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "gdn_scan",
                    "tensors": [S.buffer.buffer_id, qkv.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": n, "y": 1, "z": 1}})
    return Tensor(out).reshape(T, n), S


_gdn_scan_k = {"added": False}


# Reporting the ledger out, at most once a second.
#
# The worker is single-threaded: while a reply is being written nothing can ASK for these
# numbers, so a page that polls sees whatever was true before the reply started -- measured
# at 51 seconds outstanding on one prefill. Pushing is the only way they are live, and the
# push has to come from somewhere that runs constantly, which is why it hangs off the matmul
# rather than off the layer loop: eight pushes across a prefill is eleven seconds apart.
# No clock. The destination is a shared array, so a write is three stores and rationing it
# by time would cost more than it saves -- an earlier version did read a clock here, using a
# `time` this module does not import, and the NameError went into the blanket `except` below
# and stayed there: the counter advanced 516 times and the array never received a byte.
_stat_n = 0
_stat_why = None                  # why the last attempt failed, so a silent one cannot hide


def _gpu_stat_push(force=False):
    """Write the GPU ledger where the page can read it. Every 64th call, which is several
    times a layer -- often enough that a display refreshing once a second is never behind.

    `force` for the moments that are NOT in a hot loop and matter most: the end of a reply,
    where several gigabytes are handed back at once. Without it the last value written is
    whatever the final matmul saw, and it stands until the next reply -- a reader watching
    the number sees the peak of a prefill and no sign that it has since been released.
    """
    global _stat_n, _stat_why
    _stat_n += 1
    if not force and (_stat_n & 63):
        return
    try:
        import js
        hook = getattr(js.self, "__gpustat", None)
        if hook is None:
            _stat_why = "no hook installed"
            return
        from wgpy_backends.webgpu.platform import get_platform as _gp
        held, peak, n = _gp().gpuBytes()
        hook(held, peak, n)
        _stat_why = None
    except Exception as e:
        # Kept, not swallowed. This path is best-effort, so it must not raise -- but a
        # failure that leaves no trace is how the last one survived.
        _stat_why = type(e).__name__ + ": " + str(e)[:80]


def gpu_reap():
    """Return finished intermediates to the device. A no-op where there is nothing to return.

    Called at a LAYER boundary, not from the allocation path, and the difference is not
    subtle. A collect can only free what nothing refers to, and inside `WebGPUBuffer.__init__`
    the call chain still refers to plenty -- a byte-budgeted reap there fired every two or
    three layers and the ledger climbed straight through it, 13.5GB to 19.6GB in one prefill.
    The same collect at the boundary between layers holds it flat at 14.0-14.3GB, costs 0.2s
    across a whole prefill, and makes that prefill FASTER (100.2s to 87.2s) because what it
    stops is the paging.
    """
    try:
        import wgpy_backends.webgpu.webgpu_buffer as _b
    except Exception:
        return                                   # WebGL, or no GPU backend at all
    fn = getattr(_b, "reap_now", None)
    if fn is not None:
        fn()
    _gpu_stat_push(force=True)


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
                         self.type_name, self.Kt, self.Nt, bias=self.bias)
        return Tensor(of.reshape(*lead, self.Nt))                 # inference-only


class GGMLMoELinear(Module):
    """One projection of a sparse-MoE layer: every expert's weight, stacked in one buffer.

    A MoE layer picks a few experts per token, so which weight a matmul reads is decided at
    run time. Holding one Linear per expert forces that choice into the command stream --
    which expert kernels get dispatched -- and a captured decode step then replays whatever
    the first token selected, for every token after it. Stacking them and passing an index
    keeps the command identical: the shader offsets into this buffer by `estride` words.
    This is the same shape as llama.cpp's ggml_mul_mat_id.

    Experts are transposed one at a time into the destination, so the peak stays at a single
    expert rather than the whole stack.
    """

    def __init__(self, chunks, type_name, K, N):
        vals, blk_b = _GGML_TYPES[type_name][2], _GGML_TYPES[type_name][3]
        rowb = (int(K) // vals) * blk_b
        words = (rowb + 3) // 4
        self.estride = words * int(N)
        ne = len(chunks)
        if _webgl_ready() and not _adam_backend_ready():
            # One upload and one pass. Per-expert transposes into slices of a shared
            # destination have no fragment-shader form (see `_ggml_transpose_gl_stack`), and
            # the alternative -- one buffer per expert -- costs the routed kernel and, with
            # it, the device-side router: the host would have to read the router's scores
            # back to decide which expert's buffer to bind, once per layer per token.
            nb = max(len(c) for c in chunks)
            perb = nb + (-nb) % 4
            buf = np.zeros(perb * ne, np.uint8)
            for e, raw in enumerate(chunks):
                b = np.frombuffer(raw, np.uint8)
                buf[e * perb:e * perb + b.size] = b
            up = xp.asarray(buf.view(np.int32))
            del buf
            self.packed = _ggml_transpose_gl_stack(up, int(N), rowb, ne, perb // 4)
            del up
            self.type_name = type_name
            self.Kt = int(K); self.Nt = int(N); self.n_experts = ne
            return
        dst = _empty((self.estride * ne,))
        for e, raw in enumerate(chunks):
            b = np.frombuffer(raw, np.uint8)
            pad = (-b.size) % 4
            if pad:
                b = np.concatenate([b, np.zeros(pad, np.uint8)])
            up = xp.asarray(b.view(np.int32))
            ggml_transpose(up, int(N), rowb, dst=dst, dstoff=e * self.estride)
            del up
        self.packed = dst
        self.type_name = type_name
        self.Kt = int(K); self.Nt = int(N); self.n_experts = ne

    def forward(self, x, eidx):
        """One dispatch for every expert in `eidx`; the result is (k, N), a row per slot.

        `x` is either a single row -- shared by all slots, as the first two projections of a
        routed layer are -- or already one row per slot, which is what the third takes from
        the second. The shader is told which by the row count."""
        xd = x.data
        xper = int(xd.shape[0]) > 1
        of = ggml_matmul(_contig(xd.reshape(-1, self.Kt)), self.packed, self.type_name,
                         self.Kt, self.Nt, eidx=eidx, estride=self.estride, xper=xper)
        return Tensor(of)

    def nbytes(self):
        return int(self.packed.size) * 4


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
    out._setback(_backward)
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
