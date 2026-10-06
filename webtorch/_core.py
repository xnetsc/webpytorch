"""webtorch — a minimal PyTorch-compatible shim with define-by-run autograd,
backed by WgPy's cupy arrays (GPU via WebGPU/WebGL) inside Pyodide.

Phase-2 vertical slice: enough to build and TRAIN a small MLP and verify that
gradients (computed on the GPU) match numerical finite differences. Conv2d /
attention / more ops come later; the autograd core here is what everything else
builds on.
"""
import re
import time
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
        # Inference never traverses an autograd graph. Retaining parents anyway makes a
        # growing KV cache retain every prior cache texture and the forward intermediates
        # that produced each new K/V row. WebGL appends a row every token, so that chain
        # grows for the entire conversation even though no gradient can be requested.
        # Keep the ordered parents only when backward is actually possible.
        self._prev = tuple(_children) if requires_grad else ()
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

    def clamp(self, lo, hi):
        """Bound the values, with the gradient stopped outside the bounds.

        The mask that stops it is built inside the closure, not before it. Built before, it
        costs two comparisons, two casts and a multiply on EVERY call -- and inference never
        reads it. On a 28-layer encoder that was 180 dispatches spent on a gradient nobody
        asked for.
        """
        d = xp.minimum(xp.maximum(self.data, lo), hi)

        def grad(g, o):
            inside = ((self.data > lo).astype(np.float32) * (self.data < hi).astype(np.float32))
            return inside * g
        return self._unary(d, grad, "clamp")

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


_GELU_WGSL = """@group(0) @binding(0) var<storage,read_write> o: array<f32>;
@group(0) @binding(1) var<storage,read> s: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= arrayLength(&s)) { return; }
  let x = s[i];
  let u = clamp((x + 0.044715 * x * x * x) * 0.7978845608028654, -15.0, 15.0);
  o[i] = x * (tanh(u) + 1.0) * 0.5;
}
"""
_geluk = {"added": False}


def _gelu_data(d):
    """One dispatch for the whole activation.

    Composed from primitives it is eleven: three multiplies for the cube, two more and an
    add for the inner term, a min and a max to bound it, the tanh, an add, and two more
    multiplies. On a 28-layer encoder that measured 332 dispatches for 30 calls -- a fifth of
    everything the model issued, for one activation function.

    The bound is inside the kernel for the same reason it is outside: a tanh built from
    exponentials returns NaN once its argument is large enough to overflow, and the cubic
    gets there from x = 11.
    """
    plat = _adam_kernel["platform"]
    if not _geluk["added"]:
        plat.addKernel("gelu_fwd", {"source": _GELU_WGSL,
                                    "bindingTypes": ["storage", "read-only-storage"]})
        _geluk["added"] = True
    dc = _contig(d); out = _empty(dc.shape)
    n = 1
    for sh in dc.shape:
        n *= int(sh)
    plat.runKernel({"name": "gelu_fwd",
        "tensors": [out.buffer.buffer_id, dc.buffer.buffer_id],
        "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return out


_GEGLU_WGSL = """@group(0) @binding(0) var<storage,read_write> o: array<f32>;
@group(0) @binding(1) var<storage,read> s: array<f32>;
struct GMeta { rows: u32, half: u32, }
@group(0) @binding(2) var<storage,read> gm: GMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let n = gm.rows * gm.half;
  if (i >= n) { return; }
  let r = i / gm.half;
  let c = i % gm.half;
  let base = r * gm.half * 2u;
  let x = s[base + c];
  let u = clamp((x + 0.044715 * x * x * x) * 0.7978845608028654, -15.0, 15.0);
  o[i] = x * (tanh(u) + 1.0) * 0.5 * s[base + gm.half + c];
}
"""
_geglu_k = {"added": False}


_QKV_TAKE_WGSL = """@group(0) @binding(0) var<storage,read_write> o: array<f32>;
@group(0) @binding(1) var<storage,read> s: array<f32>;
@group(0) @binding(2) var<storage,read> cosb: array<f32>;
@group(0) @binding(3) var<storage,read> sinb: array<f32>;
struct QMeta { n: u32, T: u32, H: u32, HD: u32, which: u32, rope: u32, }
// `n` counts the whole output, so the number of sequences is implied: out is (B*H, T, HD)
// against a source of (B*T, 3*H*HD), and a sequence's rows sit together in it.
@group(0) @binding(4) var<storage,read> qm: QMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= qm.n) { return; }
  // out is (H, T, HD); src is (T, 3*H*HD) with q, k, v laid end to end on each row.
  let d = i % qm.HD;
  let t = (i / qm.HD) % qm.T;
  let bh = i / (qm.HD * qm.T);          // which (sequence, head) pair this row belongs to
  let b = bh / qm.H;
  let head = bh % qm.H;
  let D = qm.H * qm.HD;
  let base = (b * qm.T + t) * 3u * D + qm.which * D + head * qm.HD;
  let x = s[base + d];
  if (qm.rope == 0u) {
    o[i] = x;
    return;
  }
  let half = qm.HD / 2u;
  var rot: f32;
  if (d < half) {
    rot = -s[base + d + half];
  } else {
    rot = s[base + d - half];
  }
  let ci = t * qm.HD + d;
  o[i] = x * cosb[ci] + rot * sinb[ci];
}
"""
_qkv_k = {"added": False}


_MM_F16W_WGSL = """
@group(0) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> array_b: array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> array_c: array<vec4<f32>>;
struct CMeta { M: u32, N: u32, K: u32, G: u32, }
@group(0) @binding(3) var<storage,read> cmeta: CMeta;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let M = cmeta.M; let N = cmeta.N; let K = cmeta.K; let G = cmeta.G;
  let ND4 = N >> 2u; let ND8 = N >> 3u; let KD4 = K >> 2u;
  let x = gid.x; let y = gid.y; let g = gid.z;
  let row = y * 4u;
  if (x * 8u >= N || row >= M) { return; }
  let per = (KD4 + G - 1u) / G;
  let k0 = g * per;
  var k1 = k0 + per; if (k1 > KD4) { k1 = KD4; }
  let i0 = row;
  let i1 = select(row, row + 1u, row + 1u < M);
  let i2 = select(row, row + 2u, row + 2u < M);
  let i3 = select(row, row + 3u, row + 3u < M);
  var s00 = vec4<f32>(); var s01 = vec4<f32>(); var s02 = vec4<f32>(); var s03 = vec4<f32>();
  var s10 = vec4<f32>(); var s11 = vec4<f32>(); var s12 = vec4<f32>(); var s13 = vec4<f32>();
  for (var k: u32 = k0; k < k1; k = k + 1u) {
    let a0 = array_a[i0 * KD4 + k]; let a1 = array_a[i1 * KD4 + k];
    let a2 = array_a[i2 * KD4 + k]; let a3 = array_a[i3 * KD4 + k];
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let pk = array_b[(k * 4u + j) * ND8 + x];
      let lo = vec4<f32>(unpack2x16float(pk.x), unpack2x16float(pk.y));
      let hi = vec4<f32>(unpack2x16float(pk.z), unpack2x16float(pk.w));
      let av = vec4<f32>(a0[j], a1[j], a2[j], a3[j]);
      s00 = vec4<f32>(av.x) * lo + s00; s01 = vec4<f32>(av.y) * lo + s01;
      s02 = vec4<f32>(av.z) * lo + s02; s03 = vec4<f32>(av.w) * lo + s03;
      s10 = vec4<f32>(av.x) * hi + s10; s11 = vec4<f32>(av.y) * hi + s11;
      s12 = vec4<f32>(av.z) * hi + s12; s13 = vec4<f32>(av.w) * hi + s13;
    }
  }
  let sl = g * M * ND4;
  array_c[sl + x*2u+0u + (row+0u)*ND4] = s00;
  array_c[sl + x*2u+1u + (row+0u)*ND4] = s10;
  if (row+1u < M) { array_c[sl + x*2u+0u+(row+1u)*ND4] = s01; array_c[sl + x*2u+1u+(row+1u)*ND4] = s11; }
  if (row+2u < M) { array_c[sl + x*2u+0u+(row+2u)*ND4] = s02; array_c[sl + x*2u+1u+(row+2u)*ND4] = s12; }
  if (row+3u < M) { array_c[sl + x*2u+0u+(row+3u)*ND4] = s03; array_c[sl + x*2u+1u+(row+3u)*ND4] = s13; }
}
"""

_MM_REDUCE_WGSL = """
@group(0) @binding(0) var<storage,read_write> o: array<f32>;
@group(0) @binding(1) var<storage,read> p: array<f32>;
struct RMeta { n: u32, G: u32, }
@group(0) @binding(2) var<storage,read> rm: RMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= rm.n) { return; }
  var s: f32 = 0.0;
  for (var g: u32 = 0u; g < rm.G; g = g + 1u) { s = s + p[g * rm.n + i]; }
  o[i] = s;
}
"""
_mmf16_k = {"added": False}


def pack_f16_weight(w):
    """A weight matrix as half precision, two values to a word.

    Kept in an f32-typed buffer because that is the only element type this backend allocates;
    the bits are what matter, and the shader binds the same memory as `array<vec4<u32>>` and
    unpacks it. `shader-f16` is not needed for this -- `unpack2x16float` is core WGSL -- so it
    does not depend on a device feature that may not be there.

    Halves what a matmul reads, which on a 421M encoder is 1.23 GB of weights per pass. `w`
    is (K, N) as the maths wants it, N a multiple of 8.
    """
    K, N = int(w.shape[0]), int(w.shape[1])
    if N % 8:
        return None
    h = np.ascontiguousarray(w, dtype=np.float16).reshape(K, N).view(np.uint16)
    u = (h[:, 1::2].astype(np.uint32) << 16) | h[:, 0::2].astype(np.uint32)
    return np.ascontiguousarray(u.view(np.float32))


def _webgl_half_texture_extent(k, n, max_texture_size):
    """Choose an exact, row-aligned texture when both matrix axes fit."""
    size = int(k) * int(n)
    if k <= max_texture_size and n <= max_texture_size:
        return int(n), int(k)
    width = min(max(size, 1), max_texture_size)
    return width, (size + width - 1) // width


def webgl_half_matrix(weight):
    """One immutable dense matrix in native R16F texture storage, when supported.

    The logical array remains float32 for WebGL's shader interface; only its
    physical texture is half width. This is a lower-level execution candidate,
    not a model-name or decision-head route. A failed texture creation/upload
    must propagate rather than silently selecting another storage format.
    """
    if not _webgl_ready() or _adam_backend_ready():
        return None
    from wgpy_backends.webgl.platform import get_platform
    if not get_platform().getDeviceInfo().get("supportsTexture16bit"):
        return None
    from wgpy_backends.webgl.texture import (WebGL2RenderingContext as GL,
                                             WebGLArrayTextureShape, get_max_texture_size)
    from wgpy_backends.webgl.webgl_buffer import WebGLBuffer
    from wgpy_backends.webgl.ndarray import ndarray as GLArray
    value = np.ascontiguousarray(weight, dtype=np.float32)
    if value.ndim != 2:
        return None
    k, n = value.shape
    size = int(value.size)
    max_texture_size = get_max_texture_size()
    # Keep a logical weight row on one texture row whenever it fits.  Matmul
    # advances across K while holding the output column fixed; this layout
    # avoids an unrelated texture-row boundary inside each weight row.
    width, height = _webgl_half_texture_extent(k, n, max_texture_size)
    if height > max_texture_size:
        return None
    texture = WebGLArrayTextureShape(height, width,
                                     internal_format=GL.R16F, format=GL.RED,
                                     type=GL.HALF_FLOAT)
    buffer = WebGLBuffer(size, np.dtype(np.float32), texture)
    buffer.set_data(value)
    return Tensor(GLArray(value.shape, np.float32, buffer=buffer))


def _mm_split_groups(M, N):
    """How many ways to cut the K loop.

    Cutting it raises the number of workgroups, and that only matters when there are too few
    to fill the device. Measured at M=69: N=3072 gives 144 groups and splitting is a wash
    (1.05x at best); N=1024 gives 48 and splitting by four is 1.5x. So it is decided by the
    group count, not by a preference -- and above the threshold it stays at one, where the
    reduction pass is not paid for at all.
    """
    groups = (int(N) // 64) * ((int(M) + 31) // 32)
    return 4 if groups < 128 else 1


class WebGLHalfMatrix(object):
    """A (K, N) weight for `matmul_f16w` on WebGL: half precision, four K-values to a texel.

    The WebGL half of the same contract WebGPU's packed weights keep: built once by
    `pack_half_weight`, consumed only by `matmul_f16w`. Opaque on purpose -- texel (j, k/4)
    holds W[k:k+4, j], which is not row-major element order, so a generic operator that
    read it as an array would read the wrong numbers without a word. Nothing but the matmul
    built for this layout ever sees the texture.

    Why the layout: WebGL's scalar matmul fetches one value per texel on both sides, two
    fetches per multiply-add, and that -- not bandwidth -- is its bound. Four values per
    fetch and a `dot` is a quarter of the fetches for the same bytes. Measured on Apple M5
    (Chrome, ANGLE Metal), 40 queued calls, scalar -> this, including packing the
    activations each call: 519x768x2304 16.0 -> 10.0 ms, 519x768x768 5.26 -> 2.73,
    519x1152x768 7.76 -> 4.17, 8x768x2304 0.35 -> 0.20, 1x768x2304 0.091 -> 0.085. It is
    also closer to a float64 reference (4.3e-6 against 7.7e-6 at the first shape): a
    four-term dot rounds less often than four separate additions.
    """
    __slots__ = ("array", "K", "N", "__weakref__")

    def __init__(self, array, K, N):
        self.array, self.K, self.N = array, int(K), int(N)


def _webgl_half_matrix_ok(K, N):
    if not _webgl_ready() or _adam_backend_ready():
        return False
    from wgpy_backends.webgl.platform import get_platform
    from wgpy_backends.webgl.texture import get_max_texture_size
    info = get_platform().getDeviceInfo()
    mts = get_max_texture_size()
    # The weight is only ever sampled; the activations are packed by rendering into an
    # RGBA32F target, which needs float colour buffers.
    return (K % 4 == 0 and N <= mts and K // 4 <= mts
            and bool(info.get("supportsTexture16bit")) and bool(info.get("supportsTexture32bit")))


def half_weight_ok(K, N):
    """Can a (K, N) weight be held at half width for `matmul_f16w` on this backend?"""
    K, N = int(K), int(N)
    if _adam_backend_ready():
        return N % 64 == 0 and K % 4 == 0 and N % 8 == 0
    return _webgl_half_matrix_ok(K, N)


def _k4_texels(w_out_in):
    """(N, K) -> (K/4, N, 4): texel (j, k4) holds W^T[4k4:4k4+4, j] = W[j, 4k4:4k4+4]. One
    strided copy from the file's layout, the same cost as the transpose it replaces."""
    w = np.asarray(w_out_in, dtype=np.float32)
    N, K = int(w.shape[0]), int(w.shape[1])
    return np.ascontiguousarray(w.reshape(N, K // 4, 4).transpose(1, 0, 2))


def pack_half_weight(w_out_in):
    """The half-width weight `matmul_f16w` consumes, from a checkpoint's (N_out, K_in) layout.

    WebGPU: two halves to a word, (K, N). WebGL: a `WebGLHalfMatrix`. None when this backend
    cannot hold it -- ask `half_weight_ok` first and nothing is ever packed and refused.
    """
    w = np.asarray(w_out_in)
    N, K = int(w.shape[0]), int(w.shape[1])
    if _adam_backend_ready():
        packed = pack_f16_weight(np.ascontiguousarray(np.asarray(w, dtype=np.float32).T))
        return None if packed is None else Tensor(packed)
    if not _webgl_half_matrix_ok(K, N):
        return None
    from wgpy_backends.webgl.texture import (WebGL2RenderingContext as GL,
                                             WebGLArrayTextureShape)
    from wgpy_backends.webgl.webgl_buffer import WebGLBuffer
    from wgpy_backends.webgl.ndarray import ndarray as GLArray
    texels = _k4_texels(w)
    shape = WebGLArrayTextureShape(height=K // 4, width=N, internal_format=GL.RGBA16F,
                                   format=GL.RGBA, type=GL.HALF_FLOAT)
    buffer = WebGLBuffer(K * N, np.dtype(np.float32), shape)
    buffer.set_data(texels)
    return WebGLHalfMatrix(GLArray((K, N), np.float32, buffer=buffer), K, N)


def half_weight(src):
    """`pack_half_weight(src)` when the file already stores `src` at half precision.

    Half width is a STORAGE choice that must not change what the model computes with: a
    weight the checkpoint holds as float16 is computed at float16 either way, but one held as
    float32 would be silently narrowed. So only float16 sources qualify; float32 -- and BF16,
    which reads in as exact float32 -- keep the float32 path. `src` is (N_out, K_in) as
    checkpoints store Linear weights. None when it does not qualify or this backend cannot
    hold it; nothing is packed and then refused.
    """
    a = np.asarray(src)
    if a.ndim != 2 or a.dtype != np.float16:
        return None
    N, K = int(a.shape[0]), int(a.shape[1])
    if not half_weight_ok(K, N):
        return None
    return pack_half_weight(a)


_mmk4 = {"added": set()}


def _webgl_pack_x4(xd, K):
    """Rows of `xd` (M, K) packed four K-values to an RGBA32F texel, R rows side by side per
    texture row: texel (r*K4 + k4, i/R) holds row i = i/R*R + r, k = 4k4..4k4+3.

    A row per texture row would cap a call at 16,384 sequence rows, past which the only copy
    of a weight -- already dropped from the host -- could not be read any other way.
    Returns (packed buffer, R, M)."""
    from wgpy_backends.webgl.texture import (WebGL2RenderingContext as GL,
                                             WebGLArrayTextureShape, get_max_texture_size)
    from wgpy_backends.webgl.webgl_buffer import WebGLBuffer
    plat = _copy_kernel["plat"]
    M = 1
    for d in xd.shape[:-1]:
        M *= int(d)
    K4 = K // 4
    mts = get_max_texture_size()
    R = max(1, mts // K4)
    if (M + R - 1) // R > mts:
        raise ValueError("%d rows of %d exceed one WebGL texture" % (M, K))
    if "pk_x" not in _mmk4["added"]:
        plat.addKernel("mmk4_pack_x", {"source": _K4_GL_HEAD + """uniform sampler2D tex_x;
uniform int K; uniform int K4; uniform int R; uniform int M;
out vec4 fragColor;
float fx(int idx) { int tw = textureSize(tex_x, 0).x; int y = idx / tw;
                    return texelFetch(tex_x, ivec2(idx - y * tw, y), 0).r; }
void main() {
  int x = int(gl_FragCoord.x); int r = x / K4; int k4 = x - r * K4;
  int i = int(gl_FragCoord.y) * R + r;
  if (i >= M) { fragColor = vec4(0.0); return; }
  int b = i * K + k4 * 4;
  fragColor = vec4(fx(b), fx(b + 1), fx(b + 2), fx(b + 3)); }
"""})
        _mmk4["added"].add("pk_x")
    per_row = min(R, M)
    rows = (M + per_row - 1) // per_row
    xshape = WebGLArrayTextureShape(height=rows, width=per_row * K4,
                                    internal_format=GL.RGBA32F, format=GL.RGBA, type=GL.FLOAT)
    xp_buf = WebGLBuffer(rows * per_row * K, np.dtype(np.float32), xshape)
    plat.runKernel({"name": "mmk4_pack_x",
                    "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id}],
                    "output": xp_buf.buffer_id,
                    "uniforms": [{"name": "K", "value": K, "type": "int"},
                                 {"name": "K4", "value": K4, "type": "int"},
                                 {"name": "R", "value": per_row, "type": "int"},
                                 {"name": "M", "value": M, "type": "int"}]})
    return xp_buf, per_row, M


_K4_GL_HEAD = ("#version 300 es\nprecision highp float; precision highp int; "
            "precision highp sampler2D; precision highp usampler2D;\n")


def _webgl_matmul_k4(x, w):
    """`x @ w` for a `WebGLHalfMatrix` w: pack x's rows four K-values to a texel, then dot."""
    plat = _copy_kernel["plat"]
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    K, N = w.K, w.N
    if int(xd.shape[-1]) != K:
        raise ValueError("matmul_f16w: x has %d columns, the weight %d rows"
                         % (int(xd.shape[-1]), K))
    K4 = K // 4
    xp_buf, per_row, M = _webgl_pack_x4(xd, K)
    name = "mmk4_%d" % K
    if name not in _mmk4["added"]:
        # K is the loop bound, so it is compiled in; M and N are uniforms, so a new sequence
        # length does not compile a new program.
        plat.addKernel(name, {"source": _K4_GL_HEAD + """#define K4 %d
uniform int _ka_tex_output_texture_w; uniform int M; uniform int N; uniform int R;
uniform sampler2D tex_xp; uniform sampler2D tex_wp;
out float fragColor;
void main() {
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int i = idx / N; int j = idx - i * N;
  if (i >= M) { return; }
  int ry = i / R; int bx = (i - ry * R) * K4;
  float s = 0.0;
  int k4 = 0;
  for (; k4 + 4 <= K4; k4 += 4) {
    s += dot(texelFetch(tex_xp, ivec2(bx + k4, ry), 0), texelFetch(tex_wp, ivec2(j, k4), 0));
    s += dot(texelFetch(tex_xp, ivec2(bx + k4 + 1, ry), 0), texelFetch(tex_wp, ivec2(j, k4 + 1), 0));
    s += dot(texelFetch(tex_xp, ivec2(bx + k4 + 2, ry), 0), texelFetch(tex_wp, ivec2(j, k4 + 2), 0));
    s += dot(texelFetch(tex_xp, ivec2(bx + k4 + 3, ry), 0), texelFetch(tex_wp, ivec2(j, k4 + 3), 0));
  }
  for (; k4 < K4; k4++) {
    s += dot(texelFetch(tex_xp, ivec2(bx + k4, ry), 0), texelFetch(tex_wp, ivec2(j, k4), 0));
  }
  fragColor = s;
}
""" % K4})
        _mmk4["added"].add(name)
    out = _empty((M, N))
    plat.runKernel({"name": name,
                    "inputs": [{"name": "tex_xp", "id": xp_buf.buffer_id},
                               {"name": "tex_wp", "id": w.array.buffer.buffer_id}],
                    "output": out.buffer.buffer_id,
                    "uniforms": [{"name": "_ka_tex_output_texture_w",
                                  "value": out.buffer.texture_shape.width, "type": "int"},
                                 {"name": "M", "value": M, "type": "int"},
                                 {"name": "N", "value": N, "type": "int"},
                                 {"name": "R", "value": per_row, "type": "int"}]})
    lead = tuple(int(d) for d in xd.shape[:-1])
    return Tensor(out.reshape(*(lead + (N,))))


class WebGLQ8Matrix(object):
    """A Q8_0 weight (N_out, K_in) for WebGL, its blocks' two parts in two textures.

    The GGUF block is a half scale followed by 32 int8. Laid out as the stored kernel reads
    it -- 34-byte blocks transposed into 32-bit words -- every group of four int8 straddles
    two words half the time and the activations are fetched one value at a time; the Q8
    xDecision encoder spent 1980 of its 2264 ms there. Here the int8 go to an RGBA8UI
    texture, four K-values to a texel exactly as `WebGLHalfMatrix` holds halves (texel
    (j, k/4) = W[j, k:k+4], the bytes unchanged), and the scales to an R16F texture
    (texel (j, b) = d of block b of row j, the halves unchanged): the same bytes as the
    file, no value converted, one fetch per four multiplies. Opaque: only
    `_webgl_matmul_q8k4` reads it.
    """
    __slots__ = ("q", "d", "K", "N", "__weakref__")

    def __init__(self, q, d, K, N):
        self.q, self.d, self.K, self.N = q, d, int(K), int(N)


def _webgl_q8_ok(type_name, K, N):
    """Whether this WebGL device can hold a (K, N) weight as a `WebGLQ8Matrix`."""
    if type_name != "Q8_0" or not _webgl_ready() or _adam_backend_ready():
        return False
    K, N = int(K), int(N)
    if K % 32:
        return False
    from wgpy_backends.webgl.platform import get_platform
    from wgpy_backends.webgl.texture import get_max_texture_size
    info = get_platform().getDeviceInfo()
    mts = get_max_texture_size()
    return (N <= mts and K // 4 <= mts and bool(info.get("supportsTexture16bit"))
            and bool(info.get("supportsTexture32bit")))


def _q8_split(raw, K, N):
    """Q8_0 blocks (N rows of K/32 blocks, 34 bytes each) as the two textures' contents:
    the int8 bytes as (K/4, N, 4) uint8 -- texel (j, k4) = bytes of W[j, 4k4:4k4+4] -- and
    the scales as (K/32, N) halves. Bytes and halves unchanged."""
    K, N = int(K), int(N)
    nb = K // 32
    blocks = np.frombuffer(raw, np.uint8, count=N * nb * 34).reshape(N, nb, 34)
    q = np.ascontiguousarray(blocks[:, :, 2:].reshape(N, K // 4, 4).transpose(1, 0, 2))
    d = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(N, nb)
    return q, np.ascontiguousarray(d.T)


def _webgl_q8_pack(raw, K, N):
    """Split Q8_0 blocks into a `WebGLQ8Matrix` (see `_q8_split`)."""
    from wgpy_backends.webgl.texture import (WebGL2RenderingContext as GL,
                                             WebGLArrayTextureShape)
    from wgpy_backends.webgl.webgl_buffer import WebGLBuffer
    from wgpy_backends.webgl.ndarray import ndarray as GLArray
    K, N = int(K), int(N)
    nb = K // 32
    q, d = _q8_split(raw, K, N)
    # Widened only for the upload call (exact); it writes the same halves back.
    d = d.astype(np.float32)
    qbuf = WebGLBuffer(K * N, np.dtype(np.uint8),
                       WebGLArrayTextureShape(height=K // 4, width=N, internal_format=GL.RGBA8UI,
                                              format=GL.RGBA_INTEGER, type=GL.UNSIGNED_BYTE))
    qbuf.set_data(q)
    dbuf = WebGLBuffer(nb * N, np.dtype(np.float32),
                       WebGLArrayTextureShape(height=nb, width=N, internal_format=GL.R16F,
                                              format=GL.RED, type=GL.HALF_FLOAT))
    dbuf.set_data(d)
    return WebGLQ8Matrix(GLArray((K // 4 * 4, N), np.uint8, buffer=qbuf),
                         GLArray((nb, N), np.float32, buffer=dbuf), K, N)


def _webgl_matmul_q8k4(xf, w):
    """xf(M,K) @ W.T for a `WebGLQ8Matrix`: per block of 32, eight four-wide dots of the
    packed activations with the int8, then the block's scale -- d * (sum of q*x), f32,
    the same arithmetic as WebGPU's tiled kernel."""
    plat = _copy_kernel["plat"]
    xd = _contig(xf.data if isinstance(xf, Tensor) else xf)
    K, N = w.K, w.N
    if int(xd.shape[-1]) != K:
        raise ValueError("Q8_0 matmul: x has %d columns, the weight %d" % (int(xd.shape[-1]), K))
    xp_buf, per_row, M = _webgl_pack_x4(xd, K)
    name = "mmq8k4_%d" % K
    if name not in _mmk4["added"]:
        plat.addKernel(name, {"source": _K4_GL_HEAD + """#define K4 %d
#define NB %d
uniform int _ka_tex_output_texture_w; uniform int M; uniform int N; uniform int R;
uniform sampler2D tex_xp; uniform usampler2D tex_q; uniform sampler2D tex_d;
out float fragColor;
// The texel's four bytes are int8: u - 256 where u >= 128, exactly.
vec4 Q(int j, int k4) {
  vec4 u = vec4(texelFetch(tex_q, ivec2(j, k4), 0));
  return u - 256.0 * step(128.0, u);
}
void main() {
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int i = idx / N; int j = idx - i * N;
  if (i >= M) { return; }
  int ry = i / R; int bx = (i - ry * R) * K4;
  float s = 0.0;
  for (int b = 0; b < NB; b++) {
    int k4 = b * 8;
    float p = dot(texelFetch(tex_xp, ivec2(bx + k4, ry), 0), Q(j, k4));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 1, ry), 0), Q(j, k4 + 1));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 2, ry), 0), Q(j, k4 + 2));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 3, ry), 0), Q(j, k4 + 3));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 4, ry), 0), Q(j, k4 + 4));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 5, ry), 0), Q(j, k4 + 5));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 6, ry), 0), Q(j, k4 + 6));
    p += dot(texelFetch(tex_xp, ivec2(bx + k4 + 7, ry), 0), Q(j, k4 + 7));
    s += p * texelFetch(tex_d, ivec2(j, b), 0).r;
  }
  fragColor = s;
}
""" % (K // 4, K // 32)})
        _mmk4["added"].add(name)
    out = _empty((M, N))
    plat.runKernel({"name": name,
                    "inputs": [{"name": "tex_xp", "id": xp_buf.buffer_id},
                               {"name": "tex_q", "id": w.q.buffer.buffer_id},
                               {"name": "tex_d", "id": w.d.buffer.buffer_id}],
                    "output": out.buffer.buffer_id,
                    "uniforms": [{"name": "_ka_tex_output_texture_w",
                                  "value": out.buffer.texture_shape.width, "type": "int"},
                                 {"name": "M", "value": M, "type": "int"},
                                 {"name": "N", "value": N, "type": "int"},
                                 {"name": "R", "value": per_row, "type": "int"}]})
    return out


def gpu_features():
    """What the WebGPU device was created with, detected on the device at start: optional
    features (`f16`, `subgroups`), subgroup sizes and the limits kernels depend on. {} without
    WebGPU. Kernels that need a feature are offered only where it is present; which of the
    eligible ones runs is measured on the device, whatever its vendor."""
    if not _adam_backend_ready():
        return {}
    if "v" not in _GPU_FEATURES:
        try:
            info = _adam_kernel["platform"].getDeviceInfo()
            _GPU_FEATURES["v"] = dict(info) if isinstance(info, dict) else {}
        except Exception:
            _GPU_FEATURES["v"] = {}
    return _GPU_FEATURES["v"]


_GPU_FEATURES = {}


def _mm_half_src(R=4, C=8, BK=32, FLUSH=32):
    """`x @ w` for packed half weights with the products in half precision.

    Where the device has `shader-f16`, a half FMA costs half an f32 one -- 6.2 against 3.2
    TFLOPS measured on an Apple M5, and on several other GPUs the rate is likewise doubled --
    but only when the running sum is half too. So each thread sums FLUSH products in half and
    adds that partial into f32: products and short sums at the weights' own width, the long
    sum at f32. The error grows with FLUSH (as its square root): 32 measured as fast as 64
    and 16 cost 8-10%, so 32 -- about 1e-3 of the output scale, against ~1e-6 for `mm_f16w`.

    8 x 8 threads, a 32-row x 64-column tile; each BK-deep stage stages the activations
    (converted to half once, here) and the weights through workgroup memory. Thread (tx, ty)
    owns rows ty + 8r and columns tx*C.. . Every workgroup load of a 4-deep step is issued
    before its FMAs, which on the M5 took the kernel from 0.52 to 0.47 ms at 519x768x2304.
    """
    BM, BN = 8 * R, 8 * C
    AST = BK // 4 + 1
    WST = BN // 8
    nv, G = C // 4, C // 8
    pa, pw = BM * (BK // 4) // 64, BK * WST // 64
    L = []
    a = L.append
    a("enable f16;")
    a("@group(0) @binding(0) var<storage,read> A: array<vec4<f32>>;")
    a("@group(0) @binding(1) var<storage,read> W: array<vec4<u32>>;")
    a("@group(0) @binding(2) var<storage,read_write> O: array<vec4<f32>>;")
    a("struct CMeta { M: u32, N: u32, K: u32, G: u32, }")
    a("@group(0) @binding(3) var<storage,read> cm: CMeta;")
    a("var<workgroup> as_: array<vec4<f16>, %d>;" % (BM * AST))
    a("var<workgroup> ws: array<vec4<u32>, %d>;" % (BK * WST))
    a("@compute @workgroup_size(8, 8, 1)")
    a("fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,")
    a("        @builtin(local_invocation_index) li: u32) {")
    a("  let M = cm.M; let N = cm.N; let K = cm.K;")
    a("  let K4 = K / 4u; let N8 = N / 8u; let N4 = N / 4u;")
    a("  let m0 = wg.y * %du; let n0 = wg.x * %du;" % (BM, BN))
    a("  let tx = lid.x; let ty = lid.y;")
    for r in range(R):
        for v in range(nv):
            a("  var S%d_%d = vec4<f32>(); var h%d_%d = vec4<f16>();" % (r, v, r, v))
    for i in range(pa):
        a("  let ar%d = (li + %du) / %du; let ac%d = (li + %du) %% %du;"
          % (i, 64 * i, BK // 4, i, 64 * i, BK // 4))
        a("  let ag%d = min(m0 + ar%d, M - 1u) * K4 + ac%d;" % (i, i, i))
    for i in range(pw):
        a("  let wr%d = (li + %du) / %du; let wc%d = (li + %du) %% %du;"
          % (i, 64 * i, WST, i, 64 * i, WST))
    a("  for (var k0 = 0u; k0 < K; k0 = k0 + %du) {" % BK)
    for i in range(pa):
        a("    as_[ar%d * %du + ac%d] = vec4<f16>(A[ag%d + k0 / 4u]);" % (i, AST, i, i))
    for i in range(pw):
        a("    ws[wr%d * %du + wc%d] = W[(k0 + wr%d) * N8 + n0 / 8u + wc%d];" % (i, WST, i, i, i))
    a("    workgroupBarrier();")
    a("    for (var kk = 0u; kk < %du; kk = kk + 1u) {" % (BK // 4))
    for r in range(R):
        a("      let a%d = as_[(ty + %du) * %du + kk];" % (r, 8 * r, AST))
    for j in range(4):
        for g in range(G):
            a("      let p%d_%d = ws[(kk * 4u + %du) * %du + tx * %du + %du];" % (j, g, j, WST, G, g))
    for j in range(4):
        for g in range(G):
            a("      let w%d_%d = vec4<f16>(bitcast<vec2<f16>>(p%d_%d.x), bitcast<vec2<f16>>(p%d_%d.y));"
              % (j, 2 * g, j, g, j, g))
            a("      let w%d_%d = vec4<f16>(bitcast<vec2<f16>>(p%d_%d.z), bitcast<vec2<f16>>(p%d_%d.w));"
              % (j, 2 * g + 1, j, g, j, g))
        for r in range(R):
            for v in range(nv):
                a("      h%d_%d = fma(vec4<f16>(a%d[%d]), w%d_%d, h%d_%d);" % (r, v, r, j, j, v, r, v))
    if FLUSH < BK:
        a("      if (((kk + 1u) %% %du) == 0u) {" % (FLUSH // 4))
        for r in range(R):
            for v in range(nv):
                a("        S%d_%d = S%d_%d + vec4<f32>(h%d_%d); h%d_%d = vec4<f16>();"
                  % (r, v, r, v, r, v, r, v))
        a("      }")
    a("    }")
    a("    workgroupBarrier();")
    if FLUSH >= BK:
        a("    if (((k0 + %du) %% %du) == 0u || k0 + %du >= K) {" % (BK, FLUSH, BK))
        for r in range(R):
            for v in range(nv):
                a("      S%d_%d = S%d_%d + vec4<f32>(h%d_%d); h%d_%d = vec4<f16>();"
                  % (r, v, r, v, r, v, r, v))
        a("    }")
    a("  }")
    for r in range(R):
        a("  { let gm = m0 + ty + %du;" % (8 * r))
        a("    if (gm < M) {")
        for v in range(nv):
            a("      O[gm * N4 + n0 / 4u + tx * %du + %du] = S%d_%d;" % (nv, v, r, v))
        a("    } }")
    a("}")
    return "\n".join(L)


_mm_half_added = {"v": False}


def _mm_half(xd, wd, M, K, N):
    plat = _adam_kernel["platform"]
    if not _mm_half_added["v"]:
        plat.addKernel("mm_half", {"source": _mm_half_src(),
                                   "bindingTypes": ["read-only-storage", "read-only-storage",
                                                    "storage", "read-only-storage"]})
        _mm_half_added["v"] = True
    out = _empty((M, N))
    meta = _adam_kernel["make_meta"]((M, N, K, 1), "u4,u4,u4,u4")
    plat.runKernel({"name": "mm_half",
                    "tensors": [xd.buffer.buffer_id, wd.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": N // 64, "y": (M + 31) // 32, "z": 1}})
    return out


def matmul_f16w(x, wpacked, K, N, execution="auto"):
    """`x @ w` with w held as packed half precision (`pack_half_weight`). None without a GPU
    backend, or where this backend cannot take the shape.

    Two executions where the device has `shader-f16`: "f32" (`mm_f16w`, products and sums in
    f32) and "f16" (`_mm_half_src`); the faster is measured per shape bucket on the device,
    after an output check against "f32". Elsewhere "f32" is the only one."""
    if isinstance(wpacked, WebGLHalfMatrix):
        return _webgl_matmul_k4(x, wpacked)
    if not _adam_backend_ready():
        return None
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    M = 1
    for d in xd.shape[:-1]:
        M *= int(d)
    K, N = int(K), int(N)
    if int(xd.shape[-1]) != K or N % 64 or K % 4:
        return None
    wd = wpacked.data if isinstance(wpacked, Tensor) else wpacked
    lead = tuple(xd.shape[:-1])
    if execution == "auto" and K % 32 == 0 and gpu_features().get("f16"):
        reference = [None]

        def run(which):
            return matmul_f16w(xd, wd, K, N, execution=which).data

        def correct(which):
            if which == "f32":
                return True
            if reference[0] is None:
                reference[0] = np.asarray(run("f32").get(), np.float32)
            got = np.asarray(run(which).get(), np.float32)
            if not np.all(np.isfinite(got)):
                return False
            scale = max(1e-6, float(np.abs(reference[0]).max()))
            return float(np.abs(got - reference[0]).max()) / scale < 1e-2

        execution = _weight_execution("dense_half", "f16", K, N, M, run,
                                      candidates=("f32", "f16"), check=correct)
    if execution == "f16":
        return Tensor(_mm_half(xd, wd, M, K, N).reshape(*(lead + (N,))))
    plat = _adam_kernel["platform"]
    if not _mmf16_k["added"]:
        plat.addKernel("mm_f16w", {"source": _MM_F16W_WGSL,
                                   "bindingTypes": ["read-only-storage", "read-only-storage",
                                                    "storage", "read-only-storage"]})
        plat.addKernel("mm_f16w_reduce", {"source": _MM_REDUCE_WGSL,
                                          "bindingTypes": ["storage", "read-only-storage",
                                                           "read-only-storage"]})
        _mmf16_k["added"] = True
    G = _mm_split_groups(M, N)
    part = _empty((G * M, N))
    meta = _adam_kernel["make_meta"]((M, N, K, G), "u4,u4,u4,u4")
    plat.runKernel({"name": "mm_f16w",
                    "tensors": [xd.buffer.buffer_id, wd.buffer.buffer_id,
                                part.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": N // 64, "y": (M + 31) // 32, "z": G}})
    if G == 1:
        return Tensor(part.reshape(*(lead + (N,))))
    out = _empty((M, N))
    rmeta = _adam_kernel["make_meta"]((M * N, G), "u4,u4")
    plat.runKernel({"name": "mm_f16w_reduce",
                    "tensors": [out.buffer.buffer_id, part.buffer.buffer_id, rmeta.buffer_id],
                    "workGroups": {"x": (M * N + 63) // 64, "y": 1, "z": 1}})
    return Tensor(out.reshape(*(lead + (N,))))


def qkv_take(qkv, which, H, HD, T, cos=None, sin=None, B=1):
    """One of q, k or v, taken out of a fused projection and laid out for attention.

    `qkv` is (T, 3*H*HD) as the projection produced it; the result is (H, T, HD) with rotary
    applied when `cos`/`sin` are given. Written as expressions this is a slice, a transpose
    and a rotation -- and the first two are strided COPIES of the whole tensor, because the
    backend has no view of a slice of a row. Three of them per layer measured as costly as
    every multiply in the pass put together.

    Returns None without a fused backend, so the caller keeps its expression. No gradient.
    """
    gpu = _adam_backend_ready() or _webgl_ready()
    xd = _contig(qkv.data if isinstance(qkv, Tensor) else qkv)
    H, HD, T, B = int(H), int(HD), int(T), int(B)
    if tuple(xd.shape) != (B * T, 3 * H * HD):
        return None
    use_rope = cos is not None and sin is not None
    cd = _contig(cos.data if isinstance(cos, Tensor) else cos) if use_rope else xd
    sd = _contig(sin.data if isinstance(sin, Tensor) else sin) if use_rope else xd
    n = B * H * T * HD
    if not gpu:
        # NumPy can view the three projections without copying. Build only the requested
        # head-major output; the old expression materialized each strided slice, its
        # transpose, and several more full arrays for rotary embedding on every layer.
        if not isinstance(xd, np.ndarray) or int(which) not in (0, 1, 2):
            return None
        source = xd.reshape(B, T, 3, H, HD)[:, :, int(which)]
        source = source.transpose(0, 2, 1, 3).reshape(B * H, T, HD)
        if not use_rope:
            return Tensor(np.ascontiguousarray(source))
        if HD % 2 or tuple(cd.shape) != (T, HD) or tuple(sd.shape) != (T, HD):
            return None
        half = HD // 2
        out = np.empty((B * H, T, HD), dtype=source.dtype)
        out[..., :half] = source[..., :half] * cd[:, :half] - source[..., half:] * sd[:, :half]
        out[..., half:] = source[..., half:] * cd[:, half:] + source[..., :half] * sd[:, half:]
        return Tensor(out)
    if _webgl_ready() and not _adam_backend_ready():
        return Tensor(_webgl_qkv_take(xd, cd, sd, n, T, H, HD, which, use_rope)
                      .reshape(B * H, T, HD))
    plat = _adam_kernel["platform"]
    if not _qkv_k["added"]:
        plat.addKernel("qkv_take", {"source": _QKV_TAKE_WGSL,
                                    "bindingTypes": ["storage"] + ["read-only-storage"] * 4})
        _qkv_k["added"] = True
    out = _empty((n,))
    meta = _adam_kernel["make_meta"]((n, T, H, HD, int(which), 1 if use_rope else 0),
                                     "u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "qkv_take",
                    "tensors": [out.buffer.buffer_id, xd.buffer.buffer_id,
                                cd.buffer.buffer_id, sd.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return Tensor(out.reshape(B * H, T, HD))


def geglu_split(x, half):
    """A gated MLP's activation, reading the two halves where they already are.

    `Wi` produces one tensor of width `2 * half`; the transformed half and the gating half
    are slices of it. Written as slices, each one is a strided COPY -- the backend has no
    view of a half-row -- so the activation costs two full copies of the tensor before any
    arithmetic happens. On a 28-layer encoder those two copies measured as much time as the
    three QKV copies put together.

    This reads both halves out of the one tensor and writes the result, so nothing is copied
    and the activation is a single dispatch. Returns None where there is no fused backend,
    so the caller keeps its expression.
    """
    gpu = _adam_backend_ready() or _webgl_ready()
    xd = _contig(x.data if isinstance(x, Tensor) else x)
    rows = 1
    for d in xd.shape[:-1]:
        rows *= int(d)
    width = int(xd.shape[-1])
    if width != 2 * int(half):
        return None
    if not gpu:
        if not isinstance(xd, np.ndarray):
            return None
        half = int(half)
        flat = xd.reshape(rows, width)
        gate = flat[:, :half]
        up = flat[:, half:]
        # Keep the same bounded tanh GELU as the GPU kernel and composed fallback.
        inner = np.clip((gate + np.float32(0.044715) * gate * gate * gate)
                        * np.float32(0.7978845608028654), -15.0, 15.0)
        out = gate * (np.tanh(inner) + np.float32(1.0)) * np.float32(0.5) * up
        return Tensor(out.reshape(*(xd.shape[:-1] + (half,))))
    if _webgl_ready() and not _adam_backend_ready():
        return Tensor(_webgl_geglu_split(xd, rows, int(half)).reshape(*(xd.shape[:-1] + (int(half),))))
    plat = _adam_kernel["platform"]
    if not _geglu_k["added"]:
        plat.addKernel("geglu", {"source": _GEGLU_WGSL,
                                 "bindingTypes": ["storage", "read-only-storage",
                                                  "read-only-storage"]})
        _geglu_k["added"] = True
    out = _empty((rows, int(half)))
    meta = _adam_kernel["make_meta"]((rows, int(half)), "u4,u4")
    n = rows * int(half)
    plat.runKernel({"name": "geglu",
                    "tensors": [out.buffer.buffer_id, xd.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    shape = tuple(xd.shape[:-1]) + (int(half),)
    return Tensor(out.reshape(*shape))


def gelu(x):
    if _adam_backend_ready():
        out = Tensor(_gelu_data(x.data), x.requires_grad, (x,), "gelu")

        def _backward():
            # Only built when something is training. The derivative of the same tanh form:
            # 0.5(1+t) + 0.5*x*(1-t^2)*du/dx, with u bounded exactly as the forward bounds it
            # so the two agree at the ends.
            if x.requires_grad:
                xd = x.data
                a, c = 0.044715, 0.7978845608028654
                u = xp.minimum(xp.maximum((xd + a * xd * xd * xd) * c, -15.0), 15.0)
                t = xp.tanh(u)
                du = c * (1.0 + 3.0 * a * xd * xd)
                x._accum(out.grad * (0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t * t) * du))
        out._setback(_backward)
        return out
    return _gelu_composed(x)


def _gelu_composed(x):
    # tanh approximation (as used in GPT). Composed from autograd primitives, so
    # it works on every backend; WebGPU takes the fused kernel above instead.
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * c
    # The argument is bounded before the tanh, and that is load-bearing rather than tidy.
    # `tanh` saturates long before fp32 runs out of range -- it is 1.0 for any argument past
    # about 9.1 -- but a backend that computes it from exponentials overflows first. On
    # WebGPU `tanh` returns NaN once its argument passes roughly 40, and the cubic above
    # reaches that from x = 11, so `gelu(11)` came back NaN where the answer is 11.0. It
    # turned one hidden row into NaN halfway through a 28-layer encoder, and nothing before
    # that point looked wrong. Measured, both backends, x = 1 .. 50.
    #
    # Clamping costs nothing in accuracy: every value it touches would have produced exactly
    # +/-1 anyway. It was never noticed because the models this repo runs use SwiGLU, so the
    # GPU path through here had not been exercised.
    return x * (inner.clamp(-15.0, 15.0).tanh() + 1.0) * 0.5


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
struct RM { ne: u32, k: u32, norm: u32, rows: u32, }
@group(0) @binding(3)
var<storage,read> rm: RM;
var<workgroup> v: array<f32, 512>;
var<workgroup> red: array<f32, 128>;
var<workgroup> ridx: array<u32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  if (row >= rm.rows) { return; }
  let base = row * rm.ne;
  let out_base = row * rm.k;
  let t = lid.x;
  // stage the router's scores; anything past ne is -inf so it never wins a pass
  for (var e: u32 = t; e < 512u; e = e + 128u) {
    v[e] = select(-1e30, lg[base + e], e < rm.ne);
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
      eidx[out_base + s] = i32(ridx[0]);
      ew[out_base + s] = red[0];
      v[ridx[0]] = -1e30;
    }
    workgroupBarrier();
  }
  // For normalised top-k, the full-expert softmax denominator cancels exactly:
  // (exp(s_i)/sum_all exp) / sum_selected(exp(s_j)/sum_all exp)
  // = exp(s_i)/sum_selected exp. Only the unnormalised mode needs all experts.
  if (t == 0u) {
    if (rm.norm == 1u) {
      var picked_max: f32 = -1e30;
      for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
        picked_max = max(picked_max, ew[out_base + s]);
      }
      var picked_den: f32 = 0.0;
      for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
        picked_den = picked_den + exp(ew[out_base + s] - picked_max);
      }
      if (picked_den > 0.0) {
        for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
          ew[out_base + s] = exp(ew[out_base + s] - picked_max) / picked_den;
        }
      }
      return;
    }
    var mx: f32 = -1e30;
    for (var e: u32 = 0u; e < rm.ne; e = e + 1u) { mx = max(mx, lg[base + e]); }
    var den: f32 = 0.0;
    for (var e: u32 = 0u; e < rm.ne; e = e + 1u) { den = den + exp(lg[base + e] - mx); }
    for (var s: u32 = 0u; s < rm.k; s = s + 1u) {
      ew[out_base + s] = exp(ew[out_base + s] - mx) / den;
    }
  }
}
"""
_moe_r = {"added": False}


def moe_route(logits, eidx, ew, ne, k, norm=True):
    """Router scores -> chosen experts and weights for one or many rows on-device.

    Doing this on the host means reading the router's output back once per MoE layer, which
    at 48 layers is most of a decode step -- and it is a tiny amount of data, so the cost is
    all round-trip. Keeping it here also keeps the step capturable: the indices land in a
    buffer the expert matmuls already read, and no command depends on their value."""
    ne = int(ne); k = int(k)
    shape = tuple(int(v) for v in logits.shape)
    if len(shape) == 1:
        rows, width = 1, shape[0]
    elif len(shape) == 2:
        rows, width = shape
    else:
        raise ValueError("MoE router logits must be a vector or matrix")
    if not (rows >= 1 and 1 <= ne <= 512 and 1 <= k <= ne and width == ne):
        raise ValueError("MoE router requires rows >= 1, 1 <= k <= ne <= 512, "
                         "and logits width == ne")
    if int(eidx.size) < rows * k or int(ew.size) < rows * k:
        raise ValueError("MoE router output buffers are too small")
    if _webgl_ready() and not _adam_backend_ready():
        return _webgl_moe_route(logits, eidx, ew, ne, k, norm)
    plat = _adam_kernel["platform"]
    if not _moe_r["added"]:
        plat.addKernel("moe_route", {"source": _MOE_ROUTE_WGSL,
                                     "bindingTypes": ["read-only-storage", "storage",
                                                      "storage", "read-only-storage"]})
        _moe_r["added"] = True
    meta = _adam_kernel["make_meta"]((ne, k, 1 if norm else 0, rows), "u4,u4,u4,u4")
    plat.runKernel({"name": "moe_route",
                    "tensors": [logits.buffer.buffer_id, eidx.buffer.buffer_id,
                                ew.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": rows, "y": 1, "z": 1}})


_MOE_REDUCE_WGSL = """@group(0) @binding(0) var<storage,read> y: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
struct RR { rows: u32, k: u32, h: u32, pad: u32, }
@group(0) @binding(3) var<storage,read> rr: RR;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= rr.rows * rr.h) { return; }
  let row = i / rr.h;
  let col = i - row * rr.h;
  var acc: f32 = 0.0;
  for (var slot: u32 = 0u; slot < rr.k; slot = slot + 1u) {
    let p = row * rr.k + slot;
    acc = acc + y[p * rr.h + col] * w[p];
  }
  outp[i] = acc;
}
"""
_moe_reduce_added = {"webgpu": False}
_MOE_REDUCE_REJECTED = {}


def moe_weighted_sum(values, weights, k, execution="auto"):
    """Sum routed expert rows into token rows, on either GPU backend.

    ``auto`` is this operator's measured shape/device route. A containing MoE layer may
    explicitly choose a different route when its whole-layer timing favours it. Until a
    route is measured, the established two-operation path is the lower-risk baseline.
    """
    if execution not in ("auto", "composed", "fused"):
        raise ValueError("MoE reduction execution must be auto, composed, or fused")
    yd = values.data if isinstance(values, Tensor) else values
    wd = weights.data if isinstance(weights, Tensor) else weights
    k = int(k)
    if len(yd.shape) != 2 or k < 1 or int(yd.shape[0]) % k:
        raise ValueError("MoE values must have a multiple of k rows")
    rows, h = int(yd.shape[0]) // k, int(yd.shape[1])
    if int(wd.size) < rows * k:
        raise ValueError("MoE weights buffer is too small")
    backend = ("webgl" if _webgl_ready() and not _adam_backend_ready() else
               "webgpu" if _adam_backend_ready() else "cpu")
    bucket = 1 << (rows - 1).bit_length()
    key = ("moe_reduce", backend, k, h, bucket)
    def composed():
        yy = values if isinstance(values, Tensor) else Tensor(yd)
        ww = weights if isinstance(weights, Tensor) else Tensor(wd)
        return (yy * ww.reshape(rows * k, 1)).reshape(rows, k, h).sum(axis=1)

    def fused():
        if backend == "webgl":
            return _webgl_moe_weighted_sum(yd, wd, rows, k, h)
        plat = _adam_kernel["platform"]
        if not _moe_reduce_added["webgpu"]:
            plat.addKernel("moe_weighted_sum", {"source": _MOE_REDUCE_WGSL,
                           "bindingTypes": ["read-only-storage", "read-only-storage",
                                            "storage", "read-only-storage"]})
            _moe_reduce_added["webgpu"] = True
        out = _empty((rows, h))
        meta = _adam_kernel["make_meta"]((rows, k, h, 0), "u4,u4,u4,u4")
        plat.runKernel({"name": "moe_weighted_sum",
                        "tensors": [yd.buffer.buffer_id, wd.buffer.buffer_id,
                                    out.buffer.buffer_id, meta.buffer_id],
                        "workGroups": {"x": (rows * h + 127) // 128, "y": 1, "z": 1}})
        return Tensor(out)

    if backend == "cpu":
        return composed()
    mode = execution
    if mode == "auto":
        mode = _TUNED.get(key)
        if mode is None:
            # Calibrating inside a graph recording would bake the comparison's many
            # dispatches into EVERY future decode token. Defer just this unknown bucket;
            # the independent load-time one-row warm pass normally populates it first.
            if backend == "webgpu":
                from wgpy_backends.webgpu import webgpu_buffer as _wb
            else:
                from wgpy_backends.webgl import webgl_buffer as _wb
            if getattr(_wb, "_capture_depth", 0):
                mode = "composed"
            else:
                import time as _time
                try:
                    reference = np.asarray(composed().numpy(), np.float32)
                    candidate = np.asarray(fused().numpy(), np.float32)
                    scale = max(1e-6, float(np.abs(reference).max()))
                    if (not np.all(np.isfinite(candidate))
                            or float(np.abs(reference - candidate).max()) / scale > 2e-5):
                        raise RuntimeError("fused MoE reduction failed its numerical gate")
                    # A single tiny dispatch plus readback measures synchronisation
                    # jitter more than kernel work. Batch enough identical calls for
                    # the per-operator positive result to survive that noise, without
                    # keeping more than one output alive at a time.
                    repeat = 16 if rows <= 128 else 4
                    samples = {"composed": [], "fused": []}
                    for turn in range(9):
                        order = (("composed", "fused") if not (turn & 1)
                                 else ("fused", "composed"))
                        for route in order:
                            t0 = _time.perf_counter()
                            for _ in range(repeat):
                                out = composed() if route == "composed" else fused()
                            out.numpy()
                            samples[route].append((_time.perf_counter() - t0) / repeat)
                    mode = _measured_choice(samples, ("composed", "fused"),
                                            default="composed")
                except Exception as exc:
                    _MOE_REDUCE_REJECTED[key] = "%s: %s" % (type(exc).__name__, exc)
                    mode = "composed"
                _TUNED[key] = mode
    return fused() if mode == "fused" else composed()


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
def _bmm_raw(A, B):
    """Batched matmul on raw xp arrays: (Bs,M,K)@(Bs,K,N)->(Bs,M,N).
    NumPy, WebGPU and WebGL all expose native 3D matmul. An arbitrary failure
    must propagate: a global 'unsupported' latch used to turn any transient
    shader, device or shape error into a permanent per-head Python loop, hiding
    both the error and the loss of parallelism for every later question."""
    return A @ B


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


_BANDED_QK_WGSL = """
@group(0) @binding(0) var<storage,read> q: array<f32>;
@group(0) @binding(1) var<storage,read> k: array<f32>;
@group(0) @binding(2) var<storage,read> mask: array<f32>;
@group(0) @binding(3) var<storage,read_write> out: array<f32>;
struct Meta { heads: u32, T: u32, D: u32, mask_group: u32, window: u32, scale: f32, }
@group(0) @binding(4) var<storage,read> qm: Meta;
var<workgroup> tile_q: array<array<f32, 16>, 16>;
var<workgroup> tile_k: array<array<f32, 16>, 16>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) lane: vec3<u32>) {
  let row = group.y * 16u + lane.y;
  let col = group.x * 16u + lane.x;
  let head = group.z;
  let mask_index = (head / qm.mask_group) * qm.T * qm.T + row * qm.T + col;
  let out_index = head * qm.T * qm.T + row * qm.T + col;
  // Every lane in a workgroup takes this branch together. A tile wholly outside
  // the checkpoint-declared sliding window contributes zero probability after
  // the additive -1e9 mask, so it needs no QK dot product or shared-memory load.
  if (qm.window > 0u &&
      (group.x * 16u > group.y * 16u + 15u + qm.window ||
       group.y * 16u > group.x * 16u + 15u + qm.window)) {
    if (row < qm.T && col < qm.T) { out[out_index] = mask[mask_index]; }
    return;
  }
  var sum = 0.0;
  for (var base: u32 = 0u; base < qm.D; base = base + 16u) {
    let qdim = base + lane.x;
    let kdim = base + lane.y;
    var qv = 0.0;
    var kv = 0.0;
    if (row < qm.T && qdim < qm.D) {
      qv = q[(head * qm.T + row) * qm.D + qdim];
    }
    if (col < qm.T && kdim < qm.D) {
      kv = k[(head * qm.T + col) * qm.D + kdim];
    }
    tile_q[lane.y][lane.x] = qv;
    tile_k[lane.y][lane.x] = kv;
    workgroupBarrier();
    for (var d: u32 = 0u; d < 16u; d = d + 1u) {
      sum = sum + tile_q[lane.y][d] * tile_k[d][lane.x];
    }
    workgroupBarrier();
  }
  if (row < qm.T && col < qm.T) {
    out[out_index] = sum * qm.scale + mask[mask_index];
  }
}
"""
_banded_qk_kernel = {"added": False}

_BANDED_PV_WGSL = """
@group(0) @binding(0) var<storage,read> prob: array<f32>;
@group(0) @binding(1) var<storage,read> value: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
struct Meta { heads: u32, T: u32, D: u32, window: u32, }
@group(0) @binding(3) var<storage,read> pm: Meta;
var<workgroup> tile_p: array<array<f32, 16>, 16>;
var<workgroup> tile_v: array<array<f32, 16>, 16>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) lane: vec3<u32>) {
  let row = group.y * 16u + lane.y;
  let col = group.x * 16u + lane.x;
  let head = group.z;
  var sum = 0.0;
  for (var base: u32 = 0u; base < pm.T; base = base + 16u) {
    // The softmax of an additive -1e9 mask is exactly zero outside the
    // bidirectional window. A whole K tile outside every row here adds zero.
    if (base > group.y * 16u + 15u + pm.window ||
        group.y * 16u > base + 15u + pm.window) { continue; }
    let pk = base + lane.x;
    let vk = base + lane.y;
    var pv = 0.0;
    var vv = 0.0;
    if (row < pm.T && pk < pm.T) {
      pv = prob[(head * pm.T + row) * pm.T + pk];
    }
    if (vk < pm.T && col < pm.D) {
      vv = value[(head * pm.T + vk) * pm.D + col];
    }
    tile_p[lane.y][lane.x] = pv;
    tile_v[lane.y][lane.x] = vv;
    workgroupBarrier();
    for (var t: u32 = 0u; t < 16u; t = t + 1u) {
      sum = sum + tile_p[lane.y][t] * tile_v[t][lane.x];
    }
    workgroupBarrier();
  }
  if (row < pm.T && col < pm.D) {
    out[(head * pm.T + row) * pm.D + col] = sum;
  }
}
"""
_banded_pv_kernel = {"added": False}


def banded_qk_scores(q, k, mask, scale, window):
    """Inference-only bidirectional QK with the model's additive mask.

    A whole out-of-window tile skips the dot product; in-window scores combine
    QK, scale and the existing additive mask in one GPU dispatch. A zero
    window means full attention; every tile is then computed. Returning
    None means the caller uses the semantically equivalent generic operators
    (including WebGL and autograd), not that an execution error was hidden.
    """
    if q.requires_grad or k.requires_grad or mask.requires_grad:
        return None
    if not _adam_backend_ready():
        return None
    if len(q.shape) != 3 or q.shape != k.shape:
        return None
    heads, T, D = map(int, q.shape)
    if mask.shape == (T, T):
        mask_group = heads
    elif (len(mask.shape) == 3 and mask.shape[1:] == (T, T)
          and mask.shape[0] > 0 and heads % mask.shape[0] == 0):
        # One sequence's heads have the same padding/window mask. A compact
        # (B,T,T) mask can serve (B*H,T,T) attention scores without replicating
        # and uploading the same plane H times.
        mask_group = heads // int(mask.shape[0])
    else:
        return None
    plat = _adam_kernel["platform"]
    if not _banded_qk_kernel["added"]:
        plat.addKernel("banded_qk_scores", {
            "source": _BANDED_QK_WGSL,
            "bindingTypes": ["read-only-storage", "read-only-storage",
                             "read-only-storage", "storage", "read-only-storage"]})
        _banded_qk_kernel["added"] = True
    qd = _contig(q.data)
    kd = _contig(k.data)
    md = _contig(mask.data)
    out = _empty((heads, T, T))
    meta = _adam_kernel["make_meta"](
        (heads, T, D, mask_group, int(window), float(scale)),
        "u4,u4,u4,u4,u4,f4")
    plat.runKernel({"name": "banded_qk_scores",
                    "tensors": [qd.buffer.buffer_id, kd.buffer.buffer_id,
                                md.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (T + 15) // 16,
                                   "y": (T + 15) // 16, "z": heads}})
    return Tensor(out)


def banded_pv(prob, value, window):
    """Inference-only P @ V for a checkpoint-declared bidirectional window."""
    if not window or prob.requires_grad or value.requires_grad:
        return None
    if not _adam_backend_ready():
        return None
    if len(prob.shape) != 3 or len(value.shape) != 3:
        return None
    heads, T, width = map(int, prob.shape)
    if T != width or value.shape[:2] != (heads, T):
        return None
    D = int(value.shape[2])
    plat = _adam_kernel["platform"]
    if not _banded_pv_kernel["added"]:
        plat.addKernel("banded_pv", {
            "source": _BANDED_PV_WGSL,
            "bindingTypes": ["read-only-storage", "read-only-storage",
                             "storage", "read-only-storage"]})
        _banded_pv_kernel["added"] = True
    pd = _contig(prob.data)
    vd = _contig(value.data)
    out = _empty((heads, T, D))
    meta = _adam_kernel["make_meta"]((heads, T, D, int(window)), "u4,u4,u4,u4")
    plat.runKernel({"name": "banded_pv",
                    "tensors": [pd.buffer.buffer_id, vd.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (D + 15) // 16,
                                   "y": (T + 15) // 16, "z": heads}})
    return Tensor(out)



# ---- bidirectional attention straight off a packed projection ----------------------------
#
# An encoder layer's attention written as primitives is five dispatches over four full-size
# intermediates: q, k and v each taken out of the packed projection (and q, k rotated), the
# (heads, T, T) scores, their softmax, P @ V, and a transpose back to rows. Measured on a
# 22-layer 768-wide encoder at 3 x 173 tokens that is 0.95 ms a layer, 21 ms of a 82 ms pass,
# for about 0.3 GFLOP of arithmetic.
#
# Here it is two: the rotation of q and k (one pass, written out once, because every key block
# would otherwise re-rotate the same rows), then one kernel per (sequence, head, query block)
# that reads q, k and v where they are, keeps a running softmax over key blocks, and writes
# the output already in the (rows, heads * head_dim) layout the out-projection reads. Nothing
# T x T is ever stored.
#
# Workgroup memory holds only the probability tile and two reductions. A first version staged
# q, k and v tiles there as well -- 32 KB a workgroup, so one or two workgroups fit a core and
# the GPU waited on every barrier; reading them through the cache instead was twice as fast.

_ROPE_QK_WGSL = """
@group(0) @binding(0) var<storage,read_write> qk: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> qkv: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> cosb: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> sinb: array<vec4<f32>>;
struct RMeta { rows: u32, T: u32, H: u32, HD4: u32, }
@group(0) @binding(4) var<storage,read> rm: RMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  // One thread per (row, q or k, head, vec4 of the first half): the pair it rotates.
  let half = rm.HD4 / 2u;
  let per_row = 2u * rm.H * half;
  let i = g.x + g.y * 65535u * 64u;
  if (i >= rm.rows * per_row) { return; }
  let row = i / per_row;
  let r = i % per_row;
  let which = r / (rm.H * half);
  let hh = (r / half) % rm.H;
  let d = r % half;
  let t = row % rm.T;
  let src = row * 3u * rm.H * rm.HD4 + which * rm.H * rm.HD4 + hh * rm.HD4;
  let dst = row * 2u * rm.H * rm.HD4 + which * rm.H * rm.HD4 + hh * rm.HD4;
  let lo = qkv[src + d]; let hi = qkv[src + d + half];
  let cl = cosb[t * rm.HD4 + d]; let sl = sinb[t * rm.HD4 + d];
  let ch = cosb[t * rm.HD4 + d + half]; let sh = sinb[t * rm.HD4 + d + half];
  qk[dst + d] = lo * cl - hi * sl;
  qk[dst + d + half] = hi * ch + lo * sh;
}
"""
_rope_qk_kernel = {"added": False}


def rope_qk(qkv, cos, sin, H, HD, T, B=1):
    """q and k of a packed (B*T, 3*H*HD) projection, rotated, as one (B*T, 2*H*HD) tensor.

    The rotation is `qkv_take`'s: the second half of each head negated into the first. Out of
    place, so the projection is left as it was -- the attention kernel still reads v there.
    None without WebGPU or for a shape this does not take."""
    if not _adam_backend_ready():
        return None
    H, HD, T, B = int(H), int(HD), int(T), int(B)
    xd = _contig(qkv.data if isinstance(qkv, Tensor) else qkv)
    cd = _contig(cos.data if isinstance(cos, Tensor) else cos)
    sd = _contig(sin.data if isinstance(sin, Tensor) else sin)
    if (tuple(xd.shape) != (B * T, 3 * H * HD) or HD % 8
            or tuple(cd.shape) != (T, HD) or tuple(sd.shape) != (T, HD)):
        return None
    plat = _adam_kernel["platform"]
    if not _rope_qk_kernel["added"]:
        plat.addKernel("rope_qk", {"source": _ROPE_QK_WGSL,
                                   "bindingTypes": ["storage"] + ["read-only-storage"] * 4})
        _rope_qk_kernel["added"] = True
    out = _empty((B * T, 2 * H * HD))
    n = B * T * H * (HD // 4)          # pairs of vec4: 2 (q, k) x H x HD/8 per row
    meta = _adam_kernel["make_meta"]((B * T, T, H, HD // 4), "u4,u4,u4,u4")
    plat.runKernel({"name": "rope_qk",
                    "tensors": [out.buffer.buffer_id, xd.buffer.buffer_id, cd.buffer.buffer_id,
                                sd.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": min((n + 63) // 64, 65535),
                                   "y": (n + 64 * 65535 - 1) // (64 * 65535), "z": 1}})
    return Tensor(out)


def _attn_src(HD, RI, CJ):
    """WGSL for one (sequence, head) and 8*RI queries against key blocks of 8*CJ.

    8 x 8 threads. Thread (tx, ty) owns query rows ty + 8i (i < RI); for the scores, keys
    tx*CJ .. tx*CJ+CJ-1 of the block; for the output, head-dim vec4s tx + 8e. A row's softmax
    statistics are combined across its eight tx threads through workgroup memory, and each
    thread writes its probabilities as whole vec4s -- assigning one component of a shared
    vec4 compiles to a read-modify-write of all four, which two threads then race on.
    """
    HD4 = HD // 4
    BQ, BK = 8 * RI, 8 * CJ
    E = HD4 // 8
    PS = BK // 4 + 1                   # padded vec4 stride of a probability row
    L = []
    a = L.append
    a("""
@group(0) @binding(0) var<storage,read> qks: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> vsrc: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> mask: array<f32>;
@group(0) @binding(3) var<storage,read_write> out: array<vec4<f32>>;
struct AMeta { T: u32, H: u32, group: u32, window: u32, scale: f32,
               qs: u32, qo: u32, ko: u32, vs: u32, vo: u32, }
@group(0) @binding(4) var<storage,read> am: AMeta;
var<workgroup> pt: array<vec4<f32>, %d>;
var<workgroup> red: array<f32, %d>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y;
  let T = am.T; let H = am.H;
  let bh = wg.y; let b = bh / H; let h = bh %% H;
  let q0 = wg.x * %du;
  let DO = H * %du;
  let masked = am.group > 0u;
  let mbase = select(0u, (bh / max(am.group, 1u)) * T * T, masked);
  let qcol = am.qo + h * %du; let kcol = am.ko + h * %du; let vcol = am.vo + h * %du;""" % (
        BQ * PS, BQ * 8, BQ, HD4, HD4, HD4, HD4))
    for i in range(RI):
        a("  let qr%d = min(q0 + ty + %du, T - 1u); let qb%d = (b * T + qr%d) * am.qs + qcol;"
          % (i, 8 * i, i, i))
        a("  var m%d: f32 = -3.0e38; var l%d: f32 = 0.0;" % (i, i))
        for e in range(E):
            a("  var o%d_%d = vec4<f32>();" % (i, e))
    a("""  var kb0 = 0u;
  var kb1 = (T + %(BK)du - 1u) / %(BK)du;
  if (am.window > 0u) {
    // Blocks wholly outside every row's window are all mask; skipping them changes no
    // probability. The mask itself still decides every score that is computed.
    kb0 = select(0u, (q0 - am.window) / %(BK)du, q0 > am.window);
    kb1 = min(kb1, min(T - 1u, q0 + %(BQ)du - 1u + am.window) / %(BK)du + 1u);
  }
  for (var kb = kb0; kb < kb1; kb = kb + 1u) {
    let k0 = kb * %(BK)du;""" % dict(BK=BK, BQ=BQ))
    for j in range(CJ):
        a("    let kr%d = min(k0 + tx * %du + %du, T - 1u); let kp%d = (b * T + kr%d) * am.qs + kcol;"
          % (j, CJ, j, j, j))
    for i in range(RI):
        for j in range(CJ):
            a("    var s%d_%d: f32 = 0.0;" % (i, j))
    a("    for (var d = 0u; d < %du; d = d + 1u) {" % HD4)
    for i in range(RI):
        a("      let qv%d = qks[qb%d + d];" % (i, i))
    for j in range(CJ):
        a("      let kv%d = qks[kp%d + d];" % (j, j))
    for i in range(RI):
        for j in range(CJ):
            a("      s%d_%d = s%d_%d + dot(qv%d, kv%d);" % (i, j, i, j, i, j))
    a("    }")
    for i in range(RI):
        for j in range(CJ):
            a("    { let key = k0 + tx * %du + %du;" % (CJ, j))
            a("      if (key >= T) { s%d_%d = -3.0e38; } else if (masked) {"
              " s%d_%d = s%d_%d * am.scale + mask[mbase + qr%d * T + key]; }"
              " else { s%d_%d = s%d_%d * am.scale; } }"
              % (i, j, i, j, i, j, i, i, j, i, j))

    def mx(names):
        e = names[0]
        for n in names[1:]:
            e = "max(%s, %s)" % (e, n)
        return e
    for i in range(RI):
        a("    red[(ty + %du) * 8u + tx] = %s;" % (8 * i, mx(["s%d_%d" % (i, j) for j in range(CJ)])))
    a("    workgroupBarrier();")
    for i in range(RI):
        a("    var bm%d = red[(ty + %du) * 8u];" % (i, 8 * i))
        a("    for (var t = 1u; t < 8u; t = t + 1u) { bm%d = max(bm%d, red[(ty + %du) * 8u + t]); }"
          % (i, i, 8 * i))
        a("    let mn%d = max(m%d, bm%d); let al%d = exp(m%d - mn%d); m%d = mn%d;"
          % (i, i, i, i, i, i, i, i))
        ps_ = []
        for j in range(CJ):
            a("    let p%d_%d = exp(s%d_%d - m%d);" % (i, j, i, j, i))
            ps_.append("p%d_%d" % (i, j))
        for g in range(CJ // 4):
            a("    pt[(ty + %du) * %du + tx * %du + %du] = vec4<f32>(%s);"
              % (8 * i, PS, CJ // 4, g, ", ".join(ps_[4 * g:4 * g + 4])))
        a("    l%d = l%d * al%d + %s;" % (i, i, i, " + ".join(ps_)))
        for e in range(E):
            a("    o%d_%d = o%d_%d * al%d;" % (i, e, i, e, i))
    a("    workgroupBarrier();")
    a("    for (var c = 0u; c < %du; c = c + 1u) {" % (BK // 4))
    for i in range(RI):
        a("      let pp%d = pt[(ty + %du) * %du + c];" % (i, 8 * i, PS))
    for u in range(4):
        a("      let vr%d = (b * T + min(k0 + c * 4u + %du, T - 1u)) * am.vs + vcol;" % (u, u))
        for e in range(E):
            a("      let v%d_%d = vsrc[vr%d + tx + %du];" % (u, e, u, 8 * e))
    for i in range(RI):
        for e in range(E):
            a("      o%d_%d = o%d_%d + pp%d.x * v0_%d + pp%d.y * v1_%d + pp%d.z * v2_%d"
              " + pp%d.w * v3_%d;" % (i, e, i, e, i, e, i, e, i, e, i, e))
    a("    }")
    a("    workgroupBarrier();")
    a("  }")
    for i in range(RI):
        a("  red[(ty + %du) * 8u + tx] = l%d;" % (8 * i, i))
    a("  workgroupBarrier();")
    for i in range(RI):
        a("  { var lt = 0.0; for (var t = 0u; t < 8u; t = t + 1u) {"
          " lt = lt + red[(ty + %du) * 8u + t]; }" % (8 * i))
        a("    let qr = q0 + ty + %du;" % (8 * i))
        a("    if (qr < T) { let inv = 1.0 / lt;")
        for e in range(E):
            a("      out[(b * T + qr) * DO + h * %du + tx + %du] = o%d_%d * inv;"
              % (HD4, 8 * e, i, e))
        a("    } }")
    a("}")
    return "\n".join(L)


# Thread tilings raced per device: (rows, keys) per thread. The first is the default.
_ATTN_TILES = {"4x4": (4, 4), "2x8": (2, 8)}
_attn_added = set()


def _attn_run(tile, src_qk, src_v, md, H, HD, T, B, scale, window, group, qs, qo, ko, vs, vo):
    RI, CJ = _ATTN_TILES[tile]
    name = "attn_%d_%s" % (HD, tile)
    plat = _adam_kernel["platform"]
    if name not in _attn_added:
        plat.addKernel(name, {"source": _attn_src(HD, RI, CJ),
                              "bindingTypes": ["read-only-storage"] * 3
                              + ["storage", "read-only-storage"]})
        _attn_added.add(name)
    out = _empty((B * T, H * HD))
    meta = _adam_kernel["make_meta"](
        (T, H, group, int(window), float(scale), qs, qo, ko, vs, vo),
        "u4,u4,u4,u4,f4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": name,
                    "tensors": [src_qk.buffer.buffer_id, src_v.buffer.buffer_id,
                                md.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (T + 8 * RI - 1) // (8 * RI), "y": B * H, "z": 1}})
    return out


def fused_attention(qkv, H, HD, T, scale, mask=None, window=0, B=1, cos=None, sin=None):
    """Bidirectional attention of a packed (B*T, 3*H*HD) projection, as (B*T, H*HD) rows.

    q, k and v are the projection's three thirds, head-major inside each; with `cos`/`sin`
    q and k are rotated first (`rope_qk`). `mask` is additive: (T, T) for every head, or
    (P, T, T) with B*H a multiple of P, each plane serving B*H/P consecutive (sequence,
    head) pairs. `window` > 0 lets whole key blocks outside |i - j| <= window be skipped --
    the mask must still say so, it is what decides. Inference only: None with a gradient,
    without WebGPU, or for a shape this does not take, and the caller keeps its expression.
    """
    if not _adam_backend_ready():
        return None
    if getattr(qkv, "requires_grad", False) or (mask is not None and mask.requires_grad):
        return None
    H, HD, T, B = int(H), int(HD), int(T), int(B)
    xd = _contig(qkv.data if isinstance(qkv, Tensor) else qkv)
    if tuple(xd.shape) != (B * T, 3 * H * HD) or HD % 32 or T < 1:
        return None
    if mask is None:
        group, md = 0, xd
    else:
        md = _contig(mask.data if isinstance(mask, Tensor) else mask)
        if tuple(md.shape) == (T, T):
            group = B * H
        elif (len(md.shape) == 3 and tuple(md.shape[1:]) == (T, T) and int(md.shape[0]) > 0
              and (B * H) % int(md.shape[0]) == 0):
            group = (B * H) // int(md.shape[0])
        else:
            return None
    HD4 = HD // 4
    if cos is not None and sin is not None:
        qk = rope_qk(xd, cos, sin, H, HD, T, B)
        if qk is None:
            return None
        src_qk, qs, qo, ko = qk.data, 2 * H * HD4, 0, H * HD4
    else:
        src_qk, qs, qo, ko = xd, 3 * H * HD4, 0, H * HD4
    args = (src_qk, xd, md, H, HD, T, B, float(scale), int(window), group,
            qs, qo, ko, 3 * H * HD4, 2 * H * HD4)
    tile = _weight_execution("attention", "f32", HD, 1 if window else 0, T,
                             lambda which: _attn_run(which, *args),
                             candidates=tuple(_ATTN_TILES))
    return Tensor(_attn_run(tile, *args))


# ---- causal attention of new rows against the packed-half KV cache, where it lies -------
#
# A prefill's attention used to unpack the cache's live span to f32 (one copy of every key
# and value so far, per layer), run `flash_attention` -- which measured ~0.1 TFLOPS here --
# or the chunked form, and transpose the result back to rows (another copy). Per layer on a
# 16-head, head_dim-128 model: 1.41 ms at 182 tokens, 8.1 ms for 64 new tokens after 2000
# cached ones, 9.5 ms at 1024.
#
# Here one kernel reads q, the cache's halves and writes the out-projection's rows: 0.31,
# 0.70 and 2.56 ms for the same three. The design is the encoder's (`_attn_src`): only the
# probability tile in workgroup memory, q/k/v through the cache. A short prompt after a long
# conversation has few query blocks, so the keys can be split over workgroups (wg.z) and
# the partial softmax states merged after; how far to split is measured per device.

def _cattn_src(HD, RI=4, CJ=4):
    """8x8 threads, 8*RI queries of one head against key blocks of 8*CJ, over the key range
    wg.z owns. Thread (tx, ty): rows ty+8i, keys tx*CJ+j, output vec4s tx+8e. Writes the
    normalised rows (one split) or the unnormalised partial and its (max, sum)."""
    HD4 = HD // 4
    BQ, BK = 8 * RI, 8 * CJ
    E = HD4 // 8
    PS = BK // 4 + 1
    L = []
    a = L.append
    a("""
@group(0) @binding(0) var<storage,read> q4: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage,read_write> out4: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> ml: array<vec2<f32>>;
struct CMeta { T: u32, start: u32, NH: u32, rep: u32, LMAX: u32, S: u32, span: u32, scale: f32, }
@group(0) @binding(5) var<storage,read> cm: CMeta;
var<workgroup> pt: array<vec4<f32>, %d>;
var<workgroup> red: array<f32, %d>;
fn K4(base: u32, d: u32) -> vec4<f32> {
  let w = kc[base + d]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
}
fn V4(base: u32, d: u32) -> vec4<f32> {
  let w = vc[base + d]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y));
}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tx = lid.x; let ty = lid.y;
  let T = cm.T; let h = wg.y; let kvh = h / cm.rep;
  let q0 = wg.x * %du;
  let kvbase = kvh * cm.LMAX;
  // The keys this query block sees at all end at its last row's own position.
  let kend = cm.start + min(q0 + %du, T);
  let ks = wg.z * cm.span;
  let ke = min(kend, ks + cm.span);""" % (BQ * PS, BQ * 8, BQ, BQ))
    for i in range(RI):
        a("  let qr%d = min(q0 + ty + %du, T - 1u); let qp%d = cm.start + q0 + ty + %du;"
          % (i, 8 * i, i, 8 * i))
        a("  let qb%d = (h * T + qr%d) * %du;" % (i, i, HD4))
        a("  var m%d: f32 = -3.0e38; var l%d: f32 = 0.0;" % (i, i))
        for e in range(E):
            a("  var o%d_%d = vec4<f32>();" % (i, e))
    a("  for (var k0 = ks; k0 < ke; k0 = k0 + %du) {" % BK)
    for j in range(CJ):
        a("    let kr%d = min(k0 + tx * %du + %du, ke - 1u); let kb%d = (kvbase + kr%d) * %du;"
          % (j, CJ, j, j, j, HD4))
    for i in range(RI):
        for j in range(CJ):
            a("    var s%d_%d: f32 = 0.0;" % (i, j))
    a("    for (var d = 0u; d < %du; d = d + 1u) {" % HD4)
    for i in range(RI):
        a("      let qv%d = q4[qb%d + d];" % (i, i))
    for j in range(CJ):
        a("      let kv%d = K4(kb%d, d);" % (j, j))
    for i in range(RI):
        for j in range(CJ):
            a("      s%d_%d = s%d_%d + dot(qv%d, kv%d);" % (i, j, i, j, i, j))
    a("    }")
    for i in range(RI):
        for j in range(CJ):
            a("    { let key = k0 + tx * %du + %du;" % (CJ, j))
            a("      s%d_%d = select(-3.0e38, s%d_%d * cm.scale, key < ke && key <= qp%d); }"
              % (i, j, i, j, i))

    def mx(names):
        e = names[0]
        for n in names[1:]:
            e = "max(%s, %s)" % (e, n)
        return e
    for i in range(RI):
        a("    red[(ty + %du) * 8u + tx] = %s;"
          % (8 * i, mx(["s%d_%d" % (i, j) for j in range(CJ)])))
    a("    workgroupBarrier();")
    for i in range(RI):
        a("    var bm%d = red[(ty + %du) * 8u];" % (i, 8 * i))
        a("    for (var t = 1u; t < 8u; t = t + 1u) { bm%d = max(bm%d, red[(ty + %du) * 8u + t]); }"
          % (i, i, 8 * i))
        # A row that sees nothing yet keeps m at -3e38; nothing it skipped may count.
        a("    let mn%d = max(m%d, bm%d); let al%d = select(0.0, exp(m%d - mn%d), m%d > -1.0e38);"
          " m%d = mn%d;" % (i, i, i, i, i, i, i, i, i))
        ps_ = []
        for j in range(CJ):
            a("    let p%d_%d = select(0.0, exp(s%d_%d - m%d), s%d_%d > -1.0e38);"
              % (i, j, i, j, i, i, j))
            ps_.append("p%d_%d" % (i, j))
        for g in range(CJ // 4):
            a("    pt[(ty + %du) * %du + tx * %du + %du] = vec4<f32>(%s);"
              % (8 * i, PS, CJ // 4, g, ", ".join(ps_[4 * g:4 * g + 4])))
        a("    l%d = l%d * al%d + %s;" % (i, i, i, " + ".join(ps_)))
        for e in range(E):
            a("    o%d_%d = o%d_%d * al%d;" % (i, e, i, e, i))
    a("    workgroupBarrier();")
    a("    for (var c = 0u; c < %du; c = c + 1u) {" % (BK // 4))
    for i in range(RI):
        a("      let pp%d = pt[(ty + %du) * %du + c];" % (i, 8 * i, PS))
    for u in range(4):
        a("      let vb%d = (kvbase + min(k0 + c * 4u + %du, ke - 1u)) * %du;" % (u, u, HD4))
        for e in range(E):
            a("      let v%d_%d = V4(vb%d, tx + %du);" % (u, e, u, 8 * e))
    for i in range(RI):
        for e in range(E):
            a("      o%d_%d = o%d_%d + pp%d.x * v0_%d + pp%d.y * v1_%d + pp%d.z * v2_%d"
              " + pp%d.w * v3_%d;" % (i, e, i, e, i, e, i, e, i, e, i, e))
    a("    }")
    a("    workgroupBarrier();")
    a("  }")
    for i in range(RI):
        a("  red[(ty + %du) * 8u + tx] = l%d;" % (8 * i, i))
    a("  workgroupBarrier();")
    for i in range(RI):
        a("  { var lt = 0.0; for (var t = 0u; t < 8u; t = t + 1u) {"
          " lt = lt + red[(ty + %du) * 8u + t]; }" % (8 * i))
        a("    let qr = q0 + ty + %du;" % (8 * i))
        a("    if (qr < T) {")
        a("      if (cm.S == 1u) {")
        a("        let inv = 1.0 / lt;")
        for e in range(E):
            a("        out4[(qr * cm.NH + h) * %du + tx + %du] = o%d_%d * inv;" % (HD4, 8 * e, i, e))
        a("      } else {")
        a("        let pb = (wg.z * T + qr) * cm.NH + h;")
        for e in range(E):
            a("        out4[pb * %du + tx + %du] = o%d_%d;" % (HD4, 8 * e, i, e))
        a("        if (tx == 0u) { ml[pb] = vec2<f32>(m%d, lt); }" % i)
        a("      }")
        a("    } }")
    a("}")
    return "\n".join(L)


_CATTN_MERGE_WGSL = """
@group(0) @binding(0) var<storage,read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> ml: array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> out4: array<vec4<f32>>;
struct MMeta { T: u32, NH: u32, HD4: u32, S: u32, }
@group(0) @binding(3) var<storage,read> mm: MMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= mm.T * mm.NH * mm.HD4) { return; }
  let row = i / mm.HD4;                       // (t, h)
  let d = i % mm.HD4;
  var mx = -3.0e38;
  for (var z = 0u; z < mm.S; z = z + 1u) { mx = max(mx, ml[z * mm.T * mm.NH + row].x); }
  var acc = vec4<f32>(); var l = 0.0;
  for (var z = 0u; z < mm.S; z = z + 1u) {
    let e = ml[z * mm.T * mm.NH + row];
    let w = select(0.0, exp(e.x - mx), e.x > -1.0e38);
    acc = acc + part[(z * mm.T * mm.NH + row) * mm.HD4 + d] * w;
    l = l + e.y * w;
  }
  out4[row * mm.HD4 + d] = acc / l;
}
"""
_cattn_added = set()
# How many workgroups a split aims for: "p1" never splits. Raced per device, by context.
_CATTN_TARGETS = {"p1": 1, "p96": 96, "p192": 192, "p384": 384}


def _cattn_split(target, T, end, NH):
    qb = (T + 31) // 32
    have = qb * NH
    if target <= have:
        return 1
    return max(1, min((target + have - 1) // have, (end + 63) // 64))


def _cattn_run(target, qd, kd, vd, T, start, NH, NKV, HD, LMAX, scale):
    plat = _adam_kernel["platform"]
    name = "cattn_%d" % HD
    if name not in _cattn_added:
        plat.addKernel(name, {"source": _cattn_src(HD),
                              "bindingTypes": ["read-only-storage"] * 3
                              + ["storage", "storage", "read-only-storage"]})
        plat.addKernel("cattn_merge", {"source": _CATTN_MERGE_WGSL,
                                       "bindingTypes": ["read-only-storage", "read-only-storage",
                                                        "storage", "read-only-storage"]})
        _cattn_added.add(name)
    end = start + T
    S = _cattn_split(_CATTN_TARGETS[target], T, end, NH)
    span = ((end + S - 1) // S + 31) // 32 * 32
    S = (end + span - 1) // span
    if S == 1:
        out, ml = _empty((T, NH * HD)), _empty((2,))
    else:
        out, ml = _empty((S * T * NH * HD,)), _empty((S * T * NH * 2,))
    meta = _adam_kernel["make_meta"]((T, start, NH, NH // NKV, LMAX, S, span, float(scale)),
                                     "u4,u4,u4,u4,u4,u4,u4,f4")
    plat.runKernel({"name": name,
                    "tensors": [qd.buffer.buffer_id, kd.buffer.buffer_id, vd.buffer.buffer_id,
                                out.buffer.buffer_id, ml.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (T + 31) // 32, "y": NH, "z": S}})
    if S == 1:
        return out
    fin = _empty((T, NH * HD))
    mm = _adam_kernel["make_meta"]((T, NH, HD // 4, S), "u4,u4,u4,u4")
    n = T * NH * HD // 4
    plat.runKernel({"name": "cattn_merge",
                    "tensors": [out.buffer.buffer_id, ml.buffer.buffer_id, fin.buffer.buffer_id,
                                mm.buffer_id],
                    "workGroups": {"x": (n + 63) // 64, "y": 1, "z": 1}})
    return fin


def causal_attention_cache(q, kcache, vcache, start, NH, NKV, HD, LMAX, scale):
    """Attention of T new rows at positions `start`.. against a KV cache of packed halves
    (`kv_f16`), as (T, NH*HD) rows; keys 0..start+t for row t. `q` is (NH, T, HD), already
    rotated; `kcache`/`vcache` are the caches (NKV, LMAX, HD/2 words) after this prompt's
    rows were written. None where this does not apply (no WebGPU, an f32 cache, HD not a
    multiple of 32) and the caller keeps its path."""
    if not (_adam_backend_ready() and kv_f16()):
        return None
    NH, NKV, HD, LMAX, start = int(NH), int(NKV), int(HD), int(LMAX), int(start)
    qd = _contig(q.data if isinstance(q, Tensor) else q)
    if (HD % 32 or NH % NKV or len(qd.shape) != 3 or int(qd.shape[0]) != NH
            or int(qd.shape[2]) != HD):
        return None
    T = int(qd.shape[1])
    kd = kcache.data if isinstance(kcache, Tensor) else kcache
    vd = vcache.data if isinstance(vcache, Tensor) else vcache
    args = (qd, kd, vd, T, start, NH, NKV, HD, LMAX, float(scale))
    target = _weight_execution("causal_attention", "f16kv", HD, NH, start + T,
                               lambda which: _cattn_run(which, *args),
                               candidates=tuple(_CATTN_TARGETS))
    return Tensor(_cattn_run(target, *args))

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
        if T >= _ATTN_CHUNK_MIN_T:
            return chunked_attention(q, k, v, start=causal_start, scale=scale)
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
_WEIGHT_TUNE_SECONDS = 0.0
_WEIGHT_TUNE_CALLS = 0
# Route races belong after a load, not inside an answer. A weight whose routes were measured
# over a ladder of row counts (`calibrate_rows`) is "calibrated": its key prefix
# (everything but the row bucket) is in `_CALIBRATED`, and a row count the ladder did not
# visit takes the choice of the nearest bucket it did, instead of racing in front of the
# person waiting. While calibrating, every race is run and every key asked is noted.
_CALIBRATING = [0]
_CALIBRATED = set()        # full ladder measured (kept in the device profile)
_PROVISIONAL = set()       # bottom probe measured, the rest of the ladder still queued
_DEFERRED = []             # queued ladders, advanced by `calibrate_deferred` when idle
_CALIB_TOUCHED = []
_NEAREST = {}


class _calibrating(object):
    """While inside, `_weight_execution` measures instead of borrowing."""

    def __enter__(self):
        _CALIBRATING[0] += 1
        return self

    def __exit__(self, *exc):
        _CALIBRATING[0] -= 1
        return False


def _nearest_tuned(key):
    """The measured choice of the bucket nearest `key`'s, in octaves, for a calibrated
    prefix; None when the prefix was never calibrated. A tie goes to the larger bucket."""
    prefix = key[:-1]
    if prefix not in _CALIBRATED and prefix not in _PROVISIONAL:
        return None
    hit = _NEAREST.get(key)
    if hit is not None and hit[0] == len(_TUNED):
        return hit[1]
    want = int(key[-1]).bit_length()
    best = None
    for k, v in _TUNED.items():
        if k[:-1] == prefix:
            d = abs(int(k[-1]).bit_length() - want)
            if best is None or d < best[0] or (d == best[0] and k[-1] > best[1]):
                best = (d, k[-1], v)
    choice = None if best is None else best[2]
    _NEAREST[key] = (len(_TUNED), choice)
    return choice


class _Ladder(object):
    """One weight's row-count ladder: probes run one at a time, bisected in octaves."""

    def __init__(self, probe, top, lo, step):
        self.probe = probe
        self.lo = 1 << (max(1, int(lo)) - 1).bit_length()
        self.top = 1 << (max(1, int(top)) - 1).bit_length()
        self.step = max(2, int(step))
        self.seen = {}
        self.need = [self.lo] + ([self.top] if self.top > self.lo else [])
        self.todo = [(self.lo, self.top)]

    def run(self, m):
        del _CALIB_TOUCHED[:]
        with _calibrating():
            self.probe(m)
        self.seen[m] = {k[:-1]: _TUNED[k] for k in _CALIB_TOUCHED if k in _TUNED}
        del _CALIB_TOUCHED[:]

    def advance(self):
        """Run the next probe; False once the ladder is complete."""
        while not self.need and self.todo:
            a, b = self.todo.pop()
            if b <= self.step * a or a not in self.seen or b not in self.seen:
                continue
            sa, sb = self.seen[a], self.seen[b]
            if all(sa[k] == sb[k] for k in sa if k in sb):
                continue
            mid = 1 << ((a.bit_length() + b.bit_length()) // 2 - 1)
            self.need.append(mid)
            self.todo += [(a, mid), (mid, b)]
        if not self.need:
            return False
        self.run(self.need.pop(0))
        return True

    def prefixes(self):
        out = set()
        for got in self.seen.values():
            out.update(got)
        return out


def calibrate_rows(probe, top, lo=1, step=2, defer=False):
    """Measure every route race `probe(m)` sets off over a ladder of row counts, once.

    `probe(m)` runs the operation at m rows. The ladder is bisected in octaves: `lo` and
    `top` first, then the middle of any interval whose two ends chose differently for some
    race, until the ends are at most `step` times apart -- so the crossovers are found where
    they are, on this device, without visiting every bucket. Near a crossover the candidates
    are close by definition, so locating it to within `step` costs little. Each race's
    prefix is then calibrated: a row count between probes takes the nearest probe's
    measured choice, and one past `top` takes `top`'s.

    With `defer`, only `lo` -- the cheap end -- is measured now; its prefixes may be borrowed
    from at once, and the rest of the ladder is queued for `calibrate_deferred`, which the
    host runs while nothing is asking. A 27B's full ladder was 40 s of a 95 s first load,
    almost all of it the stored kernel at the top probe. A ladder already complete for every
    prefix it touches (a remembered profile) queues nothing. Returns the probes run now.
    """
    ladder = _Ladder(probe, top, lo, step)
    if defer:
        ladder.run(ladder.need.pop(0))
        got = ladder.prefixes()
        if got and got <= _CALIBRATED:
            return sorted(ladder.seen)
        _PROVISIONAL.update(got - _CALIBRATED)
        _DEFERRED.append(ladder)
        _NEAREST.clear()
        return sorted(ladder.seen)
    while ladder.advance():
        pass
    _CALIBRATED.update(ladder.prefixes())
    _PROVISIONAL.difference_update(_CALIBRATED)
    _NEAREST.clear()
    return sorted(ladder.seen)


def calibrate_deferred(budget_s=0.5):
    """Advance the queued ladders for about `budget_s` seconds of probes; return how many
    ladders remain. Called by the host when no call is running, a step at a time, so a
    caller that arrives waits for at most one probe."""
    import time as _t
    t0 = _t.perf_counter()
    while _DEFERRED:                       # at least one step per call: progress is certain
        ladder = _DEFERRED[0]
        if not ladder.advance():
            _DEFERRED.pop(0)
            got = ladder.prefixes()
            _CALIBRATED.update(got)
            _PROVISIONAL.difference_update(got)
        if _t.perf_counter() - t0 >= float(budget_s):
            break
    _NEAREST.clear()
    return len(_DEFERRED)


def calibration_drop():
    """Forget queued ladders -- their probes hold the model being released."""
    del _DEFERRED[:]
    _PROVISIONAL.clear()
    _NEAREST.clear()


def _sync_small(a):
    """Wait for the work that produced `a` by reading back one element of it, not all of
    it: a race times each candidate to completion, and reading a 519x2304 answer back cost
    more than the kernel being timed (2.3 of 3.2 ms)."""
    d = a.data if isinstance(a, Tensor) else a
    if not hasattr(d, "reshape") or not hasattr(d, "get"):
        return
    flat = d.reshape(-1)
    np.asarray(_contig(flat[flat.shape[0] - 1:]).get())


def _paired_evidence(samples, candidate, baseline):
    """Exact paired sign-test evidence for one latency candidate."""
    import math as _math
    import statistics as _s
    a = list(samples.get(candidate, ()))
    b = list(samples.get(baseline, ()))
    pairs = [(x, y) for x, y in zip(a, b) if x != y]
    wins = sum(x < y for x, y in pairs)
    n = len(pairs)
    p = (sum(_math.comb(n, k) for k in range(wins, n + 1)) / float(2 ** n)
         if n else 1.0)
    return {
        "wins": wins,
        "pairs": n,
        "p": p,
        "candidate_median": _s.median(a) if a else float("inf"),
        "baseline_median": _s.median(b) if b else float("inf"),
    }


def _paired_faster(samples, candidate, baseline, alpha=0.05):
    """Return whether ``candidate`` has a repeatable paired latency win.

    There is deliberately no minimum percentage here.  The magnitude of a positive win
    is a measurement result, not a policy threshold.  We instead use the exact one-sided
    sign test: a candidate is accepted only when its paired samples beat the incumbent
    often enough that the result is unlikely to be measurement-order noise.  Equal samples
    carry no evidence either way.  This makes a stable 1% win usable while rejecting a
    larger but alternating win/loss result.
    """
    evidence = _paired_evidence(samples, candidate, baseline)
    return (evidence["pairs"] > 0
            and evidence["candidate_median"] < evidence["baseline_median"]
            and evidence["p"] <= float(alpha))


def _measured_choice(samples, candidates, default=None):
    """Choose faster implementations without discarding small local wins.

    ``candidates`` is ordered from lower to higher memory cost.  A higher-cost candidate
    replaces the current choice only when paired measurements prove it faster.  Thus an
    inconclusive timing naturally resolves to the lower-memory implementation, while every
    statistically repeatable positive latency result is retained regardless of magnitude.
    """
    candidates = tuple(candidates)
    if not candidates:
        return default
    chosen = default if default in candidates else candidates[0]
    for candidate in candidates:
        if candidate != chosen and _paired_faster(samples, candidate, chosen):
            chosen = candidate
    return chosen



def _route_names():
    """Every value a `_weight_execution` race can record, for accepting a saved profile. Taken
    from the candidate tables themselves, so a new route is kept across reloads the day it
    is added rather than silently re-measured on every load."""
    return ({"stored", "tiled", "tiled_half", "dp4a", "materialized", "full", "packed",
             "selected_full", "selected_q", "base", "alternate", "f32", "f16",
             "slots", "grouped", "grouped_half"}
            | set(_ATTN_TILES) | set(_CATTN_TARGETS))

def _weight_execution(family, storage_format, K, N, M, run,
                      candidates=("stored", "materialized"), check=None, rounds=9,
                      repeat=4):
    """Choose a correct implementation from device-local paired measurements.

    The key contains only operator facts, never a model/repository name.  Nearby batch
    sizes share a power-of-two bucket so a prompt does not pay for a new tune at every
    length.  This is the final, device-local level of the routing hierarchy: a candidate
    that did not win globally can still win for this format and shape bucket.  There is no
    model/repository allow-list and no percentage cutoff.  Any repeatable local
    latency win is retained.  If the samples do not prove a winner, the earlier candidate
    wins; callers therefore order candidates by memory cost, with the original stored
    representation first.
    """
    m = int(M)
    bucket = 1 << (m - 1).bit_length()
    key = ("weight_exec", str(family), str(storage_format), int(K), int(N), bucket)
    if _CALIBRATING[0]:
        _CALIB_TOUCHED.append(key)
    if key in _TUNED:
        return _TUNED[key]
    if not _CALIBRATING[0]:
        near = _nearest_tuned(key)
        if near is not None:
            return near
    import time as _t
    _tune_started = _t.perf_counter()
    candidates = tuple(candidates)
    # Small decode kernels need repeated work to rise above readback noise. A large
    # prefill is already milliseconds of GPU work per candidate; repeating it four times
    # both multiplies transient expanded-weight memory and puts thousands of calibration
    # dispatches before the first answer. Keep the same paired correctness/evidence gate,
    # but size each sample to the operator's work rather than a fixed repetition count.
    repeat = max(1, min(int(repeat), 1 if m >= 128 else 2 if m >= 32 else 4))

    def batch(which):
        out = None
        t0 = _t.perf_counter()
        for _ in range(repeat):
            out = run(which)
        _sync_small(out)
        return (_t.perf_counter() - t0) / repeat

    try:
        valid = []
        for which in candidates:
            try:
                if check is not None and not check(which):
                    raise RuntimeError("%s failed the correctness gate" % which)
                batch(which)                     # compile/warm before timing
                valid.append(which)
            except Exception as exc:
                raise RuntimeError("execution candidate %r failed for %s/%s shape (%d,%d,%d)"
                                   % (which, family, storage_format, M, K, N)) from exc
        if not valid:
            raise RuntimeError("no correct execution candidate")
        valid = tuple(valid)
        times = {which: [] for which in valid}
        for r in range(max(5, int(rounds))):
            order = valid if not (r & 1) else tuple(reversed(valid))
            for which in order:
                times[which].append(batch(which))
            # Five unanimous paired rounds are already p=1/32; more repetitions cannot
            # make that decision more necessary. What has to be settled is the WINNER
            # against each other candidate -- not how two losers rank against each other,
            # which a race of three spent its remaining rounds on whenever they were close.
            # If the winner is inconclusive against anything, keep measuring through the
            # requested rounds instead of dropping a local win.
            if r >= 4:
                lead = _measured_choice(times, valid, default=valid[0])
                if all(_paired_faster(times, lead, other)
                       for other in valid if other != lead):
                    break
        chosen = _measured_choice(times, valid, default=valid[0])
        _TUNED[key] = chosen
    finally:
        # Aggregate even failed calibration; a failed candidate must never be cached as
        # the first (possibly wrong) route, but its work still belongs in load diagnostics.
        global _WEIGHT_TUNE_SECONDS, _WEIGHT_TUNE_CALLS
        _WEIGHT_TUNE_SECONDS += _t.perf_counter() - _tune_started
        _WEIGHT_TUNE_CALLS += 1
    return chosen


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
                raise RuntimeError("candidate failed its correctness gate")
            bench()
            ok.append(v)
        except Exception as exc:
            raise RuntimeError("execution candidate %r failed for tune key %r"
                               % (v, key)) from exc
    if not ok:
        raise RuntimeError("no execution candidates for tune key %r" % (key,))
    times = {v: [] for v in ok}
    for r in range(rounds):
        # Alternate the queue order so a candidate cannot win merely because it always
        # follows (or always precedes) another implementation.  This is the same paired
        # evidence rule used by the stored-weight and containing-layer tuners.
        order = ok if not (r & 1) else tuple(reversed(ok))
        for v in order:
            apply(v)
            t0 = _t.perf_counter()
            bench()
            times[v].append(_t.perf_counter() - t0)
    chosen = _measured_choice(times, ok,
                              default=(default if default in ok else ok[0]))
    _TUNED[key] = chosen
    return chosen


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

    import time as _t

    def apply(kind):
        state["kind"] = kind
        # The registry's key, exactly as `_ggml_run` asks for it -- the decode kernel's last
        # field is 0. A shorter key here built the kernel but registered it under a name
        # nothing looks up, so the first un-built variant raced (Q4_K "narrow", on a 27B)
        # failed as "never built".
        k = (type_name, 1, kind, False, 0)
        if k not in _ggml_k["added"]:
            t0 = _t.perf_counter()
            _ggml_add(type_name, 1, kind, False)
            _ggml_k["added"].add(k)
            _TUNE_COST["build_s"] += _t.perf_counter() - t0
            _TUNE_COST["variants"] += 1

    def check(kind):
        # The check validates a KERNEL, and a kernel is (format, mode, thread shape, moe) --
        # it does not depend on N or K, which only size the test data. But this is called per
        # (shape, candidate), so the same handful of kernels is re-verified once per shape:
        # measured on a 0.6B, 24 checks for 6 distinct kernels, and 5.20s of the 6.40s the
        # whole tuning phase cost. On a 27B it is 96 checks for 36 kernels.
        #
        # So the answer is remembered per kernel -- but ONLY when it was reached with the
        # coverage the check itself demands. `_selfcheck_one` explains why three blocks per
        # row is the minimum: with one, every block offset is zero and a decode fragment that
        # reads a word directly looks correct while being wrong everywhere else. A pass with
        # thinner coverage than that answers for itself and for nothing after it.
        ck = (type_name, 1, kind, False)
        if ck in _CHECKED:
            return _CHECKED[ck]
        # Check at the size a CHECK needs, not at the size this model happens to use.
        #
        # `_selfcheck_one` builds N x (NB*blk) random bytes and decodes all of it with the
        # reference decoder, in numpy, inside wasm. Handed the model's own N and K that is a
        # 5120 x 5120 matrix -- 26 million values reference-decoded, per variant, to answer a
        # question about a decode fragment. Measured on a 27B: 36.6s across 36 variants.
        #
        # `_selfcheck_shape` already exists for exactly this and is what `_ggml_selfcheck`
        # uses: an (N, blocks) chosen so the variant is actually reached, N deliberately not
        # a multiple of the group width so the partial-group guard is exercised, and at least
        # three blocks per row so an unaligned block offset cannot hide -- the Q3_K bug that
        # cost half the columns of every tensor. Coverage is what those numbers are for; the
        # model's own N adds rows, not coverage.
        shape = _selfcheck_shape(kind, vals) or (int(N), max(3, nb))
        t0 = _t.perf_counter()
        try:
            _selfcheck_one(type_name, 1, kind, False, *shape)
        finally:
            _TUNE_COST["check_s"] += _t.perf_counter() - t0
        if shape[1] >= 3:
            _CHECKED[ck] = True
        return True

    # Many dispatches per sync. A readback costs 1-2ms on this stack and one decode matmul
    # costs tens of microseconds, so timing them one at a time measures the readback and
    # ranks the candidates by noise -- which is exactly what happened: the first version of
    # this picked a mix of shapes that took a 0.6B from 104 tok/s to 64.
    #
    # Sizing this per shape instead of fixing it at 24 was tried and REVERTED: timing one
    # dispatch to choose the count costs a dispatch of its own, and the first dispatch of a
    # variant is where this stack compiles it (75-363ms, see below). Measured on a 27B, that
    # took the warming phase from 48.9s to 109.1s -- more than twice as slow, for a change
    # meant to make it faster. The cost of finding out how expensive a shape is was larger
    # than what knowing it saved.
    REP = 24

    def bench():
        o = None
        for _ in range(REP):
            o = _ggml_run(xd, packed, type_name, int(K), int(N), small=state["kind"])
        o.get()

    _t0 = _t.perf_counter()
    try:
        return tune(key, ("narrow", "balanced", "compact", "shortk", None),
                    apply, bench, check=check,
                    default=fallback)
    finally:
        _TUNE_COST["tune_s"] += _t.perf_counter() - _t0
        _TUNE_COST["shapes"] += 1


_CHECKED = {}
_TUNE_COST = {"tune_s": 0.0, "build_s": 0.0, "check_s": 0.0,
              "shapes": 0, "variants": 0}


def tune_cost():
    """Where the load's tuning time went, since the process started. Timing only."""
    return dict(_TUNE_COST)


_KBUILD = {}


def _kernel_build():
    """A stamp that changes when the KERNELS change, not when a release is cut.

    A profile says what was true of a shader, so it has to stop being trusted when that
    shader changes -- and a version string does not track that: edit the WGSL, ship it, and
    a version-stamped profile would still be accepted. So the stamp is a digest of what
    actually generates the kernels: the source of the generator, and every format's decode
    fragment. Computed once.
    """
    if "v" in _KBUILD:
        return _KBUILD["v"]
    import hashlib
    import inspect
    h = hashlib.sha1()
    for fn in (_ggml_src, _ggml_name, _cfg_for, _ggml_parallel_src,
               _ggml_src_gl, _ggml_name_gl):
        try:
            h.update(inspect.getsource(fn).encode())
        except Exception:
            h.update(b"?")
    for name in sorted(_GGML_TYPES):
        dec, helper, vals, blk, tab = _GGML_TYPES[name]
        h.update(("%s|%s|%s|" % (name, vals, blk)).encode())
        h.update(str(dec).encode())
        h.update(str(helper).encode())
    # And every shader the generator pastes those fragments INTO. The templates are
    # module-level text, so a digest of the generator's own source does not move when one of
    # them is edited -- and a profile that outlives the shader it describes is a profile that
    # asserts what the previous version did. It cost exactly that: the dequant template
    # referred to a thread index it never declared, so the kernel would not compile for the
    # two formats with a staged codebook, and the stored profile went on recording the
    # verdict from the broken build after the template was fixed.
    g = globals()
    for name in sorted(g):
        v = g[name]
        if isinstance(v, str) and ("@compute" in v or "@group(" in v or "\nfn " in v):
            h.update(name.encode())
            h.update(v.encode())
    _KBUILD["v"] = h.hexdigest()[:16]
    return _KBUILD["v"]


def _count_dispatch_names(on):
    """Count dispatches per kernel name, or stop. Off by default and for a reason: the count
    is two dictionary operations inside `runKernel`, and on a path that is not replaying a
    recording -- a prefill, or WebGL -- that is once per dispatch, hundreds of times a token.
    A load-time diagnosis must not be a tax on every step."""
    try:
        from wgpy_backends.webgpu.platform import WebGPUPlatform
        WebGPUPlatform.count_names = bool(on)
    except Exception:
        pass


def kernel_profile():
    """What this device decided about these kernels, as plain data a caller may keep.

    Two questions are answered at load and both are answers about a DEVICE, not about a
    conversation: does this kernel variant compute the right thing here, and which thread
    shape is fastest here. Neither changes between loads. But they cost 36 shader compiles
    and 36 numerical checks to derive, which on a 27B was the whole of the warming phase --
    a build-time property, re-derived on the critical path, once per load.

    So they are offered as data. The SDK does not decide where they live: a caller that has
    somewhere to put them hands them back with `use_kernel_profile` on the next load and
    pays none of it; a caller that does not is exactly as it was.

    A profile is only valid for the build that produced it and the device that ran it. The
    build is stamped here; the DEVICE is the caller's to key on, because what identifies a
    GPU is a browser fact and this file has no business knowing it.
    """
    return {
        "build": _kernel_build(),
        "tuned": {"|".join(str(x) for x in k): v for k, v in _TUNED.items()},
        "gqa_tuned": {"|".join(str(x) for x in k): v for k, v in _GQA_TUNED.items()},
        "checked": {"|".join(str(x) for x in k): bool(v) for k, v in _CHECKED.items()},
        "dequant_ok": dict(_DEQ_OK),
        "calibrated": sorted("|".join(str(x) for x in k) for k in _CALIBRATED),
    }


def use_kernel_profile(profile):
    """Take back what `kernel_profile` returned. Returns how many entries were accepted.

    A profile from another build is ignored entirely rather than partially: a kernel changes
    with the code that generates it, and half-trusting one is how a stale verdict outlives
    the shader it was about.
    """
    if not isinstance(profile, dict):
        return 0
    if str(profile.get("build")) != _kernel_build():
        return 0
    n = 0
    for k, v in (profile.get("tuned") or {}).items():
        parts = k.split("|")
        if (len(parts) == 3 and parts[0] == ("decode_plan_v3" if parts[1] == "webgpu"
                                               else "decode_plan_v1")
                and parts[1] in ("webgpu", "webgl") and len(parts[2]) == 24
                and all(c in "0123456789abcdef" for c in parts[2])
                and isinstance(v, dict)):
            plan = v.get("plan")
            ms = v.get("median_ms")
            valid_ms = (ms is None or (type(ms) in (float, int)
                                    and 0 <= ms < float("inf")))
            if parts[1] == "webgl":
                if (valid_ms and isinstance(plan, (list, tuple)) and len(plan) == 2
                        and plan[0] in ("auto", "separate", "fused")
                        and plan[1] in ("full", "device")):
                    _TUNED[(parts[0], parts[1], parts[2])] = {
                        "plan": list(plan), "median_ms": ms}
                    n += 1
                continue
            group = ("auto", "fused", "separate", "fused:default",
                     "fused:balanced", "fused:compact", "fused:narrow")
            if (isinstance(plan, (list, tuple)) and len(plan) in (8, 9)
                    and plan[0] in ("composed", "fused")
                    and isinstance(plan[1], (list, tuple)) and len(plan[1]) <= 32
                    and all(x in ("auto", "stored", "dp4a") for x in plan[1])
                    and plan[2] in ("device", "full")
                    and plan[3] in group and plan[4] in group
                    and plan[5] in ("auto", "composed", "fused")
                    and plan[6] in ("auto", "separate", "fused")
                    and plan[7] in ("auto", None, "balanced", "compact",
                                    "narrow", "shortk")
                    and (len(plan) == 8 or
                         (isinstance(plan[8], (list, tuple)) and len(plan[8]) == 2
                          and all(route in ("auto", "balanced", "compact", "narrow")
                                  for route in plan[8])))):
                if valid_ms:
                    _TUNED[(parts[0], parts[1], parts[2])] = {
                        "plan": [plan[0], list(plan[1]), *plan[2:]],
                        "median_ms": ms,
                    }
                    n += 1
            continue
        if len(parts) == 4 and parts[0] == "ggml_shape":
            _TUNED[(parts[0], parts[1], int(parts[2]), int(parts[3]))] = v
            n += 1
        elif (len(parts) == 3 and parts[0] == "add_rmsnorm"
              and v in ("fused", "composed")):
            _TUNED[(parts[0], int(parts[1]), int(parts[2]))] = v
            n += 1
        elif (len(parts) == 3 and parts[0] == "greedy_chunk_v1" and parts[1] == "webgpu"
              and len(parts[2]) == 24 and all(c in "0123456789abcdef" for c in parts[2])
              and isinstance(v, dict) and v.get("count") in (0, 1, 2, 4)
              and isinstance(v.get("row", ""), str)):
            _TUNED[(parts[0], parts[1], parts[2])] = {
                "count": int(v["count"]), "row": str(v.get("row") or "host"),
                "median_ms": v.get("median_ms")}
            n += 1
        elif (len(parts) == 3 and parts[0] == "vocab_sample_full"
              and parts[2] == "webgpu" and v in ("js", "gpu")
              and parts[1].isdigit() and 0 < int(parts[1]) <= 1 << 24):
            _TUNED[(parts[0], int(parts[1]), parts[2])] = v
            n += 1
        elif (len(parts) == 6 and parts[:2] == ["weight_exec", "repeat_rows"]
              and parts[2] in ("webgpu", "webgl") and v in ("host", "device")):
            _TUNED[(parts[0], parts[1], parts[2], int(parts[3]), int(parts[4]),
                    int(parts[5]))] = v
            n += 1
        elif (len(parts) == 6 and parts[0] == "weight_exec" and v in _route_names()):
            _TUNED[(parts[0], parts[1], parts[2], int(parts[3]), int(parts[4]),
                    int(parts[5]))] = v
            n += 1
        elif (len(parts) == 8 and parts[0] in ("moe_prefill_route", "moe_prefill_route_js_v2", "moe_prefill_route_js_v3")
              and parts[1] in ("webgpu", "webgl")
              and parts[7] in ("cold", "warm") and v in ("host", "device")):
            _TUNED[(parts[0], parts[1], int(parts[2]), int(parts[3]),
                    int(parts[4]), parts[5], parts[6], parts[7])] = v
            n += 1
        elif (len(parts) == 5 and parts[0] in ("moe_prefill_api_v1", "moe_prefill_api_v2", "moe_prefill_api_v3")
              and parts[1] in ("webgpu", "webgl") and len(parts[2]) == 24
              and all(c in "0123456789abcdef" for c in parts[2])
              and parts[4] in ("cold", "warm") and v in ("host", "device")):
            try:
                bucket = int(parts[3])
            except ValueError:
                continue
            if bucket > 1 and bucket <= 1 << 20 and bucket & (bucket - 1) == 0:
                _TUNED[(parts[0], parts[1], parts[2], bucket, parts[4])] = v
                n += 1
        elif (len(parts) == 5 and parts[0] == "moe_reduce"
              and parts[1] in ("webgpu", "webgl")
              and v in ("composed", "fused")):
            _TUNED[(parts[0], parts[1], int(parts[2]), int(parts[3]),
                    int(parts[4]))] = v
            n += 1
        elif (len(parts) == 5 and parts[0] == "qk_norm_rope"
              and v in ("composed", "fused")):
            _TUNED[(parts[0], *(int(x) for x in parts[1:]))] = v
            n += 1
        elif (len(parts) == 5 and parts[0] == "kv_write_pair"
              and parts[4] in ("True", "False")
              and v in ("separate", "fused")):
            _TUNED[(parts[0], int(parts[1]), int(parts[2]), int(parts[3]),
                    parts[4] == "True")] = v
            n += 1
        elif (len(parts) == 5 and parts[0] == "embedding_row"
              and v in ("transposed", "compact")):
            _TUNED[(parts[0], parts[1], int(parts[2]), int(parts[3]),
                    int(parts[4]))] = v
            n += 1
        elif (len(parts) == 4 and parts[0] == "flash_tile"
              and isinstance(v, (list, tuple)) and len(v) == 2):
            tile = tuple(int(x) for x in v)
            hd = int(parts[3])
            if (tile in ((16, 8), (8, 16), (16, 16), (24, 8), (8, 32))
                    and 2 * tile[0] * hd + tile[1] * hd + tile[0] * tile[1] <= 8192):
                _TUNED[(parts[0], int(parts[1]), int(parts[2]), hd)] = tile
                n += 1
    for k, v in (profile.get("checked") or {}).items():
        parts = k.split("|")
        if len(parts) == 4:
            kind = None if parts[2] == "None" else parts[2]
            _CHECKED[(parts[0], int(parts[1]), kind, parts[3] == "True")] = bool(v)
            n += 1
    for k, v in (profile.get("dequant_ok") or {}).items():
        _DEQ_OK[str(k)] = bool(v)
        n += 1
    for k in (profile.get("calibrated") or ()):
        parts = str(k).split("|")
        if (len(parts) == 5 and parts[0] == "weight_exec"
                and parts[3].isdigit() and parts[4].isdigit()):
            _CALIBRATED.add((parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])))
            n += 1
    _NEAREST.clear()
    for k, v in (profile.get("gqa_tuned") or {}).items():
        parts = k.split("|")
        if len(parts) == 4 and type(v) is int and v in (4, 8, 16, 32):
            key = tuple(int(x) for x in parts)
            if all(x > 0 for x in key):
                _GQA_TUNED[key] = v
                n += 1
    return n


_GQA_TUNED = {}

def gqa_tune(nh, nkv, hd, n, candidates=(4, 8, 16, 32), rounds=5):
    """Pick the split factor for this device at this context length, by running them.

    Cheap enough to do at load: it times the attention kernel alone, not a whole step. The
    answer is remembered per (shape, context bucket) -- `n` is bucketed by powers of four,
    because the ranking moves with the order of magnitude of the scan and not with a token.

    NOT splitting was offered as a candidate and MEASURED WORSE, so the four factors stand.
    The reasoning for offering it was sound and wrong: a split costs a second dispatch per
    attention layer, so a 0.6B at a 66-token context runs 591 a step against 563 without one,
    and a scan of 66 keys does not obviously need the parallelism. Run both on the device:

        split      591 dispatches   9.89-10.02 ms a step
        no split   563 dispatches   11.40 ms a step

    Splitting wins even here, and by more than the dispatch it costs.

    Batching the timings 24 to a sync -- the rule this file states next to the other tuner --
    was tried with it and is NOT used here either: it is what made the tuner choose the
    slower option. Twenty-four identical attention dispatches back to back are not a decode
    step, where each one sits between the layer's other work, and the ranking it produces
    does not hold there. One dispatch per timing measures a sync it should not, and measures
    the right thing anyway; that is worth knowing rather than assuming from the other tuner.
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
    kvw = kv_cache_hd(hd)                  # the tuner must measure the real cache layout
    kc = Tensor(_empty((nkv, lmax, kvw)))
    vc = Tensor(_empty((nkv, lmax, kvw)))
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
        # One dispatch per timing, deliberately -- see the docstring. Batching them 24 to a
        # sync was tried, and it chose the option that is 14% slower in a real step.
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
    # Is this cache packed? Ask the BUFFER, not a global switch. A cache of halves is
    # exactly half as wide as the query it is scanned against, and reading that off the
    # shape here means a full-width cache can never be handed to the packed kernel however
    # the flag happens to be set -- there is more than one KV cache class in this file.
    f16 = (hd2 * 2 == hd) and hd % 2 == 0
    # `acc` is sized for hd <= 256, which covers every head dimension in use (128 and 256
    # are the common ones); anything larger keeps the general path.
    if T != 1 or not (hd == hd2 or f16) or hd > 256 or nh % nkv:
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
    # A cache of packed halves is read by a different kernel, so it gets a different name:
    # the two must be able to coexist (the tuner compares them, and a model whose head
    # dimension is odd stays on the f32 pair).
    sfx = "_f16" if f16 else ""
    kvsub = _f16_kv_source if f16 else (lambda src: src)
    if ("added" + sfx) not in _gqa_k:
        plat.addKernel("gqa_decode" + sfx, {"source": kvsub(_GQA_DECODE_WGSL),
                                            "bindingTypes": ["storage"]
                                            + ["read-only-storage"] * 5})
        _gqa_k["added" + sfx] = True
    # The split factor is baked into the shader, so each value is its OWN kernel -- named
    # for it, so several can exist at once and be compared on the device that will run them.
    if (_GQA_SPLIT, sfx) not in _gqa_k:
        sub = lambda src: src.replace("SPLITu", "%du" % _GQA_SPLIT)
        plat.addKernel("gqa_split_%d%s" % (_GQA_SPLIT, sfx),
                       {"source": sub(kvsub(_GQA_SPLIT_WGSL)),
                        "bindingTypes": ["storage"] + ["read-only-storage"] * 5})
        # The merge pass never touches the cache -- it reads the partials, which are fp32
        # whatever the cache holds -- so it is the same kernel either way.
        plat.addKernel("gqa_merge_%d%s" % (_GQA_SPLIT, sfx),
                       {"source": sub(_GQA_MERGE_WGSL),
                        "bindingTypes": ["storage", "read-only-storage",
                                         "read-only-storage"]})
        _gqa_k[(_GQA_SPLIT, sfx)] = True
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
        plat.runKernel({"name": "gqa_split_%d%s" % (_GQA_SPLIT, sfx),
                        "tensors": [part.buffer.buffer_id, qc.buffer.buffer_id,
                                    kcc.buffer.buffer_id, vcc.buffer.buffer_id,
                                    meta.buffer_id, cb.buffer_id],
                        "workGroups": {"x": nh * _GQA_SPLIT, "y": 1, "z": 1}})
        plat.runKernel({"name": "gqa_merge_%d%s" % (_GQA_SPLIT, sfx),
                        "tensors": [of.buffer.buffer_id, part.buffer.buffer_id,
                                    meta.buffer_id],
                        "workGroups": {"x": nh, "y": 1, "z": 1}})
        return Tensor(of.reshape(nh, 1, hd))
    plat.runKernel({"name": "gqa_decode" + sfx,
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

# ---- half-precision KV cache -------------------------------------------------------------
#
# Decode's cost splits cleanly in two, measured on a 0.6B: 9.05ms that does not depend on the
# conversation, plus 0.00223ms for every token already in it. The second term is READING THE
# CACHE, and nothing else: 28 layers x 2 x 8 kv-heads x 128 dims x 4 bytes is 224KB per
# context token, so at a context of 3632 one decode step streams 833MB. In 8.11ms. That is
# 102.7 GB/s.
#
# Which is not a kernel that needs improving -- it is faster than anything else here reaches.
# The generic fp32 matmul streams 62-79 GB/s on the same device, and the split factor is
# already chosen by measurement from (4, 8, 16, 32) at every context bucket, so there is no
# parallelism left on the table either. The only way to make this term smaller is to make it
# fewer bytes.
#
# So the cache holds halves. Two of them go in one u32 with `pack2x16float`, and the kernels
# that read it unpack with `unpack2x16float` -- both core WGSL, needing no device feature,
# and `unpackHalf2x16` is the GLSL ES 3.0 spelling for the same thing. The arithmetic is
# free at this ratio: we are moving 833MB and adding one instruction per two values.
#
# Nothing about the BACKEND changes. The cache is allocated as an ordinary f32 tensor with
# half as many elements, and the shaders declare that same binding `array<u32>` -- a storage
# binding is bytes, and what the host believes about their dtype never reaches the shader.
#
# What it costs is precision: K and V round to fp16 (~3 decimal digits) after the rope, which
# is what llama.cpp's cache does by default. Q, the accumulation and the softmax all stay
# fp32 -- only the stored value narrows.
_KV_F16 = True


def kv_f16():
    """Is the KV cache stored as halves? WebGPU only -- the WebGL path is unchanged."""
    return bool(_KV_F16) and _adam_backend_ready()


def kv_cache_hd(hd):
    """Elements per row to ALLOCATE for a cache of head-dimension `hd`.

    The `% 2` matters and is not defensive noise: every OTHER site that decides whether a
    cache is packed requires an even head dimension, so halving here without that test
    would allocate `hd // 2` rows for a model whose readers and writers still treat them as
    `hd` -- a cache too small for what is written into it, which reads back as a kernel that
    never ran rather than as an error.
    """
    hd = int(hd)
    return (hd // 2) if (kv_f16() and hd % 2 == 0) else hd



# Pack on the way in: same scatter, same indexing, two source values per stored word.
_KVWRITE_F16_WGSL = """@group(0) @binding(0) var<storage,read_write> cache: array<u32>;
@group(0) @binding(1) var<storage,read> src: array<f32>;
struct M { pos:u32, T:u32, NKV:u32, HD:u32, LMAX:u32, }
@group(0) @binding(2) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;                       // one lane per STORED word, not per value
  let hd2 = m.HD / 2u;
  let total = m.NKV * m.T * hd2;
  if (i >= total) { return; }
  let d = i % hd2;
  let t = (i / hd2) % m.T;
  let kv = i / (hd2 * m.T);
  let so = kv * m.T * m.HD + t * m.HD + d * 2u;
  cache[kv * m.LMAX * hd2 + (m.pos + t) * hd2 + d] =
      pack2x16float(vec2<f32>(src[so], src[so + 1u]));
}
"""

# K and V are independent destinations with identical scatter geometry.  WebGPU can write
# both in one dispatch without changing either stored value.  Keep this as a separate
# addressable operator rather than silently replacing ``kv_write``: its own ``auto`` route
# is device/shape measured, and a containing decoder may explicitly request either physical
# implementation when a different composition wins at that higher layer.
_KVWRITE_PAIR_WGSL = """@group(0) @binding(0) var<storage,read_write> kc: array<f32>;
@group(0) @binding(1) var<storage,read_write> vc: array<f32>;
@group(0) @binding(2) var<storage,read> ks: array<f32>;
@group(0) @binding(3) var<storage,read> vs: array<f32>;
struct M { pos:u32, T:u32, NKV:u32, HD:u32, LMAX:u32, }
@group(0) @binding(4) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let one = m.NKV * m.T * m.HD; let i = gid.x;
  if (i >= one * 2u) { return; }
  let which = i / one; let j = i - which * one;
  let hd = j % m.HD; let t = (j / m.HD) % m.T;
  let kv = j / (m.HD * m.T);
  let dst = kv * m.LMAX * m.HD + (m.pos + t) * m.HD + hd;
  if (which == 0u) { kc[dst] = ks[j]; } else { vc[dst] = vs[j]; }
}
"""

_KVWRITE_PAIR_F16_WGSL = """@group(0) @binding(0) var<storage,read_write> kc: array<u32>;
@group(0) @binding(1) var<storage,read_write> vc: array<u32>;
@group(0) @binding(2) var<storage,read> ks: array<f32>;
@group(0) @binding(3) var<storage,read> vs: array<f32>;
struct M { pos:u32, T:u32, NKV:u32, HD:u32, LMAX:u32, }
@group(0) @binding(4) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hd2 = m.HD / 2u; let one = m.NKV * m.T * hd2; let i = gid.x;
  if (i >= one * 2u) { return; }
  let which = i / one; let j = i - which * one;
  let d = j % hd2; let t = (j / hd2) % m.T; let kv = j / (hd2 * m.T);
  let so = kv * m.T * m.HD + t * m.HD + d * 2u;
  let dst = kv * m.LMAX * hd2 + (m.pos + t) * hd2 + d;
  if (which == 0u) {
    kc[dst] = pack2x16float(vec2<f32>(ks[so], ks[so + 1u]));
  } else {
    vc[dst] = pack2x16float(vec2<f32>(vs[so], vs[so + 1u]));
  }
}
"""

# Unpack a used span back to f32, for the prefill path. Prefill already copies the span it
# attends over (`_contig(K.data[:, :end, :])`), so widening happens inside a copy that was
# there anyway and the three prefill attention kernels never learn about any of this.
_KVREAD_F16_WGSL = """@group(0) @binding(0) var<storage,read_write> outp: array<f32>;
@group(0) @binding(1) var<storage,read> cache: array<u32>;
struct M { pos:u32, T:u32, NKV:u32, HD:u32, LMAX:u32, }
@group(0) @binding(2) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let hd2 = m.HD / 2u;
  let total = m.NKV * m.T * hd2;
  if (i >= total) { return; }
  let d = i % hd2;
  let t = (i / hd2) % m.T;
  let kv = i / (hd2 * m.T);
  let v = unpack2x16float(cache[kv * m.LMAX * hd2 + t * hd2 + d]);
  let oo = kv * m.T * m.HD + t * m.HD + d * 2u;
  outp[oo] = v.x;
  outp[oo + 1u] = v.y;
}
"""
_kvf16 = {"w": False, "r": False}


def kv_unpack(cache, end, nkv, hd, lmax):
    """The first `end` rows of a packed cache, as a fresh f32 (nkv, end, hd) ndarray."""
    plat = _adam_kernel["platform"]
    if not _kvf16["r"]:
        plat.addKernel("kv_read_f16", {"source": _KVREAD_F16_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "read-only-storage"]})
        _kvf16["r"] = True
    end = int(end); nkv = int(nkv); hd = int(hd)
    out = _empty((nkv, end, hd))
    meta_id = _adam_kernel["make_meta"](
        (0, end, nkv, hd, int(lmax)), "u4,u4,u4,u4,u4").buffer_id
    total = nkv * end * (hd // 2)
    plat.runKernel({"name": "kv_read_f16",
        "tensors": [out.buffer.buffer_id, cache.buffer.buffer_id, meta_id],
        "workGroups": {"x": (total + 63) // 64, "y": 1, "z": 1}})
    return out


def _f16_kv_source(src):
    """The same attention kernel, reading a cache of packed halves instead of floats.

    Derived rather than copied: the two kernels that scan the cache are long, subtle and
    identical in the parts that touch it, and a second hand-maintained copy of each would
    drift from the original the first time anyone fixed a bug in one of them. Every
    substitution asserts, so a fragment that stops matching is a build error here and not a
    kernel that silently reads the wrong bytes.
    """
    subs = [
        # The bindings are the same bytes; only what the shader calls them changes.
        ("var<storage,read> kc: array<f32>;",
         "var<storage,read> kc: array<u32>;"),
        ("var<storage,read> vc: array<f32>;",
         "var<storage,read> vc: array<u32>;"),
        # Half the elements per row, so half the stride.
        ("  let ko = kv * gm.lmax * gm.hd;",
         "  let ko = kv * gm.lmax * (gm.hd / 2u);"),
        # q . k, two dimensions per stored word.
        ("""      let kb = ko + s * gm.hd;
      for (var i: u32 = 0u; i < gm.hd; i = i + 1u) {
        dd = dd + q[qo + i] * kc[kb + i];
      }""",
         """      let hd2 = gm.hd / 2u;
      let kb = ko + s * hd2;
      for (var i: u32 = 0u; i < hd2; i = i + 1u) {
        let kk = unpack2x16float(kc[kb + i]);
        dd = dd + q[qo + i * 2u] * kk.x + q[qo + i * 2u + 1u] * kk.y;
      }"""),
        # The weighted sum of values: one lane per stored word, so it carries two
        # accumulators. `acc` stays fp32 and full width -- only the STORED value narrows.
        ("""    for (var d: u32 = t; d < gm.hd; d = d + 128u) {
      var o: f32 = acc[d] * corr;
      for (var pp: u32 = 0u; pp < cnt; pp = pp + 1u) {
        o = o + sc[pp] * vc[ko + (base + pp) * gm.hd + d];
      }
      acc[d] = o;
    }""",
         """    let hd2v = gm.hd / 2u;
    for (var d: u32 = t; d < hd2v; d = d + 128u) {
      var o0: f32 = acc[d * 2u] * corr;
      var o1: f32 = acc[d * 2u + 1u] * corr;
      for (var pp: u32 = 0u; pp < cnt; pp = pp + 1u) {
        let vv = unpack2x16float(vc[ko + (base + pp) * hd2v + d]);
        o0 = o0 + sc[pp] * vv.x;
        o1 = o1 + sc[pp] * vv.y;
      }
      acc[d * 2u] = o0;
      acc[d * 2u + 1u] = o1;
    }"""),
    ]
    for old, rep in subs:
        if src.count(old) != 1:
            raise RuntimeError("f16 KV: fragment matched %d times, expected 1:\n%s"
                               % (src.count(old), old[:80]))
        src = src.replace(old, rep, 1)
    return src


_kvw = {"added": False}
_kvwp = {"f32": False, "f16": False}


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
    # Same rule as the reader: the cache's own width says whether it holds halves.
    packed = (int(cache.shape[-1]) * 2 == int(hd)) and int(hd) % 2 == 0
    name = "kv_write_f16" if packed else "kv_write"
    if packed and not _kvf16["w"]:
        plat.addKernel(name, {"source": _KVWRITE_F16_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "read-only-storage"]})
        _kvf16["w"] = True
    if not packed and not _kvw["added"]:
        plat.addKernel("kv_write", {"source": _KVWRITE_WGSL,
            "bindingTypes": ["storage", "read-only-storage", "read-only-storage"]})
        _kvw["added"] = True
    if ctl is not None:
        meta_id = ctl.buffer.buffer_id                 # persistent ndarray meta (capture)
    else:
        meta_id = _adam_kernel["make_meta"](
            (int(pos), int(T), int(nkv), int(hd), int(lmax)), "u4,u4,u4,u4,u4").buffer_id
    total = nkv * T * (hd // 2 if packed else hd)
    plat.runKernel({"name": name,
        "tensors": [cache.buffer.buffer_id, src.buffer.buffer_id, meta_id],
        "workGroups": {"x": (total + 63) // 64, "y": 1, "z": 1}})
    return cache


def _kv_write_pair_fused(kcache, vcache, ksrc, vsrc, pos, T, nkv, hd, lmax,
                         ctl=None):
    """One WebGPU dispatch for two exactly equivalent KV scatter writes."""
    plat = _adam_kernel["platform"]
    packed = ((int(kcache.shape[-1]) * 2 == int(hd))
              and (int(vcache.shape[-1]) * 2 == int(hd)) and int(hd) % 2 == 0)
    name = "kv_write_pair_f16" if packed else "kv_write_pair"
    flag = "f16" if packed else "f32"
    if not _kvwp[flag]:
        plat.addKernel(name, {
            "source": _KVWRITE_PAIR_F16_WGSL if packed else _KVWRITE_PAIR_WGSL,
            "bindingTypes": ["storage", "storage", "read-only-storage",
                             "read-only-storage", "read-only-storage"]})
        _kvwp[flag] = True
    meta_id = (ctl.buffer.buffer_id if ctl is not None else
               _adam_kernel["make_meta"](
                   (int(pos), int(T), int(nkv), int(hd), int(lmax)),
                   "u4,u4,u4,u4,u4").buffer_id)
    one = int(nkv) * int(T) * (int(hd) // 2 if packed else int(hd))
    plat.runKernel({"name": name,
        "tensors": [kcache.buffer.buffer_id, vcache.buffer.buffer_id,
                    ksrc.buffer.buffer_id, vsrc.buffer.buffer_id, meta_id],
        "workGroups": {"x": (2 * one + 63) // 64, "y": 1, "z": 1}})
    return kcache, vcache


def _kv_pair_auto(nkv, hd, lmax, packed):
    """Device-local exactness gate and paired timing for the KV layer operation."""
    key = ("kv_write_pair", int(nkv), int(hd), int(lmax), bool(packed))
    if key in _TUNED:
        return _TUNED[key]
    import time as _t
    try:
        width = int(hd) // 2 if packed else int(hd)
        shape = (int(nkv), int(lmax), width)
        src_shape = (int(nkv), 1, int(hd))
        # Deterministic non-special values exercise both half packing lanes as well as sign.
        base = (np.arange(np.prod(src_shape), dtype=np.float32).reshape(src_shape) % 37
                - 18.0) / 11.0
        ks = xp.asarray(base); vs = xp.asarray(base * np.float32(-0.625) + np.float32(0.125))

        def fresh():
            return xp.asarray(np.zeros(shape, np.float32)), xp.asarray(np.zeros(shape, np.float32))

        ak, av = fresh(); bk, bv = fresh()
        kv_write(ak, ks, 3, 1, nkv, hd, lmax)
        kv_write(av, vs, 3, 1, nkv, hd, lmax)
        _kv_write_pair_fused(bk, bv, ks, vs, 3, 1, nkv, hd, lmax)
        aa, ab = np.asarray(ak.get()), np.asarray(av.get())
        ba, bb = np.asarray(bk.get()), np.asarray(bv.get())
        if not (np.array_equal(aa.view(np.uint32), ba.view(np.uint32)) and
                np.array_equal(ab.view(np.uint32), bb.view(np.uint32))):
            raise RuntimeError("fused KV pair differs from the separate reference for %r" % (key,))

        candidates = ("separate", "fused")
        samples = {name: [] for name in candidates}

        def bench(name):
            ck, cv = fresh(); t0 = _t.perf_counter()
            for _ in range(16):
                if name == "fused":
                    _kv_write_pair_fused(ck, cv, ks, vs, 3, 1, nkv, hd, lmax)
                else:
                    kv_write(ck, ks, 3, 1, nkv, hd, lmax)
                    kv_write(cv, vs, 3, 1, nkv, hd, lmax)
            cv.get()
            return (_t.perf_counter() - t0) / 16.0

        bench("separate"); bench("fused")
        for r in range(9):
            order = candidates if not (r & 1) else tuple(reversed(candidates))
            for name in order:
                samples[name].append(bench(name))
        chosen = _measured_choice(samples, candidates, default="separate")
    except Exception as exc:
        raise RuntimeError("KV pair auto candidate failed for %r" % (key,)) from exc
    _TUNED[key] = chosen
    return chosen


def kv_write_pair(kcache, vcache, ksrc, vsrc, pos, T, nkv, hd, lmax, ctl=None,
                  execution="auto"):
    """Write K and V with one common layer contract on WebGPU and WebGL.

    ``auto`` means the locally fastest exact route.  ``separate`` and ``fused`` remain
    addressable so a containing decoder can choose its own best composition.  WebGL cannot
    portably render to two independently sampled cache textures in this abstraction, so its
    nearest equivalent layer implementation composes the same two writes.
    """
    if execution not in ("auto", "separate", "fused"):
        raise ValueError("KV pair execution must be auto, separate, or fused")
    if _webgl_ready() and not _adam_backend_ready():
        return (kv_write(kcache, ksrc, pos, T, nkv, hd, lmax, ctl=ctl),
                kv_write(vcache, vsrc, pos, T, nkv, hd, lmax, ctl=ctl))
    packed = ((int(kcache.shape[-1]) * 2 == int(hd)) and int(hd) % 2 == 0)
    route = (_kv_pair_auto(nkv, hd, lmax, packed)
             if execution == "auto" else execution)
    if route == "fused":
        return _kv_write_pair_fused(kcache, vcache, ksrc, vsrc, pos, T, nkv, hd,
                                    lmax, ctl=ctl)
    return (kv_write(kcache, ksrc, pos, T, nkv, hd, lmax, ctl=ctl),
            kv_write(vcache, vsrc, pos, T, nkv, hd, lmax, ctl=ctl))


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


# Attention in query chunks, using the batched path's fast matmul.
#
# The flash kernel below never writes the score matrix, which is the right instinct and the
# wrong trade on this backend: hand-written, it runs at 94 GFLOPS while the fp32 matmul next
# to it does 1850. Materialising a CHUNK of scores at a time keeps the memory bounded and
# spends the arithmetic where the machine is fast. Measured per layer, 0.6B, T=2816:
# flash 179ms, this 49ms.
#
# Two details carry most of it. The matmuls are 2-D, one per KV head, because the 3-D
# batched form is a different and much worse kernel -- 124 GFLOPS against 796 for the same
# work split into eight 2-D calls. And the key extent is rounded up to a multiple of 32 for
# the same alignment reason the prefill rounds its rows: the columns past the diagonal are
# masked to zero by the softmax anyway.
#
# Chunks are joined with the GPU concatenate, never by assigning into a preallocated output:
# WgPy's __setitem__ goes through the host, which is 14ms for 11MB.
_ATTN_CHUNK = 1024
# Below this many queries the flash kernel wins -- it is one dispatch against dozens, and at
# short lengths the dispatches are the cost. Measured: T=256 flash 5.0ms / chunked 6.0ms,
# T=512 flash 9.6 / chunked 6.2.
_ATTN_CHUNK_MIN_T = 384


def chunked_attention(q, k, v, start=0, scale=None, chunk=None):
    """Causal grouped-query attention, one query chunk at a time."""
    nh, T, hd = q.shape
    nkv, S, _ = k.shape
    rep = nh // nkv
    if scale is None:
        scale = 1.0 / (float(hd) ** 0.5)
    CH = int(chunk or _ATTN_CHUNK)
    # Padding the key extent to a multiple of 64 here does NOT pay, and the reason it looks
    # like it should is instructive.
    #
    # `lim` below is clamped to S, so on the LAST chunk of any prefill the clamp cancels the
    # rounding exactly -- and counted in a browser, 896 of a generation's matmul dispatches
    # were landing on the generic kernel because of it (n % 64 == 1 on one matmul, k % 4 == 1
    # on the other). Padding k and v to a multiple of 64 once per call does move all 896 onto
    # the tiled kernel; that was verified, `matmul` went to zero. And in isolation those 896
    # matmuls measured 557ms against 281ms, a clean 2x.
    #
    # End to end it is WORSE. At a 2385-token prompt, three samples each way, not overlapping:
    #
    #     padded     7.88  7.28  7.84 s to first token
    #     unpadded   6.43  6.42  6.51 s
    #
    # About 1.3s slower, for matmuls that are twice as fast. The copy the padding needs -- two
    # `cat`s of the whole K and V per layer, plus a wider `_contig` slice on every head -- costs
    # more than the kernel saves. Measured at 500 tokens too, where it is a wash.
    #
    # So the clamp stays. The dispatches it sends to the generic kernel are a real cost, but
    # the way to get them back is not to copy K and V to buy alignment.
    qg = q.reshape(nkv, rep * T, hd)
    kt = transpose_last2(k)
    parts = []
    for h in range(nkv):
        khT = Tensor(_contig(kt.data[h]))                  # (hd, S)
        vh = Tensor(_contig(v.data[h]))                    # (S, hd)
        for r in range(rep):
            for i0 in range(0, T, CH):
                i1 = min(T, i0 + CH)
                # Everything the LAST query of the chunk can see, rounded up for the
                # matmul. SIXTY-FOUR, not thirty-two: the alignment the fast path wants is
                # not the same on both axes. Rows need a multiple of 32, but the key extent
                # is the matmul's N and that wants 64 -- measured at M=1024 K=128, N=1056
                # runs at 45 GFLOPS and N=1088 at 193, while N=2848 gives 193 against 683 at
                # N=2880. Rounding to 32 left every continued prefill, and the last chunk of
                # every fresh one, on the slow side: a prompt resumed after 30 cached tokens
                # took 6.5s where the same work from scratch took 3.9s.
                lim = min(S, ((start + i1 + 63) // 64) * 64)
                qc = Tensor(_contig(qg.data[h][r * T + i0:r * T + i1]))
                sc = qc @ Tensor(_contig(khT.data[:, :lim]))
                pw = Tensor(_fused_causal_softmax(sc.data, i1 - i0, start + i0, scale))
                parts.append(pw @ Tensor(_contig(vh.data[:lim])))
    # The chunks were produced in (kv, rep, query) order, which is exactly the row order of
    # the grouped layout, so joining them is the answer with no permutation.
    return Tensor(cat(parts, axis=0).data.reshape(nh, T, hd))


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
    """Softmax over the last axis: each row's max and exp-sum once, then a normalise --
    or, where the rows-times-width^2 work is small, one pass (see `_one_pass`).

    The one-pass kernel recomputes both for every element of its row --
    two full passes over the row per output, so a (36, 173, 173) attention softmax did
    ~373M fetches and ~186M exps where ~3M suffice; measured on WebGL at 5.9 ms a draw. The
    statistics run the same loops as the one-pass form; measured equal bit for bit at every
    shape in the table above `_one_pass` (unlike LayerNorm, see its note). The output was also host-allocated with `_zeros`, which uploads a zeroed
    copy every call (4.3 MB here, 24 times a request) to a buffer the kernel overwrites in
    full; it is `_empty` now, as the note on `_empty` already says softmax outputs should be.
    """
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1])
    rows = int(xd.size) // width
    if _one_pass(rows, width):
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
        s = _empty(xd.shape)
        plat.runKernel({"name": name,
            "inputs": [{"name": "tex_in", "id": xd.buffer.buffer_id}],
            "output": s.buffer.buffer_id,
            "uniforms": [{"name": "_ka_tex_output_texture_w", "value": s.buffer.texture_shape.width, "type": "int"}]})
        return s
    stats_name, norm_name = f"softmax_stats_{width}", f"softmax_norm_{width}"
    if stats_name not in _softmax_gl["added"]:
        head = (f"#version 300 es\nprecision highp float; precision highp int; "
                f"precision highp sampler2D;\n#define WIDTH {width}\n"
                f"uniform int _ka_tex_output_texture_w; uniform sampler2D tex_in;\n"
                f"out float fragColor;\n{_GL_FETCH}\n")
        # Fragment 2r writes row r's max, 2r+1 its sum of exp(x - max).
        plat.addKernel(stats_name, {"source": head + f"""uniform int TOTAL;
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (idx >= TOTAL) {{ return; }}
  int row = idx / 2; int base = row * WIDTH;
  float mx = fetch(tex_in, base);
  for (int j = 1; j < WIDTH; j++) {{ float v = fetch(tex_in, base + j); if (v > mx) mx = v; }}
  if (idx - row * 2 == 0) {{ fragColor = mx; return; }}
  float sm = 0.0;
  for (int j = 0; j < WIDTH; j++) {{ sm += exp(fetch(tex_in, base + j) - mx); }}
  fragColor = sm;
}}
"""})
        plat.addKernel(norm_name, {"source": head + f"""uniform sampler2D tex_stats;
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / WIDTH;
  float mx = fetch(tex_stats, row * 2); float sm = fetch(tex_stats, row * 2 + 1);
  fragColor = exp(fetch(tex_in, idx) - mx) / sm;
}}
"""})
        _softmax_gl["added"].add(stats_name)
    stats = _empty((rows, 2))
    plat.runKernel({"name": stats_name,
        "inputs": [{"name": "tex_in", "id": xd.buffer.buffer_id}],
        "output": stats.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": stats.buffer.texture_shape.width, "type": "int"},
                     {"name": "TOTAL", "value": rows * 2, "type": "int"}]})
    s = _empty(xd.shape)
    plat.runKernel({"name": norm_name,
        "inputs": [{"name": "tex_in", "id": xd.buffer.buffer_id},
                   {"name": "tex_stats", "id": stats.buffer.buffer_id}],
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


# A decision head does not need the whole probability vector to decide whether it should
# answer.  It needs four numbers: the largest probability, its margin over the runner-up,
# normalized entropy, and the option count.  Reading the logits to Python to make those four
# numbers forces the encoder/scorer queue to finish, then uploads them again for the action
# head.  This one-workgroup reduction keeps that boundary on the device.
_DECISION_FEATURES_WGSL = """@group(0) @binding(0)
var<storage,read> x: array<f32>;
@group(0) @binding(1)
var<storage,read_write> y: array<f32>;
struct M { width: u32, k: u32, }
@group(0) @binding(2)
var<storage,read> m: M;
@compute @workgroup_size(1)
fn main() {
  var mx: f32 = x[0];
  var t1: f32 = x[0];
  var t2: f32 = -1e30;
  for (var j: u32 = 1u; j < m.width; j = j + 1u) {
    let v = x[j];
    mx = max(mx, v);
    if (v > t1) { t2 = t1; t1 = v; }
    else if (v > t2) { t2 = v; }
  }
  var den: f32 = 0.0;
  for (var j: u32 = 0u; j < m.width; j = j + 1u) {
    den = den + exp(x[j] - mx);
  }
  let p1 = exp(t1 - mx) / den;
  var p2: f32 = 0.0;
  if (m.width > 1u) { p2 = exp(t2 - mx) / den; }
  var ent: f32 = 0.0;
  for (var j: u32 = 0u; j < m.width; j = j + 1u) {
    let pp = exp(x[j] - mx) / den;
    ent = ent - pp * log(max(pp, 1e-9));
  }
  y[0] = p1;
  y[1] = p1 - p2;
  y[2] = ent / log(f32(m.k));
  y[3] = f32(m.k) / 255.0;
}
"""
_decision_features_kernel = {"added": False}

_DECISION_FEATURES_GLSL_BODY = """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  float mx = Xf(0);
  float top1 = Xf(0);
  float top2 = -1e30;
  for (int j = 1; j < u_width; j++) {
    float v = Xf(j);
    mx = max(mx, v);
    if (v > top1) { top2 = top1; top1 = v; }
    else if (v > top2) { top2 = v; }
  }
  float den = 0.0;
  for (int j = 0; j < u_width; j++) { den += exp(Xf(j) - mx); }
  float p1 = exp(top1 - mx) / den;
  float p2 = u_width > 1 ? exp(top2 - mx) / den : 0.0;
  float ent = 0.0;
  for (int j = 0; j < u_width; j++) {
    float p = exp(Xf(j) - mx) / den;
    ent -= p * log(max(p, 1e-9));
  }
  if (i == 0) fragColor = p1;
  else if (i == 1) fragColor = p1 - p2;
  else if (i == 2) fragColor = ent / log(float(u_k));
  else fragColor = float(u_k) / 255.0;
}
"""


def decision_features(logits, option_count=None):
    """Return the four action-head features for one row of decision logits.

    The GPU backends perform the reduction without reading logits into Python and uploading
    the four features again.  On WebGL that boundary also forced all pending encoder/head
    commands to finish before the action head could even be queued.
    """
    width = int(logits.data.size)
    if width < 1:
        raise ValueError("decision logits need at least one option")
    norm_k = max(2, int(width if option_count is None else option_count))
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if not _decision_features_kernel["added"]:
            plat.addKernel("decision_features", {
                "source": _DECISION_FEATURES_WGSL,
                "bindingTypes": ["read-only-storage", "storage", "read-only-storage"],
            })
            _decision_features_kernel["added"] = True
        out = _empty((1, 4))
        meta = _adam_kernel["make_meta"]((width, norm_k), "u4,u4")
        plat.runKernel({
            "name": "decision_features",
            "tensors": [_contig(logits.data).buffer.buffer_id, out.buffer.buffer_id,
                        meta.buffer_id],
            "workGroups": {"x": 1, "y": 1, "z": 1},
        })
        return Tensor(out)
    if _webgl_ready():
        out = _empty((1, 4))
        source = _gl_head([("tex_x", "Xf")], ("u_width", "u_k", "u_n")) + _DECISION_FEATURES_GLSL_BODY
        return Tensor(_gl_run("decision_features_gl", source,
                              [("tex_x", _contig(logits.data))], out,
                              [("u_width", width), ("u_k", norm_k), ("u_n", 4)]))
    z = np.asarray(logits.numpy(), dtype=np.float32).reshape(-1)
    p = np.exp(z - z.max()); p = p / p.sum()
    top = np.sort(p)[::-1][:2]
    p1 = float(top[0]); p2 = float(top[1]) if len(top) > 1 else 0.0
    ent = float(-(p * np.log(np.clip(p, 1e-9, 1.0))).sum() / np.log(norm_k))
    return Tensor(np.asarray([[p1, p1 - p2, ent, norm_k / 255.0]], np.float32))


_DECISION_FEATURES_MANY_WGSL = """@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> counts: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
struct M { rows: u32, width: u32, }
@group(0) @binding(3) var<storage,read> m: M;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= m.rows) { return; }
  let k = u32(counts[row]);
  let base = row * m.width;
  var mx = x[base];
  var top1 = x[base];
  var top2 = -1e30;
  for (var j = 1u; j < k; j = j + 1u) {
    let v = x[base + j];
    mx = max(mx, v);
    if (v > top1) { top2 = top1; top1 = v; }
    else if (v > top2) { top2 = v; }
  }
  var den = 0.0;
  for (var j = 0u; j < k; j = j + 1u) { den = den + exp(x[base + j] - mx); }
  let p1 = exp(top1 - mx) / den;
  var p2 = 0.0;
  if (k > 1u) { p2 = exp(top2 - mx) / den; }
  var ent = 0.0;
  for (var j = 0u; j < k; j = j + 1u) {
    let p = exp(x[base + j] - mx) / den;
    ent = ent - p * log(max(p, 1e-9));
  }
  let norm_k = max(2u, k);
  y[row * 4u] = p1;
  y[row * 4u + 1u] = p1 - p2;
  y[row * 4u + 2u] = ent / log(f32(norm_k));
  y[row * 4u + 3u] = f32(norm_k) / 255.0;
}
"""

_DECISION_FEATURES_MANY_GLSL = """
void main() {
  int i = _idx();
  if (i >= u_rows * 4) { fragColor = 0.0; return; }
  int row = i / 4;
  int feature = i - row * 4;
  int k = int(Cf(row) + 0.5);
  int base = row * u_width;
  float mx = Xf(base);
  float top1 = mx;
  float top2 = -1e30;
  for (int j = 1; j < k; j++) {
    float v = Xf(base + j);
    mx = max(mx, v);
    if (v > top1) { top2 = top1; top1 = v; }
    else if (v > top2) { top2 = v; }
  }
  float den = 0.0;
  for (int j = 0; j < k; j++) { den += exp(Xf(base + j) - mx); }
  float p1 = exp(top1 - mx) / den;
  float p2 = k > 1 ? exp(top2 - mx) / den : 0.0;
  float ent = 0.0;
  for (int j = 0; j < k; j++) {
    float p = exp(Xf(base + j) - mx) / den;
    ent -= p * log(max(p, 1e-9));
  }
  float norm_k = float(max(2, k));
  if (feature == 0) fragColor = p1;
  else if (feature == 1) fragColor = p1 - p2;
  else if (feature == 2) fragColor = ent / log(norm_k);
  else fragColor = norm_k / 255.0;
}
"""
_decision_features_many_added = {"gpu": False}


def decision_features_many(logits, option_counts):
    """Action features for all question rows in one device operation.

    Only the first ``option_counts[row]`` logits participate; padding never changes a
    question's softmax or entropy. The workgroup count follows the requested batch size,
    leaving actual concurrency to the GPU scheduler rather than a fixed question limit.
    """
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("decision logits must have shape (questions, options)")
    rows, width = map(int, logits.shape)
    counts = tuple(int(k) for k in option_counts)
    if len(counts) != rows or any(k < 1 or k > width for k in counts):
        raise ValueError("one valid option count is required per question row")
    if _adam_backend_ready() or _webgl_ready():
        count_data = xp.asarray(np.asarray(counts, dtype=np.float32))
        out = _empty((rows, 4))
        if _adam_backend_ready():
            plat = _adam_kernel["platform"]
            if not _decision_features_many_added["gpu"]:
                plat.addKernel("decision_features_many", {
                    "source": _DECISION_FEATURES_MANY_WGSL,
                    "bindingTypes": ["read-only-storage", "read-only-storage",
                                     "storage", "read-only-storage"],
                })
                _decision_features_many_added["gpu"] = True
            meta = _adam_kernel["make_meta"]((rows, width), "u4,u4")
            plat.runKernel({"name": "decision_features_many",
                "tensors": [_contig(logits.data).buffer.buffer_id,
                            count_data.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
                "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
        else:
            source = _gl_head([("tex_x", "Xf"), ("tex_counts", "Cf")],
                              ("u_rows", "u_width")) + _DECISION_FEATURES_MANY_GLSL
            _gl_run("decision_features_many_gl", source,
                    [("tex_x", _contig(logits.data)), ("tex_counts", count_data)],
                    out, [("u_rows", rows), ("u_width", width)])
        return Tensor(out)
    values = np.asarray(logits.numpy(), np.float32)
    features = np.empty((rows, 4), np.float32)
    for row, k in enumerate(counts):
        z = values[row, :k]
        p = np.exp(z - z.max()); p = p / p.sum()
        top = np.sort(p)[::-1][:2]
        p1 = float(top[0]); p2 = float(top[1]) if k > 1 else 0.0
        norm_k = max(2, k)
        ent = float(-(p * np.log(np.clip(p, 1e-9, 1.0))).sum() / np.log(norm_k))
        features[row] = (p1, p1 - p2, ent, norm_k / 255.0)
    return Tensor(features)


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


def _ln_rows_src(nv):
    """LayerNorm with one 64-thread workgroup per row, optionally of a sum.

    The row-per-THREAD kernel above gives a 519-row pass 519 threads -- eight workgroups on a
    GPU that wants hundreds -- and each walks its 768 values three times: 130 us a call, about
    25 GB/s. Here a row is 64 threads holding `nv` vec4s each in registers; mean and variance
    are two exact passes over those registers, reduced through workgroup memory. With `add`
    the row is `x + y` and that sum is written too: it is the residual stream the next block
    adds to, so the add costs no pass of its own."""
    L = ["""
@group(0) @binding(0) var<storage,read> xin: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> yin: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> gam: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> bet: array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> sout: array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> lout: array<vec4<f32>>;
struct LMeta { rows: u32, W4: u32, add: u32, eps: f32, }
@group(0) @binding(6) var<storage,read> lm: LMeta;
var<workgroup> red: array<f32, 64>;
fn total(t: u32, v: f32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var w = 32u; w > 0u; w = w >> 1u) {
    if (t < w) { red[t] = red[t] + red[t + w]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x + wg.y * 65535u;
  if (row >= lm.rows) { return; }
  let t = lid.x;
  let base = row * lm.W4;
  let add = lm.add != 0u;
  var s = 0.0;"""]
    for k in range(nv):
        L.append("""  var v%(k)d = vec4<f32>();
  let j%(k)d = t + %(o)du;
  if (j%(k)d < lm.W4) {
    v%(k)d = xin[base + j%(k)d];
    if (add) { v%(k)d = v%(k)d + yin[base + j%(k)d]; sout[base + j%(k)d] = v%(k)d; }
    s = s + (v%(k)d.x + v%(k)d.y) + (v%(k)d.z + v%(k)d.w);
  }""" % dict(k=k, o=64 * k))
    L.append("  let n = f32(lm.W4 * 4u);")
    L.append("  let mu = total(t, s) / n;")
    L.append("  var q = 0.0;")
    for k in range(nv):
        L.append("  if (j%(k)d < lm.W4) { let d = v%(k)d - vec4<f32>(mu); q = q + dot(d, d); }" % dict(k=k))
    L.append("  let inv = 1.0 / sqrt(total(t, q) / n + lm.eps);")
    for k in range(nv):
        L.append("  if (j%(k)d < lm.W4) { lout[base + j%(k)d] = (v%(k)d - vec4<f32>(mu)) * inv"
                 " * gam[j%(k)d] + bet[j%(k)d]; }" % dict(k=k))
    L.append("}")
    return "\n".join(L)


_ln_rows_added = set()
_ln_rows_sink = {}


def _wgpu_ln_rows(xd, yd, gd, bd, eps):
    """`(x + y, LN(x + y))` -- or just `LN(x)` with `yd` None -- by `_ln_rows_src`. None for a
    width the kernel does not take (not a multiple of 4, or past 8192)."""
    width = int(xd.shape[-1])
    if width % 4 or width > 8192:
        return None
    rows = int(xd.size) // width
    W4 = width // 4
    nv = (W4 + 63) // 64
    name = "ln_rows_%d" % nv
    plat = _adam_kernel["platform"]
    if name not in _ln_rows_added:
        plat.addKernel(name, {"source": _ln_rows_src(nv),
                              "bindingTypes": ["read-only-storage"] * 4 + ["storage", "storage",
                                                                           "read-only-storage"]})
        _ln_rows_added.add(name)
    out = _empty(xd.shape)
    if yd is not None:
        ssum = _empty(xd.shape)
    else:
        # Never written without `add`, but a binding needs a buffer -- and not `out`: one
        # buffer bound twice as writable fails validation and the pass computes nothing.
        if "sink" not in _ln_rows_sink:
            _ln_rows_sink["sink"] = _empty((4,))
        ssum = _ln_rows_sink["sink"]
    meta = _adam_kernel["make_meta"]((rows, W4, 1 if yd is not None else 0, float(eps)),
                                     "u4,u4,u4,f4")
    plat.runKernel({"name": name,
                    "tensors": [xd.buffer.buffer_id, (yd if yd is not None else xd).buffer.buffer_id,
                                gd.buffer.buffer_id, bd.buffer.buffer_id, ssum.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": min(rows, 65535), "y": (rows + 65534) // 65535, "z": 1}})
    return (ssum if yd is not None else None), out


def add_layernorm(x, y, gamma, beta, eps=1e-5):
    """`s = x + y` and `layernorm(s)`, both returned: the residual stream and the next block's
    input. One dispatch on WebGPU; elsewhere, or with a gradient, the two operations."""
    if (_adam_backend_ready() and not (x.requires_grad or y.requires_grad
                                       or gamma.requires_grad or beta.requires_grad)
            and tuple(x.shape) == tuple(y.shape)):
        got = _wgpu_ln_rows(_contig(x.data), _contig(y.data), _contig(gamma.data),
                            _contig(beta.data), eps)
        if got is not None:
            return Tensor(got[0]), Tensor(got[1])
    s = x + y
    return s, layernorm(s, gamma, beta, eps)


def _wgpu_ln_fwd(xd, gd, bd, eps):
    plat = _adam_kernel["platform"]
    if not _ln_wgpu["added"]:
        # Registered here because `_wgpu_ln_bwd` runs them without registering.
        rw = ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]
        plat.addKernel("ln_fwd", {"source": _LN_FWD_WGSL, "bindingTypes": rw})
        plat.addKernel("ln_dx", {"source": _LN_DX_WGSL, "bindingTypes": rw})
        plat.addKernel("ln_dgb", {"source": _LN_DGB_WGSL, "bindingTypes":
                                  ["read-only-storage", "read-only-storage", "storage", "storage", "read-only-storage"]})
        _ln_wgpu["added"] = True
    got = _wgpu_ln_rows(_contig(xd), None, _contig(gd), _contig(bd), eps)
    if got is not None:
        return got[1]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    # Every output lane is written by ln_fwd. Host-backed zero-fill would upload
    # the entire activation before this kernel, serialising the preceding queue.
    out = _empty(xd.shape)
    plat.runKernel({"name": "ln_fwd",
        "tensors": [xd.buffer.buffer_id, gd.buffer.buffer_id, bd.buffer.buffer_id,
                    out.buffer.buffer_id, _wgpu_ln_meta(rows, width, eps).buffer_id],
        "workGroups": {"x": (rows + 63) // 64, "y": 1, "z": 1}})
    return out


def _wgpu_ln_bwd(xd, g, gd, eps):
    plat = _adam_kernel["platform"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    dx = _empty(xd.shape); dgam = _empty((width,)); dbet = _empty((width,))
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
    # Few rows: one pass, each element deriving its row's statistics. It repeats work, but
    # across `width` fragments at once, where the two-pass form below gives each row to a
    # single fragment walking `width` dependent fetches while the GPU idles -- the trade the
    # RMSNorm note above measured, and the reason both forms exist.
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
    # Many rows: a row's mean and inverse deviation, computed ONCE per row. The fused kernel
    # replaces recomputed both for every element of the row: 1,536 fetches per output, so a
    # 519x768 LayerNorm read as much as a 519x768x768 matmul -- measured on WebGL, 10.9 ms a
    # draw and a quarter of a decision encoder's time. Two outputs per row, laid out as an
    # ordinary (rows, 2) float tensor: fragment 2r writes the mean, 2r+1 the inverse
    # deviation, each running the same loops as the one-pass form. Not bit-identical to it:
    # ANGLE compiles to Metal, whose fast math may sum the same loop in a different order in
    # a different shader. Measured at 519x768: at most 9.5e-7 apart on ~2% of elements, and
    # both exactly as far from a numpy reference (5.7e-6).
    plat.addKernel(f"ln_stats_{width}", {"source": f"""{head}
uniform sampler2D tex_x; uniform int TOTAL;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (idx >= TOTAL) {{ return; }}
  int row = idx / 2; int base = row * {width};
{_gl_ln_stats(width)}
  fragColor = (idx - row * 2) == 0 ? mu : inv;
}}
"""})
    plat.addKernel(f"ln_norm_{width}", {"source": f"""{head}
uniform sampler2D tex_x; uniform sampler2D tex_stats; uniform sampler2D tex_gamma; uniform sampler2D tex_beta;
out float fragColor;
{_GL_FETCH}
void main() {{
  int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  int row = idx / {width}; int col = idx - row * {width}; int base = row * {width};
  float mu = fetch(tex_stats, row * 2); float inv = fetch(tex_stats, row * 2 + 1);
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


# One pass or two, for LayerNorm and softmax on WebGL. The one-pass form makes every element
# re-derive its row's statistics: rows*width^2 work, spread over rows*width fragments. The
# two-pass form does the statistics once per row, but each row is one fragment walking
# `width` dependent fetches. Which wins depends on that product, not on the row count alone.
# Measured on Apple M5 (Chrome, ANGLE Metal), 40 queued calls per point, ms per call:
#
#     LayerNorm w=768   rows   1: 0.235 / 0.348    2: 0.183 / 0.200    4: 0.324 / 0.249
#                              8: 0.637 / 0.128   64: 1.490 / 0.127  519: 7.246 / 0.221
#     softmax  w=173    rows  16: 0.058 / 0.079   64: 0.169 / 0.081  6228: 5.008 / 0.904
#     softmax  w=1024   rows   1: 0.134 / 0.129    2: 0.117 / 0.127    4: 0.201 / 0.143
#     softmax  w=4096   rows   1: 0.744 / 0.339  512: 185.7 / 1.437          (one / two)
#
# `rows * width^2 <= 1.5e6` picks the faster form at every point measured; where it is wrong
# it is wrong by ~0.01 ms, and far from the line the two differ by up to 130x.
_ONE_PASS_WORK = 1_500_000


def _one_pass(rows, width):
    return rows * width * width <= _ONE_PASS_WORK


def _webgl_ln_fwd(xd, gd, bd, eps):
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    _webgl_ln_kernels(rows, width)
    if _one_pass(rows, width):
        out = _empty(xd.shape)
        plat.runKernel({"name": f"ln_fwd_{width}",
            "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id},
                       {"name": "tex_gamma", "id": gd.buffer.buffer_id},
                       {"name": "tex_beta", "id": bd.buffer.buffer_id}],
            "output": out.buffer.buffer_id,
            "uniforms": [{"name": "_ka_tex_output_texture_w", "value": out.buffer.texture_shape.width, "type": "int"},
                         {"name": "EPS", "value": eps, "type": "float"}]})
        return out
    # Statistics once per row, then a normalise that only reads them. The fragment shader
    # covers the whole output texture, like the WebGPU kernel.
    stats = _empty((rows, 2))
    plat.runKernel({"name": f"ln_stats_{width}",
        "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id}],
        "output": stats.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": stats.buffer.texture_shape.width, "type": "int"},
                     {"name": "TOTAL", "value": rows * 2, "type": "int"},
                     {"name": "EPS", "value": eps, "type": "float"}]})
    out = _empty(xd.shape)
    plat.runKernel({"name": f"ln_norm_{width}",
        "inputs": [{"name": "tex_x", "id": xd.buffer.buffer_id},
                   {"name": "tex_stats", "id": stats.buffer.buffer_id},
                   {"name": "tex_gamma", "id": gd.buffer.buffer_id},
                   {"name": "tex_beta", "id": bd.buffer.buffer_id}],
        "output": out.buffer.buffer_id,
        "uniforms": [{"name": "_ka_tex_output_texture_w", "value": out.buffer.texture_shape.width, "type": "int"}]})
    return out


def _webgl_ln_bwd(xd, g, gd, eps):
    plat = _copy_kernel["plat"]
    width = int(xd.shape[-1]); rows = int(xd.size) // width
    _webgl_ln_kernels(rows, width)
    dx = _empty(xd.shape); dgam = _empty((width,)); dbet = _empty((width,))
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


# None = decide per call by row count (see `layernorm`); True/False force it, for measuring.
_LN_FUSED = {"gpu": None}


def layernorm(x, gamma, beta, eps=1e-5):
    """LayerNorm over the last axis. gamma/beta: (D,). Fused on both backends:
    WebGPU 1 fwd + 2 bwd dispatches; WebGL 1 fwd + 3 bwd draws (one output per
    draw). Fallback: plain xp ops."""
    xd = x.data
    D = xd.shape[-1]
    # Fusing on WebGPU is decided by HOW MANY ROWS there are, because that is the thing the
    # two measurements disagree about.
    #
    # Fusing was rejected once at 0.64ms -> 1.74ms/step. That was a DECODE step: one row, so
    # a row-parallel kernel has a single row of parallelism and loses. On a 250-row encoder
    # pass the same switch measures the other way -- 791ms -> 609ms end to end, 2779 -> 1787
    # dispatches, same answer to four decimals -- because 250 rows is not one row.
    #
    # So one row keeps exactly the path that was measured for one row, and everything else
    # takes the path measured for many. The line is at 1 rather than at some round number
    # because 1 is the case the earlier measurement actually covers; nothing is being assumed
    # about the counts in between beyond that they are not the case that lost.
    rows_ln = 1
    for _d in xd.shape[:-1]:
        rows_ln *= int(_d)
    fused_gpu = (_LN_FUSED["gpu"] if _LN_FUSED["gpu"] is not None
                 else rows_ln > 1) and _adam_backend_ready()
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
_add_rms_k = {"added": False}
_RMS_FUSED = True      # A/B switch for the fused path
_ROPE_FUSED = True     # A/B switch for the fused rope


_ADD_RMS_WGSL = """@group(0) @binding(0) var<storage,read> residual: array<f32>;
@group(0) @binding(1) var<storage,read> update: array<f32>;
@group(0) @binding(2) var<storage,read> weight: array<f32>;
@group(0) @binding(3) var<storage,read_write> summed: array<f32>;
@group(0) @binding(4) var<storage,read_write> normed: array<f32>;
struct ARM { T:u32, H:u32, epsbits:u32, pad:u32, }
@group(0) @binding(5) var<storage,read> am: ARM;
var<workgroup> red: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>){
  let row=wg.x; if(row>=am.T){return;} let t=lid.x; let base=row*am.H;
  var s:f32=0.0; var i=t;
  loop { if(i>=am.H){break;} let v=residual[base+i]+update[base+i];
    summed[base+i]=v; s=s+v*v; i=i+256u; }
  red[t]=s; workgroupBarrier(); var k=128u;
  loop { if(k==0u){break;} if(t<k){red[t]=red[t]+red[t+k];}
    workgroupBarrier(); k=k/2u; }
  let scale=inverseSqrt(red[0]/f32(am.H)+bitcast<f32>(am.epsbits)); var j=t;
  loop { if(j>=am.H){break;} let v=residual[base+j]+update[base+j];
    normed[base+j]=v*scale*weight[j]; j=j+256u; }
}
"""


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


def add_rmsnorm(residual, update, w, eps, execution="auto"):
    """Return ``(residual + update, rmsnorm(residual + update, w))``.

    WebGPU writes both layer-boundary values in one dispatch.  WebGL has no portable
    multi-output fragment pass, so the same operator contract uses its exact add followed
    by the existing fused RMSNorm.  That is the nearest common layer with the best path on
    each backend; callers never lose the capability because one backend lacks an atomic
    primitive.  Inference-only, matching :func:`rmsnorm`.
    """
    if execution not in ("auto", "fused", "composed"):
        raise ValueError("execution must be 'auto', 'fused', or 'composed'")
    rd = residual.data if isinstance(residual, Tensor) else residual
    ud = update.data if isinstance(update, Tensor) else update
    wd = w.data if isinstance(w, Tensor) else w
    if tuple(rd.shape) != tuple(ud.shape):
        raise ValueError("add_rmsnorm inputs must have identical shapes")
    shape = tuple(rd.shape); H = int(shape[-1]); T = 1
    for d in shape[:-1]:
        T *= int(d)
    def composed():
        summed = Tensor(rd) + Tensor(ud)
        normed = rmsnorm(summed, w, eps)
        if normed is None:
            normed = (summed / ((summed * summed).mean(axis=-1, keepdims=True)
                                + eps).sqrt()) * w
        return summed, normed
    if not _adam_backend_ready():
        return composed()
    if execution == "auto":
        key = ("add_rmsnorm", int(T), int(H))
        execution = _TUNED.get(key)
        if execution is None:
            # The common-layer API owns this choice: callers that stop here still get the
            # best complete add+norm implementation for their device and shape.
            import time as _t
            candidates = ("fused", "composed")
            samples = {q: [] for q in candidates}

            def candidate(q):
                return add_rmsnorm(Tensor(rd), Tensor(ud), Tensor(wd), eps, execution=q)

            valid = []
            reference = None
            for q in candidates:
                try:
                    pair = candidate(q)
                    got = (np.asarray(pair[0].data.get()), np.asarray(pair[1].data.get()))
                    if reference is None:
                        reference = got
                    if not all(np.allclose(a, b, rtol=2e-5, atol=2e-5)
                               for a, b in zip(got, reference)):
                        raise RuntimeError("candidate differs from its reference")
                    valid.append(q)
                except Exception as exc:
                    raise RuntimeError("add_rmsnorm candidate %r failed for shape %r"
                                       % (q, shape)) from exc
            for r in range(9):
                order = valid if not (r & 1) else list(reversed(valid))
                for q in order:
                    t0 = _t.perf_counter(); pair = None
                    for _ in range(8):
                        pair = candidate(q)
                    pair[1].data.get()
                    samples[q].append((_t.perf_counter() - t0) / 8.0)
            execution = _measured_choice(samples, valid, default=valid[0])
            _TUNED[key] = execution
    if execution == "composed":
        return composed()
    plat = _adam_kernel["platform"]
    if not _add_rms_k["added"]:
        plat.addKernel("add_rmsnorm", {"source": _ADD_RMS_WGSL,
            "bindingTypes": ["read-only-storage", "read-only-storage",
                             "read-only-storage", "storage", "storage",
                             "read-only-storage"]})
        _add_rms_k["added"] = True
    rd = _contig(rd); ud = _contig(ud); wd = _contig(wd)
    summed = _empty((T, H)); normed = _empty((T, H))
    ebits = int(np.float32(eps).view(np.uint32))
    meta = _adam_kernel["make_meta"]((T, H, ebits, 0), "u4,u4,u4,u4")
    plat.runKernel({"name": "add_rmsnorm",
                    "tensors": [rd.buffer.buffer_id, ud.buffer.buffer_id,
                                wd.buffer.buffer_id, summed.buffer.buffer_id,
                                normed.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": T, "y": 1, "z": 1}})
    return Tensor(summed.reshape(*shape)), Tensor(normed.reshape(*shape))


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


# Per-head Q/K RMSNorm followed immediately by rotary embedding.  Decode always consumes
# those as one layer operation; keeping them separate costs four dispatches per transformer
# layer (Q norm, K norm, Q rope, K rope).  A workgroup owns one head, reduces its original
# fp32 projection values, and writes the same two results in one dispatch.  WebGL keeps the
# composed layer-equivalent path because a fragment pass cannot reduce and render two output
# textures portably.
_QK_NORM_ROPE_WGSL = """
@group(0) @binding(0) var<storage,read> qr: array<f32>;
@group(0) @binding(1) var<storage,read> kr: array<f32>;
@group(0) @binding(2) var<storage,read> qw: array<f32>;
@group(0) @binding(3) var<storage,read> kw: array<f32>;
@group(0) @binding(4) var<storage,read> cb: array<f32>;
@group(0) @binding(5) var<storage,read> sb: array<f32>;
@group(0) @binding(6) var<storage,read_write> qo: array<f32>;
@group(0) @binding(7) var<storage,read_write> ko: array<f32>;
struct QM { nh:u32, nkv:u32, hd:u32, rd:u32, epsbits:u32, pad0:u32, pad1:u32, pad2:u32, }
@group(0) @binding(8) var<storage,read> qm: QM;
var<workgroup> red: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
  let head=wg.x; if(head>=qm.nh+qm.nkv){return;}
  let isq=head<qm.nh; var h=head; if(!isq){h=head-qm.nh;}
  let base=h*qm.hd; let t=lid.x; var sum:f32=0.0; var i=t;
  loop { if(i>=qm.hd){break;} var v:f32;
    if(isq){v=qr[base+i];}else{v=kr[base+i];}
    sum=sum+v*v; i=i+256u; }
  red[t]=sum; workgroupBarrier(); var n=128u;
  loop { if(n==0u){break;} if(t<n){red[t]=red[t]+red[t+n];}
    workgroupBarrier(); n=n/2u; }
  let sc=inverseSqrt(red[0]/f32(qm.hd)+bitcast<f32>(qm.epsbits));
  let half=qm.rd/2u; var j=t;
  loop { if(j>=qm.hd){break;} var v:f32; var w:f32;
    if(isq){v=qr[base+j];w=qw[j];}else{v=kr[base+j];w=kw[j];}
    let nv=v*sc*w; var outv=nv;
    if(j<qm.rd){var p:u32;var sign:f32;
      if(j<half){p=j+half;sign=-1.0;}else{p=j-half;sign=1.0;}
      var pv:f32;var pw:f32;
      if(isq){pv=qr[base+p];pw=qw[p];}else{pv=kr[base+p];pw=kw[p];}
      outv=nv*cb[j]+sign*(pv*sc*pw)*sb[j]; }
    if(isq){qo[base+j]=outv;}else{ko[base+j]=outv;}
    j=j+256u; }
}
"""
_qk_norm_rope_added = False


def qk_norm_rope_decode(q, k, qweight, kweight, cos, sin, HD, rd, eps):
    """One-dispatch per-head Q/K norm + rope for a single decode position.

    Returns ``None`` when this exact primitive is unavailable; the decoder then composes
    the same layer operation from the backend's norm and rope primitives.
    """
    global _qk_norm_rope_added
    if not _adam_backend_ready() or qweight is None or kweight is None:
        return None
    qd = _contig(q.data if isinstance(q, Tensor) else q)
    kd = _contig(k.data if isinstance(k, Tensor) else k)
    qwd = _contig(qweight.data if isinstance(qweight, Tensor) else qweight)
    kwd = _contig(kweight.data if isinstance(kweight, Tensor) else kweight)
    if int(qwd.size) != int(HD) or int(kwd.size) != int(HD):
        return None                         # full-projection norm is a different operation
    if int(qd.size) % int(HD) or int(kd.size) % int(HD):
        return None
    nh, nkv = int(qd.size) // int(HD), int(kd.size) // int(HD)
    cd = _contig(cos.data if isinstance(cos, Tensor) else cos)
    sd = _contig(sin.data if isinstance(sin, Tensor) else sin)
    if int(cd.size) != int(HD) or int(sd.size) != int(HD):
        return None
    plat = _adam_kernel["platform"]
    if not _qk_norm_rope_added:
        plat.addKernel("qk_norm_rope_decode", {"source": _QK_NORM_ROPE_WGSL,
            "bindingTypes": ["read-only-storage"] * 6 + ["storage"] * 2
                            + ["read-only-storage"]})
        _qk_norm_rope_added = True
    qo, ko = _empty(tuple(qd.shape)), _empty(tuple(kd.shape))
    ebits = int(np.float32(eps).view(np.uint32))
    meta = _adam_kernel["make_meta"](
        (nh, nkv, int(HD), int(rd), ebits, 0, 0, 0),
        "u4,u4,u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": "qk_norm_rope_decode",
                    "tensors": [qd.buffer.buffer_id, kd.buffer.buffer_id,
                                qwd.buffer.buffer_id, kwd.buffer.buffer_id,
                                cd.buffer.buffer_id, sd.buffer.buffer_id,
                                qo.buffer.buffer_id, ko.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": nh + nkv, "y": 1, "z": 1}})
    return Tensor(qo), Tensor(ko)


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
    name = cp.get_backend_name()
    if name != "webgpu":
        _backend_why["backend_name"] = str(name)
        return False
    try:
        from wgpy_backends.webgpu.platform import get_platform
        from wgpy_backends.webgpu.webgpu_buffer import create_meta_buffer_from_structure
        _adam_kernel["platform"] = get_platform()
        _adam_kernel["make_meta"] = create_meta_buffer_from_structure
        return True
    except Exception as e:
        _backend_why["platform"] = "%s: %s" % (type(e).__name__, e)
        raise RuntimeError("the selected WebGPU backend failed to initialize") from e


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
    if not GPU or cp.get_backend_name() != "webgl":
        return False
    try:
        from wgpy_backends.webgl.platform import get_platform
        _copy_kernel["plat"] = get_platform()
        return True
    except Exception as exc:
        raise RuntimeError("the selected WebGL backend failed to initialize") from exc


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
    if _adam_backend_ready():
        from wgpy_backends.webgpu import webgpu_buffer as _wb
        plat = _adam_kernel["platform"]
    elif _webgl_ready():
        from wgpy_backends.webgl import webgl_buffer as _wb
        plat = _copy_kernel["plat"]
    else:
        return
    import gc
    gc.collect()   # buffers caught in reference cycles only reach the pools once collected
    # Order matters, and the worker->main channel is FIFO: resetCaptures must
    # clear the JS-side pin set before the disposeBuffer messages below arrive,
    # or every pinned buffer is refused and never freed.
    plat.resetCaptures()
    _wb.release_capture_buffers()
    _wb.release_pooled_buffers()
    _wb.release_comm_buffer()


def _gpu_release_idle_pool():
    """Free completed temporary buffers without disturbing a live model or its captures.

    A 30B load left 281 MB of reusable scratch after shape checks.  Those sizes are not
    needed by the following one-token warm step, and on unified memory they compete with
    the model's cold expert weights.  Unlike ``_gpu_release_memory``, this deliberately
    leaves live tensors and capture pins alone.  WebGPU and WebGL expose the same contract.
    """
    if _adam_backend_ready():
        from wgpy_backends.webgpu.webgpu_buffer import release_pooled_buffers
    elif _webgl_ready():
        from wgpy_backends.webgl.webgl_buffer import release_pooled_buffers
    else:
        return
    release_pooled_buffers()


def _release_transfer_memory():
    """Drop upload staging after loading, without releasing the live model buffers."""
    if _adam_backend_ready():
        from wgpy_backends.webgpu.webgpu_buffer import release_comm_buffer
    elif _webgl_ready():
        from wgpy_backends.webgl.webgl_buffer import release_comm_buffer
    else:
        return
    release_comm_buffer()


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
GDECLA
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
GACCB
}
fn ACC4(k: u32, v: vec4<f32>) {
  if (!live) { return; }
  let s = xsoff + (k - kbase);
  a0 = a0 + dot(vec4<f32>(xs4[s], xs4[s + 1u], xs4[s + 2u], xs4[s + 3u]), v);
GACC4B
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
  let mgroup = wgid.z * (MROWu * KSGu);
  let mbu = mgroup + ly * MROWu;
  let anyrow = mgroup < gm.M;
  mb = mbu;
  mn = select(0u, min(MROWu, gm.M - mbu), mbu < gm.M);
  nrow = n;
GINITA
  xsoff = ly * MROWu * BLKVALS;
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
GSTAGE
      }
      workgroupBarrier();
"""

_GGML_GEMM_TAIL = """
      workgroupBarrier();
    }
  }
  if (live) {
    if (mn > 0u) { outp[mbu * gm.N + n] = a0; }
GWRITE
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
#
# That is where this stopped for a while, and the conclusion was wrong in a way worth
# keeping: sharing the decoded block IS one way to raise the reuse, but it is not the only
# one and it is by far the harder one. Decoding once per WORKGROUP needs 64 columns of a
# 256-value block staged, 64 KB against a 16 KB budget, so the block has to be walked in
# K-chunks -- which means splitting every one of the 28 decode fragments, since each of them
# decodes a whole block as one unit.
#
# Decoding once per THREAD and using it more times needs none of that, and it was sitting in
# the same kernel the whole time as a literal `4u`: see `_GGML_MROW` below. Widening the
# lanes fails because each new lane re-decodes; widening the ROWS ONE THREAD OWNS does not,
# because the thread already has the value in a register.
#
# The other half of it was a guard, not a tile. `ACC` tested `mn > i` before each row's
# multiply -- MROW-1 branches in the innermost loop in this file -- to skip rows a partial
# group does not have. It never protected a result: those rows are staged as 0.0 and dropped
# at the write. Removing the test is a third of the kernel on its own.
#
# Together, clean reload on each side, whole 28-layer prefill forced onto this kernel:
#
#     T        before    after
#     512      1290.8    670.8   ms     1.92x
#     1536     4296.4   2366.0   ms     1.82x
#
# and per format at N=3072 K=1024 M=512, which is what a 27B actually runs on (84% of its
# elements are i-quants, and i-quants never take the unpacking path in `ggml_matmul`):
#
#     IQ1_S 9.613 -> 4.988   IQ2_XXS 8.295 -> 4.495   IQ3_XXS 6.060 -> 4.175
#     IQ3_S 5.728 -> 4.063   IQ2_XS  5.633 -> 4.025   IQ2_S   5.520 -> 4.013
#     IQ4_XS 5.335 -> 4.038
#
# Two things to be careful of if the tiled version is ever written after all, both already
# documented in this file and both silent: `_ggml_src`'s substitutions have an order (a
# placeholder that arrives after its own substitution has run is left in the source and
# compiles to nothing), and a compile failure on these paths does not raise -- the dispatch
# produces zeros. The ggml self-check against the numpy reference, format by format, is what
# catches both, and it only catches them for a shape it actually builds.
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

# Output rows per THREAD on the batched path -- the axis KSG is not, and the one that pays.
#
# KSG adds lane rows, which are other THREADS, and each of them decodes the weight block
# again: total decode work is (threads) * K = N*M*K / MROW whatever KSG is, which is exactly
# why the sweep above came out flat. Rows per thread is the other axis. One thread decodes a
# value once and accumulates it into MROW rows, so the decode is amortized MROW ways.
#
# Measured, Q4_K 3072x1024, interleaved medians, GFLOPS at M = 512:
#
#     MROW      4       8      12      16
#     ms     5.295   4.618   4.182   4.327
#     GF       608     697     770     744
#
# and the more expensive the format's decode, the more it buys -- Q6_K goes 386 -> 649
# GFLOPS over the same range, IQ2_S 721 -> 839. 16 turns back down: the staged activations
# are KSG * MROW * BLKVALS floats, 16 KB at MROW = 16, and a workgroup that large leaves too
# few resident to hide anything. 12 is the peak, and it is not the memory limit that puts it
# there, so `_GGML_XS_BUDGET` is a guard rather than the thing being traded against.
#
# It does NOT hold at small M, because a row group is padded up to MROW whether the rows
# exist or not and the padding is accumulated as zeros:
#
#     M          3      4      8     16     32     69    512   1536
#     MROW=4  0.525  0.586  0.264  0.275  0.423  0.798  5.160  16.19  ms
#     MROW=12 0.825  0.829  0.331  0.340  0.394  0.676  4.055  11.79  ms
#
# 1.57x the wrong way below the crossover and 0.73x the right way above it, so this is a
# routing decision with a measured threshold, not a knob to leave at one value.
_GGML_MROW = 12
_GGML_MROW_SMALL = 4
_GGML_MROW_MIN_M = 32
# Bytes of workgroup memory the staged activations may take. The WebGPU guaranteed minimum
# for the whole workgroup is 16384; the codebook staging next door is 64 bytes when it is on
# at all, and the rest is slack left deliberately -- an allocation that exactly meets a limit
# is one driver's rounding away from a silent zero-filled kernel, which on this path is not
# an error but a buffer of zeros.
_GGML_XS_BUDGET = 12288


def _ggml_mrow(vals, M):
    """Output rows per thread for a batched call of this many rows on this format."""
    r = _GGML_MROW if M >= _GGML_MROW_MIN_M else _GGML_MROW_SMALL
    return max(1, min(r, _GGML_XS_BUDGET // (4 * _GGML_KSG * max(1, vals))))


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

_Q4V_FN = """
// Four packed bytes contain four consecutive low-nibble values and four consecutive
// high-nibble values. Keep the source nibbles packed until this register-local expansion;
// no alternate-width weight buffer is produced.
fn Q4LO(p: u32) -> vec4<f32> {
  return vec4<f32>(f32(p & 15u), f32((p >> 8u) & 15u),
                   f32((p >> 16u) & 15u), f32((p >> 24u) & 15u));
}
fn Q4HI(p: u32) -> vec4<f32> {
  return vec4<f32>(f32((p >> 4u) & 15u), f32((p >> 12u) & 15u),
                   f32((p >> 20u) & 15u), f32((p >> 28u) & 15u));
}
"""

_Q5V_FN = """
fn Q5LO(p: u32, h: u32, j: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(p & 15u) + select(0.0, 16.0, (h & (1u << j)) != 0u),
    f32((p >> 8u) & 15u) + select(0.0, 16.0, (h & (1u << (j + 1u))) != 0u),
    f32((p >> 16u) & 15u) + select(0.0, 16.0, (h & (1u << (j + 2u))) != 0u),
    f32((p >> 24u) & 15u) + select(0.0, 16.0, (h & (1u << (j + 3u))) != 0u));
}
fn Q5HI(p: u32, h: u32, j: u32) -> vec4<f32> {
  let k = j + 16u;
  return vec4<f32>(
    f32((p >> 4u) & 15u) + select(0.0, 16.0, (h & (1u << k)) != 0u),
    f32((p >> 12u) & 15u) + select(0.0, 16.0, (h & (1u << (k + 1u))) != 0u),
    f32((p >> 20u) & 15u) + select(0.0, 16.0, (h & (1u << (k + 2u))) != 0u),
    f32((p >> 28u) & 15u) + select(0.0, 16.0, (h & (1u << (k + 3u))) != 0u));
}
"""

# Q8_0: 32 values / 34 bytes -- f16 d + int8[32].
_Q8_0_DEC = """
    let o = base + b * 34u;
    let d = F16(o);
    let kb = b * 32u;
    // Keep the GGUF Q8_0 block exactly as stored. B4 fetches four original signed bytes in
    // one/two coalesced word reads; sign extension and the block scale happen in registers.
    // This is mathematically the same Q8_0 x f32 operation as the scalar loop, with no
    // activation requantisation and no materialised alternate-width weight.
    for (var j: u32 = 0u; j < 8u; j = j + 1u) {
      let p = B4(o + 2u + j * 4u);
      let q = vec4<f32>(
          f32(i32((p & 255u) << 24u) >> 24u),
          f32(i32(((p >> 8u) & 255u) << 24u) >> 24u),
          f32(i32(((p >> 16u) & 255u) << 24u) >> 24u),
          f32(i32(((p >> 24u) & 255u) << 24u) >> 24u));
      ACC4(kb + j * 4u, d * q);
    }
"""

_Q8_0_SCALAR_DEC = """
    let o = base + b * 34u;
    let d = F16(o);
    let kb = b * 32u;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      ACC(kb + j, d * I8(o + 2u + j));
    }
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

_IQ4NL_VEC_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u; let q = B4(o + 2u + j);
      let c0 = q & 255u; let c1 = (q >> 8u) & 255u;
      let c2 = (q >> 16u) & 255u; let c3 = (q >> 24u) & 255u;
      ACC4(kb + j, d * vec4<f32>(kv(c0 & 15u), kv(c1 & 15u),
                                  kv(c2 & 15u), kv(c3 & 15u)));
      ACC4(kb + 16u + j, d * vec4<f32>(kv(c0 >> 4u), kv(c1 >> 4u),
                                        kv(c2 >> 4u), kv(c3 >> 4u)));
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
      for (var lw: u32 = 0u; lw < 8u; lw = lw + 1u) {
        let l = lw * 4u;
        let q = B4(qo + g * 32u + l);
        ACC4(kb + i0 * 32u + l,
             Q4LO(q) * d1 - vec4<f32>(m1, m1, m1, m1));
        ACC4(kb + (i0 + 1u) * 32u + l,
             Q4HI(q) * d2 - vec4<f32>(m2, m2, m2, m2));
      }
    }
"""

# Phase two, Q4_K decode: quantise each 32-value activation sub-block to signed int8 in
# workgroup memory, then use WebGPU's packed four-way integer dot product against the
# ORIGINAL Q4_K nibbles.  The weight buffer is neither converted nor duplicated; only the
# ephemeral activation row changes width.  Every candidate is compared with the exact
# stored-Q4_K path before it may enter the per-device/shape route.
_Q4K_DP4A_WGSL = """requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
struct QM { N:u32, K:u32, rowb:u32, pad:u32, }
@group(0) @binding(3) var<storage,read> qm: QM;
var<workgroup> xq: array<u32, 256>;
var<workgroup> xsc: array<f32, 32>;
var<workgroup> xsum: array<f32, 32>;
var<workgroup> psum: array<f32, 256>;

var<private> nrow: u32;
fn W(wo:u32)->u32 { return w[wo * qm.N + nrow]; }
fn B4(o:u32)->u32 {
  let wo=o>>2u; let sh=(o&3u)*8u; let lo=W(wo);
  if(sh==0u){return lo;} return (lo>>sh)|(W(wo+1u)<<(32u-sh));
}
fn B(o:u32)->u32 { return (W(o>>2u)>>((o&3u)*8u))&255u; }
fn U16(o:u32)->u32 { return B4(o)&65535u; }
fn HF(h:u32)->f32 {
  let m=h&1023u; let e=(h>>10u)&31u; var v:f32;
  if(e==0u){v=f32(m)*5.9604644775390625e-8;}
  else if(e==31u){v=65504.0;}
  else {v=exp2(f32(i32(e)-15))*(1.0+f32(m)*0.0009765625);}
  return select(v,-v,(h&32768u)!=0u);
}
fn F16(o:u32)->f32 { return HF(U16(o)); }
fn k4sc(so:u32,j:u32)->vec2<f32>{
  if(j<4u){return vec2<f32>(f32(B(so+j)&63u),f32(B(so+j+4u)&63u));}
  let a=B(so+j+4u);
  return vec2<f32>(f32((a&15u)|((B(so+j-4u)>>6u)<<4u)),
                   f32((a>>4u)|((B(so+j)>>6u)<<4u)));
}

@compute @workgroup_size(64,4)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
  let n=gid.x; let lx=lid.x; let ly=lid.y; nrow=n;
  let nb=qm.K/256u; let steps=(nb+3u)/4u; var acc:f32=0.0;
  for(var st:u32=0u; st<steps; st=st+1u){
    let b=st*4u+ly;
    // Eight independent Q4_K sub-blocks. One lane quantises one 32-value activation
    // sub-block, so no second dispatch or global temporary is needed.
    if(lx<8u){
      let ib=lx; let si=ly*8u+ib; var mx:f32=0.0;
      if(b<nb){
        let xb=b*256u+ib*32u;
        for(var j:u32=0u;j<32u;j=j+1u){mx=max(mx,abs(x[xb+j]));}
      }
      let sc=select(1.0,mx/127.0,mx>0.0); var sm:i32=0;
      for(var j:u32=0u;j<32u;j=j+4u){
        var iv=vec4<i32>(0); if(b<nb){let xb=b*256u+ib*32u+j;
          iv=vec4<i32>(round(vec4<f32>(x[xb],x[xb+1u],x[xb+2u],x[xb+3u])/sc));}
        xq[si*8u+j/4u]=pack4xI8Clamp(iv); sm=sm+iv.x+iv.y+iv.z+iv.w;
      }
      xsc[si]=sc; xsum[si]=f32(sm);
    }
    workgroupBarrier();
    if(b<nb && n<qm.N){
      let o=b*144u; let d=F16(o); let dm=F16(o+2u); let so=o+4u; let qo=o+16u;
      for(var g:u32=0u;g<4u;g=g+1u){
        let i0=2u*g; let s0=k4sc(so,i0); let s1=k4sc(so,i0+1u);
        let xi0=ly*8u+i0; let xi1=xi0+1u; var di0:i32=0; var di1:i32=0;
        for(var j:u32=0u;j<8u;j=j+1u){
          let q=B4(qo+g*32u+j*4u);
          let q0=pack4xI8(vec4<i32>(i32(q&15u),i32((q>>8u)&15u),
                                     i32((q>>16u)&15u),i32((q>>24u)&15u)));
          let q1=pack4xI8(vec4<i32>(i32((q>>4u)&15u),i32((q>>12u)&15u),
                                     i32((q>>20u)&15u),i32((q>>28u)&15u)));
          di0=di0+dot4I8Packed(xq[xi0*8u+j],q0);
          di1=di1+dot4I8Packed(xq[xi1*8u+j],q1);
        }
        acc=acc+xsc[xi0]*(d*s0.x*f32(di0)-dm*s0.y*xsum[xi0]);
        acc=acc+xsc[xi1]*(d*s1.x*f32(di1)-dm*s1.y*xsum[xi1]);
      }
    }
    workgroupBarrier();
  }
  psum[ly*64u+lx]=acc; workgroupBarrier();
  if(ly==0u && n<qm.N){
    outp[n]=psum[lx]+psum[64u+lx]+psum[128u+lx]+psum[192u+lx];
  }
}
"""

_q4k_dp4a_added = False


def _q4k_dp4a_matmul(xf, packed, K, N):
    """Approximate one-row Q4_K matmul using native packed INT8 dot products."""
    global _q4k_dp4a_added
    if int(xf.shape[0]) != 1:
        raise RuntimeError("Q4_K DP4A is a one-row decode candidate")
    plat = _adam_kernel["platform"]
    if not _q4k_dp4a_added:
        plat.addKernel("q4k_dp4a", {"source": _Q4K_DP4A_WGSL,
            "bindingTypes": ["read-only-storage", "read-only-storage", "storage",
                             "read-only-storage"]})
        _q4k_dp4a_added = True
    out = _empty((1, int(N)))
    rowb = (int(K) // 256) * 144
    meta = _adam_kernel["make_meta"]((int(N), int(K), rowb, 0), "u4,u4,u4,u4")
    plat.runKernel({"name": "q4k_dp4a",
                    "tensors": [xf.buffer.buffer_id, packed.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (int(N) + 63) // 64, "y": 1, "z": 1}})
    return out


# Q6_K keeps its 210-byte source block in place.  Six-bit values are assembled in registers
# from ql/qh and consumed by the native packed signed-int8 dot instruction; only one
# ephemeral activation row is requantised.  Sixteen-value activation scales match Q6_K's
# sixteen source scales, so the phase-two approximation never crosses a stored scale group.
_Q6K_DP4A_WGSL = """requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
struct QM { N:u32, K:u32, rowb:u32, pad:u32, }
@group(0) @binding(3) var<storage,read> qm: QM;
var<workgroup> xq: array<u32, 256>;
var<workgroup> xsc: array<f32, 64>;
var<workgroup> psum: array<f32, 256>;

var<private> nrow: u32;
fn W(wo:u32)->u32 { return w[wo * qm.N + nrow]; }
fn B4(o:u32)->u32 {
  let wo=o>>2u; let sh=(o&3u)*8u; let lo=W(wo);
  if(sh==0u){return lo;} return (lo>>sh)|(W(wo+1u)<<(32u-sh));
}
fn B(o:u32)->u32 { return (W(o>>2u)>>((o&3u)*8u))&255u; }
fn U16(o:u32)->u32 { return B4(o)&65535u; }
fn HF(h:u32)->f32 {
  let m=h&1023u; let e=(h>>10u)&31u; var v:f32;
  if(e==0u){v=f32(m)*5.9604644775390625e-8;}
  else if(e==31u){v=65504.0;}
  else {v=exp2(f32(i32(e)-15))*(1.0+f32(m)*0.0009765625);}
  return select(v,-v,(h&32768u)!=0u);
}
fn F16(o:u32)->f32 { return HF(U16(o)); }
fn I8V(o:u32)->i32 { return i32(B(o)<<24u)>>24u; }
fn Q6P(qs:u32,qh:u32,qshift:u32,hshift:u32)->u32 {
  return pack4xI8(vec4<i32>(
    i32(((qs>>qshift)&15u)|(((qh>>hshift)&3u)<<4u))-32,
    i32(((qs>>(8u+qshift))&15u)|(((qh>>(8u+hshift))&3u)<<4u))-32,
    i32(((qs>>(16u+qshift))&15u)|(((qh>>(16u+hshift))&3u)<<4u))-32,
    i32(((qs>>(24u+qshift))&15u)|(((qh>>(24u+hshift))&3u)<<4u))-32));
}

@compute @workgroup_size(64,4)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
  let n=gid.x; let lx=lid.x; let ly=lid.y; nrow=n;
  let nb=qm.K/256u; let steps=(nb+3u)/4u; var acc:f32=0.0;
  for(var st:u32=0u; st<steps; st=st+1u){
    let b=st*4u+ly;
    if(lx<16u){
      let sb=lx; let si=ly*16u+sb; var mx:f32=0.0;
      if(b<nb){
        let xb=b*256u+sb*16u;
        for(var j:u32=0u;j<16u;j=j+1u){mx=max(mx,abs(x[xb+j]));}
      }
      let sc=select(1.0,mx/127.0,mx>0.0);
      for(var j:u32=0u;j<16u;j=j+4u){
        var iv=vec4<i32>(0); if(b<nb){let xb=b*256u+sb*16u+j;
          iv=vec4<i32>(round(vec4<f32>(x[xb],x[xb+1u],x[xb+2u],x[xb+3u])/sc));}
        xq[si*4u+j/4u]=pack4xI8Clamp(iv);
      }
      xsc[si]=sc;
    }
    workgroupBarrier();
    if(b<nb && n<qm.N){
      let o=b*210u; let d=F16(o+208u);
      for(var half:u32=0u;half<2u;half=half+1u){
        let lo=o+half*64u; let ho=o+128u+half*32u; let so=o+192u+half*8u;
        for(var region:u32=0u;region<4u;region=region+1u){
          for(var ii:u32=0u;ii<2u;ii=ii+1u){
            let sb=half*8u+region*2u+ii; let xi=ly*16u+sb; var di:i32=0;
            for(var j:u32=0u;j<16u;j=j+4u){
              let l=ii*16u+j; let a=B4(lo+l); let c=B4(lo+l+32u); let h=B4(ho+l);
              var q:u32;
              if(region==0u){q=Q6P(a,h,0u,0u);}
              else if(region==1u){q=Q6P(c,h,0u,2u);}
              else if(region==2u){q=Q6P(a,h,4u,4u);}
              else {q=Q6P(c,h,4u,6u);}
              di=di+dot4I8Packed(xq[xi*4u+j/4u],q);
            }
            acc=acc+xsc[xi]*d*f32(I8V(so+ii+2u*region))*f32(di);
          }
        }
      }
    }
    workgroupBarrier();
  }
  psum[ly*64u+lx]=acc; workgroupBarrier();
  if(ly==0u && n<qm.N){
    outp[n]=psum[lx]+psum[64u+lx]+psum[128u+lx]+psum[192u+lx];
  }
}
"""

_q6k_dp4a_added = False


def _q6k_dp4a_matmul(xf, packed, K, N):
    """Approximate one-row Q6_K matmul using native packed INT8 dot products."""
    global _q6k_dp4a_added
    if int(xf.shape[0]) != 1:
        raise RuntimeError("Q6_K DP4A is a one-row decode candidate")
    plat = _adam_kernel["platform"]
    if not _q6k_dp4a_added:
        plat.addKernel("q6k_dp4a", {"source": _Q6K_DP4A_WGSL,
            "bindingTypes": ["read-only-storage", "read-only-storage", "storage",
                             "read-only-storage"]})
        _q6k_dp4a_added = True
    out = _empty((1, int(N)))
    rowb = (int(K) // 256) * 210
    meta = _adam_kernel["make_meta"]((int(N), int(K), rowb, 0), "u4,u4,u4,u4")
    plat.runKernel({"name": "q6k_dp4a",
                    "tensors": [xf.buffer.buffer_id, packed.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": (int(N) + 63) // 64, "y": 1, "z": 1}})
    return out

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

_Q6V_FN = """
// Four Q6_K values remain in their original low-nibble and two-bit high-plane words until
// this register-local expansion. `qs` and `qh` each contain four consecutive source bytes;
// no alternate-width weight or activation buffer is produced.
fn Q6V(qs: u32, qh: u32, qshift: u32, hshift: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(((qs >> qshift) & 15u) | (((qh >> hshift) & 3u) << 4u)) - 32.0,
    f32(((qs >> (8u + qshift)) & 15u) | (((qh >> (8u + hshift)) & 3u) << 4u)) - 32.0,
    f32(((qs >> (16u + qshift)) & 15u) | (((qh >> (16u + hshift)) & 3u) << 4u)) - 32.0,
    f32(((qs >> (24u + qshift)) & 15u) | (((qh >> (24u + hshift)) & 3u) << 4u)) - 32.0);
}
"""

_Q6K_VEC_DEC = """
    let o = base + b * 210u;
    let d = F16(o + 208u);
    let kb = b * 256u;
    for (var half: u32 = 0u; half < 2u; half = half + 1u) {
      let lo = o + half * 64u;
      let ho = o + 128u + half * 32u;
      let so = o + 192u + half * 8u;
      let k0 = kb + half * 128u;
      for (var lw: u32 = 0u; lw < 8u; lw = lw + 1u) {
        let l = lw * 4u;
        let ii = l >> 4u;
        let a = B4(lo + l); let c = B4(lo + l + 32u); let h = B4(ho + l);
        ACC4(k0 + l,       d * I8(so + ii)      * Q6V(a, h, 0u, 0u));
        ACC4(k0 + l + 32u, d * I8(so + ii + 2u) * Q6V(c, h, 0u, 2u));
        ACC4(k0 + l + 64u, d * I8(so + ii + 4u) * Q6V(a, h, 4u, 4u));
        ACC4(k0 + l + 96u, d * I8(so + ii + 6u) * Q6V(c, h, 4u, 6u));
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
fn GI8V(o: u32) -> vec4<f32> {
  let p = GSRC[o >> 2u];
  return vec4<f32>(f32(i32((p & 255u) << 24u) >> 24u),
                   f32(i32(((p >> 8u) & 255u) << 24u) >> 24u),
                   f32(i32(((p >> 16u) & 255u) << 24u) >> 24u),
                   f32(i32(((p >> 24u) & 255u) << 24u) >> 24u));
}
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

_IQ2XXS_VEC_DEC = """
    let o = base + b * 66u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let ao = o + 2u + ib * 8u;
      let qw = B4(ao); let a1 = U32(ao + 4u);
      let db = d * (0.5 + f32(a1 >> 28u)) * 0.25;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let gx = ((qw >> (8u * l)) & 255u) * 2u;
        let sm = GB((a1 >> (7u * l)) & 127u);
        let k0 = kb + ib * 32u + l * 8u;
        ACC4(k0, G4V(gx) * SGN4(sm, 0u) * db);
        ACC4(k0 + 4u, G4V(gx + 1u) * SGN4(sm, 4u) * db);
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

_IQ1S_VEC_DEC = """
    let o = base + b * 50u;
    let d = F16(o);
    let kb = b * 256u;
    for (var ib: u32 = 0u; ib < 8u; ib = ib + 1u) {
      let qh = U16(o + 34u + ib * 2u);
      let dl = d * (2.0 * f32((qh >> 12u) & 7u) + 1.0);
      let delta = select(0.125, -0.125, (qh & 32768u) != 0u);
      let k0 = kb + ib * 32u;
      let qw = B4(o + 2u + ib * 4u);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let gi = 128u + (((qw >> (8u * l)) & 255u) |
                         (((qh >> (3u * l)) & 7u) << 8u)) * 8u;
        let dv = vec4<f32>(delta, delta, delta, delta);
        ACC4(k0 + l * 8u, dl * (GI8V(gi) + dv));
        ACC4(k0 + l * 8u + 4u, dl * (GI8V(gi + 4u) + dv));
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

_IQ1M_VEC_DEC = """
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
      let qw = B4(o + ib * 4u);
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let qhb = B(o + 32u + ib * 2u + (l >> 1u));
        let gi = 128u + (((qw >> (8u * l)) & 255u) |
                         ((qhb << (8u - 4u * (l & 1u))) & 1792u)) * 8u;
        let dbit = select(8u, 128u, (l & 1u) != 0u);
        let delta = select(0.125, -0.125, (qhb & dbit) != 0u);
        let dl = select(dl1, dl2, l >= 2u);
        let dv = vec4<f32>(delta, delta, delta, delta);
        ACC4(k0 + l * 8u, dl * (GI8V(gi) + dv));
        ACC4(k0 + l * 8u + 4u, dl * (GI8V(gi + 4u) + dv));
      }
    }
"""

# Q4_0 vector candidate. Kept beside the production scalar decoder so the same-width
# benchmark can compare them without changing or materialising the stored representation.
_Q4_0_VEC_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u;
      let q = B4(o + 2u + j);
      ACC4(kb + j, d * (Q4LO(q) - vec4<f32>(8.0, 8.0, 8.0, 8.0)));
      ACC4(kb + 16u + j, d * (Q4HI(q) - vec4<f32>(8.0, 8.0, 8.0, 8.0)));
    }
"""

_Q4_1_VEC_DEC = """
    let o = base + b * 20u;
    let d = F16(o); let mn = F16(o + 2u);
    let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u;
      let q = B4(o + 4u + j);
      ACC4(kb + j, d * Q4LO(q) + vec4<f32>(mn, mn, mn, mn));
      ACC4(kb + 16u + j, d * Q4HI(q) + vec4<f32>(mn, mn, mn, mn));
    }
"""

# Scalar Q4_0/Q4_1 candidates retained for the same-width benchmark. Production uses the
# vector candidates above: at realistic 4096x3072 Linear shapes they won across M=1/32/128
# in two full interleaved runs. The earlier sub-millisecond synthetic result was dominated
# by dispatch/reclamation noise and is intentionally not used for routing.
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

_Q5_0_VEC_DEC = """
    let o = base + b * 22u;
    let d = F16(o); let qh = U32(o + 2u); let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u; let q = B4(o + 6u + j);
      ACC4(kb + j, d * (Q5LO(q, qh, j) - vec4<f32>(16.0, 16.0, 16.0, 16.0)));
      ACC4(kb + 16u + j,
           d * (Q5HI(q, qh, j) - vec4<f32>(16.0, 16.0, 16.0, 16.0)));
    }
"""

_Q5_1_VEC_DEC = """
    let o = base + b * 24u;
    let d = F16(o); let mn = F16(o + 2u); let qh = U32(o + 4u); let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u; let q = B4(o + 8u + j);
      ACC4(kb + j, d * Q5LO(q, qh, j) + vec4<f32>(mn, mn, mn, mn));
      ACC4(kb + 16u + j, d * Q5HI(q, qh, j) + vec4<f32>(mn, mn, mn, mn));
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

_TQ1V_FN = """
fn TQ4(q: u32, p3: u32) -> vec4<f32> {
  let q0 = ((q & 255u) * p3) & 255u;
  let q1 = (((q >> 8u) & 255u) * p3) & 255u;
  let q2 = (((q >> 16u) & 255u) * p3) & 255u;
  let q3 = (((q >> 24u) & 255u) * p3) & 255u;
  return vec4<f32>(f32((q0 * 3u) >> 8u) - 1.0,
                   f32((q1 * 3u) >> 8u) - 1.0,
                   f32((q2 * 3u) >> 8u) - 1.0,
                   f32((q3 * 3u) >> 8u) - 1.0);
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

_TQ1_0_VEC_DEC = """
    let o = base + b * 54u;
    let d = F16(o + 52u);
    let kb = b * 256u;
    var k: u32 = 0u;
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
      let jo = g * 32u;
      let cnt = select(32u, 16u, g == 1u);
      for (var p: u32 = 0u; p < 5u; p = p + 1u) {
        let p3 = pow3(p);
        for (var mw: u32 = 0u; mw < cnt / 4u; mw = mw + 1u) {
          let m = mw * 4u; let q = B4(o + jo + m);
          ACC4(kb + k + m, TQ4(q, p3) * d);
        }
        k = k + cnt;
      }
    }
    let qh = B4(o + 48u);
    for (var p: u32 = 0u; p < 4u; p = p + 1u) {
      let p3 = pow3(p);
      ACC4(kb + k, TQ4(qh, p3) * d);
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

_TQ2_0_VEC_DEC = """
    let o = base + b * 66u;
    let d = F16(o + 64u);
    let kb = b * 256u;
    var k: u32 = 0u;
    for (var g: u32 = 0u; g < 2u; g = g + 1u) {
      let jo = g * 32u;
      for (var l: u32 = 0u; l < 4u; l = l + 1u) {
        let shift = l * 2u;
        for (var mw: u32 = 0u; mw < 8u; mw = mw + 1u) {
          let m = mw * 4u; let q = B4(o + jo + m);
          let v = vec4<f32>(f32((q >> shift) & 3u) - 1.0,
                            f32((q >> (8u + shift)) & 3u) - 1.0,
                            f32((q >> (16u + shift)) & 3u) - 1.0,
                            f32((q >> (24u + shift)) & 3u) - 1.0);
          ACC4(kb + k + m, v * d);
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

_MXFP4_VEC_DEC = """
    let o = base + b * 17u;
    let d = e8m0h(B(o));
    let kb = b * 32u;
    for (var jw: u32 = 0u; jw < 4u; jw = jw + 1u) {
      let j = jw * 4u; let q = B4(o + 1u + j);
      ACC4(kb + j, d * vec4<f32>(fp4(q & 15u), fp4((q >> 8u) & 15u),
                                  fp4((q >> 16u) & 15u), fp4((q >> 24u) & 15u)));
      ACC4(kb + 16u + j, d * vec4<f32>(fp4((q >> 4u) & 15u), fp4((q >> 12u) & 15u),
                                        fp4((q >> 20u) & 15u), fp4((q >> 28u) & 15u)));
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

_NVFP4_VEC_DEC = """
    let o = base + b * 36u;
    let kb = b * 64u;
    for (var s: u32 = 0u; s < 4u; s = s + 1u) {
      let d = ue4m3(B(o + s)); let k0 = kb + s * 16u;
      for (var jw: u32 = 0u; jw < 2u; jw = jw + 1u) {
        let j = jw * 4u; let q = B4(o + 4u + s * 8u + j);
        ACC4(k0 + j, d * vec4<f32>(fp4(q & 15u), fp4((q >> 8u) & 15u),
                                    fp4((q >> 16u) & 15u), fp4((q >> 24u) & 15u)));
        ACC4(k0 + 8u + j, d * vec4<f32>(fp4((q >> 4u) & 15u), fp4((q >> 12u) & 15u),
                                         fp4((q >> 20u) & 15u), fp4((q >> 28u) & 15u)));
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

_Q1_0_VEC_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 128u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 2u + j); let k0 = kb + j * 8u;
      ACC4(k0, vec4<f32>(select(-d, d, (q & 1u) != 0u),
                          select(-d, d, (q & 2u) != 0u),
                          select(-d, d, (q & 4u) != 0u),
                          select(-d, d, (q & 8u) != 0u)));
      ACC4(k0 + 4u, vec4<f32>(select(-d, d, (q & 16u) != 0u),
                               select(-d, d, (q & 32u) != 0u),
                               select(-d, d, (q & 64u) != 0u),
                               select(-d, d, (q & 128u) != 0u)));
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

_Q2_0_VEC_DEC = """
    let o = base + b * 18u;
    let d = F16(o);
    let kb = b * 64u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 2u + j);
      ACC4(kb + j * 4u,
           d * vec4<f32>(f32(q & 3u) - 1.0, f32((q >> 2u) & 3u) - 1.0,
                         f32((q >> 4u) & 3u) - 1.0, f32((q >> 6u) & 3u) - 1.0));
    }
"""

# name -> (decode fragment, helper functions, values per block, bytes per block, codebook)
_GGML_TYPES = {
    "F32":     (_F32_DEC,     "",                   1,   4, None),
    "F16":     (_F16_DEC,     "",                   1,   2, None),
    "BF16":    (_BF16_DEC,    "",                   1,   2, None),
    "TQ1_0":   (_TQ1_0_VEC_DEC, _POW3_FN + _TQ1V_FN, 256, 54, None),
    "TQ2_0":   (_TQ2_0_DEC,   "",                 256,  66, None),
    "MXFP4":   (_MXFP4_VEC_DEC, _FP4_FN,           32,  17, None),
    "NVFP4":   (_NVFP4_VEC_DEC, _FP4_FN,           64,  36, None),
    "Q1_0":    (_Q1_0_VEC_DEC, "",                128,  18, None),
    "Q2_0":    (_Q2_0_DEC,    "",                  64,  18, None),
    "Q4_0":    (_Q4_0_VEC_DEC, _Q4V_FN,            32,  18, None),
    "Q4_1":    (_Q4_1_VEC_DEC, _Q4V_FN,            32,  20, None),
    "Q5_0":    (_Q5_0_VEC_DEC, _Q5V_FN,            32,  22, None),
    "Q5_1":    (_Q5_1_DEC,    "",                  32,  24, None),
    "Q8_0":    (_Q8_0_DEC,    "",                  32,  34, None),
    "IQ4_NL":  (_IQ4NL_DEC,   _KV_FN,              32,  18, None),
    "IQ4_XS":  (_IQ4XS_DEC,   _KV_FN,             256, 136, None),
    "Q4_K":    (_Q4K_DEC,     _K4SC_FN + _Q4V_FN, 256, 144, None),
    "Q5_K":    (_Q5K_DEC,     _K4SC_FN,           256, 176, None),
    "Q6_K":    (_Q6K_VEC_DEC, _Q6V_FN,            256, 210, None),
    "Q3_K":    (_Q3K_DEC,     _Q3K_HELP,          256, 110, None),
    "Q2_K":    (_Q2K_DEC,     "",                 256,  84, None),
    "IQ2_XXS": (_IQ2XXS_VEC_DEC, _GRID_FN,        256,  66, "IQ2XXS_GRID_U8"),
    "IQ2_XS":  (_IQ2XS_DEC,   _GRID_FN,           256,  74, "IQ2XS_GRID_U8"),
    "IQ2_S":   (_IQ2S_DEC,    _GRID_FN,           256,  82, "IQ2S_GRID_U8"),
    "IQ3_XXS": (_IQ3XXS_DEC,  _GRID_FN,           256,  98, "IQ3XXS_GRID_U8"),
    "IQ3_S":   (_IQ3S_DEC,    _GRID_FN,           256, 110, "IQ3S_GRID_U8"),
    "IQ1_S":   (_IQ1S_VEC_DEC, _GRID_FN,          256,  50, "IQ1S_GRID_I8"),
    "IQ1_M":   (_IQ1M_VEC_DEC, _GRID_FN,          256,  56, "IQ1S_GRID_I8"),
}

# Exact same-storage decoders selected by operator shape. These never create a second
# weight buffer or requantise activations: they only regroup the original packed bytes into
# vec4 FP32 multiply-accumulates. Sub-millisecond M=1 measurements drift enough to invert
# close results, so marginal/unstable decode paths stay scalar while the consistently faster
# M=2 and batched variants use the vector decoder. Format and operator mode are the complete
# key; model/repository names are deliberately absent.
_GGML_EXACT_MODE_DECODERS = {
    "Q5_1": {0: (_Q5_1_VEC_DEC, _Q5V_FN), 2: (_Q5_1_VEC_DEC, _Q5V_FN)},
    "Q2_0": {0: (_Q2_0_VEC_DEC, ""), 2: (_Q2_0_VEC_DEC, ""),
             3: (_Q2_0_VEC_DEC, "")},
    "TQ2_0": {0: (_TQ2_0_VEC_DEC, ""), 2: (_TQ2_0_VEC_DEC, ""),
              3: (_TQ2_0_VEC_DEC, "")},
    "IQ4_NL": {0: (_IQ4NL_VEC_DEC, ""), 2: (_IQ4NL_VEC_DEC, ""),
               3: (_IQ4NL_VEC_DEC, "")},
}

# WebGL phase-one A/B routing. Modes are 1=decode, 2=two-row verification,
# 3=small/medium batch (3..32), 0=large batch. Every entry consumes the original bytes and
# FP32 activations; only scalar versus vec4 register accumulation differs. These choices
# come from WebGL measurements, independently of WebGPU's routing.
_GGML_GL_MODE_DECODERS = {
    "MXFP4": {1: (_MXFP4_DEC, _FP4_FN), 2: (_MXFP4_DEC, _FP4_FN)},
    "NVFP4": {2: (_NVFP4_DEC, _FP4_FN)},
    "IQ1_M": {2: (_IQ1M_DEC, _GRID_FN)},
    "Q8_0": {2: (_Q8_0_SCALAR_DEC, "")},
    "Q4_0": {1: (_Q4_0_DEC, ""), 2: (_Q4_0_DEC, "")},
    "Q4_1": {3: (_Q4_1_DEC, "")},
    "Q5_0": {2: (_Q5_0_DEC, ""), 0: (_Q5_0_DEC, "")},
    "Q5_1": {1: (_Q5_1_VEC_DEC, _Q5V_FN), 2: (_Q5_1_VEC_DEC, _Q5V_FN),
             3: (_Q5_1_VEC_DEC, _Q5V_FN), 0: (_Q5_1_VEC_DEC, _Q5V_FN)},
    "Q6_K": {1: (_Q6K_DEC, ""), 2: (_Q6K_DEC, ""), 3: (_Q6K_DEC, "")},
    "Q2_0": {1: (_Q2_0_DEC, ""), 2: (_Q2_0_VEC_DEC, ""),
             3: (_Q2_0_DEC, ""), 0: (_Q2_0_VEC_DEC, "")},
    "TQ2_0": {1: (_TQ2_0_VEC_DEC, ""), 2: (_TQ2_0_VEC_DEC, ""),
              3: (_TQ2_0_VEC_DEC, ""), 0: (_TQ2_0_VEC_DEC, "")},
    "IQ4_NL": {1: (_IQ4NL_DEC, _KV_FN), 2: (_IQ4NL_VEC_DEC, _KV_FN),
               3: (_IQ4NL_VEC_DEC, _KV_FN), 0: (_IQ4NL_DEC, _KV_FN)},
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
    # Same packed decoder and original bit width, redistributed across workgroups.  These
    # are phase-two candidates, never fixed policy: self-checking and per-shape measurement
    # below decide whether either is useful on this device.
    if kind == "balanced":
        return (32, 8)
    if kind == "compact":
        return (32, 4)
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


# Dequantise a packed tensor to fp32, using the SAME decode fragment the matmuls use.
#
# Every format hands each decoded value to `ACC(k, v)`, so an ACC that STORES instead of
# multiplying turns the matmul into a dequantiser -- and there is no second copy of any
# format's bit-twiddling to keep in step with the first. The fragments, their helpers and
# their placeholder substitutions are shared verbatim.
#
# Written as (K, N) so the result is what a plain `x @ W` wants, no transpose after.
_GGML_DEQ_PRE = """
fn ACC(k: u32, v: f32) { outp[k * gm.N + nrow] = v; }
fn ACC4(k: u32, v: vec4<f32>) {
  outp[k * gm.N + nrow] = v.x;
  outp[(k + 1u) * gm.N + nrow] = v.y;
  outp[(k + 2u) * gm.N + nrow] = v.z;
  outp[(k + 3u) * gm.N + nrow] = v.w;
}
"""

_GGML_DEQ_MAIN = """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  // The staged-codebook fill is written once, for the two-dimensional decode kernels, and
  // addresses its thread as `lx + ly * WGX`. This workgroup is one-dimensional, so it says
  // so in those terms rather than carrying a second copy of the fill.
  let lx = lid.x;
  let ly = 0u;
  // Before the range guard, not after it: the fill ends in a `workgroupBarrier`, and a
  // barrier is only valid where every invocation in the workgroup reaches it. An output
  // width that is not a multiple of 64 sends the tail invocations home at that guard, and
  // with the barrier below it the shader is invalid -- which is a compile failure, and a
  // compile failure on this path is silent: the dispatch produces zeros.
HELPERINIT
  let n = gid.x;
  if (n >= gm.N) { return; }
  nrow = n;
WOFFINIT
  let base = 0u;
  let nb = gm.K / BLKVALS;
  for (var b: u32 = 0u; b < nb; b = b + 1u) {
"""

_GGML_DEQ_TAIL = """
  }
}
"""


def _ggml_src(type_name, mode, cfg=None, moe=False, mrow=None):
    """`mode`: 1 or 2 for the decode kernel with that many rows, 0 for the batched one.
    `cfg` overrides (WGX, KS) for a narrow output. `moe` selects the variant that reads its
    expert from an index buffer instead of being bound to one expert's weights."""
    dec, helpers, vals, _, _ = _GGML_TYPES[type_name]
    override = _GGML_EXACT_MODE_DECODERS.get(type_name, {}).get(mode)
    if override is not None:
        dec, extra_helpers = override
        helpers += extra_helpers
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
    if mode == 3:
        pre, main, tail = _GGML_DEQ_PRE, _GGML_DEQ_MAIN, _GGML_DEQ_TAIL
    elif mode:
        pre, main, tail = ((_GGML_GEMV_PRE_V4 if vec_win else _GGML_GEMV_PRE_F32),
                           _GGML_GEMV_MAIN, _GGML_GEMV_TAIL)
    else:
        pre, main, tail = _GGML_GEMM_PRE, _GGML_GEMM_MAIN, _GGML_GEMM_TAIL
    # helpers are functions, so they go BEFORE main -- WGSL has no nested functions.
    src = _GGML_BIND + (_MOE_BIND if moe else "") + helpers + pre + main + dec + tail
    rows = 1 if mode == 3 else max(1, mode)
    two = rows == 2
    # Multi-row accumulation is for the single-token decode path, which is the hot one. The
    # two-token variant addresses psum with its own fixed layout, so it stays at one row.
    orw = _orw_for(mode)
    xrow = _GGML_KS_L * vals                      # where the second row's activations start
    # These two go FIRST: what they expand to contains other placeholders -- XSFILL brings
    # in XLOAD1 and XBAS, XSCOMMON brings in ACCDECL1 -- and a placeholder that arrives after
    # its own substitution has run is left in the source, which compiles to nothing and reads
    # as a numerically broken kernel.
    mrow = 4 if mrow is None else mrow
    # The batched kernel's rows-per-thread, expanded here rather than carried as a shader
    # constant: every one of these fragments is a different LENGTH at a different `mrow`, so
    # there is nothing to parameterise at runtime. They go near the front of the list for the
    # same reason XSFILL does -- what they expand to still contains BLKVALS.
    _ai = range(1, mrow)
    subs = [("XSFILL", _XS_FILL_V4 if vec_win else _XS_FILL_F32),
            ("XSCOMMON", _XS_COMMON),
            ("GDECLA", "\n".join("var<private> a%d: f32;" % i for i in range(mrow))),
            # No `mn >` guard on these, and that is worth a third of the kernel. The rows
            # a group does not have are staged as 0.0 by the loop below, so accumulating
            # them adds nothing and the tail drops them anyway -- the guard was never
            # protecting a result, it was skipping a multiply, and it put MROW-1 branches
            # in the innermost loop in the file to do it. Q4_K at M=512: 8.198ms guarded
            # against 5.392ms unguarded at the same MROW=4, before any of the widening
            # below. Verified unguarded against the numpy reference on all 28 formats.
            ("GACCB", "\n".join(
                "  a%d = a%d + xs4[s + %du * BLKVALS] * v;" % (i, i, i) for i in _ai)),
            ("GACC4B", "\n".join(
                ("  { let c = s + %du * BLKVALS;\n"
                 "    a%d = a%d + dot(vec4<f32>(xs4[c], xs4[c + 1u], xs4[c + 2u], "
                 "xs4[c + 3u]), v); }") % (i, i, i) for i in _ai)),
            ("GINITA", "  " + " ".join("a%d = 0.0;" % i for i in range(mrow))),
            ("GSTAGE", "\n".join(
                "        xs4[xsoff + %du * BLKVALS + t] = select(0.0, x[sx + %du * gm.K], "
                "mn > %du);" % (i, i, i) for i in _ai)),
            ("GWRITE", "\n".join(
                "    if (mn > %du) { outp[(mbu + %du) * gm.N + n] = a%d; }" % (i, i, i)
                for i in _ai)),
            ("MROWu", "%uu" % mrow),
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
            ("XS4SZ", str(_GGML_KSG * mrow * vals)),
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
    if kind in ("balanced", "compact"):
        # Explicit candidates do not come from `_shape_kind`; this still covers an output
        # tail and three independently byte-aligned packed blocks.
        return n_wide, 3
    if kind == "narrow":
        n, nb = 2, 3
    elif kind == "shortk":
        n, nb = n_wide, 3
    else:
        n, nb = n_wide, _SHORT_K_BLOCKS + 3
    # A kind can be unreachable -- 'shortk' is, whenever its threshold is off -- and there is
    # nothing to check in that case.
    return (n, nb) if _shape_kind(n, nb * vals, vals) == kind else None


def _ggml_selfcheck(type_name, mode, small=_AUTO, moe=False, mrow=_AUTO):
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
        shape = _selfcheck_shape(None, vals) or (_SMALL_N + 64, 3)
        # Small and large batches are separate production shaders on WebGL because their
        # fastest exact decoder can differ.  Check both sides of that routing boundary;
        # validating only M=3 would leave the M>32 route completely unexecuted.
        if mode == 0:
            for gl_m in (3, 33):
                _selfcheck_one(type_name, mode, None, moe, *shape, gl_m=gl_m)
        else:
            _selfcheck_one(type_name, mode, None, moe, *shape)
        return
    if mode == 0 and mrow is _AUTO:
        # Both sides of the row-group crossover, because they are two different kernels and
        # a batch only ever lands on one of them. Checking the one this session happened to
        # want leaves the other to be discovered by a user.
        vals = _GGML_TYPES[type_name][2]
        for r in sorted({_ggml_mrow(vals, 1), _ggml_mrow(vals, _GGML_MROW_MIN_M)}):
            _ggml_selfcheck(type_name, mode, small, moe, r)
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
    if mrow is _AUTO:
        mrow = None
    vals = _GGML_TYPES[type_name][2]
    shape = _selfcheck_shape(small, vals) or (_SMALL_N + 64, 3)
    _selfcheck_one(type_name, mode, small, moe, *shape, mrow=mrow)


def _selfcheck_one(type_name, mode, small, moe, N, NB, mrow=None, gl_m=None):
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
    # The batched path is checked at more than two ROW GROUPS, not at three rows. A
    # workgroup covers `mrow * KSG` rows and each of a thread's rows is its own accumulator
    # and its own guarded write, so M = 3 leaves every accumulator above the third untouched
    # -- the kernel would pass while eight of its twelve output rows were dead.
    #
    # The M also has to ROUTE to the variant being checked, or the check builds one kernel
    # and measures another; it is stepped up by whole row groups until it does, which leaves
    # the last group partial and exercises the `mn` clamp at the same time.
    if mode:
        M = mode
    elif _webgl_ready() and not _adam_backend_ready():
        # The fragment path has no rows-per-workgroup variant.  It still needs a true
        # multi-row case, but must not feed the WebGPU-only ``mrow`` (None here) into the
        # batch-shape construction below.
        M = gl_m or 3
    else:
        M = 2 * mrow + 1
        for _ in range(64):
            if _ggml_mrow(vals, M) == mrow:
                break
            M += mrow
        else:
            raise RuntimeError("no batch size routes %s to mrow=%d" % (type_name, mrow))
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
        out_t = _ggml_run_gl(xp.asarray(x), pk, type_name, K, N, eidx=eidx,
                             eslot=(eslot or 0), estride=estride)
    else:
        key = (type_name, mode, small, moe, (_GGML_KSG, mrow) if mode == 0 else 0)
        if key not in _ggml_k["added"]:
            _ggml_add(type_name, mode, small, moe, mrow)
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


# There is deliberately no global row threshold here.  The crossover between stored-block
# compute and materialize-then-matmul changed with format, shape and device in the complete
# matrix benchmark (and even reversed around the old threshold).  `_weight_execution`
# measures the actual operator with paired samples, retains every repeatable positive win,
# and puts the device-specific answer in the reusable kernel profile.  Inconclusive samples
# keep the lower-memory stored representation; there is no minimum percentage cutoff.

# Rows the fp32 matmul wants its input to be a multiple of. Its tiled kernel only runs when
# they are, and missing it costs seven times -- at K=1024 N=3072, 1850 GFLOPS at M=2816
# against 249 at M=2800. Every multiple of 32 measured fast (2816, 2848, 2880, 2944, 3008)
# and every non-multiple slow (2800, 2801, 2802, 2804, 2808, 2832).
#
# The alignment is NOT done here. Padding inside the matmul is padding once per call -- 196
# times in a prefill -- and the copy that does it goes through the host: 14.2ms for 11.5MB,
# which measured 3.4s of a 4.8s matmul phase, more than the fast path was saving. The caller
# rounds its own rows up once instead (see `_prefill`), and a caller that has not simply
# keeps the quantised kernel, which does not care.
_MATMUL_ROW_ALIGN = 32


def _matmul_row_align():
    """Rows the batched path wants a multiple of, for a caller that can arrange it."""
    return _MATMUL_ROW_ALIGN if _adam_backend_ready() else 1


def ggml_dequant(packed, type_name, K, N):
    """A packed ggml tensor as a plain (K, N) fp32 matrix, on the device.

    For the batched path only, and only for formats `ggml_dequant_ok` allows -- an i-quant
    keeps the quantised kernel whatever the row count, which is most of a 27B.

    Unpacking once and multiplying fast beats unpacking inside the multiply, as soon as there
    are enough rows to pay for the unpacking. The margin used to be about five times and is
    now 1.5x to 2.0x, because the quantised kernel was widened (see `_GGML_MROW`): at
    N=3072 K=1024, 2.742 ms unpacked against 4.325 packed at M=512, 6.303 against 12.407 at
    M=1536. Still a clear win, and still only for the formats that can take it.

    Decode is not reimplemented here; `_ggml_src(mode=3)` reuses each format's own fragment.
    """
    plat = _adam_kernel["platform"]
    key = (type_name, 3, None, False, 0)
    # An i-quant decodes through a codebook, which is a FIFTH binding. Declaring four and
    # dispatching four leaves the shader referencing a binding that is not there, which does
    # not raise: the kernel never runs, and `of` comes back holding whatever the pooled
    # buffer held before -- measured, IQ4_XS returned the numbers Q3_K had just produced.
    # `_ggml_add` has always appended this binding; this path was written without it.
    grid = _ggml_grid(type_name)
    binds = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
    if grid is not None:
        binds.append("read-only-storage")
    if key not in _ggml_k["added"]:
        plat.addKernel(_ggml_name(type_name, 3),
                       {"source": _ggml_src(type_name, 3), "bindingTypes": binds})
        _ggml_k["added"].add(key)
    of = _empty((int(K) * int(N),))
    meta = _adam_kernel["make_meta"]((1, int(N), int(K), 0, 0, 0, 0, 0),
                                     "u4,u4,u4,u4,u4,u4,u4,u4")
    # binding 0 is the activation the matmul kernels read; nothing reads it here, so the
    # weight buffer is bound again rather than allocating something to ignore.
    bufs = [packed.buffer.buffer_id, packed.buffer.buffer_id,
            of.buffer.buffer_id, meta.buffer_id]
    if grid is not None:
        bufs.append(grid.buffer.buffer_id)
    plat.runKernel({"name": _ggml_name(type_name, 3), "tensors": bufs,
                    "workGroups": {"x": (int(N) + 63) // 64, "y": 1, "z": 1}})
    return of.reshape(int(K), int(N))


_DEQ_OK = {}


def ggml_dequant_ok(type_name):
    """Does unpacking this format agree with multiplying it packed? Asked once per format.

    Every other kernel registration in this file is followed by a numerical self-check,
    because a WGSL failure here returns a buffer rather than an exception. This path was
    added without one, and an i-quant then silently produced another format's leftovers --
    which reaches the caller as a model that scores every token identically, i.e. as a
    refused load rather than as the kernel bug it is.

    The reference is the quantised kernel itself, since agreeing with it is exactly the
    property the fast path claims. A format that disagrees keeps the quantised path: slower,
    and right.
    """
    # A format that decodes through a codebook has twice been measured returning the buffer
    # a DIFFERENT format left behind -- the signature of a kernel that did not run -- and
    # twice done so intermittently, passing the same comparison moments earlier. The binding
    # it was missing is now declared and dispatched, and that was a real fix, but it is not
    # the whole of this: something about registering these kernels is order-dependent, and
    # until that is understood a wrong answer is not worth the speed. They keep the
    # quantised kernel, which agrees with the GEMV exactly at every shape measured.
    #
    # `tab is not None` was how that was written, and it does not say it: IQ4_NL and IQ4_XS
    # decode through a codebook too, an inlined one -- sixteen values packed into four u32s
    # and staged in workgroup memory by `kvfill` -- so they have no `tab` and fell through.
    # It cost nothing while their dequant shader would not compile, and the moment that was
    # fixed they took a path that had never once run for them. Measured immediately, same
    # machine, same question, same fresh conversation, a 601-token reply: 6.x tok/s with this
    # path off, 1.x with it on. The unpacked copy is eight times the packed bytes, and this
    # model already fills the machine; the kernel is not what is slow, the residency it
    # destroys is. So the test asks what the comment above always meant.
    #
    # Asked BEFORE the remembered answer, not after. A profile records what a device
    # measured; it cannot grant a path this file refuses, and it was doing exactly that --
    # `use_kernel_profile` writes `_DEQ_OK` directly, so a profile saved by the build where
    # these formats slipped through went on re-enabling them under the build that fixed it.
    # The refusal is policy. Only the measurement below is cacheable.
    if _GGML_TYPES[type_name][4] is not None or "fn kvfill" in _GGML_TYPES[type_name][1]:
        _DEQ_OK[type_name] = False
        return False
    if type_name in _DEQ_OK:
        return _DEQ_OK[type_name]
    try:
        vals = int(_GGML_TYPES[type_name][2])
        blk = int(_GGML_TYPES[type_name][3])
        # Several blocks per column and several workgroups of columns. The first version of
        # this used one block and one workgroup -- the smallest legal shape -- and passed a
        # format that was wrong at every shape a model actually has.
        K = vals * 4
        N = 256
        nbytes = (K // vals) * N * blk
        raw = np.random.default_rng(0).integers(0, 200, nbytes + (-nbytes % 4),
                                                dtype=np.uint8)
        # A float format's bytes ARE its values: random bytes viewed as F32 include NaN,
        # infinities and 3e38, both sides come back non-finite, and the check refused F32 --
        # which failed a 30B MoE's load at its F32 router. Such formats get finite values.
        floats = {"F32": lambda v: v.view(np.uint8),
                  "F16": lambda v: v.astype(np.float16).view(np.uint8),
                  "BF16": lambda v: (v.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8)}
        if type_name in floats:
            v = np.random.default_rng(0).standard_normal(K * N).astype(np.float32) * 0.1
            enc = floats[type_name](v)
            raw = np.zeros(nbytes + (-nbytes % 4), np.uint8)
            raw[:enc.size] = enc
        W = Tensor(raw.view(np.float32).copy()).data
        # One row against many. The quantised GEMV is the reference because decode uses it
        # every token; the batched rows are what the fast path replaces. Checking only one
        # row checked the wrong kernel.
        x1 = np.random.default_rng(1).standard_normal((1, K)).astype(np.float32)
        a = np.asarray(ggml_matmul(Tensor(x1).data, W, type_name, K, N).get(),
                       np.float32).ravel()
        deq = Tensor(ggml_dequant(W, type_name, K, N))
        b = np.asarray((Tensor(x1) @ deq).data.get(), np.float32).ravel()
        scale = max(1e-6, float(np.abs(a).max()))
        ok = bool(np.all(np.isfinite(b))
                  and float(np.abs(a - b).max()) / scale < 1e-3)
        if not ok:
            raise RuntimeError("materialized GGUF %s disagrees with stored-width reference"
                               % type_name)
    except Exception as exc:
        raise RuntimeError("GGUF materialized candidate failed for %s" % type_name) from exc
    _DEQ_OK[type_name] = ok
    return ok


# Many rows against a Q8_0 weight, without changing the weight's width.
#
# The stored kernel decodes a block once per output row group it serves, and at a prompt's
# worth of rows that is the cost: on 519x768x2304 it took 2.38 ms where the same matmul on
# the half-precision copy of these weights (mm_f16w) took 0.74. The measured winner until
# now was "materialized" -- expand the whole weight to f32, then a dense matmul -- which is
# a second, four-times-wider copy of every weight per call.
#
# Here one workgroup computes 32 rows x 64 columns. For each 32-value block its 64 threads
# decode the block's 64 columns ONCE into workgroup memory, and every thread then multiplies
# 4 rows x 8 columns out of it. What is held there is the int8 values as halves -- every
# int8 is exact in f16 -- not d*q, which is not: the block's scale multiplies a per-block
# partial sum instead, d * (sum of q*x), so the inner loop reads 16 bytes a k and unpacks
# exactly as mm_f16w's does, and nothing is rounded to fewer bits than f32.
#
# Measured on Apple M5 (Chrome 154, medians of interleaved rounds, ms):
#                      stored  materialized  tiled-f32-tile  this  mm_f16w
#   519x768x2304        2.375         1.192           1.139  0.856    0.736
#   519x768x768         0.781         0.650           0.436  0.336    0.286
#   519x1152x768        1.119         0.606           0.615  0.466    0.403
# The f32-tile version (d*q, exact in f32, 32 bytes a k) lost to this by a quarter: what the
# loop reads per k is the cost, not the decode (filling the tile with a constant saved 5%)
# or the barriers (removing them saved 3%). It is a candidate, not a rule: `ggml_matmul`
# measures it against the others per format, shape and row bucket.
_GGML_TILED_Q8_0_WGSL = """
@group(0) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> packed: array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> array_c: array<vec4<f32>>;
struct QMeta { M: u32, N: u32, K: u32, RW: u32, G: u32, }
@group(0) @binding(3) var<storage,read> qm: QMeta;
// [k * 8 + c8]: one k of the block for 8 adjacent columns, as four half pairs.
var<workgroup> bs: array<vec4<u32>, 256>;
// The block's scales for the workgroup's 64 columns.
var<workgroup> ds: array<vec4<f32>, 16>;
fn sx(w: vec4<u32>, sh: u32) -> vec4<f32> {
  return vec4<f32>(vec4<i32>(w << vec4<u32>(sh)) >> vec4<u32>(24u));
}
fn h2(v: vec4<f32>, u: vec4<f32>) -> vec4<u32> {
  return vec4<u32>(pack2x16float(v.xy), pack2x16float(v.zw),
                   pack2x16float(u.xy), pack2x16float(u.zw));
}
fn scale4(us: vec4<u32>, al: bool) -> vec4<f32> {
  return vec4<f32>(unpack2x16float(select(us.x & 0xFFFFu, us.x >> 16u, al)).x,
                   unpack2x16float(select(us.y & 0xFFFFu, us.y >> 16u, al)).x,
                   unpack2x16float(select(us.z & 0xFFFFu, us.z >> 16u, al)).x,
                   unpack2x16float(select(us.w & 0xFFFFu, us.w >> 16u, al)).x);
}
@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let M = qm.M; let N = qm.N; let K = qm.K; let RW = qm.RW;
  let KD4 = K >> 2u; let ND4 = N >> 2u;
  let t = li.y * 8u + li.x;
  // Decoding: 8 adjacent columns (c8) by 4 of the block's 32 k (kq).
  let c8 = t & 7u; let kq = t >> 3u;
  let col4 = (wg.x * 64u + c8 * 8u) >> 2u;
  // Multiplying: 4 rows by 8 columns, as mm_f16w does.
  let c0 = wg.x * 64u + li.x * 8u;
  let row = wg.y * 32u + li.y * 4u;
  let i0 = select(M - 1u, row, row < M);
  let i1 = select(i0, row + 1u, row + 1u < M);
  let i2 = select(i0, row + 2u, row + 2u < M);
  let i3 = select(i0, row + 3u, row + 3u < M);
  var s00 = vec4<f32>(); var s01 = vec4<f32>(); var s02 = vec4<f32>(); var s03 = vec4<f32>();
  var s10 = vec4<f32>(); var s11 = vec4<f32>(); var s12 = vec4<f32>(); var s13 = vec4<f32>();
  // With G > 1 the blocks are cut G ways (workgroup z takes one share) and each share is
  // written to its own (M, N) slice for a reduction pass: that is how a few rows still fill
  // the device, exactly as mm_f16w does.
  let nb = K >> 5u;
  let per = (nb + qm.G - 1u) / qm.G;
  let b0 = wg.z * per;
  let b1 = min(nb, b0 + per);
  let sl = wg.z * M * ND4;
  for (var b: u32 = b0; b < b1; b = b + 1u) {
    // Block b is bytes 34b.. of the column: its scale is the low half of word w0, or the
    // high half when the block starts mid-word -- and then its int8 are word-aligned.
    let byte0 = b * 34u;
    let w0 = byte0 >> 2u;
    let al = (byte0 & 3u) == 2u;
    if (kq == 0u) {
      ds[c8 * 2u] = scale4(packed[w0 * ND4 + col4], al);
      ds[c8 * 2u + 1u] = scale4(packed[w0 * ND4 + col4 + 1u], al);
    }
    // int8 group kq (k = 4kq..4kq+3) is word w0+1+kq when aligned, else the top half of
    // w0+kq and the bottom half of w0+kq+1.
    let wl = w0 + kq;
    let in1 = wl + 1u < RW;
    let wh = select(vec4<u32>(0u), packed[(wl + 1u) * ND4 + col4], in1);
    let wh2 = select(vec4<u32>(0u), packed[(wl + 1u) * ND4 + col4 + 1u], in1);
    let ga = select((packed[wl * ND4 + col4] >> vec4<u32>(16u)) | (wh << vec4<u32>(16u)),
                    wh, vec4<bool>(al));
    let gb = select((packed[wl * ND4 + col4 + 1u] >> vec4<u32>(16u)) | (wh2 << vec4<u32>(16u)),
                    wh2, vec4<bool>(al));
    let k0 = kq * 4u;
    bs[(k0 + 0u) * 8u + c8] = h2(sx(ga, 24u), sx(gb, 24u));
    bs[(k0 + 1u) * 8u + c8] = h2(sx(ga, 16u), sx(gb, 16u));
    bs[(k0 + 2u) * 8u + c8] = h2(sx(ga, 8u), sx(gb, 8u));
    bs[(k0 + 3u) * 8u + c8] = h2(sx(ga, 0u), sx(gb, 0u));
    workgroupBarrier();
    var p00 = vec4<f32>(); var p01 = vec4<f32>(); var p02 = vec4<f32>(); var p03 = vec4<f32>();
    var p10 = vec4<f32>(); var p11 = vec4<f32>(); var p12 = vec4<f32>(); var p13 = vec4<f32>();
    let kb4 = b * 8u;
    for (var k4: u32 = 0u; k4 < 8u; k4 = k4 + 1u) {
      let a0 = array_a[i0 * KD4 + kb4 + k4]; let a1 = array_a[i1 * KD4 + kb4 + k4];
      let a2 = array_a[i2 * KD4 + kb4 + k4]; let a3 = array_a[i3 * KD4 + kb4 + k4];
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let pk = bs[(k4 * 4u + j) * 8u + li.x];
        let lo = vec4<f32>(unpack2x16float(pk.x), unpack2x16float(pk.y));
        let hi = vec4<f32>(unpack2x16float(pk.z), unpack2x16float(pk.w));
        p00 = vec4<f32>(a0[j]) * lo + p00; p10 = vec4<f32>(a0[j]) * hi + p10;
        p01 = vec4<f32>(a1[j]) * lo + p01; p11 = vec4<f32>(a1[j]) * hi + p11;
        p02 = vec4<f32>(a2[j]) * lo + p02; p12 = vec4<f32>(a2[j]) * hi + p12;
        p03 = vec4<f32>(a3[j]) * lo + p03; p13 = vec4<f32>(a3[j]) * hi + p13;
      }
    }
    let d0 = ds[li.x * 2u]; let d1 = ds[li.x * 2u + 1u];
    s00 = p00 * d0 + s00; s10 = p10 * d1 + s10;
    s01 = p01 * d0 + s01; s11 = p11 * d1 + s11;
    s02 = p02 * d0 + s02; s12 = p12 * d1 + s12;
    s03 = p03 * d0 + s03; s13 = p13 * d1 + s13;
    workgroupBarrier();
  }
  // Columns past N (the last workgroup of a width that is not a multiple of 64) decoded
  // whatever followed; each column's sum reads only its own column, so they are just not
  // written.
  if (row >= M || c0 >= N) { return; }
  let cx = sl + (c0 >> 2u);
  let wide = c0 + 4u < N;
  array_c[cx + row * ND4] = s00;
  if (wide) { array_c[cx + 1u + row * ND4] = s10; }
  if (row + 1u < M) {
    array_c[cx + (row + 1u) * ND4] = s01;
    if (wide) { array_c[cx + 1u + (row + 1u) * ND4] = s11; }
  }
  if (row + 2u < M) {
    array_c[cx + (row + 2u) * ND4] = s02;
    if (wide) { array_c[cx + 1u + (row + 2u) * ND4] = s12; }
  }
  if (row + 3u < M) {
    array_c[cx + (row + 3u) * ND4] = s03;
    if (wide) { array_c[cx + 1u + (row + 3u) * ND4] = s13; }
  }
}
"""
# Every other block format the same way, from one template. What makes it possible: in
# each of them a weight is A * q' - B, where q' is a small integer -- a nibble, a nibble
# minus 8, six bits minus 32, two bits minus 4 -- that a half holds exactly, and A, B are one
# pair per sub-block of 16 or 32 values and column (d*sc and dmin*m for the K-quants, d and
# -min for the _1 formats, B = 0 where the offset is folded into q'). So the workgroup memory
# holds q' as halves, exactly as for Q8_0, and a sub-block's contribution is
# A * (sum of q'*x) - B * (sum of x): both sums in f32, nothing rounded below the stored
# width. A format brings three functions over four adjacent columns: QV (q' for four k),
# QA and, when it has an offset, QB. Byte reads are vec4 over those columns, each one word
# of the (word, row) layout the stored kernel reads -- the same buffer, untouched.
_TILED_HELP = """
var<private> ND4: u32;
var<private> RWW: u32;
fn WV(w: u32, c4: u32) -> vec4<u32> {
  if (w >= RWW) { return vec4<u32>(0u); }
  return packed[w * ND4 + c4];
}
fn B4V(o: u32, c4: u32) -> vec4<u32> {
  let w = o >> 2u; let sh = (o & 3u) * 8u;
  let lo = WV(w, c4);
  if (sh == 0u) { return lo; }
  return (lo >> vec4<u32>(sh)) | (WV(w + 1u, c4) << vec4<u32>(32u - sh));
}
fn BV(o: u32, c4: u32) -> vec4<u32> {
  return (WV(o >> 2u, c4) >> vec4<u32>((o & 3u) * 8u)) & vec4<u32>(255u);
}
fn F16V(o: u32, c4: u32) -> vec4<f32> {
  let w = WV(o >> 2u, c4);
  let h = select(w & vec4<u32>(0xFFFFu), w >> vec4<u32>(16u), vec4<bool>((o & 2u) != 0u));
  return vec4<f32>(unpack2x16float(h.x).x, unpack2x16float(h.y).x,
                   unpack2x16float(h.z).x, unpack2x16float(h.w).x);
}
fn NIB(w: vec4<u32>, i: u32, sh: u32) -> vec4<f32> {
  return vec4<f32>((w >> vec4<u32>(8u * i + sh)) & vec4<u32>(15u));
}
fn BIT16(w: vec4<u32>, b: u32) -> vec4<f32> {
  return vec4<f32>((w >> vec4<u32>(b)) & vec4<u32>(1u)) * 16.0;
}
"""
_TILED_Q4_0 = """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let w = B4V(b * 18u + 2u + (kl & 15u), c4); let sh = select(0u, 4u, kl >= 16u);
  let e = vec4<f32>(8.0);
  return mat4x4<f32>(NIB(w, 0u, sh) - e, NIB(w, 1u, sh) - e, NIB(w, 2u, sh) - e, NIB(w, 3u, sh) - e);
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { return F16V(b * 18u, c4); }
"""
_TILED_Q4_1 = """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let w = B4V(b * 20u + 4u + (kl & 15u), c4); let sh = select(0u, 4u, kl >= 16u);
  return mat4x4<f32>(NIB(w, 0u, sh), NIB(w, 1u, sh), NIB(w, 2u, sh), NIB(w, 3u, sh));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { return F16V(b * 20u, c4); }
fn QB(b: u32, s: u32, c4: u32) -> vec4<f32> { return -F16V(b * 20u + 2u, c4); }
"""
_TILED_Q5_0 = """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 22u; let qh = B4V(o + 2u, c4);
  let w = B4V(o + 6u + (kl & 15u), c4); let sh = select(0u, 4u, kl >= 16u);
  let e = vec4<f32>(16.0);
  return mat4x4<f32>(NIB(w, 0u, sh) + BIT16(qh, kl) - e, NIB(w, 1u, sh) + BIT16(qh, kl + 1u) - e,
                     NIB(w, 2u, sh) + BIT16(qh, kl + 2u) - e, NIB(w, 3u, sh) + BIT16(qh, kl + 3u) - e);
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { return F16V(b * 22u, c4); }
"""
_TILED_Q5_1 = """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 24u; let qh = B4V(o + 4u, c4);
  let w = B4V(o + 8u + (kl & 15u), c4); let sh = select(0u, 4u, kl >= 16u);
  return mat4x4<f32>(NIB(w, 0u, sh) + BIT16(qh, kl), NIB(w, 1u, sh) + BIT16(qh, kl + 1u),
                     NIB(w, 2u, sh) + BIT16(qh, kl + 2u), NIB(w, 3u, sh) + BIT16(qh, kl + 3u));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { return F16V(b * 24u, c4); }
fn QB(b: u32, s: u32, c4: u32) -> vec4<f32> { return -F16V(b * 24u + 2u, c4); }
"""
# The K-quants' packed 6-bit (scale, min) of sub-block j, for four columns.
_TILED_K4SC = """
fn K4SCV(so: u32, j: u32, c4: u32) -> mat2x4<f32> {
  if (j < 4u) {
    return mat2x4<f32>(vec4<f32>(BV(so + j, c4) & vec4<u32>(63u)),
                       vec4<f32>(BV(so + j + 4u, c4) & vec4<u32>(63u)));
  }
  let a = BV(so + j + 4u, c4);
  return mat2x4<f32>(
    vec4<f32>((a & vec4<u32>(15u)) | ((BV(so + j - 4u, c4) >> vec4<u32>(6u)) << vec4<u32>(4u))),
    vec4<f32>((a >> vec4<u32>(4u)) | ((BV(so + j, c4) >> vec4<u32>(6u)) << vec4<u32>(4u))));
}
"""
_TILED_Q4_K = _TILED_K4SC + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 144u; let j = kl >> 5u; let l = kl & 31u;
  let w = WV((o + 16u + (j >> 1u) * 32u + l) >> 2u, c4); let sh = (j & 1u) * 4u;
  return mat4x4<f32>(NIB(w, 0u, sh), NIB(w, 1u, sh), NIB(w, 2u, sh), NIB(w, 3u, sh));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 144u; return F16V(o, c4) * K4SCV(o + 4u, s, c4)[0]; }
fn QB(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 144u; return F16V(o + 2u, c4) * K4SCV(o + 4u, s, c4)[1]; }
"""
_TILED_Q5_K = _TILED_K4SC + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 176u; let j = kl >> 5u; let l = kl & 31u;
  let w = WV((o + 48u + (j >> 1u) * 32u + l) >> 2u, c4); let sh = (j & 1u) * 4u;
  let hw = WV((o + 16u + l) >> 2u, c4);
  return mat4x4<f32>(NIB(w, 0u, sh) + BIT16(hw, j), NIB(w, 1u, sh) + BIT16(hw, 8u + j),
                     NIB(w, 2u, sh) + BIT16(hw, 16u + j), NIB(w, 3u, sh) + BIT16(hw, 24u + j));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 176u; return F16V(o, c4) * K4SCV(o + 4u, s, c4)[0]; }
fn QB(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 176u; return F16V(o + 2u, c4) * K4SCV(o + 4u, s, c4)[1]; }
"""
_TILED_Q6_K = """
fn Q6E(qs: vec4<u32>, qh: vec4<u32>, i: u32, qsh: u32, hsh: u32) -> vec4<f32> {
  return vec4<f32>(((qs >> vec4<u32>(8u * i + qsh)) & vec4<u32>(15u))
                   | (((qh >> vec4<u32>(8u * i + hsh)) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0;
}
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 210u; let h = kl >> 7u; let r = kl & 127u; let qt = r >> 5u; let l = r & 31u;
  let qs = B4V(o + h * 64u + l + 32u * (qt & 1u), c4);
  let qh = B4V(o + 128u + h * 32u + l, c4);
  let qsh = 4u * (qt >> 1u); let hsh = 2u * qt;
  return mat4x4<f32>(Q6E(qs, qh, 0u, qsh, hsh), Q6E(qs, qh, 1u, qsh, hsh),
                     Q6E(qs, qh, 2u, qsh, hsh), Q6E(qs, qh, 3u, qsh, hsh));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 210u;
  let sc = vec4<f32>(vec4<i32>(BV(o + 192u + s, c4) << vec4<u32>(24u)) >> vec4<u32>(24u));
  return F16V(o + 208u, c4) * sc;
}
"""
_TILED_Q3_K = """
fn Q3E(q: vec4<u32>, m: vec4<u32>, i: u32, sh: u32, mb: u32) -> vec4<f32> {
  return vec4<f32>((q >> vec4<u32>(8u * i + sh)) & vec4<u32>(3u))
         - select(vec4<f32>(4.0), vec4<f32>(0.0), ((m >> vec4<u32>(8u * i + mb)) & vec4<u32>(1u)) != vec4<u32>(0u));
}
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 110u; let is = kl >> 4u; let l = kl & 15u;
  let blk2 = is >> 3u; let jj = (is >> 1u) & 3u; let half = is & 1u;
  let q = B4V(o + 32u + blk2 * 32u + half * 16u + l, c4);
  let m = B4V(o + half * 16u + l, c4);
  let sh = 2u * jj; let mb = blk2 * 4u + jj;
  return mat4x4<f32>(Q3E(q, m, 0u, sh, mb), Q3E(q, m, 1u, sh, mb),
                     Q3E(q, m, 2u, sh, mb), Q3E(q, m, 3u, sh, mb));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 110u;
  let a0 = B4V(o + 96u, c4); let a1 = B4V(o + 100u, c4); let a2 = B4V(o + 104u, c4);
  let lo4 = vec4<u32>(0x0F0F0F0Fu); let lo2 = vec4<u32>(0x03030303u); let f = vec4<u32>(4u);
  let wsel = s >> 2u;
  var v: vec4<u32>;
  if (wsel == 0u) { v = (a0 & lo4) | ((a2 & lo2) << f); }
  else if (wsel == 1u) { v = (a1 & lo4) | (((a2 >> vec4<u32>(2u)) & lo2) << f); }
  else if (wsel == 2u) { v = ((a0 >> f) & lo4) | (((a2 >> f) & lo2) << f); }
  else { v = ((a1 >> f) & lo4) | (((a2 >> vec4<u32>(6u)) & lo2) << f); }
  let sc = vec4<f32>((v >> vec4<u32>(8u * (s & 3u))) & vec4<u32>(255u)) - 32.0;
  return F16V(o + 108u, c4) * sc;
}
"""
_TILED_Q2_K = """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 84u; let is = kl >> 4u; let l = kl & 15u;
  let blk2 = is >> 3u; let jj = (is >> 1u) & 3u; let half = is & 1u;
  let q = WV((o + 16u + blk2 * 32u + half * 16u + l) >> 2u, c4); let sh = 2u * jj;
  let t = vec4<u32>(3u);
  return mat4x4<f32>(vec4<f32>((q >> vec4<u32>(sh)) & t), vec4<f32>((q >> vec4<u32>(8u + sh)) & t),
                     vec4<f32>((q >> vec4<u32>(16u + sh)) & t), vec4<f32>((q >> vec4<u32>(24u + sh)) & t));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 84u; return F16V(o + 80u, c4) * vec4<f32>(BV(o + s, c4) & vec4<u32>(15u)); }
fn QB(b: u32, s: u32, c4: u32) -> vec4<f32> { let o = b * 84u; return F16V(o + 82u, c4) * vec4<f32>(BV(o + s, c4) >> vec4<u32>(4u)); }
"""
# The i-quants. Their values are codebook entries (8, 25, 43 ...; int8 for IQ4) times a
# sign, so q' is still an integer a half holds exactly; IQ1_S's +-1/8 offset is folded in by
# taking 8*(grid + delta) and A/8. The codebooks are the same buffer the stored kernel binds
# (`_ggml_grid`): 128 bytes of sign masks, then the entries from word 32. A grid lookup is
# per column, so these assemble four columns' four k and transpose.
_TILED_GRID = """
@group(0) @binding(4) var<storage,read> gr: array<u32>;
fn GBt(o: u32) -> u32 { return (gr[o >> 2u] >> ((o & 3u) * 8u)) & 255u; }
fn G4Vt(idx: u32) -> vec4<f32> { return unpack4x8unorm(gr[32u + idx]) * 255.0; }
fn GI8Vt(o: u32) -> vec4<f32> {
  let p = gr[o >> 2u];
  return vec4<f32>(vec4<i32>(vec4<u32>(p << 24u, p << 16u, p << 8u, p)) >> vec4<u32>(24u));
}
fn SGN4t(m: u32, j0: u32) -> vec4<f32> {
  return vec4<f32>(select(1.0, -1.0, (m & (1u << j0)) != 0u),
                   select(1.0, -1.0, (m & (1u << (j0 + 1u))) != 0u),
                   select(1.0, -1.0, (m & (1u << (j0 + 2u))) != 0u),
                   select(1.0, -1.0, (m & (1u << (j0 + 3u))) != 0u));
}
"""
_TILED_KV = """
fn KV(t: u32) -> f32 {
  let lo = select(0xBFAD9881u, 0xF6EADDCFu, (t & 4u) != 0u);
  let hi = select(0x26190D01u, 0x71594535u, (t & 4u) != 0u);
  let p = select(lo, hi, (t & 8u) != 0u);
  return f32(i32(((p >> (8u * (t & 3u))) & 255u) << 24u) >> 24u);
}
fn KV4(v: vec4<u32>) -> vec4<f32> { return vec4<f32>(KV(v.x), KV(v.y), KV(v.z), KV(v.w)); }
"""
_TILED_IQ4_NL = _TILED_KV + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let w = B4V(b * 18u + 2u + (kl & 15u), c4); let sh = select(0u, 4u, kl >= 16u);
  let f = vec4<u32>(15u);
  return mat4x4<f32>(KV4((w >> vec4<u32>(sh)) & f), KV4((w >> vec4<u32>(8u + sh)) & f),
                     KV4((w >> vec4<u32>(16u + sh)) & f), KV4((w >> vec4<u32>(24u + sh)) & f));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> { return F16V(b * 18u, c4); }
"""
_TILED_IQ4_XS = _TILED_KV + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 136u; let ib = kl >> 5u; let r = kl & 31u;
  let w = WV(((o + 8u + ib * 16u) >> 2u) + ((r & 15u) >> 2u), c4); let sh = select(0u, 4u, r >= 16u);
  let f = vec4<u32>(15u);
  return mat4x4<f32>(KV4((w >> vec4<u32>(sh)) & f), KV4((w >> vec4<u32>(8u + sh)) & f),
                     KV4((w >> vec4<u32>(16u + sh)) & f), KV4((w >> vec4<u32>(24u + sh)) & f));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 136u;
  let lo = (BV(o + 4u + (s >> 1u), c4) >> vec4<u32>(4u * (s & 1u))) & vec4<u32>(15u);
  let hi = (B4V(o + 2u, c4) >> vec4<u32>(2u * s)) & vec4<u32>(3u);
  return F16V(o, c4) * (vec4<f32>(lo | (hi << vec4<u32>(4u))) - 32.0);
}
"""
_TILED_IQ2_XXS = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 66u; let ib = kl >> 5u; let l = (kl >> 3u) & 3u; let hf = (kl >> 2u) & 1u;
  let qw = B4V(o + 2u + ib * 8u, c4); let a1 = B4V(o + 6u + ib * 8u, c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let gx = ((qw[c] >> (8u * l)) & 255u) * 2u + hf;
    cols[c] = G4Vt(gx) * SGN4t(GBt((a1[c] >> (7u * l)) & 127u), hf * 4u);
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 66u; let a1 = B4V(o + 6u + s * 8u, c4);
  return F16V(o, c4) * (vec4<f32>(0.5) + vec4<f32>(a1 >> vec4<u32>(28u))) * 0.25;
}
"""
_TILED_IQ2_XS = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 74u; let ib = kl >> 5u; let l = (kl >> 3u) & 3u; let hf = (kl >> 2u) & 1u;
  let qw = B4V(o + 2u + ib * 8u + select(0u, 4u, l >= 2u), c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let q = (qw[c] >> (16u * (l & 1u))) & 65535u;
    cols[c] = G4Vt((q & 511u) * 2u + hf) * SGN4t(GBt(q >> 9u), hf * 4u);
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 74u; let sc = BV(o + 66u + (s >> 1u), c4);
  let nib = select(sc & vec4<u32>(15u), sc >> vec4<u32>(4u), vec4<bool>((s & 1u) != 0u));
  return F16V(o, c4) * (vec4<f32>(0.5) + vec4<f32>(nib)) * 0.25;
}
"""
_TILED_IQ2_S = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 82u; let ib = kl >> 5u; let l = (kl >> 3u) & 3u; let hf = (kl >> 2u) & 1u;
  let qw = B4V(o + 2u + ib * 4u, c4); let sw = B4V(o + 34u + ib * 4u, c4);
  let qh = BV(o + 66u + ib, c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let gx = (((qw[c] >> (8u * l)) & 255u) | ((qh[c] << (8u - 2u * l)) & 768u)) * 2u + hf;
    cols[c] = G4Vt(gx) * SGN4t((sw[c] >> (8u * l)) & 255u, hf * 4u);
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 82u; let sc = BV(o + 74u + (s >> 1u), c4);
  let nib = select(sc & vec4<u32>(15u), sc >> vec4<u32>(4u), vec4<bool>((s & 1u) != 0u));
  return F16V(o, c4) * (vec4<f32>(0.5) + vec4<f32>(nib)) * 0.25;
}
"""
_TILED_IQ3_XXS = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 98u; let ib = kl >> 5u; let p = (kl >> 2u) & 7u;
  let a1 = B4V(o + 66u + ib * 4u, c4);
  let qw = B4V(o + 2u + ib * 8u + select(0u, 4u, p >= 4u), c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let sm = GBt((a1[c] >> (7u * (p >> 1u))) & 127u);
    cols[c] = G4Vt((qw[c] >> (8u * (p & 3u))) & 255u) * SGN4t(sm, (p & 1u) * 4u);
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 98u; let a1 = B4V(o + 66u + s * 4u, c4);
  return F16V(o, c4) * (vec4<f32>(0.5) + vec4<f32>(a1 >> vec4<u32>(28u))) * 0.5;
}
"""
_TILED_IQ3_S = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 110u; let ib = kl >> 5u; let p = (kl >> 2u) & 7u;
  let qw = B4V(o + 2u + ib * 8u + select(0u, 4u, p >= 4u), c4);
  let qh = BV(o + 66u + ib, c4); let sw = BV(o + 74u + ib * 4u + (p >> 1u), c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let gi = ((qw[c] >> (8u * (p & 3u))) & 255u) | ((qh[c] << (8u - p)) & 256u);
    cols[c] = G4Vt(gi) * SGN4t(sw[c], (p & 1u) * 4u);
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 110u; let scb = BV(o + 106u + (s >> 1u), c4);
  let nib = select(scb & vec4<u32>(15u), scb >> vec4<u32>(4u), vec4<bool>((s & 1u) != 0u));
  return F16V(o, c4) * (vec4<f32>(1.0) + 2.0 * vec4<f32>(nib));
}
"""
_TILED_IQ1_S = _TILED_GRID + """
fn QV(b: u32, kl: u32, c4: u32) -> mat4x4<f32> {
  let o = b * 50u; let ib = kl >> 5u; let l = (kl >> 3u) & 3u; let hf = (kl >> 2u) & 1u;
  let qh4 = B4V(o + 34u + ib * 2u, c4); let qw = B4V(o + 2u + ib * 4u, c4);
  var cols: array<vec4<f32>, 4>;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    let qh = qh4[c] & 65535u;
    let gi = 128u + (((qw[c] >> (8u * l)) & 255u) | (((qh >> (3u * l)) & 7u) << 8u)) * 8u + 4u * hf;
    // 8 * (grid + delta): an odd integer in [-9, 9]; A carries the 1/8.
    cols[c] = GI8Vt(gi) * 8.0 + vec4<f32>(select(1.0, -1.0, (qh & 32768u) != 0u));
  }
  return transpose(mat4x4<f32>(cols[0], cols[1], cols[2], cols[3]));
}
fn QA(b: u32, s: u32, c4: u32) -> vec4<f32> {
  let o = b * 50u; let qh = B4V(o + 34u + s * 2u, c4) & vec4<u32>(65535u);
  return F16V(o, c4) * (2.0 * vec4<f32>((qh >> vec4<u32>(12u)) & vec4<u32>(7u)) + 1.0) * 0.125;
}
"""
# name: (sub-block values, has an offset, functions)
_TILED_FORMATS = {
    "Q4_0": (32, False, _TILED_Q4_0), "Q4_1": (32, True, _TILED_Q4_1),
    "Q5_0": (32, False, _TILED_Q5_0), "Q5_1": (32, True, _TILED_Q5_1),
    "Q4_K": (32, True, _TILED_Q4_K), "Q5_K": (32, True, _TILED_Q5_K),
    "Q6_K": (16, False, _TILED_Q6_K), "Q3_K": (16, False, _TILED_Q3_K),
    "Q2_K": (16, True, _TILED_Q2_K),
    "IQ4_NL": (32, False, _TILED_IQ4_NL), "IQ4_XS": (32, False, _TILED_IQ4_XS),
    "IQ2_XXS": (32, False, _TILED_IQ2_XXS), "IQ2_XS": (16, False, _TILED_IQ2_XS),
    "IQ2_S": (16, False, _TILED_IQ2_S), "IQ3_XXS": (32, False, _TILED_IQ3_XXS),
    "IQ3_S": (32, False, _TILED_IQ3_S), "IQ1_S": (32, False, _TILED_IQ1_S),
}
_TILED_TEMPLATE = """
@group(0) @binding(0) var<storage,read> array_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> packed: array<vec4<u32>>;
@group(0) @binding(2) var<storage,read_write> array_c: array<vec4<f32>>;
struct QMeta { M: u32, N: u32, K: u32, RW: u32, G: u32, }
@group(0) @binding(3) var<storage,read> qm: QMeta;
// [k * 8 + c8]: one k of a 32-k stage for 8 adjacent columns, q' as four half pairs.
var<workgroup> bs: array<vec4<u32>, 256>;
// A (and B) of the stage's sub-blocks for the workgroup's 64 columns: [sub * 16 + col4].
var<workgroup> sA: array<vec4<f32>, 32>;
var<workgroup> sB: array<vec4<f32>, 32>;
fn h2(v: vec4<f32>, u: vec4<f32>) -> vec4<u32> {
  return vec4<u32>(pack2x16float(v.xy), pack2x16float(v.zw), pack2x16float(u.xy), pack2x16float(u.zw));
}
HELP
FUNCS
@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let M = qm.M; let N = qm.N; let K = qm.K;
  ND4 = N >> 2u; RWW = qm.RW;
  let KD4 = K >> 2u;
  let t = li.y * 8u + li.x;
  let c8 = t & 7u; let kq = t >> 3u;
  let c4 = (wg.x * 64u + c8 * 8u) >> 2u;
  let c0 = wg.x * 64u + li.x * 8u;
  let row = wg.y * 32u + li.y * 4u;
  let i0 = select(M - 1u, row, row < M);
  let i1 = select(i0, row + 1u, row + 1u < M);
  let i2 = select(i0, row + 2u, row + 2u < M);
  let i3 = select(i0, row + 3u, row + 3u < M);
  var s00 = vec4<f32>(); var s01 = vec4<f32>(); var s02 = vec4<f32>(); var s03 = vec4<f32>();
  var s10 = vec4<f32>(); var s11 = vec4<f32>(); var s12 = vec4<f32>(); var s13 = vec4<f32>();
  let ns = K >> 5u;
  let per = (ns + qm.G - 1u) / qm.G;
  let st0 = wg.z * per;
  let st1 = min(ns, st0 + per);
  let sl = wg.z * M * ND4;
  for (var st: u32 = st0; st < st1; st = st + 1u) {
    let kg = st * 32u;
    let b = kg / VALSu; let kl0 = kg - b * VALSu;
    let mlo = QV(b, kl0 + kq * 4u, c4);
    let mhi = QV(b, kl0 + kq * 4u, c4 + 1u);
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      bs[(kq * 4u + j) * 8u + c8] = h2(mlo[j], mhi[j]);
    }
    if (kq < NSUBu) {
      let s = kl0 / SBu + kq;
      sA[kq * 16u + c8 * 2u] = QA(b, s, c4);
      sA[kq * 16u + c8 * 2u + 1u] = QA(b, s, c4 + 1u);
      HASB_STORE
    }
    workgroupBarrier();
    for (var sub: u32 = 0u; sub < NSUBu; sub = sub + 1u) {
      var p00 = vec4<f32>(); var p01 = vec4<f32>(); var p02 = vec4<f32>(); var p03 = vec4<f32>();
      var p10 = vec4<f32>(); var p11 = vec4<f32>(); var p12 = vec4<f32>(); var p13 = vec4<f32>();
      var r0 = 0.0; var r1 = 0.0; var r2 = 0.0; var r3 = 0.0;
      let kb4 = (kg >> 2u) + sub * SB4u;
      for (var k4: u32 = 0u; k4 < SB4u; k4 = k4 + 1u) {
        let a0 = array_a[i0 * KD4 + kb4 + k4]; let a1 = array_a[i1 * KD4 + kb4 + k4];
        let a2 = array_a[i2 * KD4 + kb4 + k4]; let a3 = array_a[i3 * KD4 + kb4 + k4];
        HASB_ROWSUM
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          let pk = bs[((sub * SB4u + k4) * 4u + j) * 8u + li.x];
          let lo = vec4<f32>(unpack2x16float(pk.x), unpack2x16float(pk.y));
          let hi = vec4<f32>(unpack2x16float(pk.z), unpack2x16float(pk.w));
          p00 = vec4<f32>(a0[j]) * lo + p00; p10 = vec4<f32>(a0[j]) * hi + p10;
          p01 = vec4<f32>(a1[j]) * lo + p01; p11 = vec4<f32>(a1[j]) * hi + p11;
          p02 = vec4<f32>(a2[j]) * lo + p02; p12 = vec4<f32>(a2[j]) * hi + p12;
          p03 = vec4<f32>(a3[j]) * lo + p03; p13 = vec4<f32>(a3[j]) * hi + p13;
        }
      }
      let d0 = sA[sub * 16u + li.x * 2u]; let d1 = sA[sub * 16u + li.x * 2u + 1u];
      s00 = p00 * d0 + s00; s10 = p10 * d1 + s10;
      s01 = p01 * d0 + s01; s11 = p11 * d1 + s11;
      s02 = p02 * d0 + s02; s12 = p12 * d1 + s12;
      s03 = p03 * d0 + s03; s13 = p13 * d1 + s13;
      HASB_APPLY
    }
    workgroupBarrier();
  }
  if (row >= M || c0 >= N) { return; }
  let cx = sl + (c0 >> 2u);
  let wide = c0 + 4u < N;
  array_c[cx + row * ND4] = s00;
  if (wide) { array_c[cx + 1u + row * ND4] = s10; }
  if (row + 1u < M) { array_c[cx + (row + 1u) * ND4] = s01; if (wide) { array_c[cx + 1u + (row + 1u) * ND4] = s11; } }
  if (row + 2u < M) { array_c[cx + (row + 2u) * ND4] = s02; if (wide) { array_c[cx + 1u + (row + 2u) * ND4] = s12; } }
  if (row + 3u < M) { array_c[cx + (row + 3u) * ND4] = s03; if (wide) { array_c[cx + 1u + (row + 3u) * ND4] = s13; } }
}
"""


def _ggml_tiled_src(type_name):
    """The tiled kernel for a format in `_TILED_FORMATS`, from `_TILED_TEMPLATE`."""
    sb, hasb, funcs = _TILED_FORMATS[type_name]
    vals = int(_GGML_TYPES[type_name][2])
    src = (_TILED_TEMPLATE.replace("HELP", _TILED_HELP).replace("FUNCS", funcs)
           .replace("VALSu", "%du" % vals).replace("NSUBu", "%du" % (32 // sb))
           .replace("SB4u", "%du" % (sb // 4)).replace("SBu", "%du" % sb))
    if hasb:
        src = (src.replace("HASB_STORE",
                           "sB[kq * 16u + c8 * 2u] = QB(b, s, c4);\n"
                           "      sB[kq * 16u + c8 * 2u + 1u] = QB(b, s, c4 + 1u);")
               .replace("HASB_ROWSUM",
                        "let one = vec4<f32>(1.0);\n"
                        "        r0 = r0 + dot(a0, one); r1 = r1 + dot(a1, one);\n"
                        "        r2 = r2 + dot(a2, one); r3 = r3 + dot(a3, one);")
               .replace("HASB_APPLY",
                        "let e0 = sB[sub * 16u + li.x * 2u]; let e1 = sB[sub * 16u + li.x * 2u + 1u];\n"
                        "      s00 = s00 - e0 * r0; s10 = s10 - e1 * r0;\n"
                        "      s01 = s01 - e0 * r1; s11 = s11 - e1 * r1;\n"
                        "      s02 = s02 - e0 * r2; s12 = s12 - e1 * r2;\n"
                        "      s03 = s03 - e0 * r3; s13 = s13 - e1 * r3;"))
    else:
        src = (src.replace("HASB_STORE", "").replace("HASB_ROWSUM", "")
               .replace("HASB_APPLY", ""))
    return src


def _ggml_tiled_half_src(type_name):
    """The tiled kernel with its products in half precision, where the device has
    `shader-f16`.

    The stage holds the dequantised weight A*q' as halves (as llama.cpp's Metal matmul does)
    rather than q' with A applied per sub-block afterwards, so a running half sum never sees
    q' times an activation -- 63 x a large activation summed 16 times can leave the half
    range, the weight times it cannot. The activations are staged once per workgroup as
    halves; each thread sums a 32-deep stage in half and adds it into f32. Formats with a
    min keep B times the row sums per sub-block in f32. ~1e-3 of the output scale against
    ~1e-6 for the f32 kernel; 1.4-1.6x faster on an Apple M5 at prefill shapes."""
    sb, hasb, funcs = _TILED_FORMATS[type_name]
    vals = int(_GGML_TYPES[type_name][2])
    t = _TILED_TEMPLATE
    subs = [
        ("var<workgroup> sB: array<vec4<f32>, 32>;",
         "var<workgroup> sB: array<vec4<f32>, 32>;\n"
         "// The stage's 32 rows x 32 k of activations, once, as halves: [row * 9 + k4].\n"
         "var<workgroup> xsh: array<vec4<f16>, 288>;"),
        ("""    let mlo = QV(b, kl0 + kq * 4u, c4);
    let mhi = QV(b, kl0 + kq * 4u, c4 + 1u);
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      bs[(kq * 4u + j) * 8u + c8] = h2(mlo[j], mhi[j]);
    }
    if (kq < NSUBu) {
      let s = kl0 / SBu + kq;
      sA[kq * 16u + c8 * 2u] = QA(b, s, c4);
      sA[kq * 16u + c8 * 2u + 1u] = QA(b, s, c4 + 1u);
      HASB_STORE
    }
    workgroupBarrier();""",
         """    if (kq < NSUBu) {
      let s = kl0 / SBu + kq;
      sA[kq * 16u + c8 * 2u] = QA(b, s, c4);
      sA[kq * 16u + c8 * 2u + 1u] = QA(b, s, c4 + 1u);
      HASB_STORE
    }
    let mlo = QV(b, kl0 + kq * 4u, c4);
    let mhi = QV(b, kl0 + kq * 4u, c4 + 1u);
    for (var q = 0u; q < 4u; q = q + 1u) {
      let e = t + q * 64u; let rr = e >> 3u; let cc = e & 7u;
      let ir = min(wg.y * 32u + rr, M - 1u);
      xsh[rr * 9u + cc] = vec4<f16>(array_a[ir * KD4 + (kg >> 2u) + cc]);
    }
    workgroupBarrier();
    // A is per sub-block: computed once above, read here by the threads of its k-quads.
    let sub0 = (kq * 4u) / SBu;
    let alo = sA[sub0 * 16u + c8 * 2u]; let ahi = sA[sub0 * 16u + c8 * 2u + 1u];
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      bs[(kq * 4u + j) * 8u + c8] = h2(mlo[j] * alo, mhi[j] * ahi);
    }
    workgroupBarrier();
    var p00 = vec4<f16>(); var p01 = vec4<f16>(); var p02 = vec4<f16>(); var p03 = vec4<f16>();
    var p10 = vec4<f16>(); var p11 = vec4<f16>(); var p12 = vec4<f16>(); var p13 = vec4<f16>();"""),
        ("""      var p00 = vec4<f32>(); var p01 = vec4<f32>(); var p02 = vec4<f32>(); var p03 = vec4<f32>();
      var p10 = vec4<f32>(); var p11 = vec4<f32>(); var p12 = vec4<f32>(); var p13 = vec4<f32>();
""", ""),
        ("""        let a0 = array_a[i0 * KD4 + kb4 + k4]; let a1 = array_a[i1 * KD4 + kb4 + k4];
        let a2 = array_a[i2 * KD4 + kb4 + k4]; let a3 = array_a[i3 * KD4 + kb4 + k4];
        HASB_ROWSUM
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
          let pk = bs[((sub * SB4u + k4) * 4u + j) * 8u + li.x];
          let lo = vec4<f32>(unpack2x16float(pk.x), unpack2x16float(pk.y));
          let hi = vec4<f32>(unpack2x16float(pk.z), unpack2x16float(pk.w));
          p00 = vec4<f32>(a0[j]) * lo + p00; p10 = vec4<f32>(a0[j]) * hi + p10;
          p01 = vec4<f32>(a1[j]) * lo + p01; p11 = vec4<f32>(a1[j]) * hi + p11;
          p02 = vec4<f32>(a2[j]) * lo + p02; p12 = vec4<f32>(a2[j]) * hi + p12;
          p03 = vec4<f32>(a3[j]) * lo + p03; p13 = vec4<f32>(a3[j]) * hi + p13;
        }
      }
      let d0 = sA[sub * 16u + li.x * 2u]; let d1 = sA[sub * 16u + li.x * 2u + 1u];
      s00 = p00 * d0 + s00; s10 = p10 * d1 + s10;
      s01 = p01 * d0 + s01; s11 = p11 * d1 + s11;
      s02 = p02 * d0 + s02; s12 = p12 * d1 + s12;
      s03 = p03 * d0 + s03; s13 = p13 * d1 + s13;
      HASB_APPLY
    }""",
         """        let xo = sub * SB4u + k4;
        let h0 = xsh[(li.y * 4u) * 9u + xo]; let h1 = xsh[(li.y * 4u + 1u) * 9u + xo];
        let h2v = xsh[(li.y * 4u + 2u) * 9u + xo]; let h3 = xsh[(li.y * 4u + 3u) * 9u + xo];
        let a0 = vec4<f32>(h0); let a1 = vec4<f32>(h1); let a2 = vec4<f32>(h2v); let a3 = vec4<f32>(h3);
        HASB_ROWSUM
        // Every workgroup load of the step before its FMAs.
        let pk0 = bs[((sub * SB4u + k4) * 4u) * 8u + li.x];
        let pk1 = bs[((sub * SB4u + k4) * 4u + 1u) * 8u + li.x];
        let pk2 = bs[((sub * SB4u + k4) * 4u + 2u) * 8u + li.x];
        let pk3 = bs[((sub * SB4u + k4) * 4u + 3u) * 8u + li.x];
        HALFFMA
      }
      HASB_APPLY
    }
    s00 = s00 + vec4<f32>(p00); s10 = s10 + vec4<f32>(p10);
    s01 = s01 + vec4<f32>(p01); s11 = s11 + vec4<f32>(p11);
    s02 = s02 + vec4<f32>(p02); s12 = s12 + vec4<f32>(p12);
    s03 = s03 + vec4<f32>(p03); s13 = s13 + vec4<f32>(p13);"""),
    ]
    for old, new in subs:
        assert t.count(old) == 1, old[:60]
        t = t.replace(old, new)
    fma = []
    for j in range(4):
        fma.append(
            "{ let lo = bitcast<vec4<f16>>(pk%(j)d.xy); let hi = bitcast<vec4<f16>>(pk%(j)d.zw);\n"
            "          p00 = fma(vec4<f16>(h0[%(j)d]), lo, p00); p10 = fma(vec4<f16>(h0[%(j)d]), hi, p10);\n"
            "          p01 = fma(vec4<f16>(h1[%(j)d]), lo, p01); p11 = fma(vec4<f16>(h1[%(j)d]), hi, p11);\n"
            "          p02 = fma(vec4<f16>(h2v[%(j)d]), lo, p02); p12 = fma(vec4<f16>(h2v[%(j)d]), hi, p12);\n"
            "          p03 = fma(vec4<f16>(h3[%(j)d]), lo, p03); p13 = fma(vec4<f16>(h3[%(j)d]), hi, p13); }"
            % dict(j=j))
    t = "enable f16;\n" + t.replace("HALFFMA", "\n        ".join(fma))
    src = (t.replace("HELP", _TILED_HELP).replace("FUNCS", funcs)
           .replace("VALSu", "%du" % vals).replace("NSUBu", "%du" % (32 // sb))
           .replace("SB4u", "%du" % (sb // 4)).replace("SBu", "%du" % sb))
    if hasb:
        src = (src.replace("HASB_STORE",
                           "sB[kq * 16u + c8 * 2u] = QB(b, s, c4);\n"
                           "      sB[kq * 16u + c8 * 2u + 1u] = QB(b, s, c4 + 1u);")
               .replace("HASB_ROWSUM",
                        "let one = vec4<f32>(1.0);\n"
                        "        r0 = r0 + dot(a0, one); r1 = r1 + dot(a1, one);\n"
                        "        r2 = r2 + dot(a2, one); r3 = r3 + dot(a3, one);")
               .replace("HASB_APPLY",
                        "let e0 = sB[sub * 16u + li.x * 2u]; let e1 = sB[sub * 16u + li.x * 2u + 1u];\n"
                        "      s00 = s00 - e0 * r0; s10 = s10 - e1 * r0;\n"
                        "      s01 = s01 - e0 * r1; s11 = s11 - e1 * r1;\n"
                        "      s02 = s02 - e0 * r2; s12 = s12 - e1 * r2;\n"
                        "      s03 = s03 - e0 * r3; s13 = s13 - e1 * r3;"))
    else:
        src = (src.replace("HASB_STORE", "").replace("HASB_ROWSUM", "")
               .replace("HASB_APPLY", ""))
    return src


# Formats with a tiled kernel. Adding one is adding its decode, not a branch elsewhere.
_GGML_TILED = {"Q8_0": _GGML_TILED_Q8_0_WGSL}
_GGML_TILED.update({name: _ggml_tiled_src(name) for name in _TILED_FORMATS})
_ggml_tiled_k = {"added": set()}


def _ggml_tiled_ok(type_name, K, N):
    """Whether the tiled kernel takes this weight: its format has one, the output is whole
    vec4s (N % 4), and the weight is whole blocks."""
    if type_name not in _GGML_TILED:
        return False
    return (int(K) % int(_GGML_TYPES[type_name][2]) == 0 and int(K) % 32 == 0
            and int(N) % 4 == 0)


def _ggml_tiled_split(M, N):
    """How many ways the tiled kernel cuts K: four when there are too few workgroups to keep
    the device busy, else none.

    Its own line, not mm_f16w's (`_mm_split_groups`, which cuts below 128 workgroups): each
    cut repeats the per-block decode setup and adds a reduction pass, and measured that costs
    this kernel more than it costs mm_f16w. Apple M5, ms, K uncut / cut in 4:
        768x768    M=8  (12 groups) 0.095 / 0.072    M=64  (24) 0.110 / 0.114
        1152x768   M=8  (12 groups) 0.113 / 0.081    M=64  (24) 0.125 / 0.105
        768x2304   M=32 (36 groups) 0.116 / 0.109    M=64  (72) 0.154 / 0.319
        768x768    M=128 (48 groups) 0.118 / 0.157   M=256 (96) 0.176 / 0.202
    """
    groups = ((int(N) + 63) // 64) * ((int(M) + 31) // 32)
    return 4 if groups < 48 else 1


def _ggml_tiled_matmul(xf, packed, type_name, K, N, split=None, half=False):
    """xf(M,K) @ W(N,K).T from the stored blocks, through the tiled kernel above.
    `split` overrides how many ways K is cut (measurement only); `half` takes the
    half-arithmetic kernel (`_ggml_tiled_half_src`)."""
    M, K, N = int(xf.shape[0]), int(K), int(N)
    plat = _adam_kernel["platform"]
    name = ("ggml_tiledh_" if half else "ggml_tiled_") + type_name.lower()
    grid = _ggml_grid(type_name) if "binding(4) var<storage,read> gr" in _GGML_TILED[type_name] \
        else None
    if name not in _ggml_tiled_k["added"]:
        plat.addKernel(name, {"source": (_ggml_tiled_half_src(type_name) if half
                                         else _GGML_TILED[type_name]),
                              "bindingTypes": ["read-only-storage", "read-only-storage",
                                               "storage", "read-only-storage"]
                                             + (["read-only-storage"] if grid is not None
                                                else [])})
        _ggml_tiled_k["added"].add(name)
    if "ggml_tiled_reduce" not in _ggml_tiled_k["added"]:
        plat.addKernel("ggml_tiled_reduce", {"source": _MM_REDUCE_WGSL,
                                             "bindingTypes": ["storage", "read-only-storage",
                                                              "read-only-storage"]})
        _ggml_tiled_k["added"].add("ggml_tiled_reduce")
    vals, blk = int(_GGML_TYPES[type_name][2]), int(_GGML_TYPES[type_name][3])
    words = ((K // vals) * blk + 3) // 4          # ggml_transpose pads a column to whole words
    G = int(split) if split else _ggml_tiled_split(M, N)
    part = _empty((G * M, N))
    meta = _adam_kernel["make_meta"]((M, N, K, words, G), "u4,u4,u4,u4,u4")
    plat.runKernel({"name": name,
                    "tensors": [xf.buffer.buffer_id, packed.buffer.buffer_id,
                                part.buffer.buffer_id, meta.buffer_id]
                               + ([grid.buffer.buffer_id] if grid is not None else []),
                    "workGroups": {"x": (N + 63) // 64, "y": (M + 31) // 32, "z": G}})
    if G == 1:
        return part
    out = _empty((M, N))
    rmeta = _adam_kernel["make_meta"]((M * N, G), "u4,u4")
    plat.runKernel({"name": "ggml_tiled_reduce",
                    "tensors": [out.buffer.buffer_id, part.buffer.buffer_id, rmeta.buffer_id],
                    "workGroups": {"x": (M * N + 63) // 64, "y": 1, "z": 1}})
    return out


def ggml_matmul(xf, packed, type_name, K, N, eidx=None, eslot=0, estride=0,
                xper=False, bias=None, execution="stored", shape_execution="auto"):
    """xf(M,K) @ packed(N,K).T -> (M,N), decoding ggml blocks in the shader.

    `packed` must be in the transposed (word, row) layout that `ggml_transpose` produces --
    GGMLLinear does that once at upload.

    With `eidx`, `packed` holds SEVERAL weights of that shape end to end and the shader picks
    one at run time: `eidx[eslot]` is its index and `estride` its size in words. That is how a
    sparse-MoE projection runs without the choice of expert being baked into the command.

    ``execution="stored"`` is the correctness baseline: every multiply decodes the source
    blocks inside the quantized kernel, so the execution path really is the file's original
    representation.  ``"materialized"`` explicitly compares the alternative that first
    expands a supported weight on the GPU.  ``"auto"`` is reserved for a measured routing
    policy; callers must opt into it rather than silently changing representations.
    """
    if execution not in ("stored", "tiled", "tiled_half", "dp4a", "materialized", "auto"):
        raise ValueError("execution must be 'stored', 'tiled', 'tiled_half', 'dp4a', "
                         "'materialized', or 'auto'")
    if shape_execution not in ("auto", None, "narrow", "balanced", "compact", "shortk"):
        raise ValueError("stored shape execution is not a known physical route")
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
    # Prefill is 61% MLP and runs at about 0.6 GB/s against the decode path's 100+.
    #
    # The numbers above are from before `_GGML_MROW`, which nearly doubled this kernel --
    # a whole 28-layer prefill forced onto it went 1290.8 -> 670.8 ms at T=512. They are
    # kept because what they rule out has not changed: routing a batch through the two-row
    # decode kernel still loses on the copies the routing costs, whatever the batched
    # kernel is worth.
    if execution == "materialized" and not _adam_backend_ready():
        raise RuntimeError("materialized ggml comparison requires the WebGPU backend")
    if isinstance(packed, WebGLQ8Matrix):
        of = _webgl_matmul_q8k4(xf, packed)
        return of if bias is None else of + bias
    if _webgl_ready() and not _adam_backend_ready():
        return _ggml_run_gl(xf, packed, type_name, K, N, eidx=eidx, eslot=eslot,
                            estride=estride, xper=xper, bias=bias)
    mode = m if m <= 2 else 0
    # The numerical verifier for materialization calls this same entry point with
    # execution="stored" as its reference. Only discover alternative candidates
    # when the caller actually requested one; otherwise the verifier recursively
    # invokes itself before the packed kernel can run.
    can_materialize = (execution in ("auto", "materialized")
                       and eidx is None and not xper and _adam_backend_ready()
                       and ggml_dequant_ok(type_name))
    # The tiled kernel reads the same stored blocks and allocates nothing.
    can_tiled = (execution in ("auto", "tiled")
                 and eidx is None and not xper and _adam_backend_ready()
                 and _ggml_tiled_ok(type_name, K, N))
    # Its half-arithmetic twin, only where the device has `shader-f16`.
    can_tiled_half = (execution in ("auto", "tiled_half")
                      and eidx is None and not xper and _adam_backend_ready()
                      and type_name in _TILED_FORMATS and _ggml_tiled_ok(type_name, K, N)
                      and bool(gpu_features().get("f16")))
    can_dp4a = (execution in ("auto", "dp4a")
                and eidx is None and not xper and _adam_backend_ready()
                and type_name in ("Q4_K", "Q6_K") and m == 1)
    if execution == "dp4a" and not can_dp4a:
        raise RuntimeError("%s shape has no packed-dot comparison path" % type_name)
    if execution == "materialized" and not can_materialize:
        raise RuntimeError("%s has no verified materialized comparison path" % type_name)
    if execution == "tiled" and not can_tiled:
        raise RuntimeError("%s shape has no tiled stored-format path" % type_name)
    if execution == "tiled_half" and not can_tiled_half:
        raise RuntimeError("%s shape has no half-arithmetic tiled path here" % type_name)
    if execution == "auto":
        if can_tiled or can_tiled_half or can_dp4a or can_materialize:
            # The tiled kernel reads the same stored blocks and allocates nothing, so it
            # sits right after "stored": an unproven race keeps the earlier candidate.
            candidates = (("stored",) + (("tiled",) if can_tiled else ())
                          + (("tiled_half",) if can_tiled_half else ())
                          + (("dp4a",) if can_dp4a else ())
                          + (("materialized",) if can_materialize else ()))
            reference = [None]

            def run(which):
                return ggml_matmul(xf, packed, type_name, K, N, eidx=eidx,
                                   eslot=eslot, estride=estride, xper=xper,
                                   bias=bias, execution=which)

            def correct(which):
                if which == "stored":
                    return True
                if reference[0] is None:
                    reference[0] = np.asarray(run("stored").get(), np.float32)
                got = np.asarray(run(which).get(), np.float32)
                if not np.all(np.isfinite(got)):
                    return False
                scale = max(1e-6, float(np.abs(reference[0]).max()))
                # Activation INT8 is an explicit phase-two approximation.  The bound is on
                # the operator output, measured against this weight and real activation;
                # expanded-float remains a same-values comparison with the tighter bound.
                # Half arithmetic: ~1e-3 of the output scale measured, bounded at 1e-2.
                limit = (0.03 if which == "dp4a" else 1e-2 if which == "tiled_half"
                         else 1e-3)
                return float(np.abs(got - reference[0]).max()) / scale < limit

            execution = _weight_execution(
                "ggml", type_name, K, N, m, run, candidates=candidates, check=correct)
        else:
            execution = "stored"
    if execution in ("tiled", "tiled_half"):
        of = _ggml_tiled_matmul(xf, packed, type_name, K, N, half=execution == "tiled_half")
        return of if bias is None else of + bias
    if execution == "dp4a":
        of = (_q4k_dp4a_matmul(xf, packed, K, N) if type_name == "Q4_K"
              else _q6k_dp4a_matmul(xf, packed, K, N))
        return of if bias is None else of + bias
    if execution == "materialized":
        # The measured policy above has found enough rows to pay for unpacking once.
        # Bit-exact against the quantised kernel; the only difference is fp32 rounding in
        # the accumulation order.
        #
        # The tiled fp32 kernel only runs on a row count that is a multiple of
        # `_MATMUL_ROW_ALIGN`, and missing it costs seven times. That alignment belongs to
        # THE MATMUL, so it is met here rather than by lengthening the prompt: padding the
        # prompt makes every layer see the extra positions, and a recurrent layer advances
        # its state once per position -- which is how a stack that is mostly recurrent came
        # to be excluded from this path entirely, and lost it.
        #
        # The rows are added with the GPU concatenate, NOT by assigning into a preallocated
        # buffer. `xp[:m] = xf` is WgPy's `__setitem__`, which reads the whole thing back to
        # the host and uploads it again -- 11.5MB at 0.8 GB/s, 14.2ms a call, and prefill
        # calls this ~200 times. That is the version of this idea that measured 3.4s.
        pad = (-m) % _MATMUL_ROW_ALIGN
        xin = Tensor(xf)
        if pad:
            xin = cat([xin, Tensor(_empty((pad, int(K))))], axis=0)
        w32 = Tensor(ggml_dequant(packed, type_name, K, N))
        of = (xin @ w32).data
        if pad:
            of = _contig(of[:m])            # the padded rows were arithmetic, not an answer
        return of if bias is None else of + bias
    small = ((_ggml_shape_for(type_name, N, K, packed)
              if shape_execution == "auto" else shape_execution)
             if mode == 1 else None)
    moe = eidx is not None
    # Rows per thread is chosen from the batch here, so a short prefill and a long one get
    # different kernels rather than one compromise that is wrong at both ends. It is part of
    # the variant key for the same reason the thread shape is: two kernels that differ only
    # in a compile-time constant are two pipelines, and sharing a name between them means
    # the second silently runs the first.
    mrow = _ggml_mrow(_GGML_TYPES[type_name][2], m) if mode == 0 else None
    key = (type_name, mode, small, moe, (_GGML_KSG, mrow) if mode == 0 else 0)
    if key not in _ggml_k["added"]:
        _ggml_add(type_name, mode, small, moe, mrow)
        _ggml_k["added"].add(key)           # set before the check: it calls back in here
        _ggml_selfcheck(type_name, mode, small, moe, mrow)
    of = _ggml_run(xf, packed, type_name, K, N, small=small,
                   eidx=eidx, eslot=eslot, estride=estride, xper=xper)
    return of if bias is None else of + bias


# One activation feeding several independent projections is one operation at the transformer
# layer even though a plain implementation launches one matmul per weight.  The WebGPU path
# below binds the original packed buffers side by side and assigns output-row ranges to them
# inside ONE stored-format kernel.  No weight is concatenated, copied, dequantized or changed
# in width.  WebGL exposes the same parallel_linear contract and evaluates its projections
# separately at the nearest efficient layer, because a fragment pass has exactly one output.
_GGML_PARALLEL_ADDED = set()
_GGML_PARALLEL_OK = {}


def _ggml_parallel_src(type_name, count, kind=None):
    if count not in (2, 3):
        raise ValueError("parallel stored kernel supports two or three projections")
    if _GGML_TYPES[type_name][4] is not None:
        raise RuntimeError("parallel stored kernel has no spare binding for a codebook")
    src = _ggml_src(type_name, 1, _cfg_for(kind, _GGML_TYPES[type_name][2]))
    base = _GGML_BIND.replace("MOEVARS", "").replace("WOFS", "")
    if not src.startswith(base):
        raise RuntimeError("ggml source binding prefix changed")
    marker = "// Byte addressing"
    helpers = base[base.index(marker):]
    old_w = "fn W(wo: u32) -> u32 { return w[wo * gm.N + nrow]; }"
    limits = ["gm.estride", "gm.eslot", "gm.xper"]
    reads = []
    stores = []
    off = "0u"
    for i in range(count):
        ni = limits[i]
        reads.append("  if (nrow < (%s) + %s) { return w%d[wo * %s + nrow - (%s)]; }"
                     % (off, ni, i, ni, off))
        stores.append("  if (row < (%s) + %s) { out%d[row - (%s)] = v; return; }"
                      % (off, ni, i, off))
        off = "%s + %s" % (off, ni)
    new_w = "fn W(wo: u32) -> u32 {\n%s\n  return 0u;\n}" % "\n".join(reads)
    store = "fn STORE(row: u32, v: f32) {\n%s\n}" % "\n".join(stores)
    helpers = helpers.replace(old_w, new_w + "\n" + store)
    binds = ["@group(0) @binding(0)\nvar<storage,read> x: array<f32>;"]
    for i in range(count):
        binds.append("@group(0) @binding(%d)\nvar<storage,read> w%d: array<u32>;"
                     % (1 + i, i))
    for i in range(count):
        binds.append("@group(0) @binding(%d)\nvar<storage,read_write> out%d: array<f32>;"
                     % (1 + count + i, i))
    meta_binding = 1 + 2 * count
    binds.append("struct GM { M: u32, N: u32, K: u32, rowb: u32, estride: u32, "
                 "eslot: u32, xper: u32, pad: u32, }\n"
                 "@group(0) @binding(%d)\nvar<storage,read> gm: GM;" % meta_binding)
    custom = "\n".join(binds) + "\n" + helpers
    src = custom + src[len(base):]
    if "outp[nn] = tot;" not in src:
        raise RuntimeError("ggml stored output statement changed")
    return src.replace("outp[nn] = tot;", "STORE(nn, tot);")


def _ggml_parallel_fused(xd, linears, kind=None):
    count = len(linears)
    type_name = linears[0].type_name
    K = int(linears[0].Kt)
    ns = tuple(int(l.Nt) for l in linears)
    key = (type_name, count, kind)
    plat = _adam_kernel["platform"]
    name = "ggml_parallel%d_%s%s" % (
        count, type_name.lower(), "" if kind is None else "_" + str(kind))
    if key not in _GGML_PARALLEL_ADDED:
        ro, rw = "read-only-storage", "storage"
        plat.addKernel(name, {"source": _ggml_parallel_src(type_name, count, kind),
                              "bindingTypes": [ro] + [ro] * count + [rw] * count + [ro]})
        _GGML_PARALLEL_ADDED.add(key)
    outs = [_empty((1, n)) for n in ns]
    padded = ns + (0,) * (3 - count)
    meta = _adam_kernel["make_meta"]((1, sum(ns), K, 0, padded[0], padded[1], padded[2], 0),
                                     "u4,u4,u4,u4,u4,u4,u4,u4")
    tensors = ([xd.buffer.buffer_id]
               + [l.packed.buffer.buffer_id for l in linears]
               + [o.buffer.buffer_id for o in outs] + [meta.buffer_id])
    vals = _GGML_TYPES[type_name][2]
    plat.runKernel({"name": name, "tensors": tensors,
                    "workGroups": {"x": _gemv_groups(sum(ns), 1, vals, kind),
                                   "y": 1, "z": 1}})
    return tuple(Tensor(o) for o in outs)


def parallel_linear(linears, x, execution="auto"):
    """Run independent projections from one activation, returning one Tensor per weight.

    ``auto`` is the best route at this layer.  A containing transformer may pass
    ``separate`` or ``fused`` when its complete-layer or complete-API combination measures
    faster.  Unsupported formats/backends keep the same interface and use the separate
    equivalent; no model name or model category participates in the decision.
    """
    linears = tuple(linears)
    fused_modes = ("fused:default", "fused:balanced", "fused:compact", "fused:narrow")
    if execution not in ("auto", "separate", "fused") + fused_modes:
        raise ValueError("execution must be auto, separate, fused, or a fused:<shape> route")
    separate = lambda: tuple(layer(x) for layer in linears)
    xd = x.data
    rows = int(np.prod(xd.shape[:-1])) if xd.ndim > 1 else 1
    capable = (2 <= len(linears) <= 3 and rows == 1 and _adam_backend_ready()
               and all(isinstance(l, GGMLLinear) for l in linears)
               and len({l.type_name for l in linears}) == 1
               and len({int(l.Kt) for l in linears}) == 1
               and all(l.bias is None for l in linears)
               and _GGML_TYPES[linears[0].type_name][4] is None)
    if not capable:
        # This is the nearest common interface level.  WebGL has one output per fragment
        # pass, and mixed storage formats need different decode fragments, so neither can
        # share the physical dispatch.  They still implement the same parallel-projection
        # operation here with separate passes; callers do not lose the layer capability.
        return separate()
    xd = _contig(xd.reshape(1, int(linears[0].Kt)))
    ns = tuple(int(l.Nt) for l in linears)
    kinds = (None, "balanced", "compact", "narrow")
    reference = [None]

    def fused_kind(kind):
        return _ggml_parallel_fused(xd, linears, kind=kind)

    def correct_fused(kind):
        ckey = ("parallel_correct", linears[0].type_name, int(linears[0].Kt), ns, kind)
        if ckey in _GGML_PARALLEL_OK:
            return _GGML_PARALLEL_OK[ckey]
        if reference[0] is None:
            reference[0] = tuple(ggml_matmul(
                xd, l.packed, l.type_name, l.Kt, l.Nt, execution="stored")
                                 for l in linears)
            reference[0] = tuple(np.asarray(v.get(), np.float32) for v in reference[0])
        got = fused_kind(kind); ok = True
        for av, b in zip(reference[0], got):
            bv = np.asarray(b.numpy(), np.float32)
            scale = max(1e-6, float(np.abs(av).max()))
            ok = ok and bool(np.all(np.isfinite(bv))
                             and float(np.abs(av - bv).max()) / scale < 2e-5)
        _GGML_PARALLEL_OK[ckey] = ok
        return ok

    def best_fused_kind():
        key = ("parallel_fused_kind", linears[0].type_name,
               int(linears[0].Kt), ns)
        if key in _TUNED:
            return _TUNED[key]
        valid = [kind for kind in kinds if correct_fused(kind)]
        if not valid:
            raise RuntimeError("no fused stored-format projection passed validation")
        samples = {kind: [] for kind in valid}

        def bench(kind):
            out = None; t0 = time.perf_counter()
            for _ in range(4):
                out = fused_kind(kind)
            out[-1].numpy()
            return (time.perf_counter() - t0) / 4.0

        for kind in valid:
            bench(kind)
        for r in range(9):
            order = valid if not (r & 1) else list(reversed(valid))
            for kind in order:
                samples[kind].append(bench(kind))
        chosen = _measured_choice(samples, valid, default=None)
        _TUNED[key] = chosen
        return chosen

    def fused():
        return fused_kind(best_fused_kind())

    if execution.startswith("fused:"):
        kind = None if execution == "fused:default" else execution.split(":", 1)[1]
        if not correct_fused(kind):
            raise RuntimeError("fused stored-format projection shape failed validation")
        return fused_kind(kind)
    if execution == "fused":
        return fused()
    if execution == "separate":
        return separate()

    key = ("parallel_linear", linears[0].type_name, int(linears[0].Kt),
           tuple(int(l.Nt) for l in linears))
    chosen = _TUNED.get(key)
    if chosen is None:
        candidates = ["separate"] + (["fused"] if any(correct_fused(k) for k in kinds)
                                      else [])
        samples = {c: [] for c in candidates}

        def bench(which):
            out = None; t0 = time.perf_counter()
            for _ in range(4):
                out = fused() if which == "fused" else separate()
            out[-1].numpy()                # queue order completes every sibling projection
            return (time.perf_counter() - t0) / 4.0

        for c in candidates:
            bench(c)
        for r in range(9):
            order = candidates if not (r & 1) else list(reversed(candidates))
            for c in order:
                samples[c].append(bench(c))
        chosen = _measured_choice(samples, candidates, default="separate")
        _TUNED[key] = chosen
    return fused() if chosen == "fused" else separate()


def parallel_swiglu(linears, x, execution="auto"):
    """Two parallel projections followed by SwiGLU, optimised at the MLP layer.

    ``parallel_linear`` remains independently callable and returns ordinary tensors.  This
    next layer may use a different combination: on WebGL, same-format gate/up weights can
    render one combined projection texture and :func:`swiglu` consumes its two halves in
    place, avoiding both a second projection draw and materialised slices.  WebGPU uses its
    own shared-dispatch projection.  Unsupported or mixed formats take the equivalent
    separate path at this nearest common layer.
    """
    linears = tuple(linears)
    if len(linears) != 2:
        raise ValueError("parallel_swiglu needs exactly gate and up projections")
    fused_modes = ("fused:default", "fused:balanced", "fused:compact", "fused:narrow")
    if execution not in ("auto", "separate", "fused") + fused_modes:
        raise ValueError("execution must be auto, separate, fused, or a fused:<shape> route")

    def separate():
        gate, up = parallel_linear(linears, x, execution="separate")
        out = swiglu(gate, up)
        return out if out is not None else (gate / (1.0 + (-gate).exp())) * up

    xd = x.data
    rows = int(np.prod(xd.shape[:-1])) if xd.ndim > 1 else 1
    capable = (rows == 1 and all(isinstance(l, GGMLLinear) for l in linears)
               and len({l.type_name for l in linears}) == 1
               and len({int(l.Kt) for l in linears}) == 1
               and len({int(l.Nt) for l in linears}) == 1
               and all(l.bias is None for l in linears)
               and _GGML_TYPES[linears[0].type_name][4] is None
               and not any(isinstance(l.packed, WebGLQ8Matrix) for l in linears)
               and (_adam_backend_ready() or _webgl_ready()))
    if not capable:
        return separate()
    xd = _contig(xd.reshape(1, int(linears[0].Kt)))

    def fused(route="fused"):
        if _webgl_ready() and not _adam_backend_ready():
            combined = _ggml_parallel_fused_gl(xd, linears)
            return swiglu(combined)
        gate, up = parallel_linear(linears, Tensor(xd), execution=route)
        return swiglu(gate, up)

    if execution == "separate":
        return separate()
    # ``fused`` is an upper-layer request, not permission to skip the numerical gate.  The
    # first call compares the complete activation, then the result is remembered per backend
    # and operator shape.
    backend_name = "webgpu" if _adam_backend_ready() else "webgl"
    def correct_fused(route="fused"):
        ckey = ("parallel_swiglu_correct", backend_name, linears[0].type_name,
                int(linears[0].Kt), int(linears[0].Nt), route)
        if ckey in _TUNED:
            return bool(_TUNED[ckey])
        a = np.asarray(separate().numpy(), np.float32)
        b = np.asarray(fused(route).numpy(), np.float32)
        scale = max(1e-6, float(np.abs(a).max()))
        ok = bool(np.all(np.isfinite(b))
                  and float(np.abs(a - b).max()) / scale < 2e-5)
        _TUNED[ckey] = ok
        return ok

    if execution.startswith("fused:"):
        # WebGL has no workgroup decomposition at this physical layer; its nearest-layer
        # fused equivalent is nevertheless the same SwiGLU operation and is validated here.
        return fused(execution) if correct_fused(execution) else separate()
    if execution == "fused":
        return fused() if correct_fused() else separate()

    key = ("parallel_swiglu", backend_name, linears[0].type_name,
           int(linears[0].Kt), int(linears[0].Nt))
    chosen = _TUNED.get(key)
    if chosen is None:
        candidates = ["separate"] + (["fused"] if correct_fused() else [])
        samples = {c: [] for c in candidates}

        def bench(which):
            out = None; t0 = time.perf_counter()
            for _ in range(4):
                out = fused() if which == "fused" else separate()
            out.numpy()
            return (time.perf_counter() - t0) / 4.0

        for c in candidates:
            bench(c)
        for r in range(9):
            order = candidates if not (r & 1) else list(reversed(candidates))
            for c in order:
                samples[c].append(bench(c))
        chosen = _measured_choice(samples, candidates, default="separate")
        _TUNED[key] = chosen
    return fused() if chosen == "fused" else separate()


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


def _ggml_add(type_name, mode, small=None, moe=False, mrow=None):
    plat = _adam_kernel["platform"]
    binds = ["read-only-storage", "read-only-storage", "storage", "read-only-storage"]
    if moe:
        binds.append("read-only-storage")               # the expert index, before the grid
    if _GGML_TYPES[type_name][4] is not None:
        binds.append("read-only-storage")
    cfg = _cfg_for(small, _GGML_TYPES[type_name][2])
    plat.addKernel(_ggml_name(type_name, mode, small=small, moe=moe, mrow=mrow),
                   {"source": _ggml_src(type_name, mode, cfg, moe=moe, mrow=mrow),
                    "bindingTypes": binds})


def _ggml_name(type_name, mode, orw=None, small=None, moe=False, mrow=None):
    # ORW and the thread shape are compile-time constants in the shader, so a kernel is
    # identified by them too -- otherwise a second variant would silently reuse the first
    # one's pipeline.
    o = _orw_for(mode) if orw is None else orw
    # KSG is a compile-time constant of the BATCHED kernel -- it sets the workgroup's second
    # dimension and the size of the staged activations -- so it belongs in the name for the
    # same reason ORW does. Without it a second value silently reuses the first's pipeline,
    # and every measurement of the second is a measurement of the first.
    return "ggml%s_%s%s%s%s%s" % (("v", "v2", "", "deq")[mode], type_name.lower(),
                                  "" if o <= 1 else "_r%d" % o,
                                  "_%s" % small if small else "",
                                  "_e" if moe else "",
                                  "" if mode != 0 else
                                  ("" if _GGML_KSG == 1 else "_g%d" % _GGML_KSG)
                                  + ("" if (mrow or 4) == 4 else "_m%d" % mrow))


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


# Greedy language-model decoding needs one integer, not a 150k-float vocabulary copied to
# WASM every token.  One workgroup scans the row and reduces it deterministically (ties use
# the first index, matching numpy.argmax).  Keeping the reduction on-device removes both the
# full readback and the CPU argmax from the common no-sampling path.
_VOCAB_ARGMAX_WGSL = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> out_idx: array<i32>;
struct AM { n: u32, offset: u32, pad1: u32, pad2: u32, }
@group(0) @binding(2) var<storage,read> am: AM;
var<workgroup> best_v: array<f32, 256>;
var<workgroup> best_i: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var v = -3.402823466e+38;
  var bi = 0u;
  for (var i = t; i < am.n; i = i + 256u) {
    let q = x[i];
    if (q > v || (q == v && i < bi)) { v = q; bi = i; }
  }
  best_v[t] = v; best_i[t] = bi;
  workgroupBarrier();
  var width = 128u;
  loop {
    if (width == 0u) { break; }
    if (t < width) {
      let ov = best_v[t + width]; let oi = best_i[t + width];
      if (ov > best_v[t] || (ov == best_v[t] && oi < best_i[t])) {
        best_v[t] = ov; best_i[t] = oi;
      }
    }
    workgroupBarrier();
    width = width / 2u;
  }
  if (t == 0u) { out_idx[am.offset] = i32(best_i[0]); }
}
"""

_VOCAB_ARGMAX_GLSL = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform sampler2D tex_x; uniform int u_n;
out int fragColor;
float Xf(int i) { ivec2 s = textureSize(tex_x, 0); int y = i / s.x;
  return texelFetch(tex_x, ivec2(i - y * s.x, y), 0).r; }
void main() {
  float best = -3.402823466e+38; int bi = 0;
  for (int i = 0; i < u_n; i = i + 1) {
    float v = Xf(i); if (v > best) { best = v; bi = i; }
  }
  fragColor = bi;
}
"""

_vocab_argmax_k = {"gpu": False}
def vocab_argmax(x, out=None, offset=0):
    """Return a device-resident int32[1] containing the first argmax of one flat row.

    WebGPU uses a workgroup reduction and WebGL provides the same interface with a single
    fragment scan.  CPU remains the reference fallback.  The caller decides when greedy
    argmax is semantically valid; sampling, penalties and constraints still use full logits.
    """
    xd = _contig(x.data if isinstance(x, Tensor) else x).reshape(-1)
    n = int(xd.size)
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if not _vocab_argmax_k["gpu"]:
            plat.addKernel("vocab_argmax", {"source": _VOCAB_ARGMAX_WGSL,
                "bindingTypes": ["read-only-storage", "storage", "read-only-storage"]})
            _vocab_argmax_k["gpu"] = True
        out = xp.empty((1,), np.int32) if out is None else out
        if int(offset) < 0 or int(offset) >= int(out.size):
            raise ValueError("vocab_argmax output offset is outside its buffer")
        meta = _adam_kernel["make_meta"]((n, int(offset), 0, 0), "u4,u4,u4,u4")
        plat.runKernel({"name": "vocab_argmax",
                        "tensors": [xd.buffer.buffer_id, out.buffer.buffer_id,
                                    meta.buffer_id],
                        "workGroups": {"x": 1, "y": 1, "z": 1}})
        return out
    if _webgl_ready():
        if out is not None or int(offset):
            raise RuntimeError("WebGL vocab_argmax writes its complete one-value output")
        out = xp.empty((1,), np.int32)
        return _gl_run("vocab_argmax_gl", _VOCAB_ARGMAX_GLSL,
                       [("tex_x", xd)], out, [("u_n", n)])
    return np.asarray([int(np.asarray(xd).argmax())], np.int32)


# Device-side input preparation for a tied Q6_K embedding/head.  It is deliberately an
# original-format operation: one selected source row is reconstructed directly from the
# transposed 210-byte blocks already used by the vocabulary head.  The packed weight is not
# widened or duplicated.  A containing greedy decoder can place several of these between
# captured decode steps and cross the JS/WASM/GPU boundary once for the whole chunk.
_Q6K_DECODE_INPUT_WGSL = """
@group(0) @binding(0) var<storage,read> tokens: array<i32>;
@group(0) @binding(1) var<storage,read> w: array<u32>;
@group(0) @binding(2) var<storage,read> cos_table: array<f32>;
@group(0) @binding(3) var<storage,read> sin_table: array<f32>;
@group(0) @binding(4) var<storage,read_write> h_out: array<f32>;
@group(0) @binding(5) var<storage,read_write> cos_out: array<f32>;
@group(0) @binding(6) var<storage,read_write> sin_out: array<f32>;
@group(0) @binding(7) var<storage,read_write> ctl: array<i32>;
struct IM { K:u32, N:u32, HD:u32, step:u32, inc:u32, pad0:u32, pad1:u32, pad2:u32, }
@group(0) @binding(8) var<storage,read> im: IM;
var<private> nrow:u32;
fn W(wo:u32)->u32 { return w[wo*im.N+nrow]; }
fn B4(o:u32)->u32 { let wo=o>>2u; let sh=(o&3u)*8u; let lo=W(wo);
  if(sh==0u){return lo;} return (lo>>sh)|(W(wo+1u)<<(32u-sh)); }
fn B(o:u32)->u32 { return (W(o>>2u)>>((o&3u)*8u))&255u; }
fn U16(o:u32)->u32 { return B4(o)&65535u; }
fn HF(h:u32)->f32 { let m=h&1023u; let e=(h>>10u)&31u; var v:f32;
  if(e==0u){v=f32(m)*5.9604644775390625e-8;} else if(e==31u){v=65504.0;}
  else {v=exp2(f32(i32(e)-15))*(1.0+f32(m)*0.0009765625);}
  return select(v,-v,(h&32768u)!=0u); }
fn I8(o:u32)->f32 { return f32(i32(B(o)<<24u)>>24u); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let k=gid.x; nrow=u32(tokens[im.step]);
  if(k<im.K){
    let b=k/256u; let p=k-b*256u; let half=p/128u; let r=(p-half*128u)/32u;
    let l=p&31u; let ii=l>>4u; let o=b*210u;
    let lo=o+half*64u; let ho=o+128u+half*32u; let so=o+192u+half*8u;
    let a=B(lo+l); let c=B(lo+l+32u); let hh=B(ho+l); var q:u32;
    if(r==0u){q=(a&15u)|(((hh>>0u)&3u)<<4u);}
    else if(r==1u){q=(c&15u)|(((hh>>2u)&3u)<<4u);}
    else if(r==2u){q=(a>>4u)|(((hh>>4u)&3u)<<4u);}
    else {q=(c>>4u)|(((hh>>6u)&3u)<<4u);}
    h_out[k]=HF(U16(o+208u))*I8(so+ii+2u*r)*(f32(q)-32.0);
  }
  if(k<im.HD){
    cos_out[k]=cos_table[im.step*im.HD+k];
    sin_out[k]=sin_table[im.step*im.HD+k];
  }
  if(k==0u && im.inc!=0u){ ctl[0]=ctl[0]+1; }
}
"""

# The same exact Q6_K decoder over the file's compact row-major bytes.  The vocabulary
# projection transposes those bytes once for coalesced matrix multiplication; selecting one
# embedding row has the opposite access pattern, so keeping the already-present compact
# table on the device can be much faster at the cost of one compressed-size duplicate.
# Derive the shader so the quant arithmetic cannot drift between the two candidates.
_Q6K_DECODE_ROW_INPUT_WGSL = _Q6K_DECODE_INPUT_WGSL.replace(
    "fn W(wo:u32)->u32 { return w[wo*im.N+nrow]; }",
    "fn W(wo:u32)->u32 { return w[nrow*im.N+wo]; }")
if _Q6K_DECODE_ROW_INPUT_WGSL == _Q6K_DECODE_INPUT_WGSL:
    raise RuntimeError("Q6_K compact-row shader derivation did not match")

_q6k_decode_input_added = {"transposed": False, "compact": False}


def q6k_decode_input(tokens, packed, K, N, cos_table, sin_table,
                     h_out, cos_out, sin_out, ctl, step=0, increment=False,
                     layout="transposed"):
    """Prepare one captured decode input from a device token and original Q6_K row."""
    if layout not in ("transposed", "compact"):
        raise ValueError("Q6_K row layout must be transposed or compact")
    if not _adam_backend_ready():
        raise RuntimeError("device Q6_K decode input requires WebGPU")
    plat = _adam_kernel["platform"]
    name = "q6k_decode_input" if layout == "transposed" else "q6k_decode_input_compact"
    if not _q6k_decode_input_added[layout]:
        plat.addKernel(name, {"source": (_Q6K_DECODE_INPUT_WGSL
                                          if layout == "transposed"
                                          else _Q6K_DECODE_ROW_INPUT_WGSL),
            "bindingTypes": ["read-only-storage"] * 4 + ["storage"] * 4
                            + ["read-only-storage"]})
        _q6k_decode_input_added[layout] = True
    td = tokens.data if isinstance(tokens, Tensor) else tokens
    wd = packed.data if isinstance(packed, Tensor) else packed
    cd = cos_table.data if isinstance(cos_table, Tensor) else cos_table
    sd = sin_table.data if isinstance(sin_table, Tensor) else sin_table
    hd = h_out.data if isinstance(h_out, Tensor) else h_out
    co = cos_out.data if isinstance(cos_out, Tensor) else cos_out
    so = sin_out.data if isinstance(sin_out, Tensor) else sin_out
    ct = ctl.data if isinstance(ctl, Tensor) else ctl
    HDr = int(co.size)
    meta = _adam_kernel["make_meta"](
        (int(K), int(N), HDr, int(step), 1 if increment else 0, 0, 0, 0),
        "u4,u4,u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": name,
                    "tensors": [td.buffer.buffer_id, wd.buffer.buffer_id,
                                cd.buffer.buffer_id, sd.buffer.buffer_id,
                                hd.buffer.buffer_id, co.buffer.buffer_id,
                                so.buffer.buffer_id, ct.buffer.buffer_id,
                                meta.buffer_id],
                    "workGroups": {"x": (int(K) + 255) // 256, "y": 1, "z": 1}})
    return h_out


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

_GEGLU_GLSL = _gl_head([("tex_x", "Xf")], ("u_half", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int r = i / u_half; int c = i - r * u_half;
  int base = r * (2 * u_half);
  float x = Xf(base + c);
  float u = clamp((x + 0.044715 * x * x * x) * 0.7978845608028654, -15.0, 15.0);
  fragColor = x * (tanh(u) + 1.0) * 0.5 * Xf(base + u_half + c);
}
"""

_QKV_TAKE_GLSL = _gl_head([("tex_x", "Xf"), ("tex_c", "Cf"), ("tex_s", "Sf")],
                          ("u_n", "u_T", "u_H", "u_HD", "u_which", "u_rope")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int d = i % u_HD;
  int t = (i / u_HD) % u_T;
  int bh = i / (u_HD * u_T);
  int b = bh / u_H;
  int head = bh - b * u_H;
  int D = u_H * u_HD;
  int base = (b * u_T + t) * 3 * D + u_which * D + head * u_HD;
  float x = Xf(base + d);
  if (u_rope == 0) { fragColor = x; return; }
  int halfD = u_HD / 2;
  float rot = d < halfD ? -Xf(base + d + halfD) : Xf(base + d - halfD);
  int ci = t * u_HD + d;
  fragColor = x * Cf(ci) + rot * Sf(ci);
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


def _webgl_prepare_growing_cache():
    """Compile the common KV append before a large model occupies GPU memory.

    Registration is a one-time scheduling request; shader compilation and all GPU work
    stay in the JavaScript backend. The normal first use sees the same kernel and skips
    registration. This is format- and model-independent for WebGL growing-cache decode.
    """
    if _webgl_ready() and "cat2_gl" not in _gl_kernels:
        _copy_kernel["plat"].addKernel("cat2_gl", {"source": _CAT2_GLSL})
        _gl_kernels.add("cat2_gl")


# A batched MoE routes each token to k experts.  Its activation must appear k times,
# but advanced ndarray indexing reads the entire activation back to Python before
# uploading the repeated rows.  The router has a device implementation too; using
# both device operations lets the *containing layer* submit all its work without a
# per-layer CPU/GPU synchronization.  Neither backend changes weight precision.
_REPEAT_ROWS_GLSL = _gl_head([("tex_x", "Xf")], ("u_H", "u_k", "u_n")) + """
void main() {
  int i = _idx();
  if (i >= u_n) { fragColor = 0.0; return; }
  int r = i / u_H;
  fragColor = Xf((r / u_k) * u_H + i - r * u_H);
}
"""

_REPEAT_ROWS_WGSL = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
struct RM { h: u32, k: u32, n: u32, pad: u32, }
@group(0) @binding(2) var<storage,read> rm: RM;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < rm.n) {
    let row = i / rm.h;
    y[i] = x[(row / rm.k) * rm.h + i - row * rm.h];
  }
}
"""
_repeat_rows_k = {"gpu": False}


def repeat_rows(x, repeats, execution="auto"):
    """Repeat each row of a 2-D FP32 activation, without changing its values.

    ``auto`` benchmarks the operation on this backend and shape.  A containing
    layer may explicitly request ``device``: its own best composition can differ
    from this isolated operator's winner because a host readback also serializes
    every earlier dispatch in that layer.
    """
    if execution not in ("auto", "host", "device"):
        raise ValueError("repeat_rows execution must be auto, host, or device")
    xd = x.data if isinstance(x, Tensor) else x
    if len(xd.shape) != 2:
        raise ValueError("repeat_rows requires a 2-D activation")
    k = int(repeats)
    if k < 1 or k != repeats:
        raise ValueError("repeat_rows requires a positive integer repeat count")
    if k == 1:
        return xd
    rows, h = (int(v) for v in xd.shape)
    backend = "webgpu" if _adam_backend_ready() else "webgl" if _webgl_ready() else "cpu"
    device_ok = backend != "cpu" and np.dtype(xd.dtype) == np.dtype(np.float32)

    def host_array(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    def run(which):
        if which == "host" or not device_ok:
            return xp.asarray(np.repeat(host_array(xd), k, axis=0))
        source = _contig(xd)
        out = _empty((rows * k, h))
        n = rows * k * h
        if backend == "webgl":
            return _gl_run("repeat_rows_gl", _REPEAT_ROWS_GLSL,
                           [("tex_x", source)], out,
                           [("u_H", h), ("u_k", k), ("u_n", n)])
        plat = _adam_kernel["platform"]
        if not _repeat_rows_k["gpu"]:
            plat.addKernel("repeat_rows", {
                "source": _REPEAT_ROWS_WGSL,
                "bindingTypes": ["read-only-storage", "storage", "read-only-storage"]})
            _repeat_rows_k["gpu"] = True
        meta = _adam_kernel["make_meta"]((h, k, n, 0), "u4,u4,u4,u4")
        plat.runKernel({"name": "repeat_rows",
                        "tensors": [source.buffer.buffer_id, out.buffer.buffer_id,
                                    meta.buffer_id],
                        "workGroups": {"x": (n + 255) // 256, "y": 1, "z": 1}})
        return out

    if execution == "device" and not device_ok:
        raise TypeError("device repeat_rows requires a GPU-backed FP32 activation")
    if execution == "auto" and device_ok:
        reference = [None]

        def correct(which):
            if which == "host":
                return True
            if reference[0] is None:
                reference[0] = np.repeat(host_array(xd), k, axis=0)
            got = host_array(run(which))
            return bool(np.array_equal(got, reference[0]))

        execution = _weight_execution("repeat_rows", backend, h, k, rows, run,
                                     candidates=("host", "device"), check=correct,
                                     rounds=5, repeat=2)
    return run("host" if execution == "auto" else execution)


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
uniform int u_ne; uniform int u_k; uniform int u_T;
out int fragColor;
float Lf(int i) { ivec2 t = textureSize(tex_lg, 0);
  if (t.y == 1) { return texelFetch(tex_lg, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_lg, ivec2(i - y * t.x, y), 0).r; }
void main() {
  int out_i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (out_i >= u_T * u_k) { fragColor = 0; return; }
  int row = out_i / u_k;
  int s = out_i - row * u_k;
  int base = row * u_ne;
  int chosen[%d];
  for (int t = 0; t <= s; t = t + 1) {
    float best = -1e30; int bi = 0;
    for (int e = 0; e < u_ne; e = e + 1) {
      bool taken = false;
      for (int j = 0; j < t; j = j + 1) { if (chosen[j] == e) { taken = true; } }
      float v = Lf(base + e);
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
uniform int u_ne; uniform int u_k; uniform int u_norm; uniform int u_T;
out float fragColor;
float Lf(int i) { ivec2 t = textureSize(tex_lg, 0);
  if (t.y == 1) { return texelFetch(tex_lg, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_lg, ivec2(i - y * t.x, y), 0).r; }
int If(int i) { ivec2 t = textureSize(tex_idx, 0);
  if (t.y == 1) { return texelFetch(tex_idx, ivec2(i, 0), 0).r; }
  int y = i / t.x; return texelFetch(tex_idx, ivec2(i - y * t.x, y), 0).r; }
void main() {
  int out_i = int(gl_FragCoord.x) + int(gl_FragCoord.y) * _ka_tex_output_texture_w;
  if (out_i >= u_T * u_k) { fragColor = 0.0; return; }
  int row = out_i / u_k;
  int s = out_i - row * u_k;
  int base = row * u_ne;
  // Normalised top-k cancels the full softmax denominator. Retain it only when
  // the caller asks for unnormalised full-expert probabilities.
  if (u_norm == 1) {
    float mx_sel = -1e30;
    for (int j = 0; j < u_k; j = j + 1) {
      mx_sel = max(mx_sel, Lf(base + If(row * u_k + j)));
    }
    float den_sel = 0.0;
    for (int j = 0; j < u_k; j = j + 1) {
      den_sel += exp(Lf(base + If(row * u_k + j)) - mx_sel);
    }
    fragColor = exp(Lf(base + If(out_i)) - mx_sel) / den_sel;
    return;
  }
  float mx = -1e30;
  for (int e = 0; e < u_ne; e = e + 1) { mx = max(mx, Lf(base + e)); }
  float den = 0.0;
  for (int e = 0; e < u_ne; e = e + 1) { den += exp(Lf(base + e) - mx); }
  fragColor = exp(Lf(base + If(out_i)) - mx) / den;
}
"""


def _webgl_moe_route(logits, eidx, ew, ne, k, norm):
    # A fragment holds its picks in a fixed local array, so a larger k would index past it
    # and route to whatever was in that slot -- a wrong expert, silently. Loud instead.
    if int(k) > _MOE_TOPK:
        raise RuntimeError("WebGL MoE routing is built for up to %d experts per token, "
                           "not %d -- raise _MOE_TOPK" % (_MOE_TOPK, int(k)))
    rows = int(logits.shape[0]) if len(logits.shape) == 2 else 1
    _gl_run("moe_idx_gl", _MOE_IDX_GL, [("tex_lg", logits)], eidx,
            [("u_ne", int(ne)), ("u_k", int(k)), ("u_T", rows)])
    _gl_run("moe_w_gl", _MOE_W_GL, [("tex_lg", logits), ("tex_idx", eidx)], ew,
            [("u_ne", int(ne)), ("u_k", int(k)), ("u_norm", 1 if norm else 0),
             ("u_T", rows)])


_MOE_REDUCE_GL = _gl_head([("tex_y", "Yf"), ("tex_w", "Wf")],
                          ("u_rows", "u_k", "u_h")) + """
void main() {
  int i = _idx();
  if (i >= u_rows * u_h) { fragColor = 0.0; return; }
  int row = i / u_h;
  int col = i - row * u_h;
  float acc = 0.0;
  for (int slot = 0; slot < u_k; slot = slot + 1) {
    int p = row * u_k + slot;
    acc += Yf(p * u_h + col) * Wf(p);
  }
  fragColor = acc;
}
"""


def _webgl_moe_weighted_sum(yd, wd, rows, k, h):
    out = _empty((rows, h))
    _gl_run("moe_weighted_sum_gl", _MOE_REDUCE_GL,
            [("tex_y", yd), ("tex_w", wd)], out,
            [("u_rows", rows), ("u_k", k), ("u_h", h)])
    return Tensor(out)


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


def _webgl_geglu_split(xd, rows, half):
    of = _empty((rows, half))
    return _gl_run("geglu_gl", _GEGLU_GLSL, [("tex_x", xd)], of,
                   [("u_half", half), ("u_n", rows * half)])


def _webgl_qkv_take(xd, cd, sd, n, T, H, HD, which, use_rope):
    of = _empty((n,))
    return _gl_run("qkv_take_gl", _QKV_TAKE_GLSL,
                   [("tex_x", xd), ("tex_c", cd), ("tex_s", sd)], of,
                   [("u_n", n), ("u_T", T), ("u_H", H), ("u_HD", HD),
                    ("u_which", int(which)), ("u_rope", int(use_rope))])


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


def _ggml_src_gl(type_name, moe, moedec, bias=False, mode=1, exact_route="selected"):
    """GLSL ES 3.00 for one (format, routing, bias) combination.

    The bias is a compile-time variant rather than a second dispatch. It is one fetch at the
    end of a fragment that has already read a whole row of the weight, and it removes three
    launches per layer on the backend where a launch is worth the most -- a 36-layer model
    was spending more than a hundred of them a token on `out = out + bias`."""
    from . import _wgsl2glsl as w2g
    if exact_route not in ("selected", "base", "alternate"):
        raise ValueError("WebGL exact route must be 'selected', 'base', or 'alternate'")
    dec, helpers, vals, _, _ = _GGML_TYPES[type_name]
    override = _GGML_GL_MODE_DECODERS.get(type_name, {}).get(mode)
    if exact_route in ("selected", "alternate") and override is not None:
        dec, helpers = override
    elif exact_route == "alternate":
        raise ValueError("%s mode %s has no alternate exact WebGL decoder" %
                         (type_name, mode))
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
                   "acc0": "float", "gm": "GM", "xrow": "uint", "woff": "uint",
                   # The translator rewrites storage indexing to Gf/Ef calls, but type
                   # inference runs before that rewrite and must still know the element
                   # type.  Without these declarations every IQ codebook shader failed to
                   # generate on WebGL even though its WebGPU counterpart passed.
                   "gr": "uint", "eidx": "int"})
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

# The same kernel reading its activations four to an RGBA32F texel (`_webgl_pack_x4`).
# `_GL_HEAD`'s note records that this LOST on the one-row decode sweep, where the single
# activation row is cached and a scalar fetch is cheaper than a vec4 fetch plus a dynamic
# index -- so it is not a replacement. It is a candidate for the batched modes, where a
# fragment reads a whole activation row of its own: four fetches become one, and the race in
# `_ggml_run_gl` decides per format, shape and row bucket on the device.
_GL_PACKED_X = (
    ("uniform sampler2D tex_x;", "uniform sampler2D tex_xp; uniform int u_R; uniform int u_K4;"),
    ("int _xw; int _ww;", "int _ry; int _bx; int _xw; int _ww;"),
    ("  _xw = textureSize(tex_x, 0).x; _xh = textureSize(tex_x, 0).y;\n", ""),
    ("void ACC(uint k, float v) { acc0 += Xf(int(xrow + k)) * v; }",
     "void ACC(uint k, float v) {\n"
     "  acc0 += texelFetch(tex_xp, ivec2(_bx + int(k >> 2u), _ry), 0)[int(k & 3u)] * v; }"),
    ("void ACC4(uint k, vec4 v) { int i = int(xrow + k);\n"
     "  acc0 += dot(vec4(Xf(i), Xf(i + 1), Xf(i + 2), Xf(i + 3)), v); }",
     "void ACC4(uint k, vec4 v) {\n"
     "  acc0 += dot(texelFetch(tex_xp, ivec2(_bx + int(k >> 2u), _ry), 0), v); }"),
    ("  woff = 0u;\n  xrow = uint(r) * gm.K;",
     "  woff = 0u;\n  xrow = uint(r) * gm.K;\n  _ry = r / u_R; _bx = (r - _ry * u_R) * u_K4;"),
)


def _ggml_packed_x_src(src):
    """`src` (a dense batched GL kernel) reading packed activations; every substitution
    must apply, so a change to the template is an error here rather than a kernel that
    still reads `tex_x`."""
    for old, new in _GL_PACKED_X:
        if old not in src:
            raise RuntimeError("WebGL GGML template changed; packed-activation variant "
                               "cannot be derived (%r)" % old[:40])
        src = src.replace(old, new)
    # Nothing else may read the scalar activation texture.
    i = src.index("float Xf(int i)")
    j = src.index("\n}\n", i) + 3
    return src[:i] + src[j:]


def _ggml_name_gl(type_name, moe, moedec, bias=False, mode=1, exact_route="base"):
    return "ggml_gl_%s_%s_m%d_%s%s" % (type_name.lower().replace("-", "_"),
                                       "md" if moedec else ("mb" if moe else "d"), mode,
                                       {"alternate": "xa", "packed": "xp"}.get(exact_route,
                                                                                "xb"),
                                       "_b" if bias else "")


def _ggml_add_gl(type_name, moe=False, moedec=False, bias=False, mode=1,
                 exact_route="base"):
    key = (type_name, moe, moedec, bias, mode, exact_route)
    if key in _ggml_gl["added"]:
        return
    plat = _copy_kernel["plat"]
    if exact_route == "packed":
        src = _ggml_packed_x_src(_ggml_src_gl(type_name, moe, moedec, bias, mode, "selected"))
    else:
        src = _ggml_src_gl(type_name, moe, moedec, bias, mode, exact_route)
    plat.addKernel(_ggml_name_gl(type_name, moe, moedec, bias, mode, exact_route),
                   {"source": src})
    _ggml_gl["added"].add(key)


def _ggml_run_gl_exact(xf, packed, type_name, K, N, eidx=None, eslot=0, estride=0,
                       xper=False, bias=None, exact_route="base"):
    """One exact stored-width WebGL dispatch for an explicitly selected decoder."""
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    moe = eidx is not None
    M = 1 if (moe and xper) else int(xf.shape[0])
    moedec = moe and M <= 2
    mode = M if M <= 2 else (3 if M <= 32 else 0)
    slots = int(eidx.size) if moedec else 1
    rows = slots * M
    _ggml_add_gl(type_name, moe, moedec, bias is not None, mode, exact_route)
    of = _empty((rows, N))
    grid = _ggml_grid(type_name)
    extra = []
    if exact_route == "packed":
        xp_buf, per_row, _ = _webgl_pack_x4(_contig(xf), int(K))
        inputs = [{"name": "tex_xp", "id": xp_buf.buffer_id},
                  {"name": "tex_w", "id": packed.buffer.buffer_id}]
        extra = [{"name": "u_R", "value": int(per_row), "type": "int"},
                 {"name": "u_K4", "value": int(K) // 4, "type": "int"}]
    else:
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
    plat.runKernel({"name": _ggml_name_gl(type_name, moe, moedec, bias is not None, mode,
                                           exact_route),
                    "inputs": inputs, "output": of.buffer.buffer_id,
                    "uniforms": [U("_ka_tex_output_texture_w", of.buffer.texture_shape.width),
                                 U("u_M", M), U("u_N", N), U("u_K", K),
                                 U("u_rowb", (K // vals) * blk), U("u_estride", estride),
                                 U("u_eslot", eslot), U("u_xper", 1 if xper else 0),
                                 U("u_ROWS", rows)] + extra})
    return of


_GGML_GL_PARALLEL_ADDED = set()


def _ggml_parallel_src_gl(type_name, count, exact_route="base"):
    """Turn the dense exact decoder into one combined-output WebGL projection."""
    if count not in (2, 3):
        raise ValueError("WebGL combined projection supports two or three weights")
    src = _ggml_src_gl(type_name, False, False, False, 1, exact_route)
    old_sampler = "uniform isampler2D tex_w;"
    samplers = "\n".join("uniform isampler2D tex_w%d;" % i for i in range(count))
    if old_sampler not in src:
        raise RuntimeError("WebGL GGML weight sampler declaration changed")
    src = src.replace(old_sampler, samplers)
    old_width = "  _ww = textureSize(tex_w, 0).x;"
    widths = "\n".join("  _ww%d = textureSize(tex_w%d, 0).x;" % (i, i)
                       for i in range(count))
    if old_width not in src:
        raise RuntimeError("WebGL GGML weight width setup changed")
    src = src.replace(old_width, widths)
    src = src.replace("int _xw; int _ww; int _gw; int _ew; int _xh;",
                      "int _xw; int _ww; int _gw; int _ew; int _xh; "
                      + " ".join("int _ww%d;" % i for i in range(count)))
    old_w = ("uint  W(uint wo) { int i = int(woff) + int(wo) * int(gm.N) + int(nrow);\n"
             "                   int y = i / _ww; return uint(texelFetch(tex_w, ivec2(i - y * _ww, y), 0).r); }")
    branches = []
    offset = "0"
    for i in range(count):
        ni = "u_N%d" % i
        branches.append(
            "  if (int(nrow) < (%s) + %s) { int r = int(nrow) - (%s); "
            "int z = int(wo) * %s + r; int y = z / _ww%d; "
            "return uint(texelFetch(tex_w%d, ivec2(z - y * _ww%d, y), 0).r); }"
            % (offset, ni, offset, ni, i, i, i))
        offset = "(%s) + %s" % (offset, ni)
    new_w = "uint W(uint wo) {\n%s\n  return 0u;\n}" % "\n".join(branches)
    if old_w not in src:
        raise RuntimeError("WebGL GGML W() helper changed")
    src = src.replace(old_w, new_w)
    decl = "uniform int u_ROWS;"
    extra = decl + " " + " ".join("uniform int u_N%d;" % i for i in range(count))
    if decl not in src:
        raise RuntimeError("WebGL GGML row uniform changed")
    return src.replace(decl, extra)


def _ggml_parallel_run_gl_exact(xd, linears, exact_route="base"):
    count = len(linears); type_name = linears[0].type_name
    K = int(linears[0].Kt); ns = tuple(int(l.Nt) for l in linears)
    key = (type_name, count, exact_route)
    name = "ggml_gl_parallel%d_%s_%s" % (
        count, type_name.lower().replace("-", "_"), exact_route)
    plat = _copy_kernel["plat"]
    if key not in _GGML_GL_PARALLEL_ADDED:
        plat.addKernel(name, {"source": _ggml_parallel_src_gl(type_name, count,
                                                               exact_route)})
        _GGML_GL_PARALLEL_ADDED.add(key)
    out = _empty((1, sum(ns)))
    inputs = [{"name": "tex_x", "id": _contig(xd).buffer.buffer_id}]
    inputs += [{"name": "tex_w%d" % i, "id": l.packed.buffer.buffer_id}
               for i, l in enumerate(linears)]
    _, _, vals, blk, _ = _GGML_TYPES[type_name]
    U = lambda n, v: {"name": n, "value": int(v), "type": "int"}
    uniforms = [U("_ka_tex_output_texture_w", out.buffer.texture_shape.width),
                U("u_M", 1), U("u_N", sum(ns)), U("u_K", K),
                U("u_rowb", (K // vals) * blk), U("u_estride", 0),
                U("u_eslot", 0), U("u_xper", 0), U("u_ROWS", 1)]
    uniforms += [U("u_N%d" % i, n) for i, n in enumerate(ns)]
    plat.runKernel({"name": name, "inputs": inputs,
                    "output": out.buffer.buffer_id, "uniforms": uniforms})
    return Tensor(out)


def _ggml_parallel_fused_gl(xd, linears):
    """Measured same-width combined projection used by WebGL's MLP layer."""
    linears = tuple(linears); type_name = linears[0].type_name
    mode = 1
    alternate = _GGML_GL_MODE_DECODERS.get(type_name, {}).get(mode)

    def run(route):
        return _ggml_parallel_run_gl_exact(xd, linears, route).data

    if alternate is None:
        return _ggml_parallel_run_gl_exact(xd, linears, "base")
    reference = [None]

    def correct(route):
        if route == "base":
            return True
        if reference[0] is None:
            reference[0] = np.asarray(run("base").get(), np.float32)
        got = np.asarray(run(route).get(), np.float32)
        scale = max(1e-6, float(np.abs(reference[0]).max()))
        return bool(np.all(np.isfinite(got))
                    and float(np.abs(got - reference[0]).max()) / scale < 1e-4)

    route = _weight_execution("ggml_gl_parallel", type_name, linears[0].Kt,
                              sum(int(l.Nt) for l in linears), 1, run,
                              candidates=("base", "alternate"), check=correct,
                              rounds=9, repeat=2)
    return _ggml_parallel_run_gl_exact(xd, linears, route)


def _ggml_run_gl(xf, packed, type_name, K, N, eidx=None, eslot=0, estride=0, xper=False,
                 bias=None):
    """Measured exact-width WebGL dispatch, independently selected on this device.

    The base and alternate shaders decode the same original packed bytes into FP32
    accumulators; only their scalar/``vec4`` register schedule differs.  When a format has
    both, correctness is checked first and paired measurements select by format, routing
    mode, shape and row bucket.  WebGPU's result is never consulted.
    """
    moe = eidx is not None
    M = 1 if (moe and xper) else int(xf.shape[0])
    mode = M if M <= 2 else (3 if M <= 32 else 0)
    override = _GGML_GL_MODE_DECODERS.get(type_name, {}).get(mode)

    def run(route):
        return _ggml_run_gl_exact(xf, packed, type_name, K, N, eidx=eidx,
                                  eslot=eslot, estride=estride, xper=xper, bias=bias,
                                  exact_route=route)

    # Packed activations: dense batched modes only (a MoE kernel indexes its activation
    # row differently), and only where RGBA32F renders.
    packed_ok = (not moe and mode in (0, 3) and int(K) % 4 == 0
                 and bool(get_platform_info_gl().get("supportsTexture32bit")))
    candidates = (("base",) + (("alternate",) if override is not None else ())
                  + (("packed",) if packed_ok else ()))
    if len(candidates) == 1:
        return run("base")
    reference = [None]

    def correct(route):
        if route == "base":
            return True
        if reference[0] is None:
            reference[0] = np.asarray(run("base").get(), np.float32)
        got = np.asarray(run(route).get(), np.float32)
        scale = max(1e-6, float(np.abs(reference[0]).max()))
        return bool(np.all(np.isfinite(got))
                    and float(np.abs(got - reference[0]).max()) / scale < 1e-4)

    storage = "%s:m%d:%s:%s:%s" % (type_name, mode,
                                     "moe" if moe else "dense",
                                     "slot" if moe and M <= 2 else "batch",
                                     "bias" if bias is not None else "nobias")
    route = _weight_execution("ggml_gl_exact", storage, K, N, M, run,
                              candidates=candidates, check=correct,
                              rounds=9, repeat=2)
    return run(route)


_GL_INFO = {}


def get_platform_info_gl():
    """This WebGL device's capability flags, asked of the platform once."""
    if "v" not in _GL_INFO:
        from wgpy_backends.webgl.platform import get_platform
        _GL_INFO["v"] = get_platform().getDeviceInfo()
    return _GL_INFO["v"]


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
    mrow = _ggml_mrow(vals, M) if mode == 0 else None
    # Asking for a variant nobody built is the same silent failure the self-check exists to
    # catch: the platform does not know the name, runs nothing, and leaves the output buffer
    # zeroed -- which reads as a numerically wrong kernel. It cost a full sweep reported as
    # "168 of 168 formats broken" while the model beside it generated perfectly.
    if (type_name, mode, small, moe,
            (_GGML_KSG, mrow) if mode == 0 else 0) not in _ggml_k["added"]:
        raise RuntimeError("ggml kernel variant %r was never built -- go through ggml_matmul, "
                           "or pass the same `small` it derives (_AUTO works)"
                           % ((type_name, mode, small, moe),))
    # A MoE decode runs every routed slot in one dispatch, z indexing the slot, and returns
    # one row per slot for the caller to weight and sum. Batched prefill keeps z for its own
    # row blocking and takes a slot at a time.
    slots = int(eidx.size) if (moe and mode == 1) else 1
    name = _ggml_name(type_name, mode, small=small, moe=moe, mrow=mrow)
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
                                   "z": slots if M <= 2 else
                                   (M + mrow * _GGML_KSG - 1) // (mrow * _GGML_KSG)}})
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
GPTQACC
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
GPTQGLACC
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
GPTQACC
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

_PACK_I8_WGSL = """requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read_write> q: array<u32>;
@group(0) @binding(2) var<storage,read_write> scales: array<f32>;
@group(0) @binding(3) var<storage,read_write> sums: array<f32>;
struct PM { M:u32, K:u32, B:u32, NB:u32, }
@group(0) @binding(4) var<storage,read> c: PM;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; let m = gid.y;
  if (b >= c.NB || m >= c.M) { return; }
  let xb = m * c.K + b * c.B;
  var mx: f32 = 0.0;
  for (var j:u32=0u; j<c.B; j=j+1u) { mx = max(mx, abs(x[xb+j])); }
  let sc = select(1.0, mx / 127.0, mx > 0.0);
  let qb = (m * c.NB + b) * (c.B / 4u);
  var sm:i32 = 0;
  for (var j:u32=0u; j<c.B; j=j+4u) {
    let v = vec4<f32>(x[xb+j], x[xb+j+1u], x[xb+j+2u], x[xb+j+3u]);
    let iv = vec4<i32>(round(v / vec4<f32>(sc)));
    q[qb + j/4u] = pack4xI8Clamp(iv);
    sm = sm + iv.x + iv.y + iv.z + iv.w;
  }
  scales[m*c.NB+b] = sc; sums[m*c.NB+b] = f32(sm);
}
"""

_GPTQ_DP4A_WGSL = """requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> xq: array<u32>;
@group(0) @binding(1) var<storage,read> xsc: array<f32>;
@group(0) @binding(2) var<storage,read> xsum: array<f32>;
@group(0) @binding(3) var<storage,read> qweight: array<u32>;
@group(0) @binding(4) var<storage,read> qzeros: array<u32>;
@group(0) @binding(5) var<storage,read> scales: array<f32>;
@group(0) @binding(6) var<storage,read_write> outp: array<f32>;
struct C { M:u32, N:u32, K:u32, GS:u32, }
@group(0) @binding(7) var<storage,read> c: C;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let n=gid.x; let m=gid.y; if(n>=c.N || m>=c.M){return;}
  let per=PERu; let ng=c.K/c.GS; let qpg=c.GS/per; let xpg=c.GS/4u;
  let ndp=c.N/per; var acc:f32=0.0;
  for(var g:u32=0u; g<ng; g=g+1u){
    let zword=qzeros[g*ndp+n/per];
    let z=i32((zword>>((n%per)*BITSu))&MASKu)+ZOFFi;
    var di:i32=0;
    for(var t:u32=0u; t<qpg; t=t+1u){
      let qw=qweight[(g*qpg+t)*c.N+n];
DP4ACC
    }
    let xi=m*ng+g;
    acc=acc+xsc[xi]*scales[g*c.N+n]*(f32(di)-f32(z)*xsum[xi]);
  }
  outp[m*c.N+n]=acc;
}
"""

_dp4a_k = {"pack": False, "gptq": set()}

# Phase-two cross-width routing.  WebGPU can measure activation-INT8 DP4A, including its
# packing cost, per format/shape/device.  WebGL exposes neither compute packing nor packed
# integer dot products, so its equivalent QuantizedLinear implementation converges at the
# public operator result and keeps the exact stored GLSL path.  This is model-agnostic.
_PHASE2_CROSS_WIDTH = {
    "gptq_activation_int8_dp4a": {
        "webgpu": "measured_per_format_shape_device",
        "webgl": "primitive_unavailable_keep_stored",
    },
    # Activations and running sums in half (`shader-f16`), output in f32. Offered only where
    # the device was created with the feature; gated per input against the f32 route.
    "dense_half_activation_f16": {            # `_mm_half_src`, raced in `matmul_f16w`
        "webgpu": "measured_per_shape_device_when_shader_f16",
        "webgl": "primitive_unavailable_keep_f32",
    },
    "moe_grouped_activation_f16": {           # "grouped_half", raced in `forward_routed`
        "webgpu": "measured_per_format_shape_device_when_shader_f16",
        "webgl": "primitive_unavailable_keep_stored",
    },
    "ggml_tiled_activation_f16": {            # `_ggml_tiled_half_src`, raced in `ggml_matmul`
        "webgpu": "measured_per_format_shape_device_when_shader_f16",
        "webgl": "primitive_unavailable_keep_stored",
    },
}


def _pack_i8_rows(xf, K, block):
    M = int(xf.shape[0]); nb = int(K) // int(block)
    plat = _adam_kernel["platform"]
    if not _dp4a_k["pack"]:
        plat.addKernel("pack_i8_rows", {"source": _PACK_I8_WGSL,
            "bindingTypes": ["read-only-storage", "storage", "storage", "storage",
                             "read-only-storage"]})
        _dp4a_k["pack"] = True
    q = _empty((M * nb * (int(block) // 4),)); sc = _empty((M * nb,)); sm = _empty((M * nb,))
    meta = _adam_kernel["make_meta"]((M, int(K), int(block), nb), "u4,u4,u4,u4")
    plat.runKernel({"name": "pack_i8_rows",
        "tensors": [xf.buffer.buffer_id, q.buffer.buffer_id, sc.buffer.buffer_id,
                    sm.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": nb, "y": M, "z": 1}})
    return q, sc, sm


def _gptq_dp4a_src(bits, zoff=0.0):
    per = 32 // int(bits)
    if int(bits) == 4:
        acc = """      let q0=pack4xI8(vec4<i32>(i32(qw&15u),i32((qw>>4u)&15u),i32((qw>>8u)&15u),i32((qw>>12u)&15u)));
      let q1=pack4xI8(vec4<i32>(i32((qw>>16u)&15u),i32((qw>>20u)&15u),i32((qw>>24u)&15u),i32((qw>>28u)&15u)));
      let xo=(m*ng+g)*xpg+t*2u;
      di=di+dot4I8Packed(xq[xo],q0)+dot4I8Packed(xq[xo+1u],q1);"""
    else:
        acc = """      let xo=(m*ng+g)*xpg+t;
      di=di+dot4I8Packed(xq[xo],qw^0x80808080u)+128*xsum_i(xq[xo]);"""
        # The correction above would need the per-word activation sum. The group sum is
        # already available outside the loop, so use the equivalent group-level correction.
        acc = """      let xo=(m*ng+g)*xpg+t;
      di=di+dot4I8Packed(xq[xo],qw^0x80808080u);"""
    src = _GPTQ_DP4A_WGSL.replace("DP4ACC", acc)
    if int(bits) == 8:
        src = src.replace("(f32(di)-f32(z)*xsum[xi])",
                          "(f32(di)+f32(128-z)*xsum[xi])")
    for k, v in (("ZOFFi", str(int(zoff))), ("PERu", f"{per}u"),
                 ("BITSu", f"{bits}u"), ("MASKu", f"{(1 << bits)-1}u")):
        src = src.replace(k, v)
    return src


def _gptq_dp4a_matmul(xf, qweight, qzeros, scales, K, N, gs, bits, zoff=0.0):
    M = int(xf.shape[0]); plat = _adam_kernel["platform"]
    key = (int(bits), int(gs), int(bool(zoff)))
    name = "gptq_dp4a_%d_g%d_z%d" % key
    if key not in _dp4a_k["gptq"]:
        plat.addKernel(name, {"source": _gptq_dp4a_src(bits, zoff),
            "bindingTypes": ["read-only-storage"] * 6 + ["storage", "read-only-storage"]})
        _dp4a_k["gptq"].add(key)
    xq, xsc, xsum = _pack_i8_rows(xf, K, gs)
    out = _empty((M, N)); meta = _adam_kernel["make_meta"]((M, N, K, gs), "u4,u4,u4,u4")
    plat.runKernel({"name": name,
        "tensors": [xq.buffer.buffer_id, xsc.buffer.buffer_id, xsum.buffer.buffer_id,
                    qweight.buffer.buffer_id, qzeros.buffer.buffer_id, scales.buffer.buffer_id,
                    out.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (N+63)//64, "y": M, "z": 1}})
    return out


def _gptq_acc(bits, gemv=False, vector=True):
    """Exact packed-weight accumulation; vector=False is retained for the phase-one A/B.

    Both variants read the original GPTQ u32 and FP32 activations. The vector form merely
    unpacks one stored word into register-local vec4 values and uses FP32 dot products; it
    does not requantise either operand and therefore remains a same-width implementation.
    """
    if not vector:
        if gemv:
            return """        for (var j: u32 = 0u; j < PERu; j = j + 1u) {
          part = part + xsg[ly * GSu + ko + j] *
              (f32((qw >> (j * BITSu)) & MASKu) - zv);
        }"""
        return """        for (var j: u32 = 0u; j < PERu; j = j + 1u) {
          let qv = f32((qw >> (j * BITSu)) & MASKu) - zv;
          let kk = ko + j;
          let xv = vec4<f32>(xs[kk], xs[GSu + kk], xs[2u * GSu + kk], xs[3u * GSu + kk]);
          part = part + xv * qv;
        }"""
    if bits == 8:
        qvecs = (("q0", "unpack4x8unorm(qw) * 255.0 - vec4<f32>(zv)"),)
    elif bits == 4:
        qvecs = (
            ("q0", "vec4<f32>(f32(qw & 15u), f32((qw >> 4u) & 15u), "
                   "f32((qw >> 8u) & 15u), f32((qw >> 12u) & 15u)) - vec4<f32>(zv)"),
            ("q1", "vec4<f32>(f32((qw >> 16u) & 15u), f32((qw >> 20u) & 15u), "
                   "f32((qw >> 24u) & 15u), f32((qw >> 28u) & 15u)) - vec4<f32>(zv)"),
        )
    else:
        raise ValueError("GPTQ native compute supports int4/int8, got %r" % bits)
    lines = []
    for qi, (name, expr) in enumerate(qvecs):
        off = qi * 4
        lines.append("        let %s = %s;" % (name, expr))
        if gemv:
            lines.append(
                "        part = part + dot(vec4<f32>(xsg[ly * GSu + ko + %du], "
                "xsg[ly * GSu + ko + %du], xsg[ly * GSu + ko + %du], "
                "xsg[ly * GSu + ko + %du]), %s);" %
                (off, off + 1, off + 2, off + 3, name))
        else:
            for row, comp in enumerate("xyzw"):
                base = ("ko" if row == 0 else "%du * GSu + ko" % row)
                lines.append(
                    "        part.%s = part.%s + dot(vec4<f32>(xs[%s + %du], "
                    "xs[%s + %du], xs[%s + %du], xs[%s + %du]), %s);" %
                    (comp, comp, base, off, base, off + 1, base, off + 2,
                     base, off + 3, name))
    return "\n".join(lines)


def _gptq_gl_acc(bits, vector=True):
    """GLSL equivalent of the exact packed-word GPTQ accumulation.

    WebGL has no compute workgroups, but its fragment shader still has native ``vec4`` and
    ``dot`` operations.  Like the WGSL path this only regroups the original packed word in
    registers; weights stay at their stored bit width and activations stay FP32.
    """
    if not vector:
        return """      for(int j=0;j<PER;j++){
        float qv = float((qw>>(j*BITS))&MASK) - zv;
        part += fetch(tex_x, m*K + kb0 + j) * qv;
      }"""
    if int(bits) == 8:
        groups = ((0, (0, 8, 16, 24)),)
    elif int(bits) == 4:
        groups = ((0, (0, 4, 8, 12)), (4, (16, 20, 24, 28)))
    else:
        raise ValueError("GPTQ native compute supports int4/int8, got %r" % bits)
    lines = []
    for off, shifts in groups:
        q = ", ".join("float((qw >> %d) & MASK)" % s for s in shifts)
        x = ", ".join("fetch(tex_x, m*K + kb0 + %d)" % (off + j)
                      for j in range(4))
        lines.append("      part += dot(vec4(%s), vec4(%s) - vec4(zv));" % (x, q))
    return "\n".join(lines)


def _gptq_gemv_src(bits, gs, ks=_GPTQ_KS, zoff=0.0, vector=True):
    per = 32 // bits
    src = _GPTQ_GEMV_WGSL.replace("GPTQACC", _gptq_acc(bits, gemv=True,
                                                        vector=vector))
    # longest/most-specific tokens first
    for k, v in [("ZOFFf", "%.1f" % float(zoff)), ("KSxGS", str(ks * gs)),
                 ("KSx64", str(ks * 64)), ("KSu", f"{ks}u"), ("KS", str(ks)),
                 ("GSu", f"{gs}u"), ("PERu", f"{per}u"),
                 ("BITSu", f"{bits}u"), ("MASKu", f"{(1 << bits) - 1}u")]:
        src = src.replace(k, v)
    return src


def _gptq_src(tmpl, bits, gs=None, zoff=0.0, vector=True):
    per = 32 // bits
    if "GPTQACC" in tmpl:
        tmpl = tmpl.replace("GPTQACC", _gptq_acc(bits, gemv=False, vector=vector))
    if "GPTQGLACC" in tmpl:
        tmpl = tmpl.replace("GPTQGLACC", _gptq_gl_acc(bits, vector=vector))
    d = {"ZOFFf": "%.1f" % float(zoff)}
    if gs is not None:                       # tiled dequant-matmul kernel only
        # XSSZu must be substituted before XSSZ; likewise PERu before PER, etc.
        d.update({"XSSZu": f"{4 * gs}u", "XSSZ": str(4 * gs), "GSu": f"{gs}u"})
    d.update({"PERu": f"{per}u", "PER": str(per), "BITSu": f"{bits}u", "BITS": str(bits),
              "MASKu": f"{(1 << bits) - 1}u", "MASK": str((1 << bits) - 1)})
    for k, v in d.items():
        tmpl = tmpl.replace(k, v)
    return tmpl


def _gptq_exact_vector(bits, rows=1):
    """Measured same-width packed-word route, independently selected per backend."""
    bits = int(bits); rows = int(rows)
    if _webgl_ready() and not _adam_backend_ready():
        if bits == 4:
            return rows <= 32
        if bits == 8:
            return rows == 1 or rows > 32
        return False
    return bits == 4


def _gptq_matmul(xf, qweight, qzeros, scales, K, N, gs, bits, zoff=0.0):
    M = int(xf.shape[0])
    gemv = (M == 1)                                   # decode path: split-K GEMV
    zt = 1 if zoff else 0                             # AutoGPTQ stores zero-1
    # Exact packed-word vec4 dots consistently win for int4. Int8 is neutral and unstable
    # (including a measured M=2 regression), so keep its scalar unpack. Both consume the
    # same original storage and FP32 activations; this is a phase-one implementation choice.
    exact_vector = _gptq_exact_vector(bits, M)
    key = (bits, gs, gemv, zt, exact_vector)
    name = f"gptq{'v' if gemv else ''}{bits}_g{gs}_z{zt}_{'xv' if exact_vector else 'xs'}"
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if key not in _gptq_k["wgpu"]:
            src = (_gptq_gemv_src(bits, gs, zoff=zoff, vector=exact_vector) if gemv
                   else _gptq_src(_GPTQ_WGSL, bits, gs, zoff=zoff,
                                  vector=exact_vector))
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


# A bounded number of in-flight expert sources: 32 replaces 32 per-expert readback
# synchronisations with one while keeping transient upload buffers to tens of MB.
# The limit is transport memory hygiene; the stored weight bits and order do not change.
_MOE_TRANSPOSE_WINDOW = 32


def _ggml_transpose_drain(pending):
    """Finish queued sliced writes before their source/meta buffers can be recycled."""
    if pending:
        cp.asnumpy(pending[-1][2])
        pending.clear()


def ggml_transpose(src, n, rowb, dst=None, dstoff=0, pending=None):
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
    # The dispatch is queued, not done. Keep every source, metadata and flag alive until
    # a readback drains the queue. A standalone transpose still synchronises here; a
    # stacked MoE weight may batch bounded slices and synchronise once per window.
    #
    # Read the flag, not the tensor: a read-back is sized to the whole buffer, and asking
    # for a 210 MB head block back exceeds what the backend will stage ("buffer size
    # insufficient"). Four bytes drain the queue just as well.
    if pending is None:
        cp.asnumpy(flag)
    else:
        pending.append((src, meta, flag))
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
        if _webgl_ready() and not _adam_backend_ready():
            # The WebGL JS backend writes live texture bytes directly to the shared
            # resource array; Python merely signals the worker to refresh WASM size.
            hook(None, None, None)
        else:
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
    if not _adam_backend_ready():
        return                                   # WebGL, or no GPU backend at all
    import wgpy_backends.webgpu.webgpu_buffer as _b
    fn = getattr(_b, "reap_now", None)
    if fn is not None:
        fn()
    _gpu_stat_push(force=True)


class GGMLWeight(object):
    """One tensor kept in the encoding its storage source supplied.

    This is a weight protocol implementation, not a model type.  Any loader can hand one to
    any model: a Linear consumer obtains the matching native module with ``as_linear``;
    another operator explicitly materializes it.  The model never branches on GGUF, a
    quantization name, or a repository name.
    """

    def __init__(self, raw, type_name, shape, type_id=None):
        self.raw = bytes(raw)
        self.type_name = str(type_name)
        self.shape = tuple(int(x) for x in shape)
        self.type_id = type_id

    def materialize(self, dtype=None):
        from . import ggufload as G
        raw, self.raw = self.raw, None
        ttype = self.type_id if self.type_id is not None else G.GGML_IDS[self.type_name]
        count = int(np.prod(self.shape, dtype=np.int64))
        out = G.dequant(ttype, raw, count).reshape(self.shape)
        if dtype is None:
            dtype = np.float32 if self.type_name == "F32" else np.float16
        return out.astype(dtype)

    def as_linear(self, bias=None, execution="auto"):
        if len(self.shape) != 2:
            raise ValueError("stored Linear weight must be two-dimensional")
        if not ggml_native_supported(self.type_name):
            raise RuntimeError("%s weight has no active native compute backend" % self.type_name)
        raw, self.raw = self.raw, None
        n_out, n_in = self.shape
        return GGMLLinear(raw, self.type_name, n_in, n_out, bias, execution=execution)


def weight_shape(weight):
    """The logical tensor shape, independent of how its values are stored."""
    return tuple(getattr(weight, "shape", np.shape(weight)))


def materialize_weight(weight, dtype=None):
    """Explicitly turn a stored weight into an array for a non-Linear operator."""
    fn = getattr(weight, "materialize", None)
    value = fn(dtype=dtype) if callable(fn) else weight
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def stored_linear(weight, bias=None, execution="auto"):
    """Return a ready native Linear for an encoded weight/module, or None for a dense array.

    GGMLWeight and future storage encodings implement ``as_linear``.  AutoGPTQ already
    arrives as QuantizedLinear, so it passes through unchanged.  Model code only asks this
    one question and contains no format-specific branch.
    """
    if isinstance(weight, Module):
        return weight
    fn = getattr(weight, "as_linear", None)
    return fn(bias=bias, execution=execution) if callable(fn) else None


class GGMLLinear(Module):
    """Inference-only Linear whose weight stays in the encoding the GGUF shipped it in.

    Nothing is dequantized or requantized at load: the file's bytes go to the GPU as they
    are and `ggml_matmul` unpacks each block while it multiplies. That removes the whole
    conversion pass -- the bulk of a load -- and the second rounding it imposed."""

    def __init__(self, raw, type_name, K, N, bias=None, execution="auto"):
        if execution not in ("stored", "tiled", "tiled_half", "dp4a", "materialized", "auto"):
            raise ValueError("execution must be 'stored', 'tiled', 'tiled_half', 'dp4a', "
                             "'materialized', or 'auto'")
        vals, blk = _GGML_TYPES[type_name][2], _GGML_TYPES[type_name][3]
        if _webgl_q8_ok(type_name, K, N):
            self.packed = _webgl_q8_pack(raw, K, N)
        else:
            b = np.frombuffer(raw, np.uint8)
            pad = (-b.size) % 4
            if pad:
                b = np.concatenate([b, np.zeros(pad, np.uint8)])
            up = xp.asarray(b.view(np.int32))
            # Transposed on the way in, once, so every later matmul reads it coalesced. Done
            # on the device: the host copy is already the largest thing in flight in a load.
            self.packed = ggml_transpose(up, int(N), (int(K) // vals) * blk)
            del up
        self.type_name = type_name
        self.storage_format = type_name
        self.execution = execution
        # A containing decoder may override only its one-row composition without changing
        # this Linear's own auto route or forcing that choice onto multi-row prefill.
        self.decode_execution = None
        self.decode_shape = "auto"
        self.Kt = int(K); self.Nt = int(N)
        self.bias = None if bias is None else xp.asarray(np.asarray(bias, np.float32))

    def forward(self, x):
        xd = x.data
        lead = xd.shape[:-1]
        rows = int(xd.reshape(-1, self.Kt).shape[0])
        execution = (self.decode_execution
                     if rows == 1 and self.decode_execution is not None else self.execution)
        of = ggml_matmul(_contig(xd.reshape(-1, self.Kt)), self.packed,
                         self.type_name, self.Kt, self.Nt, bias=self.bias,
                         execution=execution,
                         shape_execution=(self.decode_shape if rows == 1 else "auto"))
        return Tensor(of.reshape(*lead, self.Nt))                 # inference-only



# ---- routed experts for a whole prompt: one weight read per expert, not per (token, slot) --
#
# A prefill's routed layer ran every (token, slot) pair as its own GEMV against the expert it
# was routed to (`GGMLMoELinear.forward` with a row per slot), which reads an expert's weights
# once for EVERY token routed to it. On a 30B (128 experts, 8 per token) at 248 tokens that
# is 1984 reads where 128 do: the expert GEMVs were 1.33 of a 1.62 s prefill. Here the slots
# are first grouped by expert on the device, then each expert's rows are one small GEMM over
# its weights (the tiled kernel, re-pointed): 39.9 -> 9.9 ms for that model's gate/up.

_MOE_GROUP_WGSL = """
// Group the routed slots by expert, on the device: counts, starts, a permutation that lists
// each expert's slots together, and a table of (expert, first row) for every 32-row tile.
@group(0) @binding(0) var<storage,read> eidx: array<i32>;
@group(0) @binding(1) var<storage,read_write> grp: array<u32>;
struct GMeta { S: u32, E: u32, PERM0: u32, TT0: u32, TOT: u32, }
@group(0) @binding(2) var<storage,read> gmm: GMeta;
var<workgroup> cnt: array<atomic<u32>, 256>;
var<workgroup> fill: array<atomic<u32>, 256>;
var<workgroup> tst: array<u32, 257>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
  let t = li.x; let S = gmm.S; let E = gmm.E;
  atomicStore(&cnt[t], 0u);
  workgroupBarrier();
  for (var i = t; i < S; i = i + 256u) { atomicAdd(&cnt[u32(eidx[i])], 1u); }
  workgroupBarrier();
  if (t == 0u) {
    var s = 0u; var ts = 0u;
    for (var e = 0u; e < E; e = e + 1u) {
      let c = atomicLoad(&cnt[e]);
      grp[e] = s; atomicStore(&fill[e], s); tst[e] = ts;
      s = s + c; ts = ts + (c + 31u) / 32u;
    }
    grp[E] = s; tst[E] = ts; grp[gmm.TOT] = ts;
  }
  workgroupBarrier();
  for (var i = t; i < S; i = i + 256u) {
    let p = atomicAdd(&fill[u32(eidx[i])], 1u);
    grp[gmm.PERM0 + p] = i;
  }
  if (t < E) {
    let c = atomicLoad(&cnt[t]);
    for (var j = 0u; j < (c + 31u) / 32u; j = j + 1u) {
      grp[gmm.TT0 + 2u * (tst[t] + j)] = t;
      grp[gmm.TT0 + 2u * (tst[t] + j) + 1u] = j * 32u;
    }
  }
}
"""
_moe_group_added = {"v": False}


def moe_group(eidx, S, E):
    """Group S routed slots by expert, on the device: (grp, info) for `_ggml_tiled_moe_src`.
    `grp` holds the experts' start offsets, the permutation and the tile table."""
    S, E = int(S), int(E)
    if E > 256:
        return None
    plat = _adam_kernel["platform"]
    if not _moe_group_added["v"]:
        plat.addKernel("moe_group", {"source": _MOE_GROUP_WGSL,
                                     "bindingTypes": ["read-only-storage", "storage",
                                                      "read-only-storage"]})
        _moe_group_added["v"] = True
    PERM0 = E + 1
    TT0 = PERM0 + S
    maxT = (S + 31) // 32 + E
    TOT = TT0 + 2 * maxT
    grp = _empty_i32((TOT + 1,))
    meta = _adam_kernel["make_meta"]((S, E, PERM0, TT0, TOT), "u4,u4,u4,u4,u4")
    plat.runKernel({"name": "moe_group",
                    "tensors": [eidx.buffer.buffer_id, grp.buffer.buffer_id, meta.buffer_id],
                    "workGroups": {"x": 1, "y": 1, "z": 1}})
    return grp, (PERM0, TT0, TOT, maxT)


def _ggml_tiled_moe_src(type_name, half):
    """The tiled kernel (f32 or half) over the rows `moe_group` grouped by expert.

    One workgroup per (expert, 32-row tile) x 64 columns, its expert's weights at
    `expert * estride` and its rows read through the permutation: a token's row directly
    for the first projection (no k-fold copy of the activations), a slot's row for the
    second. Each row is written to its slot."""
    src = _ggml_tiled_half_src(type_name) if half else _GGML_TILED[type_name]
    gbind = 5 if "binding(4) var<storage,read> gr" in src else 4
    subs = [
        ("struct QMeta { M: u32, N: u32, K: u32, RW: u32, G: u32, }",
         "struct QMeta { M: u32, N: u32, K: u32, RW: u32, G: u32, ESTR4: u32, KSLOT: u32, SLOTROWS: u32, PERM0: u32, TT0: u32, TOT: u32, }\n"
         "@group(0) @binding(%d) var<storage,read> grp: array<u32>;\nvar<private> EOFF: u32;" % gbind),
        ("  return packed[w * ND4 + c4];", "  return packed[EOFF + w * ND4 + c4];"),
        ("""  let row = wg.y * 32u + li.y * 4u;
  let i0 = select(M - 1u, row, row < M);
  let i1 = select(i0, row + 1u, row + 1u < M);
  let i2 = select(i0, row + 2u, row + 2u < M);
  let i3 = select(i0, row + 3u, row + 3u < M);""",
         """  let tt = wg.y;
  if (tt >= grp[qm.TOT]) { return; }
  let ge = grp[qm.TT0 + 2u * tt]; let r0 = grp[qm.TT0 + 2u * tt + 1u];
  let es = grp[ge]; let cnt = grp[ge + 1u] - es;
  EOFF = ge * qm.ESTR4;
  let lrow = r0 + li.y * 4u;
  let o0 = grp[qm.PERM0 + es + min(lrow, cnt - 1u)];
  let o1 = grp[qm.PERM0 + es + min(lrow + 1u, cnt - 1u)];
  let o2 = grp[qm.PERM0 + es + min(lrow + 2u, cnt - 1u)];
  let o3 = grp[qm.PERM0 + es + min(lrow + 3u, cnt - 1u)];
  let i0 = select(o0 / qm.KSLOT, o0, qm.SLOTROWS != 0u);
  let i1 = select(o1 / qm.KSLOT, o1, qm.SLOTROWS != 0u);
  let i2 = select(o2 / qm.KSLOT, o2, qm.SLOTROWS != 0u);
  let i3 = select(o3 / qm.KSLOT, o3, qm.SLOTROWS != 0u);"""),
        ("""  if (row >= M || c0 >= N) { return; }
  let cx = sl + (c0 >> 2u);
  let wide = c0 + 4u < N;
  array_c[cx + row * ND4] = s00;
  if (wide) { array_c[cx + 1u + row * ND4] = s10; }
  if (row + 1u < M) { array_c[cx + (row + 1u) * ND4] = s01; if (wide) { array_c[cx + 1u + (row + 1u) * ND4] = s11; } }
  if (row + 2u < M) { array_c[cx + (row + 2u) * ND4] = s02; if (wide) { array_c[cx + 1u + (row + 2u) * ND4] = s12; } }
  if (row + 3u < M) { array_c[cx + (row + 3u) * ND4] = s03; if (wide) { array_c[cx + 1u + (row + 3u) * ND4] = s13; } }""",
         """  if (c0 >= N) { return; }
  let cx = c0 >> 2u;
  let wide = c0 + 4u < N;
  if (lrow < cnt) { array_c[cx + o0 * ND4] = s00; if (wide) { array_c[cx + 1u + o0 * ND4] = s10; } }
  if (lrow + 1u < cnt) { array_c[cx + o1 * ND4] = s01; if (wide) { array_c[cx + 1u + o1 * ND4] = s11; } }
  if (lrow + 2u < cnt) { array_c[cx + o2 * ND4] = s02; if (wide) { array_c[cx + 1u + o2 * ND4] = s12; } }
  if (lrow + 3u < cnt) { array_c[cx + o3 * ND4] = s03; if (wide) { array_c[cx + 1u + o3 * ND4] = s13; } }"""),
    ]
    if half:
        subs.append(("      let ir = min(wg.y * 32u + rr, M - 1u);",
                     "      let os = grp[qm.PERM0 + es + min(r0 + rr, cnt - 1u)];\n"
                     "      let ir = select(os / qm.KSLOT, os, qm.SLOTROWS != 0u);"))
    for old, new in subs:
        assert src.count(old) == 1, (type_name, half, old[:50])
        src = src.replace(old, new)
    return src, gbind


_moe_tiled_added = set()


def _moe_grouped_run(stack, xd, group, k, slot_rows, S, half):
    t = stack.type_name
    src, gbind = _ggml_tiled_moe_src(t, half)
    name = "moe_tiled%s_%s" % ("h" if half else "", t.lower())
    grid = _ggml_grid(t) if gbind == 5 else None
    plat = _adam_kernel["platform"]
    if name not in _moe_tiled_added:
        plat.addKernel(name, {"source": src,
                              "bindingTypes": ["read-only-storage", "read-only-storage",
                                               "storage", "read-only-storage"]
                                             + (["read-only-storage"] if grid is not None
                                                else []) + ["read-only-storage"]})
        _moe_tiled_added.add(name)
    vals, blk = int(_GGML_TYPES[t][2]), int(_GGML_TYPES[t][3])
    K, N = int(stack.Kt), int(stack.Nt)
    words = ((K // vals) * blk + 3) // 4
    grp, (PERM0, TT0, TOT, maxT) = group
    out = _empty((int(S), N))
    meta = _adam_kernel["make_meta"](
        (int(xd.shape[0]), N, K, words, 1, int(stack.estride) // 4, int(k),
         1 if slot_rows else 0, PERM0, TT0, TOT), "u4,u4,u4,u4,u4,u4,u4,u4,u4,u4,u4")
    plat.runKernel({"name": name,
                    "tensors": [xd.buffer.buffer_id, stack.packed.buffer.buffer_id,
                                out.buffer.buffer_id, meta.buffer_id]
                               + ([grid.buffer.buffer_id] if grid is not None else [])
                               + [grp.buffer.buffer_id],
                    "workGroups": {"x": (N + 63) // 64, "y": maxT, "z": 1}})
    return out

class GGMLMoELinear(Module):
    """One projection of a sparse-MoE layer: every expert's weight, stacked in one buffer.

    A MoE layer picks a few experts per token, so which weight a matmul reads is decided at
    run time. Holding one Linear per expert forces that choice into the command stream --
    which expert kernels get dispatched -- and a captured decode step then replays whatever
    the first token selected, for every token after it. Stacking them and passing an index
    keeps the command identical: the shader offsets into this buffer by `estride` words.
    This is the same shape as llama.cpp's ggml_mul_mat_id.

    Experts are transposed in bounded windows into the destination. Only one window of
    source buffers stays live while GPU work is in flight, not the whole stack.
    """

    def __init__(self, chunks, type_name, K, N, also_chunks=None):
        vals, blk_b = _GGML_TYPES[type_name][2], _GGML_TYPES[type_name][3]
        rowb = (int(K) // vals) * blk_b
        words = (rowb + 3) // 4
        self.estride = words * int(N)
        ne = len(chunks)
        if also_chunks is not None and len(also_chunks) != ne:
            raise ValueError("joined expert projections must have the same expert count")
        if _webgl_ready() and not _adam_backend_ready():
            # One upload and one pass. Per-expert transposes into slices of a shared
            # destination have no fragment-shader form (see `_ggml_transpose_gl_stack`), and
            # the alternative -- one buffer per expert -- costs the routed kernel and, with
            # it, the device-side router: the host would have to read the router's scores
            # back to decide which expert's buffer to bind, once per layer per token.
            nb = max(len(c) + (len(also_chunks[e]) if also_chunks is not None else 0)
                     for e, c in enumerate(chunks))
            perb = nb + (-nb) % 4
            buf = np.zeros(perb * ne, np.uint8)
            for e, raw in enumerate(chunks):
                offset = e * perb
                for part in ((raw, also_chunks[e]) if also_chunks is not None else (raw,)):
                    b = np.frombuffer(part, np.uint8)
                    buf[offset:offset + b.size] = b
                    offset += b.size
            up = xp.asarray(buf.view(np.int32))
            del buf
            self.packed = _ggml_transpose_gl_stack(up, int(N), rowb, ne, perb // 4)
            del up
            self.type_name = type_name
            self.Kt = int(K); self.Nt = int(N); self.n_experts = ne
            return
        dst = _empty((self.estride * ne,))
        pending = []
        try:
            for e, raw in enumerate(chunks):
                b = np.frombuffer(raw, np.uint8)
                if also_chunks is not None:
                    # Only one expert is joined at a time.  The WebGPU transpose
                    # already drains in bounded windows; a whole-layer joined list
                    # defeats that bound before the first upload starts.
                    b = np.concatenate((b, np.frombuffer(also_chunks[e], np.uint8)))
                pad = (-b.size) % 4
                if pad:
                    b = np.concatenate([b, np.zeros(pad, np.uint8)])
                up = xp.asarray(b.view(np.int32))
                ggml_transpose(up, int(N), rowb, dst=dst, dstoff=e * self.estride,
                               pending=pending)
                # A 32-expert window bounds transient GPU uploads even for hundreds of
                # experts. Queue order makes one four-byte flag read wait for all slices.
                if len(pending) >= _MOE_TRANSPOSE_WINDOW:
                    _ggml_transpose_drain(pending)
        finally:
            _ggml_transpose_drain(pending)
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

    def forward_routed(self, x, eidx, k, slot_rows, cache=None):
        """Every routed slot of a prompt: (S, N), a row per slot, S = len(eidx).

        `x` is (T, K) token rows when `slot_rows` is False (slot s reads token s // k) and
        (S, K) slot rows when True. Three executions, raced per format/shape/slot count:
        "slots" (a GEMV per slot -- the old path, with the token rows repeated), "grouped" and
        "grouped_half" (`_ggml_tiled_moe_src`; half only with `shader-f16`). `cache` shares
        one grouping between the projections of a layer."""
        xd = _contig(x.data if isinstance(x, Tensor) else x)
        S = int(eidx.shape[0])
        k = int(k)

        def slots():
            rows = xd if slot_rows else repeat_rows(xd, k, execution="device")
            return ggml_matmul(_contig(rows.reshape(-1, self.Kt)), self.packed,
                               self.type_name, self.Kt, self.Nt, eidx=eidx,
                               estride=self.estride, xper=True)
        can = (_adam_backend_ready() and self.type_name in _GGML_TILED
               and self.type_name in _TILED_FORMATS and int(self.n_experts) <= 256
               and _ggml_tiled_ok(self.type_name, self.Kt, self.Nt) and S > 2)
        if not can:
            return Tensor(slots())

        def grouped(half):
            group = cache.get("group") if cache is not None else None
            if group is None:
                group = moe_group(eidx, S, self.n_experts)
                if cache is not None:
                    cache["group"] = group
            return _moe_grouped_run(self, xd, group, k, slot_rows, S, half)

        def run(which):
            if which == "slots":
                return slots()
            return grouped(which == "grouped_half")
        candidates = ("slots", "grouped") + (("grouped_half",) if gpu_features().get("f16")
                                              else ())
        reference = [None]

        def correct(which):
            if which == "slots":
                return True
            if reference[0] is None:
                reference[0] = np.asarray(run("slots").get(), np.float32)
            got = np.asarray(run(which).get(), np.float32)
            if not np.all(np.isfinite(got)):
                return False
            scale = max(1e-6, float(np.abs(reference[0]).max()))
            limit = 1e-2 if which == "grouped_half" else 1e-4
            return float(np.abs(got - reference[0]).max()) / scale < limit
        which = _weight_execution("moe_routed", self.type_name, self.Kt, self.Nt, S, run,
                                  candidates=candidates, check=correct)
        return Tensor(run(which))

    def nbytes(self):
        return int(self.packed.size) * 4


class QuantizedLinear(Module):
    """Inference-only GPTQ-format weight-quantized Linear (group-wise int4/int8)."""
    def __init__(self, qweight, qzeros, scales, bias, Kt, Nt, Kp, Np, gs, bits,
                 zero_offset=0.0, execution="auto"):
        if execution not in ("stored", "materialized", "auto"):
            raise ValueError("execution must be 'stored', 'materialized', or 'auto'")
        self.qweight = xp.asarray(qweight)     # int32 GPU
        self.qzeros = xp.asarray(qzeros)       # int32 GPU
        self.scales = xp.asarray(scales)       # f32 GPU
        self.bias = xp.asarray(bias.astype(np.float32))
        self.Kt = Kt; self.Nt = Nt; self.Kp = Kp; self.Np = Np; self.gs = gs; self.bits = bits
        self.zero_offset = float(zero_offset)  # AutoGPTQ stores (zero-1) -> use 1.0
        self.storage_format = "GPTQ_INT%d" % int(bits)
        self.execution = execution

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
        def run(which):
            if which == "stored":
                return _gptq_matmul(xf, self.qweight, self.qzeros, self.scales,
                                    self.Kp, self.Np, self.gs, self.bits,
                                    zoff=self.zero_offset)
            if which == "dp4a":
                return _gptq_dp4a_matmul(xf, self.qweight, self.qzeros, self.scales,
                                         self.Kp, self.Np, self.gs, self.bits,
                                         zoff=self.zero_offset)
            full = _dequant_full(self.qweight, self.qzeros, self.scales, self.Kp,
                                 self.Np, self.gs, self.bits, self.zero_offset)
            return xf @ full

        execution = self.execution
        # Materialise-and-retune is a WebGPU alternative.  WebGL always executes the
        # original packed GLSL path; do not enter a WebGPU-only tuner and rely on an
        # exception as backend routing.
        if execution == "auto" and _adam_backend_ready():
            candidates = ("stored", "dp4a", "materialized")
            reference = [None]

            def correct(which):
                if which == "stored":
                    return True
                if reference[0] is None:
                    reference[0] = np.asarray(run("stored").get(), np.float32)
                got = np.asarray(run(which).get(), np.float32)
                if not np.all(np.isfinite(got)):
                    return False
                scale = max(1e-6, float(np.abs(reference[0]).max()))
                limit = 0.03 if which == "dp4a" else 1e-3
                return float(np.abs(got - reference[0]).max()) / scale < limit

            execution = _weight_execution("gptq", self.storage_format, self.Kp, self.Np,
                                          int(xf.shape[0]), run, candidates=candidates,
                                          check=correct)
        elif execution == "auto":
            execution = "stored"
        of = run(execution)
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
  outp[i] = scales[g*c.Np + n] * (f32(qv) - (f32(zv) + ZOFFf));
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
  fragColor = fetch(tex_s,g*Np+n)*(float(qv)-(float(zv)+ZOFFf));
}
""".replace("FETCH", _GL_FETCH)
_dqf_k = {"wgpu": set(), "gl": set()}


def _dequant_full(qweight, qzeros, scales, Kp, Np, gs, bits, zoff=0.0):
    zt = 1 if zoff else 0
    key = (bits, zt)
    name = f"dqf{bits}_z{zt}"
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if key not in _dqf_k["wgpu"]:
            plat.addKernel(name, {"source": _gptq_src(_DQF_WGSL, bits, zoff=zoff),
                "bindingTypes": ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "read-only-storage"]})
            _dqf_k["wgpu"].add(key)
        of = _empty((Kp, Np))
        meta = _adam_kernel["make_meta"]((Kp, Np, gs), "u4,u4,u4")
        plat.runKernel({"name": name, "tensors": [qweight.buffer.buffer_id, qzeros.buffer.buffer_id, scales.buffer.buffer_id, of.buffer.buffer_id, meta.buffer_id],
                        "workGroups": {"x": (Kp * Np + 63) // 64, "y": 1, "z": 1}})
        return of
    _webgl_ready()
    plat = _copy_kernel["plat"]
    if key not in _dqf_k["gl"]:
        plat.addKernel(name, {"source": _gptq_src(_GL_DQF, bits, zoff=zoff)})
        _dqf_k["gl"].add(key)
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
    full = _dequant_full(ql.qweight, ql.qzeros, ql.scales, ql.Kp, ql.Np, ql.gs,
                         ql.bits, ql.zero_offset)
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


_GATHER_ROWS_WGSL = """@group(0) @binding(0) var<storage,read> src:array<f32>;
@group(0) @binding(1) var<storage,read> rows:array<f32>;
@group(0) @binding(2) var<storage,read_write> dst:array<f32>;
struct GatherMeta { count:u32, width:u32, }
@group(0) @binding(3) var<storage,read> gather_meta:GatherMeta;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let i=gid.x; if(i>=gather_meta.count*gather_meta.width){return;}
  let r=i/gather_meta.width; let c=i-r*gather_meta.width;
  dst[i]=src[u32(rows[r])*gather_meta.width+c];
}
"""
_GATHER_ROWS_GL = """#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform int _ka_tex_output_texture_w; uniform int count; uniform int width;
uniform sampler2D tex_src; uniform sampler2D tex_rows;
out float fragColor;
float read_at(sampler2D t,int i){int w=textureSize(t,0).x;return texelFetch(t,ivec2(i%w,i/w),0).r;}
void main(){int i=int(gl_FragCoord.x)+int(gl_FragCoord.y)*_ka_tex_output_texture_w;
  if(i>=count*width){fragColor=0.0;return;}
  int r=i/width; int c=i-r*width;
  fragColor=read_at(tex_src,int(read_at(tex_rows,r)+0.5)*width+c);
}
"""
_gather_rows_added = {"gpu": False, "gl": False}


def gather_rows(x, rows):
    """Inference-only direct row gather; never build a dense one-hot matmul.

    ``rows`` are small scheduling metadata. The tensor payload remains on the
    device, and each output lane reads just its selected source lane. Both GPU
    backends expose the same shape and ordering; differentiable callers retain
    their existing matmul route.
    """
    if not isinstance(x, Tensor) or x.ndim != 2 or x.requires_grad:
        return None
    indices = tuple(int(r) for r in rows)
    width = int(x.shape[1]); height = int(x.shape[0])
    if any(r < 0 or r >= height for r in indices):
        raise IndexError("gather row outside tensor")
    if not indices:
        return Tensor(_empty((0, width))) if (_adam_backend_ready() or _webgl_ready()) else Tensor(np.empty((0, width), np.float32))
    if not (_adam_backend_ready() or _webgl_ready()):
        return Tensor(np.asarray(x.data)[list(indices)])
    source = _contig(x.data)
    # Float32 exactly represents every row index addressable by these backends.
    index_data = xp.asarray(np.asarray(indices, dtype=np.float32))
    out = _empty((len(indices), width))
    if _adam_backend_ready():
        plat = _adam_kernel["platform"]
        if not _gather_rows_added["gpu"]:
            plat.addKernel("gather_rows", {"source": _GATHER_ROWS_WGSL,
                "bindingTypes": ["read-only-storage", "read-only-storage",
                                 "storage", "read-only-storage"]})
            _gather_rows_added["gpu"] = True
        meta = _adam_kernel["make_meta"]((len(indices), width), "u4,u4")
        plat.runKernel({"name": "gather_rows",
            "tensors": [source.buffer.buffer_id, index_data.buffer.buffer_id,
                        out.buffer.buffer_id, meta.buffer_id],
            "workGroups": {"x": (len(indices) * width + 63) // 64, "y": 1, "z": 1}})
    else:
        plat = _copy_kernel["plat"]
        if not _gather_rows_added["gl"]:
            plat.addKernel("gather_rows", {"source": _GATHER_ROWS_GL})
            _gather_rows_added["gl"] = True
        plat.runKernel({"name": "gather_rows",
            "inputs": [{"name": "tex_src", "id": source.buffer.buffer_id},
                       {"name": "tex_rows", "id": index_data.buffer.buffer_id}],
            "output": out.buffer.buffer_id,
            "uniforms": [{"name": "_ka_tex_output_texture_w",
                          "value": out.buffer.texture_shape.width, "type": "int"},
                         {"name": "count", "value": len(indices), "type": "int"},
                         {"name": "width", "value": width, "type": "int"}]})
    return Tensor(out)


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
