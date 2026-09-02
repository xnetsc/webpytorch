"""pytorch-compatible shim backed by webtorch.

Goal: run existing torch model code UNMODIFIED. `torchshim.install()` registers
`torch`, `torch.nn`, `torch.nn.functional`, `torch.nn.init`, `torch.optim` in
sys.modules; then `import torch` resolves to this shim. Kept separate from
webtorch.py -- this file only adds torch-compatible surface at runtime.

nn.Module here mirrors torch's real semantics (_parameters/_modules/_buffers via
__setattr__, named_parameters, apply, buffers), and nn.Linear stores a genuine
(out, in) `weight` Parameter with y = x @ Wt -- so weight tying
(`wte.weight = lm_head.weight`) and state_dict round-trips behave like torch.
"""
import sys, types, math, functools
import numpy as np
from . import _core as wt

T = wt.Tensor
xp = wt.xp
_ORIG_MATMUL = wt.Tensor.matmul     # webtorch's native 2D/3D matmul (before override)


# ============ Tensor compatibility (runtime aliases on webtorch.Tensor) ======
def _view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return self.reshape(*shape)


def _size(self, dim=None):
    return self.shape if dim is None else self.shape[dim]


def _transpose(self, d0, d1):
    nd = self.ndim
    d0 %= nd; d1 %= nd
    axes = list(range(nd)); axes[d0], axes[d1] = axes[d1], axes[d0]
    return self.permute(*axes)


def _ndmatmul(self, other):
    """N-D batched matmul (fold all but last 2 dims into one batch)."""
    a, b = self, other
    if a.ndim <= 3 and b.ndim <= 3:
        return _ORIG_MATMUL(a, b)
    lead = a.shape[:-2]
    M, K, N = a.shape[-2], a.shape[-1], b.shape[-1]
    nb = 1
    for d in lead:
        nb *= d
    return wt.bmm(a.reshape(nb, M, K), b.reshape(nb, K, N)).reshape(*lead, M, N)


def _masked_fill(self, mask, value):
    mf = mask.data.astype(np.float32) if isinstance(mask, T) else xp.asarray(np.asarray(mask, np.float32))
    v = -1e9 if (value == float("-inf") or value < -1e30) else float(value)
    out = T(self.data * (1.0 - mf) + v * mf, self.requires_grad, (self,), "masked_fill")

    def _bw():
        if self.requires_grad:
            self._accum(out.grad * (1.0 - mf))
    out._setback(_bw)
    return out


def _split(self, size, dim=0):
    dim %= self.ndim
    outs, off = [], 0
    n = self.shape[dim]
    for start in range(0, n, size):
        end = min(start + size, n)
        idx = [slice(None)] * self.ndim
        idx[dim] = slice(start, end)
        outs.append(T(wt._contig(self.data[tuple(idx)]), self.requires_grad, (self,), "split"))

    def make_bw(o, s, e):
        def _bw():
            if self.requires_grad:
                g = xp.zeros_like(self.data)
                idx = [slice(None)] * self.ndim
                idx[dim] = slice(s, e)
                g[tuple(idx)] = o.grad
                self._accum(g)
        return _bw
    for o in outs:
        e = off + o.shape[dim]
        o._setback(make_bw(o, off, e)); off = e
    return tuple(outs)


def _getitem(self, key):
    """Detached indexing (inference). Falls back to host for fancy indexing."""
    try:
        return T(wt._contig(self.data[key]))
    except Exception:
        return T(np.ascontiguousarray(self.numpy()[key]))


class _Device:
    def __init__(self, kind="cpu"): self.type = kind
    def __repr__(self): return "device(type='%s')" % self.type
    def __eq__(self, o): return getattr(o, "type", o) == self.type
    def __hash__(self): return hash(self.type)


_CPU = _Device("cpu")


def _install_tensor():
    T.view = _view
    T.size = _size
    T.transpose = _transpose
    T.contiguous = lambda self: self
    T.to = lambda self, *a, **k: self
    T.float = lambda self: self
    T.half = lambda self: self
    T.cuda = lambda self, *a, **k: self
    T.cpu = lambda self: self
    T.detach = lambda self: T(self.data)
    T.masked_fill = _masked_fill
    T.split = _split
    T.__matmul__ = _ndmatmul
    T.matmul = _ndmatmul
    T.dim = lambda self: self.ndim
    T.numel = lambda self: int(self.data.size)
    T.__getitem__ = _getitem
    T.device = property(lambda self: _CPU)
    T.dtype = property(lambda self: np.float32)
    # torch tensors support `t == v` -> mask tensor. webtorch autograd is
    # id-based, so overriding __eq__ is safe; keep identity hashing.
    T.__eq__ = lambda self, other: T((self.data == (other.data if isinstance(other, T) else other)).astype(np.float32))
    T.__ne__ = lambda self, other: T((self.data != (other.data if isinstance(other, T) else other)).astype(np.float32))
    T.__hash__ = lambda self: id(self)


# ============ torch.nn.Module (real torch semantics) ========================
class Module:
    def __init__(self):
        d = self.__dict__
        d["_parameters"] = {}; d["_modules"] = {}; d["_buffers"] = {}
        d["training"] = True

    def _slots(self):
        d = self.__dict__
        for k in ("_parameters", "_modules", "_buffers"):
            if k not in d:
                d[k] = {}
        return d

    def __setattr__(self, name, value):
        d = self._slots()
        if isinstance(value, wt.Parameter):
            d["_parameters"][name] = value; d.pop(name, None)
        elif isinstance(value, Module):
            d["_modules"][name] = value; d.pop(name, None)
        else:
            d["_parameters"].pop(name, None); d["_modules"].pop(name, None)
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        d = self.__dict__
        for slot in ("_parameters", "_modules", "_buffers"):
            s = d.get(slot)
            if s is not None and name in s:
                return s[name]
        raise AttributeError("%s has no attribute %r" % (type(self).__name__, name))

    def register_buffer(self, name, tensor, persistent=True):
        self._slots()["_buffers"][name] = tensor

    def children(self):
        return list(self._slots()["_modules"].values())

    def named_children(self):
        return list(self._slots()["_modules"].items())

    def modules(self):
        out = [self]
        for m in self.children():
            out.extend(m.modules())
        return out

    def named_parameters(self, prefix=""):
        out = []
        for n, p in self._slots()["_parameters"].items():
            if p is not None:
                out.append((prefix + n, p))
        for n, m in self._slots()["_modules"].items():
            out.extend(m.named_parameters(prefix + n + "."))
        return out

    def parameters(self):
        seen, uniq = set(), []
        for _, p in self.named_parameters():
            if id(p) not in seen:
                seen.add(id(p)); uniq.append(p)
        return uniq

    def buffers(self):
        out = list(self._slots()["_buffers"].values())
        for m in self.children():
            out.extend(m.buffers())
        return out

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def zero_grad(self, set_to_none=True):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.__dict__["training"] = mode
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *a, **k): return self
    def cuda(self, *a, **k): return self
    def cpu(self): return self
    def float(self): return self

    def state_dict(self, prefix=""):
        sd = {}
        for n, p in self._slots()["_parameters"].items():
            if p is not None:
                sd[prefix + n] = p.numpy()
        for n, b in self._slots()["_buffers"].items():
            sd[prefix + n] = b.numpy() if isinstance(b, T) else b
        for n, m in self._slots()["_modules"].items():
            sd.update(m.state_dict(prefix + n + "."))
        return sd

    def load_state_dict(self, sd, strict=True, prefix=""):
        for n, p in self._slots()["_parameters"].items():
            k = prefix + n
            if p is not None and k in sd:
                v = sd[k]
                v = v.numpy() if isinstance(v, T) else np.asarray(v, np.float32)
                p.data = xp.asarray(np.ascontiguousarray(v.astype(np.float32)))
        for n, b in self._slots()["_buffers"].items():
            k = prefix + n
            if k in sd and isinstance(b, T):
                v = sd[k]
                b.data = xp.asarray(np.ascontiguousarray(
                    (v.numpy() if isinstance(v, T) else np.asarray(v, np.float32)).astype(np.float32)))
        for n, m in self._slots()["_modules"].items():
            m.load_state_dict(sd, strict, prefix + n + ".")

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


def Parameter(data, requires_grad=True):
    if isinstance(data, wt.Parameter):
        return data
    d = data.data if isinstance(data, T) else np.asarray(data, np.float32)
    return wt.Parameter(d)


# ============ nn layers ======================================================
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32))
        self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None

    def forward(self, x):
        wt_ = self.weight.transpose(0, 1)          # autograd-aware transposed view
        if x.ndim == 2:
            y = x.matmul(wt_)
        else:
            lead = x.shape[:-1]
            y = x.reshape(-1, self.in_features).matmul(wt_).reshape(*lead, self.out_features)
        return (y + self.bias) if self.bias is not None else y


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, **k):
        super().__init__()
        self.num_embeddings = num_embeddings; self.embedding_dim = embedding_dim
        self.weight = Parameter((np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32))

    def forward(self, idx):
        ids = idx.numpy().astype(np.int64) if isinstance(idx, T) else np.asarray(idx, np.int64)
        return wt.embedding(self.weight, ids)


def _layer_norm(x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
    d = x.shape[-1]
    if isinstance(normalized_shape, (tuple, list)) and normalized_shape:
        d = normalized_shape[-1]
    g = weight if weight is not None else T(np.ones((d,), np.float32))
    b = bias if bias is not None else T(np.zeros((d,), np.float32))
    return wt.layernorm(x, g, b, eps)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        d = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        self.eps = eps
        self.weight = Parameter(np.ones((d,), np.float32))
        self.bias = Parameter(np.zeros((d,), np.float32)) if bias else None

    def forward(self, x):
        return _layer_norm(x, (self.weight.shape[-1],), self.weight, self.bias, self.eps)


class ModuleList(Module):
    def __init__(self, mods=()):
        super().__init__()
        for i, m in enumerate(mods):
            self._modules[str(i)] = m

    def append(self, m):
        self._modules[str(len(self._modules))] = m; return self

    def __iter__(self): return iter(list(self._modules.values()))
    def __getitem__(self, i): return list(self._modules.values())[i]
    def __len__(self): return len(self._modules)


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules:
            for k, v in modules.items():
                self._modules[k] = v

    def __getitem__(self, k): return self._modules[k]
    def __setitem__(self, k, v): self._modules[k] = v
    def keys(self): return self._modules.keys()
    def values(self): return self._modules.values()
    def items(self): return self._modules.items()
    def __iter__(self): return iter(self._modules)
    def __len__(self): return len(self._modules)


class Dropout(Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x): return x


class GELU(Module):
    def __init__(self, approximate="none"): super().__init__()
    def forward(self, x): return wt.gelu(x)


class ReLU(Module):
    def forward(self, x): return x.relu()


class Identity(Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, x): return x


def _as_ids(t):
    return t.numpy().astype(np.int64) if isinstance(t, T) else np.asarray(t, np.int64)


class CrossEntropyLoss(Module):
    def __init__(self, **k): super().__init__()
    def forward(self, logits, target): return wt.cross_entropy(logits, _as_ids(target))


class MSELoss(Module):
    def __init__(self, **k): super().__init__()
    def forward(self, a, b): return wt.mse_loss(a, b)


class BCEWithLogitsLoss(Module):
    def __init__(self, **k): super().__init__()
    def forward(self, a, b): return wt.bce_loss(a.sigmoid(), b)


class SiLU(Module):
    def forward(self, x): return wt.silu(x)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, m in enumerate(layers):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


# ============ torch namespace ================================================
class _NoGrad:
    """Works as `with torch.no_grad():` AND as `@torch.no_grad()` decorator."""
    def __enter__(self): return self
    def __exit__(self, *a): return False

    def __call__(self, fn=None):
        if fn is None:
            return self

        @functools.wraps(fn)
        def wrapper(*a, **k):
            return fn(*a, **k)
        return wrapper


def _softmax_dim(x, dim):
    nd = x.ndim; dim %= nd
    if dim == nd - 1:
        return wt.softmax(x)
    axes = list(range(nd)); axes[dim], axes[-1] = axes[-1], axes[dim]
    return wt.softmax(x.permute(*axes)).permute(*axes)


def _topk(x, k, dim=-1, largest=True, sorted=True):
    a = x.numpy()
    order = np.argsort(a, axis=dim)
    if largest:
        order = np.flip(order, axis=dim)
    idx = np.take(order, np.arange(k), axis=dim)
    return T(np.take_along_axis(a, idx, axis=dim).astype(np.float32)), T(idx.astype(np.float32))


def _multinomial(p, num_samples=1, replacement=False):
    a = p.numpy().reshape(-1, p.shape[-1]).astype(np.float64)
    out = np.empty((a.shape[0], num_samples), np.float32)
    for i in range(a.shape[0]):
        pi = np.clip(a[i], 0, None); s = pi.sum()
        pi = pi / s if s > 0 else np.full_like(pi, 1.0 / len(pi))
        out[i] = np.random.choice(len(pi), size=num_samples, replace=replacement, p=pi)
    return T(out)


def _shape_args(s):
    return s[0] if (len(s) == 1 and isinstance(s[0], (tuple, list))) else s


class _FInfo:
    def __init__(self, dtype=np.float32):
        i = np.finfo(np.float32)
        self.min = float(i.min); self.max = float(i.max)
        self.eps = float(i.eps); self.tiny = float(i.tiny)
        self.smallest_normal = float(i.tiny); self.resolution = float(i.resolution)


def _make_utils():
    """torch.utils._pytree + torch.utils.checkpoint -- transformers imports both
    at module scope, so they must exist before `import transformers` works."""
    utils = types.ModuleType("torch.utils")
    pytree = types.ModuleType("torch.utils._pytree")
    nodes = {}

    def register_pytree_node(cls, flatten, unflatten, serialized_type_name=None, **k):
        nodes[cls] = (flatten, unflatten)

    def tree_flatten(x):
        t = type(x)
        if t in nodes:
            children, ctx = nodes[t][0](x)
            return list(children), ("node", t, ctx)
        if isinstance(x, (list, tuple)):
            return list(x), ("seq", t, None)
        if isinstance(x, dict):
            return list(x.values()), ("dict", t, list(x.keys()))
        return [x], ("leaf", t, None)

    def tree_unflatten(vals, spec):
        kind, t, ctx = spec
        if kind == "node":
            return nodes[t][1](vals, ctx)
        if kind == "seq":
            return t(vals)
        if kind == "dict":
            return dict(zip(ctx, vals))
        return vals[0]

    def tree_map(fn, x):
        v, s = tree_flatten(x)
        return tree_unflatten([fn(i) for i in v], s)

    pytree.register_pytree_node = register_pytree_node
    pytree._register_pytree_node = register_pytree_node
    pytree.tree_flatten = tree_flatten
    pytree.tree_unflatten = tree_unflatten
    pytree.tree_map = tree_map
    pytree.SUPPORTED_NODES = nodes

    ckpt = types.ModuleType("torch.utils.checkpoint")
    ckpt.checkpoint = lambda fn, *a, **k: fn(*a)
    utils._pytree = pytree
    utils.checkpoint = ckpt
    return utils, pytree, ckpt


def _make_torch():
    torch = types.ModuleType("torch")
    torch.Tensor = T
    torch.float32 = np.float32; torch.float = np.float32; torch.float16 = np.float16
    torch.long = np.int64; torch.int64 = np.int64; torch.int = np.int32; torch.bool = np.bool_
    torch.device = _Device
    torch.tensor = lambda data, dtype=None, **k: T(data.data if isinstance(data, T) else np.asarray(data, dtype or np.float32))
    torch.zeros = lambda *s, **k: T(np.zeros(_shape_args(s), np.float32))
    torch.ones = lambda *s, **k: T(np.ones(_shape_args(s), np.float32))
    torch.full = lambda s, v, **k: T(np.full(s, v, np.float32))
    torch.empty = lambda *s, **k: T(np.zeros(_shape_args(s), np.float32))
    torch.arange = lambda *a, **k: T(np.arange(*a).astype(np.float32))
    torch.randn = lambda *s, **k: T(np.random.randn(*_shape_args(s)).astype(np.float32))
    torch.tril = lambda x, diagonal=0: T(np.tril(x.numpy() if isinstance(x, T) else np.asarray(x), diagonal))
    torch.triu = lambda x, diagonal=0: T(np.triu(x.numpy() if isinstance(x, T) else np.asarray(x), diagonal))
    torch.cat = lambda ts, dim=0: wt.cat(list(ts), axis=dim)
    torch.stack = lambda ts, dim=0: wt.stack(list(ts), axis=dim)
    torch.matmul = _ndmatmul
    torch.softmax = _softmax_dim
    torch.topk = _topk
    torch.multinomial = _multinomial
    torch.sqrt = lambda x: x.sqrt() if isinstance(x, T) else math.sqrt(x)
    torch.exp = lambda x: x.exp()
    torch.no_grad = _NoGrad
    torch.inference_mode = _NoGrad
    torch.bfloat16 = np.float32
    torch.Size = tuple
    torch.finfo = lambda dtype=np.float32: _FInfo(dtype)
    torch.get_default_dtype = lambda: np.float32
    torch.is_floating_point = lambda t: True
    torch.is_tensor = lambda t: isinstance(t, T)
    torch.where = lambda c, a, b: T(np.where(c.numpy() != 0, a.numpy() if isinstance(a, T) else a,
                                             b.numpy() if isinstance(b, T) else b).astype(np.float32))

    # ---- torch.nn ----
    nn = types.ModuleType("torch.nn")
    for _n, _v in dict(Module=Module, Parameter=Parameter, Linear=Linear, Embedding=Embedding,
                       LayerNorm=LayerNorm, ModuleList=ModuleList, ModuleDict=ModuleDict,
                       Dropout=Dropout, GELU=GELU, ReLU=ReLU, SiLU=SiLU, Sequential=Sequential,
                       Identity=Identity, CrossEntropyLoss=CrossEntropyLoss, MSELoss=MSELoss,
                       BCEWithLogitsLoss=BCEWithLogitsLoss).items():
        setattr(nn, _n, _v)

    # ---- torch.nn.init ----
    init = types.ModuleType("torch.nn.init")

    def normal_(t, mean=0.0, std=1.0):
        t.data = xp.asarray(np.random.normal(mean, std, t.shape).astype(np.float32)); return t

    def zeros_(t):
        t.data = xp.zeros(tuple(t.shape), np.float32); return t

    def ones_(t):
        t.data = xp.asarray(np.ones(tuple(t.shape), np.float32)); return t

    init.normal_ = normal_; init.zeros_ = zeros_; init.ones_ = ones_
    nn.init = init

    # ---- torch.nn.functional (NOTE: intentionally no scaled_dot_product_attention,
    #      so frameworks that `hasattr`-probe for flash attention take the manual path)
    F = types.ModuleType("torch.nn.functional")
    F.softmax = lambda x, dim=-1, **k: _softmax_dim(x, dim)
    F.gelu = lambda x, approximate="none": wt.gelu(x)
    F.relu = lambda x, inplace=False: x.relu()
    F.silu = lambda x, inplace=False: wt.silu(x)
    F.layer_norm = _layer_norm
    F.dropout = lambda x, p=0.0, training=False, inplace=False: x
    F.cross_entropy = lambda logits, tgt, **k: wt.cross_entropy(
        logits, tgt.numpy().astype(np.int64) if isinstance(tgt, T) else np.asarray(tgt, np.int64))
    F.linear = lambda x, w, b=None: (x.matmul(w.transpose(0, 1)) + b) if b is not None else x.matmul(w.transpose(0, 1))
    nn.functional = F

    # ---- torch.optim ----
    optim = types.ModuleType("torch.optim")
    optim.Adam = lambda params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **k: wt.Adam(
        list(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    optim.AdamW = lambda params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **k: wt.Adam(
        list(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    optim.SGD = lambda params, lr=0.01, momentum=0.0, weight_decay=0.0, **k: wt.SGD(
        list(params), lr=lr, momentum=momentum, weight_decay=weight_decay)

    torch.nn = nn; torch.optim = optim
    return torch, nn, F, optim, init


TORCH_VERSION = "2.5.0"


def _as_real_package(mod, name, is_pkg=True):
    """Frameworks probe importlib.util.find_spec(...) / __version__; a bare
    ModuleType has __spec__ = None which makes find_spec() raise ValueError."""
    import importlib.machinery
    mod.__name__ = name
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
    mod.__loader__ = None
    mod.__version__ = TORCH_VERSION
    if is_pkg:
        mod.__path__ = []
    return mod


def install():
    _install_tensor()
    torch, nn, F, optim, init = _make_torch()
    utils, pytree, ckpt = _make_utils()
    torch.utils = utils
    mods = {"torch": torch, "torch.nn": nn, "torch.nn.functional": F,
            "torch.nn.init": init, "torch.optim": optim, "torch.utils": utils,
            "torch.utils._pytree": pytree, "torch.utils.checkpoint": ckpt}
    pkgs = ("torch", "torch.nn", "torch.optim", "torch.utils")
    for name, m in mods.items():
        _as_real_package(m, name, is_pkg=(name in pkgs))
        sys.modules[name] = m
    return torch
