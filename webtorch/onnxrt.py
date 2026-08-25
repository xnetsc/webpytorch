"""Generic ONNX runtime for webtorch — runs ANY ONNX model in the browser.

Two independent, reusable pieces:
  1. A pure-Python ONNX protobuf reader (no `onnx`/protobuf dependency, works in
     Pyodide) -> a lightweight Graph IR (nodes, initializers, inputs, outputs).
  2. An interpreter that executes the graph on host numpy / webtorch GPU ops via
     a generic op registry (`@op("Conv")` etc.).

This is model-agnostic: any ONNX graph whose ops are registered will run. It is
the SDK's generic "run an ONNX model" capability, used by e.g. speaker-embedding
and speech-tokenizer models for voice cloning, but not specific to them.
"""
import numpy as np

# ============================ protobuf wire reader ============================
# ONNX files are serialized protobuf. We only need to *read* a handful of message
# types. Wire format: each field = (tag varint)(payload). tag = (field_num<<3)|wire.
# wire: 0=varint, 1=64bit, 2=length-delimited, 5=32bit.

def _rv(buf, p):                       # read varint -> (value, new_pos)
    shift = 0; out = 0
    while True:
        b = buf[p]; p += 1
        out |= (b & 0x7F) << shift
        if not (b & 0x80): return out, p
        shift += 7

def _fields(buf, start=0, end=None):
    """Yield (field_num, wire_type, value) for a protobuf message slice.
    value: int for varint/32/64, memoryview for length-delimited."""
    if end is None: end = len(buf)
    p = start
    while p < end:
        tag, p = _rv(buf, p)
        fn = tag >> 3; wt = tag & 7
        if wt == 0:
            v, p = _rv(buf, p); yield fn, wt, v
        elif wt == 2:
            ln, p = _rv(buf, p); yield fn, wt, buf[p:p + ln]; p += ln
        elif wt == 1:
            yield fn, wt, buf[p:p + 8]; p += 8
        elif wt == 5:
            yield fn, wt, buf[p:p + 4]; p += 4
        else:
            raise ValueError("bad wire type %d" % wt)

def _grp(buf):                         # collect fields -> {field_num: [values]}
    d = {}
    for fn, wt, v in _fields(buf):
        d.setdefault(fn, []).append(v)
    return d

def _s64(x):                           # unsigned varint -> signed int64
    return x - (1 << 64) if x >= (1 << 63) else x

def _ints(vals):                       # repeated int field (packed wire2 or unpacked wire0) -> [int]
    out = []
    for v in vals:
        if isinstance(v, int):
            out.append(_s64(v))
        else:
            p = 0
            while p < len(v):
                x, p = _rv(v, p); out.append(_s64(x))
    return out

# ONNX enum: TensorProto.DataType -> numpy dtype
_DT = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
       6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64,
       12: np.uint32, 13: np.uint64, 14: np.complex64, 16: np.float32}  # 16=bf16 (handled)

def _str(mv): return bytes(mv).decode("utf-8", "replace")


# ============================ Graph IR ============================
class Node:
    __slots__ = ("op", "inputs", "outputs", "attrs", "name")
    def __init__(self, op, inputs, outputs, attrs, name):
        self.op, self.inputs, self.outputs, self.attrs, self.name = op, inputs, outputs, attrs, name
    def __repr__(self): return "Node(%s, in=%s, out=%s)" % (self.op, self.inputs, self.outputs)


def _pf32(vals):                       # repeated float32 (packed wire2 or unpacked wire5) -> [float]
    out = []
    for v in vals:
        if isinstance(v, int): out.append(float(np.frombuffer(int(v).to_bytes(4, "little"), np.float32)[0]))
        else:
            a = np.frombuffer(bytes(v), np.float32); out.extend(a.tolist())
    return out

def _parse_tensor(buf):                # TensorProto -> (name, np.ndarray)
    # fields: dims=1(int64,packed), data_type=2, float_data=4, int32_data=5,
    #         name=8, raw_data=9, int64_data=7, double_data=10
    g = _grp(buf)
    dt = int(g.get(2, [0])[0])
    name = _str(g[8][0]) if 8 in g else ""
    shape = tuple(int(d) for d in _ints(g.get(1, [])))
    if 9 in g:                                             # raw_data
        raw = bytes(g[9][0])
        if dt == 16:                                       # bfloat16 -> float32
            from . import webio
            arr = webio.bf16_to_f32(raw)
        else:
            arr = np.frombuffer(raw, _DT.get(dt, np.float32))
    elif dt in (1, 16) and 4 in g:  arr = np.array(_pf32(g[4]), np.float32)
    elif dt == 7 and 7 in g:        arr = np.array(_ints(g[7]), np.int64)
    elif dt in (6, 5, 4, 3, 2, 9) and 5 in g: arr = np.array(_ints(g[5]), np.int64).astype(_DT.get(dt, np.int32))
    else:                            arr = np.zeros(shape, _DT.get(dt, np.float32)).reshape(-1)
    # NOTE: np.ascontiguousarray promotes 0-d -> (1,), so reshape LAST (shape=() -> scalar)
    return name, np.ascontiguousarray(arr).reshape(shape)

def _parse_attr(buf):                  # AttributeProto -> (name, value)
    # fields: name=1, f=2(float32 wire5), i=3(int varint), s=4(bytes), t=5(tensor),
    #         floats=7, ints=8, strings=9, type=20
    g = _grp(buf); name = _str(g[1][0]); typ = int(g.get(20, [0])[0])
    if typ == 1:  return name, _pf32(g[2])[0]                     # FLOAT
    if typ == 2:  return name, _s64(int(g[3][0]))                 # INT
    if typ == 3:  return name, _str(g[4][0])                      # STRING
    if typ == 4:  return name, _parse_tensor(g[5][0])[1]          # TENSOR
    if typ == 6:  return name, _pf32(g.get(7, []))                # FLOATS
    if typ == 7:  return name, _ints(g.get(8, []))               # INTS
    if typ == 8:  return name, [_str(x) for x in g.get(9, [])]    # STRINGS
    # fallback by field presence
    if 3 in g:    return name, _s64(int(g[3][0]))
    if 8 in g:    return name, _ints(g[8])
    if 2 in g:    return name, _pf32(g[2])[0]
    if 4 in g:    return name, _str(g[4][0])
    return name, None


class OnnxGraph:
    def __init__(self, nodes, inits, inputs, outputs):
        self.nodes = nodes; self.inits = inits          # {name: ndarray}
        self.inputs = inputs; self.outputs = outputs    # [name]

    @classmethod
    def parse(cls, data):
        """data: bytes of a .onnx file -> OnnxGraph."""
        buf = memoryview(bytes(data)) if not isinstance(data, memoryview) else data
        model = _grp(buf)
        graph_buf = model[7][0]                          # ModelProto.graph = field 7
        g = _grp(graph_buf)
        nodes = []
        for nb in g.get(1, []):                          # GraphProto.node = 1
            ng = _grp(nb)
            attrs = dict(_parse_attr(a) for a in ng.get(5, []))
            nodes.append(Node(
                op=_str(ng[4][0]) if 4 in ng else "",     # op_type = 4
                inputs=[_str(x) for x in ng.get(1, [])],  # input = 1
                outputs=[_str(x) for x in ng.get(2, [])], # output = 2
                attrs=attrs,
                name=_str(ng[3][0]) if 3 in ng else ""))  # name = 3
        inits = {}
        for tb in g.get(5, []):                          # initializer = 5
            nm, arr = _parse_tensor(tb); inits[nm] = arr
        def _io(field):
            out = []
            for vb in g.get(field, []):
                vg = _grp(vb); out.append(_str(vg[1][0]) if 1 in vg else "")
            return out
        return cls(nodes, inits, _io(11), _io(12))       # input=11, output=12


# ============================ executor ============================
# Correctness-first numpy interpreter. Heavy ops (Conv/MatMul) can later be routed
# to webtorch GPU; shape/control ops stay on host. Registry maps op_type -> fn.

_OPS = {}
def op(name):
    def deco(fn): _OPS[name] = fn; return fn
    return deco

def _bc(a, b): return np.broadcast_arrays(a, b)

@op("Add")
def _(x, y, **k): return [np.add(x, y)]
@op("Sub")
def _(x, y, **k): return [np.subtract(x, y)]
@op("Mul")
def _(x, y, **k): return [np.multiply(x, y)]
@op("Div")
def _(x, y, **k):
    if np.issubdtype(np.asarray(x).dtype, np.integer) and np.issubdtype(np.asarray(y).dtype, np.integer):
        return [(np.asarray(x) // np.asarray(y)).astype(np.asarray(x).dtype)]
    return [np.divide(x, y)]
@op("Pow")
def _(x, y, **k): return [np.power(x, y).astype(np.asarray(x).dtype)]
@op("Neg")
def _(x, **k): return [-x]
@op("Sqrt")
def _(x, **k): return [np.sqrt(x)]
@op("Exp")
def _(x, **k): return [np.exp(x)]
@op("Log")
def _(x, **k): return [np.log(x)]
@op("Abs")
def _(x, **k): return [np.abs(x)]
@op("Reciprocal")
def _(x, **k): return [1.0 / x]
@op("Relu")
def _(x, **k): return [np.maximum(x, 0)]
@op("LeakyRelu")
def _(x, alpha=0.01, **k): return [np.where(x >= 0, x, x * alpha)]
@op("Sigmoid")
def _(x, **k): return [1.0 / (1.0 + np.exp(-x))]
@op("Tanh")
def _(x, **k): return [np.tanh(x)]
@op("Erf")
def _(x, **k):
    from math import sqrt
    s = np.sign(x); ax = np.abs(x); t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return [(s * y).astype(x.dtype)]
@op("Softmax")
def _(x, axis=-1, **k):
    e = np.exp(x - x.max(axis, keepdims=True)); return [e / e.sum(axis, keepdims=True)]
@op("Sin")
def _(x, **k): return [np.sin(x)]
@op("Cos")
def _(x, **k): return [np.cos(x)]
@op("Round")
def _(x, **k): return [np.rint(x)]
@op("Sqrt")
def _(x, **k): return [np.sqrt(x)]
@op("Clip")
def _(x, lo=None, hi=None, **k): return [np.clip(x, lo, hi)]

@op("MatMul")
def _(a, b, **k): return [np.matmul(a, b)]
@op("Gemm")
def _(a, b, c=None, alpha=1.0, beta=1.0, transA=0, transB=0, **k):
    A = a.T if transA else a; B = b.T if transB else b
    y = alpha * (A @ B)
    if c is not None: y = y + beta * c
    return [y]

@op("Not")
def _(x, **k): return [~x.astype(bool)]
@op("Equal")
def _(x, y, **k): return [np.equal(x, y)]
@op("Greater")
def _(x, y, **k): return [np.greater(x, y)]
@op("GreaterOrEqual")
def _(x, y, **k): return [np.greater_equal(x, y)]
@op("Less")
def _(x, y, **k): return [np.less(x, y)]
@op("Where")
def _(c, x, y, **k): return [np.where(c.astype(bool), x, y)]
@op("Cast")
def _(x, to=1, **k): return [np.asarray(x).astype(_DT.get(int(to), np.float32))]
@op("Identity")
def _(x, **k): return [x]

@op("Shape")
def _(x, start=0, end=None, **k):
    s = np.array(np.asarray(x).shape, np.int64); return [s[start:end]]
@op("Size")
def _(x, **k): return [np.array(np.asarray(x).size, np.int64)]
@op("Reshape")
def _(x, shp, allowzero=0, **k):
    shp = [int(v) for v in np.asarray(shp).astype(np.int64).ravel()]
    if not allowzero:
        shp = [x.shape[i] if v == 0 else v for i, v in enumerate(shp)]
    return [np.reshape(np.asarray(x), shp)]
@op("Flatten")
def _(x, axis=1, **k):
    sh = x.shape; a = int(np.prod(sh[:axis])) if axis > 0 else 1
    return [x.reshape(a, -1)]
@op("Transpose")
def _(x, perm=None, **k): return [np.transpose(x, perm)]
@op("Concat")
def _(*xs, axis=0, **k): return [np.concatenate([np.asarray(x) for x in xs], axis)]
@op("Unsqueeze")
def _(x, axes=None, **k):
    if axes is None: axes = k.get("_axes")
    axes = sorted(int(a) for a in np.atleast_1d(axes))
    for a in axes: x = np.expand_dims(x, a)
    return [x]
@op("Squeeze")
def _(x, axes=None, **k):
    if axes is None or (hasattr(axes, "__len__") and len(axes) == 0): return [np.squeeze(x)]
    return [np.squeeze(x, tuple(int(a) for a in np.atleast_1d(axes)))]
@op("Gather")
def _(x, idx, axis=0, **k):
    # np.intp (int32 on 32-bit WASM) — int64 indices fail the safe-cast to intp there
    return [np.take(x, np.asarray(idx).astype(np.intp), axis=axis)]
@op("GatherElements")
def _(x, idx, axis=0, **k):
    return [np.take_along_axis(x, np.asarray(idx).astype(np.intp), axis)]
@op("Slice")
def _(x, starts=None, ends=None, axes=None, steps=None, **k):
    r = [slice(None)] * x.ndim
    starts = np.atleast_1d(starts); ends = np.atleast_1d(ends)
    axes = np.atleast_1d(axes) if axes is not None else np.arange(len(starts))
    steps = np.atleast_1d(steps) if steps is not None else np.ones(len(starts), np.int64)
    for st, en, ax, sp in zip(starts, ends, axes, steps):
        ax = int(ax) % x.ndim; en = min(int(en), x.shape[ax]) if int(en) < 9e18 else x.shape[ax]
        r[ax] = slice(int(st), int(en), int(sp))
    return [x[tuple(r)]]
@op("Expand")
def _(x, shp, **k):
    shp = np.asarray(shp).astype(np.int64)
    return [np.broadcast_to(x, np.broadcast_shapes(x.shape, tuple(int(s) for s in shp))).copy()]
@op("ConstantOfShape")
def _(shp, value=None, **k):
    v = 0.0 if value is None else (value.ravel()[0] if hasattr(value, "ravel") else value)
    return [np.full(tuple(int(s) for s in np.asarray(shp)), v, dtype=(value.dtype if hasattr(value, "dtype") else np.float32))]
@op("Range")
def _(start, limit, delta, **k):
    return [np.arange(np.asarray(start).item(), np.asarray(limit).item(), np.asarray(delta).item())]
@op("Tile")
def _(x, reps, **k): return [np.tile(x, tuple(int(r) for r in np.asarray(reps)))]
@op("Pad")
def _(x, pads=None, value=0.0, axes=None, mode="constant", **k):
    pads = np.asarray(pads).astype(np.int64); n = x.ndim
    if axes is None:
        pw = [(int(pads[i]), int(pads[i + n])) for i in range(n)]
    else:
        pw = [(0, 0)] * n
        for j, ax in enumerate(np.atleast_1d(axes)):
            pw[int(ax) % n] = (int(pads[j]), int(pads[j + len(axes)]))
    m = {"constant": "constant", "reflect": "reflect", "edge": "edge"}.get(mode, "constant")
    return [np.pad(x, pw, mode=m, constant_values=value) if m == "constant" else np.pad(x, pw, mode=m)]

def _reduce(fn):
    def f(x, axes=None, keepdims=1, noop_with_empty_axes=0, **k):
        if axes is not None and hasattr(axes, "__len__") and len(axes) == 0:
            if noop_with_empty_axes: return [x]
            axes = None
        ax = None if axes is None else tuple(int(a) for a in np.atleast_1d(axes))
        return [fn(x, axis=ax, keepdims=bool(keepdims))]
    return f
_OPS["ReduceMean"] = _reduce(np.mean); _OPS["ReduceSum"] = _reduce(np.sum)
_OPS["ReduceMax"] = _reduce(np.max); _OPS["ReduceMin"] = _reduce(np.min)
_OPS["ReduceProd"] = _reduce(np.prod); _OPS["ReduceL2"] = _reduce(lambda a, axis, keepdims: np.sqrt(np.sum(a * a, axis=axis, keepdims=keepdims)))

@op("LayerNormalization")
def _(x, scale, bias=None, axis=-1, epsilon=1e-5, **k):
    ax = tuple(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    mu = x.mean(ax, keepdims=True); var = x.var(ax, keepdims=True)
    y = (x - mu) / np.sqrt(var + epsilon) * scale
    if bias is not None: y = y + bias
    return [y]
@op("BatchNormalization")
def _(x, scale, B, mean, var, epsilon=1e-5, momentum=0.9, **k):
    sh = [1, -1] + [1] * (x.ndim - 2)
    return [(x - mean.reshape(sh)) / np.sqrt(var.reshape(sh) + epsilon) * scale.reshape(sh) + B.reshape(sh)]
@op("InstanceNormalization")
def _(x, scale, B, epsilon=1e-5, **k):
    ax = tuple(range(2, x.ndim)); mu = x.mean(ax, keepdims=True); var = x.var(ax, keepdims=True)
    sh = [1, -1] + [1] * (x.ndim - 2)
    return [(x - mu) / np.sqrt(var + epsilon) * scale.reshape(sh) + B.reshape(sh)]

def _conv_nd(x, w, b, strides, pads, dils, groups):
    # x (N,C,*spatial); w (O,C/g,*k). Generic 1d/2d via im2col.
    N = x.shape[0]; C = x.shape[1]; O = w.shape[0]; sp = x.ndim - 2
    strides = strides or [1] * sp; dils = dils or [1] * sp
    pads = pads or [0] * (2 * sp)
    pw = [(0, 0), (0, 0)] + [(pads[i], pads[i + sp]) for i in range(sp)]
    xp = np.pad(x, pw)
    ksz = w.shape[2:]
    outsp = [ (xp.shape[2 + i] - (dils[i] * (ksz[i] - 1) + 1)) // strides[i] + 1 for i in range(sp) ]
    Cg = C // groups; Og = O // groups
    out = np.empty((N, O) + tuple(outsp), np.float32)
    # build index grids for im2col
    for g in range(groups):
        xg = xp[:, g * Cg:(g + 1) * Cg]
        wg = w[g * Og:(g + 1) * Og]                     # (Og, Cg, *k)
        if sp == 1:
            L = outsp[0]; K = ksz[0]
            idx = (strides[0] * np.arange(L)[:, None] + dils[0] * np.arange(K)[None, :]).astype(np.intp)
            cols = xg[:, :, idx]                          # (N,Cg,L,K)
            cols = cols.transpose(0, 2, 1, 3).reshape(N, L, Cg * K)
            og = cols @ wg.reshape(Og, Cg * K).T          # (N,L,Og)
            out[:, g * Og:(g + 1) * Og] = og.transpose(0, 2, 1)
        else:
            H, W2 = outsp; KH, KW = ksz
            ih = (strides[0] * np.arange(H)[:, None, None, None] + dils[0] * np.arange(KH)[None, None, :, None]).astype(np.intp)
            iw = (strides[1] * np.arange(W2)[None, :, None, None] + dils[1] * np.arange(KW)[None, None, None, :]).astype(np.intp)
            cols = xg[:, :, ih, iw]                        # (N,Cg,H,W,KH,KW)
            cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N, H * W2, Cg * KH * KW)
            og = cols @ wg.reshape(Og, Cg * KH * KW).T
            out[:, g * Og:(g + 1) * Og] = og.transpose(0, 2, 1).reshape(N, Og, H, W2)
    if b is not None:
        out += b.reshape([1, O] + [1] * sp)
    return out

@op("Conv")
def _(x, w, b=None, strides=None, pads=None, dilations=None, group=1, kernel_shape=None, auto_pad="NOTSET", **k):
    sp = x.ndim - 2
    if auto_pad in ("SAME_UPPER", "SAME_LOWER") and (pads is None):
        pads = []
        st = strides or [1] * sp; dl = dilations or [1] * sp; ks = kernel_shape or list(w.shape[2:])
        lo = []; hi = []
        for i in range(sp):
            out = -(-x.shape[2 + i] // st[i]); tot = max(0, (out - 1) * st[i] + (dl[i] * (ks[i] - 1) + 1) - x.shape[2 + i])
            a = tot // 2; lo.append(tot - a if auto_pad == "SAME_LOWER" else a); hi.append(a if auto_pad == "SAME_LOWER" else tot - a)
        pads = lo + hi
    return [_conv_nd(x, w, b, strides, pads, dilations, int(group))]

@op("AveragePool")
def _(x, kernel_shape=None, strides=None, pads=None, ceil_mode=0, count_include_pad=0, **k):
    import math as _m
    sp = x.ndim - 2; ks = kernel_shape; st = strides or ks; pd = pads or [0] * (2 * sp)
    pw = [(0, 0), (0, 0)] + [(pd[i], pd[i + sp]) for i in range(sp)]
    xp = np.pad(x, pw, constant_values=0.0)
    _div = (lambda a, b: -(-a // b)) if ceil_mode else (lambda a, b: a // b)
    if sp == 1:
        L = max(1, _div(xp.shape[2] - ks[0], st[0]) + 1)
        out = np.empty(x.shape[:2] + (L,), np.float32)
        for j in range(L):                                    # partial last window (ceil_mode)
            s0 = st[0] * j; seg = xp[:, :, s0:min(s0 + ks[0], xp.shape[2])]
            out[:, :, j] = seg.mean(-1)
        return [out]
    H = (xp.shape[2] - ks[0]) // st[0] + 1; W2 = (xp.shape[3] - ks[1]) // st[1] + 1
    ih = st[0] * np.arange(H)[:, None, None, None] + np.arange(ks[0])[None, None, :, None]
    iw = st[1] * np.arange(W2)[None, :, None, None] + np.arange(ks[1])[None, None, None, :]
    return [xp[:, :, ih, iw].mean((-2, -1))]
@op("GlobalAveragePool")
def _(x, **k): return [x.mean(tuple(range(2, x.ndim)), keepdims=True)]

@op("Constant")
def _(value=None, **k):
    for key in ("value", "value_float", "value_int", "value_ints", "value_floats"):
        if key in k and k[key] is not None: value = k[key]
    return [np.asarray(value)]


# ---- interpreter ----
_ATTR_INPUT_OPS = {   # ops where some inputs are really attributes in newer opsets
}
def run(graph, feeds, want=None):
    env = dict(graph.inits)
    for kf, vf in feeds.items(): env[kf] = np.asarray(vf)
    for nd in graph.nodes:
        ins = [env[i] if i != "" else None for i in nd.inputs]
        fn = _OPS.get(nd.op)
        if fn is None:
            raise NotImplementedError("ONNX op not implemented: %s (node %s)" % (nd.op, nd.name))
        kw = dict(nd.attrs)
        # ops that moved scalar params from attrs to inputs (opset>=13): merge trailing inputs as kwargs
        outs = _dispatch(nd.op, fn, ins, kw)
        for onm, ov in zip(nd.outputs, outs):
            if onm != "": env[onm] = ov
    want = want or graph.outputs
    return [env[w] for w in want]

# map (op, positional inputs) for ops whose ONNX signature uses inputs for what our
# python fn takes as named kwargs (Reshape shape, Slice starts/ends/axes/steps, etc.)
def _dispatch(opn, fn, ins, kw):
    if opn == "Reshape": return fn(ins[0], ins[1], **kw)
    if opn == "Expand": return fn(ins[0], ins[1], **kw)
    if opn == "ConstantOfShape": return fn(ins[0], value=kw.get("value"))
    if opn == "Slice":
        return fn(ins[0], starts=ins[1] if len(ins) > 1 else kw.get("starts"),
                  ends=ins[2] if len(ins) > 2 else kw.get("ends"),
                  axes=ins[3] if len(ins) > 3 else kw.get("axes"),
                  steps=ins[4] if len(ins) > 4 else kw.get("steps"))
    if opn in ("Unsqueeze", "Squeeze"):
        ax = ins[1] if len(ins) > 1 else kw.get("axes")
        return fn(ins[0], axes=ax)
    if opn in ("ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL2"):
        ax = ins[1] if len(ins) > 1 and ins[1] is not None else kw.get("axes")
        return fn(ins[0], axes=ax, keepdims=kw.get("keepdims", 1), noop_with_empty_axes=kw.get("noop_with_empty_axes", 0))
    if opn == "Pad":
        return fn(ins[0], pads=ins[1] if len(ins) > 1 else kw.get("pads"),
                  value=(ins[2] if len(ins) > 2 and ins[2] is not None else kw.get("value", 0.0)),
                  axes=ins[3] if len(ins) > 3 else kw.get("axes"), mode=kw.get("mode", "constant"))
    if opn == "Clip":
        return fn(ins[0], lo=ins[1] if len(ins) > 1 else kw.get("min"), hi=ins[2] if len(ins) > 2 else kw.get("max"))
    if opn == "Range": return fn(ins[0], ins[1], ins[2])
    if opn == "Constant": return fn(**kw)
    if opn == "Tile": return fn(ins[0], ins[1])
    if opn in ("Gather", "GatherElements"): return fn(ins[0], ins[1], axis=kw.get("axis", 0))
    # default: positional inputs + attr kwargs
    return fn(*ins, **kw)


class OnnxModel:
    """Load a served .onnx and run it: OnnxModel(bytes).run({'input': arr}).
    IO is injected — `from_source` accepts a url (str), raw bytes, or a reader callback,
    with an optional `fetch(url)->bytes` injection (defaults to the host/browser reader)."""
    def __init__(self, data): self.graph = OnnxGraph.parse(data)
    @classmethod
    async def from_source(cls, src, io=None):
        from . import webio
        return cls(await webio.read_bytes(src, io))
    @classmethod
    async def from_url(cls, url, io=None):     # back-compat alias
        return await cls.from_source(url, io)
    def run(self, feeds, want=None): return run(self.graph, feeds, want)
