"""Pure-Python stand-ins for the HuggingFace Rust-wheel packages that Pyodide
cannot install (`safetensors`, and a placeholder `tokenizers`). Independent
module -- not part of webtorch, not part of torchshim.

`safetensors` is a trivial format (8-byte header length + JSON header + raw
tensor data), so a pure-Python reader is exact, not an approximation.
"""
import sys, types, json
import numpy as np


def _dtype_to_np(dt):
    return {"F64": np.float64, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
            "U8": np.uint8, "BOOL": np.bool_}.get(dt)


def _decode(raw, dt, shape):
    if dt == "BF16":
        u16 = np.frombuffer(raw, np.uint16).astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    else:
        np_dt = _dtype_to_np(dt)
        if np_dt is None:
            raise ValueError("unsupported safetensors dtype %s" % dt)
        arr = np.frombuffer(raw, np_dt)
    n = 1
    for d in shape:
        n *= d
    return arr[:n].reshape(shape)


def read_safetensors(buf):
    """bytes -> {name: numpy array} (BF16/F16 upcast to fp32 on read)."""
    b = bytes(buf)
    hlen = int.from_bytes(b[:8], "little")
    hdr = json.loads(b[8:8 + hlen].decode("utf-8"))
    hdr.pop("__metadata__", None)
    base = 8 + hlen
    out = {}
    for name, info in hdr.items():
        a, z = info["data_offsets"]
        arr = _decode(b[base + a:base + z], info["dtype"], info["shape"])
        out[name] = arr.astype(np.float32) if arr.dtype in (np.float16, np.float32, np.float64) else arr
    return out


def install(torch_module=None):
    """Register pure-python `safetensors` (+ `safetensors.torch`) in sys.modules."""
    torch = torch_module or sys.modules.get("torch")

    st = types.ModuleType("safetensors")
    st_torch = types.ModuleType("safetensors.torch")
    st_np = types.ModuleType("safetensors.numpy")

    class safe_open:
        def __init__(self, filename, framework="pt", device="cpu"):
            with open(filename, "rb") as f:
                self._t = read_safetensors(f.read())
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def keys(self): return list(self._t.keys())
        def get_tensor(self, name):
            a = self._t[name]
            return torch.tensor(a) if torch is not None else a
        def get_slice(self, name): return self._t[name]
        def metadata(self): return {}

    def load_file(filename, device="cpu"):
        with open(filename, "rb") as f:
            t = read_safetensors(f.read())
        return {k: (torch.tensor(v) if torch is not None else v) for k, v in t.items()}

    def load(data):
        t = read_safetensors(data)
        return {k: (torch.tensor(v) if torch is not None else v) for k, v in t.items()}

    def save_file(tensors, filename, metadata=None):
        raise NotImplementedError("safetensors writing not implemented in the pure-python shim")

    st.safe_open = safe_open
    st.serialize = save_file
    st.deserialize = load
    st.__version__ = "0.4.5"
    st_torch.load_file = load_file
    st_torch.load = load
    st_torch.save_file = save_file
    st_torch.save_model = save_file
    st_torch.storage_ptr = lambda t: 0
    st_torch.storage_size = lambda t: (t.numel() * 4 if hasattr(t, "numel") else 0)
    st_torch.safe_open = safe_open
    st_np.load_file = lambda fn: read_safetensors(open(fn, "rb").read())
    st.torch = st_torch
    st.numpy = st_np

    import importlib.machinery
    for name, m, pkg in (("safetensors", st, True), ("safetensors.torch", st_torch, False),
                         ("safetensors.numpy", st_np, False)):
        m.__name__ = name
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=pkg)
        m.__loader__ = None
        if pkg:
            m.__path__ = []
        sys.modules[name] = m
    return st
