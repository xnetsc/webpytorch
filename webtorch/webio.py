"""IO injection for the SDK. The library CORE never touches files — every read/write
goes through a caller-supplied **async** callback (awaited, so IO never blocks the
worker). This module holds the ONLY adapters that actually perform IO, and resolvers
that auto-distinguish the accepted argument forms so public APIs stay framework-
compatible (accept a path) while ALSO accepting async callbacks / in-memory buffers.

All injected callbacks are ASYNC (return awaitables). Small things (config/index/
tokenizer json) are passed as dict/str/bytes objects, not callbacks.

Tensor source forms (resolve_tensor_reader -> async read_tensor, async has_tensor):
  - async callback read_tensor(name)->ndarray   (used directly)
  - dict {name: ndarray}                          (in-memory)
  - str path  (local safetensors dir/file)        (adapter mmaps — IO isolated here)
  - safetensors bytes
Sink forms (resolve_shard_writer -> async write_shard):
  - async callback write_shard(filename, dict)    (used directly)
  - str dir path                                  (adapter save_file's — IO here)
"""
import numpy as np


async def _maybe_await(v):
    return await v if hasattr(v, "__await__") else v


# ----------------------------- tensor readers -----------------------------
def resolve_tensor_reader(src, io=None):
    """-> (async read_tensor(name)->fp32 ndarray, async has_tensor(name)->bool). For a
    str path, every read is streamed through the global IO callback (or `io` override)."""
    if callable(src):
        has = getattr(src, "has", None)
        async def read(n): return await _maybe_await(src(n))
        async def hasf(n): return (await _maybe_await(has(n))) if callable(has) else True
        return read, hasf
    if isinstance(src, dict):
        async def read(n): return np.asarray(src[n])
        async def hasf(n): return n in src
        return read, hasf
    if isinstance(src, (bytes, bytearray, memoryview)):
        return _safetensors_bytes_reader(bytes(src), io)
    if isinstance(src, str):
        return _local_safetensors_reader(src, io)
    raise TypeError("unsupported tensor source: %r" % type(src))


# safetensors decoding — pure numpy, no `safetensors` package dependency, so it works
# identically on host and in-browser. All bytes come from `io_read` (the global callback).
_ST_DT = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64,
          "I8": np.int8, "U8": np.uint8, "I16": np.int16}

def _st_decode(raw, dt, shape):
    if dt == "BF16":
        arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.frombuffer(raw, _ST_DT.get(dt, np.float32))
    return arr.reshape(shape).astype(np.float32)


def _st_ranged_reader(path, io=None):
    """Reader for ONE .safetensors file, streamed via the global IO callback: the header
    is fetched once (8-byte length + json), then each tensor is a single ranged read —
    the file never needs to fit in memory."""
    st = {}
    async def hdr():
        if "h" not in st:
            import json
            n = int.from_bytes(await io_read(path, 0, 8, io), "little")
            h = json.loads(bytes(await io_read(path, 8, n, io))); h.pop("__metadata__", None)
            st["h"], st["base"] = h, 8 + n
        return st["h"], st["base"]
    async def read(name):
        h, base = await hdr(); info = h[name]; a, z = info["data_offsets"]
        raw = await io_read(path, base + a, z - a, io)
        return _st_decode(raw, info["dtype"], info["shape"])
    async def hasf(name):
        h, _ = await hdr(); return name in h
    return read, hasf, hdr


def _shard_files(path, io=None):
    """(async) name->shard-path map for a served safetensors dir. Prefers the HF index
    (one small ranged json), else a local listdir (host), else a single model.safetensors."""
    d = path.rstrip("/")
    async def resolve():
        try:
            idx = await read_json(d + "/model.safetensors.index.json", io)
            return {k: d + "/" + v for k, v in idx["weight_map"].items()}
        except Exception:
            try:
                import os; shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
            except Exception:
                shards = ["model.safetensors"]
            m = {}
            for s in shards:
                _, _, hdr = _st_ranged_reader(d + "/" + s, io); h, _b = await hdr()
                for k in h: m[k] = d + "/" + s
            return m
    return resolve


def _local_safetensors_reader(path, io=None):
    if path.endswith(".safetensors"):
        r, h, _ = _st_ranged_reader(path, io); return r, h
    resolve = _shard_files(path, io); st = {"map": None, "rd": {}}
    async def _map():
        if st["map"] is None: st["map"] = await resolve()
        return st["map"]
    def _reader(sp):
        if sp not in st["rd"]: st["rd"][sp] = _st_ranged_reader(sp, io)
        return st["rd"][sp]
    async def read(name):
        m = await _map(); r, _h, _hd = _reader(m[name]); return await r(name)
    async def hasf(name):
        m = await _map(); return name in m
    return read, hasf


def _safetensors_bytes_reader(buf, io=None):
    import json
    n = int.from_bytes(buf[:8], "little")
    hdr = json.loads(buf[8:8 + n]); base = 8 + n
    hdr.pop("__metadata__", None)
    async def read(name):
        info = hdr[name]; a, z = info["data_offsets"]
        return _st_decode(buf[base + a:base + z], info["dtype"], info["shape"])
    async def hasf(name): return name in hdr
    return read, hasf


async def tensor_names(src, io=None):
    if isinstance(src, dict): return list(src.keys())
    if isinstance(src, (bytes, bytearray, memoryview)):
        import json; b = bytes(src); n = int.from_bytes(b[:8], "little")
        h = json.loads(b[8:8 + n]); h.pop("__metadata__", None); return list(h.keys())
    if isinstance(src, str):
        if src.endswith(".safetensors"):
            _, _, hdr = _st_ranged_reader(src, io); h, _b = await hdr(); return list(h)
        return list((await _shard_files(src, io)()).keys())
    names = getattr(src, "names", None); return list(names) if names else None


# safetensors serialization — pure numpy, no `safetensors` package. Mirrors _st_decode
# so writes and reads use the SAME format and both flow through the global IO callbacks.
_NP2ST = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("int64"): "I64",
          np.dtype("int32"): "I32", np.dtype("int16"): "I16", np.dtype("int8"): "I8",
          np.dtype("uint8"): "U8", np.dtype("bool"): "BOOL"}

def _st_serialize(tensors, metadata=None):
    import json, struct
    hdr, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        dt = _NP2ST.get(arr.dtype)
        if dt is None:
            raise TypeError("unsupported dtype %s for tensor %r" % (arr.dtype, name))
        b = arr.tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b); off += len(b)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    out = bytearray(); out += struct.pack("<Q", len(hj)); out += hj
    for b in blobs: out += b
    return bytes(out)


# ----------------------------- shard writers -----------------------------
def resolve_shard_writer(dst, io=None):
    """-> async write_shard(filename, {name: ndarray}). A str dst is a directory/prefix;
    the shard is serialized to safetensors bytes and written via the GLOBAL io_write
    callback (identical logic to reads). A callable dst stays a tensor-level sink."""
    if callable(dst):
        async def write(fn, tensors): return await _maybe_await(dst(fn, tensors))
        return write
    if isinstance(dst, str):
        d = dst.rstrip("/")
        async def write(fn, tensors):
            await io_write(d + "/" + fn, _st_serialize(tensors), io=io)
        return write
    raise TypeError("unsupported shard sink: %r" % type(dst))


async def write_json(dst, name, obj, io=None):
    """Persist a small json. dst: dir path (-> global io_write) | async callback(name,obj)
    | None. Path writes go through the same io_write callback as everything else."""
    import json
    if dst is None: return
    if callable(dst): return await _maybe_await(dst(name, obj))
    if isinstance(dst, str):
        await io_write(dst.rstrip("/") + "/" + name, json.dumps(obj).encode(), io=io)
        return
    raise TypeError("unsupported json sink: %r" % type(dst))


# ============================ the SINGLE global IO callback ============================
# EVERY file/weight the SDK reads goes through one async callback:
#     async def io(name: str, offset: int = 0, length: int | None = None) -> bytes
# `length is None` => read the whole file; otherwise read `length` bytes from `offset`
# (used for streaming / HTTP-Range). The user installs their own via `set_io(...)`; the
# default reads via the browser (pyfetch, with a Range header) or the host (open+seek).

async def _default_io(name, offset=0, length=None):
    try:                                                    # browser
        from pyodide.http import pyfetch
        if length is not None:
            r = await pyfetch(name, headers={"Range": "bytes=%d-%d" % (offset, offset + length - 1)})
        else:
            r = await pyfetch(name)
        return bytes(await r.bytes())
    except Exception:                                       # host
        with open(name, "rb") as f:
            if offset: f.seek(offset)
            return f.read() if length is None else f.read(length)

_IO = _default_io

def set_io(callback):
    """Install the GLOBAL async IO callback used for EVERY file/weight the SDK reads:
        async def callback(name, offset=0, length=None) -> bytes
    Your callback fetches the bytes from wherever the data actually lives (disk / OPFS /
    S3 / socket / …); `length is None` means whole-file, otherwise a byte range. Pass
    `None` to restore the built-in default (browser pyfetch / host open)."""
    global _IO
    _IO = callback or _default_io

def get_io():
    return _IO

async def io_read(name, offset=0, length=None, io=None):
    """The one entry point the whole SDK uses to read bytes — routes to the global
    callback (or a per-call `io` override with the same signature)."""
    return bytes(await (io or _IO)(name, offset, length))


# ---- the SINGLE global WRITE callback (mirror of io_read) ----
# EVERY file the SDK writes (quantized shards, index.json, config.json, cached downloads)
# goes through one async callback:
#     async def io_write(name: str, data: bytes, offset: int = 0) -> None
# `offset == 0` writes/creates the whole file; a positive offset patches in place (for
# streamed/chunked writes). Install your own via `set_io_write(...)`; the default writes
# through the local/Pyodide filesystem (open+seek), the exact counterpart of the reader.

async def _default_io_write(name, data, offset=0):
    import os
    d = os.path.dirname(name)
    if d:
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass
    if offset:
        try:
            f = open(name, "r+b")
        except FileNotFoundError:
            f = open(name, "wb")
        with f:
            f.seek(offset); f.write(bytes(data))
    else:
        with open(name, "wb") as f:
            f.write(bytes(data))

_IOW = _default_io_write

def set_io_write(callback):
    """Install the GLOBAL async write callback used for EVERY file the SDK writes:
        async def callback(name, data, offset=0) -> None
    Your callback persists the bytes wherever they should live (disk / OPFS / IndexedDB /
    a download / S3 / …). `offset == 0` means whole-file; a positive offset patches in
    place. Pass `None` to restore the built-in default (local/Pyodide open+seek)."""
    global _IOW
    _IOW = callback or _default_io_write

def get_io_write():
    return _IOW

async def io_write(name, data, offset=0, io=None):
    """The one entry point the whole SDK uses to write bytes — routes to the global write
    callback (or a per-call `io` override with the same signature)."""
    await (io or _IOW)(name, bytes(data), offset)


# ----------------------------- byte-source helpers (built on io_read) -----------------------------
async def read_bytes(src, io=None):
    """Bytes from a source, auto-distinguished: bytes -> as-is; async/sync callable() ->
    awaited; str name -> the global IO callback (via io_read)."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    if callable(src):
        return bytes(await _maybe_await(src()))
    if isinstance(src, str):
        return await io_read(src, io=io)
    raise TypeError("unsupported byte source: %r" % type(src))

async def read_text(src, io=None):
    return (await read_bytes(src, io)).decode("utf-8")

async def read_json(src, io=None):
    import json
    return json.loads(await read_text(src, io))

async def load_npz(src, io=None):
    import io as _io
    return dict(np.load(_io.BytesIO(await read_bytes(src, io)), allow_pickle=True))
