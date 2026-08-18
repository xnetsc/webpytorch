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


# ============================ the SINGLE global IO callbacks ============================
# A symmetric pair the integrator MUST install — the core ships with NO default IO, so an
# unconfigured SDK fails fast instead of silently reaching the network/disk:
#     async def read(name, offset=0, length=None) -> bytes   # length None = whole file
#     async def write(name, data, offset=0) -> None           # offset 0   = whole file
#     webtorch.set_io_read(read); webtorch.set_io_write(write)
# `offset`/`length` mark ranged/streaming access (int4 weight shards, streamed shard writes)
# so a callback can issue an HTTP Range request or seek into local/remote storage. The
# built-in browser-fetch / host-open implementations are provided as an OPT-IN one-liner
# (`use_default_io()`); nothing is installed until you ask for it.

_IO = None        # read  callback — None until the integrator installs one
_IOW = None       # write callback — None until the integrator installs one

_UNSET_READ = ("webtorch IO is not configured: install a global read callback with "
               "webtorch.set_io_read(async def read(name, offset=0, length=None) -> bytes), "
               "or call webtorch.use_default_io() for the built-in browser-fetch / host-open reader.")
_UNSET_WRITE = ("webtorch IO is not configured: install a global write callback with "
                "webtorch.set_io_write(async def write(name, data, offset=0) -> None), "
                "or call webtorch.use_default_io() for the built-in host/Pyodide writer.")


def set_io_read(callback):
    """Install the GLOBAL async READ callback used for EVERY file/weight the SDK reads:
        async def callback(name, offset=0, length=None) -> bytes
    Your callback fetches the bytes from wherever the data actually lives (disk / OPFS /
    S3 / socket / …); `length is None` means whole-file, otherwise a byte range. REQUIRED
    — until it (or `use_default_io()`) is called, any read raises. `None` clears it back to
    the unconfigured state. Mirror of `set_io_write`."""
    global _IO
    _IO = callback

def get_io_read():
    return _IO

async def io_read(name, offset=0, length=None, io=None):
    """The one entry point the whole SDK uses to read bytes — routes to the global read
    callback (or a per-call `io` override). Raises RuntimeError if none is configured."""
    fn = io or _IO
    if fn is None:
        raise RuntimeError(_UNSET_READ)
    return bytes(await fn(name, offset, length))


def set_io_write(callback):
    """Install the GLOBAL async WRITE callback used for EVERY file the SDK writes:
        async def callback(name, data, offset=0) -> None
    Your callback persists the bytes wherever they should live (disk / OPFS / IndexedDB /
    a download / S3 / …). `offset == 0` means whole-file; a positive offset patches in
    place. REQUIRED — until it (or `use_default_io()`) is called, any write raises. `None`
    clears it back to the unconfigured state. Mirror of `set_io_read`."""
    global _IOW
    _IOW = callback

def get_io_write():
    return _IOW

async def io_write(name, data, offset=0, io=None):
    """The one entry point the whole SDK uses to write bytes — routes to the global write
    callback (or a per-call `io` override). Raises RuntimeError if none is configured."""
    fn = io or _IOW
    if fn is None:
        raise RuntimeError(_UNSET_WRITE)
    await fn(name, bytes(data), offset)


# ---- OPT-IN built-in IO (browser fetch+Range / host+Pyodide open+seek) ----
# Not installed automatically. Call `use_default_io()` (or install these individually) to
# use them; a bare SDK has no IO until you do.

async def _fetch_once(url, rng, headers):
    try:
        from pyodide.http import pyfetch                     # browser (Pyodide)
    except ImportError:
        pyfetch = None
    if pyfetch is not None:
        h = dict(headers or {})
        if rng: h["Range"] = rng
        r = await pyfetch(url, headers=h)
        return bytes(await r.bytes())
    if url.startswith(("http://", "https://")):             # host: urllib
        import urllib.request
        req = urllib.request.Request(url, headers=dict(headers or {}))
        if rng: req.add_header("Range", rng)
        with urllib.request.urlopen(req, timeout=60) as f:
            return f.read()
    raise FileNotFoundError(url)                             # not a URL -> caller reads locally

async def _fetch_range(url, offset=0, length=None, headers=None, retries=4):
    """Read bytes from an http(s) URL or a local path, with an optional byte range. Uses the
    browser's `fetch` inside Pyodide (with a `Range` header) and `urllib`/`open` on the host
    — the shared transport under `default_io_read`, `hf_read`, and `modelscope_read`. Network
    reads are retried with exponential backoff so a streamed load (hundreds of ranged reads)
    survives a transient drop / reset."""
    rng = ("bytes=%d-%d" % (offset, offset + length - 1)) if length is not None else None
    if not url.startswith(("http://", "https://")):         # local file (host)
        try:
            from pyodide.http import pyfetch                 # in browser, a bare path is a URL
            return await _fetch_once(url, rng, headers)
        except ImportError:
            with open(url, "rb") as f:
                if offset: f.seek(offset)
                return f.read() if length is None else f.read(length)
    import asyncio
    last = None
    for attempt in range(retries):
        try:
            return await _fetch_once(url, rng, headers)
        except Exception as e:                              # transient network / TLS / 5xx
            last = e
            if attempt < retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
    raise last

async def default_io_read(name, offset=0, length=None):
    """Built-in reader: browser `fetch`+Range / host `urllib` (for URLs) or `open`+seek (for
    local paths). `name` is resolved as a URL relative to the page origin in the browser."""
    return await _fetch_range(name, offset, length)

async def default_io_write(name, data, offset=0):
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

def use_default_io():
    """Opt in to the built-in IO: browser `fetch`+Range for reads / host+Pyodide `open`
    for reads & writes. One explicit call installs BOTH global callbacks. Convenience for
    demos/examples and the common browser case; production integrators install their own."""
    set_io_read(default_io_read)
    set_io_write(default_io_write)


# ---- model-hub read callbacks (io_read-shaped; you INSTALL them, they are NOT a default) ----
# These are convenience READ callbacks so you can load a model straight from Hugging Face or
# ModelScope by its repo id — no separate download step. They are NOT installed automatically;
# you pick one and pass it to `set_io_read` (writes, if you quantize, need a separate writer).
#
# How they work: a loader turns the repo id you gave it into file names like
# "<org>/<repo>/config.json", "<org>/<repo>/model.safetensors". The callback splits off the
# first two segments as the repo, maps the rest to that hub's file URL, and streams the bytes
# with an HTTP Range request (via `_fetch_range`, same transport as `default_io_read`). A full
# http(s) URL in `name` is fetched as-is, so mixed sources still work.

def _split_repo(name):
    parts = name.lstrip("/").split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])         # (org/repo, filepath)

def hf_read(revision="main", endpoint="https://huggingface.co", token=None):
    """Return an `io_read`-shaped async callback that fetches files directly from the
    **Hugging Face Hub**. Install it, then load by repo id — nothing to pre-download:

        webtorch.set_io_read(webtorch.hf_read())        # + set_io_write(...) only if quantizing
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to `{endpoint}/{org}/{repo}/resolve/{revision}/{path}`,
    read with an HTTP Range request (streaming). `token` (optional) is sent as a Bearer header
    for gated/private repos; `endpoint` can point at a mirror. Reads only."""
    hdr = {"Authorization": "Bearer " + token} if token else None
    async def read(name, offset=0, length=None):
        if name.startswith(("http://", "https://")):
            url = name
        else:
            repo, path = _split_repo(name)
            url = "%s/%s/resolve/%s/%s" % (endpoint.rstrip("/"), repo, revision, path)
        return await _fetch_range(url, offset, length, hdr)
    return read

def modelscope_read(revision="master", endpoint="https://modelscope.cn", token=None):
    """Return an `io_read`-shaped async callback that fetches files directly from
    **ModelScope (魔搭)**. Same shape as `hf_read`:

        webtorch.set_io_read(webtorch.modelscope_read())
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to
    `{endpoint}/api/v1/models/{org}/{repo}/repo?Revision={revision}&FilePath={path}`, read
    with an HTTP Range request. `revision` defaults to ModelScope's `master`. Reads only."""
    hdr = {"Authorization": "Bearer " + token} if token else None
    async def read(name, offset=0, length=None):
        if name.startswith(("http://", "https://")):
            url = name
        else:
            repo, path = _split_repo(name)
            url = "%s/api/v1/models/%s/repo?Revision=%s&FilePath=%s" % (
                endpoint.rstrip("/"), repo, revision, path)
        return await _fetch_range(url, offset, length, hdr)
    return read


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
