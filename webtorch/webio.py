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

class HttpError(Exception):
    """A non-2xx HTTP response, carrying the status and a snippet of the body so callers can
    tell a rate-limit apart from a real error and surface the server's message."""
    def __init__(self, status, body=""):
        self.status = int(status); self.body = body or ""
        super().__init__("HTTP %d%s" % (self.status, (": " + self.body[:300]) if self.body else ""))

# Wording that marks a rate-limit even when the status is not 429 (some hubs use 200/403/404).
_RATE_WORDS = ("rate limit", "ratelimit", "too many requests", "try again later", "slow down",
               "quota", "throttl", "限速", "限流", "太快", "过于频繁", "频繁", "请求过多", "超出",
               "请稍后", "稍后再试")

def _is_rate_limited(status, body):
    if status == 429:
        return True
    t = (body or "").lower()
    return any(w in t for w in _RATE_WORDS)


async def _fetch_once(url, rng, headers):
    try:
        from pyodide.http import pyfetch                     # browser (Pyodide)
    except ImportError:
        pyfetch = None
    if pyfetch is not None:
        h = dict(headers or {})
        if rng: h["Range"] = rng
        r = await pyfetch(url, headers=h)
        data = bytes(await r.bytes())
        status = int(getattr(r, "status", 200) or 200)
        if status not in (200, 206):                        # pyfetch does not raise on 4xx/5xx
            raise HttpError(status, data[:2048].decode("utf-8", "replace"))
        return data
    if url.startswith(("http://", "https://")):             # host: urllib
        import urllib.request, urllib.error
        req = urllib.request.Request(url, headers=dict(headers or {}))
        if rng: req.add_header("Range", rng)
        try:
            with urllib.request.urlopen(req, timeout=60) as f:
                return f.read()
        except urllib.error.HTTPError as e:                 # 4xx/5xx -> carry status + body
            body = b""
            try: body = e.read()
            except Exception: pass
            raise HttpError(e.code, body[:2048].decode("utf-8", "replace"))
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

def _is_url(name):
    return name.startswith(("http://", "https://"))

def _in_browser():
    try:
        import pyodide.http  # noqa: F401
        return True
    except Exception:
        return False

def use_default_io(cache=True, cache_dir=None, max_parallel=8, prefetch=True, chunk_mb=16, persist=True):
    """Opt in to the built-in IO for **your own files** (your server / CDN / local disk):
    browser `fetch`+Range or host `urllib`/`open`, with the `name` used as-is (NOT a hub — no
    repo-id→URL mapping). Installs both global callbacks in one call.

    By default it **caches network reads** with the same read-ahead + persistence as the hub
    readers, so a reload / re-run does not re-download: in the browser every read is a network
    fetch (from the page origin) and is cached (IndexedDB-persistent by default); on the host,
    `http(s)://` URLs are cached while a **local file path is read directly, never cached**
    (caching a local file would just duplicate it). `cache=False` restores plain, uncached
    fetch/open; `cache_dir`/`max_parallel`/`prefetch`/`chunk_mb`/`persist` mirror `hf_read` and
    apply to the cached (network) reads. Writes always go to the local/Pyodide filesystem."""
    set_io_write(default_io_write)
    if not cache:
        set_io_read(default_io_read); return
    async def fetch(name, offset, length):                   # the network transport (browser fetch / host urllib)
        return await http_get(name, offset, length)
    async def size(name):
        try: return await http_size(name)
        except Exception: return None
    tr = throttle_reads(fetch, max_parallel, http_rate_limited)
    if prefetch:
        tr = prefetch_whole_file(tr, size=size, cache_dir=cache_dir, chunk_mb=chunk_mb)
    cached = make_cached_reader(tr, size=size, cache_dir=cache_dir,
                                chunk_mb=chunk_mb, persist=persist)
    async def read(name, offset=0, length=None):
        if _is_url(name) or _in_browser():                   # network read -> cache it
            return await cached(name, offset, length)
        return await default_io_read(name, offset, length)   # local host file -> read directly
    set_io_read(read)


# ---- model-hub read callbacks (io_read-shaped; you INSTALL them, they are NOT a default) ----
# These are convenience READ callbacks so you can load a model straight from Hugging Face or
# ModelScope by its repo id — no separate download step. They are NOT installed automatically;
# you pick one and pass it to `set_io_read` (writes, if you quantize, need a separate writer).
#
# How they work: a loader turns the repo id you gave it into file names like
# "<org>/<repo>/config.json", "<org>/<repo>/model.safetensors". The callback splits off the
# first two segments as the repo and maps the rest to that hub's file URL. By default it
# CACHES to a local dir with read-ahead:
#   * The first partial (ranged) read of a file starts a background whole-file PREFETCH that
#     streams the file in chunks into a sparse cache — it does NOT block reads.
#   * Every range read is served from cache if those bytes are already present; otherwise it
#     fetches JUST that range now (a separate request, so it never waits for the prefetch to
#     reach that offset) and stores it. The prefetch keeps going and skips ranges already
#     cached. Once the whole file is present it is marked complete and later runs read it
#     straight from disk — no network.
#   * All range reads (prefetch chunks included) go through a bounded queue: at most
#     `max_parallel` concurrent network reads (configurable).
# A full http(s) URL in `name` is fetched as-is. Pass cache=False for pure streaming.

def _split_repo(name):
    parts = name.lstrip("/").split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])         # (org/repo, filepath)

def _default_hub_cache():
    """Cache namespace: $WEBTORCH_CACHE, else ~/.cache/webtorch/hub. On the host it is a real
    directory. In the browser it only names the entries -- they are IndexedDB records written
    per chunk, not files, so nothing here goes through Pyodide's FS."""
    import os
    return os.environ.get("WEBTORCH_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "webtorch", "hub")

def _url_key(url):
    import re
    return re.sub(r"[^A-Za-z0-9._/-]", "_", url.split("://", 1)[-1])   # readable, filesystem-safe

async def _remote_size(url, headers):
    """Total byte length of a remote file (needed to drive prefetch). HEAD Content-Length
    first, then a `bytes=0-0` GET's Content-Range total. None if the server won't say."""
    try:
        from pyodide.http import pyfetch
    except ImportError:
        pyfetch = None
    if pyfetch is not None:
        try:
            r = await pyfetch(url, method="HEAD", headers=dict(headers or {}))
            cl = r.headers.get("content-length") or r.headers.get("Content-Length")
            if cl and str(cl).isdigit(): return int(cl)
        except Exception: pass
        try:
            h = dict(headers or {}); h["Range"] = "bytes=0-0"
            r = await pyfetch(url, headers=h)
            cr = r.headers.get("content-range") or r.headers.get("Content-Range")
            if cr and "/" in cr and cr.rsplit("/", 1)[-1].isdigit():
                return int(cr.rsplit("/", 1)[-1])
        except Exception: pass
        return None
    import urllib.request
    try:
        req = urllib.request.Request(url, headers=dict(headers or {}), method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as f:
            cl = f.headers.get("Content-Length")
            if cl and str(cl).isdigit(): return int(cl)
    except Exception: pass
    try:
        req = urllib.request.Request(url, headers=dict(headers or {}))
        req.add_header("Range", "bytes=0-0")
        with urllib.request.urlopen(req, timeout=60) as f:
            cr = f.headers.get("Content-Range")
            if cr and "/" in cr and cr.rsplit("/", 1)[-1].isdigit():
                return int(cr.rsplit("/", 1)[-1])
    except Exception: pass
    return None


# ---- chunk stores: the cache's persistence layer -----------------------------------
# The unit of storage is a chunk, matching the unit of download. Pyodide's IDBFS cannot be
# used for model-sized files: it holds a file's whole contents in the wasm heap and its
# persistence unit is the file, so a multi-GB download is buffered entirely in memory and
# every sync rewrites the whole record -- chunking the download does not help while the
# storage granularity is the file. Writing chunk-by-chunk fixes both: memory holds one
# chunk, an interrupted download keeps the chunks it already wrote, and nothing is ever
# rewritten. Which chunks exist IS the coverage map, so resuming needs no extra bookkeeping.

_CHUNK_DEFAULT = 16 << 20


class _Store:
    """Chunk-addressed cache storage. Chunk size is fixed per entry at creation and kept in
    its metadata, so indices stay meaningful across sessions."""

    async def open(self):
        pass

    async def meta(self, key):
        """-> {"size": int|None, "chunk": int, "complete": bool} (defaults if absent)."""
        raise NotImplementedError

    async def set_meta(self, key, **kw):
        raise NotImplementedError

    async def have(self, key):
        """-> set of chunk indices currently stored."""
        raise NotImplementedError

    async def get(self, key, i):
        raise NotImplementedError

    async def put(self, key, i, data):
        """-> True when stored, False when storage is full (the caller keeps streaming)."""
        raise NotImplementedError

    async def delete(self, key):
        raise NotImplementedError

    async def keys(self):
        raise NotImplementedError

    async def stored(self, key):
        """-> bytes currently held for this key."""
        raise NotImplementedError


class _DiskStore(_Store):
    """Host filesystem: one sparse file per entry plus a small sidecar recording which
    chunks are present, so an interrupted download resumes here too."""

    def __init__(self, root):
        self.root = root

    def _paths(self, key):
        import os
        p = os.path.join(self.root, _url_key(key))
        return p, p + ".meta"

    def _read_meta(self, key):
        import json, os
        _p, mp = self._paths(key)
        try:
            with open(mp, "r") as f:
                m = json.load(f)
        except Exception:
            m = {}
        # No stored chunk size means the entry is new, so the reader's configured size
        # applies. Defaulting here would silently override it.
        m.setdefault("size", None); m.setdefault("chunk", None)
        m.setdefault("have", []); m.setdefault("complete", False)
        return m

    def _write_meta(self, key, m):
        import json, os
        p, mp = self._paths(key)
        d = os.path.dirname(mp)
        if d:
            try: os.makedirs(d, exist_ok=True)
            except Exception: pass
        with open(mp, "w") as f:
            json.dump(m, f)

    async def meta(self, key):
        m = self._read_meta(key)
        return {"size": m["size"], "chunk": m["chunk"], "complete": m["complete"]}

    async def set_meta(self, key, **kw):
        m = self._read_meta(key)
        for k, v in kw.items():
            if v is not None:
                m[k] = v
        self._write_meta(key, m)

    async def have(self, key):
        return set(self._read_meta(key)["have"])

    async def get(self, key, i):
        import os
        m = self._read_meta(key)
        if i not in set(m["have"]) or not m["chunk"]:
            return None
        p, _mp = self._paths(key)
        try:
            with open(p, "rb") as f:
                f.seek(i * m["chunk"])
                return f.read(m["chunk"])
        except OSError:
            return None

    async def put(self, key, i, data):
        import os
        p, _mp = self._paths(key)
        d = os.path.dirname(p)
        if d:
            try: os.makedirs(d, exist_ok=True)
            except Exception: pass
        if not os.path.exists(p):
            open(p, "wb").close()
        m = self._read_meta(key)
        if not m["chunk"]:
            m["chunk"] = _CHUNK_DEFAULT
            self._write_meta(key, m)
        with open(p, "r+b") as f:
            f.seek(i * m["chunk"]); f.write(data)
        if i not in set(m["have"]):
            m["have"] = sorted(set(m["have"]) | {i})
            self._write_meta(key, m)
        return True

    async def delete(self, key):
        import os
        existed = False
        for q in self._paths(key):
            try:
                os.remove(q); existed = True
            except OSError:
                pass
        return existed

    async def keys(self):
        import os
        out = []
        if os.path.isdir(self.root):
            for dp, _dn, fns in os.walk(self.root):
                for f in fns:
                    if f.endswith(".meta"):
                        continue
                    ap = os.path.join(dp, f)
                    out.append(os.path.relpath(ap, self.root).replace("\\", "/"))
        return sorted(out)

    async def stored(self, key):
        import os
        p, _mp = self._paths(key)
        try: return os.path.getsize(p)
        except OSError: return 0


def _is_quota_error(e):
    """True for the browser's "storage is full" signal, whatever it is wrapped in."""
    name = getattr(e, "name", "") or ""
    text = "%s %s" % (name, e)
    return ("QuotaExceeded" in text or "quota" in text.lower()
            or "NS_ERROR_DOM_QUOTA" in text)


def _idb_req(req):
    """An IDBRequest as an awaitable."""
    from js import Promise
    from pyodide.ffi import create_proxy
    def executor(resolve, reject):
        def ok(_e=None): resolve(req.result)
        def err(_e=None): reject(req.error)
        req.onsuccess = create_proxy(ok)
        req.onerror = create_proxy(err)
    return Promise.new(create_proxy(executor))


class _IdbStore(_Store):
    """Browser: one IndexedDB record per chunk.

    Each chunk is written through as it arrives, so the wasm heap holds one chunk rather
    than one model, whatever the file's size. Nothing is rewritten as the download grows,
    and the set of stored chunk records is itself the coverage map, so a download that was
    interrupted -- by a reload, a closed tab, a network failure -- resumes from exactly
    what it had.
    """

    DB = "webtorch-chunks"
    CHUNKS = "chunks"
    META = "meta"

    def __init__(self, root):
        self.root = root                       # kept only so entries stay separable per cache dir
        self.db = None
        self.full = False                      # set once the origin's quota is exhausted

    async def open(self):
        if self.db is not None:
            return
        from js import indexedDB
        from pyodide.ffi import create_proxy
        req = indexedDB.open(self.DB, 1)
        def upgrade(_e=None):
            db = req.result
            if not db.objectStoreNames.contains(self.CHUNKS):
                db.createObjectStore(self.CHUNKS)
            if not db.objectStoreNames.contains(self.META):
                db.createObjectStore(self.META)
        req.onupgradeneeded = create_proxy(upgrade)
        self.db = await _idb_req(req)

    def _store(self, name, mode):
        return self.db.transaction(name, mode).objectStore(name)

    def _ck(self, key, i):
        return "%s#%d" % (key, i)

    async def meta(self, key):
        await self.open()
        m = await _idb_req(self._store(self.META, "readonly").get(key))
        d = m.to_py() if m is not None and hasattr(m, "to_py") else (m or None)
        d = dict(d) if d else {}
        return {"size": d.get("size"), "chunk": d.get("chunk"),
                "complete": bool(d.get("complete", False))}

    async def set_meta(self, key, **kw):
        await self.open()
        cur = await self.meta(key)
        for k, v in kw.items():
            if v is not None:
                cur[k] = v
        from pyodide.ffi import to_js
        from js import Object
        await _idb_req(self._store(self.META, "readwrite").put(
            to_js(cur, dict_converter=Object.fromEntries), key))

    async def have(self, key):
        await self.open()
        from js import IDBKeyRange
        rng = IDBKeyRange.bound(key + "#", key + "#\uffff")
        ks = await _idb_req(self._store(self.CHUNKS, "readonly").getAllKeys(rng))
        out = set()
        for k in (ks or []):
            try: out.add(int(str(k).rsplit("#", 1)[1]))
            except (ValueError, IndexError): pass
        return out

    async def get(self, key, i):
        await self.open()
        v = await _idb_req(self._store(self.CHUNKS, "readonly").get(self._ck(key, i)))
        if v is None:
            return None
        return bytes(v.to_py()) if hasattr(v, "to_py") else bytes(v)

    async def put(self, key, i, data):
        """Store one chunk. Returns False when storage is full rather than raising.

        A model can legitimately be larger than the origin's storage quota -- the browser
        grants roughly a share of free disk, which is unrelated to what the GPU can run.
        Because chunks are independent, a cache that only holds part of a file is still
        useful: the stored chunks are served from storage and the rest streams. So hitting
        the quota degrades caching instead of failing the load.
        """
        await self.open()
        if self.full:
            return False
        from js import Uint8Array
        try:
            buf = Uint8Array.new(len(data))
            buf.assign(data)                   # one copy into the JS heap
        except Exception:
            from pyodide.ffi import to_js
            buf = to_js(data)
        try:
            await _idb_req(self._store(self.CHUNKS, "readwrite").put(buf, self._ck(key, i)))
        except Exception as e:
            if _is_quota_error(e):
                self.full = True               # stop trying; reads keep working, writes stream
                return False
            raise
        finally:
            del buf                            # nothing here outlives the write
        return True

    async def delete(self, key):
        await self.open()
        from js import IDBKeyRange
        rng = IDBKeyRange.bound(key + "#", key + "#\uffff")
        existed = bool(await self.have(key))
        await _idb_req(self._store(self.CHUNKS, "readwrite").delete(rng))
        await _idb_req(self._store(self.META, "readwrite").delete(key))
        return existed

    async def keys(self):
        await self.open()
        ks = await _idb_req(self._store(self.META, "readonly").getAllKeys())
        return sorted(str(k) for k in (ks or []))

    async def stored(self, key):
        """Summed from the chunk records, so a partial entry reports what it really holds."""
        await self.open()
        from js import IDBKeyRange
        rng = IDBKeyRange.bound(key + "#", key + "#\uffff")
        vals = await _idb_req(self._store(self.CHUNKS, "readonly").getAll(rng))
        return int(sum(int(v.byteLength) for v in (vals or [])))


def _in_browser():
    try:
        import js                                   # noqa: F401
        from js import indexedDB                    # noqa: F401
        return True
    except Exception:
        return False


# Progress is reported on bytes SERVED, not bytes fetched. A model already in the cache
# does no network at all, and counting fetches leaves that load looking frozen -- which is
# exactly what a cache is for. Counting reads covers both, with one number.
_progress = {"cb": None, "state": {}}


def set_read_progress(cb):
    """Install `cb(info)`, called as reads are served -- how far a LOAD has got. `None` clears it.

    `info` is a dict, so it can gain fields without breaking callers:

        key       the entry being read
        done      cumulative bytes served for it since the hook was installed
        total     its full length, or None when it is not known
        elapsed   seconds since the first read of this entry
        rate      bytes/second being served, smoothed over recent reads

    This is deliberately about loading, not about transport. `read(name, offset, length)`
    is the whole contract; whether the bytes come from HTTP, a local file, OPFS or a
    database is the callback's business, so there is no download speed here -- a reader
    backed by a local disk has no download. For that, see `set_download_progress`, which
    belongs to the HTTP transport the hub readers are built on.
    """
    _progress["cb"] = cb
    _progress["state"].clear()      # a fresh hook measures a fresh load, not the last one


def get_read_progress():
    return _progress["cb"]


def _prog_state(key):
    st = _progress["state"].get(key)
    if st is None:
        import time
        now = time.monotonic()
        st = _progress["state"][key] = {"t0": now, "t": now, "done": 0, "rate": 0.0}
    return st


def _report(key, done, total):
    cb = _progress["cb"]
    if cb is None:
        return
    import time
    st = _prog_state(key)
    now = time.monotonic()
    dt = now - st["t"]
    if dt >= 0.2:                             # smooth over a window, not per read
        a = 0.4                               # recent enough to feel live, steady enough to read
        st["rate"] = a * ((done - st["done"]) / dt) + (1 - a) * st["rate"]
        st["t"] = now; st["done"] = done
    el = max(now - st["t0"], 1e-6)
    try:
        # Until the window has produced a figure, the average since the first read is the
        # honest answer -- better than reporting zero while bytes are plainly moving.
        cb({"key": key, "done": done, "total": total, "elapsed": now - st["t0"],
            "rate": st["rate"] or (done / el)})
    except Exception:
        pass                                  # a broken meter must not break a load


def _make_store(root):
    """IndexedDB in the browser, real files on the host -- same chunk-addressed interface."""
    return _IdbStore(root) if _in_browser() else _DiskStore(root)


class _CachedFile:
    """One remote file, cached chunk by chunk, with a background whole-file read-ahead.

    A read is served from stored chunks; any chunk it needs that is missing is fetched on
    the spot -- never waiting for the read-ahead to reach it -- written through to storage,
    and only then used.

    On memory: what this bounds is *growth with file size*. A 13 GB file costs the same
    live bytes as a 400 MB one, because nothing accumulates. It is not a hard cap. The
    live set is roughly

        (max_parallel + chunks spanned by the current read) x chunk

    so with the defaults (8 x 16 MB) a read of one chunk peaks near 144 MB, and each chunk
    is briefly duplicated while it is copied into the JS heap on its way to IndexedDB.
    Beyond that: the wasm heap never shrinks, so the high-water mark persists; repeated
    allocations of this size fragment it; and a caller that asks for a whole multi-GB file
    in one read still gets exactly what it asked for. Lower `max_parallel` or `chunk_mb` to
    trade throughput for footprint.
    """

    def __init__(self, cache, key):
        self.c = cache
        self.key = key
        self.size = None
        self.chunk = cache.chunk
        self.have = set()                     # chunk indices in storage == the coverage map
        self.complete = False
        self.served = 0                       # cumulative bytes handed to the caller
        self._loaded = False

    async def _load(self):
        """Adopt whatever a previous session stored for this key."""
        if self._loaded:
            return
        self._loaded = True
        m = await self.c.store.meta(self.key)
        self.size = m["size"]
        self.chunk = m["chunk"] or self.c.chunk
        self.complete = m["complete"]
        self.have = await self.c.store.have(self.key)

    def _count(self):
        return None if self.size is None else (self.size + self.chunk - 1) // self.chunk

    async def _chunk_bytes(self, i):
        """Chunk `i`, fetched and stored first if it is not there yet.

        The total length is often unknowable in a browser: `Content-Length` and
        `Content-Range` are not CORS-safelisted, so a cross-origin host that does not
        expose them leaves the size unknown. That must not disable caching, so a chunk is
        simply requested at full width; one that comes back short is the last one, which
        is where the total length is learnt.
        """
        # Always ask the cache first, never a remembered answer. Whoever else is filling
        # this entry -- a transport reading ahead, another tab, a previous session -- writes
        # to the same store, and a set captured when the entry was opened would send us
        # back to the reader for bytes that are already here.
        b = await self.c.store.get(self.key, i)
        if b is not None:
            self.have.add(i)
            return b
        self.have.discard(i)
        off = i * self.chunk
        n = self.chunk if self.size is None else max(0, min(self.chunk, self.size - off))
        data = await self.c._net(self.key, off, n)
        stored = await self.c.store.put(self.key, i, data)
        if stored:                            # a refused write must not look like a hit
            self.have.add(i)
        if self.size is None and len(data) < self.chunk:
            self.size = off + len(data)       # short read == EOF
            await self.c.store.set_meta(self.key, size=self.size, chunk=self.chunk)
        elif self.size is None:
            await self.c.store.set_meta(self.key, chunk=self.chunk)
        cnt = self._count()
        if cnt is not None and not self.complete and len(self.have) >= cnt:
            self.complete = True
            await self.c.store.set_meta(self.key, complete=True)
        return data


    async def read(self, offset, length):
        await self._load()
        if self.size is None:
            self.size = await self.c._size(self.key)   # None whenever the host hides it
            if self.size is not None:
                await self.c.store.set_meta(self.key, size=self.size, chunk=self.chunk)
        if length is None and self.size is None:
            return await self.c._net(self.key, offset, None)   # open-ended, unknown extent
        end = self.size if length is None else offset + length
        if self.size is not None:
            end = min(end, self.size)
        if end <= offset:
            return b""
        c0 = offset // self.chunk
        c1 = (end - 1) // self.chunk
        out = bytearray(end - offset)                  # filled in place: no second full copy
        pos = 0
        for i in range(c0, c1 + 1):
            base = i * self.chunk
            b = await self._chunk_bytes(i)
            lo = max(offset, base) - base
            hi = min(end, base + len(b)) - base        # a short chunk is the file's end
            if hi > lo:
                out[pos:pos + (hi - lo)] = b[lo:hi]
                pos += hi - lo
            short = len(b) < self.chunk
            del b
            if short:
                break
        return bytes(out[:pos])


# Escalating cooldowns (seconds) applied each time concurrency is forced to 0 by repeated
# rate-limiting; capped at 3 minutes. A rate-limit after the last one aborts the read.
_COOLDOWNS = [30, 60, 120, 180]

class _AdaptiveLimiter:
    """A concurrency gate whose live limit adapts to a hub's real capacity. `ceiling` is the
    max parallel network reads; the live `limit` self-tunes:

    - **Success** additively climbs `limit` back toward `ceiling` (and reopens it from 0).
    - **Explicit rate-limit** **halves** `limit`. What counts as one is decided by the
      `is_rate_limited(exc)` predicate the transport supplies -- this gate caches, it does not
      know whether the reader behind it speaks HTTP or reads a local disk. With no predicate,
      nothing is treated as rate limiting and only the rules below apply.
    - **A non-rate-limit error while other reads are still in flight** is treated as an
      *undisclosed* capacity signal — a transport may simply error instead of saying so —
      so the concurrency is capped to the number still succeeding (the sweet spot) and the read
      is retried, NOT raised.
    - **A non-rate-limit error with no read in flight** is a genuine failure: it is not retried
      and propagates immediately with the server's message.
    - When (and only when) an explicit rate-limit drives `limit` to 0 **with nothing in
      flight**, it cools down for an escalating interval (30→60→120→180s, cap 3 min) then
      reopens to 1; a rate-limit persisting past the last cooldown aborts. A cooldown/abort
      never fires while any read is still succeeding."""
    def __init__(self, ceiling, is_rate_limited=None):
        self.is_rate_limited = is_rate_limited   # supplied by the transport, not assumed here
        self.ceiling = max(1, int(ceiling))
        self.limit = self.ceiling
        self.inflight = 0
        self.succ = 0
        self.cool_level = 0
        self.cooling = False
        self.failed = False
        self._cond = None

    def _c(self):
        import asyncio
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def _acquire(self):
        c = self._c()
        async with c:
            while True:
                if self.failed:
                    raise RuntimeError("still rate-limited after cooling down to 3 min")
                if self.limit >= 1 and self.inflight < self.limit:
                    self.inflight += 1
                    return
                await c.wait()

    async def _release_ok(self):
        c = self._c()
        async with c:
            self.inflight -= 1
            self.cool_level = 0
            if self.limit == 0:
                self.limit = 1; self.succ = 0                 # a success reopens from 0
            else:
                self.succ += 1
                if self.limit < self.ceiling and self.succ >= 2:
                    self.limit += 1; self.succ = 0
            c.notify_all()

    async def _on_error(self, e):
        """Classify a failed read -> 'raise' (genuine, propagate e) | 'abort' (rate-limit gave
        up) | 'retry' (back off and try again)."""
        import asyncio
        c = self._c()
        async with c:
            self.inflight -= 1
            # Asked of the transport, not decided here: the cache does not know what a
            # reader speaks, so it cannot know what "rate limited" looks like. Without a
            # classifier no error counts as rate limiting, and the transport-agnostic rules
            # below still apply.
            is_rate = bool(self.is_rate_limited and self.is_rate_limited(e))
            if not is_rate and self.inflight == 0:
                c.notify_all(); return "raise"                # solo non-rate error -> genuine failure
            # explicit rate-limit, OR a non-rate error while peers still succeed (undisclosed
            # rate-limit): back off to a sustainable concurrency and retry.
            self.succ = 0
            if is_rate:
                self.limit = (self.limit // 2) if self.limit > 1 else 0
            else:
                self.limit = self.inflight                    # peers still running = the sweet spot
            if self.limit == 0 and self.inflight == 0 and not self.cooling:
                if self.cool_level >= len(_COOLDOWNS):
                    self.failed = True; c.notify_all(); return "abort"
                dur = _COOLDOWNS[self.cool_level]; self.cool_level += 1
                self.cooling = True
                asyncio.ensure_future(self._cooldown(dur))
            c.notify_all(); return "retry"

    async def _cooldown(self, dur):
        import asyncio
        await asyncio.sleep(dur)
        c = self._c()
        async with c:
            self.cooling = False
            if self.limit < 1: self.limit = 1                 # reopen to a single probe
            c.notify_all()

    async def run(self, attempt):
        while True:
            await self._acquire()
            try:
                r = await attempt()
            except Exception as e:
                action = await self._on_error(e)
                if action == "raise":
                    raise e                                     # genuine non-rate error (solo)
                if action == "abort":
                    # The transport's own error is the message; this only says why the gate
                    # stopped retrying it.
                    raise RuntimeError("gave up after repeated rate-limiting: %s" % e) from e
                continue                                        # retry (respects reduced limit / cooldown)
            await self._release_ok()
            return r


class _CacheLayer:
    """Owns a cache dir and one `_CachedFile` per key. Caching, and nothing else.

    Bytes it does not have come from the injected `fetch(key, offset, length)`, and a key's
    total length from the injected `size(key)`. It does not know what those callbacks speak,
    so it does not throttle them, does not decide how many may run at once, and does not
    interpret their failures -- an error from the reader is raised as it came, because only
    the reader knows what went wrong. Concurrency limits, rate-limit handling and backoff
    belong to the transport, which is where the meaning of a 429 exists."""
    def __init__(self, fetch, size, cache_dir, chunk=16 << 20, persist=True):
        self.fetch = fetch; self.size_fn = size; self.cache_dir = cache_dir
        self.chunk = int(chunk); self.persist = persist
        self.files = {}; self.tasks = []; self._loaded = False
        self.store = _make_store(cache_dir) if cache_dir else None

    async def _net(self, key, offset, length):
        """Bytes the cache does not have. Whatever the reader raises bubbles up unchanged."""
        return await self.fetch(key, offset, length)

    async def _size(self, key):
        if self.size_fn is None: return None
        try: return await self.size_fn(key)
        except Exception: return None

    async def read(self, key, offset, length):
        if not self.cache_dir:
            data = await self._net(key, offset, length)       # cache disabled -> pure streaming
            self.served = getattr(self, "served", 0) + len(data)
            _report(key, self.served, None)
            return data
        if not self._loaded:
            self._loaded = True
            await self.store.open()
        st = self.files.get(key)
        if st is None:
            st = self.files[key] = _CachedFile(self, key)
        data = await st.read(offset, length)
        st.served += len(data)
        _report(key, st.served, st.size)
        return data


# ============================ generic cached-reader tool ============================
# `make_cached_reader` wraps ANY async transport with the cache + read-ahead + adaptive
# concurrency + (browser) persistence used by the hub readers. Use it when you implement your
# own `set_io_read` callback over a custom source (S3, a signed CDN, your own server, …); the
# built-in `hf_read` / `modelscope_read` are just clients of it (see below).

# Download speed is a property of THESE TOOLS, not of the SDK. `hf_read` /
# `modelscope_read` / `use_default_io` are convenience functions we ship -- a kind of user
# callback, not part of the contract -- and they happen to speak HTTP, which is the only
# reason a download speed exists to report. The SDK's own contract is
# `read(name, offset, length) -> bytes`; a reader someone writes over a local disk has no
# download to measure. Loading progress lives with the loader, in `set_read_progress`.
_dl = {"cb": None, "t0": None, "t": 0.0, "n": 0, "last": 0, "rate": 0.0}


def set_download_progress(cb):
    """Install `cb(info)` for the built-in HTTP transport these tools use. `None` clears it.

        url    the request that just completed
        bytes  cumulative bytes received since the hook was installed
        rate   bytes/second, smoothed -- the download speed

    Fires for anything going through `http_get`, which is what `hf_read`,
    `modelscope_read` and `use_default_io` are built on. A reader of your own that does not
    use `http_get` will not report here, and should not: it may not be downloading anything.
    """
    import time
    _dl.update({"cb": cb, "t0": time.monotonic(), "t": time.monotonic(),
                "n": 0, "last": 0, "rate": 0.0})


def get_download_progress():
    return _dl["cb"]


def _note_download(url, n):
    cb = _dl["cb"]
    if cb is None:
        return
    import time
    now = time.monotonic()
    _dl["n"] += n
    dt = now - _dl["t"]
    if dt >= 0.2:
        a = 0.4
        _dl["rate"] = a * ((_dl["n"] - _dl["last"]) / dt) + (1 - a) * _dl["rate"]
        _dl["t"] = now; _dl["last"] = _dl["n"]
    el = max(now - (_dl["t0"] or now), 1e-6)
    try:
        cb({"url": url, "bytes": _dl["n"], "rate": _dl["rate"] or (_dl["n"] / el)})
    except Exception:
        pass


def http_rate_limited(exc):
    """True when `exc` from the HTTP transport means "slow down".

    This is the transport's own knowledge -- a 429, or a body that says so in as many words
    -- and it is handed to `make_cached_reader` rather than assumed by it, because the cache
    has no idea what its reader speaks.
    """
    return isinstance(exc, HttpError) and _is_rate_limited(exc.status, exc.body)


async def http_get(url, offset=0, length=None, headers=None):
    """The built-in HTTP range transport (browser `fetch` / host `urllib`). Raises `HttpError`
    on a non-2xx status. A ready-made `fetch` building block for `make_cached_reader`."""
    rng = ("bytes=%d-%d" % (offset, offset + length - 1)) if length is not None else None
    data = await _fetch_once(url, rng, headers)
    _note_download(url, len(data))            # this transport IS HTTP, so it has a speed
    return data

async def http_size(url, headers=None):
    """Total byte length of an http(s) file (HEAD, else a `bytes=0-0` GET's Content-Range).
    A ready-made `size` building block for `make_cached_reader` (drives prefetch)."""
    return await _remote_size(url, headers)

def throttle_reads(fetch, max_parallel=8, is_rate_limited=None):
    """Wrap a transport with an adaptive concurrency gate. Returns a `fetch`-shaped callable.

    This belongs to the transport, not to the cache: how many requests a host will take at
    once, what it does when pushed too hard, and what "slow down" even looks like are facts
    about the thing being read, and only its reader knows them. Compose it around your own
    reader before handing that to `make_cached_reader`; the built-in HTTP tools do exactly
    that.

    `is_rate_limited(exc) -> bool` tells the gate which failures mean "slow down" -- for
    HTTP that is `http_rate_limited`. See `_AdaptiveLimiter` for how the limit self-tunes.
    """
    lim = _AdaptiveLimiter(max_parallel, is_rate_limited)

    async def gated(key, offset, length):
        return await lim.run(lambda: fetch(key, offset, length))
    return gated


def prefetch_whole_file(fetch, size=None, cache_dir=None, chunk_mb=16, key=None):
    """Wrap a transport so that touching a file starts filling the cache with the rest of it.

    Reading sixteen bytes of a header usually means the whole file is wanted, but only the
    transport knows whether fetching ahead is a good idea -- whether ranges are cheap, whether
    the host minds, whether the bytes even come from somewhere worth reading ahead from. So
    the policy lives here, not in the cache: this fetches the remaining chunks in the
    background and writes them through the cache's own write interface. The cached reader
    then simply finds them and never calls the transport for those ranges again.

    Returns a `fetch`-shaped callable to hand to `make_cached_reader`.
    """
    import asyncio
    chunk = chunk_mb << 20
    started = set()
    tasks = []

    async def fill(k):
        try:
            store = _make_store(cache_dir or _default_hub_cache())
            await store.open()
            total = await size(k) if size is not None else None
            meta = await store.meta(k)
            if total is not None and meta["size"] is None:
                await store.set_meta(k, size=total, chunk=chunk)
            have = await store.have(k)
            i = 0
            while total is None or i * chunk < total:
                if i not in have:
                    n = chunk if total is None else min(chunk, total - i * chunk)
                    b = await fetch(k, i * chunk, n)
                    if not await store.put(k, i, b):
                        return                       # storage full: stop, reads still work
                    if total is None and len(b) < chunk:
                        await store.set_meta(k, size=i * chunk + len(b), chunk=chunk)
                        break
                i += 1
            m = await store.meta(k)
            if m["size"] is not None:
                need = (m["size"] + chunk - 1) // chunk
                if len(await store.have(k)) >= need:
                    await store.set_meta(k, complete=True)
        except Exception:
            pass                                     # read-ahead is an optimisation, never a failure

    async def ahead(k, offset, length):
        if k not in started:
            started.add(k)
            tasks.append(asyncio.ensure_future(fill(k)))
        return await fetch(k, offset, length)

    return ahead


def make_cached_reader(fetch, size=None, key=None, cache=True, cache_dir=None,
                       chunk_mb=16, persist=True):
    """Wrap an async transport with the cache, giving back an `io_read`-shaped callback
    `read(name, offset=0, length=None) -> bytes` for `webtorch.set_io_read`.

    Caching, and only caching. Bytes it already has come from storage; bytes it does not
    come from `fetch`, and are kept. It does not read ahead, does not decide how many
    requests may run at once, and does not interpret failures -- whatever `fetch` raises is
    raised as it came, because only `fetch` knows what went wrong. Those are properties of
    the transport, so they are composed around it before it gets here:

        tr = throttle_reads(my_fetch, max_parallel=8, is_rate_limited=my_classifier)
        tr = prefetch_whole_file(tr, size=my_size, cache_dir=None)   # optional
        webtorch.set_io_read(make_cached_reader(tr, size=my_size))

    which is exactly what `hf_read` and `modelscope_read` do. The cache is also reachable on
    its own through `read_cache` / `write_cache`, which is how a transport that wants to fill
    it in the background does so.

      fetch(key, offset, length) -> bytes   (async): read a byte range for `key`
          (length None = whole file). `webtorch.http_get` is a ready-made HTTP one.
      size(key) -> int | None               (async, optional): total length of `key`. Without
          it the length is learnt from the first short read.
      key(name) -> str                       (optional): map the incoming `name` to a stable
          cache key (default: identity). The hub readers map "org/repo/path" -> the file URL,
          so one platform's cache never collides with another's.
      cache / cache_dir / chunk_mb / persist: where and how the entries are stored.
    """
    keyfn = key or (lambda n: n)
    cdir = (cache_dir or _default_hub_cache()) if cache else None
    layer = _CacheLayer(fetch, size, cdir, chunk=chunk_mb << 20, persist=persist)
    async def read(name, offset=0, length=None):
        return await layer.read(keyfn(name), offset, length)
    return read


# ============================ cache management ============================
# The persistent cache would otherwise grow write-only. These inspect / read / write / delete
# it, keyed by cache dir (default `default_cache_dir()`). Every entry's key is the URL without
# scheme, so its FIRST path segment is the HOST/domain — HF ("huggingface.co/…") and ModelScope
# ("modelscope.cn/…") entries never mix and can be listed/cleared per host. All are async so the
# browser's IndexedDB-backed cache is loaded first (and flushed after any change).

def default_cache_dir():
    """The cache directory the hub readers / `make_cached_reader` use by default
    ($WEBTORCH_CACHE or ~/.cache/webtorch/hub)."""
    return _default_hub_cache()

def _cache_host(key):
    return key.replace("\\", "/").split("/", 1)[0]

async def list_cache(cache_dir=None, host=None):
    """List cached entries. Returns, sorted by key, `[{"key", "host", "size", "complete",
    "path"}]`, where `size` is what is actually stored (a partly-downloaded entry reports its
    real extent, not the file's full length). `host` (e.g. "huggingface.co") filters to one
    domain. `key` is the URL without scheme; pass it to `read_cache`/`delete_cache`."""
    import os
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    out = []
    for key in await store.keys():
        h = _cache_host(key)
        if host is not None and h != host:
            continue
        m = await store.meta(key)
        out.append({"key": key, "host": h, "size": await store.stored(key),
                    "complete": m["complete"], "total": m["size"],
                    "path": os.path.join(root, _url_key(key))})
    out.sort(key=lambda e: e["key"])
    return out


async def cache_hosts(cache_dir=None):
    """Per-domain summary of the cache: `[{"host", "files", "size"}]`, so entries from
    different hubs stay distinguishable."""
    agg = {}
    for e in await list_cache(cache_dir):
        a = agg.setdefault(e["host"], {"host": e["host"], "files": 0, "size": 0})
        a["files"] += 1; a["size"] += e["size"]
    return sorted(agg.values(), key=lambda a: a["host"])


async def cache_size(cache_dir=None, host=None):
    """Total bytes currently held (optionally for one host)."""
    return sum(e["size"] for e in await list_cache(cache_dir, host))


async def read_cache(key, offset=0, length=None, cache_dir=None):
    """Read bytes from the cache. Returns None on a miss -- nothing else happens.

    This is local storage and nothing else. It never fetches, never falls back, and has no
    idea where the data would otherwise come from; on a miss the caller decides what to do,
    which is usually to read it from wherever it lives and hand it to `write_cache`. Only
    the chunks the range touches are read, so this stays cheap on a multi-GB entry. None
    also means a hole: a range that crosses a gap in a partly-filled entry is a miss.
    """
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    m = await store.meta(key)
    have = await store.have(key)
    if not have:
        return None
    chunk = m["chunk"]
    total = m["size"] if m["size"] is not None else (max(have) + 1) * chunk
    end = total if length is None else min(offset + length, total)
    if end <= offset:
        return b""
    out = bytearray(end - offset)
    pos = 0
    for i in range(offset // chunk, (end - 1) // chunk + 1):
        base = i * chunk
        b = await store.get(key, i)
        lo = max(offset, base) - base
        hi = min(end, base + chunk) - base
        if b is None:                                  # hole in a partial entry
            return None
        out[pos:pos + (hi - lo)] = b[lo:hi]
        pos += hi - lo
        del b
    return bytes(out)


async def write_cache(key, data, cache_dir=None, offset=None, complete=None, total=None):
    """Put bytes into the cache. A later `read_cache` (or a reader built on it) finds them.

    This is local storage and nothing else -- IndexedDB in a browser, files on a host. It
    fetches nothing and knows nothing about where `data` came from; that is the caller's
    business. Which is exactly how a transport fills it: read a range however you read
    ranges, then write it here, and the next read of that range never reaches you again.

    `offset=None` (the default) replaces the whole entry with `data`. Passing an `offset`
    writes just that span and leaves the rest, so a file can be filled chunk by chunk as it
    arrives -- give `total` on the first such write if the full length is known, so progress
    and completeness can be reported. `complete` sets the "whole file is here" flag; leaving
    it None updates it automatically once every chunk is present.
    """
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    data = bytes(data)

    if offset is None:                                 # replace the entry outright
        await store.delete(key)
        chunk = _CHUNK_DEFAULT
        await store.set_meta(key, size=len(data), chunk=chunk,
                             complete=True if complete is None else bool(complete))
        for i in range((len(data) + chunk - 1) // chunk):
            await store.put(key, i, data[i * chunk:(i + 1) * chunk])
        return

    meta = await store.meta(key)
    chunk = meta["chunk"] or _CHUNK_DEFAULT
    if meta["size"] is None and total is not None:
        await store.set_meta(key, size=int(total), chunk=chunk)
        meta = await store.meta(key)
    if offset % chunk or (len(data) % chunk and
                          (meta["size"] is None or offset + len(data) < meta["size"])):
        raise ValueError("a partial write must start on a %d-byte boundary and run to the "
                         "next one (or to the end of the file)" % chunk)
    for j in range(0, len(data), chunk):
        await store.put(key, (offset + j) // chunk, data[j:j + chunk])
    if complete is not None:
        await store.set_meta(key, complete=bool(complete))
    elif meta["size"] is not None:
        need = (meta["size"] + chunk - 1) // chunk
        if len(await store.have(key)) >= need:
            await store.set_meta(key, complete=True)


async def delete_cache(key, cache_dir=None):
    """Delete one cached entry -- every chunk plus its metadata. -> True if it existed.
    (A reader created earlier may still hold chunks it already read.)"""
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    return await store.delete(key)


async def clear_cache(cache_dir=None, host=None):
    """Delete all cached entries (or only one `host`/domain's). -> number removed."""
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    n = 0
    for e in await list_cache(cache_dir, host):
        if await store.delete(e["key"]):
            n += 1
    return n


# ---- model-hub readers: thin clients of make_cached_reader over the HTTP transport ----
def _hub_reader(to_url, token, cache, cache_dir, max_parallel, prefetch, chunk_mb, persist):
    hdr = {"Authorization": "Bearer " + token} if token else None
    async def fetch(url, offset, length): return await http_get(url, offset, length, hdr)
    async def size(url): return await http_size(url, hdr)
    def key(name):
        return name if name.startswith(("http://", "https://")) else to_url(*_split_repo(name))
    # Composed the way the layers actually stack: the HTTP transport carries its own
    # concurrency and rate-limit handling, and the cache is wrapped around the result.
    # Stacked the way the responsibilities actually sit: the HTTP transport carries its own
    # concurrency and rate-limit handling, optionally reads ahead into the cache, and the
    # cache is wrapped around the result and only ever asks for what it does not have.
    tr = throttle_reads(fetch, max_parallel, http_rate_limited)
    if prefetch:
        tr = prefetch_whole_file(tr, size=size, cache_dir=cache_dir, chunk_mb=chunk_mb)
    return make_cached_reader(tr, size=size, key=key, cache=cache, cache_dir=cache_dir,
                              chunk_mb=chunk_mb, persist=persist)

def hf_read(revision="main", endpoint="https://huggingface.co", token=None,
            cache=True, cache_dir=None, max_parallel=8, prefetch=True, chunk_mb=16, persist=True):
    """Return an `io_read`-shaped async callback that fetches files directly from the
    **Hugging Face Hub** (a client of `make_cached_reader` over the HTTP transport). Install it,
    then load by repo id:

        webtorch.set_io_read(webtorch.hf_read())        # + set_io_write(...) only if quantizing
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to `{endpoint}/{org}/{repo}/resolve/{revision}/{path}`.
    Caching (default on) prefetches each touched file in the background and serves later reads
    from the cache; it **persists by default** — in the browser each chunk is its own
    IndexedDB record, written as it arrives, so a reload keeps whatever had been downloaded
    and resumes from there; on the host it is a real directory.
    `cache=False` streams with no cache, `cache_dir` picks the namespace, `prefetch=False`
    disables read-ahead, `chunk_mb` sets the chunk (and so the storage) granularity.
    `token` is a Bearer header for gated repos.
    `max_parallel` is the **ceiling** on concurrent range reads; the live limit adapts to a
    hub's real capacity. An explicit rate-limit (429 or a rate-limit body) halves it; it cools
    down (30→60→120→180s, cap 3 min) only if pushed to 0 with nothing in flight, and climbs
    back on success — aborting only when a rate-limit persists past the last cooldown with
    nothing in flight. A non-rate error while other reads still succeed is treated as an
    undisclosed capacity signal (concurrency capped to the in-flight count, read retried); a
    non-rate error with nothing in flight is genuine and raises immediately. Reads only."""
    ep = endpoint.rstrip("/")
    return _hub_reader(lambda repo, path: "%s/%s/resolve/%s/%s" % (ep, repo, revision, path),
                       token, cache, cache_dir, max_parallel, prefetch, chunk_mb, persist)

def modelscope_read(revision="master", endpoint="https://modelscope.cn", token=None,
                    cache=True, cache_dir=None, max_parallel=8, prefetch=True, chunk_mb=16, persist=True):
    """Return an `io_read`-shaped async callback that fetches files directly from
    **ModelScope (魔搭)** — another client of `make_cached_reader`. Same shape and options as
    `hf_read` (incl. `persist=True` → IndexedDB-backed in the browser):

        webtorch.set_io_read(webtorch.modelscope_read())
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to
    `{endpoint}/models/{org}/{repo}/resolve/{revision}/{path}`; `revision` defaults to
    ModelScope's `master`.

    This `resolve` route is used rather than the `/api/v1/.../repo?FilePath=` one because it
    (and the CDN it redirects large files to) returns `Access-Control-Allow-Origin: *` and
    allows the `Range` header — so a browser page, which can only read a cross-origin file when
    the host opts in, can stream weights directly. Reads only."""
    ep = endpoint.rstrip("/")
    return _hub_reader(
        lambda repo, path: "%s/models/%s/resolve/%s/%s" % (ep, repo, revision, path),
        token, cache, cache_dir, max_parallel, prefetch, chunk_mb, persist)


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
