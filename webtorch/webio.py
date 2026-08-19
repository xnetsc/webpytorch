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
    cached = make_cached_reader(fetch, size=size, cache_dir=cache_dir, max_parallel=max_parallel,
                                prefetch=prefetch, chunk_mb=chunk_mb, persist=persist)
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
    """Cache root: $WEBTORCH_CACHE, else ~/.cache/webtorch/hub. On the host this persists
    across runs; in the browser it lives in Pyodide's FS (persist across reloads by mounting
    IDBFS/OPFS at this path, e.g. via `webtorch.models.webenv`)."""
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


# ---- browser persistence: back the cache dir with IndexedDB so it survives page reloads ----
# On the host the cache dir is already a real disk (persistent). In the browser Pyodide's
# default FS (MEMFS) is wiped on reload, so we mount IDBFS at the cache dir and syncfs it:
# load prior contents on first use, flush each newly-completed file back to IndexedDB.
_persist = {"mounted": set(), "loaded": set()}

def _browser_fs():
    try:
        from js import pyodide as _p          # exposed by the Pyodide worker bootstrap
        return _p.FS
    except Exception:
        return None                           # host / non-Pyodide -> real disk, already persistent

async def _fs_syncfs(FS, populate):
    from js import Promise
    from pyodide.ffi import create_proxy
    def executor(resolve, reject):
        def cb(err):
            (reject if err else resolve)(err if err else None)
        FS.syncfs(populate, create_proxy(cb))
    return await Promise.new(create_proxy(executor))

async def _persist_load(cache_dir):
    """Mount IDBFS at cache_dir and load prior contents (IndexedDB -> FS), once. No-op on host."""
    if not cache_dir or cache_dir in _persist["loaded"]:
        return
    FS = _browser_fs()
    if FS is None:
        _persist["loaded"].add(cache_dir); return
    cur = ""
    for part in cache_dir.strip("/").split("/"):
        cur += "/" + part
        try: FS.mkdir(cur)
        except Exception: pass                # already exists
    if cache_dir not in _persist["mounted"] and hasattr(FS.filesystems, "IDBFS"):
        try:
            FS.mount(FS.filesystems.IDBFS, {}, cache_dir); _persist["mounted"].add(cache_dir)
        except Exception: pass
    if cache_dir in _persist["mounted"]:
        try: await _fs_syncfs(FS, True)       # populate FS <- IndexedDB
        except Exception: pass
    _persist["loaded"].add(cache_dir)

async def _persist_flush(cache_dir):
    """Flush newly-written cache files FS -> IndexedDB. No-op on host / when not IDBFS-mounted."""
    FS = _browser_fs()
    if FS is None or cache_dir not in _persist["mounted"]:
        return
    try: await _fs_syncfs(FS, False)          # push FS -> IndexedDB
    except Exception: pass


class _CachedFile:
    """Sparse local cache of one remote file + a background whole-file prefetch. A range read
    is served from the cache when those bytes are present, else fetched on the spot (never
    waiting for the prefetch to reach it) and stored. Sparse-file writes and coverage updates
    are synchronous, so they are atomic with respect to the single-threaded event loop; only
    the network fetches await."""
    def __init__(self, cache, key):
        import os
        self.c = cache; self.key = key
        self.path = os.path.join(cache.cache_dir, _url_key(key))
        self.marker = self.path + ".complete"
        self.covered = []                                    # merged sorted [start,end) present on disk
        self.size = None
        self.complete = os.path.exists(self.marker)
        self._prefetching = False

    def _has(self, a, b):
        for s, e in self.covered:
            if s <= a and b <= e: return True
            if s >= b: break
        return False

    def _mark(self, a, b):
        self.covered.append((a, b)); self.covered.sort()
        out = []
        for s, e in self.covered:
            if out and s <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], e))
            else: out.append((s, e))
        self.covered = out

    def _read_local(self, offset, length):
        with open(self.path, "rb") as f:
            if offset: f.seek(offset)
            return f.read() if length is None else f.read(length)

    def _store(self, offset, data):
        import os
        d = os.path.dirname(self.path)
        if d:
            try: os.makedirs(d, exist_ok=True)
            except Exception: pass
        if not os.path.exists(self.path):
            open(self.path, "wb").close()
        with open(self.path, "r+b") as f:
            f.seek(offset); f.write(data)
        self._mark(offset, offset + len(data))
        if self.size is not None and not self.complete and self._has(0, self.size):
            try: open(self.marker, "w").close()
            except Exception: pass
            self.complete = True
            return True                                       # just completed -> caller persists
        return False

    def _ensure_prefetch(self):
        if self._prefetching or self.complete or not self.c.prefetch or self.size is None:
            return
        self._prefetching = True
        import asyncio
        self.c.tasks.append(asyncio.ensure_future(self._prefetch()))

    async def _prefetch(self):
        try:
            o = 0
            while self.size is not None and o < self.size and not self.complete:
                b = min(self.size, o + self.c.chunk)
                if not self._has(o, b):
                    if self._store(o, await self.c._net(self.key, o, b - o)):
                        await self.c._flush()                 # persist a completed file
                o = b
        except Exception:
            pass

    async def read(self, offset, length):
        if self.complete:
            return self._read_local(offset, length)
        if self.size is None:
            self.size = await self.c._size(self.key)
        end = self.size if length is None else offset + length
        if self.size is not None and end is not None:
            end = min(end, self.size)
        if end is not None and self._has(offset, end):        # cache hit
            return self._read_local(offset, length)
        partial = length is not None and not (offset == 0 and (self.size is None or length >= self.size))
        if partial:
            self._ensure_prefetch()                           # read-ahead the rest, in background
        data = await self.c._net(self.key, offset, length)    # miss -> fetch just this range now
        if self._store(offset, data):
            await self.c._flush()                             # persist a completed file
        return data


# Escalating cooldowns (seconds) applied each time concurrency is forced to 0 by repeated
# rate-limiting; capped at 3 minutes. A rate-limit after the last one aborts the read.
_COOLDOWNS = [30, 60, 120, 180]

class _AdaptiveLimiter:
    """A concurrency gate whose live limit adapts to a hub's real capacity. `ceiling` is the
    max parallel network reads; the live `limit` self-tunes:

    - **Success** additively climbs `limit` back toward `ceiling` (and reopens it from 0).
    - **Explicit rate-limit** (HTTP 429, or a body with rate-limit wording) **halves** `limit`.
    - **A non-rate-limit error while other reads are still in flight** is treated as an
      *undisclosed* capacity/rate signal — some servers just error instead of returning 429 —
      so the concurrency is capped to the number still succeeding (the sweet spot) and the read
      is retried, NOT raised.
    - **A non-rate-limit error with no read in flight** is a genuine failure: it is not retried
      and propagates immediately with the server's message.
    - When (and only when) an explicit rate-limit drives `limit` to 0 **with nothing in
      flight**, it cools down for an escalating interval (30→60→120→180s, cap 3 min) then
      reopens to 1; a rate-limit persisting past the last cooldown aborts. A cooldown/abort
      never fires while any read is still succeeding."""
    def __init__(self, ceiling):
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
                    raise RuntimeError("hub reads aborted: still rate-limited after cooling down to 3 min")
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
            is_rate = isinstance(e, HttpError) and _is_rate_limited(e.status, e.body)
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
                    raise RuntimeError("hub reads aborted after repeated rate-limiting: %s" % e) from e
                continue                                        # retry (respects reduced limit / cooldown)
            await self._release_ok()
            return r


class _CacheLayer:
    """Owns a cache dir, the adaptive read queue, and one `_CachedFile` per key. GENERIC — it
    fetches bytes for a key via the injected `fetch(key, offset, length)` and (optionally)
    learns a key's total size via `size(key)`. It knows nothing about HTTP or model hubs."""
    def __init__(self, fetch, size, cache_dir, max_parallel=8, prefetch=True, chunk=16 << 20, persist=True):
        self.fetch = fetch; self.size_fn = size; self.cache_dir = cache_dir
        self.max_parallel = max(1, int(max_parallel)); self.prefetch = prefetch
        self.chunk = int(chunk); self.persist = persist
        self.files = {}; self.tasks = []; self._loaded = False
        self.limiter = _AdaptiveLimiter(self.max_parallel)

    async def _net(self, key, offset, length):               # one adaptive, queued fetch
        return await self.limiter.run(lambda: self.fetch(key, offset, length))

    async def _size(self, key):
        if self.size_fn is None: return None
        try: return await self.size_fn(key)
        except Exception: return None

    async def _flush(self):
        if self.persist:
            await _persist_flush(self.cache_dir)

    async def read(self, key, offset, length):
        if not self.cache_dir:
            return await self._net(key, offset, length)       # cache disabled -> pure streaming
        if not self._loaded:                                  # first use: load prior cache (browser: IDBFS)
            self._loaded = True
            if self.persist:
                await _persist_load(self.cache_dir)
        st = self.files.get(key)
        if st is None:
            st = self.files[key] = _CachedFile(self, key)
        return await st.read(offset, length)


# ============================ generic cached-reader tool ============================
# `make_cached_reader` wraps ANY async transport with the cache + read-ahead + adaptive
# concurrency + (browser) persistence used by the hub readers. Use it when you implement your
# own `set_io_read` callback over a custom source (S3, a signed CDN, your own server, …); the
# built-in `hf_read` / `modelscope_read` are just clients of it (see below).

async def http_get(url, offset=0, length=None, headers=None):
    """The built-in HTTP range transport (browser `fetch` / host `urllib`). Raises `HttpError`
    on a non-2xx status. A ready-made `fetch` building block for `make_cached_reader`."""
    rng = ("bytes=%d-%d" % (offset, offset + length - 1)) if length is not None else None
    return await _fetch_once(url, rng, headers)

async def http_size(url, headers=None):
    """Total byte length of an http(s) file (HEAD, else a `bytes=0-0` GET's Content-Range).
    A ready-made `size` building block for `make_cached_reader` (drives prefetch)."""
    return await _remote_size(url, headers)

def make_cached_reader(fetch, size=None, key=None, cache=True, cache_dir=None,
                       max_parallel=8, prefetch=True, chunk_mb=16, persist=True):
    """Wrap an async transport with caching + background read-ahead + adaptive concurrency +
    (browser IndexedDB) persistence, returning an `io_read`-shaped callback
    `read(name, offset=0, length=None) -> bytes` you can pass to `webtorch.set_io_read`.

      fetch(key, offset, length) -> bytes   (async): read a byte range for `key` (length None =
          whole file). Raise `webtorch.HttpError(status, body)` for a 429 / rate-limit response
          so the adaptive limiter can back off; any OTHER exception is a generic error (fatal
          only when no other read is in flight). Use `webtorch.http_get` for HTTP sources.
      size(key) -> int | None               (async, optional): total length of `key`, used to
          drive read-ahead. Return None (or omit) to disable prefetch for that key. `http_size`
          works for HTTP sources.
      key(name) -> str                       (optional): map the incoming `name` to a stable
          cache/fetch key (default: identity). Hub readers map "org/repo/path" -> the file URL
          here, so the cache is keyed uniquely per platform.
      cache / cache_dir / max_parallel / prefetch / chunk_mb / persist: same as `hf_read`
          (see its docstring / the API reference for the full adaptive-concurrency behavior)."""
    keyfn = key or (lambda n: n)
    cdir = (cache_dir or _default_hub_cache()) if cache else None
    layer = _CacheLayer(fetch, size, cdir, max_parallel=max_parallel, prefetch=prefetch,
                        chunk=chunk_mb << 20, persist=persist)
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
    """List cached entries (loads the browser IndexedDB cache first). Returns, sorted by key,
    `[{"key", "host", "size", "complete", "path"}]`. `host` (e.g. "huggingface.co") filters to
    one domain. `key` is the URL without scheme; pass it to `read_cache/delete_cache`."""
    import os
    root = cache_dir or _default_hub_cache()
    await _persist_load(root)
    out = []
    if os.path.isdir(root):
        for dp, _dn, fns in os.walk(root):
            for f in fns:
                if f.endswith(".complete") or f.endswith(".part"):
                    continue
                ap = os.path.join(dp, f); key = os.path.relpath(ap, root).replace("\\", "/")
                h = _cache_host(key)
                if host is not None and h != host:
                    continue
                try: sz = os.path.getsize(ap)
                except OSError: sz = 0
                out.append({"key": key, "host": h, "size": sz,
                            "complete": os.path.exists(ap + ".complete"), "path": ap})
    out.sort(key=lambda e: e["key"])
    return out

async def cache_hosts(cache_dir=None):
    """Per-domain summary so HF vs ModelScope (etc.) are separated:
    `[{"host", "files", "size"}]`, largest first."""
    agg = {}
    for e in await list_cache(cache_dir):
        a = agg.setdefault(e["host"], {"host": e["host"], "files": 0, "size": 0})
        a["files"] += 1; a["size"] += e["size"]
    return sorted(agg.values(), key=lambda a: -a["size"])

async def cache_size(cache_dir=None, host=None):
    """Total bytes cached (optionally for one `host`)."""
    return sum(e["size"] for e in await list_cache(cache_dir, host))

async def read_cache(key, offset=0, length=None, cache_dir=None):
    """Read bytes from a cached entry by its `key` (a `list_cache` key or a full URL). Returns
    None if that entry is not cached."""
    import os
    root = cache_dir or _default_hub_cache(); await _persist_load(root)
    p = os.path.join(root, _url_key(key))
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        if offset: f.seek(offset)
        return f.read() if length is None else f.read(length)

async def write_cache(key, data, cache_dir=None, complete=True):
    """Write/replace a cache entry's bytes (pre-seed the cache). `complete=True` marks it fully
    cached (so a reader serves it from disk). Persists to IndexedDB in the browser."""
    import os
    root = cache_dir or _default_hub_cache(); await _persist_load(root)
    p = os.path.join(root, _url_key(key)); d = os.path.dirname(p)
    if d:
        try: os.makedirs(d, exist_ok=True)
        except Exception: pass
    with open(p, "wb") as f:
        f.write(bytes(data))
    marker = p + ".complete"
    if complete:
        try: open(marker, "w").close()
        except Exception: pass
    else:
        try: os.remove(marker)
        except OSError: pass
    await _persist_flush(root)

async def delete_cache(key, cache_dir=None):
    """Delete one cached entry (its data + `.complete` + `.part`). Persists. -> True if it
    existed. (Applies on disk; a live reader created earlier may still hold it in memory.)"""
    import os
    root = cache_dir or _default_hub_cache(); await _persist_load(root)
    p = os.path.join(root, _url_key(key)); existed = os.path.exists(p)
    for q in (p, p + ".complete", p + ".part"):
        try: os.remove(q)
        except OSError: pass
    await _persist_flush(root)
    return existed

async def clear_cache(cache_dir=None, host=None):
    """Delete all cached entries (or only one `host`/domain's). Persists. -> # data files
    removed."""
    import os
    root = cache_dir or _default_hub_cache()
    entries = await list_cache(cache_dir, host)          # loads persistence, host-filtered
    n = 0
    for e in entries:
        for q in (e["path"], e["path"] + ".complete", e["path"] + ".part"):
            try: os.remove(q)
            except OSError: pass
        n += 1
    if os.path.isdir(root):                              # prune now-empty subdirs
        for dp, _dn, _fn in os.walk(root, topdown=False):
            if dp != root:
                try: os.rmdir(dp)
                except OSError: pass
    await _persist_flush(root)
    return n


# ---- model-hub readers: thin clients of make_cached_reader over the HTTP transport ----
def _hub_reader(to_url, token, cache, cache_dir, max_parallel, prefetch, chunk_mb, persist):
    hdr = {"Authorization": "Bearer " + token} if token else None
    async def fetch(url, offset, length): return await http_get(url, offset, length, hdr)
    async def size(url): return await http_size(url, hdr)
    def key(name):
        return name if name.startswith(("http://", "https://")) else to_url(*_split_repo(name))
    return make_cached_reader(fetch, size=size, key=key, cache=cache, cache_dir=cache_dir,
                              max_parallel=max_parallel, prefetch=prefetch, chunk_mb=chunk_mb, persist=persist)

def hf_read(revision="main", endpoint="https://huggingface.co", token=None,
            cache=True, cache_dir=None, max_parallel=8, prefetch=True, chunk_mb=16, persist=True):
    """Return an `io_read`-shaped async callback that fetches files directly from the
    **Hugging Face Hub** (a client of `make_cached_reader` over the HTTP transport). Install it,
    then load by repo id:

        webtorch.set_io_read(webtorch.hf_read())        # + set_io_write(...) only if quantizing
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to `{endpoint}/{org}/{repo}/resolve/{revision}/{path}`.
    Caching (default on) prefetches each touched file in the background and serves later reads
    from disk; it **persists by default** (`persist=True`) — in the browser the cache dir is
    backed by IndexedDB (IDBFS) so it survives page reloads, on the host it is a real dir.
    `cache=False` streams with no cache, `cache_dir` picks the location, `prefetch=False`
    disables read-ahead, `chunk_mb` is the prefetch chunk size, `persist=False` keeps an
    in-session-only (browser MEMFS) cache. `token` is a Bearer header for gated repos.
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
    `{endpoint}/api/v1/models/{org}/{repo}/repo?Revision={revision}&FilePath={path}`;
    `revision` defaults to ModelScope's `master`. Reads only."""
    ep = endpoint.rstrip("/")
    return _hub_reader(
        lambda repo, path: "%s/api/v1/models/%s/repo?Revision=%s&FilePath=%s" % (ep, repo, revision, path),
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
