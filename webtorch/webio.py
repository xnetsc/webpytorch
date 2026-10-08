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

# BF16 is the top half of a float32, so widening it goes through a uint32 array, a shifted
# copy, and a float32 view -- several times the source in temporaries, all live at once. A
# 64MB tensor peaks over 300MB that way, which 32-bit WASM refuses outright; it is what
# stopped a 3B safetensors model from loading on a machine with room to spare for it.
# Converting in bounded slices makes the peak independent of the tensor's size.
_CONV_SLICE = 1 << 22                                  # elements per step (~32MB of peak)


def bf16_to_f32(raw):
    """BF16 bytes -> float32, without a large intermediate."""
    src = np.frombuffer(raw, np.uint16)
    out = np.empty(src.size, np.float32)
    for i in range(0, src.size, _CONV_SLICE):
        j = min(src.size, i + _CONV_SLICE)
        out[i:j] = (src[i:j].astype(np.uint32) << 16).view(np.float32)
    return out


def to_f32(raw, dt):
    """Raw safetensors bytes of dtype `dt` -> a flat float32 array, bounded peak."""
    if dt == "BF16":
        return bf16_to_f32(raw)
    src = np.frombuffer(raw, _ST_DT.get(dt, np.float32))
    if src.dtype == np.float32:
        return src
    out = np.empty(src.size, np.float32)
    for i in range(0, src.size, _CONV_SLICE):
        j = min(src.size, i + _CONV_SLICE)
        out[i:j] = src[i:j].astype(np.float32)
    return out


def _st_decode(raw, dt, shape):
    return to_f32(raw, dt).reshape(shape)


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

# Cooperative cancellation: Python here cannot be preempted from outside, so an in-flight
# load checks this flag at every IO checkpoint (io_read/io_write, and every served chunk)
# and raises once someone asked it to stop. The flag is STICKY on purpose: once set it stays
# set, and neither a checkpoint nor the transport's abort-conversion clears it. That is what
# makes a stop reliable — a stop also aborts background read-ahead fetches, and if whichever
# of them died first could clear the flag, a background task could swallow a stop meant for
# the load and the load would run on. So the flag stays up and keeps raising until the load's
# OWN error path runs and the next load clears it on the way in (cancel(False)). Between a
# stop and the next load, no read/write callback fires at all — which is exactly "stopped".
# `n` counts stop requests: the built-in transport stamps each fetch with the current count,
# so a fetch in flight when a stop fires reports Cancelled by generation, independent of
# the flag. `_INFLIGHT` holds those fetches' abort controllers, which is how a stop
# interrupts them at once instead of waiting for each to finish on its own.
_CANCEL = {"on": False, "n": 0, "probe": None}
# Keys written since the last load began. A stop has to be able to tidy up after
# itself, and only this layer knows which cache entries a load touched -- the caller
# asked for a model, not for a set of URLs.
_WROTE = set()
_INFLIGHT = {}             # fetch id -> (controller, stop-count): the id is the key because
                           # the controller is a JS proxy — unhashable, and equality there is
                           # not ours to rely on
_next_fetch_id = 0


class Cancelled(BaseException):
    """Raised from inside a load/read after webtorch.cancel() asked it to stop.

    Derives from BaseException the same way asyncio.CancelledError does: loaders are full
    of broad `except Exception` fallbacks (optional files, retry loops, rate-limit gates),
    and none of them may swallow a stop — it must always reach the load's own error path."""


def cancel_requested():
    """Has a stop been asked for? True until `cancel(False)` withdraws it.

    Reading the flag rather than raising on it is what a decode loop needs: a generation is
    not IO, has no checkpoint to raise at, and should hand back the tokens it already has
    instead of losing them to an exception."""
    if _CANCEL["on"]:
        return True
    probe = _CANCEL["probe"]
    if probe is not None:
        try:
            if probe():
                # Go through `cancel` rather than setting the flag: a stop is not only a flag,
                # it also aborts the reads already in flight. Without that a load waits for
                # its current chunk to finish before it can notice -- measured at 987ms, where
                # aborting the fetch ends it at once.
                cancel(True)
                return True
        except Exception:
            pass                              # a probe that breaks must not break the run
    return False


def set_cancel_probe(fn):
    """Install `fn()` as a second source of "stop was asked for". `None` clears it.

    A stop has to be able to arrive while the interpreter is BUSY. In the browser the
    decode loop is a plain Python loop inside one `runPythonAsync` call, so the worker
    thread is inside that call for the whole generation: a stop sent as a message is not
    queued behind the work, it is not received at all until the work ends, and the button
    reads as dead. The flag has to come from somewhere the caller can write without the
    interpreter's help -- a `SharedArrayBuffer` the page stores into directly -- and this is
    where the SDK reads it. Called between tokens and at every IO checkpoint, so it must be
    cheap; it is a single index into shared memory."""
    _CANCEL["probe"] = fn


def cancel(flag=True):
    """Ask the in-flight load/read/generation to stop. `cancel(False)` withdraws the request.

    The stop lands at the next IO checkpoint (io_read/io_write, or the next served chunk),
    where it raises `Cancelled`; after that the SDK issues no more read/write callbacks. For
    the built-in HTTP transport a stop also aborts every fetch in flight right away, so it
    does not wait for a slow request to finish on its own — aborting a fetch someone's OWN
    read callback issued is that callback's business, not the SDK's. The flag is sticky: it
    stays set — keeping any stray IO from running — until `cancel(False)` clears it, which
    every load does on the way in.

    A generation stops the same way, but gracefully: the decode loop checks the flag between
    tokens (see `cancel_requested`) and returns the reply as far as it got, because half an
    answer is worth keeping and a half-finished load is not."""
    _CANCEL["on"] = bool(flag)
    if not flag:
        _WROTE.clear()                    # a new load: what the last one wrote is its own
    if flag:
        _CANCEL["n"] += 1
        for _fid, (ctl, _gen) in list(_INFLIGHT.items()):
            try: ctl.abort()
            except Exception: pass


def _check_cancel():
    # Through `cancel_requested`, so an out-of-band stop (see `set_cancel_probe`) reaches the
    # IO checkpoints too, not only the decode loop.
    if cancel_requested():
        raise Cancelled("load cancelled")

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
    _check_cancel()
    fn = io or _IO
    if fn is None:
        raise RuntimeError(_UNSET_READ)
    return bytes(await fn(name, offset, length))


# The container formats a source can be served in, by the extension each one is published
# under. Everything that is published is published under these names.
_CONTAINERS = (".gguf", ".onnx", ".safetensors")


def container_of(name):
    """Which container `name` is — `"gguf"`, `"onnx"`, `"safetensors"`, or `""` for a
    directory, a repo id, or a single file in some other shape.

    A *format* is a file naming convention and reading it off the name is honest. What this
    is NOT for is deciding what **model** is inside: a GGUF may hold a decoder, a decision
    model or something nobody has published yet, and only its contents say which. So this
    answers "how are these bytes packed" and nothing else, and it lives in one place so that
    the answer cannot drift between the loaders that need it.
    """
    low = str(name).rstrip("/").lower()
    for ext in _CONTAINERS:
        if low.endswith(ext):
            return ext[1:]
    return ""


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
    _check_cancel()
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


async def _pyfetch(url, **kw):
    """pyfetch wired to webtorch.cancel(): the request registers its AbortController, so a
    stop interrupts it at once instead of letting it run to completion. An interrupt is
    reported as Cancelled — decided by the stop counter, not the flag, so it stays right
    whichever of several aborted fetches surfaces first. It does NOT clear the stop flag:
    only the load's own checkpoint (or the next load, clearing it on the way in) accounts
    for the stop — a background fetch dying here must not swallow a stop meant for the load.
    Lives here (not in one caller) so EVERY fetch the SDK issues — data reads and size
    probes alike — stops the same way."""
    from pyodide.http import pyfetch
    import js
    global _next_fetch_id
    ctl = js.AbortController.new()
    _next_fetch_id += 1
    fid = _next_fetch_id
    gen = _CANCEL["n"]
    _INFLIGHT[fid] = (ctl, gen)
    try:
        return await pyfetch(url, signal=ctl.signal, **kw)
    except Exception:
        if _CANCEL["n"] != gen:
            raise Cancelled("load cancelled") from None
        raise
    finally:
        _INFLIGHT.pop(fid, None)


async def _read_streaming(r, url):
    """The response body, reported as it arrives rather than when it is complete.

    `r.bytes()` waits for the whole chunk, and a chunk here is 16 MiB. On a slow link that is
    one progress update every sixteen seconds -- during which a download that is working
    looks exactly like one that has hung, which is what it was taken for. Reading the stream
    reports every piece the network hands over, so the meter moves at the speed the bytes do.

    Falls back to `r.bytes()` wherever the stream is not reachable (a host runtime, an older
    response object), because a progress meter is not worth failing a read over.
    """
    try:
        body = r.js_response.body
        reader = body.getReader()
    except Exception:
        return bytes(await r.bytes())
    # What the response says it will deliver, so a body that ends early can be told from one
    # that ended. Reading a stream makes that failure reachable in a way `r.bytes()` did not:
    # a connection dropped mid-chunk closes the stream cleanly, and the bytes gathered so far
    # would otherwise be returned as a whole chunk and cached as one -- a silently truncated
    # model, which is the worst outcome available here.
    want = None
    try:
        headers = r.js_response.headers
        # Fetch exposes the DECODED response body, while Content-Length describes the
        # compressed wire representation. Content-Encoding is not necessarily CORS-exposed,
        # so a 200 response cannot safely be checked against Content-Length at all. A 206's
        # Content-Range is the requested representation and is the exact check needed for
        # the large model chunks this transport issues.
        status = int(getattr(r, "status", 200) or 200)
        if status == 206:
            cr = headers.get("content-range")
            if cr and "/" in cr and "-" in cr:
                span = cr.split(" ")[-1].split("/")[0]
                a, b = span.split("-")
                want = int(b) - int(a) + 1
    except Exception:
        want = None
    parts, total = [], 0
    try:
        while True:
            res = await reader.read()
            if res.done:
                break
            buf = res.value.to_py().tobytes()
            parts.append(buf)
            total += len(buf)
            _note_download(url, len(buf))     # the rate line, at the rate it is happening
    except Exception:
        # An abort lands here. Nothing partial escapes: the accumulated pieces go with the
        # stack, and the caller sees the failure rather than a short read.
        raise
    finally:
        try: reader.releaseLock()
        except Exception: pass
    if want is not None and total != want:
        raise HttpError(0, "truncated response: got %d of %d bytes from %s"
                           % (total, want, url))
    return b"".join(parts)


async def _fetch_once(url, rng, headers):
    _check_cancel()      # a retry loop's next attempt is a checkpoint too, not just the first
    try:
        from pyodide.http import pyfetch                     # browser (Pyodide)
    except ImportError:
        pyfetch = None
    if pyfetch is not None:
        gen = _CANCEL["n"]
        h = dict(headers or {})
        if rng: h["Range"] = rng
        try:
            r = await _pyfetch(url, headers=h)
            data = await _read_streaming(r, url)
        except Exception:
            # Aborted because a stop fired while it ran -> the stop, not a transport fault.
            # Decided by the stop counter (not the flag), so it stays right whichever of
            # several aborted fetches surfaces first. The flag is deliberately left set: a
            # load's own checkpoint accounts for the stop (or the next load clears it on the
            # way in), so a fetch dying here can never swallow a stop meant for the load.
            if _CANCEL["n"] != gen:
                raise Cancelled("load cancelled") from None
            raise
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
    rng = (("bytes=%d-%d" % (offset, offset + length - 1)) if length is not None
           else (("bytes=%d-" % offset) if offset else None))
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

def use_default_io(cache=True, cache_dir=None, max_parallel=16, prefetch=True, chunk_mb=16, persist=True):
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
    cdir = cache_dir or _default_hub_cache()
    known = {}

    async def size(name):
        try: return await http_size(name)
        except Exception: return None

    async def raw(name, offset, length):                     # browser fetch / host urllib
        return await http_get(name, offset, length)

    get = throttle_reads(raw, max_parallel, http_rate_limited)
    if prefetch:
        get = prefetch_whole_file(get, size=size, cache_dir=cdir, chunk_mb=chunk_mb)

    async def read(name, offset=0, length=None):
        if not (_is_url(name) or _in_browser()):
            return await default_io_read(name, offset, length)   # a local file: read it
        n = str(name)
        # Full name first, then basename: a file imported under its own identity
        # ("1a2b….gguf", "mydir/config.json") matches exactly, and a bare "model.gguf"
        # still matches the plain-basename registration.
        h = _local_files.get(n) or _local_files.get(n.rsplit("/", 1)[-1])
        if h is not None:                                    # a file the person pointed at
            data = await _read_local_file(h, offset, length)
            _report(name, len(data), None, offset)
            return data
        if _local_miss(n):
            raise FileNotFoundError("%s is not in the folder that was picked" % n)
        # Ask the cache; if it has not got them, that is this callback's problem to solve.
        hit = await read_cache(name, offset, length, cdir)
        if hit is not None and (length is None or len(hit) >= length):
            _report(name, len(hit), known.get(name), offset)
            return hit
        got = len(hit) if hit else 0
        if await await_inflight(name, offset + got, chunk_mb << 20):
            again = await read_cache(name, offset, length, cdir)
            if again is not None and (length is None or len(again) >= length):
                _report(name, len(again), known.get(name), offset)
                return again
            hit = again if again is not None else hit
            got = len(hit) if hit else 0
        if name not in known:
            known[name] = await size(name)
        total = known.get(name)
        want = None if length is None else length - got
        data = await get(name, offset + got, want)
        # See the same inference in the hub reader below: a short answer ends the file, and
        # so does a read that asked for no length -- what came back is the rest of it.
        if want is None or (total is None and len(data) < want):
            total = offset + got + len(data)
            known[name] = total
        await write_cache(name, data, cdir, offset=offset + got, total=total, chunk_mb=chunk_mb)
        out = (hit or b"") + data
        _report(name, len(out), total, offset)
        return out

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
        # Only a successful answer says how long the FILE is. An error page has a length
        # too -- its own: modelscope.cn answers a HEAD for a repo it does not carry with 404
        # and Content-Length 145, and taking that as the size made the reader reject the
        # .ai copy as "a different file" and fail the load on the .cn 404. (urllib below
        # raises on an error status, so the host path never had this.)
        try:
            r = await _pyfetch(url, method="HEAD", headers=dict(headers or {}))
            cl = r.headers.get("content-length") or r.headers.get("Content-Length")
            if r.ok and cl and str(cl).isdigit(): return int(cl)
        except Exception: pass
        try:
            h = dict(headers or {}); h["Range"] = "bytes=0-0"
            r = await _pyfetch(url, headers=h)
            cr = r.headers.get("content-range") or r.headers.get("Content-Range")
            if r.status == 206 and cr and "/" in cr and cr.rsplit("/", 1)[-1].isdigit():
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
_idb_warned = []            # warn once, not once per chunk
# Below this, too little time has passed to quote a rate; report none rather than a number
# that is an artefact of dividing by nearly zero.
_RATE_MIN_S = 0.05


def _merge(spans, lo, hi):
    """Add [lo, hi) to a sorted, disjoint span list, coalescing anything it meets.

    Writes arrive in whatever pieces the transport read, so 0-16 followed by 16-32 has to
    become 0-32 rather than two records -- otherwise a file written in many small pieces
    accumulates a span per piece. Once every byte is present the list is a single span,
    which is also how "complete" is recognised.
    """
    if hi <= lo:
        return list(spans)
    out = []
    placed = False
    for a, b in sorted(list(spans) + [[lo, hi]]):
        if out and a <= out[-1][1]:                    # touching or overlapping -> one span
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _covered_from(spans, offset):
    """How far a contiguous run starting at `offset` reaches, or None if it does not start."""
    for a, b in spans:
        if a <= offset < b:
            return b
    return None


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

    async def delete_chunk(self, key, i):
        """Remove one chunk's bytes. Default: nothing, because not every backend can.

        A disk entry is ONE sparse file, so there is no record to remove from the middle of
        it -- the coverage map is what says which bytes count, and trimming that is the
        whole operation there. A store that keeps a record per chunk overrides this and
        reclaims the space.

        What this must never be is a write of empty bytes. That leaves a record which reads
        back as null: `get` reports the chunk missing, `stored` trips over it, and an entry
        that was complete becomes one that cannot be exported or resumed."""
        return False

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
        m.setdefault("covered", [])            # byte spans actually present, merged
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
        return {"size": m["size"], "chunk": m["chunk"], "complete": m["complete"],
                "covered": [list(x) for x in m["covered"]]}

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
        if not m["chunk"]:
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
                "complete": bool(d.get("complete", False)),
                "covered": [list(x) for x in (d.get("covered") or [])]}

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
        recs = await _idb_req(self._store(self.CHUNKS, "readonly").getAll(rng))
        ks = await _idb_req(self._store(self.CHUNKS, "readonly").getAllKeys(rng))
        out = set()
        for n, k in enumerate(ks or []):
            # An empty record is not a stored chunk -- see `stored`. Counting one as present
            # is what would keep a damaged entry from refetching the bytes it is missing.
            v = (recs or [])[n] if recs is not None and n < len(recs) else None
            if v is None or int(getattr(v, "byteLength", 0) or 0) == 0:
                continue
            try: out.add(int(str(k).rsplit("#", 1)[1]))
            except (ValueError, IndexError): pass
        return out

    async def get(self, key, i):
        """One chunk, or None if it is not there -- or cannot be read right now.

        A read can fail on a value the store definitely holds: Chrome raises "Failed to
        read large IndexedDB value" when it cannot materialize one under memory pressure,
        which is exactly the situation loading a large model creates. That is a cache miss,
        not a load failure -- the caller's reader fetches the range instead -- so it must
        not propagate. Raising here failed a 27B load outright with the file fully cached.
        """
        await self.open()
        try:
            v = await _idb_req(self._store(self.CHUNKS, "readonly").get(self._ck(key, i)))
        except Exception as e:
            if not _idb_warned:
                _idb_warned.append(1)
                print("webtorch: IndexedDB read failed (%s); treating as a cache miss" % e)
            return None
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
        except Exception as e:
            # Out of room in the JS heap, not a type problem. A chunk is 16 MiB and the
            # allocation can simply fail ("RangeError: Array buffer allocation failed") while
            # a model is resident. Falling through to `to_js` hid that behind a
            # ConversionError -- a memory failure wearing a type failure's face, which is a
            # long way to look in the wrong direction.
            #
            # And it must not end the load. The cache is an OPTIMISATION: the bytes are in
            # hand, the model can be built from them, and the only thing lost is not having
            # to fetch them again. Treated like being over quota -- stop caching, keep
            # loading -- which is what the quota path next to this already does.
            if "allocation failed" in str(e) or isinstance(e, MemoryError):
                self.full = True
                _note_full(key)
                return False
            from pyodide.ffi import to_js
            try:
                buf = to_js(data)
            except Exception:
                self.full = True
                _note_full(key)
                return False
        try:
            await _idb_req(self._store(self.CHUNKS, "readwrite").put(buf, self._ck(key, i)))
        except Exception as e:
            if _is_quota_error(e):
                self.full = True               # stop trying; reads keep working, writes stream
                _note_full(key)
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

    async def delete_chunk(self, key, i):
        """Remove one chunk's record, reclaiming its space."""
        await self.open()
        try:
            await _idb_req(self._store(self.CHUNKS, "readwrite").delete(self._ck(key, i)))
            return True
        except Exception:
            return False

    async def stored(self, key):
        """Summed from the chunk records, so a partial entry reports what it really holds."""
        await self.open()
        from js import IDBKeyRange
        rng = IDBKeyRange.bound(key + "#", key + "#\uffff")
        vals = await _idb_req(self._store(self.CHUNKS, "readonly").getAll(rng))
        # A record with no bytes is not a chunk. One release wrote empty records to "drop" a
        # chunk, and every entry it touched then failed here rather than reporting what it
        # held; treating them as absent is what lets such an entry heal by refetching.
        return int(sum(int(v.byteLength) for v in (vals or []) if v is not None))


def _in_browser():
    try:
        import js                                   # noqa: F401
        from js import indexedDB                    # noqa: F401
        return True
    except Exception:
        return False


# Progress is reported on UNIQUE FILE POSITIONS SERVED, not bytes fetched. A model already
# in the cache does no network at all, and counting fetches leaves that load looking frozen.
# Counting raw reads is also wrong: retries and overlapping tensor ranges can make one file
# appear larger than it is. The covered-range union below measures actual load coverage.
_progress = {"cb": None, "state": {}}


def set_read_progress(cb):
    """Install `cb(info)`, called as reads are served -- how far a LOAD has got. `None` clears it.

    `info` is a dict, so it can gain fields without breaking callers:

        key       the entry being read
        done      unique byte positions served for it since the hook was installed
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


_stage = {"cb": None, "at": None}


def set_load_progress(cb):
    """Install `cb(info)`, called as a load moves between STAGES. `None` clears it.

    Reading the weights is not the whole of a load, and on a large model it is not even the
    slow part: what follows is measuring the kernel shapes this model will use, checking the
    weights against the file they came from, and one forward pass to prove the result is
    usable. Read progress cannot report any of that -- it counts bytes served, and by then
    nothing is being read -- so a 13 GB model sat at a finished progress bar for minutes,
    which reads as a load that has hung.

    `info` is a dict, so it can gain fields without breaking callers:

        stage   "reading" | "warming" | "checking" | "proving" | "ready"
        done    items finished in this stage, when the stage counts them
        total   items in this stage, when that is known ahead of time

    A stage without a count still reports, so a caller always knows what is happening even
    when it cannot say how far along. Stages are a fixed vocabulary rather than free text:
    a caller decides what to show, and should not have to parse a sentence to do it.
    """
    _stage["cb"] = cb
    _stage["at"] = None             # a fresh hook times a fresh load, not the last one


def get_load_progress():
    return _stage["cb"]


def load_stage(stage, done=None, total=None):
    """Report that a load has reached `stage`. Called by the loaders, not by callers."""
    import time
    now = time.monotonic()
    prev, t0 = _stage.get("at") or (None, now)
    _stage["at"] = (stage, now)
    cb = _stage["cb"]
    if cb is None:
        return
    info = {"stage": stage}
    # How long the stage that just ENDED took. Without this a slow load is an impression
    # rather than a number: every part of this phase measures fast in isolation on a settled
    # machine, and the one time it matters is right after several gigabytes have been
    # uploaded. Carrying the elapsed time means the next slow load says where it went
    # instead of being argued about.
    if prev is not None:
        info["after"] = prev
        info["elapsed"] = round(now - t0, 2)
    if done is not None:
        info["done"] = int(done)
    if total is not None:
        info["total"] = int(total)
    try:
        cb(info)
    except Exception:
        pass                        # a reporting hook must never take a load down with it


def _prog_state(key):
    st = _progress["state"].get(key)
    if st is None:
        import time
        now = time.monotonic()
        st = _progress["state"][key] = {"t0": now, "t": now, "done": 0, "mark": 0,
                                        "rate": 0.0, "ranges": []}
    return st


def _covered_bytes(st, offset, length, total):
    """Merge one served range and return the exact union size.

    A total learned after early reads also clamps the older ranges. That matters for servers
    which reveal their length only after an open-ended request reaches EOF.
    """
    start = max(0, int(offset))
    end = start + max(0, int(length))
    if total is not None:
        limit = max(0, int(total))
        start, end = min(start, limit), min(end, limit)
    ranges = list(st["ranges"])
    if end > start:
        ranges.append((start, end))
    ranges.sort()
    merged = []
    for a, z in ranges:
        if total is not None:
            a, z = min(a, limit), min(z, limit)
        if z <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], z))
        else:
            merged.append((a, z))
    st["ranges"] = merged
    return sum(z - a for a, z in merged)


def _report(key, delta, total, offset=None):
    """Add one served range for `key` and report unique file coverage.

    The total lives HERE, not in the reader, because the reader is installed once and stays
    installed while any number of loads come and go -- so a counter it owned would keep
    climbing. Loading a 13 GB file, releasing it and loading it again reported 26 GB. This
    state is cleared by `set_read_progress`, which is called at the start of every load.
    """
    _check_cancel()      # every served chunk is a stop checkpoint, progress hook or not
    cb = _progress["cb"]
    if cb is None:
        return
    import time
    st = _prog_state(key)
    if offset is None:
        # Compatibility for custom/internal callers that have no positional information.
        st["done"] += delta
        if total is not None:
            st["done"] = min(st["done"], total)
    else:
        st["done"] = _covered_bytes(st, offset, delta, total)
    done = st["done"]
    now = time.monotonic()
    dt = now - st["t"]
    if dt >= 0.2:                             # smooth over a window, not per read
        a = 0.4                               # recent enough to feel live, steady enough to read
        # Learning the real EOF can clamp an earlier open-ended range. Coverage cannot have
        # a negative rate merely because the estimate became more accurate.
        st["rate"] = a * (max(0, done - st["mark"]) / dt) + (1 - a) * st["rate"]
        st["t"] = now; st["mark"] = done
    el = now - st["t0"]
    # Until the window has produced a figure, the average since the first read is the honest
    # answer -- but only once enough time has passed to divide by. Clamping the elapsed time
    # to a microsecond instead reported 12583 GB/s on the first callback.
    rate = st["rate"] or ((done / el) if el >= _RATE_MIN_S else 0.0)
    try:
        cb({"key": key, "done": done, "total": total, "elapsed": el, "rate": rate})
    except Exception:
        pass                                  # a broken meter must not break a load


class _FsaStore(_Store):
    """A directory the person picked, through the File System Access API.

    The origin storage quota is a browser policy, not a property of the machine: the same
    page is given 4.46 GB in one browser and 11.5 GB in another while the disk has 382 GB
    free, and `persist()` is refused in both. A model larger than whatever that number
    happens to be simply cannot be kept -- which is why this exists. A directory handed
    over by the person carries no quota at all, so what fits is what fits on the disk.

    It is also a better fit than a chunk store: a real file takes writes at an offset, so
    the bytes go where they belong and the file IS the model -- openable by anything else,
    and exportable by copying it. Only the span bookkeeping lives beside it, in a sidecar.
    """

    def __init__(self, root, handle):
        self.root = root
        self.dir = handle                     # FileSystemDirectoryHandle, from the page
        self._h = {}

    async def open(self):
        pass

    def _name(self, key):
        return _url_key(key).replace("/", "_")

    async def _file(self, key, create=True):
        from pyodide.ffi import to_js
        from js import Object
        n = self._name(key)
        h = self._h.get(n)
        if h is None:
            fh = await self.dir.getFileHandle(n, to_js({"create": create},
                                                       dict_converter=Object.fromEntries))
            h = self._h[n] = await fh.createSyncAccessHandle()
        return h

    async def _meta_file(self, key, data=None):
        import json
        from pyodide.ffi import to_js
        from js import Object, Uint8Array
        n = self._name(key) + ".meta"
        fh = await self.dir.getFileHandle(n, to_js({"create": True},
                                                   dict_converter=Object.fromEntries))
        if data is None:
            f = await fh.getFile()
            txt = await f.text()
            try: return json.loads(txt) if txt else {}
            except Exception: return {}
        w = await fh.createWritable()
        await w.write(json.dumps(data))
        await w.close()
        return data

    async def meta(self, key):
        d = await self._meta_file(key)
        return {"size": d.get("size"), "chunk": d.get("chunk"),
                "complete": bool(d.get("complete", False)),
                "covered": [list(x) for x in (d.get("covered") or [])]}

    async def set_meta(self, key, **kw):
        d = await self._meta_file(key)
        for k, v in kw.items():
            if v is not None or k == "size":
                d[k] = v
        await self._meta_file(key, d)

    async def have(self, key):
        m = await self.meta(key)
        ch = m["chunk"] or _CHUNK_DEFAULT
        return {i for a, b in m["covered"] for i in range(a // ch, (b - 1) // ch + 1)}

    async def get(self, key, i):
        from js import Uint8Array
        m = await self.meta(key)
        ch = m["chunk"] or _CHUNK_DEFAULT
        h = await self._file(key)
        size = h.getSize()
        off = i * ch
        if off >= size:
            return None
        n = min(ch, size - off)
        buf = Uint8Array.new(n)
        got = h.read(buf, _at(off))
        return bytes(buf.to_py()[:got])

    async def put(self, key, i, data):
        from js import Uint8Array
        m = await self.meta(key)
        ch = m["chunk"] or _CHUNK_DEFAULT
        h = await self._file(key)
        buf = Uint8Array.new(len(data))
        buf.assign(data)
        h.write(buf, _at(i * ch))
        h.flush()
        return True                           # no quota to run out of

    async def delete(self, key):
        n = self._name(key)
        h = self._h.pop(n, None)
        if h is not None:
            try: h.close()
            except Exception: pass
        existed = False
        for nm in (n, n + ".meta"):
            try:
                await self.dir.removeEntry(nm); existed = True
            except Exception:
                pass
        return existed

    async def keys(self):
        out = []
        it = self.dir.values()
        while True:
            r = await it.next()
            if r.done:
                break
            nm = r.value.name
            if not nm.endswith(".meta"):
                out.append(nm)
        return sorted(out)

    async def stored(self, key):
        try:
            h = await self._file(key, create=False)
            return int(h.getSize())
        except Exception:
            return 0


def _at(offset):
    """`{at: offset}` for a sync access handle, as a JS object."""
    from pyodide.ffi import to_js
    from js import Object
    return to_js({"at": offset}, dict_converter=Object.fromEntries)


_stores = {}
_dir_handle = {"h": None}
_local_files = {}          # basename -> FileSystemFileHandle, read in place
# Folders the person picked: a name under one of them is that folder's file or no file at
# all. A loader probing for a layout's optional files ("encoder/config.json",
# "decision_config.json", "tokenizer/tokenizer.json") asks for names the folder does not
# have; those fell through to the network -- HEAD and GET against the page's own origin,
# on a deployed page another site -- for a model that was never anywhere but on disk.
_local_roots = {}          # root folder name -> how many registered files lie under it


def _root_of(name):
    n = str(name)
    if "://" in n or n.startswith(("/", "./")) or "/" not in n:
        return None
    return n.split("/", 1)[0]


def _local_miss(name):
    """True for a name inside a picked folder that the folder does not contain: answered
    "not found" here, never fetched."""
    root = _root_of(name)
    return root is not None and root in _local_roots and str(name) not in _local_files


def use_model_file(handle, name=None):
    """Read a model straight out of a local file the person picked. Nothing is copied.

    Importing a model should not mean duplicating it: a 12 GB file copied into origin
    storage is slow and would not fit anyway. This registers the handle instead, and reads
    come from it at the offsets they ask for -- no download, no quota, no second copy.

    Matching is by file name, because a model's key ends in the same name the file has
    ("…/resolve/master/model.gguf" against "model.gguf"), so a file picked from disk
    satisfies the very id the loader was going to fetch. Pass `name` to override.
    """
    key = name or handle.name
    if key not in _local_files:
        root = _root_of(key)
        if root is not None:
            _local_roots[root] = _local_roots.get(root, 0) + 1
    _local_files[key] = handle
    return key


def local_files():
    """Names currently satisfied from local files, in registration order."""
    return list(_local_files)


def files_under(folder):
    """The files under `folder`, as sorted paths relative to it, or None when it cannot be
    listed: HTTP has no directory listing, so a served folder answers None rather than an
    empty list that would read as "nothing there"."""
    base = str(folder).rstrip("/")
    pre = base + "/"
    names = [n[len(pre):] for n in _local_files if n.startswith(pre)]
    if names:
        return sorted(names)
    if _local_miss(pre + "?"):                 # inside a picked folder, and nothing under it
        return []
    if _is_url(base) or _in_browser():
        return None
    import os
    if not os.path.isdir(base):
        return None
    return sorted(os.path.relpath(os.path.join(d, f), base).replace(os.sep, "/")
                  for d, _, fs in os.walk(base) for f in fs)


async def local_size(name):
    """The size in bytes of a file `files_under` can list, or None for one it cannot."""
    n = str(name)
    h = _local_files.get(n)
    if h is not None:
        get_file = getattr(h, "getFile", None)
        f = await get_file() if callable(get_file) else h
        return int(f.size)
    if _is_url(n) or _in_browser():
        return None
    import os
    try:
        return os.path.getsize(n)
    except OSError:
        return None


def forget_model_file(name):
    """Stop reading `name` from a local file."""
    if _local_files.pop(name, None) is None:
        return False
    root = _root_of(name)
    if root is not None and root in _local_roots:
        _local_roots[root] -= 1
        if _local_roots[root] <= 0:
            del _local_roots[root]
    return True


async def _read_local_file(handle, offset, length):
    """A byte range of a picked file, read where it lies."""
    # FileSystemFileHandle.getFile() and an <input>-supplied File converge here. File is a
    # Blob already and structured-clones without first becoming one giant ArrayBuffer.
    get_file = getattr(handle, "getFile", None)
    f = await get_file() if callable(get_file) else handle
    total = int(f.size)
    end = total if length is None else min(offset + length, total)
    if end <= offset:
        return b""
    blob = f.slice(offset, end)
    buf = await blob.arrayBuffer()
    from js import Uint8Array
    return bytes(Uint8Array.new(buf).to_py())
_full_cb = {"cb": None, "fired": False}


def set_storage_full(cb):
    """Install `cb(info)`, called once when origin storage runs out. `None` clears it.

        key    the entry being written when it hit the wall
        quota  what the browser said the origin may use, if it said

    Origin storage is capped by browser policy -- the same page is given a few GB in one
    browser and a few more in another, while the disk has hundreds free -- so a large model
    can run out of room through no fault of the machine. That is not an error to raise: the
    load continues by streaming. It is a moment to offer the person a directory, which has
    no such cap. See `use_directory` and `migrate_cache`.
    """
    _full_cb["cb"] = cb
    _full_cb["fired"] = False


def _note_full(key):
    if _full_cb["cb"] is None or _full_cb["fired"]:
        return
    _full_cb["fired"] = True
    try:
        _full_cb["cb"]({"key": key, "quota": None})
    except Exception:
        pass


def use_directory(handle):
    """Keep cached models in a directory the person picked, instead of origin storage.

    The browser's storage quota is a policy, not a limit of the machine -- the same page is
    given 4.46 GB in one browser and 11.5 GB in another while the disk has hundreds of GB
    free, and `persist()` is refused in both. A model bigger than that number cannot be
    kept at all. A directory carries no quota, so what fits is what fits on the disk, and
    what lands there is the file itself: openable by other tools, and copied rather than
    exported.

    `handle` is a `FileSystemDirectoryHandle` from `showDirectoryPicker()`, which needs a
    gesture, so the page asks for it and passes it in. `None` goes back to origin storage.
    """
    _dir_handle["h"] = handle
    _stores.clear()


async def migrate_cache(handle, cache_dir=None, on_progress=None, keep=False):
    """Move what is already cached into a directory, then keep using that directory.

    Entry by entry, chunk by chunk, and each entry is deleted from origin storage as soon
    as it is safely written -- which is the point: when the quota is already full, freeing
    as you go is the only way there is room to continue. Memory holds one chunk, never a
    model. `keep=True` copies instead of moving, for when origin storage is not the problem.

    Returns the bytes moved. Afterwards reads come from the directory, which has no quota.
    """
    root = cache_dir or _default_hub_cache()
    src = _make_store(root)
    await src.open()
    dst = _FsaStore(root, handle)
    await dst.open()
    moved = 0
    for key in await src.keys():
        m = await src.meta(key)
        chunk = m["chunk"] or _CHUNK_DEFAULT
        await dst.set_meta(key, size=m["size"], chunk=chunk,
                           covered=m["covered"], complete=m["complete"])
        for a, b in m["covered"]:
            i = a // chunk
            while i * chunk < b:
                blk = await src.get(key, i)
                if blk is not None:
                    await dst.put(key, i, blk)
                    moved += len(blk)
                    if on_progress is not None:
                        on_progress(moved, key)
                del blk
                i += 1
        if not keep:
            await src.delete(key)              # freed now, so the next entry has room
    use_directory(handle)
    return moved


def get_directory():
    return _dir_handle["h"]


def _make_store(root):
    """IndexedDB in the browser, real files on the host -- same chunk-addressed interface.

    One instance per root, reused: opening a fresh IndexedDB connection on every read would
    cost more than the read.
    """
    st = _stores.get(root)
    if st is None:
        if _dir_handle["h"] is not None:
            st = _FsaStore(root, _dir_handle["h"])
        elif _in_browser():
            st = _IdbStore(root)
        else:
            st = _DiskStore(root)
        _stores[root] = st
    return st


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
            except Cancelled:
                c = self._c()
                async with c:
                    self.inflight -= 1               # a stop is not a transport fault: free
                    c.notify_all()                   # the slot and let it through, never retried
                raise
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


# The HTTP transport, the throttle and the read-ahead below are the pieces a read
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
    el = now - (_dl["t0"] or now)
    rate = _dl["rate"] or ((_dl["n"] / el) if el >= _RATE_MIN_S else 0.0)
    try:
        cb({"url": url, "bytes": _dl["n"], "rate": rate})
    except Exception:
        pass


def http_rate_limited(exc):
    """True when `exc` from the HTTP transport means "slow down".

    This is the transport's own knowledge -- a 429, or a body that says so in as many words
    -- and it is handed to the throttle rather than assumed by it, because the cache
    has no idea what its reader speaks.
    """
    return isinstance(exc, HttpError) and _is_rate_limited(exc.status, exc.body)


async def http_get(url, offset=0, length=None, headers=None):
    """The built-in HTTP range transport (browser `fetch` / host `urllib`). Raises `HttpError`
    on a non-2xx status. A ready-made `fetch` building block for a read callback."""
    rng = (("bytes=%d-%d" % (offset, offset + length - 1)) if length is not None
           else (("bytes=%d-" % offset) if offset else None))
    data = await _fetch_once(url, rng, headers)
    # No `_note_download` here: the streaming reader already reported these bytes as they
    # arrived, and counting them again would double the rate it shows.
    return data

async def http_size(url, headers=None):
    """Total byte length of an http(s) file (HEAD, else a `bytes=0-0` GET's Content-Range).
    A ready-made `size` building block for a read callback (drives read-ahead)."""
    return await _remote_size(url, headers)

def throttle_reads(fetch, max_parallel=16, is_rate_limited=None):
    # 16, not 8: reading a 12 GB model measured 13.5 s at eight parallel reads and 9.7 s at
    # twenty-four, so eight was leaving the link idle. This is a starting point rather than a
    # ceiling -- the AIMD loop below backs off on its own if a host objects.
    """Wrap a transport with an adaptive concurrency gate. Returns a `fetch`-shaped callable.

    This belongs to the transport, not to the cache: how many requests a host will take at
    once, what it does when pushed too hard, and what "slow down" even looks like are facts
    about the thing being read, and only its reader knows them. Compose it around your own
    reader before caching it; the built-in HTTP readers do exactly
    that.

    `is_rate_limited(exc) -> bool` tells the gate which failures mean "slow down" -- for
    HTTP that is `http_rate_limited`. See `_AdaptiveLimiter` for how the limit self-tunes.
    """
    lim = _AdaptiveLimiter(max_parallel, is_rate_limited)

    async def gated(key, offset, length):
        return await lim.run(lambda: fetch(key, offset, length))
    return gated


# Which chunks a background read-ahead has in flight right now. A foreground read that
# wants one of them should wait for it and then take it from the cache, rather than asking
# for the same bytes a second time; a chunk nobody is fetching it takes itself, at once.
_inflight = {}


def _mark_inflight(key, i):
    import asyncio
    fut = asyncio.get_running_loop().create_future()
    _inflight[(key, i)] = fut
    return fut


def _clear_inflight(key, i, fut, exc=None):
    _inflight.pop((key, i), None)
    if not fut.done():
        if exc is not None:
            fut.set_exception(exc)
            fut.exception()                   # marked as retrieved; waiters see it too
        else:
            fut.set_result(None)


async def await_inflight(key, offset, chunk):
    """If a read-ahead is already fetching the chunk at `offset`, wait for it. -> did we wait.

    Whatever went wrong for the read-ahead is raised here rather than swallowed. Asking for
    the same bytes again after they just failed has no reason to go differently, so the
    caller is told what happened instead of quietly retrying and failing a second time.
    """
    import asyncio
    fut = _inflight.get((key, offset // chunk))
    if fut is None:
        return False
    await asyncio.shield(fut)                 # raises what the read-ahead hit
    return True


def prefetch_whole_file(fetch, size=None, cache_dir=None, chunk_mb=16, key=None):
    """Wrap a transport so that touching a file starts filling the cache with the rest of it.

    Reading sixteen bytes of a header usually means the whole file is wanted, but only the
    transport knows whether fetching ahead is a good idea -- whether ranges are cheap, whether
    the host minds, whether the bytes even come from somewhere worth reading ahead from. So
    the policy lives here, not in the cache: this fetches the remaining chunks in the
    background and writes them through the cache's own write interface. The cached reader
    then simply finds them and never calls the transport for those ranges again.

    Returns a `fetch`-shaped callable.
    """
    import asyncio
    chunk = chunk_mb << 20
    bg = {}                       # key -> fill task; a dead one restarts on the next touch
                                  # (a stop kills it mid-file; the next load resumes where
                                  # the cache says the covered spans end)

    async def fill(k):
        gen = _CANCEL["n"]            # a stop fired after this read-ahead started ends it,
                                      # even if its own fetch escaped the abort (it may have
                                      # been queued, not in flight, when the stop landed)
        try:
            total = await size(k) if size is not None else None
            off = 0
            while total is None or off < total:
                if _CANCEL["n"] != gen:
                    return
                # Skip what is already there, then read and keep the next span. Everything
                # goes through write_cache, which is what records the bytes as present --
                # writing chunks behind its back leaves them invisible to every reader.
                have = await read_cache(k, off, 1, cache_dir)
                if have is not None:
                    store = _make_store(cache_dir or _default_hub_cache())
                    await store.open()
                    reach = _covered_from((await store.meta(k))["covered"], off)
                    if reach is not None:
                        off = reach
                        continue
                n = chunk if total is None else min(chunk, total - off)
                fut = _mark_inflight(k, off // chunk)     # so a reader waits instead of refetching
                try:
                    b = await fetch(k, off, n)
                    if not b:
                        _clear_inflight(k, off // chunk, fut)
                        break
                    await write_cache(k, b, cache_dir, offset=off,
                                      total=total if total is not None
                                      else (off + len(b) if len(b) < n else None),
                                      chunk_mb=chunk_mb)
                except (Exception, Cancelled) as e:
                    # Hand it to whoever is waiting on this chunk, then stop reading ahead.
                    _clear_inflight(k, off // chunk, fut, e)
                    raise
                _clear_inflight(k, off // chunk, fut)
                off += len(b)
                if len(b) < n:
                    break                                # short read == end of file
                del b
        except (Exception, Cancelled):
            pass                                     # read-ahead is an optimisation, never a failure

    async def ahead(k, offset, length):
        # A whole-file read is already doing exactly what read-ahead would do. Starting a
        # second fetch races two writers and, for compressed small files, can use the wire
        # Content-Length as if it were the decoded file length. Read-ahead is only useful
        # when the foreground asked for a bounded range.
        if length is not None:
            t = bg.get(k)
            if t is None or t.done():
                bg[k] = asyncio.ensure_future(fill(k))
        return await fetch(k, offset, length)

    return ahead


# ============================ cache management ============================
# The persistent cache would otherwise grow write-only. These inspect / read / write / delete
# it, keyed by cache dir (default `default_cache_dir()`). Every entry's key is the URL without
# scheme, so its FIRST path segment is the HOST/domain — HF ("huggingface.co/…") and ModelScope
# ("modelscope.cn/…") entries never mix and can be listed/cleared per host. All are async so the
# browser's IndexedDB-backed cache is loaded first (and flushed after any change).

def default_cache_dir():
    """The cache directory the hub readers use by default
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
        # What is actually held, summed from the spans -- not the file's extent on disk.
        # A sparse file whose last byte was written looks full-size to the filesystem while
        # holding almost nothing, and reporting that would overstate every partial entry.
        held = sum(b - a for a, b in m["covered"])
        out.append({"key": key, "host": h, "size": held,
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
    the chunks the range touches are read, so this stays cheap on a multi-GB entry.

    A range that runs into a gap returns everything up to the gap -- a short result, not a
    failure -- so the caller can ask its transport for just the remainder instead of the
    whole range again. `None` means the very first byte is missing, i.e. nothing to build on.
    """
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    m = await store.meta(key)
    cov = m["covered"]
    reach = _covered_from(cov, offset)
    if reach is None:                                  # the first byte is not here
        return None
    chunk = m["chunk"] or _CHUNK_DEFAULT
    total = m["size"] if m["size"] is not None else reach
    end = total if length is None else min(offset + length, total)
    end = min(end, reach)                              # stop where the run stops
    if end <= offset:
        return None
    out = bytearray(end - offset)
    pos = 0
    for i in range(offset // chunk, (end - 1) // chunk + 1):
        base = i * chunk
        b = await store.get(key, i)
        if b is None:
            break
        lo = max(offset, base) - base
        hi = min(min(end, base + chunk) - base, len(b))
        if hi <= lo:
            break
        out[pos:pos + (hi - lo)] = b[lo:hi]
        pos += hi - lo
        del b
    return bytes(out[:pos]) if pos else None


async def _cache_has_complete_tail(key, offset=0, cache_dir=None):
    """Whether the cache covers every byte from ``offset`` through the known EOF."""
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    m = await store.meta(key)
    size = m["size"]
    reach = _covered_from(m["covered"], offset)
    return bool(m["complete"] and size is not None and reach is not None and reach >= size)


_write_locks = {}


def _write_lock(key):
    """One lock per entry: a write is read-modify-write over the span list, and the
    foreground read and the background read-ahead are both writing the same file. Without
    this they interleave, each having read the spans before the other's write, and the
    second one back overwrites the first one's record of what it just stored."""
    import asyncio
    lk = _write_locks.get(key)
    if lk is None:
        lk = _write_locks[key] = asyncio.Lock()
    return lk


async def write_cache(key, data, cache_dir=None, offset=None, complete=None, total=None,
                      chunk_mb=None):
    """Put bytes into the cache. A later `read_cache` (or a reader built on it) finds them.

    This is local storage and nothing else -- IndexedDB in a browser, files on a host. It
    fetches nothing and knows nothing about where `data` came from; that is the caller's
    business. Which is exactly how a transport fills it: read a range however you read
    ranges, then write it here, and the next read of that range never reaches you again.

    `offset=None` (the default) replaces the whole entry with `data`. Passing an `offset`
    writes just that span and leaves the rest, so a file can be filled as it arrives, in
    whatever ranges the transport happened to read -- there is no alignment to respect,
    because how the bytes are chunked underneath is not the caller's problem. Give `total`
    when the full length is known, so progress and completeness can be reported. `complete`
    sets the "whole file is here" flag; leaving it None sets it once every chunk is present.
    """
    async with _write_lock(key):
        root = cache_dir or _default_hub_cache()
        store = _make_store(root)
        await store.open()
        data = bytes(data)

        if offset is None:                                 # replace the entry outright
            await store.delete(key)
            chunk = (chunk_mb << 20) if chunk_mb else _CHUNK_DEFAULT
            await store.set_meta(key, size=len(data), chunk=chunk, covered=[[0, len(data)]],
                                 complete=True if complete is None else bool(complete))
            for i in range((len(data) + chunk - 1) // chunk):
                await store.put(key, i, data[i * chunk:(i + 1) * chunk])
            return

        _WROTE.add(key)
        meta = await store.meta(key)
        chunk = meta["chunk"] or ((chunk_mb << 20) if chunk_mb else _CHUNK_DEFAULT)
        if meta["chunk"] is None or (meta["size"] is None and total is not None):
            await store.set_meta(key, size=None if total is None else int(total), chunk=chunk)
            meta = await store.meta(key)
        # Any offset, any length. How the bytes are chunked underneath is this function's
        # business, not the caller's -- a transport writes the range it happened to read.
        # Storage is chunked at a fixed size, but a chunk need not be full and may have holes
        # in it; which bytes are actually present is kept as a merged span list, so writing
        # 0-16 and then 16-32 leaves one span rather than two, and a file that ends up complete
        # is a single span.
        first, last = offset // chunk, (offset + len(data) - 1) // chunk
        # `covered` records what is actually STORED, so a chunk the store refused -- it is
        # full -- must not be claimed. Recording it anyway marks a partial file complete and
        # makes reads ask for bytes that are not there.
        done = offset
        for i in range(first, last + 1):
            base = i * chunk
            lo, hi = max(offset, base), min(offset + len(data), base + chunk)
            piece = data[lo - offset:hi - offset]
            if lo == base and (hi - lo == chunk or
                               (meta["size"] is not None and hi >= meta["size"])):
                if await store.put(key, i, piece) is False:  # a whole chunk, or the file's tail
                    break
                done = hi
                continue
            cur = await store.get(key, i)
            buf = bytearray(cur if cur is not None else b"")
            if len(buf) < hi - base:                       # grow to hold this span
                buf.extend(b"\x00" * (hi - base - len(buf)))
            buf[lo - base:hi - base] = piece
            if await store.put(key, i, bytes(buf)) is False:
                break
            done = hi
        if done > offset:
            await store.set_meta(key, covered=_merge(meta["covered"], offset, done))
        meta = await store.meta(key)

        if complete is not None:
            await store.set_meta(key, complete=bool(complete))
        elif meta["size"] is not None:
            cov = meta["covered"]
            if len(cov) == 1 and cov[0][0] <= 0 and cov[0][1] >= meta["size"]:
                await store.set_meta(key, complete=True)   # one span covering it all


async def trim_partial(key, cache_dir=None):
    """Drop every chunk of `key` that is not fully present. -> bytes dropped.

    A stopped download leaves whole chunks and, at the point it stopped, one that is only
    part-written. The part-written ones are not wrong -- `covered` records byte for byte what
    is there, and a read never asks outside it -- but they are dead weight: nothing can be
    served from them until the rest of the chunk arrives, and they are what makes a stopped
    load look like a cache entry that is "there" without being usable.

    Whole chunks are kept, so a stopped load still resumes from where it got to; only the
    ragged edge goes. `covered` is trimmed to match, so what the entry claims and what it
    holds stay the same thing.
    """
    root = cache_dir or _default_hub_cache()
    store = _make_store(root)
    await store.open()
    meta = await store.meta(key)
    chunk = meta["chunk"]
    if not chunk or not meta["covered"]:
        return 0
    size = meta["size"]
    kept, dropped, freed = [], [], 0
    for lo, hi in meta["covered"]:
        first, last = lo // chunk, (hi - 1) // chunk
        for i in range(first, last + 1):
            base = i * chunk
            end = base + chunk if size is None else min(base + chunk, size)
            if lo <= base and hi >= end:                 # this chunk is entirely present
                kept.append((max(lo, base), min(hi, end)))
            else:
                dropped.append(i)
                freed += min(hi, end) - max(lo, base)
    if not dropped:
        return 0
    for i in sorted(set(dropped)):
        await store.delete_chunk(key, i)                 # a real removal, or nothing at all
    spans = []
    for lo, hi in sorted(kept):
        spans = _merge(spans, lo, hi)
    await store.set_meta(key, covered=spans)
    if size is not None:
        whole = bool(spans) and spans[0][0] <= 0 and spans[0][1] >= size
        await store.set_meta(key, complete=whole)
    return freed


async def trim_stopped(cache_dir=None):
    """Trim every entry the load that just stopped had written. -> bytes dropped.

    The stop itself cannot do this: `cancel` is called from wherever the person pressed the
    button, often while the load is mid-chunk, and the tidying has to happen after the load
    has actually unwound. So the loader's caller calls this once the `Cancelled` has come
    back to it, and each entry keeps its whole chunks and loses its ragged edge.
    """
    freed = 0
    for key in sorted(_WROTE):
        try:
            freed += await trim_partial(key, cache_dir)
        except Exception:
            pass                          # tidying must not turn a stop into a failure
    _WROTE.clear()
    return freed


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


# ---- model-hub readers: ready-made read callbacks over the HTTP transport ----
def hub_read(to_url, token=None, cache=True, cache_dir=None, max_parallel=16,
             prefetch=True, chunk_mb=16, persist=True, digest=None):
    """An `io_read`-shaped callback for ANY host that serves files over HTTP.

    `hf_read` and `modelscope_read` are two lines each on top of this -- a lambda that builds
    a URL -- and everything that makes them worth using is here: the cache, the read-ahead,
    the concurrency that adapts to what a host will actually take, the rate-limit backoff,
    and the chunked persistence that lets a half-finished download resume. It is public so
    that "write your own reader" is a real option rather than an invitation to reimplement
    all of that; a source that is not HTTP at all still goes through `set_io_read` directly.

        webtorch.set_io_read(webtorch.hub_read(
            lambda repo, path: "https://mirror.example/%s/raw/%s" % (repo, path)))

    `to_url(repo, path)` is the only thing a host-specific reader has to know. The rest of
    the arguments are `hf_read`'s, and mean the same things.

    **Several places, one file.** `to_url` may be a LIST of such functions, and then a block
    is fetched from whichever host is answering fastest, with the next one tried when a host
    stops answering. A mirror has to earn that first: same length, and one block from the
    middle that matches, because mixing blocks from two hosts that disagree gives a model
    that is corrupt with nothing reported. The first function in the list names the file for
    the cache, so which host a block came from leaves no trace and a resumed download does
    not care that the network changed under it.

        webtorch.set_io_read(webtorch.hub_read([
            lambda r, p: "https://huggingface.co/%s/resolve/main/%s" % (r, p),
            lambda r, p: "https://modelscope.cn/models/%s/resolve/master/%s" % (r, p)]))

    `digest` is how a mirror is checked properly: one `async digest(repo, path) -> str|None`
    per builder, returning the hash THAT HOST publishes for the file. A hash the host
    computed covers the whole file and costs one small request, so it is asked for first and
    the byte sampling is only what happens when a host will not answer. `hf_digest` and
    `modelscope_digest` are the two this package ships; `mirrored_read` wires them up.
    """
    return _hub_reader(to_url, token, cache, cache_dir, max_parallel, prefetch, chunk_mb,
                       persist, digest=digest)


def _hub_reader(to_url, token, cache, cache_dir, max_parallel, prefetch, chunk_mb, persist,
                digest=None):
    """The read callback behind `hub_read`, and so behind `hf_read` / `modelscope_read`.

    It asks the cache first, and when the cache says it does not have the bytes, it deals
    with that itself -- which is the whole arrangement. `read_cache` answers one question,
    "do I have this", and takes no transport because it has no idea what one would be. What
    to do about a miss is knowledge this function has and the cache does not: that these
    bytes live behind HTTP, that the host rate-limits, that reading ahead is worth it.
    """
    hdr = {"Authorization": "Bearer " + token} if token else None
    cdir = (cache_dir or _default_hub_cache()) if cache else None
    known = {}

    # ---- several places the same bytes live ---------------------------------------------
    #
    # A file mirrored on two hubs is one file. There is no reason a download has to pick a
    # host and stay with it: whichever answers a given block fastest can serve that block,
    # and a host that stops answering costs one retry rather than the whole model. Measured
    # on one mirrored checkpoint, 64 KB at three offsets: 3.1 / 1.7 / 1.2 s from one hub
    # against 1.4 / 0.48 / 0.43 s from the other, and the first aborted partway through the
    # real download. Pinning to either is leaving that on the table.
    #
    # The cache stays keyed by ONE url -- the first builder's -- or blocks fetched from
    # different hosts would land under different keys and neither copy would ever be whole.
    import time
    builders = list(to_url) if isinstance(to_url, (list, tuple)) else [to_url]
    # One per builder: `async digest(repo, path) -> str | None`, the hash the HOST publishes
    # for that file. Supplied by whoever knows the hub's API, because `hub_read` knows no
    # hub. None where a host has no such thing, or will not tell this context.
    digests = list(digest) if isinstance(digest, (list, tuple)) else [digest] * len(builders)
    digests += [None] * (len(builders) - len(digests))
    _digest_of = {}         # url -> the builder index it came from, for asking the right one
    _alt = {}               # canonical url -> [url, ...] the same file is also served at
    mirrors = {}            # canonical url -> [url, ...] proven to be the same bytes
    rate = {}               # url -> bytes per second, last seen
    _SAMPLE = 65536

    async def size(url):
        return await http_size(url, hdr)

    _said = {}              # url -> hash or None, asked once
    _mute = set()           # builder indexes that have shown they cannot answer here

    async def _published(url):
        """The hash the host publishes for this file, or None if it will not say.

        Asked once per file, and once per HOST per session. A hub whose hash lives behind an
        API that does not answer cross-origin will not start answering: the first failure
        settles it, and without that this asked again for every one of a checkpoint's 170
        tensors -- measured at 353 blocked requests and a load that stopped moving.
        """
        if url in _said:
            return _said[url]
        known_at = _digest_of.get(url)
        if not known_at:
            return None
        i, repo, path = known_at
        fn = digests[i] if i < len(digests) else None
        if fn is None or i in _mute:
            return None
        try:
            got = await fn(repo, path)
        except Exception:
            _mute.add(i)
            _said[url] = None
            return None
        # No hash for THIS file is not a host that cannot answer: a small file is stored
        # inline rather than as an LFS object and simply has none. Muting on that would give
        # up checking the weights because a config was too small to have a hash.
        out = str(got).strip().strip('"').lower() if got else None
        _said[url] = out
        return out

    async def _same_file(a, b, sa):
        """Is `b` the file `a` is? Asked before a mirror is ever MIXED IN.

        Mixing blocks from two hosts that disagree produces a model that is corrupt in a way
        nothing reports: every block arrives, the file is the right length, and the weights
        are nonsense. So a mirror has to earn it.

        The host's own hash settles it when both hosts will give one -- it covers the whole
        file and costs one small request, which is better in both directions than reading
        bytes to compare. It is not always reachable: measured from a browser, one hub loses
        the header to its CDN redirect and exposes nothing on the file route, its API answers
        but CORS-blocked, while the other hub's API answers fine. So this asks, and falls
        back when the answer is not available.

        The fallback is evidence rather than proof, and says so: equal length, and three
        blocks spread through the file. The alternative to evidence here is downloading both
        copies in full to compare them, which is the thing the mirror existed to avoid.
        """
        try:
            da, db = await _published(a), await _published(b)
            if da and db:
                return da == db                  # the hosts' own word, on the whole file
            if not sa or sa != await size(b):
                return False
            for frac in (0.1, 0.5, 0.9):
                at = int(sa * frac)
                at = max(0, min(at - at % _SAMPLE, max(0, sa - _SAMPLE)))
                ba = await http_get(a, at, _SAMPLE, hdr)
                bb = await http_get(b, at, _SAMPLE, hdr)
                if len(ba) != len(bb) or bytes(ba) != bytes(bb):
                    return False
            return True
        except Exception:
            return False

    async def _origins(url):
        """The hosts that may serve this file, fastest first."""
        if url not in mirrors:
            alts = list(_alt.get(url, []))
            try:
                here = await size(url)
            except Exception:
                here = None
            if here is None and alts:
                # The named host is not answering at all. A mirror cannot be cross-checked
                # against a host that will not speak, and refusing it on that ground means
                # the model does not load -- which is worse than what a single-source reader
                # pointed at this same mirror would have done, and no less safe: nothing is
                # MIXED here, the file simply comes from somewhere else entirely.
                mirrors[url] = alts
            else:
                proven = [url]
                for other in alts:
                    if await _same_file(url, other, here):
                        proven.append(other)
                mirrors[url] = proven
        # Unmeasured hosts go first: one block is what it costs to find out, and a host
        # nobody has tried is the only way a better one is ever discovered.
        return sorted(mirrors[url], key=lambda u: -rate.get(u, float("inf")))

    async def raw(url, offset, length):
        origins = await _origins(url)
        last = None
        for i, cand in enumerate(origins):
            t0 = time.monotonic()
            try:
                data = await http_get(cand, offset, length, hdr)
            except Exception as e:
                # A host that will not answer this block is demoted, not abandoned: hubs
                # rate-limit and recover, and the next block may go through.
                rate[cand] = 0.0
                last = e
                if i + 1 == len(origins):
                    raise
                continue
            dt = max(1e-3, time.monotonic() - t0)
            seen = rate.get(cand)
            measured = len(data) / dt
            # Smoothed, because one slow block is weather and one fast block is luck.
            rate[cand] = measured if seen in (None, 0.0) else (seen * 0.7 + measured * 0.3)
            return data
        raise last if last else RuntimeError("no origin answered " + url)

    # Concurrency and rate-limit handling are properties of HTTP, so they wrap the transport
    # here rather than living in the cache; read-ahead likewise, filling the cache in the
    # background so later reads find the bytes already there.
    get = throttle_reads(raw, max_parallel, http_rate_limited)
    if prefetch and cache:
        get = prefetch_whole_file(get, size=size, cache_dir=cdir, chunk_mb=chunk_mb)

    _settled = {}           # canonical url -> the key this file is actually cached under

    async def _same_copy(a, b):
        """Are two CACHED entries the same file? Asked before one of them is deleted.

        The hosts' own hashes settle it when both will give one. They do not always: this
        hub serves one checkpoint straight from its own domain with no object URL to read a
        hash out of, and its API does not answer cross-origin, so from a page there is no
        hash to be had for it at all -- and the duplicate sat there because of it.

        Both copies are already on this machine, so they can simply be compared. Nothing is
        downloaded; this reads what is already stored, in windows, and stops at the first
        difference. Slower than comparing two hashes and it is proof rather than evidence,
        which is the right trade when the alternative is deleting 678 MB on a guess.
        """
        da, db = await _published(a), await _published(b)
        if da and db:
            return da == db
        try:
            sa, sb = await _cached_extent(a), await _cached_extent(b)
        except Exception:
            return False
        if not sa or sa != sb:
            return False
        WIN = 8 << 20
        off = 0
        while off < sa:
            n = min(WIN, sa - off)
            x = await read_cache(a, off, n, cdir)
            y = await read_cache(b, off, n, cdir)
            if x is None or y is None or len(x) != n or len(y) != n or bytes(x) != bytes(y):
                return False
            off += n
        return True

    async def _cached_extent(key):
        """How much of this entry is actually held, or None if it is not complete."""
        for e in await list_cache(cdir):
            if e["key"] == key or key.split("://", 1)[-1] == e["key"]:
                return e["total"] if e["complete"] else None
        return None

    async def _cache_key(url):
        """The one key this file is held under, whichever host it came from.

        A cache keyed by URL treats the same bytes on two hubs as two files. That is not
        hypothetical: switching this reader from one hub to the other put a second 678 MB
        copy of the same checkpoint on disk, both complete, both identical -- the hashes had
        already been checked equal, which is how blocks from either host were allowed to mix
        in the first place.

        So before fetching, the other hosts' urls are asked whether they are already holding
        it, and the first that is becomes the key for reads AND writes. Nothing is
        re-downloaded because a race picked a different winner this time, and nothing is
        stored twice.
        """
        if cdir is None or url in _settled:
            return _settled.get(url, url)
        cands = [url] + list(_alt.get(url, []))
        if len(cands) == 1:
            _settled[url] = url
            return url
        held = []
        for c in cands:
            try:
                if await read_cache(c, 0, 1, cdir):
                    held.append(c)
            except Exception:
                continue
        if not held:
            _settled[url] = url
            return url
        keep = held[0]
        # More than one copy. They are only the same file if the hosts say so -- `_alt` is a
        # list of places this file is EXPECTED to be, not proof that it is, and deleting
        # 678 MB on an expectation is not a thing to get wrong. Where both hosts publish a
        # hash and the hashes agree, the extra copy goes now rather than waiting for someone
        # to run a cleanup they do not know exists.
        for other in held[1:]:
            if await _same_copy(keep, other):
                try:
                    await delete_cache(other, cdir)
                except Exception:
                    pass
        _settled[url] = keep
        return keep

    def to_key(name):
        """The one url this file is cached under, remembering where else it lives.

        The first builder names the file; the rest are places to fetch the same bytes from.
        Only the first is ever a cache key, so which host a block came from leaves no trace
        and a resumed download does not care that the network changed under it.
        """
        if name.startswith(("http://", "https://")):
            return name
        repo, path = _split_repo(name)
        urls = [b(repo, path) for b in builders]
        for i, u in enumerate(urls):
            _digest_of[u] = (i, repo, path)
            if len(urls) > 1:
                # Each candidate lists the others, so the key may be ANY of them -- which is
                # what lets a file already cached under one host go on being read from there
                # while the rest stay available to fetch from.
                _alt.setdefault(u, [v for v in urls if v != u])
        return urls[0]

    async def read(name, offset=0, length=None):
        # A file the person pointed at IS the model: read it where it lies, before
        # anything else. No download, no copy, no quota. Check BEFORE to_key so a
        # plain basename like "model.gguf" matches even when the hub reader would
        # turn it into a URL whose trailing-slash tail is empty.
        n = str(name)
        # Full name first, then basename (see the other lookup): directory-registered files
        # ("mydir/config.json") match exactly, plain basenames still work.
        h = _local_files.get(n) or _local_files.get(n.rsplit("/", 1)[-1])
        if h is not None:
            data = await _read_local_file(h, offset, length)
            _report(name, len(data), None, offset)
            return data
        if _local_miss(n):
            raise FileNotFoundError("%s is not in the folder that was picked" % n)
        # A same-origin path ("/models/…", "./x.gguf") is a location, not a repo id: fetch it
        # where it is served and do NOT copy it into the hub cache — it is already local, so
        # caching it would burn quota for nothing. Only repo ids and full URLs map to the hub.
        if str(name).startswith(("/", "./")):
            data = await http_get(name, offset, length)
            _report(name, len(data), None, offset)
            return data
        url = await _cache_key(to_key(name))
        if cdir is None:                               # caching off: straight to HTTP
            data = await get(url, offset, length)
            _report(url, len(data), None, offset)
            return data

        hit = await read_cache(url, offset, length, cdir)
        complete = (length is None and hit is not None
                    and await _cache_has_complete_tail(url, offset, cdir))
        if hit is not None and (complete or (length is not None and len(hit) >= length)):
            _report(url, len(hit), known.get(url), offset)  # cache is load coverage too
            return hit

        # A miss, or a short answer that ran into a gap. If the read-ahead is already
        # fetching exactly this chunk, wait for it and take it from the cache -- asking for
        # the same bytes again would just download them twice. A chunk nobody is fetching we
        # take ourselves, right now, rather than waiting for the read-ahead to reach it.
        got = len(hit) if hit else 0
        if await await_inflight(url, offset + got, chunk_mb << 20):
            again = await read_cache(url, offset, length, cdir)
            complete = (length is None and again is not None
                        and await _cache_has_complete_tail(url, offset, cdir))
            if again is not None and (complete or (length is not None and len(again) >= length)):
                _report(url, len(again), known.get(url), offset)
                return again
            hit = again if again is not None else hit
            got = len(hit) if hit else 0
        if url not in known:
            try: known[url] = await size(url)
            except Exception: known[url] = None
        total = known.get(url)
        want = None if length is None else length - got
        data = await get(url, offset + got, want)
        # A short answer means the file ends here -- often the only way to learn its length,
        # since a cross-origin host need not expose Content-Length. A read that asked for no
        # length at all ends the file too: what came back IS the rest of it. Without that
        # second case a whole-file read leaves the entry with no known size, and an entry
        # with no size can never be marked complete -- which is why small configs fetched in
        # one go sat in the cache reading "incomplete" forever while being entirely present.
        if want is None or (total is None and len(data) < want):
            total = offset + got + len(data)
            known[url] = total
        await write_cache(url, data, cdir, offset=offset + got, total=total, chunk_mb=chunk_mb)
        out = (hit or b"") + data
        _report(url, len(out), total, offset)
        return out

    return read


# Hub APIs that have already shown they cannot be reached from here. Per session, because
# that is how long the answer stays true: a CORS policy does not change mid-page.
_API_DEAD = {}


async def _final_url(url, headers=None):
    """Where a URL actually lands, after redirects. One byte is fetched to find out."""
    try:
        from pyodide.http import pyfetch                     # browser
    except ImportError:
        pyfetch = None
    h = dict(headers or {}); h["Range"] = "bytes=0-0"
    if pyfetch is not None:
        r = await _pyfetch(url, headers=h)
        # Read the one byte. An unread body is an open connection, and a few hundred of
        # those is a browser that stops making requests at all.
        try: await r.bytes()
        except Exception: pass
        return str(getattr(r, "url", "") or "")
    import urllib.request
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.geturl()


async def lfs_digest(url, token=None):
    """The sha256 a host's own object URL names, or None.

    Git-LFS stores an object under its hash, so a host that serves one from a CDN usually
    says the hash in the path: `/lfs-objects/df/85/3bf7fe…95b72`, which is the oid split by
    the usual two-and-two. That is the host's own statement about the file and it costs
    nothing extra -- it comes back on the redirect of a request that was going to happen
    anyway -- and it is readable from a browser, which the headers and APIs carrying the same
    number frequently are not.

    None whenever the path does not spell one out, which is not a failure: the caller falls
    back to whatever else it knows.
    """
    import re
    hdr = {"Authorization": "Bearer " + token} if token else None
    try:
        final = await _final_url(url, hdr)
    except Exception:
        return None
    for part in reversed(final.split("?")[0].split("/")):
        if not re.fullmatch(r"[0-9a-f]+", part or ""):
            continue
        # Walk back up the path joining hex segments, which is how the two-and-two split
        # reads: .../df/85/3bf7…  ->  df85 3bf7…
        pieces, segs = [], final.split("?")[0].split("/")
        i = segs.index(part)
        j = i
        while j > 0 and re.fullmatch(r"[0-9a-f]+", segs[j - 1] or ""):
            j -= 1
        joined = "".join(segs[j:i + 1])
        if len(joined) == 64:
            return joined
    return None


async def hf_digest(repo, path, endpoint="https://huggingface.co", revision="main", token=None):
    """The sha256 Hugging Face publishes for one file, or None.

    From the LFS pointer, which this hub serves at `/raw/` as about 130 bytes of text and
    which reads cross-origin. Not from the file's own headers: `X-Linked-ETag` carries the
    number, but a browser following the redirect to the CDN never sees that header, and the
    CDN's own `ETag` is a different hash entirely. The repo API carries it too and is the
    fallback, being the larger request of the two.
    """
    import json, re
    base = endpoint.rstrip("/")
    hdr = {"Authorization": "Bearer " + token} if token else None
    try:
        ptr = bytes(await http_get("%s/%s/raw/%s/%s" % (base, repo, revision, path),
                                   0, None, hdr)).decode("utf-8", "replace")
        m = re.search(r"oid sha256:([0-9a-f]{64})", ptr)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        body = await http_get("%s/api/models/%s?blobs=true" % (base, repo), 0, None, hdr)
        for f in (json.loads(bytes(body).decode("utf-8")) or {}).get("siblings") or []:
            if f.get("rfilename") == path:
                lfs = f.get("lfs") or {}
                return lfs.get("sha256") or lfs.get("oid")
    except Exception:
        pass
    return None


# ModelScope answers on two origins, and they are NOT copies of each other. Measured, three
# reads each: `mccoysc/xDecision` is 200 on .ai and 404 on .cn, `convaiinnovations/laya-multilingual`
# is 200 on .cn and 404 on .ai, and `Qwen/Qwen2-0.5B-Instruct` is on both. Neither is a superset,
# and which origin carries a repo is not something a caller can be expected to know -- so the
# reader this package ships is given both and falls over between them. That is not the SDK
# choosing a hub: the caller chose ModelScope, and this is what reaching ModelScope takes.
#
# .cn first because it is the established one and the first entry names the file for the cache;
# a second origin still has to prove it holds the same bytes before any of them are mixed in.
MODELSCOPE_ORIGINS = ("https://modelscope.cn", "https://modelscope.ai")


async def modelscope_digest(repo, path, endpoint="https://modelscope.cn", revision="master",
                            token=None):
    """The sha256 ModelScope publishes for one file, or None.

    Two ways, because neither works everywhere. Its APIs carry the number and do not answer
    cross-origin -- measured, both the file listing and the raw-pointer route fail from a
    page -- so from a browser the answer comes from where the content redirect LANDS: this
    hub serves LFS objects from a path that spells the oid out, and a final URL is readable
    where a header is not.
    """
    import json
    base = endpoint.rstrip("/")
    hdr = {"Authorization": "Bearer " + token} if token else None
    # Tried once. Where this API is reachable it answers for every file; where it is not --
    # a browser, because it sends no CORS header -- it will not start, and asking again per
    # file is a failed request and a console line each time for nothing.
    try:
        if _API_DEAD.get("modelscope@" + base):
            raise RuntimeError("asked before")
        body = await http_get("%s/api/v1/models/%s/repo/files?Revision=%s" % (base, repo, revision),
                              0, None, hdr)
        data = json.loads(bytes(body).decode("utf-8")) or {}
        for f in (data.get("Data") or {}).get("Files") or []:
            if f.get("Path") == path:
                if f.get("Sha256"):
                    return f["Sha256"]
    except Exception:
        _API_DEAD["modelscope@" + base] = True
    return await lfs_digest("%s/models/%s/resolve/%s/%s" % (base, repo, revision, path), token)


def mirrored_read(readers=None, **kw):
    """Read from several hubs at once, taking each block from whichever is answering.

    The ready-made case of `hub_read`'s list form: the hubs this package already knows how
    to address, used together rather than chosen between. A repo that only exists on one of
    them still works -- a mirror that cannot be proven to hold the same bytes is simply not
    used -- so this is a reasonable default for a host that does not want to care.

        webtorch.set_io_read(webtorch.mirrored_read())
        webtorch.set_io_read(webtorch.mirrored_read(token=MY_HF_TOKEN))

    `readers` overrides the hubs with your own `to_url` functions, in preference order.
    """
    if readers is None:
        readers = [
            lambda repo, path: "https://huggingface.co/%s/resolve/main/%s" % (repo, path),
        ] + [(lambda ep: lambda repo, path: "%s/models/%s/resolve/master/%s" % (ep, repo, path))(e)
             for e in MODELSCOPE_ORIGINS]
        kw.setdefault("digest", [hf_digest] + [
            (lambda ep: lambda repo, path: modelscope_digest(repo, path, endpoint=ep))(e)
            for e in MODELSCOPE_ORIGINS])
    return hub_read(list(readers), **kw)


def hf_read(revision="main", endpoint="https://huggingface.co", token=None,
            cache=True, cache_dir=None, max_parallel=16, prefetch=True, chunk_mb=16, persist=True):
    """Return an `io_read`-shaped async callback that fetches files directly from the
    **Hugging Face Hub**, caching as it goes. Install it,
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

def modelscope_read(revision="master", endpoint=None, token=None,
                    cache=True, cache_dir=None, max_parallel=16, prefetch=True, chunk_mb=16, persist=True):
    """Return an `io_read`-shaped async callback that fetches files directly from
    **ModelScope (魔搭)**, caching as it goes. Same shape and options as
    `hf_read` (incl. `persist=True` → IndexedDB-backed in the browser):

        webtorch.set_io_read(webtorch.modelscope_read())
        lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    `name` ("<org>/<repo>/<path>") maps to
    `{endpoint}/models/{org}/{repo}/resolve/{revision}/{path}`; `revision` defaults to
    ModelScope's `master`.

    **Both of ModelScope's origins**, unless `endpoint` names one. They are not copies of each
    other -- see `MODELSCOPE_ORIGINS` -- so a reader pinned to one silently cannot see part of
    the hub. Pass `endpoint=` to pin it anyway.

    This `resolve` route is used rather than the `/api/v1/.../repo?FilePath=` one because it
    (and the CDN it redirects large files to) returns `Access-Control-Allow-Origin: *` and
    allows the `Range` header — so a browser page, which can only read a cross-origin file when
    the host opts in, can stream weights directly. Reads only."""
    eps = [endpoint.rstrip("/")] if endpoint else [e.rstrip("/") for e in MODELSCOPE_ORIGINS]
    # Bound per origin rather than closing over the loop variable, which would give every
    # builder the last one.
    def to_url(ep):
        return lambda repo, path: "%s/models/%s/resolve/%s/%s" % (ep, repo, revision, path)

    def to_digest(ep):
        return lambda repo, path: modelscope_digest(repo, path, endpoint=ep,
                                                    revision=revision, token=token)

    return _hub_reader([to_url(e) for e in eps], token, cache, cache_dir, max_parallel,
                       prefetch, chunk_mb, persist,
                       digest=[to_digest(e) for e in eps] if len(eps) > 1 else None)


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
