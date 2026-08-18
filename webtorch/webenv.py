"""Independent environment configuration — NOT part of webtorch.

Gives Pyodide an IndexedDB-backed *persistent* filesystem mounted at the paths
ML frameworks already use (HF_HOME, torch hub, generic ~/.cache). Existing
download/cache/load logic then works unchanged: a framework fetches a model over
HTTP, writes it to its usual cache dir, and on the next page load reads it back
from IndexedDB without re-downloading.

Usage (once, at startup):

    import webenv
    await webenv.setup()                       # mount IDBFS + load prior cache
    path = await webenv.cached_download(url, "models/foo/model.bin")
    ...                                        # hand `path` to the framework
    await webenv.persist()                     # flush new files to IndexedDB

`import torch`-style frameworks that call urllib/requests keep working because
setup() also installs pyodide_http.
"""
import os
from js import pyodide as _pyodide          # exposed by the worker bootstrap
from pyodide.ffi import create_proxy

FS = _pyodide.FS

# Framework cache roots to back with IndexedDB. Each becomes its own IDBFS mount.
CACHE_ROOTS = (
    "/root/.cache/huggingface",   # HF_HOME default -> hub/, transformers/
    "/root/.cache/torch",         # torch.hub / torch download cache
    "/root/.cache/vwebgpu",       # generic app cache
)

_mounted = set()


def _mkdirs(path):
    parts, cur = path.strip("/").split("/"), ""
    for p in parts:
        cur += "/" + p
        try:
            FS.mkdir(cur)
        except Exception:
            pass  # already exists


def _syncfs(populate):
    """FS.syncfs is callback-async; wrap it as an awaitable JS Promise."""
    from js import Promise

    def executor(resolve, reject):
        def cb(err):
            if err:
                reject(err)
            else:
                resolve(None)
        FS.syncfs(populate, create_proxy(cb))
    return Promise.new(create_proxy(executor))


async def setup(roots=CACHE_ROOTS, install_http=True):
    """Mount IDBFS at each cache root and populate from IndexedDB (prior runs)."""
    has_idbfs = bool(hasattr(FS.filesystems, "IDBFS"))
    for root in roots:
        if root in _mounted:
            continue
        _mkdirs(root)
        if has_idbfs:
            FS.mount(FS.filesystems.IDBFS, {}, root)
            _mounted.add(root)
    if _mounted:
        await _syncfs(True)   # populate MEMFS <- IndexedDB

    # point frameworks at the persistent roots
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/root/.cache/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/.cache")

    if install_http:
        try:
            import pyodide_http
            pyodide_http.patch_all()   # make urllib/requests work in-browser
        except Exception:
            pass
    return {"idbfs": has_idbfs, "mounts": sorted(_mounted)}


async def persist():
    """Flush everything written since setup() out to IndexedDB."""
    if _mounted:
        await _syncfs(False)   # push MEMFS -> IndexedDB


def is_cached(path):
    return os.path.exists(_abspath(path))


def _abspath(path):
    return path if path.startswith("/") else "/root/.cache/vwebgpu/" + path


async def cached_download(url, path, force=False):
    """Return a local FS path for `url`, downloading only if not already cached.

    `path` may be absolute (under a mounted root) or relative (-> app cache).
    After a fresh download the file is persisted to IndexedDB automatically, so a
    later page load finds it already present and skips the network entirely.
    """
    dest = _abspath(path)
    if os.path.exists(dest) and not force:
        return dest
    _mkdirs(os.path.dirname(dest))
    from . import webio
    data = await webio.io_read(url)              # global read callback
    await webio.io_write(dest, data)             # global write callback
    await persist()
    return dest
