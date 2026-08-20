"""Which compute backend is actually live.

The GPU backend is wired up outside Python -- across the main thread and the worker,
before the interpreter starts -- and when that wiring is wrong nothing raises: every
tensor op falls back to numpy inside wasm. It stays correct and gets roughly two orders
of magnitude slower, which reads as "the model is big" rather than "the GPU is missing".
So the state is queryable, and `require_gpu()` turns it into an error for callers that
would rather not run at all than run at that speed.
"""

from . import _core as _wt

__all__ = ["backend", "has_gpu", "require_gpu"]


def backend():
    """`"webgpu"`, `"webgl"` or `"cpu"` -- what tensor ops will actually run on."""
    if _wt._adam_backend_ready():
        return "webgpu"
    if _wt._webgl_ready():
        return "webgl"
    return "cpu"


def has_gpu():
    """True when a GPU backend is live."""
    return backend() != "cpu"


def require_gpu(what="this model"):
    """Raise unless a GPU backend is live, naming the usual cause.

    Call it before a long download, so a misconfigured page fails in a second instead of
    after several gigabytes and a very slow first reply.
    """
    b = backend()
    if b == "cpu":
        raise RuntimeError(
            "no GPU backend is active, so %s would run on numpy inside wasm (far too "
            "slow to be usable). The backend is initialized outside Python and both "
            "halves are required: `webtorch.initMain(worker)` on the main thread before "
            "any message reaches the worker, then `webtorch.initWorker()` inside it "
            "before Pyodide starts. See webtorch/js/." % what)
    return b
