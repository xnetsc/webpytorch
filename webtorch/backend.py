"""Which compute backend is actually live.

The GPU backend is wired up outside Python -- across the main thread and the worker,
before the interpreter starts -- and when that wiring is wrong nothing raises: every
tensor op falls back to numpy inside wasm. It stays correct and gets roughly two orders
of magnitude slower, which reads as "the model is big" rather than "the GPU is missing".
So the state is queryable, and `require_gpu()` turns it into an error for callers that
would rather not run at all than run at that speed.
"""

from . import _core as _wt

__all__ = ["backend", "has_gpu", "require_gpu", "parity_contracts"]


# Efficient-common-layer audit for the public browser capabilities. ``common_level`` records
# where the two paths become semantically identical after correctness, end-to-end latency,
# persistent/temporary memory, transfers, compilation and first-call cost are considered.
# Keep it low only when that is also the best complete path; otherwise align one level up.
_PARITY_CONTRACTS = {
    "tensor_autograd": {"common_level": "operator", "webgpu": "WGSL kernels",
                        "webgl": "GLSL kernels"},
    "dense_linear": {"common_level": "operator", "webgpu": "compute matmul",
                     "webgl": "fragment matmul"},
    "ggml_stored_linear": {"common_level": "decoder", "webgpu": "WGSL packed decoder",
                           "webgl": "translated GLSL packed decoder"},
    "ggml_phase2_linear": {
        "common_level": "operator",
        "webgpu": "measured stored/packed-dot/materialized candidates per format and shape",
        "webgl": "measured native packed GLSL path with equivalent Linear result",
    },
    "gptq_stored_linear": {"common_level": "operator", "webgpu": "packed WGSL matmul",
                           "webgl": "packed GLSL matmul"},
    "moe_linear": {"common_level": "operator", "webgpu": "routed packed dispatch",
                   "webgl": "routed packed fragment pass"},
    "attention": {"common_level": "attention", "webgpu": "flash/fused attention",
                  "webgl": "fused softmax plus matmul attention"},
    "qk_norm_rope": {
        "common_level": "layer",
        "webgpu": "validated one-dispatch per-head Q/K norm plus rotary embedding",
        "webgl": "equivalent measured norm plus rotary layer composition",
    },
    "kv_cache": {
        "common_level": "attention",
        "webgpu": "measured separate/paired in-place packed scatter cache writes",
        "webgl": "equivalent paired-write interface over the growing texture cache",
    },
    "generation": {"common_level": "api", "webgpu": "captured decode replay",
                   "webgl": "eager decode"},
    "gdn_linear_attention": {"common_level": "layer", "webgpu": "scan or step kernel",
                             "webgl": "step kernel"},
    "transformer_layer_boundary": {
        "common_level": "layer",
        "webgpu": "one-pass residual add plus RMSNorm with two outputs",
        "webgl": "equivalent residual add plus fused RMSNorm layer path",
    },
    "parallel_projection": {
        "common_level": "layer",
        "webgpu": "measured shared-dispatch original-format projections",
        "webgl": "measured separate projections behind the same layer operation",
    },
    "parallel_mlp_activation": {
        "common_level": "layer",
        "webgpu": "measured shared-dispatch projections plus SwiGLU",
        "webgl": "measured combined original-format projection plus in-place SwiGLU",
    },
    "conv_training": {"common_level": "operator", "webgpu": "WGSL forward/backward",
                      "webgl": "GLSL forward/backward"},
    "decision_features": {"common_level": "operator", "webgpu": "WGSL reduction",
                          "webgl": "equivalent tensor reduction"},
    "onnx_model": {"common_level": "api",
                   "webgpu": "shared graph interpreter; ORT WebGPU vision adapter",
                   "webgl": "shared graph interpreter; ORT WASM vision adapter"},
    "model_io_cache_progress": {"common_level": "api", "webgpu": "shared transport",
                                "webgl": "shared transport"},
}


def parity_contracts():
    """Return the audited WebGPU/WebGL implementation paths for public capabilities."""
    return {name: dict(contract) for name, contract in _PARITY_CONTRACTS.items()}


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
