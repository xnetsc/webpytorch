"""webtorch — a PyTorch-compatible ML SDK that runs in the browser.

Train and run CNNs, Transformers, and LLMs on WebGPU/WebGL (via Pyodide + WgPy),
with a torch-compatible core and a transformers-style model/quantization API. Third
parties use ONLY the names re-exported here — no need to touch internal modules.

Quickstart
----------
    import webtorch as torch
    torch.install_torch()                      # `import torch` now resolves to webtorch

    # --- train, like PyTorch ---
    import torch, torch.nn as nn
    net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8*28*28, 10))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()(net(x), y); loss.backward(); opt.step()

    # --- run an LLM (quantized OR fp16), transformers-style ---
    lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen-gptq")   # GPTQ / GGUF
    print(lm.generate("Hello", max_new=64))

    # --- quantize ANY fp16 model to int4 (streaming, IO-free core) ---
    await webtorch.Quantizer.quantize(src_path, out_dir, config, bits=4)

    # --- task pipelines ---
    tts = await webtorch.pipeline("text-to-speech", "cosyvoice2"); wav, mel, toks = tts.tts("hi")
    onnx = await webtorch.OnnxModel.from_source("/models/any.onnx")   # run any ONNX
"""

__version__ = "0.1.0"

# ---- torch-compatible core (Tensor / autograd / nn layers / optim / functional) ----
from . import _core
from ._core import *                                  # Tensor, Module, Linear, Conv2d, ... (public names)
from . import _core as core                           # full core (incl. kernels) if needed

# ---- high-level SDK surface (transformers/torch style) ----
from ._sdk import install_torch, AutoTokenizer, AutoModelForCausalLM, Quantizer, pipeline, register_pipeline, OnnxModel

# ---- generic LM engine + samplers (CausalLM + MoE series) ----
from .lm_engine import TransformerLM, build_lm, SAMPLERS

# The PUBLIC API is generic / task-level only (below). Concrete model implementations
# (CosyVoice2, VITS, DETR, YOLO, Qwen-VL, …) are NOT public interfaces — reach them
# generically via `pipeline(task, model=…)` / `AutoModelForCausalLM`. They remain
# importable as advanced submodules (e.g. `import webtorch.models.cosyvoice`) the same
# way `transformers.models.*` are, but are intentionally absent from the public surface.
from . import lm_engine, quantize, webio, onnxrt   # generic building blocks (advanced)

# ---- symmetric global async IO callbacks (REQUIRED) ---------------------------
# EVERY file the SDK reads/writes (weights, configs, tokenizers, ONNX, npz, quantized
# shards) flows through ONE pair of injection points. The core ships with NO default IO —
# you MUST install both, or the first read/write raises. All loaders, the quantizer, and
# the cache then use them:
#     async def my_read(name, offset=0, length=None) -> bytes: ...
#     async def my_write(name, data, offset=0) -> None: ...
#     webtorch.set_io_read(my_read); webtorch.set_io_write(my_write)
# For demos / the common browser case, `webtorch.use_default_io()` installs the built-in
# browser-fetch+Range / host-open pair in one explicit call. `offset`/`length` mark
# ranged/streaming access (e.g. int4 weight shards) so a callback can issue an HTTP Range
# request or seek into local/remote storage.
from .webio import (set_io_read, get_io_read, io_read, set_io_write, get_io_write, io_write,
                    use_default_io, default_io_read, default_io_write)

__all__ = [
    # torch-compatible core
    "install_torch", "Tensor", "core",
    # generic model / tokenizer / quantization / pipeline / onnx entry points
    "AutoTokenizer", "AutoModelForCausalLM", "pipeline", "register_pipeline", "Quantizer", "OnnxModel",
    # generic decoder engine (CausalLM + MoE series)
    "TransformerLM", "build_lm", "SAMPLERS",
    # symmetric global async IO callbacks (REQUIRED) — read: (name, offset, length) -> bytes ; write: (name, data, offset) -> None
    "set_io_read", "get_io_read", "io_read", "set_io_write", "get_io_write", "io_write",
    "use_default_io", "default_io_read", "default_io_write",
    "__version__",
]


# advanced/internal namespace for concrete model impls (not the public interface)
class _Models:
    def __getattr__(self, name):
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
models = _Models()   # webtorch.models.cosyvoice / .tts / .detection / .vl / .llm
