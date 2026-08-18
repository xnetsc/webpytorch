# Architecture

webtorch is a thin, PyTorch-shaped SDK layered on the **WgPy** array backend, all inside a
Pyodide (Python-in-WASM) worker in the browser.

```
        your code  ──►  webtorch (SDK, this repo's webtorch/ package)
                              │  Tensor / autograd / nn / optim / LLM / quant / onnx / pipelines
                              ▼
        WgPy cupy-shim  ──►  WgPy backend  ──►  WebGPU (WGSL)  /  WebGL (GLSL)
        (numpy fallback off-browser)          GPU kernels: matmul, bmm, conv, softmax,
                                              layernorm, gqa-attention, kv-scatter, int4/int8
```

- **Runtime**: Pyodide runs Python on a Web Worker. WgPy bridges Python↔GPU synchronously
  via SharedArrayBuffer + `Atomics.wait` (hence the COOP/COEP headers `serve-coi.mjs` sets).
- **Backends**: WebGPU (compute shaders, batched submits, graph capture → ~sub-ms replay)
  and WebGL (fragment-shader kernels, fallback). Off-browser, `webtorch` falls back to numpy
  so pure-Python logic can be unit-tested on the host.

## SDK module map (`webtorch/`)

| module | role |
|---|---|
| `__init__.py` | **public API** (generic/task-level only) |
| `_core.py` | torch-compatible `Tensor`, autograd, `nn.*`, `optim.*`, and all GPU op wrappers + fused/quantized kernels |
| `torchshim.py` | builds the fake `torch` module so `import torch` resolves to `_core` |
| `_sdk.py` | transformers-style facade (`AutoModelForCausalLM`, `AutoTokenizer`) + the task **pipeline registry** (`pipeline`, `register_pipeline`) |
| `lm_engine.py` | generic decoder `TransformerLM` (RMSNorm + GQA + rope + SwiGLU **or** MoE), samplers (`greedy/nucleus/ras`), KV-cache + capture-replay, `build_lm` |
| `quantize.py` | streaming quantizer — IO-free `Quantizer.stream(read,has,names,write)`; convenience `Quantizer.quantize` |
| `webio.py` | the **only** IO layer: two REQUIRED global async callbacks (`set_io_read`/`io_read`, `set_io_write`/`io_write`; `use_default_io()` for built-ins; `hf_read`/`modelscope_read` to load by hub repo id) + path/bytes/callback/dict resolvers and pure-numpy safetensors read/write on top |
| `onnxrt.py` | generic ONNX runtime (pure-Python protobuf parser + ~50-op interpreter) |
| `llm.py` | `CausalLM` — loads AutoGPTQ (int4/int8), GGUF, or plain fp16/bf16 HF; runs int4/int8 (capture kernel) or fp16 (`UnquantizedLinear`, plain matmul); `BPETokenizer` |
| `cosyvoice.py` `tts.py` `detection.py` `vl.py` `audiofe.py` | concrete model impls (**internal** — reached via `pipeline` / `webtorch.models.*`) |

## Design rules

- **Generic public surface.** Users touch only generic/task-level entry points
  (`pipeline`, `AutoModelFor*`, `Quantizer`, `OnnxModel`, torch-compat). Concrete models
  (CosyVoice2, VITS, DETR, YOLO, Qwen-VL) are internal, reached via `pipeline(task, model)`
  or the advanced `webtorch.models.*` namespace. Pipelines are protocol + registry: a model
  is a set of methods (`.synth/.clone`, `.detect`, `.generate`), and third parties add models
  with `register_pipeline` without changing the SDK.
- **No hardcoded IO — two REQUIRED global async callbacks.** The core never opens files/URLs
  and ships with **no default IO**: every byte read goes through one global
  `io_read(name, offset, length)` callback and every byte written through its mirror
  `io_write(name, data, offset)` (`webio`, installed via `set_io_read`/`set_io_write`).
  Until both are installed the first read/write raises — a misconfigured SDK fails fast
  instead of silently hitting the network/disk. `webtorch.use_default_io()` opts into the
  built-in browser-fetch / host-open pair in one explicit call. `offset`/`length` enable
  ranged/streaming access (int4 weight shards, streamed quantizer output), so a too-big-to-fit
  model streams in and out without ever fully residing in memory. Path/bytes/dict adapters
  (and pure-numpy safetensors read/write) sit on top — a str `dst` is just a name handed to
  `io_write`; the SDK never assumes it is local.
- **Backend kernels** (the WgPy fork, `src/` + `webgl/` + `webgpu/`) add: batched matmul,
  graph capture/replay, fused Adam/softmax/layernorm, in-place KV-scatter, int4/int8 dequant-
  matmul. See [NOTICE](../NOTICE) and [WGPY_BACKEND.md](WGPY_BACKEND.md).
