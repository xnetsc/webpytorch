# webtorch

**Train and run PyTorch-style models — CNNs, Transformers, LLMs — in the browser, on the GPU.**

webtorch is a PyTorch-compatible ML SDK that runs inside [Pyodide](https://pyodide.org/)
on **WebGPU** (with a **WebGL** fallback), backed by a modified
[WgPy](https://github.com/mil-tokyo/wgpy). It gives you a torch-compatible core
(`Tensor`/autograd/`nn`/`optim`), a transformers-style model API, a generic ONNX
runtime, and a streaming quantizer — so a third party can drop it in for PyTorch and
train/infer real models client-side, with **no server and no native install**.

```python
import webtorch
webtorch.install_torch()                 # `import torch` now resolves to webtorch

import torch, torch.nn as nn             # train, exactly like PyTorch
net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(16*28*28, 10))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()(net(x), y); loss.backward(); opt.step()
```

## Features

- **Drop-in PyTorch** — `Tensor` + autograd, `nn.{Linear,Conv1d/2d/3d,LayerNorm,RMSNorm,
  MultiheadAttention,…}`, `optim.{SGD,Adam,AdamW}`. Trains real GPT/CNN/Transformer on
  WebGPU **and** WebGL.
- **LLMs, quantized or fp16** — `AutoModelForCausalLM.from_pretrained(...)` for the
  CausalLM series (Qwen2/Qwen3/Llama-shaped) and the **MoE series** (Qwen2-MoE/Qwen3-MoE),
  from AutoGPTQ int4 or GGUF. WebGPU capture-replay decode (~20×).
- **Streaming quantization** — turn any fp16 model into int4/int8 without holding it in
  RAM; streams weights **in** and quantized shards **out** through two global async IO
  callbacks (`set_io`/`set_io_write`), so nothing needs to fit in memory; output is
  standard AutoGPTQ, loadable by auto_gptq / vLLM / transformers.
- **Generic ONNX runtime** — run any `.onnx` (pure-Python parser + ~50 ops, no deps).
- **Task pipelines** — uniform, model-agnostic `pipeline("text-to-speech" | "object-detection"
  | "image-to-text" | "text-generation")`; register your own models with `register_pipeline`.
  Ships text-to-speech (CosyVoice2 incl. zero-shot voice cloning, VITS), detection (YOLO/DETR),
  and vision-language (Qwen2.5-VL).

## Quickstart

Requires a browser with WebGPU (Chrome/Edge) or WebGL. Build the backend + serve, then
open the demo — see **[docs/BUILD.md](docs/BUILD.md)**. Short version:

```bash
# 1. build the WgPy backend wheels + fetch Pyodide (one-time)   → docs/BUILD.md
# 2. serve with the required COOP/COEP headers
node serve-coi.mjs . 8119
# 3. open http://localhost:8119/webapp/ , pick an example, Run
```

More usage (LLMs, quantization, pipelines): **[docs/SDK_README.md](docs/SDK_README.md)**
and the **[API reference](docs/API.md)**.

## Repository layout

```
webtorch/            ← the SDK package  (public API in __init__.py; internals under webtorch.models.*)
  __init__.py          public API: install_torch / AutoModelForCausalLM / Quantizer / pipeline / OnnxModel / …
  _core.py             torch-compatible Tensor / autograd / nn / optim + GPU kernels
  _sdk.py              transformers-style facade + task pipeline registry
  lm_engine.py         generic decoder (CausalLM + MoE series) + samplers + capture
  quantize.py          streaming quantizer (IO-free core)
  webio.py             the only IO layer: two global async callbacks (read + write) + resolvers
  onnxrt.py            generic ONNX runtime
  torchshim.py         `import torch` compatibility
  cosyvoice.py tts.py detection.py vl.py audiofe.py   model implementations (internal)
examples/            runnable examples (loaded by the web demo)
webapp/              browser demo harness (index.html, worker.js, main.js) + run.py
serve-coi.mjs        dev server with COOP/COEP + HTTP Range
tools/               weight-preparation scripts
docs/                API.md, SDK_README.md, ARCHITECTURE.md, BUILD.md, WGPY_BACKEND.md
pyproject.toml       package metadata

# vendored backend (modified WgPy — WebGPU/WebGL array kernels; see NOTICE):
src/ webgl/ webgpu/ wgpy/ cupy/ cupyx/   (+ webpack.config.js, setup_*.py, package.json)
```

Weights (`models/`), the Pyodide runtime (`lib/`), and build output (`dist/`, `build/`,
`node_modules/`) are **not** in git — see [docs/BUILD.md](docs/BUILD.md) to obtain/build them.

## Documentation

- [docs/SDK_README.md](docs/SDK_README.md) — SDK usage (torch / LLMs / quantization / pipelines)
- [docs/API.md](docs/API.md) — API reference
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how webtorch sits on WgPy; module map
- [docs/BUILD.md](docs/BUILD.md) — build the backend, fetch models, run the demo
- [docs/WGPY_BACKEND.md](docs/WGPY_BACKEND.md) — the WgPy backend (upstream)

## License & credits

MIT — see [LICENSE](LICENSE). Built on **WgPy** (© The University of Tokyo, Edge
Intelligence Systems, Inc.; MIT), whose WebGPU/WebGL backend we modified (batched matmul,
graph capture/replay, fused kernels, int4/int8 kernels). See [NOTICE](NOTICE).
