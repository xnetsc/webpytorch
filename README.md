<div align="center">

# webtorch

### Run a 30-billion-parameter model in a browser tab.

**No server. No install. No native runtime.**
PyTorch-compatible, on the GPU, in Python — inside the page.

<!-- HERO -->

[Quickstart](#quickstart) · [What it does](#what-it-does) · [Speed](#speed) ·
[The chat app](#the-chat-app) · [Docs](docs/API.md)

</div>

---

```python
# ── this is running inside the browser tab ──
import webtorch
webtorch.use_default_io()

lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen3-30b-a3b.gguf")
print(lm.generate("Why is this surprising?", max_new=64))
```

Nothing was downloaded to the machine. Nothing was installed. The weights were read
by range straight from disk or a hub, the layers ran on the GPU through WebGPU, and the
whole thing was CPython — compiled to WebAssembly — executing in a worker.

## What it does

**Drop-in PyTorch.** `install_torch()` and `import torch` resolves here. `Tensor` with
autograd, `nn.{Linear, Conv1d/2d/3d, LayerNorm, RMSNorm, MultiheadAttention, …}`,
`optim.{SGD, Adam, AdamW}`. Trains real GPT/CNN/Transformer models on WebGPU **and** WebGL.

**LLMs by config, not by a supported-model list.** `AutoModelForCausalLM.from_pretrained`
reads the model's own config and runs the CausalLM family (Qwen2/Qwen3/Llama-shaped) and the
MoE family, from an AutoGPTQ directory, a **GGUF** file, or a plain **fp16/bf16 HF** folder —
at int4, int8, or fp16. Twenty-eight quantisation formats, from `Q4_K` to the 2-bit i-quants.

**Streaming quantisation.** Turn an fp16 model into int4/int8 without ever holding it in RAM:
weights stream in and quantised shards stream out through your own async IO callbacks. The
output is standard AutoGPTQ, loadable by auto_gptq / vLLM / transformers.

**Multimodal, generically.** `register_encoder` + `MultimodalLM` pair **any** decoder with
**any** media encoder, so vision and audio are not welded to one model family. Ships
text-to-speech (CosyVoice2 with zero-shot voice cloning, VITS), detection (YOLO/DETR) and
vision-language (Qwen2.5-VL).

**A generic ONNX runtime.** Any `.onnx`, a pure-Python parser and ~50 ops, no dependencies.

**Task pipelines, an open registry.** `pipeline("text-generation" | "text-to-speech" |
"object-detection" | "image-to-text" | …)`. The built-in names are *pre-registered* loaders;
register your own task without touching the SDK.

**Bring your own IO — and you must.** The core does no IO itself. The callback you install
decides where bytes come from: `use_default_io()` for your own files, `hf_read()` /
`modelscope_read()` for a hub repo id, or your own `set_io_read` / `set_io_write` for any
storage at all. Reads are cached, resumable, and persist across reloads.

## Speed

Measured on Apple silicon (`metal-3`), one captured decode step, from a clean load. These are
the models running *in the tab*, not a server:

| Model | Size on disk | GPU step | tok/s |
|---|---:|---:|---:|
| Qwen3-30B-A3B · MoE · Q3_K_XL | 13.8 GB | 25.3 ms | **39.6** |
| Qwen3.8-27B · dense · hybrid SSM | 12.0 GB | 134.5 ms | **7.4** |
| Qwen3-0.6B · Q4_K_M | 0.4 GB | 6.7 ms | **149.5** |

Weight streaming runs at **106–121 GB/s** — the hardware's read ceiling — for every
quantisation format. What is left is decode arithmetic, and it is close to uniform across
formats at 199–233 G values/s.

## The chat app

A complete local chat client lives in [`chat/`](chat/) — the SDK driving a real interface.

<!-- SCREENSHOT: chat/desktop -->
<!-- SCREENSHOT: chat/models -->
<!-- SCREENSHOT: chat/mobile -->

- **Any model that fits.** Load a GGUF or an HF folder from the device, or any repo id from a
  hub. The list is examples, not a whitelist; the only real limit is GPU memory.
- **Replies that render.** Markdown, syntax-highlighted code, LaTeX typeset with KaTeX,
  tables. Sanitised before it reaches the DOM.
- **Press the code.** A Python block gets a ▶ and runs in its own Pyodide — separate from the
  one holding the model — with output, tracebacks and matplotlib figures inline. numpy,
  pandas and matplotlib are loaded before you ask; add wheels from a URL or from disk.
- **Edit anything, block by block.** A paragraph as text, a code block in the block, a table
  cell by cell — a formula cell opens as LaTeX, an image cell as an image.
- **Works offline.** A service worker keeps every wheel, the wasm and the runtime
  permanently; the app's own files stay network-first, so an update still lands the moment
  there is a network.
- **On a phone**, too.

## Quickstart

Needs a browser with WebGPU (Chrome/Edge) or WebGL, and the COOP/COEP headers that
`SharedArrayBuffer` requires.

```bash
# 1. build the WgPy backend wheels + fetch Pyodide (one-time) → docs/BUILD.md
# 2. serve with the required headers
node serve-coi.mjs . 8119
# 3. open http://localhost:8119/chat/   (or /webapp/ for the example runner)
```

Full steps, including where to get weights: **[docs/BUILD.md](docs/BUILD.md)**.

## Where things are

```
webtorch/            the SDK
  _core.py             torch-compatible Tensor / autograd / nn / optim + the GPU kernels
  _sdk.py              transformers-style facade + the task-pipeline registry
  lm_engine.py         generic decoder (dense + MoE) + samplers + capture/replay
  quantize.py          streaming quantiser (IO-free core)
  webio.py             the only IO layer: global callbacks, hub readers, cache management
  onnxrt.py            generic ONNX runtime
  torchshim.py         `import torch` compatibility
chat/                the chat app (index.html, app.js, worker.js, pyworker.js)
webapp/              example runner
examples/            runnable examples
docs/                API.md · SDK_README.md · ARCHITECTURE.md · BUILD.md · WGPY_BACKEND.md
src/ webgl/ webgpu/ wgpy/ cupy/ cupyx/     vendored WgPy backend (modified — see NOTICE)
```

Weights (`models/`), the Pyodide runtime (`lib/`) and build output are not in git.

## Docs

- [SDK usage](docs/SDK_README.md) — torch, LLMs, quantisation, pipelines
- [API reference](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md) — how webtorch sits on WgPy
- [Build](docs/BUILD.md) — backend, models, running the demo
- [WgPy backend](docs/WGPY_BACKEND.md)

## License

MIT — see [LICENSE](LICENSE). Built on **WgPy** (© The University of Tokyo, Edge Intelligence
Systems, Inc.; MIT), whose WebGPU/WebGL backend is modified here: batched matmul, graph
capture/replay, fused kernels, and the quantised matmul kernels. See [NOTICE](NOTICE).

The chat app renders with [marked](https://github.com/markedjs/marked),
[DOMPurify](https://github.com/cure53/DOMPurify), [KaTeX](https://katex.org) and
[highlight.js](https://highlightjs.org), loaded from a CDN at runtime.
