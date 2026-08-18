# webtorch

A **PyTorch-compatible ML SDK that runs in the browser.** Train and run CNNs,
Transformers, and LLMs on **WebGPU/WebGL** (via Pyodide + WgPy), with a
transformers-style model API and a streaming quantizer. Third parties use the
public API only — no need to touch internals.

```python
import webtorch
```

---

## 1. Drop-in PyTorch

```python
import webtorch
webtorch.install_torch()          # `import torch` now resolves to webtorch

import torch, torch.nn as nn
net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(16*28*28, 10))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
logits = net(x); loss = nn.CrossEntropyLoss()(logits, y)
loss.backward(); opt.step()       # autograd + optimizer, on WebGPU/WebGL
```

The core mirrors torch 2.x: `Tensor`, `nn.{Module,Linear,Conv1d/2d/3d,LayerNorm,
RMSNorm,MultiheadAttention,…}`, activations/losses, `optim.{SGD,Adam,AdamW}`,
autograd, `.to()/.view()/.transpose()/.reshape()`. Real GPT + CNN + Transformer
train end-to-end on both backends.

## 2. LLMs — quantized **or** fp16, transformers-style

```python
# already-quantized (AutoGPTQ int4 safetensors / llama.cpp GGUF)
lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen-gptq")
text = lm.generate("Hello", max_new=64)          # WebGPU capture-replay decode

# streaming token output
for tok in lm.stream("...", max_new=128): render(tok)
```

Covers the **CausalLM series** (Qwen2/Qwen3/Llama-shaped) and the **MoE series**
(Qwen2-MoE / Qwen3-MoE) through one generic layer — router top-k + optional shared
expert. Verified end-to-end on the real Qwen1.5-MoE-A2.7B (int4) → coherent text.

## 3. Quantization — dedicated, streaming, IO-free, framework-agnostic

Turn **any fp16/bf16 model** into int4/int8 without ever holding either model in RAM,
and write the result for loading by **any** framework (auto_gptq / vLLM / transformers /
this SDK). The core does no IO — inject **async** read/write callbacks:

```python
# IO-free core: caller owns all IO (disk / S3 / socket); streams in AND out
manifest = await webtorch.Quantizer.stream(read_tensor, has_tensor, names, write_shard,
                                           bits=4, group_size=128, shard_bytes=512<<20)

# framework-compat convenience: src/dst = path | async callback | bytes | dict (auto-distinguished)
await webtorch.Quantizer.quantize(src, dst, config, bits=4)
```

Peak RAM = one input tensor + one output shard, independent of model size. Output =
AutoGPTQ safetensors + index + `config.json`.

## 4. Task pipelines — uniform, model-agnostic

Interact by **task**, not by model. Each pipeline exposes the same methods regardless of
the model underneath; the concrete model (CosyVoice2/VITS/DETR/YOLO/Qwen-VL) is an
internal detail, never a public interface.

```python
tts = await webtorch.pipeline("text-to-speech")                  # model="vits" | "cosyvoice2"
wav = tts("Hello.")                                              # same call for any TTS model
wav = tts("Hello.", reference_audio=(w16, w24))                  # generic zero-shot voice clone
                                                                 # (pipeline(..., clone=True) to enable)

det = await webtorch.pipeline("object-detection")               # model="yolo" | "detr"
boxes = det(image, threshold=0.3)

vl  = await webtorch.pipeline("image-to-text", path="/models/qwen2.5-vl-3b")
caption = vl(image, prompt="Describe the image.")

gen = await webtorch.pipeline("text-generation", path="/models/qwen-gptq")
text = gen("Hello", max_new=64)                                  # or: for t in gen.stream("Hi"): ...

onnx = await webtorch.OnnxModel.from_source("/models/any.onnx")  # run ANY onnx graph
```

> Concrete implementations are reachable for advanced use as `webtorch.models.cosyvoice`
> etc. (like `transformers.models.*`) but are **not** part of the public API.

## IO injection (no hardcoded IO anywhere)

The library core never touches the filesystem. Every read/write is an **async**
callback; public APIs also accept a `path` / `bytes` / `dict` and an optional
`fetch=` injection, auto-distinguished (framework-compatible). Small configs are
passed as plain objects. See [docs/API.md](docs/API.md).

## Layout

```
webtorch/            the importable SDK package
  __init__.py        public API (everything above)
  _core.py           torch-compatible Tensor/autograd/nn/optim + GPU kernels
  _sdk.py            AutoModel*/AutoTokenizer/Quantizer/pipeline/OnnxModel facade
  torchshim.py       `import torch` compatibility
  lm_engine.py       generic decoder (CausalLM + MoE series) + samplers + capture
  quantize.py        streaming quantizer (IO-free core)
  webio.py           the only IO adapters (async), + auto-distinguishing resolvers
  onnxrt.py          generic ONNX runtime
  llm.py / cosyvoice.py / tts.py / detection.py / vl.py / audiofe.py   model impls
```

Backend: WgPy (WebGPU/WebGL) in the browser, numpy on the host. See
[../WGPY_BACKEND.md](../WGPY_BACKEND.md) and [docs/API.md](docs/API.md).
