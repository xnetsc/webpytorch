# webtorch

A **PyTorch-compatible ML SDK that runs in the browser.** Train and run CNNs,
Transformers, and LLMs on **WebGPU/WebGL** (via Pyodide + WgPy), with a
transformers-style model API and a streaming quantizer. Third parties use the
public API only — no need to touch internals.

> **Runtime:** this is **Python-in-the-browser**. All code below runs **inside Pyodide**
> (CPython→WASM), typically in a Web Worker, via `pyodide.runPythonAsync(code)` on the WgPy
> WebGPU/WebGL backend — hence top-level `await` works. webtorch is **not** `pip install`ed
> into system Python; its files are loaded into Pyodide's virtual FS next to the backend
> wheels (see [`webapp/worker.js`](../webapp/worker.js) and [BUILD.md](BUILD.md)). Host
> CPython import is a numpy-fallback smoke test only; GPU paths need the browser.

```python
import webtorch          # inside Pyodide (browser / Web Worker)
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

## 2. LLMs — any CausalLM/MoE by config (int4 / int8 / GGUF)

> **Set IO first.** Loading reads files, and the core has no default IO — call
> `webtorch.use_default_io()` (or your own `set_io_read`/`set_io_write`) once before any
> `from_pretrained`, or it raises. See [IO injection](#io-injection-required-not-optional).

```python
webtorch.use_default_io()                        # REQUIRED before loading (built-in fetch/open)

# Any CausalLM/MoE model, identified by its config — int4 / int8 / fp16:
lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen-gptq")
text = lm.generate("Hello", max_new=64)          # WebGPU capture-replay decode

for tok in lm.stream("...", max_new=128): render(tok)   # streaming token output

# load straight from a hub by repo id (install a hub reader instead of the default):
webtorch.set_io_read(webtorch.hf_read())         # Hugging Face; or webtorch.modelscope_read()
lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype="fp16")
```

The model is picked by its `config`, **not** a hard-coded name — one config-driven engine
covers the **CausalLM series** (Qwen2/Qwen3/Llama-shaped) and the **MoE series** (Qwen2-MoE
/ Qwen3-MoE: router top-k + optional shared expert). Verified end-to-end on the real
Qwen1.5-MoE-A2.7B (int4) → coherent text.

**Precision — int4 / int8 / fp16 (`dtype=`).** `dtype="auto"` (default): an AutoGPTQ dir
loads at whatever bits its `quantization_config` declares (**int4 or int8**); a GGUF is
requantized to int`bits`; a plain **fp16/bf16 HF dir runs unquantized** (fp16 weights, fp32
compute, via the `UnquantizedLinear` layer). Force it with `dtype="fp16"`, or quantize a
fp16 model on load with `dtype="int8"`/`"int4"`. int4/int8 use the capture-accelerated int
kernel (~20× decode); fp16 uses a plain `x @ W.T + b`. (The general torch core in §1 runs
native fp32.)

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

## 4. Task pipelines — uniform interface, open registry (not a fixed model list)

Interact by **task**, not by model. Each pipeline exposes the same methods regardless of
the model underneath; the concrete model is an internal detail, never a public interface.

```python
tts = await webtorch.pipeline("text-to-speech")                  # model="auto" → default loader
wav = tts("Hello.")                                              # same call for any TTS model
wav = tts("Hello.", reference_audio=(w16, w24))                  # generic zero-shot voice clone
                                                                 # (pipeline(..., clone=True) to enable)

det = await webtorch.pipeline("object-detection")
boxes = det(image, threshold=0.3)

vl  = await webtorch.pipeline("image-to-text", path="/models/qwen2.5-vl-3b")
caption = vl(image, prompt="Describe the image.")

gen = await webtorch.pipeline("text-generation", path="/models/qwen-gptq")
text = gen("Hello", max_new=64)                                  # or: for t in gen.stream("Hi"): ...

onnx = await webtorch.OnnxModel.from_source("/models/any.onnx")  # run ANY onnx graph
```

**The `model="…"` names are registry keys, not a whitelist.** The SDK *pre-registers* a
few built-in loaders (`"cosyvoice2"`/`"vits"`, `"yolo"`/`"detr"`, `"qwen-vl"`, `"auto"`);
`model="auto"` uses the task default. You register your own **without editing the SDK** —
the wrapper only calls the task protocol (`.synth/.clone`, `.detect`, `.generate`,
`.stream`) on your impl, never branching on model identity:

```python
# a custom TTS model — impl must expose .sr / .synth / .clone / .can_clone
async def load_my_tts(**kw):
    return await MyTTS.load(kw.get("path", "/models/my-tts"))    # uses io_read internally
webtorch.register_pipeline("text-to-speech", "my-tts", load_my_tts, default=True)
tts = await webtorch.pipeline("text-to-speech", "my-tts");  wav = tts("hello")

# a custom LLM behind the text-generation task
async def load_my_llm(**kw):
    return await webtorch.AutoModelForCausalLM.from_pretrained(kw["path"])   # any int4/int8/GGUF
webtorch.register_pipeline("text-generation", "my-llm", load_my_llm)
gen = await webtorch.pipeline("text-generation", "my-llm", path="/models/my-qwen-gptq")
print(gen("Hello", max_new=64))
```

> Concrete implementations are reachable for advanced use as `webtorch.models.cosyvoice`
> etc. (like `transformers.models.*`) but are **not** part of the public API.

## IO injection (required, not optional)

The library core never touches the filesystem — and it ships with **no default IO**, so you
**must** install a global read + write callback before loading anything, or the first
read/write raises `RuntimeError` (fail-fast, never a silent fallback). The two are mirror
images:

```python
# Option A — built-ins (browser fetch+Range / host+Pyodide open). One explicit call:
webtorch.use_default_io()

# Option B — bring your own storage (OPFS / IndexedDB / S3 / socket / …):
async def my_read(name, offset=0, length=None) -> bytes:   # length None = whole file
    ...                                                    # `name` is just a key/URL/path you gave a loader
async def my_write(name, data, offset=0) -> None:          # offset 0 = whole file
    ...
webtorch.set_io_read(my_read)
webtorch.set_io_write(my_write)
```

`offset`/`length` mark ranged/streaming access (int4 weight shards, streamed quantizer
output) so a callback can issue an HTTP Range request or seek into storage. Public APIs also
accept a `path` / `bytes` / `dict`, auto-distinguished; small configs are passed as plain
objects. Because it is one injection point, oversized models stream in and out of external
storage without ever fully residing in memory. See [API.md](API.md).

### Built-in and model-hub read callbacks (you install them; none is a default)
The SDK ships ready-made `io_read`-shaped callbacks — you pass one to `set_io_read`; they are
never auto-installed:

```python
# Built-ins (browser fetch+Range / host urllib+open). use_default_io() just installs both:
webtorch.set_io_read(webtorch.default_io_read); webtorch.set_io_write(webtorch.default_io_write)

# Load straight from a hub by repo id — no separate download step:
webtorch.set_io_read(webtorch.hf_read())          # Hugging Face   (token=… for gated repos)
webtorch.set_io_read(webtorch.modelscope_read())  # ModelScope 魔搭 (revision defaults to master)
lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype="fp16")
```

**Caching & read-ahead (default on).** Each hub reader caches to a local dir
(`$WEBTORCH_CACHE` or `~/.cache/webtorch/hub`; set `cache_dir=…`). The first partial read of a
file kicks off a **background whole-file prefetch** (streamed in `chunk_mb` chunks) that never
blocks reads; each range read is served from cache when present, else fetched **as its own
request right then** (never waiting for the prefetch to reach it) and stored. When a file is
fully cached it is marked complete, so a **later run reads it from disk with zero network**.
The cache is **persistent by default** (`persist=True`): on the host it is a real dir, and in
the browser it is automatically backed by **IndexedDB (IDBFS)** and synced, so it **survives
page reloads** with no setup. All range reads share a **bounded queue** (`max_parallel`,
default 8). `cache=False` = pure streaming; `prefetch=False` = cache without read-ahead;
`persist=False` = in-session-only (browser MEMFS).

**URL mapping.** A loader turns the repo id into file names like `"<org>/<repo>/config.json"`;
the callback splits the first two segments as the repo and maps the rest to the hub file URL —
HF `…/resolve/{revision}/{path}`, ModelScope
`…/api/v1/models/{org}/{repo}/repo?Revision={revision}&FilePath={path}` — fetched with HTTP
`Range` (shared transport, backoff retry). A full `http(s)://` `name` is fetched as-is. Reads
only — install a writer separately if you quantize. `default_io_read`/`default_io_write` use
the same transport for plain paths/URLs and the local (or Pyodide) filesystem.

## Layout

```
webtorch/            the importable SDK package
  __init__.py        public API (everything above)
  _core.py           torch-compatible Tensor/autograd/nn/optim + GPU kernels
  _sdk.py            AutoModel*/AutoTokenizer/Quantizer/pipeline/OnnxModel facade
  torchshim.py       `import torch` compatibility
  lm_engine.py       generic decoder (CausalLM + MoE series) + samplers + capture
  quantize.py        streaming quantizer (IO-free core)
  webio.py           the only IO layer: global async read + write callbacks + resolvers
  onnxrt.py          generic ONNX runtime
  llm.py / cosyvoice.py / tts.py / detection.py / vl.py / audiofe.py   model impls
```

Backend: WgPy (WebGPU/WebGL) in the browser, numpy on the host. See
[WGPY_BACKEND.md](WGPY_BACKEND.md) and [API.md](API.md).
