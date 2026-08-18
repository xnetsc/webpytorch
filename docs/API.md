# webtorch API reference

All public symbols are on the top-level `webtorch` package. **Everything here runs inside
Pyodide** (CPython→WASM in the browser, usually a Web Worker) via
`pyodide.runPythonAsync(...)` on the WgPy WebGPU/WebGL backend — that is where you
`import webtorch` and where `await` at top level is valid; it is not a system-CPython
package (see [BUILD.md](BUILD.md) / [`webapp/worker.js`](../webapp/worker.js) for the
bootstrap). Async functions are marked `await`. Every IO argument accepts a **path** (str),
**bytes**, an **async callback**, or a **dict** (auto-distinguished). The library core performs no IO itself:
you **must** install two global async callbacks first — `set_io_read` and `set_io_write`
(or `use_default_io()` for the built-ins) — or the first read/write raises. See
[IO injection](#io-injection--required-global-read--write-callbacks).

## torch compatibility
- `webtorch.install_torch()` → registers the shim so `import torch` yields webtorch
  (nn / optim / autograd / functional, torch 2.x surface). Idempotent.
- `webtorch.Tensor`, `webtorch.core` → the tensor engine (also under `torch.*` after install).

## Causal LM (dense + MoE series)
- `await webtorch.AutoModelForCausalLM.from_pretrained(path, bits=4, lmax=320)`
  - `path`: AutoGPTQ dir (**int4 or int8** — bits read from its `quantization_config`) |
    `*.gguf` (llama.cpp; dequantized and requantized to int`bits`, 4 or 8) | fp16 HF
    (quantize first via `Quantizer`). LLM inference runs on the int (4/8) capture-
    accelerated kernel — there is no fp16 linear path for decoding yet.
  - Config-driven across the whole **CausalLM series** (Qwen2/Qwen3/Llama-shaped) and the
    **MoE series** — no per-model code; the model is identified by its `config`, not a name.
  - returns a model with `.generate(prompt, max_new=…) -> str` and
    `.stream(prompt, max_new=…) -> iterator[token]`.
- `await webtorch.AutoTokenizer.from_pretrained(path)` → byte-BPE tokenizer
  (`.encode/.decode`); accepts a vocab/merges dir or a `{vocab,merges}` json.
- `webtorch.TransformerLM(cfg)` / `webtorch.build_lm(cfg, get, linear, tensor)` → the
  generic engine directly (config drives dense vs MoE); `webtorch.SAMPLERS` =
  `{greedy, nucleus, ras}`.
  - `lm.init_capture(lmax)` → enable WebGPU capture-replay decode (~20×).
  - `lm.generate_stream(prompt_ids, max_new, stop_ids, sampler, …)` → yields tokens.

## Quantizer  (`webtorch.Quantizer`)
- `await Quantizer.stream(read_tensor, has_tensor, names, write_shard, bits=4, group_size=128, shard_bytes=…)`
  — **IO-free** streaming core; all callbacks are async and awaited. Returns a manifest
  `{weight_map, shards, quantization_config, nq}`. Output tensors are AutoGPTQ-format.
- `await Quantizer.quantize(src, dst, config, bits=4, group_size=128, …)` — convenience:
  `src` accepts path | async callback | bytes | dict; `dst` accepts a name/prefix (str) or
  an async shard callback. Path forms stream through the global `io_read`/`io_write`
  callbacks (nothing is assumed local — a str `dst` is just a name), and it also emits
  `model.safetensors.index.json` + `config.json` via `io_write`.
- `Quantizer.pack(W, bits, group_size)` / `Quantizer.dequant(packed)` — single-tensor.
- `webtorch.core.quantize_model(module, group_size, bits)` — quantize an in-memory nn.Module.

## Pipelines — uniform task interface (open registry, not a fixed model list)
`await webtorch.pipeline(task, model="auto", **kw)` returns a task object whose methods are
identical across models (the concrete model is never a public interface):
- `"text-to-speech"` → callable `pipe(text, reference_audio=None, reference_text="") ->
  waveform`; `.sampling_rate`, `.supports_cloning()`. Pass `clone=True` to `pipeline(...)`
  for zero-shot cloning; `reference_audio=(wav16, wav24)`.
- `"object-detection"` → `pipe(image, threshold=…) -> detections`.
- `"image-to-text"` → `pipe(image, prompt=…) -> text`.
- `"text-generation"` → `pipe(prompt, max_new=…) -> text`; `.stream(prompt) -> tokens`.

**The `model="…"` names are registry keys, not a whitelist.** The SDK *pre-registers* a
few built-in loaders for convenience — `"cosyvoice2"`/`"vits"` (TTS), `"yolo"`/`"detr"`
(detection), `"qwen-vl"` (image-to-text), `"auto"` (text-generation, → `AutoModelForCausalLM`).
`model="auto"` (the default) picks the task's default loader. You are not limited to these:
register your own **without modifying the SDK**.

```python
webtorch.register_pipeline(task, name, loader, default=False)
#   loader:  async def loader(**kw) -> impl   (impl implements the task PROTOCOL below)
```

A pipeline wrapper only ever calls its task's **protocol** on the impl — it never branches
on which model it is:
- text-to-speech: `impl.sr`, `impl.synth(text)->wav`, `impl.clone(text, ref, ref_text)->wav`, `impl.can_clone`
- object-detection: `impl.detect(image, threshold=…)`
- image-to-text: `impl.generate(prompt, image=…)`
- text-generation: `impl.generate(prompt, …)`, `impl.stream(prompt, …)`

Register and use a custom model (any task, including an **LLM**):

```python
# --- a custom TTS model ---
async def load_my_tts(**kw):
    m = await MyTTS.load(kw.get("path", "/models/my-tts"))   # your code; uses io_read internally
    return m                                                 # must expose .sr/.synth/.clone/.can_clone
webtorch.register_pipeline("text-to-speech", "my-tts", load_my_tts, default=True)
tts = await webtorch.pipeline("text-to-speech", "my-tts")
wav = tts("hello")

# --- a custom LLM behind the text-generation task ---
async def load_my_llm(**kw):
    return await webtorch.AutoModelForCausalLM.from_pretrained(kw["path"])  # any CausalLM/MoE, int4/int8/GGUF
webtorch.register_pipeline("text-generation", "my-llm", load_my_llm)
lm = await webtorch.pipeline("text-generation", "my-llm", path="/models/my-qwen-gptq")
print(lm("Hello", max_new=64))
```

For LLMs you usually don't even need the registry — `AutoModelForCausalLM.from_pretrained(path)`
already loads *any* CausalLM/MoE-series model by its `config`. The registry is for giving a
model a stable task name or plugging in a non-CausalLM architecture.

## Generic ONNX  (`webtorch.OnnxModel`)
- `await webtorch.OnnxModel.from_source(src, io=None)` — `src`: url | bytes | async
  callback (str urls read via the global `io_read`; `io=` overrides per call).
  `.run({name: ndarray}) -> [outputs]`. ~50 ops; runs any registered graph.

## IO injection — REQUIRED global read + write callbacks
Every byte the SDK reads or writes — weights, configs, tokenizers, ONNX graphs, npz,
quantized shards, index/config json — flows through **two** global async callbacks. The
core ships with **no default IO**: you must install both, or the first read/write raises
`RuntimeError` (fail-fast, so a misconfigured SDK never silently reaches the network/disk).
The two signatures mirror each other:

```python
async def my_read(name, offset=0, length=None) -> bytes: ...   # length None = whole file
async def my_write(name, data, offset=0) -> None:       ...    # offset 0 = whole file
webtorch.set_io_read(my_read)     # + webtorch.get_io_read()  / io_read(name, offset, length)
webtorch.set_io_write(my_write)   # + webtorch.get_io_write() / io_write(name, data, offset)
```

For demos and the common browser case, one explicit call installs the built-ins
(browser `fetch`+Range for reads / host+Pyodide `open` for reads & writes):

```python
webtorch.use_default_io()         # installs default_io_read + default_io_write
```

`offset`/`length` are populated for ranged/streaming access (int4 weight shards use an HTTP
`Range` read; the quantizer streams shards out) so a callback can seek, issue a Range
request, or chunk into OPFS/IndexedDB/S3. `set_io_read(None)`/`set_io_write(None)` clear a
callback back to the unconfigured (raising) state. Because it is one injection point, a
too-big-to-fit fp16 model can be streamed **in** and its quantized weights streamed **out**
to external storage without either fitting in memory. `name` is just a string handed to
your callback (a path, URL, or object key) — the SDK never assumes it is local; the browser
default resolves it as a URL relative to the page origin.

Lower-level adapters (built on the two callbacks; normally you won't call them directly):
`resolve_tensor_reader(src, io=None)`, `resolve_shard_writer(dst, io=None)`,
`await read_bytes/read_text/read_json/load_npz(src, io=None)`,
`await write_json(dst, name, obj, io=None)`. safetensors is serialized/parsed in pure
numpy, so a str `dst` (e.g. `"s3://bucket/model"`) is just a name handed to `io_write`.

## Advanced: concrete model impls  (`webtorch.models.*`, NOT the public interface)
Reach these only if you need model-internal control; the generic `pipeline` / `AutoModel*`
cover normal use. All loaders accept url | bytes | async callback; str paths read through
the global `io_read` (per-call `io=` overrides it).
- `webtorch.models.cosyvoice.CosyVoice2TTS.from_npz(flow, hift, baked)` + `.load_llm(...)` + `.load_clone(...)`
- `webtorch.models.llm.CausalLM.from_gptq/from_gguf`, `webtorch.models.tts.VitsTTS.from_npz`,
  `webtorch.models.detection.{DetrDetector,YoloDetector}.from_npz`, `webtorch.models.vl.QwenVL.from_pretrained`.
