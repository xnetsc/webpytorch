# webtorch API reference

All public symbols are on the top-level `webtorch` package. Async functions are
marked `await`. Every IO argument accepts a **path** (str), **bytes**, an **async
callback**, or a **dict** (auto-distinguished). The library core performs no IO itself:
all reads/writes route through two global async callbacks — `set_io` (reads) and
`set_io_write` (writes) — see [IO injection](#io-injection--one-global-async-read--one-global-async-write-callback).

## torch compatibility
- `webtorch.install_torch()` → registers the shim so `import torch` yields webtorch
  (nn / optim / autograd / functional, torch 2.x surface). Idempotent.
- `webtorch.Tensor`, `webtorch.core` → the tensor engine (also under `torch.*` after install).

## Causal LM (dense + MoE series)
- `await webtorch.AutoModelForCausalLM.from_pretrained(path, bits=4, lmax=320)`
  - `path`: AutoGPTQ int4 dir | `*.gguf` | (fp16 HF — quantize first via `Quantizer`).
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

## Pipelines — uniform task interface (models are internal)
`await webtorch.pipeline(task, model="auto", **kw)` returns a task object whose methods are
identical across models (the concrete model is never a public interface):
- `"text-to-speech"` (`model="cosyvoice2"|"vits"`) → callable `pipe(text, reference_audio=None,
  reference_text="") -> waveform`; `.sampling_rate`, `.supports_cloning()`. Pass `clone=True`
  to `pipeline(...)` to enable zero-shot cloning; `reference_audio=(wav16, wav24)`.
- `"object-detection"` (`"yolo"|"detr"`) → `pipe(image, threshold=…) -> detections`.
- `"image-to-text"` → `pipe(image, prompt=…) -> text`.
- `"text-generation"` → `pipe(prompt, max_new=…) -> text`; `.stream(prompt) -> tokens`.

## Generic ONNX  (`webtorch.OnnxModel`)
- `await webtorch.OnnxModel.from_source(src, io=None)` — `src`: url | bytes | async
  callback (str urls read via the global `io_read`; `io=` overrides per call).
  `.run({name: ndarray}) -> [outputs]`. ~50 ops; runs any registered graph.

## IO injection — one global async read + one global async write callback
Every byte the SDK reads or writes — weights, configs, tokenizers, ONNX graphs, npz,
quantized shards, index/config json — flows through **two** global async callbacks. The
core never touches a filesystem itself; install your own callbacks once and all loaders,
the quantizer, and the cache use them. Both signatures mirror each other:

```python
async def my_read(name, offset=0, length=None) -> bytes: ...   # length None = whole file
async def my_write(name, data, offset=0) -> None:       ...    # offset 0 = whole file
webtorch.set_io(my_read)          # webtorch.get_io() / io_read(name, offset, length)
webtorch.set_io_write(my_write)   # webtorch.get_io_write() / io_write(name, data, offset)
```

`offset`/`length` are populated for ranged/streaming access (e.g. int4 weight shards use
an HTTP `Range` read; the quantizer streams shards out) so a callback can seek, issue a
Range request, or chunk into OPFS/IndexedDB/S3. Pass `None` to `set_io`/`set_io_write` to
restore the built-ins (browser `pyfetch`+Range / host `open`+seek). Because it is one
injection point, a too-big-to-fit fp16 model can be streamed **in** and its quantized
weights streamed **out** to external storage without either fitting in memory.

Lower-level adapters (built on the two callbacks; normally you won't call them directly):
`resolve_tensor_reader(src, io=None)`, `resolve_shard_writer(dst, io=None)`,
`await read_bytes/read_text/read_json/load_npz(src, io=None)`,
`await write_json(dst, name, obj, io=None)`. safetensors is serialized/parsed in pure
numpy, so a str `dst` (e.g. `"s3://bucket/model"`) is just a name handed to `io_write`.

## Advanced: concrete model impls  (`webtorch.models.*`, NOT the public interface)
Reach these only if you need model-internal control; the generic `pipeline` / `AutoModel*`
cover normal use. All loaders accept url | bytes | async callback (+ `fetch=`).
- `webtorch.models.cosyvoice.CosyVoice2TTS.from_npz(flow, hift, baked)` + `.load_llm(...)` + `.load_clone(...)`
- `webtorch.models.llm.CausalLM.from_gptq/from_gguf`, `webtorch.models.tts.VitsTTS.from_npz`,
  `webtorch.models.detection.{DetrDetector,YoloDetector}.from_npz`, `webtorch.models.vl.QwenVL.from_pretrained`.
