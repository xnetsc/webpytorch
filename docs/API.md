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
- `await webtorch.AutoModelForCausalLM.from_pretrained(path, dtype="auto", bits=4, lmax=320)`
  - Runs inference at **int4, int8, or fp16** — `dtype` selects it:
    - `"auto"` (default): AutoGPTQ dir → int at its declared bits; `*.gguf` → int`bits`;
      plain fp16/bf16 HF dir → **fp16** (unquantized, weights in fp16/bf16, compute fp32).
    - `"fp16"`: force unquantized fp16 execution (plain HF dir).
    - `"int4"` / `"int8"`: quantize a plain fp16 HF dir on load, or requantize a GGUF to
      that width. (An AutoGPTQ dir always loads at its own stored bits.)
  - `path`: AutoGPTQ dir | `*.gguf` | plain fp16/bf16 HF dir (`config.json` +
    `model.safetensors[.index.json]` + `vocab.json`/`merges.txt`).
  - Config-driven across the whole **CausalLM series** (Qwen2/Qwen3/Llama-shaped) and the
    **MoE series** — no per-model code; the model is identified by its `config`, not a name.
  - int4/int8 use the capture-accelerated int kernel (~20× decode); fp16 uses a plain
    `x @ W.T + b` matmul (the `UnquantizedLinear` layer), same engine and KV-cache.
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

### Installable read callbacks — choose your data source
These are `io_read`-shaped async callbacks you **install yourself** via `set_io_read`; none is
a hidden default. They are **three parallel data sources** — installing one selects *where the
bytes come from*. Pick the one matching where your files live:

| install | bytes come from | `name` handling | cache / read-ahead |
|---|---|---|---|
| `use_default_io()` (= `default_io_read`/`default_io_write`) | **your own server / local disk** | `name` used **verbatim** as a path/URL (browser: relative to the page origin) | none |
| `set_io_read(hf_read())` | **Hugging Face Hub** | `"<org>/<repo>/<path>"` → the HF file URL | yes |
| `set_io_read(modelscope_read())` | **ModelScope (魔搭)** | `"<org>/<repo>/<path>"` → the ModelScope file URL | yes |

> **`use_default_io()` is *not* a hub.** It does no repo-id→URL mapping and no caching — it
> just `fetch`es / `open`s the `name` string as-is. So with `use_default_io()`,
> `from_pretrained("Qwen/Qwen2-0.5B-Instruct")` resolves `"Qwen/Qwen2-0.5B-Instruct/config.json"`
> **relative to your page origin** (e.g. `https://your.site/Qwen/…`), **not** to Hugging Face.
> To load from a hub by repo id, install `hf_read()` / `modelscope_read()` instead. Use
> `use_default_io()` when *you* host the weight files (your server, a CDN, `/models/…`, local disk).

- `webtorch.default_io_read` / `webtorch.default_io_write` — the built-ins for **self-hosted /
  local** files. `default_io_read` reads via browser `fetch`+Range (in Pyodide) or host
  `urllib`/`open`; `default_io_write` writes via host/Pyodide `open`+seek. `name` is used
  verbatim (a path or full URL); no hub mapping, no cache. Install them yourself, or call
  `use_default_io()` (which just does `set_io_read(default_io_read); set_io_write(default_io_write)`):
  ```python
  webtorch.use_default_io()                           # serve your OWN files (name used as-is)
  lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen-gptq")  # /models/… on your server
  # explicit form: webtorch.set_io_read(webtorch.default_io_read); webtorch.set_io_write(webtorch.default_io_write)
  ```
- `webtorch.hf_read(revision="main", endpoint=…, token=None, cache=True, cache_dir=None,
  max_parallel=8, prefetch=True, chunk_mb=16, persist=True)` — returns a callback that
  fetches straight from the **Hugging Face Hub**, so you load by repo id, no pre-download:
  ```python
  webtorch.set_io_read(webtorch.hf_read())           # + set_io_write only if you quantize
  lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype="fp16")
  print(lm.generate("The capital of France is", max_new=8))
  ```
- `webtorch.modelscope_read(revision="master", endpoint=…, token=None, cache=True, …)` —
  same options, for **ModelScope (魔搭)**:
  ```python
  webtorch.set_io_read(webtorch.modelscope_read())
  lm = await webtorch.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
  ```

**Build your own cached reader (generic tool).** The caching, background read-ahead, adaptive
concurrency (see `max_parallel` below), and browser persistence are **not** hub-specific — they
live in one reusable wrapper, `webtorch.make_cached_reader`, and `hf_read`/`modelscope_read`
are just clients of it (each supplies only a repo-id→URL mapping). Use it to give your own
source (S3, a signed CDN, your own server, …) the same behavior:

```python
async def my_fetch(key, offset, length):     # read a byte range for `key` (length None = whole)
    ...                                       # raise webtorch.HttpError(429, body) to signal rate-limit
async def my_size(key):                       # total length of `key` (drives prefetch); optional
    ...
reader = webtorch.make_cached_reader(my_fetch, size=my_size, key=lambda name: name,
                                     cache_dir=None, max_parallel=8, prefetch=True, persist=True)
webtorch.set_io_read(reader)
```
- `fetch(key, offset, length) -> bytes` (async, required): raise `webtorch.HttpError(status,
  body)` for a 429 / rate-limit response so the limiter can back off; any other exception is a
  generic error (fatal only when no other read is in flight). `webtorch.http_get(url, offset,
  length, headers)` is the built-in HTTP-range transport you can call inside `fetch`.
- `size(key) -> int | None` (async, optional): total length; `webtorch.http_size(url, headers)`
  probes it for HTTP sources. Return `None` to skip prefetch for that key.
- `key(name) -> str` (optional): map an incoming `name` to a stable cache/fetch key (default:
  identity). The hub readers map `"org/repo/path"` to the file URL here so the cache is keyed
  uniquely per platform.
- `cache` / `cache_dir` / `max_parallel` / `prefetch` / `chunk_mb` / `persist`: identical to
  `hf_read`.

#### Managing the persistent cache
Because the cache persists, it needs housekeeping — list / inspect / read / write / delete /
clear it, so it never becomes dead data. All are **async** (they load the browser's
IndexedDB-backed cache first, and flush any change back). Every entry's **key is the URL
without scheme, so its first path segment is the host/domain** — HF (`huggingface.co/…`) and
ModelScope (`modelscope.cn/…`) entries are separated and can be listed or cleared per host.

- `webtorch.default_cache_dir() -> str` — the dir the readers use by default
  (`$WEBTORCH_CACHE` or `~/.cache/webtorch/hub`). All functions below take `cache_dir=None`
  meaning this default.
- `await webtorch.list_cache(cache_dir=None, host=None) -> [ {"key","host","size","complete","path"} ]`
  — every cached entry, sorted by key; `host="huggingface.co"` filters to one domain.
- `await webtorch.cache_hosts(cache_dir=None) -> [ {"host","files","size"} ]` — per-domain
  summary (largest first), so you can see HF vs ModelScope usage at a glance.
- `await webtorch.cache_size(cache_dir=None, host=None) -> int` — total bytes cached
  (optionally for one host).
- `await webtorch.read_cache(key, offset=0, length=None, cache_dir=None) -> bytes | None` —
  read a cached entry's bytes (a `list_cache` key, or a full URL); `None` if not cached.
- `await webtorch.write_cache(key, data, cache_dir=None, complete=True) -> None` — write /
  replace an entry (pre-seed the cache); `complete=True` marks it fully cached so a reader
  serves it from disk. Persists to IndexedDB in the browser.
- `await webtorch.delete_cache(key, cache_dir=None) -> bool` — delete one entry (its data +
  markers); `True` if it existed. Persists.
- `await webtorch.clear_cache(cache_dir=None, host=None) -> int` — delete everything (or only
  one `host`'s entries); returns the number of files removed. Persists.

```python
for h in await webtorch.cache_hosts():          # e.g. [{"host":"huggingface.co","files":9,"size":9_999_999}, …]
    print(h["host"], h["files"], h["size"])
await webtorch.clear_cache(host="modelscope.cn")  # free just the ModelScope cache
```

Note: these act on disk; a reader object created earlier may still hold a just-deleted entry
in memory for its lifetime — clear the cache between loads or before creating the reader.

A complete, runnable demo of `make_cached_reader` (with `HttpError` rate-limit handling) and
every cache-management function is in **`examples/io_cache_tools.py`** — it uses no GPU or
models, so it runs on the host directly (`python examples/io_cache_tools.py`).

**Caching & read-ahead (default on).** Each hub reader caches to a local dir
(`$WEBTORCH_CACHE` or `~/.cache/webtorch/hub`, override with `cache_dir`):
- The first *partial* read of a file starts a **background whole-file prefetch** that streams
  the file into a sparse cache in `chunk_mb` chunks — it never blocks reads.
- Every range read is served from cache when those bytes are present; on a miss it fetches
  **just that range now** (a separate request — it never waits for the prefetch to reach that
  offset) and stores it. The prefetch keeps going and skips ranges already cached.
- Once the whole file is cached it is marked complete, so a **later run reads it straight
  from disk — zero network**.
- **Persistent by default (`persist=True`).** On the host the cache dir is a real directory.
  In the browser the cache dir is automatically backed by **IndexedDB (IDBFS)** and synced,
  so the cache **survives page reloads** — no setup needed. `persist=False` keeps an
  in-session-only cache (browser MEMFS, wiped on reload).
- All range reads (prefetch chunks included) go through an **adaptive queue** governed by
  `max_parallel` — see the box below. `cache=False` disables caching (pure streaming);
  `prefetch=False` keeps the cache but no read-ahead.

#### `max_parallel` — adaptive concurrency (the read queue)
`max_parallel` (default **8**) is the **ceiling** on concurrent network range reads, not a
fixed count. The live limit *self-tunes* to how the server responds, so it drives the pipe as
fast as the hub allows and backs off under rate-limiting instead of failing:

- **What counts as rate-limiting.** HTTP **429**, or any response whose body contains
  rate-limit wording — English or 中文: `rate limit`, `too many requests`, `try again later`,
  `slow down`, `quota`, `throttled`, `限速`, `限流`, `太快`, `过于频繁`, `频繁`, `请求过多`,
  `超出`, `请稍后`. (Some hubs signal a limit with a 200/403/**404** body rather than 429 —
  ModelScope does — so the wording check matters.)
- **Back-off (rate-limited).** The live limit **halves**. It is **not** immediately reduced
  to 0 while other reads are still succeeding — that current value is treated as the
  sustainable concurrency and held. A **successful** read additively climbs the limit back up
  toward the ceiling.
- **Cooldown (stalled to 0).** Only when the limit reaches **0 concurrency and no read is in
  flight** does it cool down, for an **escalating** interval — **30 → 60 → 120 → 180s, capped
  at 3 minutes** — then reopens to a single probe and retries. A success resets the escalation.
- **Abort (rate-limit).** It raises **only** when it is stalled to 0 concurrency, **nothing is
  in flight**, and the cooldowns are exhausted (a rate-limit persists past the final 3-minute
  cooldown). As long as *any* range request is still succeeding, it never aborts — the current
  limit is simply the best sustainable value.
- **Non-rate-limit errors depend on whether reads are in flight.** Some servers throttle by
  simply erroring (no 429, no rate-limit wording), so a non-rate error is only trusted as
  *genuine* when it happens with **no other read in flight**: then it is not retried and
  propagates immediately with the server's actual message (a real 404 / 5xx / malformed
  response / TLS failure / timeout). But a non-rate error **while other reads are still
  succeeding** is taken as an *undisclosed* capacity signal — the concurrency is capped to the
  number still in flight (that is the sweet spot) and the read is **retried**, not raised. A
  genuinely broken read still surfaces: as concurrency winds down it eventually errors with
  nothing else in flight, and then raises.

Net effect: on a healthy hub it runs up to `max_parallel` reads at once; under throttling it
settles at the highest concurrency the hub tolerates (cooling down and retrying if pushed to
zero); and it fails fast, with the real message, on genuine errors.

**How the URL mapping works.** A loader turns the repo id into file names like
`"<org>/<repo>/config.json"`; the callback splits the first two segments as the repo and maps
the rest to the hub file URL — HF `{endpoint}/{org}/{repo}/resolve/{revision}/{path}`;
ModelScope `{endpoint}/api/v1/models/{org}/{repo}/repo?Revision={revision}&FilePath={path}` —
fetched with an HTTP `Range` request (shared transport `_fetch_range`, exponential-backoff
retry). A full `http(s)://` `name` is fetched as-is. `token` is a Bearer header for
gated/private repos. **Reads only** — set a writer separately if you also quantize.

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
