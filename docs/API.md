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

## Unified loader — `webtorch.load` / `Model`

One entry point for every model type; the specialised APIs below remain available.

- `await webtorch.load(source=None, task=None, dtype="auto", encoder=None, reuse=True, **kw) -> Model`
  - `source`: a model dir, a hub repo id (`"org/repo"`, with a hub reader installed), a
    `.gguf`, or a `.onnx`. Omit it and pass `task=` to build a registered pipeline.
  - `task`: force a task (`"text-to-speech"`, `"asr"`, …); `encoder=`: a registered media
    encoder, giving a multimodal model; `dtype`: `"auto"|"fp16"|"int4"|"int8"`.
  - Detection is by content (extension, then the served `config.json`) — never by model name.
- `Model.infer(...)` / `model(...)` — **the single inference call for every model type**. It
  dispatches to whatever the impl exposes (`__call__`, then `generate`/`run`/`synth`/`detect`/
  `transcribe`/`classify`/`encode`). `.stream(...)` streams where supported. `.kind` and
  `.impl` stay visible and unknown attributes forward to the impl.

**A model is never loaded twice.** `load()` caches by request, so loading the same source again
returns the *same* object instead of re-downloading and rebuilding multi-GB weights, and
concurrent `load()` calls for one model fetch it once.

```python
m  = await webtorch.load("Qwen/Qwen3-0.6B")      # first call builds it
m2 = await webtorch.load("Qwen/Qwen3-0.6B")      # same object, zero extra reads (m2 is m)
m.release()                                       # free the weights (or: with await load(...) as m:)
m3 = await webtorch.load("Qwen/Qwen3-0.6B")      # released -> loads fresh
```
- `model.release()` frees the weights and drops it from the cache; using a released model
  raises. `webtorch.release_all()` releases everything; `webtorch.loaded_models()` lists what
  is held; `load(..., reuse=True)` forces a separate instance.

### Releasing any model — `webtorch.release(model)`
On-demand load/unload has to work for **every** entry point, not just `load()`. `release()`
frees a model however it was built — a `Model`, an `AutoModelForCausalLM.from_pretrained()`
result, a `pipeline(...)` task object, an `OnnxModel`, or a `MultimodalLM`:

```python
lm = await webtorch.AutoModelForCausalLM.from_pretrained(path)   # not via load()
...
webtorch.release(lm)          # weights dropped; using it afterwards raises
```
`model.release()` / `pipe.release()` work directly too, and task objects and `Model` are
context managers (`with await webtorch.load(...) as m:`). Releasing also clears any `load()`
cache entry, so a later `load()` rebuilds. Repeated load/release keeps memory flat: the freed
weights are reused by the next load.

## Generation parameters
`generate(...)` / `stream(...)` take `temperature`, `top_p`, `top_k`, `min_p`, `seed`,
`do_sample`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`min_new_tokens`, `stop`, `constraint` and `require_known_tools`.

**Sampling is the default, not greedy.** Anything the model's own `generation_config.json`
states wins; a model loaded from a bare `.gguf` states nothing, and then the default
temperature is 0.6. An EXPLICIT `temperature=0` is greedy and reproducible; so is any
`seed` with sampling on.

## Output constraints — deciding what the model may say next
No model supports this itself. A model emits a distribution over the vocabulary and has no
way to forbid anything, so the restriction is always the inference side's to apply, by
removing candidates before sampling. `constraint=` is where you say what to remove.

```python
m.generate(prompt, constraint="json")                    # built-in: valid JSON only
m.generate(prompt, constraint={"regex": r"\d{4}-\d{2}"})  # a pattern (needs `regex`)
m.generate(prompt, constraint=["END", "\n\n"])            # stop strings
m.generate(prompt, constraint=lambda text, piece: piece.isdigit())   # your own rule
```

### The callback forms
Both see decoded **text**, never token ids — which is what makes a constraint portable
across tokenizers and expressible in your terms: "a digit may follow" is a statement about
characters, and which ids spell them is not your problem.

**`allows(text, piece) -> Verdict`** is asked about one candidate at a time, walking the
model's ranking until something is accepted. Easy to write, and bounded at 64 questions per
token while the model's own top choices are acceptable. When they are not, the sampler
widens, and a constraint permitting only something rare can be asked about the whole
vocabulary.

**`decide(text, candidates) -> (Verdict, allowed)`** is asked **once** per token, handed
every candidate best-first, and answers with the permitted set as text:

```python
def decide(text, candidates):
    return webtorch.ALLOW,     ["python", "javascript"]  # only these
    return webtorch.ALLOW,     None                      # no opinion; keep the ranking
    return webtorch.ALLOW,     []                        # none of these -- widen, ask again
    return webtorch.ALLOW_END, ["}"]                     # exactly this, and it is the last
    return webtorch.END                                  # the reply is complete

m.generate(prompt, constraint={"decide": decide,
                               "finished": lambda t: t.endswith("}")})
```

A candidate counts as permitted when either string is a prefix of the other, so a name that
takes three tokens to spell is reachable without knowing how it splits. **A permitted string
that matches no candidate is looked up in the vocabulary and used anyway** — that is how you
force a token the model ranked nowhere, which a predicate cannot do: it can only reject what
it is shown.

### Verdicts — `webtorch.Verdict`
A verdict is three independent things: whether the candidate is allowed, whether it goes
into the reply, and what happens next. Twelve combinations exist and six mean something:

| | allow | take | then | meaning |
|---|---|---|---|---|
| `DENY` | no | — | `THEN_ASK` | not this one; keep asking |
| `DENY_FREE` | no | — | `THEN_FREE` | not this one, and stop asking from here |
| `END` | no | — | `THEN_END` | the reply is complete as it stands |
| `ALLOW` | yes | yes | `THEN_ASK` | the ordinary answer |
| `ALLOW_FREE` | yes | yes | `THEN_FREE` | take it, then stop asking |
| `ALLOW_END` | yes | yes | `THEN_END` | take it; it is the last |

`True` / `False` work as `ALLOW` / `DENY`. Anything else raises rather than being silently
truthy. `ALLOW_FREE` and `DENY_FREE` release the constraint, so steering only a prefix costs
nothing after the prefix.

### Cost
Measured on a 0.6B: a permissive callback is free (10.21 ms/token against 10.21 unconstrained);
a restrictive one that rejects most of the window costs about 6% (10.85). The `decide` form
is one call per token instead of 64.

### A note on `{"regex": ...}`
Pattern constraints need the **`regex`** package, for its partial matching: deciding what
may come next means asking whether a prefix can still grow into a match, and the standard
library's `re` cannot answer that — it only reports what already matches. There is no
fallback on purpose. The old one used `re.match`, which rejects every incomplete prefix
including the correct ones, so the constraint silently applied to nothing. It now raises
with the fix in the message (`await micropip.install("regex")`), or use a callback, which
needs no package at all.

### Writing your own constraint class
Anything with `allows` (or `decide`) is accepted; subclass `webtorch.Constraint` for
`reset()` and `finished()` as well. Compose with `constrain.AllOf([...])`.

## Tool calling
You decide **which** tools to offer and what they do. The SDK owns everything about how a
model is told they exist, how what it writes back is read, and how a result returns to it —
none of which varies between callers, and all of which is read from the model's own chat
template rather than assumed.

```python
tools = [{"type": "function",
          "function": {"name": "python",
                       "description": "Run Python, get the output back.",
                       "parameters": {"type": "object",
                                      "properties": {"code": {"type": "string"}},
                                      "required": ["code"]}}}]

if not m.tools_supported():            # can this model be given tools at all?
    ...

reply = m.generate(prompt, tools=tools, require_known_tools=True)
calls = m.tool_calls(reply, tools)     # [{"name", "args", "id", "known"}]
shown = m.strip_tool_calls(reply, tools)   # what a reader should see: prose, no protocol
results = [run(c) for c in calls]      # your implementations
msgs += m.tool_round_messages(shown, calls, results)   # the turns to append
```

- `m.tools_supported()` → whether this model's template renders tool definitions at all.
  Which **shape** it renders, how it writes a call, and how a result reaches it are settled
  inside `generate(tools=...)` and the methods below; a caller never needs them.
- `m.parse_tool_calls(text, tools)` / `m.tool_calls(text, tools)` → the calls, with or
  without the span each occupies. Names come back as **registered**, so noise like
  `"run_ Python"` is already resolved.
- `m.strip_tool_calls(text, tools)` → the reply with the call markup removed. Same single
  scan as the parse, so the two cannot disagree.
- `m.render_tool_call(name, args, tools)` → one call written the way this model writes them.
- `m.tool_result_message(call, content)` → one result, shaped for this template.
- `m.tool_round_messages(text, calls, results)` → the assistant turn plus every result,
  including whether the calls travel as structured `tool_calls` or as the model's own text.
- `m.suggest_tool(name, tools, args)` → ranked candidates for a name that matched nothing:
  `[{"name", "name_score", "args_match"}]`. **Evidence, not a decision** — the threshold and
  what to do about it are yours.
- `m.split_reasoning(text)` → `{"reasoning", "answer", "open"}` for a reply stored as one
  string. A live stream does not need it: `stream(channels=True)` labels the pieces.

`require_known_tools=True` holds the model to the names you registered, by removing the
others from the candidate set. Asked for, never assumed: it also makes a model that would
have called the wrong tool sometimes call none at all (measured 2 of 8 → 4 of 8 on a 0.6B),
and which failure is better depends on what you do next.

## Causal LM (dense + MoE series)
- `await webtorch.AutoModelForCausalLM.from_pretrained(path, dtype="auto", bits=None, lmax=None, weights="native")`
  - Runs inference at **int4, int8, or fp16** — `dtype` selects it:
    - `"auto"` (default): AutoGPTQ dir → int at its declared bits; `*.gguf` → **its own
      encoding**, kept packed (see `weights` below); plain fp16/bf16 HF dir → **fp16**
      (unquantized, weights in fp16/bf16, compute fp32).
    - `"fp16"`: force unquantized fp16 execution (plain HF dir).
    - `"int4"` / `"int8"`: quantize a plain fp16 HF dir on load, or requantize a GGUF to
      that width. (An AutoGPTQ dir always loads at its own stored bits.)
  - `weights=` (GGUF only) — `"native"` (default): upload each tensor in its own encoding
    and multiply it packed; no dequantize, no requantize, no fp32 intermediate. Twenty-eight
    formats, `Q4_K` through the 2-bit i-quants, each checked against a reference decoder on
    both backends. `"requant"`: dequantize and requantize to int`bits`, then run the ordinary
    int kernel. A type with no native decode falls back to that path per tensor either way.
    Any other value raises.
  - `path`: AutoGPTQ dir | `*.gguf` | plain fp16/bf16 HF dir (`config.json` +
    `model.safetensors[.index.json]` + `vocab.json`/`merges.txt`).
  - Config-driven across the whole **CausalLM series** (Qwen2/Qwen3/Llama-shaped) and the
    **MoE series** — no per-model code; the model is identified by its `config`, not a name.
  - int4/int8 use the capture-accelerated int kernel (~20× decode); fp16 uses a plain
    `x @ W.T + b` matmul (the `UnquantizedLinear` layer), same engine and KV-cache.
  - returns a model with `.generate(prompt, max_new=…) -> GenResult` and
    `.stream(prompt, max_new=…) -> iterator[token]`. `GenResult` carries `.text`,
    `.tokens`, `.ttft_s` and `.decode_tok_s`; printing it shows the timings above the text.
- `await webtorch.AutoTokenizer.from_pretrained(path)` → byte-BPE tokenizer
  (`.encode/.decode`); accepts a vocab/merges dir or a `{vocab,merges}` json.
- `webtorch.TransformerLM(cfg)` / `webtorch.build_lm(cfg, get, linear, tensor)` → the
  generic engine directly (config drives dense vs MoE); `webtorch.SAMPLERS` =
  `{greedy, nucleus, ras}`.
  - `lm.init_capture(lmax)` → enable WebGPU capture-replay decode (~20×).
  - `lm.generate_stream(prompt_ids, max_new, stop_ids, sampler, …)` → yields tokens.

## Multimodal — any decoder + any media encoder
Multimodality is not tied to one model family. Every VLM/ALM does the same three things:
**encode** media into embeddings that live in the decoder's hidden space, **splice** them into
the token embeddings at placeholder-token positions, then **decode** normally. Only the encoder
and the placeholder id differ, so that is all a new modality supplies.

- `webtorch.register_encoder(name, loader, default=False)` — register a media encoder.
  `loader(**kw)` is async and returns an object implementing the **encoder protocol**:
  `.token_id` (the placeholder token spliced over), `.encode(media, **kw) -> (n, H)` and
  optionally `.n_tokens(media)`. `webtorch.list_encoders()` / `load_encoder(name)` discover
  and build them.
- `webtorch.MultimodalLM(lm, encoder, placeholder_id=None)` — pair **any** CausalLM (the whole
  generic decoder family: Llama/Qwen2/Qwen3, dense or MoE, int4/int8/fp16) with **any**
  registered encoder. `.generate(prompt, media=None, …)` falls back to plain text generation
  when `media` is None, so a multimodal model is a strict superset of a text one.
  `.embed_prompt(ids, media)` / `.build_prompt(...)` expose the assembled embeddings.
- `webtorch.splice_embeddings(token_embeds, ids, media_embeds, placeholder_id)` — the shared
  splice step, usable on its own.
- Decoders accept prebuilt inputs directly: `generate(..., ids=…, embeds=…)`. That is the
  generic hook the multimodal path uses, so no model-specific decode path is required.

```python
class MyVision:                          # any third-party encoder
    token_id = 151655
    def n_tokens(self, img, **kw): return 4
    def encode(self, img, **kw): return embeddings          # (n, H)
webtorch.register_encoder("my-vision", loader, default=True)

m = await webtorch.load("/models/decoder", encoder="my-vision")   # or MultimodalLM(lm, enc)
print(m.generate("Describe this.", media=img))
```

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
- `"automatic-speech-recognition"` (aliases `"asr"`, `"speech-to-text"`) →
  `pipe(audio, sampling_rate=…) -> text`; `.sampling_rate`.
- `"audio-classification"` → `pipe(audio) -> labels/scores`; `.sampling_rate`.

**The task list is open too.** `pipeline()` accepts *any* task name: register a model under a
name webtorch has never heard of and it is wrapped in a generic forwarder (calls the impl, or
its `.run(...)`, and proxies attributes). For a new task that deserves its own uniform call
signature, register the wrapper with `webtorch.register_task(task, wrapper)`.
`webtorch.list_pipelines()` returns `{task: [model names]}` (or a list for one task) so callers
can discover what is registered instead of hard-coding names.

```python
class MyASR:                                    # any third-party model
    sr = 16000
    def transcribe(self, audio, **kw): return "..."
webtorch.register_pipeline("asr", "my-asr", lambda **kw: _load(), default=True)
asr = await webtorch.pipeline("asr");  text = asr(waveform)

webtorch.register_pipeline("feature-extraction", "my-embed", loader)   # a brand-new task
emb = await webtorch.pipeline("feature-extraction", "my-embed")        # generic wrapper
```

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

## GGUF quantization support
`.gguf` files are loaded generically (architecture read from the file's own `{arch}.*`
metadata). Dequantization is implemented for **F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K,
Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS** — the common
builds (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`) and the mixed/"dynamic" files that combine
several of these. The i-quants index ggml's codebook grids, kept as data in
`webtorch/iqtables.py` (extracted from `ggml-common.h`). Only the 1-bit IQ1 family is missing.

Both the quantization types and the architecture are **checked up front from the header**, so
an unsupported file fails in seconds naming exactly what is missing, instead of after
downloading gigabytes. `dtype="fp16"` runs a GGUF unquantized (the int kernel is GPU-only, so
this is what makes GGUF work without a GPU).

## Generic ONNX  (`webtorch.OnnxModel`)
- `await webtorch.OnnxModel.from_source(src, io=None)` — `src`: url | bytes | async
  callback (str urls read via the global `io_read`; `io=` overrides per call).
  `.run({name: ndarray}) -> [outputs]`. ~50 ops; runs any registered graph.

## Bringing up the GPU backend  (`webtorch.initMain` / `initWorker` / `backend`)

Tensor ops run on WebGPU (or WebGL) through a backend that spans **two JavaScript contexts**:
the main thread owns the device, the worker reaches it over shared memory. Both halves must
be initialised, in this order — the worker's half after the main thread's, and **before
Pyodide starts**.

Getting it wrong does not raise. Every op silently falls back to numpy inside wasm: still
correct, roughly two orders of magnitude slower. That failure mode is why the SDK owns the
handshake instead of leaving it to callers, and why the live backend is queryable.

- `webtorch.backend_reason()` → what is stopping the GPU path, as a sentence, or `None` when
  nothing is. Reading this is the supported way to find out why a machine that should be
  fast is not — the reason is recorded where the failure happened, which is the only way to
  tell a browser without WebGPU from a page that is merely not cross-origin isolated.
- `webtorch.__version__` — the SDK version string.

**Main thread** — load `dist/wgpy-main.js`, then `webtorch/js/webtorch-main.js`:

```html
<script src="../dist/wgpy-main.js"></script>
<script src="../webtorch/js/webtorch-main.js"></script>
```
```js
const worker = new Worker('worker.js');
// Picks the first backend this browser can actually provide and tells the worker.
// Hold every message to the worker until it resolves.
const { backend } = await webtorch.initMain(worker, { backendOrder: ['webgpu', 'webgl'] });
console.log('compute:', backend);            // 'webgpu' | 'webgl' | 'cpu'
```

- `backendOrder` — preference order; entries the browser cannot provide are skipped.
- `requireGpu: true` — throw instead of reporting `'cpu'`, when a CPU fallback is useless
  to you.

**Worker** — load `dist/wgpy-worker.js` and the Pyodide loader, then
`webtorch/js/webtorch-worker.js`. One call brings up the backend, Pyodide, the matching
backend wheel and the package modules, in the order they require:

```js
importScripts('../lib/pyodide/pyodide.js');
importScripts('../dist/wgpy-worker.js');
importScripts('../webtorch/js/webtorch-worker.js');

const { pyodide, backend } = await webtorch.initWorker({
  baseURL: '../',                            // prefix for dist/ and webtorch/
  onStatus: (t) => postMessage({ type: 'status', text: t }),
});
```

`backend` is what actually came up, probed after the fact — not what was requested. The
module list lives in the bootstrap, so callers do not track it.

**From Python** — the same question, answerable inside the runtime:

- `webtorch.backend() -> "webgpu" | "webgl" | "cpu"` — what ops will really run on.
- `webtorch.has_gpu() -> bool`
- `webtorch.require_gpu(what="this model") -> str` — raise unless a GPU backend is live,
  naming the usual cause. Call it **before** a long download, so a misconfigured page fails
  in a second instead of after several gigabytes and a very slow first reply:

```python
import webtorch
webtorch.require_gpu("Qwen3-30B")            # fails fast if the handshake was missed
m = await webtorch.load("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/…Q3_K_XL.gguf")
```

Show the value in your UI rather than inferring it from how long a reply takes — a silent
CPU fallback is otherwise indistinguishable from "the model is big".

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
webtorch.use_default_io()         # built-in reader (caches network reads) + writer
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
| `use_default_io()` | **your own server / local disk** | `name` used **verbatim** as a path/URL (browser: relative to the page origin) | **network reads cached** (browser fetches + http/https URLs); local host files read directly |
| `set_io_read(hf_read())` | **Hugging Face Hub** | `"<org>/<repo>/<path>"` → the HF file URL | yes |
| `set_io_read(modelscope_read())` | **ModelScope (魔搭)** | `"<org>/<repo>/<path>"` → the ModelScope file URL | yes |

> **`use_default_io()` is *not* a hub.** It does no repo-id→URL mapping — it just `fetch`es /
> `open`s the `name` string as-is. So with `use_default_io()`,
> `from_pretrained("Qwen/Qwen2-0.5B-Instruct")` resolves `"Qwen/Qwen2-0.5B-Instruct/config.json"`
> **relative to your page origin** (e.g. `https://your.site/Qwen/…`), **not** to Hugging Face.
> To load from a hub by repo id, install `hf_read()` / `modelscope_read()` instead. Use
> `use_default_io()` when *you* host the weight files (your server, a CDN, `/models/…`, local disk).

**Caching in `use_default_io()`.** By default it **caches network reads** with the same
read-ahead + persistence as the hub readers, so a reload / re-run does not re-download: in the
browser every read is a network fetch (from the page origin) and is cached (IndexedDB-persistent
by default); on the host, `http(s)://` URLs are cached while a **local file path is read
directly and never cached** (caching a local file would only duplicate it). Cached entries share
the same store and management functions (`list_cache` / `clear_cache` / …), keyed by host. Pass
`use_default_io(cache=True)` for plain, uncached fetch/open; `cache_dir` / `max_parallel` /
`prefetch` / `chunk_mb` / `persist` mirror `hf_read` and apply to the cached (network) reads.

- `webtorch.use_default_io(cache=True, cache_dir=None, max_parallel=16, prefetch=True, chunk_mb=16, persist=True)`
  — install the built-in IO for **self-hosted / local** files (browser `fetch`+Range or host
  `urllib`/`open`; `name` used verbatim, no hub mapping), caching network reads as above. Writes
  always go to the local/Pyodide filesystem.
  ```python
  webtorch.use_default_io()                           # serve your OWN files (cached; name used as-is)
  lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen-gptq")  # /models/… on your server
  ```
- `webtorch.default_io_read` / `webtorch.default_io_write` — the raw (uncached) built-in
  transports underneath: `default_io_read` reads via browser `fetch`+Range or host `urllib`/`open`;
  `default_io_write` writes via host/Pyodide `open`+seek. Install directly if you want the
  transport without caching (equivalent to `use_default_io(cache=True)`).
- `webtorch.hf_read(revision="main", endpoint=…, token=None, cache=True, cache_dir=None,
  max_parallel=16, prefetch=True, chunk_mb=16, persist=True)` — returns a callback that
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
live inside the ready-made readers `hf_read`/`modelscope_read`. Writing a callback for your
own source (S3, a signed CDN, your own server) means composing the same pieces yourself: read
from the cache, and on a miss read from wherever the bytes live and put them in the cache.

```python
async def my_read(name, offset=0, length=None):
    hit = await webtorch.read_cache(name, offset, length)
    if hit is not None:                       # cached -> done, nothing else happens
        return hit
    data = await my_fetch(name, offset, length)   # your transport: HTTP, S3, a local disk…
    await webtorch.write_cache(name, data, offset=offset, total=await my_size(name))
    return data

webtorch.set_io_read(my_read)
```

That is the whole model. The cache never fetches and never decides what an error means -- it
does not know what your transport speaks. Anything transport-shaped is yours to add around it,
and two ready-made pieces are here if your transport is HTTP:

- `webtorch.http_get(url, offset, length, headers)` -- the built-in HTTP range transport.
- `webtorch.http_size(url, headers)` -- its length probe. Returns `None` when the host does not
  expose `Content-Length` (ModelScope does not, cross-origin); the readers then learn the length
  from the first short read.
- `webtorch.throttle_reads(fetch, max_parallel=16, is_rate_limited=…)` -- concurrency and
  rate-limit backoff. `webtorch.http_rate_limited` is the HTTP classifier (429, or a body that
  says so). This belongs to the transport: only it knows what "slow down" looks like.
- `webtorch.prefetch_whole_file(fetch, size=…)` -- reads the rest of a touched file in the
  background and writes it through the cache, so later reads never reach your transport at all.

Note that `write_cache(..., offset=…)` is what lets a transport fill an entry chunk by chunk as
the bytes arrive, rather than holding a multi-gigabyte file to write it in one go.

  uniquely per platform.
- `cache` / `cache_dir` / `max_parallel` / `prefetch` / `chunk_mb` / `persist`: identical to
  `hf_read`.

#### How the cache is stored
The unit of storage is a **chunk**, matching the unit of download: in the browser each chunk
is its own IndexedDB record, written as it arrives. Three consequences worth knowing:

- **Memory does not grow with file size.** A 13 GB model costs the same live bytes as a
  400 MB one, because no object accumulates the file. (Pyodide's IDBFS is *not* used for
  this: it keeps a file's whole contents in the wasm heap and only copies them to IndexedDB
  on `syncfs`, so a multi-GB download would be buffered entirely in memory.)
- **An interrupted download resumes.** The set of stored chunks *is* the coverage map, so a
  reload, a closed tab or a dropped connection costs only the chunks that were in flight.
- **Nothing is rewritten** as the download grows, and there is no flush step.

What this bounds is growth with file size, not absolute footprint. The live set is roughly
`(max_parallel + chunks spanned by the current read) × chunk`, so the defaults (8 × 16 MB)
peak near 144 MB; lower `max_parallel` or `chunk_mb` to trade throughput for footprint. It
is not a hard cap — the wasm heap never shrinks, and asking for a whole multi-GB file in one
`read` still returns one.

The file's total length is often unknowable in a browser: `Content-Length` and
`Content-Range` are not CORS-safelisted, and some hosts (ModelScope among them) do not
expose them. That does not disable caching — chunk indices never needed the total, and the
size is learnt from the first short chunk.

#### Managing the persistent cache
Because the cache persists, it needs housekeeping — list / inspect / read / write / delete /
clear it, so it never becomes dead data. All are **async**. Every entry's **key is the URL
without scheme, so its first path segment is the host/domain** — HF (`huggingface.co/…`) and
ModelScope (`modelscope.cn/…`) entries are separated and can be listed or cleared per host.

- `webtorch.default_cache_dir() -> str` — the dir the readers use by default
  (`$WEBTORCH_CACHE` or `~/.cache/webtorch/hub`). All functions below take `cache_dir=None`
  meaning this default.
- `await webtorch.list_cache(cache_dir=None, host=None) -> [ {"key","host","size","complete","total","path"} ]`
  — every cached entry, sorted by key; `host="huggingface.co"` filters to one domain.
  `size` is what is actually stored, so a partly-downloaded entry reports its real extent;
  `total` is the file's full length once known (`None` if the host never revealed it), and
  `complete` says whether every chunk is present.
- `await webtorch.cache_hosts(cache_dir=None) -> [ {"host","files","size"} ]` — per-domain
  summary (largest first), so you can see HF vs ModelScope usage at a glance.
- `await webtorch.cache_size(cache_dir=None, host=None) -> int` — total bytes cached
  (optionally for one host).
- `await webtorch.read_cache(key, offset=0, length=None, cache_dir=None) -> bytes | None` —
  read a cached entry's bytes (a `list_cache` key, or a full URL); `None` if not cached.
- `await webtorch.read_cache(...)` reads only the chunks the range touches, so it stays cheap
  on a multi-GB entry. It returns `None` if the range crosses a gap in a partial entry.
- `await webtorch.write_cache(key, data, cache_dir=None, complete=True) -> None` — write /
  replace an entry (pre-seed the cache), stored chunk by chunk; `complete=True` marks it
  fully cached so a reader serves it without touching the network. Replaces any existing
  entry rather than merging with its chunks.
- `await webtorch.delete_cache(key, cache_dir=None) -> bool` — delete one entry: every chunk
  plus its metadata; `True` if it existed.
- `await webtorch.clear_cache(cache_dir=None, host=None) -> int` — delete everything (or only
  one `host`'s entries); returns the number of entries removed.

```python
for h in await webtorch.cache_hosts():          # e.g. [{"host":"huggingface.co","files":9,"size":9_999_999}, …]
    print(h["host"], h["files"], h["size"])
await webtorch.clear_cache(host="modelscope.cn")  # free just the ModelScope cache
```

Note: these act on storage; a reader object created earlier may still hold chunks it already
read — clear the cache between loads, or before creating the reader.

A complete, runnable demo of a cache-backed read callback (with `HttpError` rate-limit handling) and
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
  `max_parallel` — see the box below. `cache=True` disables caching (pure streaming);
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

## Where the bytes live — local files and directories

Origin storage is capped by browser **policy**, not by the machine: the same page is offered
4.46 GB in one browser and 11.5 GB in another while the disk has hundreds of GB free, and
`persist()` is refused in both. A model larger than that number cannot be kept there at all.
Two ways out, and neither copies the model:

**Read it where it lies.** The person picks a file; it is registered, not imported.

- `webtorch.use_model_file(handle, name=None)` — serve reads for a model out of a local
  file. `handle` is a `FileSystemFileHandle`. Matching is by file name, because a model's
  key ends in the name the file has (`…/resolve/master/model.gguf` against `model.gguf`),
  so a file picked from disk satisfies the very id the loader was about to fetch. Pass
  `name` to override that.
- `await webtorch.import_model(handle, name=None)` — the same idea for either kind of
  handle: a `FileSystemFileHandle` registers one file, a `FileSystemDirectoryHandle`
  registers every file in it, which is what a multi-file model is. Nothing touches origin
  storage; reads go to the offsets the loader asks for.
- `webtorch.local_files()` → the names currently satisfied this way, in registration order.
- `webtorch.forget_model_file(name)` — stop serving `name` from a local file.

**Or keep the cache in a directory.** A directory carries no quota: what fits is what fits
on the disk, and what lands there is the file itself — openable by other tools.

- `webtorch.use_directory(handle)` — cache into a directory the person picked instead of
  origin storage. `webtorch.get_directory()` returns the handle in use, or `None`.
- `await webtorch.migrate_cache(handle, cache_dir=None, on_progress=None, keep=False)` —
  move what is already cached into that directory and keep using it. Entry by entry, chunk
  by chunk, **deleting from origin storage as each is safely written** — which is the point:
  when the quota is already full, freeing as you go is the only way there is room to
  continue. Memory holds one chunk, never a model. `keep=True` copies instead of moving.
  Returns the bytes moved.

**Taking a model elsewhere.**

- `await webtorch.model_groups(cache_dir=None)` → cached entries grouped into models,
  `[{"name", "label", "keys", "size", "files"}]`. Grouping is by the directory part of the
  key, which is what separates one repo's files from another's.
- `await webtorch.export_model(keys, write, cache_dir=None, on_progress=None)` — write
  cached entries out through `write` (async, called with `bytes`). One key is written as the
  file itself; several become a stored ZIP. Returns bytes written.

## Progress and storage feedback

Each installs one callback and returns nothing; passing `None` clears it. The paired getter
returns whatever is installed. `info` is a dict in every case, so it can gain fields without
breaking callers.

- `webtorch.set_read_progress(cb)` / `webtorch.get_read_progress()` — how far a **load** has
  got, called as reads are served. `info`: `key`, `done` (cumulative bytes for that entry
  since the hook was installed), `total` (or `None` when unknown), `elapsed`, `rate`
  (bytes/second, smoothed).
- `webtorch.set_download_progress(cb)` / `webtorch.get_download_progress()` — the **network**
  underneath, for anything going through the built-in `http_get`, which is what `hf_read`,
  `modelscope_read` and `use_default_io` are built on. `info`: `url`, `bytes`, `rate`. A
  read callback of your own that does not use `http_get` will not report here, and should
  not — it may not be downloading anything.
- `webtorch.set_storage_full(cb)` — called **once**, when origin storage runs out. `info`:
  `key` (the entry being written when it hit the wall), `quota` (what the browser said the
  origin may use, if it said). This is not an error and is not raised: the load continues by
  streaming. It is the moment to offer the person a directory — see `use_directory`.

## Stopping a load or a generation

- `webtorch.cancel(flag=True)` — ask the in-flight load, read or generation to stop.
  `cancel(False)` withdraws the request. The flag is **sticky**: it stays set, keeping stray
  IO from starting, until it is withdrawn, which every load does on its way in.
- `webtorch.Cancelled` — raised at the next IO checkpoint once a stop is asked for; after
  that the SDK issues no further read/write callbacks. It derives from `BaseException`, the
  way `asyncio.CancelledError` does, because loaders are full of broad `except Exception`
  fallbacks (optional files, retry loops, rate-limit gates) and **none of them may swallow a
  stop**.
- `webtorch.cancel_requested()` → whether a stop has been asked for. A decode loop reads the
  flag rather than being raised at: a generation is not IO, has no checkpoint to raise at,
  and should hand back the tokens it already has instead of losing them to an exception.
- `webtorch.set_cancel_probe(fn)` — install `fn()` as a second source of "stop was asked
  for". A stop has to be able to arrive while the interpreter is **busy**: in the browser the
  decode loop is a plain Python loop inside one `runPythonAsync` call, so a stop sent as a
  message is not queued behind the work, it is **not received at all** until the work ends,
  and the button reads as dead. The flag then has to come from somewhere the caller can
  write without the interpreter's help — a `SharedArrayBuffer` the page stores into — and
  this is where the SDK reads it. Called between tokens and at every IO checkpoint, so it
  must be cheap.

**Cleaning up after a stop.** A stopped download leaves whole chunks plus, where it stopped,
one that is only part-written. Part-written chunks are not *wrong* — `covered` records byte
for byte what is there and a read never asks outside it — but nothing can be served from them
until the rest arrives, which is what makes a stopped load look like a cache entry that is
"there" without being usable.

- `await webtorch.trim_partial(key, cache_dir=None)` → bytes dropped. Whole chunks are kept,
  so a stopped load still resumes from where it got to; only the ragged edge goes.
- `await webtorch.trim_stopped(cache_dir=None)` → bytes dropped, for every entry the load
  that just stopped had written. The stop itself cannot do this — `cancel` is called from
  wherever the person pressed the button, often mid-chunk — so the loader's caller calls this
  once the `Cancelled` has come back to it.

## Advanced: concrete model impls  (`webtorch.models.*`, NOT the public interface)
Reach these only if you need model-internal control; the generic `pipeline` / `AutoModel*`
cover normal use. All loaders accept url | bytes | async callback; str paths read through
the global `io_read` (per-call `io=` overrides it).
- `webtorch.models.cosyvoice.CosyVoice2TTS.from_npz(flow, hift, baked)` + `.load_llm(...)` + `.load_clone(...)`
- `webtorch.models.llm.CausalLM.from_gptq/from_gguf`, `webtorch.models.tts.VitsTTS.from_npz`,
  `webtorch.models.detection.{DetrDetector,YoloDetector}.from_npz`, `webtorch.models.vl.QwenVL.from_pretrained`.
