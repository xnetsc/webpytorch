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
| `llm.py` | `CausalLM` — loads AutoGPTQ (int4/int8), GGUF, or plain fp16/bf16 HF; prefill/decode, KV prefix reuse, chat templates and the tool-call API; `BPETokenizer` (which also *reads* the model's own template to learn its tool syntax) |
| `constrain.py` | output constraints — `Verdict`, the callback protocol, and the built-ins (`json`, `regex`, `choices`, tool names) |
| `toolcall.py` | pure functions for reading/writing tool calls in whatever delimiters and shape a model uses; knows no model family by name |
| `ggufload.py` `hfcompat.py` `iqtables.py` | weight readers: GGUF (28 formats incl. i-quants) and HF/safetensors; `iqtables` is the i-quant codebook data |
| `multimodal.py` | `register_encoder` + `MultimodalLM` — pairs any decoder with any media encoder |
| `linear_attn.py` | SSM / linear-attention layers (the hybrid models) |
| `backend.py` `webenv.py` `portable.py` | backend selection, browser-vs-host environment, and the numpy fallback that lets pure-Python logic be tested off-browser |
| `_wgsl2glsl.py` | translates the WGSL kernels to GLSL so WebGL gets the same kernels rather than a second implementation |
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

## How a token gets produced

Prefill and decode look like the same arithmetic and are not the same problem, so they do
not share a path. Decode is one row against every weight: nothing to amortise, latency is
everything. Prefill is hundreds or thousands of rows at once: enough work that the *shape*
of the work decides the time.

**Decode** runs the quantised matmul directly — the weights are never unpacked, because
with a single row the unpacking would cost more than the multiply. The whole step is
captured as a WebGPU command graph on the first token and replayed afterwards, so per-token
CPU work is a handful of buffer writes rather than a re-record of every dispatch. Attention
uses a split-K GQA kernel; how far to split is measured at load, not assumed.

Its cost splits in two, and only one half is about the model. Measured on a 0.6B: 9.05 ms
that does not depend on the conversation, plus 0.00223 ms for every token already in it.
The second term is **reading the cache**, and nothing else — 28 layers × 2 × 8 kv-heads ×
128 dims is 224 KB per context token in fp32, so at a context of 3632 one step streams
833 MB, in 8.11 ms. That is 102.7 GB/s, which is *faster* than anything else here reaches
(the general fp32 matmul streams 62–79 GB/s on the same device) and the split factor is
already chosen by measurement from 4/8/16/32 at every context bucket. There was no kernel
left to improve, so the bytes had to go: **K and V are stored as halves**, two to a `u32`,
unpacked in the shader with `unpack2x16float` — core WGSL needing no device feature, and
`unpackHalf2x16` is the GLSL spelling of the same thing. Measured end to end on the 0.6B at
greedy: 101.2 → 125.2 tok/s at short context and 67.3 → 83.8 at a context of 2848, with the
generated text identical in both, token for token. Halving the bytes does not halve the
time (the packed kernel sustains 66.6 GB/s against 88.0), so the win is ~1.5× on the
attention term rather than 2×.

Which kernel reads which cache is decided from the **buffer**, never from a flag: a packed
cache is exactly half as wide as the query scanned against it, and there is more than one
KV cache class in this file. The two attention kernels that scan the cache are long and
subtle, so the packed pair is *derived* from the fp32 pair by substitution rather than
copied — every fragment asserts, so one that stops matching is an error and not a kernel
quietly reading the wrong bytes. Prefill is unaffected: it already copied the span it
attends over, so that copy widens back to fp32 and the three prefill kernels never learn
about any of this.

**Prefill** does the opposite twice over:

- Above `_GGML_DEQ_M` rows the weights *are* unpacked, once, and the plain fp32 matmul runs
  on them. Measured on this machine that kernel sustains 2117 GFLOPS against the quantised
  kernel's 426 — the quantised one is not bandwidth-bound, it is bound by unpacking the same
  weights again for every row. Above a few dozen rows, paying once is 4.8× cheaper.
- Attention materialises the score matrix `_ATTN_CHUNK` queries at a time (above
  `_ATTN_CHUNK_MIN_T` tokens) instead of streaming it. A flash kernel avoids writing the
  scores down but runs at 94 GFLOPS here; the chunked form spends more memory traffic in
  order to spend its arithmetic in the 2117-GFLOPS kernel, and wins by 3.6×. Below that
  threshold flash still wins — it is one dispatch against dozens, and at short lengths the
  dispatches *are* the cost — so both kernels stay, chosen by length.

**Alignment is load-bearing, not a detail.** The backend's fp32 matmul falls off a cliff
when the row count is not a multiple of `_MATMUL_ROW_ALIGN` (32) or the key extent not a
multiple of 64. So prefill pads its token sequence to 32 *once* and takes the last real row
back, and chunked attention rounds each chunk's key extent up to 64. Padding per call
instead — `xp[:m] = xf` — is the trap: `__setitem__` goes through the host at 0.8 GB/s and
cost 3.4 s, more than the cliff it was fixing.

**Nothing above is hardcoded on faith.** `tune(key, candidates, apply, bench, check)` runs
the real kernel over the candidates at load time and keeps what measured fastest, per shape
(`_warm_shapes`). That phase runs every distinct `(format, N, K)` at **two row counts**, not
one: a shader compiles on its first dispatch rather than when it is registered, and the
prefill path branches on the row count — one row reaches the decode GEMV, three the batched
kernel. Warming only the first left the batched kernel to compile in front of the reader.
The third branch, at `_GGML_DEQ_M` rows, unpacks the weights to fp32 and is deliberately
**not** warmed: building it here would also materialise an unpacked copy per shape during
the load, and on a machine the model already fills, making room for those is what pushes the
weights back out.

Feeding the weights one at a time is not the same as running a step, so a **whole decode
step** runs at the end of `_init_state` as well (`_warm_decode_step`), outside any recording.
Whatever a first `_decode_fwd` does that a later one does not, it must not do it while the
decode graph is being recorded: measured on a dense 27B, the first recording of that graph
held 3477 dispatches against 1749 for the same graph recorded again mid-reply, and replayed
the difference for every token until it was replaced. Two rules were learned the hard way and are enforced in `bench`: batch
24 dispatches per sync, or you measure the 1–2 ms readback instead of the kernel; and
interleave the candidates, because the same configuration measured 7.61 ms and 4.49 ms in
one session when run in blocks. Where measurement said a knob does not pay (`_GGML_KSG`,
widening `_SMALL_N`) it is *not* made dynamic, and the negative result is recorded next to
the constant so it is not rediscovered.

**KV reuse.** A reply keeps its cache; the next turn re-uses the longest common prefix of
token ids rather than re-reading the prompt. The invariant that makes it safe: the cache's
id list is committed in a `finally`, so an aborted or errored generation leaves the recorded
ids matching the tensor. Committing only on the happy path is what produced a cache that
claimed a prefix it did not contain — and the symptom was not a crash but fluent, degenerate
output.

## Where a decision belongs

The SDK is an abstraction over a *class* of needs, not a set of features for the chat app.
The line is drawn like this:

- **Model internals never reach the caller.** Probing a template for its tool syntax,
  rendering a call, constraining a decode — each is an API or a parameter to one, not
  something a caller is expected to assemble.
- **Parameters speak the business's language**, not the implementation's:
  `require_known_tools=True`, not `constraint="tool_names"`.
- **The SDK offers, it does not install.** Defaults do not make a business decision on the
  caller's behalf.
- **Application-specific behaviour stays in the application.** Which tools exist, and what
  closing a tab means, are the chat app's; parsing and constraining them are the SDK's.

Everything functional lives in `webtorch/`. Outside it there is UI and orchestration only —
`chat/worker.js` marshals calls, `chat/app.js` renders. The chat app is a user of this SDK,
and holds no capability of its own.
