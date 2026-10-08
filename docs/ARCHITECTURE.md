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
| `ggufload.py` `hfcompat.py` `iqtables.py` | weight readers: GGUF (28 formats incl. i-quants; header read incrementally, metadata arrays decoded on access) and HF/safetensors; `iqtables` is the i-quant codebook data |
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

**Greedy decode keeps the GPU busy while the host handles tokens.** For a plain greedy call
on a model whose tied embedding/head is Q6_K, a chunk of 1, 2 or 4 steps is recorded with
nothing per step from the host: a step reads its token from the slot the previous step's
argmax wrote, advances the position counter `ctl[0]` on the device, and takes its rotary rows
from a per-position table covering the cache. So chunk n+1 is queued before chunk n has been
read. Each round is one call into JS, `replayStaged`: queue the next chunk and a *staged
read* of its token slots — a copy recorded behind the queued work, mapped when the GPU gets
there, delivered to a shared-memory slot — then collect the oldest staged read. The host's
per-token work (accepting, detokenising, streaming) runs while the GPU computes the next
chunk; before, the GPU sat through the readback, that work and the next chunk's uploads.
0.6B, same session, 200 tokens: 138–145 tok/s host-fed against 175–181 pipelined (chunk 1),
text identical token for token. `_greedy_chunks` never queues past the caller's token budget
or the cache (which grows only with nothing in flight), and a caller that stops early leaves
at most one chunk running, whose rows lie past everything the cache records as held. The
chunk size is raced at load through the same driver.

Printing a decode step by kernel name is how you find work that is not work. On a 0.6B it
showed `out0 = in0` — an identity copy — 28 times, one per layer, from asking for a layout
the tensor already had (see `_attn_out`); `reshape` could not tell, because an axis of
length 1 makes the strides say non-contiguous when the memory is not. That one is removed
because it computes nothing, which is reason enough.

It is NOT removed on the theory that the step is bound by its command count. That theory
was tested here and failed: fusing a MoE layer's gate and up into one weight and one
dispatch moved the step from 35.62 ms to 35.57 (see `llm.py`). A 0.6B step reads 0.4 GB in
9.7 ms, which is ~41 GB/s against a device that reaches ~100 — so the headroom is in what
the quantised GEMV does per value, not in how many times it is asked.

Its cost splits in two, and only one half is about the model. Measured on a 0.6B: 9.05 ms
that does not depend on the conversation, plus 0.00223 ms for every token already in it.
The second term is **reading the cache**, and nothing else — 28 layers × 2 × 8 kv-heads ×
128 dims is 224 KB per context token in fp32, so at a context of 3632 one step streams
833 MB, in 8.11 ms. That is 102.7 GB/s, which is *faster* than anything else here reaches
(the general fp32 matmul streams 62–79 GB/s on the same device) and the split factor is
already chosen by measurement from 4/8/16/32 at every context bucket. Not splitting was
offered as a fifth candidate and measured worse even on a 0.6B at a 66-token context — 563
dispatches at 11.40 ms against 591 at 9.89 — so splitting earns the dispatch it costs. There was no kernel
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
  on them — but only for formats `ggml_dequant_ok` allows. Every i-quant is excluded, which
  is 84% of a 27B's elements, so for that model this is not the prefill path at all.
- A same-width route, `"tiled"`, decodes each block once per 32-row workgroup into
  workgroup memory -- the integer values as halves, which are exact -- and multiplies from
  there, the block's scale multiplying a per-block partial sum. It reads the stored buffer
  and allocates nothing, and `ggml_matmul(execution="auto")` races it against the others per
  format, shape and row bucket. Q8_0 has it: 519×768×2304 takes 0.86 ms against 1.19
  materialized, 2.38 stored and 0.74 for the same product on half weights, and the Q8
  xDecision encoder went 117.3 → 103.8 ms with unchanged answers. Q4_0, Q4_1, Q5_0, Q5_1,
  Q4_K, Q5_K, Q6_K, Q3_K and Q2_K share one template (`_TILED_TEMPLATE`): each is A·q′ − B
  per sub-block with q′ exact in a half, so a format supplies only how to read q′, A and B.
  Where the device has `shader-f16` the same template also yields `"tiled_half"`
  (`_ggml_tiled_half_src`): the stage holds A·q′ as halves (the dequantised weight, so a half
  sum never sees q′ times an activation), the activations are staged once as halves, a
  32-deep stage is summed in half and added into f32. A phase-two cross-width route
  (`_PHASE2_CROSS_WIDTH`), raced with a 1e-2 output gate: ~1e-3 of the output scale,
  1.4–1.6× the f32 kernel at 0.6B prefill shapes; a 161-token first token 112 → 94 ms.
  The half sum's error depends on the input. On a 27B's 6144×5120 Q4_K weight it was 0.3%
  to 1.6% of the output scale from one input to the next, so a gate run on whichever input
  came first admitted it at load and refused it at a remeasure. An approximation
  (`_APPROXIMATE`: `tiled_half`, `dp4a`) is now admitted per weight only if it also holds
  its bound on two fixed inputs that are hard for it: a few large values among small ones,
  and a few very large ones among ordinary ones (`_approx_holds`, once per weight). One
  that misses its bound is left out of the race (`_GATE_EXCLUDED`) rather than failing it;
  an exact candidate that misses still raises. Summing in f32 instead keeps the error at
  0.02–0.16% but gives back most of the speed (0–30% over the f32 kernel against 30–120%),
  so the half sum stays where it holds. The profile's build stamp covers these gates.
- The quantised kernel it falls back to was bound by decoding the same weight again for
  every output row. One thread owned four rows, so a decoded value fed four multiplies; it
  now owns twelve (`_GGML_MROW`), and the per-row guard that put eleven branches in the
  innermost loop is gone because the rows a partial group lacks are staged as zero anyway.
  A whole 28-layer prefill forced onto this kernel went 1290.8 → 670.8 ms at T=512 and
  4296.4 → 2366.0 at T=1536; per format it is 1.2× to 1.9×, biggest where the decode is
  most expensive. Unpacking still wins where it is allowed, by 1.5–2.0× rather than 5×.
- Attention, where the cache is packed halves (WebGPU), is `causal_attention_cache`: one
  kernel reads q, keys and values where they lie in the cache and writes the
  out-projection's rows. Only the probability tile goes through workgroup memory; a
  short prompt after a long conversation splits its keys over workgroups and merges the
  softmax states after, how far measured per device by context length. Per layer of a
  16-head, head_dim-128 model it took 1.41 → 0.31 ms at 182 tokens, 8.1 → 0.70 ms for 64 new
  tokens after 2000, 9.5 → 2.6 ms at 1024 — against widening the span to f32, the flash or
  chunked kernel and a transpose back. Those remain the path elsewhere: the score matrix
  `_ATTN_CHUNK` queries at a time above `_ATTN_CHUNK_MIN_T` tokens (a flash kernel avoids
  writing the scores down but runs at 94 GFLOPS here; the chunked form spends memory
  traffic to spend its arithmetic in the 2117-GFLOPS kernel), flash below it.

- A routed (MoE) layer in a prefill used to run every (token, slot) pair as its own GEMV
  against its expert, reading an expert's weights once per token routed to it — 1984 reads
  where 128 would do on a 30B at 248 tokens, 1.33 of a 1.62 s prefill. `moe_group` now
  groups the slots by expert on the device (counts, starts, a permutation and a table of
  (expert, 32-row tile), one workgroup, no readback) and `_ggml_tiled_moe_src` re-points the
  tiled kernel at it: one workgroup per (expert, tile) x 64 columns, the expert's weights at
  its stride, the rows read through the permutation (a token's row for gate/up, so the
  activations are no longer copied k times; a slot's row for down), each written to its
  slot. `GGMLMoELinear.forward_routed` races per slot over GEMV, grouped and grouped in half,
  laddered at load from 16 to 4096 slots: the slot GEMVs keep small prompts, the grouped
  kernels take the rest. Qwen3-30B-A3B: first token at 157 tokens 1.71 → 0.67 s.

**Alignment is load-bearing, not a detail.** The backend's fp32 matmul falls off a cliff
when the row count is not a multiple of `_MATMUL_ROW_ALIGN` (32) or the key extent not a
multiple of 64. So prefill pads its token sequence to 32 *once* and takes the last real row
back, and chunked attention rounds each chunk's key extent up to 64. Padding per call
instead — `xp[:m] = xf` — is the trap: `__setitem__` goes through the host at 0.8 GB/s and
cost 3.4 s, more than the cliff it was fixing.

**WebGL Q8_0 is split, not decoded per word.** On WebGL a Q8_0 linear that fits holds its
int8 four to an RGBA8UI texel and its scales as an R16F texture (`WebGLQ8Matrix`), the same
bytes as the file. The matmul packs activations four to a texel and does a block's eight
`dot`s before one scale multiply — 3–4.5× the word-layout kernel from 64 rows up; the Q8
decision model went 2264 → 835 ms on WebGL, level with F16.

**Races happen once, after a load — never inside an answer.** A route race
(`_weight_execution`) for a new row bucket used to run the first time an answer needed it:
461 ms of a 526 ms first request on the Q8 decision model. Now a model calibrates right
after its weights arrive: `calibrate_rows` measures every race a weight can set off over a
row-count ladder bisected in octaves (both ends, then the middle of any interval whose ends
disagree), and from then on a row count no probe visited takes the nearest probe's measured
choice. Only the ladder's cheap end runs inside the load; the rest (the top probe, where the
stored kernel at 512 rows is most of the cost, and any bisection) is queued and advanced by
the host a step at a time while no call is running (`calibrate_deferred`), answers borrowing
the nearest measured bucket meanwhile. What is measured is per device and kept by the SDK
(IndexedDB, keyed by adapter, dropped when the kernels change), so a second load measures
nothing. Decision models
ladder every stored linear and the head's row selection; LLMs ladder the weights their
layers hold (16→512 rows) — not the output head, which a prefill reads for one row.

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
one session when run in blocks. The first rule does NOT carry over to `gqa_tune`: applied
there it chose a configuration 14% slower in a real step, because 24 identical attention
dispatches back to back are not a decode step, where each sits between the layer's other
work. Where measurement said a knob does not pay (`_GGML_KSG`,
widening `_SMALL_N`) it is *not* made dynamic, and the negative result is recorded next to
the constant so it is not rediscovered.

**What a race costs is the GPU work it times**, so measurements that cannot change the
answer are not taken (`_race`): from the third round, a candidate all of whose samples are
slower than all of the current fastest's is out — complete separation, three against three,
which two equal candidates produce one time in twenty and a genuinely faster one essentially
never — and a race whose winner has a repeatable paired win over every survivor stops. A
candidate is warmed with one run (it only has to build and prove it runs), not a timed batch,
and a variant raced by name is self-checked at the small coverage shape rather than at the
model's own N × K. A 27B's load: warming 32 → 17 s, its row ladder 11.5 → 6.7 s, checks
4.6 → 0.4 s; the replies of the 27B, the 30B MoE and the decision model came out identical.

A hybrid model's recurrent state is **zeroed where it lives**: a reset clears the device
buffers it already has (WebGPU's `clearBuffer`, WebGL's `clearBufferfv` on the texture), and
a fresh state is made as device zeros, so a reply starts without sending the host's zeros —
3 MB a layer, each upload waiting for everything queued ahead of it. 27B first token
5.7–6.5 → 4.5–5.0 s. The layers' constants go up at load, while nothing else is queued.

**KV reuse.** A reply keeps its cache; the next turn re-uses the longest common prefix of
token ids rather than re-reading the prompt. The invariant that makes it safe: the cache's
id list is committed in a `finally`, so an aborted or errored generation leaves the recorded
ids matching the tensor. Committing only on the happy path is what produced a cache that
claimed a prefix it did not contain — and the symptom was not a crash but fluent, degenerate
output.

## Adapters on stored weights

A PEFT LoRA adapter is read from PEFT's own files and placed by the Hugging Face module path it
was trained on (`adapters.py`, `CausalLM.attach_adapter`). It stays beside each weight as two
float32 matrices and adds `scale*(x A^T) B^T` to that projection's output; it is never merged,
because merging would decode a quantized weight and store it at a width the file does not have.
A projection without one (`lora` None) runs exactly as before. The paths that read several
stored weights in one dispatch (q/k/v, gate/up) keep their route and add each adapter to its
own output afterwards. On a decode row the adapter can be one dispatch (`_lora_fused`): a
workgroup reduces t = A x in workgroup memory and adds `t . B^T` to the outputs it owns,
instead of two matmuls and an add, the first of which no tiled shape fits (r outputs, each a
K-long dot product walked by one thread). Which of the composed and fused routes runs is raced
per adapter shape while the model warms; prefill keeps the composed GEMMs. Linear-attention
value heads are stored tiled in a GGUF, so an adapter's value rows (and out_proj's columns) are
moved to that order when attached. An adapter is attached during the load, before warm-up, so
route races and recorded decode steps include it; attaching later drops the model's own
recordings. A slot-head decision model is the same language model with its adapter plus a
linear head over the last hidden state (`decision.SlotDecisionModel`).

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

Everything functional lives in `webtorch/`, and that now includes crossing into it. The SDK
runs as Python in a worker, so every call used to cross a message boundary the host had to
build: `chat/worker.js` was 690 lines of which about 400 were marshalling that no application
should be writing — `toolScan` was `m.strip_tool_calls` plus `m.tool_calls`, `cacheList` was
three documented calls and a dict comprehension. `webtorch.start()` creates the worker and
returns functions, and that file is gone.

What is left outside is what an application really owns. `chat/app.js` renders. `chat/cache-sw.js`
says which files this page keeps and for how long — it rides inside the SDK's service worker,
because a client has exactly one controller and on a static host that one has to be the SDK's:
only a worker controlling the document can put the isolation headers on the document's own
response, and without them the whole model runs on the CPU.

The line between them is drawn by one question — **can the application build this itself?**
It can build a service worker, a scope and a caching policy; it cannot work out that COEP has
to be `require-corp` rather than `credentialless` (a measurement on WebKit, where the wrong
one is ignored and every browser on an iPhone lands on the CPU with nothing said). So the
worker is the SDK's and the caching is the page's. By the same test the SDK decides nothing
about storage: caching models, and keeping what a GPU measured, are both off until asked for.

## Making it faster: what this backend is actually bound by

Six attempts to speed up one encoder pass, all measured end to end on the same machine. The
first five are below; the sixth has a section of its own. Four of them reduced something real
and made no difference or made it worse. They are
recorded because the reasoning behind each one was sound and still wrong, and the next
person to have the idea should get the number rather than the afternoon.

| attempt | what it reduced | result |
|---|---|---|
| fuse `gelu` into one dispatch | 1577 → 1187 dispatches | 280 → 271 ms |
| pad K/V so attention's matmuls align | 896 dispatches onto a kernel 2× faster in isolation | first token 6.4 → 7.5 s, **worse** |
| RGBA textures in the WebGL matmul | 32 → 5 texture fetches per 16 multiply-adds | 9.5 → 13.8 ms, **worse** |
| int8 / int4 weights | 4–8× fewer bytes read | 0.42–0.71× as fast, **worse** |
| **the `m % 32` gate on the tiled matmul** | **nothing** | **131 ms, 6× on the matmuls** |
| WebGL LayerNorm/softmax: row statistics once per row, not per element | 1,536 → ~3 fetches per LayerNorm output | decision request 2230 → 1773 ms |
| WebGL scalar matmul: step the LHS texel address instead of dividing | one integer division per multiply-add | 2232 → 2466 ms, **worse** |
| WebGL dense matmul: four K-values per RGBA texel and `dot` (at 519 rows, not 69) | 2 → 0.5 fetches per multiply-add | 1.6–1.9× per matmul; decision request 1773 → 856 ms |

The one that worked did not reduce anything. There were already two matmul kernels, and the
fast one required the row count to be a multiple of 32 — which is the token count, the one
axis a caller does not choose. Every matmul in an encoder pass was falling to the naive
kernel because of the shape of a check. Letting the tiled kernel take any row count is the
whole of it.

What the failures have in common is that each reduced the thing it assumed was the bound.
At the sizes a decision model works at — 69 rows against a 1024-wide hidden state — this
backend is short of **parallel work**, not of bandwidth:

- The same kernel on the same weights measures 234 GFLOPS at 69 rows and 1573 at 800. It is
  not the weights that are slow, it is having too few rows to fill the device.
- Halving the bytes by holding weights at half precision is worth 1.33×, not the 2× the
  arithmetic suggests — and nothing at all on the narrowest shape.
- Quantising further makes it *slower*, because the dequantisation is arithmetic, and
  arithmetic is what is already scarce. Half precision is the exception only because
  `unpack2x16float` is one instruction where unpacking int4 is a shift, a mask, a multiply
  and a subtract.
- Fewer texture fetches was never less traffic: consecutive fragments reading consecutive
  elements already share cache lines.
- Dispatch count is not time. The note in `layernorm` saying so was right, and removing 390
  dispatches to save 8 ms is what finally made that believable.

Quantisation remains necessary for large models, but as a **capacity** decision and not a
speed one: a 27B model does not fit otherwise. A 421M model fits either way, so nothing is
buying the slowdown.

A multi-question decision request is also not one encoder sequence. Each question and its
options are placed beside the state before bidirectional attention, so the state cannot be
encoded once and appended to unrelated questions afterward. The runtime reports the resulting
per-question lengths and encoder pass count in `usage`; exact duplicate sequences reuse one
pass, while short compatible sequences may be batched. A decision batch stays on the GPU
through sequence extraction and head scoring. Reading its full `(batch × padded length ×
hidden)` result back to the host and uploading each question again erased most of the batch
gain; only the final logits cross that boundary now.

### The sixth attempt: stop issuing the pass at all

The same reasoning one step further. If the device is short of work and not of bandwidth, then
the cost is the host: measured on one 82-token pass of a 22-layer 768-wide encoder, **36.2 ms
of it is Python issuing 550 dispatches and 0.7 ms is waiting for the GPU.** So the pass is
recorded once as a command graph and re-issued with one call afterwards, with the inputs
written into the buffers the recording bound. Dispatches per pass: 485 → 0.

What makes that hard is that a recording is only valid for the shape it saw, and decision
sequences are never the same length twice — 24 real ones gave 14 distinct lengths. Keyed by
the exact length it is not merely useless, it is **negative**: the handful of retained slots
fill up with lengths that occur once, each having paid to record a pass that is never
replayed. Lengths are therefore rounded up to a multiple of 32 and the answer trimmed back.
Median of three runs over those 24 calls:

| keyed by | total | warm half | slots used |
|---|---|---|---|
| the exact length | 1451 ms | 692 ms | 4, all cold |
| nothing (no capture) | 1131 ms | 553 ms | — |
| a 32 bucket | **1024 ms** | **440 ms** | 3 |
| a 64 bucket | 1121 ms | 471 ms | 2 |

A 64 bucket is worth about nothing, because a fifth of the positions it computes are padding
and attention is quadratic in them. 32 is also the multiple the tiled matmul wanted in the
row above, which is not a coincidence — it is the same axis.

Two things are worth being plain about. **Padding does not change the result**: measured at
105 to 225 tokens, padding to a bucket and trimming back leaves the hidden states
bit-identical, difference 0.0 at every length against values of magnitude 42. What does change
it is handing the padded *rows* onward, because a decision head's attention has no mask and
reads them — 0.012 on its logits, and the reason the rounding was once removed as if padding
were unsound. And **1.26× is the whole of it**, not the 1.44× one captured length suggests:
that figure was measured where the sequence already fitted its bucket. The host cost is real
and this removes most of it; what is left is arithmetic, and 4× the length still costs 3.94×
the time.

One caveat on the last row: the quantised path has a dedicated GEMV kernel for a single row,
which is the decode case. All of the above is measured at 69 rows, on the general path.

### The seventh: once the pass is fed, what the GPU itself is bound by

With the pass replayed, the request is GPU time plus whatever the host does around it. On the
three-question request (3 × 173 padded tokens, a 22-layer 768-wide encoder, F16 GGUF) that was
100.2 ms, of which 82.2 was GPU and 49.0 of that the dense matmuls. An MLX reference on the
same input takes 19.1 ms in f16 and 51.5 in f32. What closed most of the distance, in order:

| change | GPU time it took | request |
|---|---|---|
| attention as two dispatches (`rope_qk`, `fused_attention`) | 21.0 → 4.5 ms | 100.2 → 79.2 ms |
| LayerNorm a workgroup per row, the residual add folded in (`add_layernorm`) | 8.8 → 2.3 ms | 79.2 → 75.5 ms |
| the per-request scratch release moved after the answer, to idle time | — (host) | 75.5 → 65.1 ms |
| dense half weights race half arithmetic (`_mm_half_src`) against f32 | 47.3 → 33.9 ms | 65.1 → 51.5 ms |
| the questions end to end instead of padded to the longest (`_replayed_packed`) | 44.1 → 39.7 ms | 49.8 → 44.5 ms (same-session A/B) |

Answers were bit-identical through the first three. The last one changes them in the third
or fourth decimal (billing 0.8936 → 0.8935, duplicate charge 0.8755 → 0.8794, urgency 1.7199
→ 1.7194) — the same order as MLX's own f16-against-f32 difference on this input (0, 0.0020,
0.0012). That is what computing at the weights' width costs here, and why it is a raced
*candidate* gated by an output check, not the only path.

What sets the ceiling is the arithmetic the device exposes to standard WebGPU, measured with
timestamp queries on an Apple M5 (Chrome 154, no flags): **3.2 TFLOPS f32 and 6.2 f16 FMA**,
and no dual issue — f32 and f16 work mixed in one kernel take the sum of their times, and so
does integer arithmetic. MLX reaches 12 TFLOPS on the same GPU through Apple's own matrix
path, which WebGPU does not reach at all: `chromium-experimental-subgroup-matrix`, exposed only
behind `--enable-unsafe-webgpu`, offers f32→f32 and f16→f16 8×8×8 here and runs at 3.9 TFLOPS
(f16) and 3.8 (f32) on independent multiply-accumulate chains — the ordinary ALUs, no faster
than the half kernel below. So the encoder's matmuls cannot match MLX on this GPU through
WebGPU, flags or not; they can get to the FMA peak.

The half kernel gets to about 65% of it (4.0 TFLOPS at 519×768×2304, 1.35× `mm_f16w`, which
is itself at 91% of the f32 peak). Two things made the difference and several did not:

- **The running sum must be half for the half rate**; f16 operands into an f32 sum run at the
  f32 rate (3.1 TFLOPS). So a thread sums 32 products in half and adds that into f32. The
  error grows as the square root of that span; 32 cost nothing over 64 and 16 cost 8–10%.
- **Issue the workgroup loads of a 4-deep step before its FMAs**: 0.52 → 0.47 ms. Without
  the loads at all the same FMAs run at 6.0 TFLOPS, so what remains is the loads and the
  staging, not the arithmetic.
- No help: activations already in half (−3%), larger tiles, 8 rows a thread, double-buffered
  staging (slower — registers), wider workgroups, re-mapping threads so shared reads coalesce,
  and weights paired along K so every FMA is `vec2<f16> × vec2<f16>` with no scalar splatted
  (3.2–3.6 TFLOPS against 3.6–4.0: the splat of a scalar of A costs nothing on this GPU).

**Padding was the other cost.** A batch of questions ran each one at the longest one's length,
rounded up: 160, 153 and 173 tokens became 3 × 192 = 576 rows for 486 real ones. Laid end to
end (`_replayed_packed`), the total is rounded once (512 rows), projections, norms and MLPs run
over the real tokens, and attention reads each question alone at its own length
(`fused_attention_packed`: per sequence an offset and a length, the padding and the window
computed in the kernel, no mask tensor; RoPE at each row's own position, `rope_qk_pos`). The
JS side stages token rows, segments, positions and the gather back to the head's layout in one
call; the vocabulary lookup is the pass's first dispatch. Per query the arithmetic and its
order are the padded layout's: with the same routes the answers are identical, and different
row counts can only change which tile or width a race picks per bucket. The first sight of a
packed shape runs eagerly in that layout, so its races settle before anything is recorded —
racing inside a recording records every candidate, and the first version replayed 62
attention dispatches a pass instead of 22.

**Recorded at a capacity, replayed for the rows a call has.** Per (questions, 32-row bucket),
almost every request met a shape it had not recorded: seconds apart, its first sight ran
eagerly (52–75 ms against 36–40 replayed) and its second recorded (60–88 ms) — a first
request several times slower and a steady state reached only gradually. A recording is
reissued by JS one dispatch at a time, so a replay can issue a dispatch with fewer workgroups
than it was recorded with: while `wt.elastic(rows, segments)` is open, each row-wise launcher
(the tiled and half matmuls and their reductions, row LayerNorm, GEGLU, positioned RoPE, the
row gather) attaches the rule its own workgroup formula follows, and packed attention its
(longest sequence, sequences) rule; `_dyn` checks each against the count computed at the
capacity. `replay(name, {rows, segments, longest})` then issues every ruled dispatch for the
live quantities, never past what was recorded. Every kernel here computes a row from that row
alone and attention reads only its segments, so the rows past the live ones (the rest of a
32-row tile) are harmless. The encoder records its packed pass at 128, 256 and 512 rows while
the model loads (`prepare_tiers`, ~51 KB pinned a row), larger capacities the first time a
call needs one; each recording is checked bit for bit against its own recording run at a row
count that is not a whole tile before it is used. Laya seconds apart: first request 25.6 →
17.7 ms, no request records any more; the Q8_0 GGUF's first two 69.6/77.6 → 22.8/29.3 ms.

**Which kernel wins is measured by the GPU's own clock.** A race sample that a collect landed
in measured the collect: every 1.4–2.1 ms sample of a 0.5–0.7 ms kernel had a young collect
and a reap inside it, and two of them made a 30%-faster kernel's win unprovable (7 of 9
paired wins, p = 0.09), so the slower one stayed — 10 ms on every three-question request.
Samples now run with no collection inside (`paused_reaping`); where the device has
`timestamp-query` they are the summed length of their passes, from the device's timestamps
(`timingBegin`/`timingEnd`), with nothing submitted early so a sample's dispatches run back
to back — not the browser's clock, which is coarsened and jittered on purpose and counts the
host and the readback too. Chrome without developer flags reports timestamps in steps of
65.5 µs, so a sample is 30 steps of GPU work (0.5 ms where the step is finer), sized from the
step the readings themselves show. An idle GPU clocks up over its first 15–20 ms of work
(a 0.77 ms kernel ran 3.6, 2.4, 2.0, 1.5, 1.2, 0.98 ms from idle), so a race that starts
after idle runs rounds until one is no faster than the last (`_settle`) before it times
anything; one that follows another race directly does not. A composite issued op by op from
Python (the decision head's row selection) is still timed by the browser's clock: its
candidates differ in what the host issues as much as in GPU work.

**Racing again, on request.** `remeasure(budget_s)` races again the routes the loaded model
has used — what its warm-up, recordings and requests looked up, not every bucket a ladder
explored — through the probes that raced them first (each ladder probe is kept per route
prefix; each `tune` keeps its own arguments; a model registers its composite choices with
`register_remeasure`, or says with `cannot_remeasure` why one cannot be). Each new choice
takes effect when its race ends; `cancel` or the budget ends the race in progress without a
verdict; recordings that looked up a changed route (`route_keys` around each recording) are
rebuilt when idle (`on_routes_changed`). The routes raced longest ago go first, so a budget
that runs out never strands the same ones, and the operator races it did not reach go on
between calls. Laya: 38 routes in 1.2 s, answers unchanged. 0.6B: 30 in 4.2 s, decode 177 tok/s
before and after.

Raced again, the choice in use is the one to beat (`_anchored_choice`): a challenger replaces
it only on paired evidence, and an inconclusive race changes nothing. The budget is a limit:
a race starts only when the duration of the race that last decided it (`_RACE_SECONDS`) fits
in what is left after a reserve for stopping (`_REMEASURE_RESERVE_S`), and a checkpoint inside
a race asks for room for its next round. A 27B's remeasure: 59.1 s of 60, twice.

**Two sets of routes.** A remeasure that changes routes puts what it made in use and keeps
the set that was in use as the set not in use (`_ROUTE_ALT`: its value for each route that
differs; `_ROUTE_IDS`); `switch_routes` swaps them and the next changing remeasure pushes out
the set not in use. A swap writes the other values into `_TUNED`, puts a model's own composite
choices back in force through what it registered (`register_route_apply`: the decode plan as
a load applies a kept one, the greedy chunk from its kept verdict; nothing is raced), and
rebuilds the recordings that used a route that changed. Each set's speed in use is the last
three replies (decode seconds per token, per decode path) or decision requests (by input size)
it served (`note_speed`); when each of the last three in use is slower than each of the other
set's last three, `on_routes_slower` callbacks are told and nothing is switched.

The samples wait out the remeasure's heat. On the M5 the same stored GEMV of a 27B's largest
weight took 1.06 ms of GPU time cool and 1.72 ms after a minute of ordinary replies (no
remeasure) or 1.48-1.80 ms right after a remeasure; decode went 8.0 → 4.0 tok/s, and was back
after a minute idle. Paging was not it: 12 MB read back from swap and 93 MB decompressed over
those replies. So a sample counts only once as long as the remeasure ran has passed since it
ended, and comparing a set's speed before and after a remeasure without alternating them is
not evidence either way: two such comparisons on the 27B came out +7% and -12%.

Every operator choice is made by one race (`tune` or `_weight_execution`): the decode thread
shapes, flash tiles, MoE weighted sum, KV pair write, add+RMSNorm, the parallel projections
and SwiGLU, Q/K norm+rope and the embedding row layout each had a loop of their own (host
clock, a collect free to land in a sample, no settling, and no way to race them again). They
are raced on inputs of their own shape made for the race and dropped after it, by the GPU's
clock and sized to it (`tune(sized=True)`); a 27B's shape tuning went from 9.9 to 4.0 s of
its load. What remains its own loop is a composite timed end to end: the greedy chunk (host
clock, registered for `remeasure`) and the decode composition. A load does not search the
decode composition: one full-model record can outlast any load budget, so a load applies the
exact original-width plan, or the plan a search kept in the profile. `remeasure` searches it
locally from the plan in force: one axis and one value at a time, each challenger checked
against the original-width oracle and raced against the plan in force on its own recording,
replayed alternately; adopted only when proven faster, and at once. The full product of the
axes (147 plans in one tournament) runs only offline. The plan in force is held before the
search (`_decode_selection`) and put back exactly on a stop before anything was proven, or
when nothing was. Each indivisible piece (trace, record, timed sample) starts only when the
budget has room for the longest that piece has taken on this model. Where a search stopped
is kept with the profile (`decode_search_v1`) and the next starts there: a 27B fits three
challengers in a remeasure, and walks the axes over successive ones. 30B: 18 tried, 1 adopted,
34 s, decode 41.3 → 42.1 tok/s. An earlier version raced it "again" through the load's path,
which applied the reference plan over a greedy session's pick mode and turned the chunked
greedy path off (0.6B 175 → 152 tok/s).

All of this is per device. `gpu_features()` reports what the device was created with —
`shader-f16` and `subgroups` are requested whenever the adapter has them, whatever its vendor
— and a kernel that needs a feature is offered only where it is present; which eligible kernel
runs is raced after load on the device itself (`_weight_execution`) and remembered per
adapter. Nothing is keyed by vendor, so another GPU with a different f16 rate, or none,
simply picks differently.

**No capability is assumed.** Every kernel is checked when it is registered
(`WebGPUPlatform.addKernel` → `unsupported_reason`): the features it enables (`f16`,
`subgroups`), the WGSL language features it `requires` (the browser's
`wgslLanguageFeatures`), the storage buffers it binds, its workgroup memory and size —
against what the device reported in `features()`, or WebGPU's guaranteed minimum (8 storage
buffers, 16 KB, 256 invocations) where it reported nothing. A kernel that does not fit raises
`KernelUnsupported` before anything is compiled, instead of a pipeline that fails later and
writes nothing. A race leaves such a candidate out; any other failure still stops it. The
choices made without a race ask first: the decode thread shape falls back to the default
where the narrow one does not fit (`_auto_kind`), flash tiles are sized to the device's
workgroup memory (`_flash_fits`; it had been a fixed 32 KB), the packed-dot routes need
`packed_4x8_integer_dot_product`, and the greedy chunk path needs the nine storage buffers
its input kernel binds. An audit of every kernel the 0.6B, 30B and 27B load (178) found 36
that a device with only the guarantees could not run — all optional variants — and with
Python told it was such a device, the 0.6B and the decision model ran correctly on the
remaining routes (138–143 tok/s; the decision answers exactly the full-width ones).

The load's deferred route ladders now run one probe per call into Python. With a 0.25 s
budget per call, a request that arrived right after a load waited 320 ms behind them.
