# Progress

## 2026-10-05 ▸ LLMs race their routes after load too (0.6B first reply 466 → 255 ms to first token)

The first reply of Qwen3-0.6B Q4_K_M spent 204 ms racing stored against unpacked weights for
the 256-row prefill bucket of seven layer shapes. `_warm_shapes` now runs `calibrate_rows`
over a 16→512 ladder for every distinct (format, N, K) the layers hold, so a prompt of any
length takes the nearest measured bucket. The output head is excluded: a prefill reads its
last row only, and laddering the 151936-wide head to 512 rows had cost six seconds of load.

Measured (local file via the page's input, WebGPU, M5, greedy, three different prompts):

| Qwen3-0.6B Q4_K_M | load | load-time races | 1st reply first token | races in replies | decode |
|---|---|---|---|---|---|
| before, cold | 12.8 s | 2.7 s (15) | 466 ms | 204 ms | 136–141 tok/s |
| now, cold | 11.9–15.9 s | 3.3–3.5 s (29–31) | 254–257 ms | 0 | 135–141 tok/s |
| now, saved profile | 13.2 s | 0 | 243 ms | 0 | 137–138 tok/s |

Load wall time varies ±2 s between identical runs; the races add ~0.6 s of it on a cold
device. Prefill picked "materialized" from 64 rows up for every Q4_K/Q6_K layer shape: those
formats have no tiled kernel yet. The 0.6B's ~13 s load is mostly not tuning (0 with a
profile) — not yet attributed.

## 2026-10-05 ▸ Route races move from the first answer to once after load (Q8 first request 526 → 121 ms)

**Why Q8's first three-question request took 500–600 ms and F16's ~120:** measured on a
fresh browser, 461 ms of Q8's 526 ms first request was `_weight_execution` racing routes for
the new 1024-row bucket (5 weight shapes × stored/tiled/materialized): ~123 ms compiling the
stored kernel's large-batch variants, ~211 ms of timed samples that each read back the whole
1.6–4.8 MB output, ~87 ms of checks and warm runs. F16 weights have one route (`mm_f16w`) and
race nothing. With a saved profile the same request was 117 ms — the race was all of it.

**Change (SDK, not page):**
- `calibrate_rows(probe, top, lo, step)`: right after a load, each route race a weight can
  set off is measured over a row-count ladder bisected in octaves — both ends, then the
  middle of any interval whose ends chose differently. Its prefix is then *calibrated*.
- `_weight_execution`: for a calibrated prefix, a row count no probe visited takes the
  nearest probe's measured choice instead of racing in front of the caller. Uncalibrated
  prefixes (LLMs today) keep the old behaviour. Calibration measures; answers never do.
- Decision models (`_warm_decision`): the warm-up question runs measured (it creates every
  projection and races the small buckets), then every distinct stored linear (format, K, N)
  gets a 16→512 ladder and the head's row-selection race a 16→512 ladder by token count.
- A race stops once its winner beats every other candidate (p ≤ 1/32), not once every pair
  of losers is ranked; samples sync on one element instead of reading the output back.
- Tiled K-split has its own line (cut below 48 workgroups): measured, cutting at mm_f16w's
  128 made 768×2304 at M=64 0.154 → 0.319 ms.
- Persistence of the measurements is the SDK's by default (`rememberTuning` now defaults on;
  `false` opts out). The chat page no longer passes it; laya-service and the browser-use
  skill, which never passed it, now keep their measurements too.

**Measured** (local Q8_0 GGUF via the page's file input, WebGPU, M5; answers identical
`billing 0.8966 0.979 1.7244`):

| Q8_0 | load | load-time races | first 3-question request | races during requests |
|---|---|---|---|---|
| before, cold | 2.3 s | 0.9 s | 526 ms | 461 ms |
| now, cold (first load on a device/build) | 3.2 s | 1.67 s | 121 ms | 0 |
| now, saved profile | 1.76 s | 0 | 107 ms | 0 |

F16 GGUF cold: load 2.0 s (head ladder 0.45 s), first request 100.5 ms, no races in requests.

**Limits:** only decision models calibrate; LLM prefill buckets still race on first use.
The ladder stops at 512 rows: a longer request uses the 512 choice. Only measured on M5.

## 2026-10-05 ▸ WebGPU Q8_0 at many rows: a tiled stored-format kernel (117.3 → 103.8 ms)

**Why Q8 was slower than F16** (GPU timestamps, xDecision encoder, three questions): every
multi-row Q8 projection had measured "materialized" -- expand the weight to f32 per call
(`ggmldeq_q8_0`, 93 × 0.123 ms) then the f32 matmul (`matmul_m32n64k4`, 93 × 0.638 ms) --
70.8 ms against F16's `mm_f16w` 47.2 ms. Everything else was the same kernels.

**Change:** `ggml_matmul` gains a same-width candidate, `"tiled"`, measured by the existing
`_weight_execution` per format, shape and row bucket, ordered right after `"stored"` because
it reads the same buffer and allocates nothing. One workgroup computes 32 rows × 64 columns;
per 32-value block its 64 threads decode the block once into workgroup memory -- the int8
values as halves, which are exact -- and each thread multiplies 4 rows × 8 columns from it,
the block's scale multiplying a per-block partial sum (d·Σq·x, f32). Q8_0 has it today.

**Measured before choosing** (M5, Chrome 154, medians of interleaved rounds, ms):

| 519 rows × K × N | stored | materialized | f32 tile | **tiled (this)** | mm_f16w |
|---|---|---|---|---|---|
| 768 × 2304 | 2.375 | 1.192 | 1.139 | **0.856** | 0.736 |
| 768 × 768 | 0.781 | 0.650 | 0.436 | **0.336** | 0.286 |
| 1152 × 768 | 1.119 | 0.606 | 0.615 | **0.466** | 0.403 |

Ablations on the 768 × 2304 case: decode replaced by a constant −5%, barriers removed −3%,
the multiply loop removed −80%; an f32 tile (d·q, 32 bytes a k) +33%; 8 rows a thread and
bank-conflict-free layouts made no difference. What the inner loop reads per k is the cost.
Measured FMA ceiling of this GPU from WebGPU: 3.2 TFLOPS; `mm_f16w` runs at 2.4.

**End to end** (local disk file through the page's file input, WebGPU, six requests):
`xDecision-Q8_0.gguf` steady median 117.3 → **103.8 ms**, every answer unchanged
(`billing 0.8966 0.979 1.7244`), 423 dispatches. All 93 multi-row projections chose
`ggml_tiled_q8_0`: 54.2 ms GPU against 70.8.

**Limits:** Q8 is still ~9% behind F16 (~95 ms), all of it in this kernel against
`mm_f16w` (54.2 vs 47.2 ms GPU). Other block formats still race stored against
materialized only. WebGL Q8 (2232 ms) is untouched.

## 2026-10-05 ▸ WebGL dense matmul: four K-values per texel (1773 → 856 ms); the head's weights joined it

**Measured first, in isolation** (40 queued calls, including packing the activations per
call): scalar → RGBA K4 `dot`, 519×768×2304 16.0 → 10.0 ms, 519×768×768 5.26 → 2.73,
519×1152×768 7.76 → 4.17, 8×768×2304 0.35 → 0.20, 1×768×2304 0.091 → 0.085. The scalar
kernel's two single-value fetches per multiply-add are its bound; four values per fetch is
a quarter of the fetches for the same bytes. Closer to a float64 reference too (4.3e-6 vs
7.7e-6). The earlier "RGBA was worse" row in ARCHITECTURE was measured at 69 rows.

**Built as WebGL's half of the existing `_packed`/`matmul_f16w` contract**, not a new route:
`pack_half_weight` returns an opaque `WebGLHalfMatrix` (RGBA16F, texel (j, k/4) = Wᵀ[k:k+4, j])
that only `matmul_f16w` reads — a generic operator reading that texture as an array would
get wrong numbers silently, so none can reach it. Kernels compile per K only (M, N are
uniforms). Packed activation rows sit side by side in the texture, so a call is not capped at
16,384 rows. The encoder's code did not change; `half_weight_ok` asks the backend.

**The decision head** had no half path on either backend — its projections were float32
matrices, and on WebGL not row-aligned either (27 ms a matmul by knockout, twice the
encoder's). It now takes the same `half_weight`/`matmul_f16w` path.

**Width kept (gate 1).** `half_weight` only accepts weights the file stores as float16. The
encoder had been packing any source to half width, which narrows a float32 checkpoint
silently; float32 — and BF16, which reads in as exact float32 — now stay float32, here and in
the R16F texture fallback.

**Results** (486-token three-question request, median of 6):

| | before | after |
|---|---|---|
| WebGL xDecision F16 | 1773 ms | **856–858 ms** |
| WebGL Laya safetensors | — | 858 ms |
| WebGPU xDecision F16 / Laya | 96.9 ms | 94.5 / 96.3 ms (same 427-dispatch sequence) |
| WebGL / WebGPU xDecision Q8 | 2226 / 117.3 ms | 2232 / 117.4 ms (unchanged, as on HEAD) |

Answers unchanged everywhere (each model its own). Host 235 passed (3 new: the float16 gate,
backend refusal asked first, the K4 texel layout); JS 102.

**Open:** Q8 is 2.6× slower than F16 on WebGL and 1.24× on WebGPU; WebGPU F16 at ~95 ms is
4–5× the ~20 ms an MLX implementation reports for the same model.

## 2026-10-05 ▸ WebGL decision: where the GPU time goes, and the first two fixes (2230 → 1773 ms)

**Instrument first.** Tried WebGL GPU timer queries: Chrome on Apple M5 exposes
`EXT_disjoint_timer_query_webgl2`, but on ANGLE-Metal a `TIME_ELAPSED` reading is the queue
latency of its command buffer, not the draw — every one of 486 draws read ~90–130 ms, an
add the same as a matmul, 46.8 s summed for a 2.2-s request — and a `gl.flush()` per draw
did not change that. `TIMESTAMP_EXT` reports 0 bits. So per-kernel GPU time is not
available from WebGL on this device; the experiment was reverted rather than shipped.

**Knockout attribution instead.** Each kernel family in turn keeps its output shape, draw
count and bindings but computes nothing (`main()` writes a zero of its output type); the
drop in the steady request time is what that family costs the pipeline. Committed scalar
baseline, local `xDecision-F16.gguf`, 486-token three-question request, headless Chrome,
median of 6: 2230 ms. Savings when knocked out: dense projections (88 draws) 1190 ms,
LayerNorm (50) 544, attention bmm (48) 375, other matmul (12) 323, add (89) 218, qkv
take (71) 213, softmax (24) 141, mul and geglu ~18. They sum to 135%: knocking one family
out changes the data and the load the others see, so this ranks and sizes, it does not
add up.

**LayerNorm (24% by knockout).** Every output element recomputed its row's mean and
variance: 1,536 fetches per element, a 519×768 LayerNorm reading as much as a 519×768×768
matmul (10.9 ms a draw). Now a statistics pass writes each row's mean and inverse deviation
once into an ordinary (rows, 2) tensor and a normalise pass reads them. Not bit-identical
to the one-pass form (Metal's fast math orders the same loop differently in a different
shader): at most 9.5e-7 apart on ~2% of elements, both 5.7e-6 from a numpy reference.
Request 2230 → 1878 ms, same answers.

**Softmax (6%).** The same per-element recomputation, plus a host-allocated `_zeros`
output — a 4.3-MB upload of zeros per call, 24 a request, into a buffer the kernel
overwrites; `_empty`'s own note already said softmax outputs belong there. Two-pass and
`_empty`: 1878 → 1771 ms, bit-identical to the one-pass form at every shape measured.

**When one pass still wins.** The RMSNorm path already keeps a one-pass form for few rows
(one fragment per row in the two-pass form leaves the GPU idle). Measured for LayerNorm and
softmax at 23 shapes (40 queued calls each): the crossover follows rows × width², not rows —
16 rows of a 173-wide softmax are still one-pass work, one row 4096 wide already is not.
`rows·width² ≤ 1.5e6` picks the faster form at every point (where it errs, by ~0.01 ms);
the measured table is above `_one_pass` and pinned by `test_webgl_one_pass_rule.py`.
Request with the rule: 1773 ms, same answers. WebGPU unchanged (96.9 ms, same 422-dispatch
sequence and weights). Host 232 passed (13 new), JS 102.

**Also tried, rejected:** stepping the LHS texel address instead of dividing per
multiply-add in the scalar matmul — slower (2232 → 2466 ms); the division was not the bound.
Codex's `pair2` stays stashed (`scratchpad/codex_pair2_binary.patch` was also saved).

**Next:** dense projections are now the larger half of the request. The scalar kernel does
two single-channel fetches per multiply-add; fetching four along K per texel (RGBA) is the
remaining WebGL-native storage/instruction option that has not been built and measured at
these row counts.

## 2026-10-05 ▸ WebGPU: F16 GGUF xDecision vs Laya safetensors — same path, same speed; load gap closed

**Trigger:** the user asked whether the local `xDecision-F16.gguf` (post-trained from Laya)
takes exactly Laya's path on WebGPU, with the same performance. Laya is the performance
reference; performance is the goal.

**Method.** Local page `127.0.0.1:8119/chat/?backend=webgpu&profile_decision=1`, driven by a
Playwright-launched system Chrome (headless, `--enable-unsafe-webgpu --use-angle=metal`).
Each model is fed to the page's own `#localModelInput` / `#localDirInput` with
`setInputFiles`, so it takes the app's `localPick → cache.import` route and the browser
reads the file from disk — no HTTP for model bytes, no download. One model per browser,
released before exit. Same three-question request (the app's billing example, 486 tokens).
Path identity is checked two ways: a fingerprint of every resident weight (name, container
type, dtype, shape), and the exact sequence of kernels dispatched on one request with the
encoder capture forced off (replay would hide individual dispatches).

**Inference (identical):**

| | Laya safetensors | xDecision F16 GGUF |
|---|---|---|
| weight fingerprint (169 tensors) | `498026a00a26dc09` | `498026a00a26dc09` |
| dispatch sequence, capture off | 422 · `1f3b06b3758d4e26` | 422 · `1f3b06b3758d4e26` |
| steady median (10 requests) | 96.1 ms (94.9–100.5) | 95.6 ms (94.9–98.7) |
| eager (capture off) | 93.3 ms | 94.5 ms |

Answers differ, as post-training should make them, and each model's are stable run to run.

**Load (was not identical, now near).** Per-function attribution of the loader: Laya 1598 ms,
xDecision 4486 ms, and the whole difference was `_gguf_header` (2800 ms vs 1.1 ms for the
safetensors header); everything else matched (warm 910/962, tensor reads 259/271,
tokenizer 241/269 ms). The file's header is 60.3 MB: the tokenizer JSON its loader uses, plus
a llama.cpp tokenizer it never reads (256,000 tokens, 256,000 types, 580,604 merges). The
parser decoded every array element in Python, and the read ladder (12 → 24 → 48 → 96 MB)
re-walked the header from the top each time; the 48-MB attempt alone cost 551 ms on the host.

**Change (generic, any GGUF, both loaders):** one `ggufload.read_header(read)` replaces the
two copies of the ladder in `decision.py` and `llm.py`; reads fetch only new bytes and the
walk resumes at the last finished entry. Arrays become `LazyArray` (decoded on first access,
otherwise the same list). Skipping them is one flat loop with no call per element — the
nested merges had been 886,983 recursive calls — and in a browser that loop runs in JS over
a zero-copy view of Python's bytes, falling back to Python on the host.

**Measured, Chrome, same file:** header 2800 → 417 (lazy + resume + flat loop) → 79–100 ms
(JS loop); whole load 4515 → 1707–1725 ms vs Laya 1612 ms. A browser check parsed the same
header with the JS loop and with Python only: identical version, metadata (every array's
length, offset and type), 170 tensor infos and data start (60,279,520). After the change the
weight fingerprint, the 422-dispatch sequence and the answers are unchanged.
Host: 219 passed, 1 skipped (4 new header tests); JS 102 passed.

**Left as found:** Codex's uncommitted WebGL `pair2` matmul candidate and encoder-scoped
route selection are stashed (`git stash list`), not committed: correct, but on this device
the encoder selects scalar and the full request gains nothing (median 1920 ms). The WebGL
decision latency itself (~1.9 s for three questions) remains the open performance item.

## 2026-10-05 ▸ measured WebGL decision operator costs before choosing a kernel change

- Used the already-loaded local `xDecision-F16.gguf` in Chrome WebGL; no model
  download or second resident model. The representative three-question request
  has lengths 137/130/150 (417 total), padded to 3×150 rows. This checkpoint
  has 22 encoder layers, each with attention QKV/output and MLP input/output
  projections, followed by two decision-head layers. The browser's temporary
  accumulator experiment was set back to incumbent mode 1; its on-disk code
  has already been reverted. No new performance candidate is enabled here.
- The product's `head_ms` is not head-only GPU work: it includes the final
  readback fence for queued encoder work. With an explicit fence just after
  encoding on three same-shape changed states, encoder GPU/readback wait was
  2018/1871/1871 ms, while the first full head layer took 267/261/272 ms,
  selected final layer 90/86/90 ms, and scorer readback 0.8/1.9/0.7 ms.
  Without intrusive fences, three further complete requests were
  2448/2306/2305 ms and retained the checked `billing` answer.
- A full-request 22-layer diagnostic with a barrier after each attention and
  MLP stage measured 22 attention stages at 1220/1150 ms net of a repeated
  idle readback and 22 MLP stages at 815/804 ms. These are diagnostic fenced
  wall costs, **not** additive production GPU timestamps. They establish that
  the encoder, particularly attention and MLP, dominates the ~2.3-s route.
- Fencing immediately before and after each of the 88 real encoder linears,
  then subtracting same-output idle readback cost, measured per-request sums:
  QKV 439–452 ms (22 calls, ~19 ms each), MLP input 404–417 ms (22, ~18 ms),
  MLP output 177–191 ms (22, ~8 ms), attention output 127–142 ms (22, ~6 ms).
  The intervening work before attention-output projection was another
  ~431 ms in a separately fenced run. Within attention, targeted fences
  measured QK batched matmul 228 ms (22), PV matmul 136 ms (22), and softmax
  107 ms (22); that run's total rose to 4296 ms because 66 extra barriers
  change scheduling/readback, so these are **relative operator costs**, not
  numbers to add to the normal request. The strongest observed individual
  targets are QKV and MLP input projections, then QK/PV/softmax collectively.
- Temporary Python method/function wrappers were restored after each run;
  the original product route and answer were rechecked. No speculative shader
  change is being selected from this measurement alone. Next work must test
  a targeted candidate against this baseline for correctness and end-to-end
  latency, including WebGL/other-backend parity.

## 2026-10-05 ▸ local xDecision F16 GGUF dense-route correction

- Used `/Users/mccoy/Desktop/code/laya-model/release/xDecision-hf/models/gguf/xDecision-F16.gguf`
  from the native browser file picker, not xDecision's safetensors checkpoint
  and not a network source. Parsed the GGUF header read-only: 170 tensors,
  169 F16 plus one F32, with exactly the same names, shapes and dtypes as the
  local Laya checkpoint; embedded encoder config and tokenizer JSON are
  semantically equal. Only one model was resident at a time.
- Root cause: the decision GGUF loader represented every supported 2D type,
  including unquantized F16, as `GGMLWeight`. This forced WebGL's generic
  byte-decoding shader for matrices that were already dense F16, while the
  same tensors from safetensors used WebGL's native R16F matrix route. The
  format-capability fix returns ordinary arrays for F16/F32 GGUF matrices;
  quantized GGML types still keep their packed blocks and native decoder.
- Matched Chrome WebGL 417-token three-question inputs before the fix took
  8011/7553 ms on the first two requests. After a release, reload and native
  file re-selection, four requests took 2386/2256/2260/2347 ms. Route was
  `batch` with `batched_selected_q` throughout; the checked billing choice
  and duplicate-charge score remained `billing` and 0.9994. Browser runtime
  inspection after the fix found zero `stored-linear` objects in both encoder
  and head, only dense `Tensor` objects.
- Released WebGL before switching to WebGPU and selected the same local GGUF.
  Its four matched requests took 172/138/94/93 ms with the same batch route
  and checked first two answers. These are serial local-browser diagnostics,
  not a universal benchmark. The shared WebGL ~2.3-second latency is still
  present in both dense Laya and F16 GGUF paths and remains open.
- Added a unit regression covering both dense F16 and F32 GGUF matrices while
  the existing quantized-block test remains in force. Host Python suite:
  215 passed, one skipped; this change does not claim complete model accuracy.

## 2026-10-05 ▸ requested remote checkpoint before further F16 GGUF work

- Audited the pending decision profile, memory cleanup, capture-local reuse,
  browser timing display, and version stamp changes. WebGL capture-local reuse
  is absent after its same-input correctness failure; no WebGL capture speedup
  is enabled. Decision-time tuning persistence avoids repeating measured route
  selection on reload, while request cleanup and explicit Release reclaim
  different classes of scratch and pinned resources respectively.
- Host tests: 214 passed, one skipped (`test/test_*.py`); Node tests: 102
  passed (`test/*.test.mjs`). The modified WebGPU wheel contains the same
  `webgpu_buffer.py` bytes as source; `git diff --check`, JS syntax checks, and
  content-version stamping passed. The WebGPU capture-local reuse has a
  lifetime test, but its benefit is not yet browser-quantified. This checkpoint
  does not claim full latency, accuracy, or backend acceptance.
- The next isolated diagnostic uses the local `xDecision-F16.gguf`, not the
  xDecision safetensors checkpoint. Serial Chrome WebGL testing found the same
  417-token batched route as Laya but ~8.0/7.6 seconds on its first two
  requests versus Laya's ~2.3–2.4 seconds. Root cause and correction remain
  open; no network model download was used.

## 2026-10-05 ▸ WebGL decision wait attribution and capture correctness

- Chrome's loaded local Laya WebGL three-question request reported 2704 ms, with
  2371 ms under `head` and 1749 ms in its final readback wait. That wait also
  fences queued encoder work. On a comparable 423-token three-question input,
  a diagnostic fence immediately after the encoder measured 1799 ms of pending
  encoder work; the two head-layer fences measured
  257 and 91 ms. The apparent head dominance in the unsynchronised profile is
  therefore a timing-boundary artifact, not proof that the head owns the stall.
- Controlled 22-layer encoder probes on that diagnostic shape found ~1106 ms in attention stages and
  ~748 ms in MLP stages (44 diagnostic readbacks). Fencing each of 88 linears
  associated ~670 ms with QKV projections and ~677 ms with MLP input
  projections, versus ~522/218 ms for the attention/MLP output segments.
  These are instrumented wall costs including readback and intervening work,
  not additive production GPU timestamps. They identify dense projections as
  the next candidate for isolated same-shape measurement; they do not justify
  changing a kernel yet.
- A temporary WebGL encoder capture with capture-local buffer aliasing produced
  reproducibly different answers from eager execution on the SAME input.
  Disabling only that new aliasing restored exact answer equality, so WebGL
  capture-local reuse was removed. Six changed-input eager/capture pairs with
  aliasing disabled were all answer-equal, but full-request median was 240.87
  ms eager versus 276.05 ms captured for a short one-question shape. Capture
  removed ~38 ms of encoder host submission yet lost ~35 ms end to end. It is
  not a measured positive WebGL route and remains disabled in the SDK. All
  temporary browser overrides and graphs were released; the user's loaded
  Laya model remains available. Inspecting the bad short capture's dispatches
  found zero same-dispatch input/output texture IDs, so the exact alias hazard
  is not yet proven; do not attribute it specifically to framebuffer feedback.
  Full host suite: 214 passed, one skipped; JavaScript suite: 102 passed.
  No model was downloaded.

## 2026-10-05 ▸ xDecision Q8 memory attribution; capture residency remains open

- The existing disk-backed GGUF is 402,546,752 bytes. On the same local Chrome
  WebGPU build it reported 134 MB GPU buffers after load and 127 MB after the
  first three-question answer. On the second request, encoder capture for
  `(batch=3, bucket=192)` raised the steady GPU ledger to 1.09 GB.
- The backend's capture accounting then reported 776 pinned buffers totaling
  1,033,578,476 bytes and zero bytes in its reusable idle pool. Only one
  encoder capture slot existed. Thus the excess is retained capture buffers
  (including touched weights/intermediates), not the on-disk GGUF expanding
  into 1 GB of persistent model weights. The decision cleanup added this step
  releases ordinary scratch but cannot free live capture pins by design.
- Releasing the model returned reported GPU buffers to 0 KB and WASM heap
  capacity from 2.38 GB to 50 MB. No across-release GPU leak was observed.
  Captures can still hold excessive live memory, and the current four-slot
  bound is not a byte budget. Any capture-memory change needs a measured
  latency/semantics comparison; none is claimed here.
- The three-question outputs kept `billing` 95.4% and duplicate-charge `true`
  99.9%; the third answer remains outside the requested accuracy conclusion.
  Full JS suite passed 102/102 and host Python suite 212 passed, one skipped.
  No commit or push in this step.

## 2026-10-05 ▸ decision cold-route reuse verified; hot GPU variance remains open

- On the same local Chrome WebGPU page and 486-token three-question example,
  five Laya hot diagnostic repeats took 213/129/196/141/185 ms. Their route
  stayed `batch`, weight tuning remained 0 calls, and the final combined GPU
  wait ranged from 86 to 185 ms. A separately observed Laya run reached 119
  ms. These repeated-input runs diagnose variance, not changed-question product
  throughput. The 119-ms path has not disappeared, but it is not stable.
- Released Laya before loading the existing disk xDecision Q8 GGUF. Its first
  request took 850 ms, of which encoder/head weight tuning was 299.5/379.2 ms
  (3/5 calls). Its second request took 352 ms, including a 256.7-ms capture
  submission. Five subsequent hot repeats took 264/160/193/172/177 ms, all
  with zero tuning calls. Their range overlaps Laya's; no stable per-format
  hot gap is established by these small, serial samples. Both models still
  selected `billing` and treated duplicate charging as true on the first two
  checked questions; the third is excluded from accuracy comparison.
- Found a concrete repeat-cold root cause: the worker saved reply-time kernel
  profiles for LLM generation but not for decision `decide`. Added conditional
  persistence when a decision request creates new device/shape tuning entries.
  A hot request neither exports nor writes an unchanged profile. After a fresh
  local build stamp, first xDecision request took 955 ms with 825.2 ms of
  tuning; after release and page reload, the next load reused 37 profile entries
  (formerly 29) and its first identical request reported 333 ms and 0/0 tuning
  calls. The first two answers were unchanged (billing 95.4%, true 99.9%).
  The second request still took 332 ms, including 270.5 ms capture submission.
  Thus repeated weight tuning is fixed, but first-ever tuning, capture cost,
  steady GPU wait variance, and the target absolute decision latency remain open.
- Two new JS profile-persistence tests pass. Full JavaScript suite: 102 passed;
  focused Python decision suite: 60 passed. No commit or push in this step.

## 2026-10-04 ▸ xDecision/Laya local parity and latency attribution (open)

- Used only the existing local `xDecision-Q8_0.gguf` and
  `convaiinnovations_laya-multilingual` files, never a model download and never
  two resident models at once. Their 170 tensor names/shapes match exactly;
  embedded encoder config and tokenizer JSON also match exactly. The decision
  head config differs in post-training calibration, so exact probabilities are
  not a parity requirement.
- On the same three-question example, native CPU xDecision took 235.0 ms for
  486 tokens and Laya took 226.9 ms; both selected `billing` and scored the
  duplicate-charge statement true (0.9993 and 0.9939). The third question's
  accuracy is excluded per user instruction. Both used one batched encoder and
  one batched head pass. This establishes these checked semantics on CPU, not
  full benchmark accuracy or browser backend equivalence.
- Chrome WebGPU xDecision showed 1115 ms on the first three-question request;
  the head's 621.5 ms included 453.0 ms queuing its first layer and 166.8 ms
  queuing scores. The second request took 420 ms, including 144.7 ms capture
  submission. Later requests still fluctuated around 313–330 ms in this
  profile. The final head readback fence includes outstanding encoder and head
  GPU work; it must not be described as head-only kernel time. Per-pass GPU
  timestamp callbacks spilled into subsequent requests, so their totals are
  not yet a reliable per-request attribution.
- Added opt-in encoder/head weight-route tuning milliseconds and call counts
  to `profile_decision` output so a fresh local browser load can distinguish
  calibration from shader compilation and GPU execution. Related 60 Python
  tests pass, `chat/app.js` syntax and `git diff --check` pass. The browser
  re-run is blocked by macOS lock; no performance fix or GPU parity claim yet.

## 2026-10-04 ▸ serial WebGL 0.6B and 30B browser check (open)

- Following the user's provisional acceptance of current WebGPU performance,
  selected only the local Qwen3-0.6B Q4_K_M GGUF on explicit WebGL. With
  reasoning and autonomous Python tools off, the actual chat page copied
  lines 1–40 correctly in 110 tokens at 18.2 tok/s and 2.4 s first token.
  A WebGL layer trace measured 42.065 ms for an unsplit synthetic step; its
  split stage sums are diagnostic, not additive production time. The explicit
  correctness-gated full-decoder calibration selected the existing `auto` / full
  path (41.9225 ms diagnostic median); a second correct page reply ran at
  18.4 tok/s. No positive route or source change was claimed. The lower
  same-format WebGL SwiGLU route was already tuned to fused on this device.
- Released 0.6B and observed GPU buffers 0 KB / WASM 42 MB before choosing the
  sole local Qwen3-30B-A3B-Instruct Q3_K_XL GGUF through the native file input;
  no model bytes crossed HTTP and no two models were resident. The 13.83 GB
  load reached ready, but `warming` took 4.0 s and `proving` took 16.1 s,
  violating the 10 s load-stage requirement. Declared WebGL textures were
  12.73 GB after load, peak 13.33 GB, and WASM 1.40 GB.
- On the complete-answer 1–40 prompt, the first visible token appeared only
  after roughly 20 s; 13 generated tokens had reached about 0.3 tok/s with
  the product reporting tight paging and severe frame delay (up to 1951 ms).
  This was not a completed accuracy or settled-throughput sample. To avoid
  prolonging severe memory pressure, generation was stopped and the 30B was
  released. Verified afterward: GPU buffers 0 KB, WASM 42 MB. Prior evidence
  of ~9 GB swap activity in a serial 30B layer sweep makes cross-layer weight
  residency/paging the leading diagnosis, not a proved sole mechanism.
- WebGL 0.6B throughput optimisation, 30B <=10 s load and seconds-level first
  token, broad format/API parity, decision-model work and final acceptance
  remain open. No source optimisation or final commit occurred in this check.

## 2026-10-04 ▸ restore and validate historical 0.6B path (in progress)

- User clarified the acceptance meaning: ordinary throughput should return to the
  historical 140+ tok/s regime, with occasional falls no lower than about 135;
  a 135 tok/s median with frequent 120s is not acceptance. Prior recorded evidence
  has direct SDK 139–141 tok/s and visible chat 136.41–137.54 tok/s on the same
  local Qwen3-0.6B GGUF. Do not relabel direct SDK measurements as page speed.
- The current ordinary WebGPU page, one local model, 231 context / 110 generated
  tokens, initially returned the correct 1–40 sequence at 119.64–130.27 tok/s
  over eight runs. A lower-level `down=balanced` win did not transfer by itself
  to the full API. `o=balanced` plus `down=balanced`, fused add/RMS and compact
  fused QKV preserved the full answer and had a 135.28 tok/s median over twelve
  page runs, but lower results remained well below the user's condition. It is
  not accepted as a substitute for the historical state.
- The historically recorded full fusion (add/RMS, QKV, QK norm/RoPE, KV write,
  compact head, stored Q4_K and GPU sampling) was restored as a runtime trial;
  on the prior loaded build its twelve correct page runs were 101.7–135.82
  tok/s. On the newest locally reloaded build, explicit fused QKV physical
  shapes reached ~139.7 tok/s in some complete SDK replies but the visible
  page did not reproduce ordinary 140+. The old machine profile had been
  overwritten by a later `decode_plan_v3` entry; only the newer cache remains.
- Added a generic, correctness-gated upper-level projection-shape composition
  stage plus explicit historical fused combinations to the offline candidate
  tournament; no model name or category selects a route. Existing eight-field
  profiles still load; a ninth field carries independently addressable `o` and
  `down` shapes. Invalid saved shapes are rejected. 170 Python and 79 JS tests
  pass. Stamped browser SDK `9ef7514292` was released/reloaded from the same
  disk GGUF; GPU buffers were 0 KB and WASM 42 MB before load. A one-model
  explicit offline complete-API tune was started (result below). No network model download,
  WebGL/decision completion claim, final commit, or clean-worktree claim.
- The explicit offline tune finished after 332.4 s and chose the older
  composed/auto/separate plan (10.66 ms diagnostic median). Three actual page
  replies on that plan were correct at 109.16/124.30/124.21 tok/s, so this
  tuner result was not accepted as the user's historical-performance baseline.
  The recorded full-fusion plan was installed in the current device-local
  `decode_plan_v3` profile and verified by releasing the only model, reaching
  0 KB GPU / 42 MB WASM, reloading the same disk GGUF, and checking
  `profile_reused=True` with all six recorded upper-layer route fields.
  Twelve post-reload page replies were exact 1–40 but only 115.01–133.53
  tok/s. This proves route persistence on the same build, **not** recovery
  of the 140+ product speed or safe reuse after a source-stamp change.
- An independent empty-history page at `127.0.0.1` initially measured about
  60 tok/s, but it was **not a controlled comparison**: that origin defaulted
  to reasoning and autonomous Python tools on, while the original page had
  both off. The tool constraint forced JS rather than GPU sampling and changed
  output semantics. Those rates and wrong-format replies must not be used to
  attribute the slowdown to chat history. The browser turn interruption then
  closed the temporary tabs; no model remains loaded. The next check must
  align both toggles and the same source/device profile before comparing.
- The controlled follow-up copied the *same-build, same-device* kernel profile
  into a separate local browser origin with only 13 test chats, then selected
  the same GGUF via the native disk chooser. Both pages had reasoning and
  Python tools off, `full/gpu` sampling, 231 context and 110 generated tokens;
  the fresh page confirmed `profile_reused=True`. Its 24 exact 1–40 chat
  replies ranged 126.88–138.90 tok/s, roughly 135 median, whereas the prior
  729-chat page had ranged 115.01–133.53. This is evidence that page state
  matters, but origin and prior runtime activity also changed, so it is not
  proof that the chat count alone is causal. Eight fixed-seed complete SDK
  replies on the clean page were exact at 137.09–143.39 tok/s (median 141.83);
  the *SDK* historical 140+ regime is reproduced, while the *chat product*
  acceptance remains open. Unseeded direct calls likewise reached 141–145
  with two lower outliers. Paired page/direct runs varied, so no deterministic
  UI tax is claimed. Temporary per-token callback timing was only 10.22 ms
  over 440 tokens; live rendering was 9.68 ms over four replies. Neither alone
  accounts for the whole page/direct gap. Both wrappers were restored.
- A regression now round-trips a compatible complete fusion plan through
  `kernel_profile()` / `use_kernel_profile()` and verifies that reload restores
  all six upper routes, original-width Q4 execution, compact head and GPU
  sampler identity. Full local checks: 172 Python, 79 JavaScript, and clean
  `git diff --check`. A source-stamp change still invalidates device profiles
  by design, and the current browser-only profile is not a universal default.
- The measured same-build/device profile is now checked in as
  `profiles/webgpu_apple_metal3_2026-10-04.json`, including its exact route,
  source fingerprint, build stamp and separate SDK/page outcome labels. A
  regression imports the artifact and checks the route and GPU sampler; it
  does not add a runtime model-name branch or claim portability to other
  devices/builds. The sole locally selected model remains ready in the in-app
  WebGPU page, whose latest visible 110-token answer is correct at 136.9 tok/s.
  Current scoped checks: 173 Python, 79 JavaScript, `git diff --check` clean.
  This is historical-path preservation, **not** a claim that normal chat-page
  throughput has reached stable 140+, nor that the other project gates passed.
- Six additional same-page 231-context/110-token replies confirmed
  `profile_reused=True`, stored Q4_K execution, fused upper path and full GPU
  sampling. All six were exact; actual page rates were 128.41, 136.55, 139.08,
  136.98, 134.71 and 136.02 tok/s. A preserved route therefore did not remove
  the occasional sub-135 dip. Do not mark the 0.6B product gate complete.

## 2026-10-04 ▸ same-instance sampler submission experiment (rejected)

- On the sole native-file Qwen3-0.6B Q4_K_M in the in-app WebGPU browser,
  five additional 231-context/110-token product replies were correct but
  104.92–128.55 tok/s, below stable 140+.
- Fixed the decode-plan status when returning from an unprofiled greedy call
  to the already measured sampled route. Its stale `mode_profile_missing`
  marker is now cleared; a browser check showed active `full/gpu` and no
  missing marker. Regression added; 168 scoped Python tests passed.
- Prototyped queuing the sampler's 16-byte metadata before submitting the
  pending decoder graph so the decoder and sampler could share one submit.
  Eight interleaved real-page pairs preserved the 1–40 output but favored
  that candidate only 3/8 times. Eight fixed-seed full-SDK pairs preserved
  identical complete text and favored the incumbent two-submit path 8/8.
  The candidate source and test were removed; fewer submissions was not a
  positive upper-API optimization on this device. No fixed percentage cutoff
  was used. The final `380ffcae77` SDK stamp was loaded from the same native
  GGUF after GPU buffers/WASM heap returned to 0 KB/42 MB. Four more complete
  browser replies were correct at 117.23/107.85/128.69/128.27 tok/s, and
  active plan diagnostics showed `full/gpu` without a stale missing marker.
  A same-instance position-231 layer trace measured 7.94 ms whole-step and
  2.079/1.1475/1.1355/1.0841 ms aggregate MLP/QKV/attention-output/head
  net stages. Five subsequent correct page replies measured 84.59–128.87
  tok/s; profiler-related buffers returned from a transient 606 MB to the
  555 MB steady level, with 984 capture pins. System memory was 64% free and
  macOS reported no thermal/performance warning; this does not prove a GPU
  frequency or contention cause.
  Maintained suites: 168 Python, 79 JS; `git diff --check` clean. The 0.6B
  performance gate and seven-point acceptance remain open, so no final commit.

## 2026-10-04 ▸ current `bf9fed4c0b` local 0.6B browser continuation (open)

- Subsequently fixed a profile-key gap: WebGPU's complete-decode key is v3
  and distinguishes the *effective* JS/GPU sampling path, including fallback
  when an option such as top-p makes GPU sampling ineligible. WebGL's v1
  topology/key is preserved. A regression covers both distinct WebGPU keys
  and identical WebGL keys. Rebuilt/stamped to `72e61e0a7b`, released the
  prior single model (GPU 0 KB/WASM 42 MB), and selected the same disk GGUF
  alone. Its saved GPU sampler route was reused; the missing v3 upper plan
  correctly fell back to a budget-limited stored-width path. One baseline
  product answer was correct at 121.24 tok/s. The explicit v3 GPU-sampler
  full-composition tune finished in 216.57 s, chose a stored-width route with
  separate QKV and composed QK norm/RoPE (8.46 ms diagnostic median), and
  saved its profile. Eight correct full product replies were 86.79–128.52
  tok/s, still not 140+. The former JS-tuned fused route won only 5/8 real-page
  pairs under GPU sampling and was restored to the v3 winner.
- A 12-run worker profile associated slow 73–87 tok/s replies with 10.9–12.9
  ms GPU-completion/readback waits, versus 7.38–7.48 ms on 127–130 tok/s
  replies. An opt-in hardware timestamp run on the same sole model then showed
  5.90–6.42 ms main-compute-pass medians for faster diagnostic replies and
  10.49–13.17 ms for slower ones, with the same 562 main dispatches and one
  sampler dispatch per token. Profiling perturbs absolute rate; it does establish
  that GPU execution itself varies substantially, not just page rendering.
  External contention, GPU clock behavior and other causes are not isolated.
  The browser was returned from `profile_gpu=1` to ordinary WebGPU; the same
  local file reloaded with `profile_reused=True` and four correct replies at
  119.71/120.37/128.02/128.78 tok/s.
- Fixed idle greedy-graph retention under normal sampled generation. The
  loaded-model intervention lowered capture pins from 2,956/19.55 MB to
  984/4.65 MB with a correct answer. The new SDK stamp `303f1177d3` leaves
  zero pins at load and holds 984/4.65 MB after four sampled replies. A forced
  chunk-2 greedy test reproduced the scalar greedy's full 110-token answer;
  returning to sampled chat automatically freed its graph and returned to
  984/4.65 MB. The forced route was restored to the measured choice. This
  does not close throughput. Maintained suites: 167 Python, 79 JS;
  acceptance and final commit remain open.
- Rechecked the prior fused QKV/QK combination under GPU sampling. A 16-pair
  alternating product test gave only 7/16 fused wins; all 32 replies were
  correct. A graph-route switch leaves roughly 3.45 MB temporarily in the
  reuse pool, so first-after-switch timings are not clean steady-state
  comparisons. Six two-run paired blocks (compare second runs) split 3/6;
  four continuous eight-run A/B/A/B blocks still drifted within each route
  (settled medians 125.8/130.4/110.9/90.9 tok/s). The v3 measured route
  was restored. `powermetrics` requires superuser on this host; no thermal or
  GPU-frequency root cause is claimed. The same local 0.6B is left loaded on
  the ordinary in-app WebGPU page.
- Kept exactly one local Qwen3-0.6B Q4_K_M GGUF loaded in the in-app browser;
  there was no model download or parallel model load. The explicit v2 full
  decoder search finished in 222.86 s, chose stored Q4_K arithmetic and
  measured an 8.525 ms diagnostic median. Six actual chat replies were all
  semantically correct at 86.93–123.76 tok/s, not stable 140+.
- The loaded-model layer profiler reported 6.565 ms at position 0; MLP, head,
  QKV and attention-output net stage sums were 1.991/1.059/1.005/0.653 ms.
  JS full-vocabulary sampling added about 0.65 ms per token in its diagnostic
  worker scope; 4-byte GPU sampling removed that JS sampling work, but still
  waited for GPU completion. Do not label the 6.565 ms value product latency.
- Five order-alternated, complete real-page JS/GPU sampling pairs all preserved
  the 1–40 answer and favored GPU (132.06–134.04 versus 72.12–123.05
  tok/s). Eight fixed-seed full-SDK pairs agreed in entire output but favored
  GPU only 5/8. Set the measured local device/vocabulary profile to GPU and
  verified 111 selections read back 444 bytes in total. Six further page
  replies remained correct but measured 111.41–133.67 tok/s, so this has not
  closed the 140+ gate. Disabled profiling before those six calls.
- An upper-level candidate combining fused add/RMS, fused KV write and compact
  output-head shape preserved the complete fixed-seed output in ten paired
  SDK comparisons but won only 6/10; its temporary runtime overrides were
  restored. No unsupported speed claim or unproven source route was adopted.

## 2026-10-04 ▸ actual-sampler upper composition and loaded 0.6B (open)

- Rebuilt and stamped `793af5ae3d`, released the previous sole 0.6B model to
  0 GPU buffers / 42 MB WASM, then reselected that same GGUF from the native
  browser file chooser. No additional model or remote weight source was used.
- Fixed the WebGPU top-level decode tuner to time the actual seeded JS/GPU
  sampling path instead of full-logit readback plus argmax; semantic screening
  compares the selected token sequence as well as four steps of logits.
  Only old WebGPU whole-decode profiles are invalidated (`decode_plan_v2`);
  WebGL's previously real-path profiles retain v1.
- The explicit full-model v2 search finished in 217.4 s and kept the
  original-width composition (`median_ms=9.6925` in its diagnostic context).
  A new behavioral test proves that matching logits with divergent sampled
  tokens cannot enter the timed candidate set; it also checks restoration of
  the prior sampling state.
- Seven post-search, single-model product replies all returned the requested
  1–40 sequence at 120.22/123.66/110.46/127.51/128.30/124.08/94.58 tok/s
  (231 context, 110 generated tokens). The 140+ settled-product goal is **not**
  met. GPU buffers were 570 MB and JS heap 146 MB after the search. Releasing
  the only model returned to GPU 0 KB / WASM 42 MB; a page reload lowered the
  JS heap from 172 MB to 22 MB. The same local file then reloaded with
  `profile_reused=True` for the v2 plan; six further correct page replies
  measured 127.57/130.07/122.84/130.46/121.67/125.36 tok/s. The sole 0.6B
  is left loaded for subsequent work.
- Current source suites pass: 163 Python, 79 JavaScript. A blanket root
  `pytest` discovery also picked up legacy Pyodide-only tests and failed at
  collection; the maintained local suite is `test/test_*.py`. No final commit.
- A first same-instance Q4_K `balanced`/`compact` A/B is invalid: the diagnostic
  helper changed Python attributes but failed at an incorrect `_adam_kernel`
  module reference before invalidating the captured graph. Its speed/output
  observations must not select a route. Explicitly clearing the temporary
  graph reduced capture pins to 0; one ordinary reply recaptured 984 pins /
  5.58 MB and six more replies held that count exactly, so no steady per-turn
  pin accumulation was observed. The A/B below has now been rerun with reset
  success asserted.
- Corrected A/B verified every 168-matrix route switch and graph reset, then
  ran fixed-seed whole-SDK pairs: `balanced` won 10/18; `compact` won 8/8
  initially but only 6/12 on an independent repeat. On the real chat page
  `compact` won only 6/10. Every fixed-seed full text matched and every page
  reply preserved 1–40, but no repeatable containing-API win was established.
  Restored all linears to `auto`; the final correct page reply was 126.88 tok/s.
- The existing opt-in WebGPU readback-buffer pool was screened in 8 alternating
  product pairs. Speed fluctuated widely on both sides (79–131 tok/s) without
  a repeatable benefit, so it remains off. The source was not changed.
- A `narrow` Q4_K layout lost all 8 fixed-seed, complete-SDK pairs (about
  101–108 versus 128–133 tok/s) with identical full text. It remains disabled.
  The adapter offers shader-f16/subgroups/timestamp queries; feature presence
  alone does not justify a width change or establish a fast, semantic route.
- A diagnostic Q4_K `tiny` (16×4) same-width workgroup passed the independent
  packed-weight GPU selfcheck and kept identical fixed-seed full text, but won
  only 2/8 complete-SDK pairs. Its source changes were reverted.
- Rebuilding an unchanged backend wheel had changed the browser SDK stamp
  because wheel ZIP timestamps differed. Added a default fixed
  `SOURCE_DATE_EPOCH` to the WebGPU, WebGL and test wheel entrypoints.
  Consecutive independent builds of all three produced identical SHA-256
  hashes; the stamp script ran twice without changing `bf9fed4c0b`.
  This fixes needless tuning-profile invalidation across same-source builds.

## 2026-10-04 ▸ loaded 0.6B product performance and rejected candidates (open)

- Used only the Qwen3-0.6B Q4_K_M GGUF selected from disk in the in-app browser.
  The current tab initially showed the model ready on WebGPU. Five complete,
  same-prompt 231-context, 110-token page runs all returned numbers 1–40 in
  order at 118.56/124.44/106.84/135.82/123.26 tok/s. This is still below
  a reproducible 140+ product result, despite earlier direct-SDK 140+ samples.
- A loaded-model layer profile gave 6.38 ms for the unsplit one-token step at
  position 0 and 6.785 ms at position 231. The latter staged diagnosis summed
  to 6.771 ms net, dominated by MLP 2.313, attention output 1.263, head
  1.257 and QKV 1.253 ms. These are diagnostic complete-step and staged
  measurements, not additive production GPU timestamps. Python input setup
  was 0.260 ms median (embedding row 0.155, RoPE 0.015), so moving that work
  alone cannot account for the whole product-speed gap.
- An existing device-side Q6_K embedding-row route reproduced the original
  packed row and both RoPE inputs bit-for-bit at three token/position probes.
  A runtime-only A–B–A over nine full page replies gave baseline
  121.27/106.22/133.96, GPU-row 139.60/137.31/106.33, returning baseline
  130.84/136.94/137.31 tok/s. It was restored, not promoted to `auto`.
- A TypeScript candidate coalesced decode replay and GPU vocabulary sampling
  into one shared-memory signal after the readback arena was registered. It
  passed the queue unit test and six complete browser replies with the same
  output, but its rates 125.43/123.89/136.97/130.50/137.52/114.64 had
  median 127.97 tok/s. After reverting, rebuilding and reloading the same
  disk GGUF, the ordinary route gave 134.69/131.71/133.96/137.05/134.67/
  122.77 (median 134.32). The candidate source and test were removed;
  no unproven optimisation is enabled. The restored SDK stamp is `4bbb2e651f`.
- Four alternating direct-SDK/page pairs on the restored instance showed
  overlapping, time-varying rates rather than a consistent page-only tax.
  Runtime-only complete-SDK comparisons then examined the upper-level output
  head, gate/up and QKV physical layouts without changing source. A narrow
  Q6_K head was substantially slower; balanced and short-K head layouts were
  mixed against the returning compact baseline. Fused gate/up did not beat
  separate repeatably. With a fixed sampling seed, gate/up routes produced
  identical complete text. Twelve order-alternated fixed-seed QKV pairs gave
  `fused:compact` only 6/12 wins over `fused:default`, with mean delta
  −1.46 tok/s; all 24 outputs were identical. All instance overrides were
  restored. A final page reply returned the 1–40 sequence at 132.19 tok/s.
  A temporary, bounded diagnostic cache for decoded embedding rows and RoPE
  values lowered Python input preparation from 0.220 to 0.090 ms/call, an
  upper-bound saving of about 0.13 ms/token; it was **not** added to the
  product because Python-side data caching conflicts with the JS/GPU ownership
  requirement and is too small to explain the observed 1–2 ms rate variance.
  The temporary instance override, its bound-method reference, cached arrays
  and diagnostic globals were removed from the live Python worker.
  An opt-in sampler-boundary diagnostic measured 913.1 ms over 111 calls,
  which includes waiting for the preceding GPU decode and cannot isolate the
  sampler's own arithmetic. One incorrectly initialised diagnostic object
  caused a test reply to fail; its temporary flag was removed and the next
  full 1–40 reply succeeded. The browser was left with only the restored local
  0.6B loaded. Full local checks: 79 JavaScript, 161 Python tests, webpack,
  stamp and whitespace checks pass. The seven-point acceptance and final
  commit remain open.

## 2026-10-04 ▸ WebGL 0.6B browser rerun and slow-warning correction (open)

- The only observable in-app tab was `?backend=webgl`, initially with no
  resident model. Selected the sole local Qwen3-0.6B Q4_K_M through the
  native GGUF chooser; no model HTTP transfer or concurrent model. It loaded
  332.9 MB from disk. A completed 216-context, 110-token chat reply was
  exactly lines 1–40 at 18.76 tok/s, 2.231 s first-token latency. The saved
  WebGL upper plan was reused. Visible GPU buffers read 448 MB.
- Browser-verified the UI correction: WebGL 18.76 tok/s no longer triggers
  the WebGPU-only 40 tok/s slow warning. The functional VM test also checks
  that the same rate still warns on WebGPU. Full local regression on this
  source: 161 Python, 77 JavaScript, TypeScript build, stamp, diff check.
- This closes only that warning regression and one WebGL correctness/speed
  sample; the seven-point project gate and stable WebGPU product 140+
  requirement remain open. No final commit.
- After WebGL release (0 KB GPU, 42 MB WASM), reloaded the same local GGUF
  alone under WebGPU. Reused full upper plan and GPU sampler. A default
  six-reply series returned the 1–40 sequence at
  114.95/132.43/125.52/100.92/121.00/133.16 tok/s; the observed slowdown
  tracks the mixed GPU-sampling/completion/readback/host scope
  (7.1–9.3 ms/token); the separate 0.3–0.4 ms scope is replay submission,
  **not** a measurement of GPU arithmetic. A same-prompt 1024→2048→1024 dispatch
  flush A–B–A gave six-run medians ~123.3/130.5/130.9 tok/s. The candidate
  had no repeatable gain against the returning default, so source remained
  at the default 1024. All replies preserved the number sequence (some added
  a code fence). The default WebGPU model is the sole resident instance.
- A 512-dispatch six-run candidate measured
  122.75/133.64/137.19/121.02/123.44/136.95 tok/s, followed by a
  default-1024 series at 121.34/130.72/112.93/108.58/115.42/100.92.
  The same route drifted strongly over time; neither 512 nor 2048 is a
  repeatable positive winner, and production stays at default 1024. An
  isolated fresh tab produced 119.23–136.76 tok/s in six page runs and
  112.03–142.50 in three direct SDK runs. Do not call 140+ stable.
- Repeated release reported 0 KB GPU and 42 MB WASM, but the shared browser
  GPU process RSS moved only ~1.81→1.78 GB immediately and ~1.65 GB after
  empty-page reload / temporary-tab closure. Driver caching versus retained
  resources is unresolved. Corrected the page's false `GPU + host` breakdown:
  GPU sampling plus completion/readback happens inside the historic `pick_ms`
  scope, so the UI now says `step + pick/readback`. The new stamped browser
  showed 0.34 + 7.25 ms at 129.85 tok/s with 1–40 sequence intact.
  Targeted UI test and all 78 JavaScript tests pass; `git diff --check` passes.
- Corrected the attribution with the existing opt-in WebGPU hardware timestamp
  path, one resident local 0.6B. In two warm three-token diagnostics, the
  first 366-dispatch compute pass took ~15.0 ms, the next two ~6.29–6.42 ms.
  A 12-token diagnostic's following main passes ranged 6.29–7.21 ms,
  plus ~2.03 ms summed over its small auxiliary passes. Timestamp query
  instrumentation changes the short-run wall speed, so this is not a new
  product throughput result. It establishes that the old 0.3–0.4 ms `gpu_ms`
  was submit time, not GPU arithmetic, and that the pick/readback scope
  includes substantial real GPU work.
- Fixed the page's unconditional Markdown/fenced-code system instruction that
  contradicted an explicit plain-text user prompt. Both backends now receive
  a model-agnostic "user format first" display default. The same sole local
  Qwen3-0.6B produced unfenced 1–40 sequences in three WebGPU product runs
  (127.72/129.06/130.92 tok/s, 231 context) and two sequential WebGL runs
  (18.36/18.47 tok/s, 2.607/0.055 s first token). The model still generated
  trailing spaces on some lines; this is not byte-exact copying. WebGL's
  WebGPU-only slow warning stayed absent. A shorter system prompt gave mixed
  direct-API paired results and was not adopted without page-level evidence.
  Browser returned to one locally loaded WebGPU 0.6B on stamp `76dfea11a1`.
  Full local suites pass: 161 Python, 79 JavaScript; diff check is clean.

## 2026-10-04 ▸ 0.6B sampled-path A/B and current product regression (open)

- Used only the user-selected local Qwen3-0.6B Q4_K_M in the in-app browser.
  Tested a generic two-token GPU sampling graph under the same full-decoder
  plan as baseline. Three fixed-seed 1–40 outputs were identical, but the
  candidate lost every paired throughput comparison (134.04/135.10/136.12
  versus 134.82/138.68/139.36 tok/s). Removed the candidate from Python and
  WGSL source and its test; no slower fallback remains selected.
- Released the experimental resident model (0 GPU buffers, 42 MB WASM),
  stamped the reverted source, and reselected the same local GGUF. A fresh
  build lacking the saved source-version profile used a budget-limited exact
  plan and yielded one correct 110.8 tok/s page reply. Reusing the prior
  correctness-checked full plan on this build and pairing JS/GPU sampling with
  identical 115-token outputs gave GPU 138.50–139.37 versus JS
  115.12–127.50 tok/s across five interleaved pairs. `auto` again selects the
  GPU route; the next fresh load must verify profile persistence.
- Actual `auto` chat-page replies counted 1–40 correctly at
  122.3/135.4/132.4 tok/s. A 115-token callback cost 4.25 ms total;
  increasing Markdown refresh interval from adaptive 200 ms to 1000 ms did
  not improve the three-run product outcome and was not retained. The 140+
  product target, source-version profile behavior, broader model/backend
  acceptance, full regression, and final clean commit remain open.
- A release/reselect of the same native local file on the clean `?backend=webgpu`
  page confirmed the full decode plan and GPU vocabulary sampler were restored
  automatically from device-local profile. Three correct product replies were
  114.6/130.2/135.7 tok/s, with the first cold. Five direct API replies using
  the product's unseeded sampling produced four warm 140.24–141.30 tok/s runs
  and one 126.45 outlier. Delaying page token-event consumption by 16 ms was
  not a repeatable win; 50 ms batching was worse, and an unseeded 4-token
  reply leaves its semantic cause unresolved.
  Both diagnostic wrappers were restored. A full offline upper-composition
  search was then run on the one loaded 0.6B. It completed in 295.9 s after
  four-step semantic checks and chose the same prior plan; diagnostic median
  10.215 ms, no new upper-layer winner. Three post-search direct replies were
  138.03/140.50/133.74 tok/s and three page replies 133.1/135.0/135.8 tok/s,
  all exact 1–40. Search raised main JS heap to ~153 MB; model release returned
  GPU buffers to zero and WASM to 42 MB, and page reload lowered JS heap to
  ~23 MB. The same local GGUF was reselected; its measured plan automatically
  reused. Three clean-reload product replies were correct at
  123.1/134.5/137.1 tok/s. Synchronous pair-coalescing and a collapsed
  resources panel also did not yield a safe speed win; neither was kept.
  Latest automated checks: 161
  Python, 76 JavaScript, webpack build, version stamp and diff check pass.

## 2026-10-03 ▸ Qwen3-0.6B current-build baseline and route diagnosis (open)

- Shifted the active performance check to the one local
  `models/Qwen3-0.6B-Q4_K_M.gguf` as requested. In-app browser native file
  chooser, no model HTTP source, one resident model at a time. Each WebGPU/WebGL
  run ended by release: GPU 0 KB, WASM 42 MB. The user now accepts stable 140+
  tok/s rather than pursuing 150; this target has **not** been met on the
  current build.
- WebGPU product at default sampling: 183 tokens at 118.8 tok/s with an
  incorrect code/table presentation, then a correct 1–40 copy at 115 tokens,
  110.6 tok/s, 216-token context. A timestamp-instrumented 63.4 tok/s run is
  diagnostic overhead, not a production regression. On an uninstrumented fresh
  runtime a seeded complete API reply stayed correct at 120.5 tok/s. The
  opt-in worker timing separated the reported `host` bucket into mean
  GPU-completion/full-logit readback 7.19 ms/token, options preparation 0.015 ms,
  JS sampling 0.64 ms. The old label conflates GPU waiting with host work.
- Lower-level WebGPU stage profile at position 128: complete deterministic
  step 7.1 ms, segmented net sum 6.62 ms; head 1.04, dense MLP 2.16,
  attention out 1.37, QKV 1.20 ms (segmented timings are diagnostic, not an
  additive production prediction). Four separate/fused QKV/MLP whole-API
  candidates were correct with fixed seed and measured 119.0–125.5 tok/s.
  An offline correctness-gated full composition search finished after about
  two minutes, chose fused add/RMS and per-shape stored/DP4A Q4_K routes;
  its correct 1–40 reply was only 123.8 tok/s. That result was persisted and
  auto-reused on the next load, which still measured 120.5 tok/s. Neither is
  sufficient evidence of a stable local win or the accepted 140+ target.
- WebGL product, same local file/prompt: correct 1–40, 115 tokens at 18.2
  tok/s, 2.4 s first token. A separate direct seeded reply measured 20.68
  tok/s; worker split was 39.3 ms/token readback/wait and 0.71 ms sampling.
  Unsplit whole-step WebGL profile at position 128 was 45.5 ms, versus a
  107.7 ms segmented net sum with boundary synchronizations. WebGL still needs
  backend-specific optimization and a device-appropriate accepted speed gate.
- Added an opt-in JS-worker sample-timing aggregate for both backends; Python
  only flips the diagnostic flag and reads the final JSON. TypeScript/webpack
  build and version stamp pass; the broader automated suite and complete
  seven-point acceptance have **not** been rerun/closed for this change.

## 2026-10-03 ▸ WebGL 30B context-loss recovery and upload experiment (open)

- A profiled local-file 30B load showed one `cat2_gl` compile/link/status path
  taking 6994.82 ms after weights had been uploaded, followed by WebGL context
  loss at the required first-forward readback. The failed page retained about
  1.40 GB of WASM heap with Release disabled. The load-failure path now closes
  the entire SDK runtime, releases its GPU context, invalidates the old local
  File registration, and starts a fresh worker. A tiny invalid local GGUF and
  a real 30B context-loss load both recovered to ready(WebGL) with about 42 MB
  WASM and a usable Load button. No model was fetched from the network.
- The generic WebGL growing-KV append shader is now registered before the
  large GGUF weight upload. On the next profiled 30B load it had no long
  compilation log, but the context still failed during the full-forward
  proof. This removes a local compilation stall; it is **not** a complete
  load-time or context-loss fix. A Python regression checks one-time kernel
  registration before weight upload.
- An opt-in 64 MiB periodic WebGL upload flush candidate loaded the sole
  local 13.83 GB model without context loss: warming 2.4 s, proving 33.4 s.
  Its product answer reached only 18 in the 1–40 count at roughly 0.4 tok/s
  before the run was stopped, so it has neither a complete semantic result nor
  a demonstrated performance win; the candidate remains disabled in default
  routing. A later no-flush load also succeeded but spent 35.4 s proving, so
  context loss and timing vary and this A/B does **not** isolate flushing.
- The JS-side WebGL texture ledger now writes directly to existing shared
  memory, closing a backend-parity hole where the product showed GPU memory as
  “—”. The real no-flush 30B load held 12.73 GB after loading and peaked at
  13.33 GB declared texture storage; this excludes driver copies. Its independent
  98-context product reply counted 1–40 correctly: 114 tokens, 2.9 tok/s
  overall, 2.8 s first token, and about 180–185 ms in the last ten GPU steps.
  Release visibly changed GPU buffers to 0 KB and WASM from 1.40 GB to 42 MB.
  Physical free memory fell to tens of MiB and the host compressor held several
  GiB during these runs. The precise driver-versus-OS memory cause and the
  ≤10 s WebGL proof/warm gate remain open. The test tab was closed.
- 0.6B performance has **not** been retested or accepted on this build.
  WebGL 30B load/proving and seconds-level first-token gates, both 0.6B backend
  performance gates, Laya, all seven final acceptance items, and final commit
  remain open. Only one model was resident at a time. Local regression after
  these edits: 160 Python tests, 71 JavaScript tests, TypeScript/webpack build,
  and `git diff --check` pass. The invalid-file browser fixture was removed.

## 2026-10-03 ▸ new-session native 30B WebGL regression (open)

- The earlier file-picker failure was specific to an old in-app browser tab
  that was no longer owned by the current browser-control session; a fresh tab
  selected the **local disk GGUF** successfully. The model loaded without an
  HTTP model endpoint or network download. One full product reply at a
  240-token context counted 1–40 correctly: 114 tokens, 3.8 tok/s, 27.1 s
  first token. This is not a clean-context latency baseline.
- An independent 96-context reply also completed the correct 1–40: 114 tokens,
  1.5 tok/s overall, 3.8 s first token; its first steps took up to 2.1 s while
  the settled tail took about 180–190 ms/token. The load log showed warming
  2.4 s and proving 20.9 s, so the strict ≤10 s load gate **fails**. During
  the run the host showed about 100 MiB free physical memory and 6.5 GiB in
  compressor; this supports memory pressure as a contributor, not a proven
  sole cause. A transient single digit seen while streaming was not a final
  correctness error; the final answer was correct.
- Releasing the sole model returned the WASM heap from 1.40 GB to 50 MB.
  Opt-in WebGL program-compile timing was added to distinguish compile/link
  stalls from subsequent execution, built and locally tested. A diagnostic
  reload currently cannot be counted: two native-file chooser attempts after
  page navigation emitted no chooser event. No HTTP fallback, no second model,
  and no final commit. 0.6B performance remains untested on this source.

## 2026-10-03 ▸ readback memory and synchronous failure handling (open)

- WebGPU and WebGL previously allocated a 64 MiB shared readback arena on the
  first read, even for a 4-byte token. The shared arena now begins at 64 KiB,
  grows to the required power-of-two capacity, and rebinds the main thread on
  growth. The worker still reads directly from the shared buffer, without an
  RPC tensor payload or additional copy. An arena unit test covers reuse,
  growth and rebinding.
- A WebGPU readback could throw synchronously before a Promise existed and
  leave the worker asleep in `Atomics.wait`. It now catches both synchronous
  and asynchronous failures and wakes with an error, as WebGL already does.
  The targeted regression test passes. Full local checks: 159 Python, 68 JS,
  TypeScript build, clean `git diff --check`; source stamps were regenerated.
- The sole local-disk 30B browser retest is **not** complete: the native file
  picker did not emit a chooser event. A separate no-model WebGL test page
  stalled during backend initialisation, so this source is not browser-verified.
  No model was loaded via HTTP, no network model fetch was made, and no commit
  was created. WebGL 30B latency, 0.6B and Laya remain open.

## 2026-10-03 ▸ upload calibration does not duplicate tensors (open)

- Replaced duplicate-upload calibration with alternating natural calls: each
  `set_data` uploads exactly once while a repeated physical shape gathers
  samples, then uses the faster median path. Both browser backends returned
  the same final 16-value tensor after 15 writes, with 7 staged and 7 direct
  measured calls and one settled-path call. A regression test checks that
  calibration never duplicates an upload.
- A failed WebGPU mapped metadata write now unmaps and releases its GPU
  allocation. The complete local run is 159 Python and 65 JavaScript tests,
  TypeScript build passing; `git diff --check` is clean. Full 30B browser
  regression remains blocked by the locked Mac's native local-file chooser.
  No alternate HTTP model route was used; no commit.

## 2026-10-03 ▸ fewer hot-path copies, measured upload choices (open)

- WebGL 2D/2DArray readback now passes its existing shared target directly to
  `readPixels`; WebGPU maps its staging buffer into the same worker-visible
  target. This removes the prior extra main-thread tensor copy, but not the
  unavoidable GPU staging transfer. Browser checks covered WebGL float32,
  half-float, int32, uint8 and 2DArray plus WebGPU arithmetic; results matched.
- MoE host routing now writes indices and weights directly into two regions of
  reusable shared upload memory, without allocating/copying two output arrays.
  Its top-k scratch arrays are reused across rows and exponentials calculated
  once; WebGL and WebGPU browser routes both returned the same 2-row indices
  `[1,2,1,2]` and weights. LLM token-history counts now stay incrementally in
  JS instead of being rebuilt from the entire history every token. Both
  backends' consecutive-token browser check returned `[2,1]` as expected.
- An optional direct NumPy-buffer upload avoids Python's intermediate WASM
  staging copy for exact contiguous physical formats. Paired browser timings
  showed mixed gains/losses by backend and byte size, so it is **not** forced
  globally. A never-repeated shape uses the established upload; repeated
  physical shapes alternate paths across natural calls, with exactly one
  upload per call, and settle on the faster median, with no fixed percentage
  gate. Unit and both-backend browser checks
  verified identical output. WebGPU per-op metadata now moves Python bytes to
  shared JS staging in one copy and a signal-only RPC, eliminating the old
  Python comm-buffer and transferred-array intermediates.
- The SDK cache stamp now covers both generated JS bundles and both backend
  wheels; the wheel install URL carries that version so a browser reload does
  not silently reuse a stale wheel. 159 Python, 64 JavaScript tests and the
  TypeScript build pass. A full 30B reload is not verified on this source:
  the Mac is locked, and its native local-file chooser does not open in the
  in-app browser. CDP cannot set files by protocol on this surface. No HTTP
  model load or network model fetch was substituted. The latest prior WebGL
  30B result remains 2.6 tok/s with a 21.0 s proving stage, so performance
  gates and the broader seven-point acceptance remain open. No commit.

## 2026-10-03 ▸ shared-memory transport and JS token selection (open)

- Both WebGL and WebGPU now batch ordered nonblocking GPU descriptors into
  shared-memory slots; worker↔main hot-path messages only identify ready slots.
  Tensor uploads use reusable shared staging with a completion signal instead
  of transferring a fresh data array in each RPC. An oversized upload arena is
  released after acknowledgement. Both backends' no-model browser arithmetic
  returned `[5,10,15,20]` from `((a+b)*a)` and JS-side greedy sampling returned
  token 2 from `[0,1,2]` without changing the GPU tensor.
- Forced browser WebGL context loss exposed an additional synchronous readback
  throw before Promise creation. This is now caught, sends -1 to the blocked
  worker and raises immediately in Python; no infinite wait. Forced WebGPU
  device destruction also raised immediately. Temporary WebGPU test tab was
  closed, and the user's WebGL tab remains.
- The ordinary unconstrained LLM sampling path now reads logits into JS shared
  memory and applies top-k/top-p/min-p, penalties and selection in JS; Python
  gets only the selected scalar. The worker retains token history, so the
  whole prompt ID list crosses the bridge only on the first selection. This
  **does not** cover constrained sampling or all NumPy/model-loader paths.
- On the sole native-file 30B WebGL model, the complete product reply counted
  1–40 correctly, 114 tokens at 2.6 tok/s, ~180 ms hot tail, 9.7 s first
  token. The runtime reported JS sampling active. Load stages were warming
  2.4 s and proving 21.0 s: the strict ≤10 s load gate is still FAILED. The
  model was released; WASM heap returned from 1.40 GB to 50 MB. No other model
  was loaded concurrently, and no network model fetch was used.
- 155 Python tests, 53 JavaScript tests and TypeScript compilation passed on
  this source. MoE prefill host routing still computes in Python and must be
  moved or removed only after equivalent correctness/performance evidence;
  30B WebGL latency, 0.6B/Laya, full JS ownership and seven-point acceptance
  remain open. No commit.

## 2026-10-03 ▸ JS ownership of hot GPU paths (open)

- The user tightened the architecture requirement: both WebGPU and WebGL must
  keep high-frequency tensor data movement, arithmetic and GPU execution in
  JS/GPU; Python should schedule with handles and make as few bridge calls as
  possible. This is a new acceptance gate, not satisfied by command batching.
- Source audit shows GPU buffers already live on the JS/GPU side, but Python
  still constructs each operation, converts its descriptor through Pyodide,
  performs NumPy sampling and routing, and sometimes reads full logits back.
  WebGPU has a JS-side captured decode replay; WebGL's growing-KV path does
  not. Both backends now queue nonblocking GPU commands in bounded 128-command
  JS batches, flushing before upload/readback. The queue has ordering tests for
  both backends and TypeScript compilation passes. This reduces worker/main
  messages but **does not yet remove per-op Python→JS calls**.
- A confirmed WebGL context-loss failure produced zero-valued readbacks and
  garbage replies. Loss guards now reject uploads/dispatch/readbacks, WebGL
  readback failures notify a blocked worker with -1, and Python raises instead
  of retrying stale bytes. WebGPU keeps the analogous device-loss path. The
  browser product has not yet been reloaded/tested on this source. The single
  30B model was released; no model is currently resident.
- New JS build and WebGL wheel were packed and source stamps updated. The
  30B WebGL warm≤10 s, first-token, semantic and throughput gates remain open,
  as do the broader JS-ownership migration and full seven-point acceptance.

## 2026-10-03 ▸ WebGL 30B paired MoE route and browser loss (open)

- The complete 128-token product response on the sole local Q3_K_XL 30B correctly
  listed 1–40: 114 generated tokens at 2.7 tok/s, 3.7 s to first token and
  about 0.17–0.20 s per token after its slow initial steps. Earlier apparent
  number repetitions were incomplete streamed numerals, not final output errors.
- The corrected underlying-model T=4 A/B showed a positive *combined* device
  router plus device row-repeat result: host 31.648/21.347 s versus device
  21.827/7.812 s, identical greedy token 5 and hidden-state relative error
  8.7e-7. A GPU router or GPU repeat alone did not improve whole-model timing.
  Five paired hot single-layer runs also favored the combined route with
  relative error 3.76e-8. WebGPU and WebGL now expose the same generic
  `repeat_rows` primitive; 151 Python tests passed.
- A new source-stamped browser load (`bdb2e13a3f`) chose `host` in two
  cold, single-layer auto profiles even though the whole-model combination
  favored `device`. It took `warming 3.3s · proving 13.7s`: improved from
  36.6 s, but beyond the strict 10 s limit. A separate forced-device T=4
  complete forward on this same resident model took 20.016 s, returned token
  5 and a finite hidden norm; timing variability remains significant.
- A subsequent single-row browser diagnostic never returned: CDP timed out
  twice and the in-app-browser tab disappeared. It supplies no valid latency
  measurement. The renderer loss/crash mechanism is not yet established.
  There is currently no resident browser model. Do not count the WebGL load,
  first-token, performance, or full seven-point acceptance gates as closed.
  Continue 30B before 0.6B, then Laya; never co-load models.

## 2026-10-03 ▸ WebGL 30B cross-layer residency diagnosis and bounded expert assembly (open)

- On the user-loaded, native-file Q3_K_XL 30B in the in-app browser, a 30-token
  product reply ran at 0.7 tok/s with 30.4 s first token. Its per-tenth GPU
  step curve fell from 5573/3423/2230/2251/2166 ms to about 174–213 ms;
  this is a cold-to-hot cliff, not a stable 0.7 tok/s arithmetic baseline.
  That reply was capped at 30 tokens, so its incomplete 1–40 text is not an
  accuracy result.
- The first synthetic MoE route script assigned controls to the SDK `Model`
  wrapper, not its `.impl`. Its route timings and varying sampled token IDs
  are **invalid evidence**, not a WebGL correctness failure. After correcting
  the harness, T=4 host/device/host/device forwards took
  47.165/47.791/47.774/47.453 s. All chose token 5; final hidden states
  differed by at most 8.7e-7 relative. A GPU repeat-row alternative hit 49
  gathers but still took 47.356 s with identical hidden state and token,
  so neither candidate has a measured positive whole-model result.
- A hot repeated layer-0 MoE MLP took 13–15 ms after its first pass. A
  48-layer serial sweep took 41.931 s (median 0.908 s/layer), close to the
  47-second whole-model forward. System swap-ins and swap-outs rose by about
  9 GB during that sweep. This localizes the cold whole-model cost to
  cross-layer weight residency/OS paging much more strongly than to row
  indexing, but does not prove a unique driver-level mechanism. The product
  panel's GPU pressure is based on the last reply's timing spread, not live
  VRAM occupancy, so it is not used as memory proof.
- GGUF expert assembly now passes memoryviews of the original gate/up bytes
  and concatenates only in the bounded WebGPU expert upload or the single
  WebGL staging array. Six targeted regressions and 147 Python tests pass.
  Browser Release reduced the GPU process from about 14 GB to 551 MB. On the
  stamped reload of the same local file, the WASM heap was about 1.40 GB
  versus 1.48 GB before, but load `proving` worsened to 36.6 s; this change
  is memory hygiene, **not** a WebGL latency fix. A complete 1–40 product
  reply with a 128-token cap is in progress. The ≤10 s load and seconds-level
  first-token gates remain open; no 0.6B/Laya switch or final commit is claimed.

## 2026-10-03 ▸ in-app-browser WebGL 30B and inference graph lifetime (open)

- The sole local `Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf` was loaded through
  the browser's native file chooser on explicit `backend=webgl`; no model was
  downloaded or co-loaded. The WebGL renderer is ANGLE Metal on Apple M5, not a
  CPU software fallback. The initial build spent `warm-state 250.1s` in a
  synchronous full-API composition tournament and answered a 12-token product
  prompt at 0.3 tok/s, 11.6s to first token, with 3273 ms per decode step.
- Interactive WebGL load now reuses a measured device/model profile when available
  and otherwise exposes an exact `profile_pending` baseline; its full semantic
  tournament remains explicitly callable offline. Python and profile roundtrip
  tests pass. The next stamped load removed the 250-second tournament but spent
  26.5s in the first-forward `proving` stage. The user-refreshed load of the same
  local file measured `warming 2.6s · proving 17.5s`. Seconds-level load is
  still unmet; the work was shifted, not declared complete.
- A live WebGL trace at 107 cached rows attributed 19.2s of 21.3s instrumented
  stage time to MoE MLP, with a large instrumentation penalty; uninstrumented
  production is the speed authority. An old-source cache probe after one decode
  found 384 reachable inference tensors and 288 parent edges. `Tensor` now keeps
  autograd parents only when `requires_grad`, with regressions for inference
  release and gradient parent order. On the new stamped browser, the same cache
  roots have zero parent edges. The new 12-token product reply improved to 0.4
  tok/s and normal GPU pressure, but its first token took 34.1s and its decode
  averaged 2114ms; this is memory hygiene, not a solved speed or load gate.
- Explicit WebGL selection no longer claims WebGPU failed. The current model
  remains loaded for serial WebGL operator and layer optimization. No WebGL
  acceptance, 0.6B/Laya work, seven-point completion, or final commit is claimed.

## 2026-10-03 ▸ independent Chrome 30B reload and prompt-size diagnosis (open)

- The only resident model was the same 13.83 GB local
  `Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf`, chosen through Chrome's native
  file picker after the in-app browser released its model (0 KB GPU, 42 MB WASM).
  No model was fetched over HTTP or loaded alongside another model.
- Fresh Chrome load logged `warming 11.6s` and `warm-step 14.5s`, both beyond the
  10-second warm-step acceptance bound. Its first product 1–40 reply was correct
  but ran at 25.9 tok/s with a 55.4s first token: 1168 context tokens, 21223
  prefill dispatches, 47.689s prefill and 7.689s after prefill. Its saved
  decode composition was still budget-limited and had tested no full-API candidates.
- This prompt-size discrepancy was traced exactly, without blaming chat history:
  the conversation contained one 26-character user turn, and the model's own
  tokenizer rendered 98 tokens without tools versus 1168 with the page's seven
  default tool definitions. The tools are a real product capability and were
  restored after a temporary A/B test. A fresh 98-token no-tool product reply
  still took 52.6s to first token (46.442s prefill, 6.188s after), so tool
  inflation is not the sole cause. A subsequent same-prompt, one-token top-level
  API call took 7.403s to first token; another 98-row call took 19.567s.
- On the same resident model, a direct full 1168-row MoE prefill comparison was
  host 41.853s versus device 48.472s, with the same first token ID (16). For
  98 rows, a nearby pair was host 4.490s versus device 10.893s, with the same
  first token ID (13874) and 0.000158 relative final-hidden-state difference.
  The host route varied from 4.49 to 12.79 seconds even on this loaded model;
  neither device-route comparison establishes a positive result. Experimental
  execution overrides and the temporary GPUQueue timing wrapper were restored.
- These observations keep the Chrome warm, first-token, and near-40 tok/s gates
  open. They do not authorize a move to 0.6B or Laya, a performance winner, or
  a final commit. The cause of long GPU queue/residency delays on this host is
  still not isolated, and Chrome's profile/capability path is not directly
  comparable to the in-app browser's saved execution plan.

## 2026-10-03 ▸ actual GPU/WASM release across both backends (WebGPU browser passed)

- A no-model 4-byte copy on a separate WebGPU device finished in about 2.3 ms
  while the 30B's first token waited 18.128 s. The problem is specific to the
  30B queue/resources rather than a whole-GPU outage. The model held 13.79 GB
  of GPU buffers, only 19 MB capture pins and 47 MB of reusable scratch; there
  was no gigabyte-scale pool to trim. These measurements strengthen, but do not
  uniquely prove, a residency/driver-preparation cause.
- Browser Release previously reported `GPU buffers 0 KB` but left the browser
  GPU process at roughly 14 GB and the Pyodide WASM heap at 987 MB. Reloading
  the page dropped the GPU process to roughly 118 MB, proving that model-buffer
  deletion alone was not the final physical resource boundary on this device.
- SDK `close()` now terminates the worker and destroys the backend device/context;
  `wgpy.initMain` exposes an idempotent disposer. WebGPU and WebGL each release
  their captured buffers and context resources at the same API level. The chat
  page uses `release()` followed by `close()` and starts a fresh runtime, clears
  stale local-file IDs, and asks for the native file again. An SDK close test
  covers pending-call rejection, idempotency and backend disposal.
- On the stamped build, the sole local 30B loaded correctly. After Release,
  without a page reload, the GPU process fell from about 14 GB to 813 MB and
  the WASM heap from 987 to 42 MB; the chat then selected the same disk file
  and loaded it successfully. Its first correct 1–40 reply after the restart
  ran at 38.34 tok/s with 4.685 s first-token latency. The two latest loads
  still showed `warm-step 14.1s` and 13.7s, so the strict ≤10 s gate remains
  failed. WebGL browser disposal/parity, repeatable TTFT, and full seven-point
  acceptance are still open. No second model or network model transfer occurred.

## 2026-10-03 ▸ 30B sampled-route browser reload and GPU queue stall (open)

- The full offline composition search now measured all valid candidates on the
  actual sampling/full-logit path, took 112.355 s, and retained the incumbent
  fused/add-RMS plus lower-layer auto plan at a 25.37 ms median. The profile was
  persisted and reused after a source-stamped browser reload (`pick_mode=full`);
  no greedy-only result was substituted for this chat route.
- The same local 13.83 GB GGUF, selected through the native file picker, produced
  correct 1–40 replies at 35.83, 36.96, 36.38 and 34.53 tok/s. Their first-token
  times were 4.428, 6.405, 3.254 and 20.035 s respectively. The last run's
  one-row graph replay spent 19.994 s in the first full-logit readback, while
  input setup and command enqueue were under 3 ms total. A subsequent cold
  load displayed `warm-step 10.2s`, failing the strict 10-second gate; another
  measured 8.8 s. Neither warm-up nor first-token repeatability is accepted.
- An opt-in main-thread WebGPU timestamp run measured 13.419 s and 16.613 s
  first-token latencies on two one-row replays. The first compute passes actually
  occupied only 32.57 and 30.08 ms on GPU; the second passes took 22.68 and
  22.94 ms. The 607,744-byte logits map and independent 16-byte timestamp map
  both waited roughly 13.38/16.58 s after queue submission and resolved together.
  This localizes the outlier to pre-execution queue/residency delay, not slow
  matrix arithmetic, Python input setup, a 607 KB transfer, or page rendering.
  Concurrent 1-second macOS VM samples showed heavy page-ins, decompression and
  swap-ins under a 24 GB machine with the 13.8 GB model resident. Memory pressure
  is strongly implicated but not proven to be the only driver cause.
- A temporary diagnostic change to forward page query parameters to the Python
  Worker was reverted after inspection showed that the WebGPU device and
  `dispatch_flush`/`profile_gpu` parsing already live on the main thread.
  Earlier flush comparisons therefore remain valid. Main-thread timestamp
  instrumentation and the one-token reply cap were restored after the test.
  The SDK/page source stamp was restored to `adbdb8d679` / `297ab34597`;
  the current browser tab remains on the earlier, functionally equivalent
  diagnostic build until the next reload. No model was downloaded or co-loaded.

## 2026-10-03 ▸ real-sampling 30B composition audit (open)

- The user reloaded the same native local 30B GGUF; no model download or second model was
  used. Its saved whole-decoder plan was reused. `warm-step` displayed 10.0 s; the first
  correct 1–40 reply measured 3.486 s first token and 34.46 tok/s, and the repeated
  cached-prefix reply measured 0.212 s and 35.55 tok/s. Earlier identical one-row
  replays still had 7.579 and 12.3 s first-token outliers, so latency repeatability
  and near-40 tok/s throughput are not accepted.
- A 12.3 s replay was instrumented at the actual blocking call: input writes took
  about 0.001 s and `logits.numpy()` waited 12.2256 s. The call samples at temperature
  0.6, so forcing device greedy selection would change semantics. The Mac concurrently
  showed heavy swapping; this is strong residency evidence, not sole-cause proof.
- Native Q3_K MoE route tests matched the output token; host/device hidden states differed
  by at most 0.004459, logits by 0.003614 on a 39.67 scale, and top-16 rank was unchanged.
  Paired timings were unstable under memory pressure, so no route was promoted. MoE
  gate/up and down physical-shape variants were bit-identical on the tested tensors;
  alternatives did not reliably beat the existing `balanced`/default choices.
- The prior complete-decoder tuner timed device greedy even though the product default
  samples and reads full logits. A 49.614 s same-model search of the real readback path
  proposed composed residuals plus separate KV writes, but a direct five-pair comparison
  against the incumbent won only 2/5 pairs and was not adopted. Source now keys plans by
  actual pick mode, keeps a measured incumbent in the final tournament, and times every
  correctness-passing offline candidate rather than only the first four. WebGL's top-level
  tuner likewise runs the default sampling path. Browser verification of this new source
  and the expanded offline search are still pending; 140 Python, 35 JavaScript, and the
  TypeScript checks pass.

## 2026-10-03 ▸ one-row cached-prefix first-token fix (30B source verified; task open)

- A repeated native-file 30B turn reproduced 15.717 s first-token latency with one new
  prefill row and 1,109 prefill dispatches. An instrumented repeat took 12.060 s;
  11.943 s was inside the final head/logit readback, while the following captured
  decode took 0.039 s. System swap counters climbed during another 6.582 s run,
  supporting device residency pressure but not proving a sole external cause.
- A runtime-only test reused the already captured decode graph for the single new
  prefix row. The 1–40 answer stayed correct and first-token latency fell to 1.742 s.
  On the same cached row, full-vocabulary logits differed by at most 0.000979,
  the top-16 ranking was identical, and all 16 paired seeded sampling draws matched.
- The generic SDK now uses that path only with an existing cached prefix, exactly
  one new row, no embedding override, full-attention capability, and an identical
  graph key/buffer plan. Streaming and non-streaming APIs share it; WebGL/CPU and any
  incompatible graph retain the general prefill. The product stats now carry the
  prefill/after-prefill split and route. 135 Python tests, 35 JavaScript tests,
  TypeScript checking and whitespace checking pass; the SDK stamp is `6809862eb3`.
- On the new source stamp, the user-loaded local 30B was the only model. The cold
  product turn answered 1–40 correctly at 33.6 tok/s, 5.493 s first token, and
  `warm-step 8.6s`. The same prompt as a fresh chat then answered 1–40 correctly
  at 34.9 tok/s with `decode_replay` prefill, 0.194 s first token (0.167 s prefill,
  0.027 s after) and zero prefill dispatches. This validates the specific latency
  route; it does **not** establish a repeatable ≤10 s cold bound, near-40 tok/s
  throughput, complete upper-API auto optimisation, WebGL browser results, or the
  seven-point project acceptance.
- Subsequent same-model alternating chat trials exposed a `decode_replay` turn with
  12.261 s first token and zero prefill dispatches. Thus graph reuse removed one
  command-building path, **not** the underlying GPU residency/queue stall; the
  section title's "fix" is scoped to one-row command reuse, not the latency gate.
  Under the same local model, five latest paired 1–40 replies all favoured the
  fused add/RMS + Q/K norm/RoPE route on token throughput, but the first fused
  switch took 14.163 s to first token and another baseline replay took 12.261 s.
  Fused is a positive settled candidate, not yet a safe automatic whole-API
  choice or a cold/first-token latency solution. No alternative model was loaded.
- An explicit offline full-composition search on that same loaded 30B completed in
  52.166 s. It selected fused add/RMS, per-layer `auto` for QKV, Q/K norm/RoPE,
  gate/up, KV write and head shape, plus device greedy when sampling allows it;
  its median complete-step sample was 25.67 ms. The search's synthetic KV rows
  were invalidated before product use. The first post-search 1–40 chat was
  correct but 34.6 tok/s and 14.7 s first token (98 fresh rows); the next was
  correct at 35.63 tok/s and 0.245 s first token with one-row graph reuse.
  This is an actual upper-layer selection, but it does not prove the product's
  best throughput or solve system-paging cold latency. A source fix now always
  invalidates KV prefix metadata after explicit tuner traces, with regression.

## 2026-10-03 ▸ stamped 30B reload and product-path retest (open)

- Reloaded the existing browser page after the SDK source stamp changed and selected the
  same local `Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf` through the native file
  chooser. No hub download or localhost model HTTP transfer occurred. The new Python
  source stamp was `fab66f35a9`; the page showed the 13.83 GB model ready.
- Two correct 1–40 chat turns on the new source measured 33.2 tok/s with 4.6 s first
  token at 98 context rows, then 33.3 tok/s with 2.9 s first token at 243 rows. The
  runtime graph signature was present after the second turn. This is a browser regression
  pass for correctness and a positive first-token observation, not proof of a
  repeatable latency bound or near-40 tok/s throughput.
- The interactive complete-API plan still reported `budget_limited` and
  `tested_candidates: 0`, so `auto` is not yet proven optimal. A runtime-only fused
  add/RMS + Q/K norm/RoPE + device-pick reply measured 34.1 tok/s, while an immediately
  paired baseline/candidate pair measured 33.6/33.7 tok/s. All gave the same ordered
  1–40 answer. This does not establish a repeatable product-path benefit, so the browser
  worker was restored to composed/full and the diagnostic capture freed.
- Native Q3_K MoE gate/up shape testing on the loaded 30B favoured the existing balanced
  route (0.140 ms median). A Q3_K down-projection compact shape initially appeared
  faster, but a 31-pair retest won only 12/31 and had medians 0.166 versus 0.163 ms.
  It was not selected. The active MoE weights are Q3_K gate/up in all 48 layers and
  Q3_K down in 35 layers, Q4_K down in 13; this is a format/shape observation, not a
  model-name routing rule.
- The user clarified that WebGL performance work follows WebGPU optimisation. WebGL
  token/s targets may be lower, informed by comparable industry benchmarks and measured
  same-device ratios rather than one universal multiplier. Its load and first-token
  latency must still be seconds-level, with type-by-type semantic parity and the same
  layered API obligations.
- A fresh same-file 30B browser load with `dispatch_flush=2048` displayed `warm-step
  9.9s`: one successful run but too close to the 10-second boundary to prove a stable
  guarantee. Its first correct 1–40 chat ran at 33.7 tok/s with 3.6 s first token.
  A three-pair loaded-model comparison of 2048 against 512 dispatches measured
  36.8/37.9, 36.3/35.7 and 36.6/35.3 tok/s. Against production-default 1024,
  three pairs measured 35.5/35.8, 37.2/36.2 and 37.0/36.3 tok/s. All answers
  remained correct, but neither alternative wins repeatably. The browser was reset to
  2048; the opt-in live-threshold diagnostic was removed from source and its bundle
  rebuilt/restamped.
- One baseline reply took 18.9 s to its first token despite just one new prefill row.
  This remains a real latency failure even while other settled turns took under one
  second. The same 30B complete-step profile measured 25.945 ms, led by MoE MLP
  12.473 ms, attention output 4.004 ms, QKV 3.157 ms and head 2.343 ms.
- A one-layer Q3_K MoE candidate fused SwiGLU activation into the original-width
  down-projection shader without changing weights. It matched the existing output
  exactly, but 21 paired captures measured 0.0970 versus 0.0963 ms median and it
  won only 13/21. No source kernel was added; diagnostic captures and temporary
  Python/GPU references were released.


## 2026-10-03 ▸ user-confirmed 30B warm run, throughput and chat status (open)

- The sole resident model was the user-selected local 30B Q3_K GGUF. The current
  `dispatch_flush=2048` load displayed `warming 1.9s` and `warm-step 7.4s`;
  the user confirmed latency is met for this run. An actual 97-context chat
  produced the correct 1–40 reply: 114 tokens, 33.6 tok/s, 28 ms GPU + 1 ms
  host per token, 5.4 s first token. Another same-model chat produced the same
  answer at 35.1 tok/s and 1.9 s first token. These are product-path results;
  they do not erase earlier >10 s cold-run variance or meet the near-40 target.
- The loaded model's top-level `decode_plan` says `budget_limited` and
  `tested_candidates: 0`, selecting composed add/RMS, separate QKV, composed
  Q/K norm+RoPE, separate KV write and full-vocabulary greedy readback. A
  stage profile measured about 24.5 ms for a whole token, about 12.7 ms of
  which is MoE MLP. Automatic complete-API routing remains unproven.
- A completed reply retained the composer hint `Stopping…` after the button
  returned to `Send`. The chat source now clears that hint on turn completion
  without erasing a different status; a functional JavaScript regression
  passes. The live tab's stale hint was cleared in place, without reloading
  or releasing its model. Source stamping is updated for a later safe reload.
  All 35 JavaScript tests and `git diff --check` pass. This is not the full
  seven-point acceptance or a final commit.
- On the resident 30B, a runtime-only complete-step check compared composed
  add/RMS plus Q/K norm+RoPE to their fused combination. Four sequential
  full-logit outputs and chosen tokens matched exactly on the tested state;
  nine alternating captured-step pairs favoured fused 8/9 times, with medians
  26.92 versus 24.85 ms. Real chat still only measured 35.6 tok/s on one
  fused run, and a later baseline/fused pair measured 33.6/34.9 tok/s, all
  with the correct 1–40 answer. One diagnostic baseline recapture had 15.2 s
  first-token latency despite just one new prefill row, so capture/paging
  variance is part of the upper-API decision. The experiment reset its
  diagnostic graphs and restored composed/full in the running worker; fused
  is not persisted as `auto` on this evidence alone.
- An alternating same-graph position check gave 25.748 ms at 97 context rows
  versus 25.825 ms at 200 (later slower in only 9/16 pairs). The product
  decode-rate drift is therefore not explained by this context increase alone.
  Another correct baseline 1–40 reply measured 35.3 tok/s but took 7.8 s to
  first token despite just one new prefill row. During that run macOS sampled
  a three-second interval with 42,186 swap-in and 5,372 swap-out 16-KiB pages,
  while wired memory spiked and then returned. This is system-wide correlation
  with severe memory movement, not proof that any one model buffer or process
  alone caused the first-token delay. The user's 30B remains the only model.
- At an idle post-reply boundary, the already-supported GPU idle-pool release
  freed 131,202,540 bytes without changing the 1,833 pinned decode buffers.
  The next correct product reply was slower, 32.9 tok/s with 17.2 s first
  token for one new prefill row. This single run is not a causal proof that
  release itself caused the stall, but it supplies no positive performance
  evidence for automatically draining the pool between replies. No new
  per-reply pool eviction was added; the existing bounded pool/release path
  remains in place on both backends.
- The scalar WebGPU generate/stream APIs always re-recorded `decode` at the start
  of every reply, even with unchanged full-attention buffers and execution
  route. A new capability-checked path retains its output tensor and replays
  that graph across compatible turns; KV capacity, GPU split, device-pick
  semantics, execution overrides and new weight tuning invalidate reuse.
  Recurrent layers intentionally re-record because prefill may replace their
  state buffers. WebGL remains eager behind the same generation API. Both
  streaming and non-streaming mock tests cover single execution, reuse,
  invalidation and synchronous first-token accounting; full local suites pass
  (132 Python, 35 JavaScript, TypeScript,
  whitespace). The SDK source stamp is updated. The live 30B browser worker
  still runs the older source and was not reloaded, so the size of the actual
  first-token win and any browser-only regression are **unverified**.

## 2026-10-03 ▸ early transfer-window release trial (warm gate still open)

- The GGUF loader now releases the CPU/JS transfer window immediately after
  the last weight upload, before model warm-up, on the shared WebGPU/WebGL SDK
  path; the host's final release remains idempotent. This is a resource-lifetime
  correction, not a proven latency optimisation. A backend-routing unit test
  covers both implementations.
- Three same-file, same-build native-file cold 30B loads at
  `dispatch_flush=2048` measured first/second decode GPU sync at
  8.966/0.042, 15.899/0.050 and 13.789/0.042 seconds. The 8.966-second run's
  `warm-step` UI stage was 9.1 seconds, and a correct 1–40 product reply ran
  at 34.6 tok/s with 2.9-second first-token latency. The next two cold runs
  fail the strict ≤10-second gate; do not attribute a stable warm-up gain to
  this change. No other model was loaded or downloaded.
- During the third local cold load, 1-second macOS VM samples showed available
  pages falling from roughly 807,000 to below 4,000 (16 KiB/page), while the
  compressor grew from roughly 620,000 to 1.4 million pages and bursts of
  20,000–54,000 swap-out pages/sec occurred. This strongly supports physical
  memory pressure as part of the cold queue wait, but does not exclude lazy
  GPU-driver work. It is system-wide evidence, not proof of which process owns
  every page. The 30B warm gate remains open; no 0.6B or Laya test has replaced
  this priority.
- The third cold load also answered the same 1–40 chat prompt correctly at
  35.1 tok/s; first token was 6.1 s for 98 prompt rows. A post-load GC
  diagnostic collected only three objects and did not change GPU bytes
  (13,723,627,352) or pinned bytes (zero before the next reply), so an
  unreaped Python cycle after load is not the observed resident-memory gap.
- After the source change and SDK version stamp: 127 Python tests, 34 JavaScript
  tests, TypeScript, wheel consistency and `git diff --check` all pass. This is
  not the full seven-point browser/format/API acceptance and no final commit
  has been made.

## 2026-10-03 ▸ 30B 512-dispatch and loaded-model fusion check (open)

- The user-selected sole local 30B GGUF was cold-loaded with the diagnostic
  `dispatch_flush=512` setting. Construction took 16.7 s; the first/second decode
  GPU synchronisations took 18.166/0.950 s. A correct 1–40 product reply ran at
  36.5 tok/s with 2.8 s first-token latency. This is not a warm-up win against
  the prior 2048-dispatch run (10.393/0.041 s sync, 34.5 tok/s), nor paired
  evidence that 512 is the best complete-API route.
- On that already-loaded model, a runtime-only fused add/RMS plus Q/K norm/RoPE
  plus device-greedy candidate still returned 1–40 correctly. Its first use
  was 36.1 tok/s with 18.5 s first-token latency; the next fresh-chat reply
  was 37.6 tok/s. The first-use stall and lack of a paired cold comparison
  disqualify automatic production selection. The worker's diagnostic execution
  fields were restored to the original composed/full route after the test.
- OS sampling with the model resident showed the Codex service using about
  14 GB, physical memory 23 GB used of 24 GB, roughly 200 MB unused, and an
  11 GB compressor. This makes memory pressure a serious candidate for the
  cold queue wait, but does not isolate it from driver first-use work. The
  30B ≤10 s warming gate and near-40 tok/s target remain open; 0.6B and Laya
  have not been loaded concurrently or advanced ahead of this gate.
- On the same loaded model at decoder position 98, five output-head shape
  candidates were screened on a complete captured decode. `balanced` first
  appeared 0.49 ms faster than the baseline, but 30 interleaved pairs split
  exactly 15/30 (24.190 versus 23.915 ms medians); `shortk` won only 11/30
  (23.500 versus 23.685 ms). Both preserved the argmax and had <6e-7 relative
  logits error. Neither has a repeatable positive win, so neither was selected.
  Diagnostic captures were reset, pinned ids returned to zero and both head
  blocks were restored to their original `auto`/null setting.

## 2026-10-03 ▸ 30B loaded-session rerun and stream state correction (open)

- The sole browser model was again the same 13,833,048,480-byte local 30B Q3_K GGUF,
  selected through the native file input. Baseline cold first decode sync was 10.503 s,
  above the strict 10 s gate; its correct 1–40 chat reply was 34.4 tok/s. On that loaded
  instance, the previously numerically checked fused add/RMS and Q/K norm/RoPE pair
  produced the same ordered answer at 36.4 and 36.6 tok/s. This is a candidate, not
  a proven complete-API auto route or a near-40 acceptance result.
- A diagnostic-only duplicate `createComputePipelineAsync` trial finished 29/29 promises
  without errors but had 11.729 s cold first-sync on the same local file. It did not
  meet the 10 s limit or prove a gain, so its code and URL gate were removed. A separate
  runtime-only fused-plus-device-argmax product reply gave 37.0 tok/s and the same 1–40
  content, but its cold 98-row first-token time was 13.1 s; that does not establish a
  complete-API route win. The unmodified source is rebuilt and restamped.
- Source review exposed a separate correctness/performance bug: streaming generation
  executed a decode step while recording the graph and then replayed that **same** token
  position. This wastes one step and doubles the recurrent-state transition. It now
  consumes the capture's logits just as non-streaming `generate` does. Both captured
  WebGPU and non-captured/WebGL paths stop before a forward for the final output token.
  Three new mock regressions cover the token sequence, execution count and committed KV
  prefix. Complete local tests: 126 Python, 34 JavaScript,
  TypeScript and both wheel consistency checks pass. No final commit; 30B warm gate open.
- After reloading the stamped Python fix, the same local 30B gave a correct 1–40
  streamed reply at 34.5 tok/s; the worker source contained the captured-step
  correction. Cold first sync was still 10.393 s, so the warm gate remains open.
  At decoder position 0, a GQA single-kernel candidate matched four sequential
  greedy tokens within 1.8e-5 relative logits error and won 25/30 full-graph pairs
  (23.385 versus 24.028 ms). This was **not** the real chat context. At actual
  position 98, it won only 19/30 pairs (23.857 versus 23.895 ms), output differed
  by 6.1e-5 relative, and a runtime-only product reply slowed to 33.8 tok/s.
  It was not selected; browser diagnostic overrides were restored.
- A source-gated early-MoE test queued one single-row MLP immediately after each stacked
  expert layer loaded, attempting to overlap first-use work with the subsequent GGUF
  reads. On the same local 30B the model-construction interval was 17.1 s, but the
  first decode sync worsened to 17.675 s versus 10.393 s on the prior baseline run.
  It failed both the strict warm gate and total-work test, so the experimental source
  flag and warm calls were removed and the production assets restamped.
- A second gated trial selected the numerically checked fused add/RMS plus QK-norm/RoPE
  route *before* the first decode warm step, rather than only for steady replies. The
  same local 30B constructed in 16.8 s but first sync took 12.041 s, slower than the
  10.393 s baseline and still above the limit. Warm-stage and steady-state choices
  therefore cannot be conflated; the diagnostic flag was removed and assets restamped.

## 2026-10-03 ▸ 30B first-decode submission root cause and verified correction (open)

- Reused the user-selected local 30B Q3_K GGUF exclusively; no HTTP model load or second
  resident model. A 50-pass WebGPU timestamp trace of two warm decode steps recorded
  exactly 1061 dispatches each. The first was fragmented into 49 passes (65.99 ms total
  GPU time) while the second was one pass (61.67 ms). Host warm-step stage was 19.52 s,
  including 16.99 s first sync. Pipeline creation on the JavaScript host was 0.09 ms
  total across 25 registrations, but that alone does not rule out lazy driver work.
- The fragmentation matched 48 MoE layers: each first T=1 route allocated a zeroed
  tiny weight buffer. `setDataRaw` flushed pending work before each upload, even though
  `moe_route` overwrites all output slots before consumers read them. Replaced this with
  a device-native empty buffer for the shared WebGPU/WebGL layer contract and added a
  regression using NaN-filled mock storage to prove complete overwrite.
- Same-file, same diagnostic build after the change: first warm step became one 1061-
  dispatch pass (52.10 ms GPU execution), second one pass (40.30 ms). First sync fell
  to 10.435 s and second to 0.043 s; host stage was 10.573 s. Correct browser output
  remained 1–40, with 33.5 tok/s and 3.0 s first token. This is an improvement but the
  ≤10 s warm limit and near-40 tok/s target remain **open**. First-request latency has
  shown large variance (a prior same-model reply took 29.0 s), still unresolved.
- Two further uninstrumented cold loads of that same local file measured 13.085 s and
  12.942 s first sync, so the 10.435 s diagnostic run cannot be called a pass. A queue
  completion probe on another cold run showed its critical submission took 0.005 ms on
  the JavaScript host yet `GPUQueue.onSubmittedWorkDone()` resolved at 12.395 s, matching
  Python's 12.351 s wait. Thus the remaining wait is in GPU queue completion, not the
  JS upload call, `mapAsync` readback, or the chat UI. GPU timestamps show the arithmetic
  itself is only tens of milliseconds; driver-side compilation/residency is not yet split.
- System VM sampling: with this 13.8 GB model released, swap activity fell to essentially
  zero; during a loaded idle sample it reached about 1,091 swap-ins and 7,540 swap-outs
  of 16 KiB pages in one second. During a local reload, physical free pages fell from
  roughly 770,000 to under 5,000 and compressed pages rose sharply. This supports
  memory pressure as a contributing condition, not proof that it is the sole cause of
  the GPU queue wait. Production browser replies after the fix were correct at 35.7
  tok/s baseline and 36.1 tok/s for a runtime-only fused combination; the latter has
  not been made an automatic route based on those isolated product readings.
- TypeScript, 33 JavaScript tests, the 15-test MoE route subset and `git diff --check`
  passed after the change. A subsequent full collection of `test/test_*.py` passed
  123/123; all 33 JavaScript tests, TypeScript, wheel consistency and whitespace checks
  passed. The WebGL browser check and final acceptance
  matrix still need rerun. No final commit has been made.
- A separate failure-path audit found `WebGPUTensorBuffer.getDataRaw()` could strand its
  temporary GPU readback buffer if `mapAsync`, copy encoding, or the data copy threw.
  It now destroys that buffer in `finally`, and a unit test covers successful readback
  and rejected mapping. WebGL reads through its persistent framebuffer and allocates
  no corresponding temporary GPU readback buffer. This does not claim a throughput gain.

## 2026-10-03 ▸ Same 30B browser A/B and progress-log correction (open)

- The only resident model was the same native-file-picked 30B Q3_K GGUF. A reverse layer
  upload-order trial gave 14.17 s first decode GPU sync; a same-build default-order
  follow-up gave 12.09 s (second sync 0.28 s). Upload order therefore has no proved win;
  its diagnostic source branch was removed. Default-order load was ready after 24.5 s of
  model construction and answered 1–40 correctly at 35.0 tok/s, first token 5.5 s.
- Layer profiling on that instance: unsplit whole step 24.13 ms; MoE MLP 12.16 ms, QKV
  3.24 ms, attention out 3.42 ms, head 2.13 ms. A real-file routed-Q3 shape comparison
  found the current balanced gate/up and default down shapes fastest among the tested
  original-width variants; alternative outputs differed by at most 3.5e-7 relative.
- The fused add+RMSNorm plus Q/K norm+RoPE candidate matched full-step logits exactly
  on the tested input. An expanded 20-pair, order-alternated, complete-graph measurement
  favoured it 17/20 times: median 24.71 versus 25.32 ms, one-sided exact sign-test
  p=0.00129. Ten actual default-sampling browser replies all contained 1–40 in order;
  fused and composed page rates remained roughly 35–38 tok/s and did not reach the
  30B target. This candidate is not yet an automatic, profile-persisted API route.
- Fixed the chat load-stage log reset at the start of **every** load attempt. Native-file
  loads may emit no `reading` stage, which previously made the status and console append
  timings from earlier loads. A regression test passes. Browser verification of this
  source change awaits a later safe reload; the currently loaded 30B was not discarded.
- Rebuilt both Python wheels and updated the asset stamp after removing the unsuccessful
  upload-order trial. Current subset checks: 119 Python, 32 JavaScript, TypeScript
  type-check, and `git diff --check` pass. Warm ≤10 s, 30B near-40 tok/s, WebGL browser
  parity and the remaining seven-point acceptance gates remain open. No commit made.
- WebGPU's stacked MoE constructor was synchronising a four-byte transpose flag after
  **every expert**. It now keeps source and metadata buffers alive in bounded 32-expert
  windows and synchronises once per window. WebGL retains its equivalent one-pass stack
  at the same `GGMLMoELinear` interface. Three new mock tests cover lifetime, bounded
  windows and WebGL routing; the full Python suite rose to 122 passing tests.
- On fresh workers with the same local 30B and same source build, a runtime transport
  control measured window 1 at 19.4 s model construction / 17.34 s first GPU sync,
  window 8 at 16.9 / 17.62 s, and window 32 at 16.8 / 16.20 s. The default 32 window
  improved construction against its same-build single-expert control and answered 1–40
  correctly at 35.5 tok/s, first token 2.7 s. It did **not** meet the ≤10 s warming gate;
  no warm-latency benefit is claimed. The running worker's experimental window was
  restored to 32 after the control.

## 2026-10-03 ▸ 30B upper-API candidate and cold MoE residency (open)

- The same locally picked 30B Q3_K GGUF was the only resident model. Four-step sequential
  logits/token checks passed for fused add+RMSNorm and fused Q/K norm+RoPE. Fused QKV was
  numerically eligible but about 47 ms versus 24 ms per full step, so rejected.
- Five interleaved complete-generation pairs (90 tokens, greedy, same prompt) all favoured
  the combined fused add/QK route: median 40.99 versus 39.38 tok/s. Browser chat with the
  live combination still answered 1–40 correctly at 37.8 tok/s under default sampling.
  This is a verified candidate, **not yet automatic production routing**; the two API
  conditions and the initial context work differ.
- A bounded large-upload path now waits for each tensor's queued GPU copies to finish
  before destroying staging sources and acknowledging the blocked worker. Unit tests cover
  byte identity, source lifetime, and asynchronous success/failure wakeup; TypeScript
  type-check and JS build pass. The first fresh same-file load still had 16.06 s of first
  decode GPU sync, so this change is not accepted as a warmup fix.
- A second cold load with an ephemeral per-stage probe put 15.81 of 16.66 s in MoE MLP,
  primarily layers 9–24 at 0.7–0.97 s each. QKV summed to 0.35 s. The ordinary step
  immediately after this probe needed 1.97 s then 0.044 s of GPU sync. Browser pipeline
  creation totaled 0.085 ms over 33 pipelines. The data points to cold expert-weight
  residency or paging, but that causal hypothesis still needs a controlled fix/test.
- Both model releases returned WebGPU buffers to zero. No network model download, second
  model, commit, or completion claim. The ≤10 s warm gate and remaining seven-point gates
  are open.
- Bounded direct `queue.writeBuffer` was separately tested on the same local model. First
  decode GPU sync was 28.38 s, worse than the bounded mapped-staging trial's 16.06 s;
  the direct-write candidate and its flag were removed from source and the JS bundle
  restored to the mapped-staging build.
- A runtime-only GPU kernel touched the pages of the eight actually selected experts
  before their first MoE projection. It read 96 stacked buffers; the corrected cold load
  measured 14.41 s first GPU sync against 16.06/16.41 s on two bounded-staging loads.
  With non-paired variance and the ≤10 s gate still failed, it is **not** selected. The
  monkeypatch was restored and all 96 temporary GPU scratch buffers were released.
- Releasing only completed idle GPU scratch immediately before the decode warm step
  recovered 280,951,332 bytes in 0.0017 s on a fresh same-file load. The first/second
  GPU syncs were still 15.147/1.865 s, so the route did not meet the ≤10 s gate.
  The safe pool cleanup was retained in source for both WebGPU and WebGL and has a
  mock-backed regression test; it does not release weights or capture pins.
- A separate runtime-only trial skipped the whole-model decode warm step after releasing
  and re-picking the **same local file**. The later required full-forward proof then took
  11.5 s by itself; the first independent 98-context chat answered 1–40 correctly at
  32.4 tok/s, 2.6 s first token, but its capture pinned 1,929 buffers rather than the
  previously warmed route's 1,833. The 96 additional pins may be persistent MoE routing
  buffers first allocated inside capture; pin counts alone do not prove extra dispatches.
  This moves cold work; it
  neither meets the ≤10 s loading gate nor establishes an API win. The monkeypatch was
  restored and no skip was added to source.
- Reordering the existing correctness proof before decode warming and reducing its
  probe to two positions was also tested only in the running worker. The first proof
  still took 12.296 s, and the subsequent two-step decode warm took 0.559 s
  (first/second GPU sync 0.437/0.024 s). Combined 12.855 s misses the same gate;
  the original methods were restored. The same local GGUF was the only resident model.
- After the pool experiment, the running worker's temporary warm-step monkeypatch was
  restored. The Python suite passed 117 tests, JavaScript suite 29, TypeScript type-check
  and diff whitespace check passed. These are subset gates, not final acceptance.
- Browser reload on the source-stamped build using the same local GGUF confirmed the
  local-file status no longer claims a cache copy. The browser's IndexedDB now has a
  `webgpu/apple/metal-3//` key distinct from the legacy unprefixed key, and contains the
  measured flash/format entries for this backend. The actual first decode GPU sync was
  still 17.210 s (second 0.767 s), so warm latency remains open. Host profile keys now
  distinguish WebGPU/WebGL and cache the identity once per worker; GQA, flash and several
  layer-composition choices have validated profile round-trips. Unit suites: 119 Python,
  31 JavaScript; this does not replace a WebGL browser run.
- On that source-stamped WebGPU page, the first independent 98-context chat returned the
  correct 1–40 sequence (114 tokens) at 36.7 tok/s; first token was 6.7 s, including
  1.5 s of seven new prefill shape measurements. This is a product-path result, not the
  fused upper-route candidate's result and not evidence that the 17.210 s warm gate passed.

## 2026-10-02 ▸ 30B cold-warm root-cause investigation (gate remains open)

- Serially released and reloaded only the same disk-backed 30B GGUF through the browser
  file picker. The model was never fetched over HTTP and never co-resident with another
  full model. A release plus explicit cleanup returned WebGPU buffers to zero; one
  diagnostic reference temporarily held a 124 MB head buffer and was cleaned before
  the next load.
- An interactive-load check based on *remaining* time still let a 16.6 s semantic oracle
  recording start, because the operation is indivisible. Loading now selects the exact
  original-width composition without starting that oracle; explicit offline composition
  tuning remains available. An automated regression tests both expired and ample future
  deadlines. The source also records first/second decode dispatch and GPU-sync timings.
- Removing the oracle did not meet ≤10 s. Baseline 30B `warm-step` runs took 13.3 and
  18.3 s at dispatch flush 2048, and 15.4 s at flush 512. Their first GPU synchronisations
  were 12.74 and 14.69 s, while host dispatch was about 0.05–0.10 s and the second GPU
  synchronisation was 0.41–0.61 s. A further baseline run under heavier system pressure
  took 40.7 s warm-step, 37.79 s of it first GPU synchronisation. GPU pipeline creation
  accumulated under 1 ms, and bind-group creation only a few ms during these runs.
- Two opt-in upload hypotheses were tested on the real local model and rejected: touching
  9.1 GB of buffers during upload made shape warming 6.8 s and warm-step 27.6 s; replacing
  staging copies with `queue.writeBuffer` for 13.4 GB made warm-step 21.6 s. Both candidates
  were removed and the checked-in JS bundle rebuilt. The 512-flush product run answered
  1–40 correctly at 37.0 tok/s versus a separate 2048-flush run at 35.4 tok/s; that
  unpaired comparison does not establish a route winner.
- During a serial baseline load on the 24 GB Apple M5, system VM counters registered large
  increases in decompressions and swap-ins while available pages became scarce. This is
  consistent with cold resource residency, but a controlled causal test is still needed.
  Do not label the ≤10 s warmup or the full seven-point acceptance matrix complete.
- Python unit suite: 116 passed. Browser worker/source and HTML were rebuilt/stamped after
  removing negative candidates; no final commit has been made.

## 2026-10-02 ▸ Loaded 30B MoE routing/reduction measurements and WebGL parity repair

**Trigger:** The user had the local 30B GGUF loaded and asked that its real API path,
first-answer latency, memory and warmup be improved without another model download.

### Changes

- Generalised the device MoE router to a batch of token rows. A previous per-prefill-layer
  host readback was measured at 26.89 s across 48 layers during a 27.49 s cold prefill.
  A first-layer auto calibration that executed both routes was measured negative and removed;
  route choices now accept device/backend/format/shape/cold-or-warm profile data with a host
  fallback, plus explicit containing-layer overrides.
- Simplified normalised top-k by cancelling the full-expert softmax denominator on both GPU
  backends. The non-normalised branch retains the full softmax. This does not change the
  quantised weight representation or activation width.
- Fixed a real WebGL shader compile failure: `flat` is a GLSL qualifier, so the batched
  router's local variable prevented both shaders from compiling and silently yielded zeros.
- Added an equivalent WebGPU/WebGL fused routed-expert weighted sum, leaving both the fused
  and composed routes addressable at the operator and containing-layer levels.
- Corrected the source first-token timer to include the post-prefill decode-graph capture.
  The already-loaded browser worker predates this timer change, so its UI timing is not
  accepted as post-capture latency evidence.

### Measurements and correctness evidence

- WebGPU router: old/new indices identical for 1, 4 and 98 rows, both normalised and
  unnormalised; worst weight relative difference 3.1e-7. Isolated normalised routing won
  9/9 paired rounds at 1 and 98 rows. Complete captured 30B token step at position 98:
  old median 27.795 ms, new 26.845 ms, new route faster 8/9 paired rounds.
- WebGL real browser, with no second model loaded: fixed router matched an independent
  NumPy top-k/softmax reference for 1, 3 and 98 rows, both normalisation modes; worst
  weight absolute error 1.7e-7. Fake-projection MoE layer host/device outputs differed by
  at most 7.5e-9. The original shader returned all-zero outputs before the compile fix.
- Fused weighted sum: WebGPU and WebGL outputs matched the composed operator to about
  1.6e-7 relative at the real 30B shape (k=8, H=2048). The isolated primitive won 9/9
  paired rounds on both backends at 1 and 98 rows. On the real 30B, four interleaved
  full-step combinations gave medians 25.315/25.070 ms (old router, composed/fused) and
  24.395/24.210 ms (new router, composed/fused); all picked the same token. The fused
  route by itself had been inconclusive in an earlier 5/9 paired comparison, so the
  combined whole-API result matters.
- Real chat with the same loaded 30B, runtime hotpatched but no reload: correct ordered
  1–40 response, 114 tokens, 36.2 tok/s, GPU 26.08 ms and host pick 1.19 ms, versus
  32.6 tok/s immediately before these candidates. The prompt hit a one-row prefix cache,
  so it does not establish cold prefill or warmup performance. The current browser URL's
  build stamp still refers to the pre-hotpatch build.
- Local tests: 114 Python and 29 JavaScript passed. `pytest test` without a file filter
  incorrectly collects the Pyodide-only top-level-`await` runner; the valid local command
  is `pytest test/test_*.py`.

### Explicit limits

- Warmup ≤10 s is **not** verified; a live repeat of two decode warm steps took 16.13 s
  once and 0.18 s when settled. The initial load's exact semantic recording remains the
  next critical-path investigation. The source timer fix also awaits a fresh browser build.
- No full 30B WebGL model was loaded alongside WebGPU. WebGL evidence here covers the
  real GPU operators and equivalent MoE layer interface, not 30B product throughput.
- The new operator choices are not yet integrated into an automatic, persisted top-API
  composition tuner for fresh builds; the loaded-tab run used explicit runtime overrides.
  The seven-point acceptance matrix is therefore still open. 0.6B and the BERT-based Laya
  decision path follow the remaining 30B gates, in the user's order.

### Unchanged

- The local 30B model remains loaded in the original browser tab. No model bytes were
  downloaded or read over localhost HTTP; only SDK source was fetched for hotpatch tests.

## 2026-10-02 ▸ 30B latency diagnosis, not yet accepted

- Direct local GGUF WebGPU load and full-chat generation produced the correct ordered 1–40
  output twice at 34.4 and 33.5 tok/s. Model release returned GPU buffers from 13.19 GB
  to effectively zero; no second model was resident.
- The first bounded-warm build still spent 23.1 s warming. Fine-grained stage reporting
  located 20.2 s in the four eager sequential logits forwards used as the exact-width
  semantic oracle. This is above the required 10 s and remains open.
- First-token waits of 13.6 and 17.0 s were prefill, not decode: the later turn reprocessed
  436 prompt rows after a page reload, including 11,816 GPU dispatches. Browser-persisted
  tuning held weight-execution buckets 1 and 4, but not the larger batch bucket reached
  by that prompt. A first-time measurement was therefore on the answer's critical path.
- Current unverified fix captures and replays the exact sequential oracle instead of four
  eager command streams; it registers nested stacked-MoE weight shapes before the oracle,
  trims redundant large-batch calibration repetitions while retaining paired sign-test
  evidence, and saves new shape decisions after generation as well as after load.
- Next gate: rebuild/stamp, one local 30B browser rerun, verify both warm ≤10 s and
  first-token latency with correctness and GPU/CPU memory. Do not claim completion before
  that evidence; then 0.6B, then Laya decision model, in the requested order.

## 2026-10-02 ▸ Correction — no arbitrary phase-two gain cutoff

**This supersedes the percentage-margin statements in the earlier completion entry below.**

- Removed the fixed latency-gain cutoff and the special-case that skipped M=1/M=2. Routing
  now uses nine interleaved paired rounds and an exact one-sided sign test. A stable small
  win is enabled; inconclusive timings keep the lower-memory implementation.
- Connected activation-INT8 DP4A to production `QuantizedLinear(auto)` routing on WebGPU.
  Candidate availability and numerical correctness are checked before it is timed, and the
  cache key remains model-agnostic: family, stored format, K/N, shape bucket and device
  runtime only. WebGL converges at the same `QuantizedLinear.forward` result contract.
- Re-ran the target WebGPU. DP4A speedups versus exact stored were INT4
  `0.569/0.946/0.546/0.503×` and INT8 `0.612/0.928/0.384/0.393×` for M=1/2/32/128, so every
  bucket selected stored from negative or inconsistent paired evidence—not from a minimum
  gain rule. AutoGPTQ zero-offset validation also passed below 0.7% maximum relative error.
- Re-ran stored versus materialized across all accepted formats. Stable local wins remain
  enabled (for example F16 M=8/32/128 and several packed M=128 buckets), while noisy results
  are explicitly reported as inconclusive and keep stored memory use.
- Final browser gates: 140/140 native GGML cases on WebGPU and 140/140 on WebGL; phase-two
  WebGL explicit-equivalence gates; local Qwen3 native Q4_K/Q6_K smoke `OK` on both backends.
  Automated gates: 67 Python tests and 19 JavaScript tests.

## 2026-10-02 ▸ Two-phase completion and efficient parity

- Completed the original-width phase for all accepted GGML formats and GPTQ INT4/INT8 on
  both WebGPU and WebGL. Backend-, format- and batch-specific scalar/vec4 routes retain local
  wins instead of requiring one global winner.
- Completed measured stored-versus-materialized routing on WebGPU. Decisions are cached by
  family, format, K/N and shape bucket in the device-specific kernel profile. The original
  percentage rule recorded here is superseded by the correction above.
- Implemented and measured activation-INT8 DP4A as the phase-two cross-width candidate. The
  benchmark-only conclusion recorded here is superseded: DP4A is now a production candidate
  and is selected per device/format/shape from stable paired evidence. WebGL declares the
  primitive unavailable and converges at the `QuantizedLinear.forward` contract.
- Added the WebGPU/WebGL efficient-common-layer manifest and executable parity tests. Scope
  descends global → backend → format → operator mode → shape bucket → device profile;
  absence of a global win never discards a local win.
- Fixed the WebGPU backend wheel so partially constructed buffers cannot raise noisy
  destructor errors during alternative-allocation probes.
- Fixed two WebGL end-to-end gaps found only after operator gates passed: the CausalLM smoke
  proof now uses the growing-cache path when capture is unavailable, and native 3D batched
  matmul now indexes every batch/head instead of writing only batch zero.

### Evidence

- WebGPU and WebGL: 28 GGML formats × 5 labelled cases = 140/140 per backend; WebGL also
  executes both M=3 and M=33 batch shader variants inside its GEMM gates.
- GPTQ INT4/INT8: both backends match an independent NumPy dequant reference, worst relative
  error `2.15e-6`.
- Independent Qwen2 reference: WebGPU and WebGL maximum logits error `4.84e-8`, identical
  argmax and 8/8 greedy tokens. WebGL first-layer attention error after the atomic BMM fix is
  `7.45e-9` (previously `0.1178`).
- Local Qwen3-0.6B Q4_K_M: deterministic `OK` on both backends with 168 Q4_K and 28 Q6_K
  native linears; WebGPU capture and WebGL growing-cache generation both pass.
- WebGL training: loss `0.614 → 0.0`, accuracy `1.0`.
- Automated: 66 Python tests, 19 JavaScript tests; wheel freshness and diff checks pass.

## 2026-10-02 ▸ Non-stop two-phase completion rule

**Highest project principle:** Neither a partial benchmark nor a commit is a stopping point.
Work continues until phase one has completed all same-width correctness and applicable
hardware optimisation gates, then phase two has completed the full measured performance
comparison and production routing. Both phases, not merely their start, are required.

## 2026-10-02 ▸ Phase-one exact Q6_K vector kernel

- Added a same-width Q6_K candidate that reads four original low-nibble bytes and their
  two-bit high plane together, expands only in registers, and keeps FP32 activations.
- Fixed the benchmark fixture so an already-present helper is not emitted twice; the old
  duplicate made WGSL registration fail and compared a stale pooled buffer as if it were a
  candidate result. The numerical gate caught it before timing was accepted.
- Two independent K=4096, N=3072 runs measured Q6_K at 1.19/1.21/1.22× and
  1.17/1.28/1.22× for M=1/32/128, so the vector path is enabled for every shape.

### Evidence

- Chrome WebGPU stored matrix: 28 formats × 5 cases = 140/140 after enabling Q6_K.
- Local Qwen3-0.6B Q4_K_M: 168 Q4_K + 28 Q6_K native linears, deterministic `OK`,
  103 ms TTFT, captured decode enabled.

## 2026-10-02 ▸ Local Qwen3 native-format end-to-end gate

- Corrected the distinction between browser-persistent model cache and repository-local
  model files: the cache settings page intentionally does not enumerate disk files.
- Added a repeatable browser smoke test for `models/Qwen3-0.6B-Q4_K_M.gguf`. It fetches the
  served local file with persistence disabled, explicitly requests native weights, checks
  the live model's packed Linear formats, and performs deterministic generation.
- Updated the old GGUF example and loader comments that still described every GGUF as being
  converted to INT4, which has not been true since stored-format kernels became the default.

### Evidence

- Chrome WebGPU: deterministic reply `OK`; 168 Q4_K + 28 Q6_K native Linear modules;
  load 1.2 s, TTFT 112 ms, captured decode enabled.
- Python suite: 49 passed. JavaScript suite: 19 passed.

## 2026-10-02 ▸ Phase-one exact Q5 vector kernels

- Added register-vector candidates that consume Q5_0/Q5_1's original four-bit nibble stream
  and separate fifth-bit plane directly. FP32 activations, block scales, minima, and output
  accumulation semantics are unchanged.
- Realistic K=4096, N=3072 interleaved runs showed Q5_0 at 1.00–1.13× and 1.01–1.11×
  across M=1/32/128, so its exact vector path is enabled globally.
- Q5_1 batch execution was consistently positive (M32 +7–8%, M128 +10–11%), while M1
  contradicted itself (+8%, then −12%). Production therefore keeps exact scalar GEMV/GEMV2
  and selects exact vector execution only for GEMM. This is same-width operator routing,
  not the deferred cross-width phase.

### Evidence

- Final Chrome WebGPU matrix: 140/140 stored-format operator cases; decision-feature CPU/GPU
  comparison passed.
- Python suite: 49 passed. JavaScript suite: 19 passed.

## 2026-10-02 ▸ Phase-one exact Q4 vector kernels

**Constraint:** Optimise within the source format first. No weight re-encoding, activation
requantisation, or materialised alternate-width comparison may decide this phase.

- Added register-local packed-nibble vector decoders for Q4_0, Q4_1, and Q4_K. Four source
  bytes are read together, split into their original low/high Q4 values, scaled with the
  format's exact block parameters, and accumulated against unchanged FP32 activations.
- Added an interleaved same-width scalar/vector benchmark. It labels itself phase one and
  cannot be used as evidence for cross-width routing.
- Rejected the first tiny-shape timing set: its 0.06–0.38 ms kernels were dominated by
  dispatch/reclamation noise and repeated runs contradicted one another.
- At realistic K=4096, N=3072 shapes, two independent interleaved runs showed Q4_0
  1.03–1.12× / 1.06–1.10×, Q4_1 1.02–1.09× / 1.03–1.10×, and Q4_K
  1.01–1.14× / 1.07–1.14× across M=1/32/128. Relative numerical differences stayed near
  1e-6 and came only from equivalent FP32 accumulation order.

### Evidence

- Final Chrome WebGPU matrix: 28 formats × 5 operator cases = 140/140, no failures; the
  decision-feature reduction also matched its CPU reference.
- Python suite: 47 passed. JavaScript suite: 19 passed.

## 2026-10-01 ▸ Native-width performance methodology correction

**Correction:** The earlier stored-versus-materialized timings were taken before the stored
formats had all received their applicable native integer-dot implementations. They are useful
diagnostics, but cannot decide production routing yet.

- Restored stored/original-format execution as the public default for GGML and GPTQ modules.
- Kept `auto` available only as an explicit diagnostic opt-in; it is not the default policy.
- Marked the existing benchmark output as provisional and without routing authority.
- Completion order is now explicit: implement and verify each original-format kernel first;
  then benchmark; only then consider an alternate representation for a measured shape/device.
- "Original format" also constrains the computation candidate: activation requantisation to
  satisfy a packed integer-dot instruction is a cross-width algorithm and therefore belongs
  to phase two. The phase-one Q8_0 change instead vectorises exact reads from each stored
  signed-byte block and preserves the existing FP32 activation path.

### Evidence

- Chrome WebGPU: 28 stored GGML formats × GEMV/GEMV2/GEMM/MoE-GEMV/MoE-GEMM = 140/140;
  the decision-feature GPU reduction also matched its independent CPU reference.
- Real cached xDecision Q8 inference retained the corrected distributions (billing 89.7%,
  duplicate true 97.9%, urgency 15.7/26.6/27.3/30.4) and completed in 304 ms for 486 tokens.
- Python suite: 45 passed. JavaScript suite: 19 passed.

## 2026-10-01 ▸ Decision-output correctness correction

**Trigger:** The real xDecision page showed uniform 50/50 and 25/25/25/25 distributions,
so its displayed conclusions were not credible even though all stored-weight format tests
passed.

### Root cause and changes

- The scorer itself produced non-zero logits. A validation-sensitive WGSL spelling in the
  subsequent decision-feature reduction made its dispatch invalid and discarded the whole
  pending WebGPU command buffer, including the scorer work; the zero logits then became a
  uniform softmax.
- Replaced that reduction with the WGSL form verified on the target browser and added a
  numerical browser gate for its four outputs. This tests dispatch and readback, not merely
  pipeline creation.
- Expanded the SDK version digest to include all Python modules, the module manifest, and
  both worker bootstraps. A Python-kernel fix now changes the browser URLs on the first
  reload instead of leaving a stale cached module behind.

### Evidence

- Browser correctness gate: 140/140 stored GGML operator comparisons plus decision-feature
  reduction passed; GPU output matched the CPU reference component-by-component.
- Real cached xDecision Q8 request completed in 595 ms: billing 89.7%, duplicate charge true
  97.9%, urgency distribution 15.7% / 26.6% / 27.3% / 30.4%.
- Python suite: 44 passed. JavaScript suite: 19 passed.

## 2026-10-01 ▸ Native stored-weight execution and truthful load progress

**Trigger:** Quantized weights must compute from their stored representation without
model-name or model-category special cases, then choose an alternative only from measured
operator performance. The browser also reported 527.9 MB loaded for a 402.5 MB GGUF.

### Changes

- Added a generic encoded-weight protocol and native stored execution for every GGML type
  accepted by the loader, plus GPTQ INT4/INT8 through the same execution policy.
- Added a shape-, format-, and device-measured `stored` versus `materialized` policy. Model
  names and decision-model categories are not inputs to the policy.
- Kept xDecision Q8 in GGUF form through its Linear operators and moved its final scoring
  features onto WebGPU so inference needs one final readback.
- Made browser model-source selection probe artifact existence first and run throughput
  samples only when multiple configured sources actually have it.
- Changed read progress from cumulative responses to the union of unique byte ranges; added
  a UI-side total clamp as a final invariant.
- Added browser matrix and benchmark pages, worker startup diagnostics, and module inventory
  loading for the standalone webapp harness.

### Evidence

- 28 GGML storage formats × 5 operator shapes: 140/140 WebGPU comparisons passed.
- Stored execution was typically 1.1–3.7× faster for one to 32 rows; at 128 rows several
  materialized GGML paths won by 5–30%, which is why the runtime measures rather than guesses.
- Python suite: 44 passed. JavaScript suite: 17 passed.
- Real cached `mccoysc/xDecision/models/gguf/xDecision-Q8_0.gguf` load displayed 402.5 MB,
  matching 402,546,752 bytes, and completed a 486-token decision request in 668 ms.

### Explicit limits

- The generic non-GGUF quantized implementation exercised here is GPTQ INT4/INT8. Quantized
  ONNX requires its quantized operator graph semantics and is not claimed by this change.
- Performance measurements describe the tested WebGPU device and shapes, not a universal
  threshold; the runtime profile is deliberately device-specific.

### Unchanged

- Dense FP16/BF16/F32 execution remains available.
- The model list still exposes only the published xDecision Q8 GGUF artifact.
# 2026-10-03 ▸ latest 30B live-browser cold/replay and release evidence

- The user-selected sole model was the native local `Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf`.
  A settled cached-prefix chat on the previous diagnostic build answered 1–40 correctly
  at 35.6 tok/s, but needed 15.3s to first token for one replay row; timestamp-query
  put the first GPU compute pass at only 31.2ms. The next same-model run was correct
  at 36.4 tok/s with 1.2s first token. This is latency variance, not a slow steady kernel.
- `Release` without page reload showed 0KB page GPU buffers and 42MB WASM heap. The page
  then loaded the newest stamped production bundle (no diagnostic URL settings), and
  reselected that exact local GGUF through the native file picker. The cold load showed
  `warm-step 9.3s` (first GPU sync 9.025s, second 0.050s). A correct 1–40 full-prefill
  chat ran at 34.1 tok/s, 3.6s first token; a second correct cached-prefix chat ran
  at 35.2 tok/s with 4.9s first token. This is one passing warm sample; previous
  13–14s warms and the new 15.3s replay keep both latency gates open.
- System VM sampling during reply work showed major transient wired-page growth and
  decompression/swap activity, consistent with resource-residency pressure but not a
  controlled attribution of the whole stall. The 677-dispatch captured step is smaller
  than both 1024 and 2048 flush thresholds; those settings do not split its main replay,
  so small unpaired tok/s differences between these thresholds cannot establish a
  causal winner for this decode path.
- Hardened `close()` so backend disposal still runs if `Worker.terminate()` throws; both
  backend singleton references clear before disposal, and main-thread teardown attempts
  WebGL even if WebGPU teardown throws. New exceptional-path test passes. Local checks:
  140 Python, 37 JavaScript, TypeScript compiler, and `git diff --check`. Not committed.
- A 256-byte-aligned shared WebGPU metadata-buffer arena was tested as a candidate to
  reduce the roughly 642 small metadata allocations held by this 30B. It passed local
  alias/lifetime tests and browser 1–40 correctness. Two serial native-file cold loads
  displayed `warm-step 10.0s` and `10.4s`; the first full reply ran at 35.2 tok/s,
  and its cached-prefix repeat at 35.1 tok/s, versus a nearby unpooled baseline at
  9.3s warm and 34.1/35.2 tok/s. This did **not** establish a positive latency or
  throughput result (the 10.4s load fails the hard limit). The arena experiment and
  its tests were removed; the bundle was rebuilt and restamped to the previous
  unpooled implementation. No candidate is enabled on allocation count alone.
  The restored production bundle was itself reloaded with the same local 30B:
  `warm-step 9.2s`, correct 1–40 at 35.5 tok/s and 3.9s first token. The model is
  left loaded in the browser; this still does not prove a stable ≤10s warm bound.

## 2026-10-03 ▸ 0.6B exact-output WebGPU profile and complete chat-path retest

- The only loaded model was the native local Qwen3-0.6B Q4_K_M GGUF
  (`ce11278f` fingerprint). Each script reload released it first, verified zero
  GPU buffers and about 42 MB WASM, then used the browser's disk file picker for
  the same file. No network model read and no simultaneous model load occurred.
- A device-local IndexedDB kernel profile reuses the measured full-sampling
  composition: fused add/RMS, fused QKV, separate gate/up, fused QK norm/RoPE,
  fused KV write, compact head, five Q4_K stored-format shapes, and GPU full-vocab
  selection. The profile is keyed by backend/device/source, not by model name.
  Four-step complete logits matched the stored-width reference exactly in a
  diagnostic readback, and seeded complete 1–40 replies matched the reference
  text. Direct SDK settled around 139–141 tok/s; this is not a chat-page claim.
- Fixed-seed chat-page generation with the same 115-token output was measured
  repeatedly. The original 60 ms Markdown refresh typically settled near
  134–136 tok/s; an adaptive 200 ms refresh on streams under 12 ms/token raised
  the seven settled runs to 136.41–137.54 tok/s, after a 127.85 opening run.
  All eight replies were correct; the product has **not** demonstrated stable
  140+ tok/s. The per-token callback itself averaged 0.051 ms; 20 live Markdown
  renders cost about 19.5 ms total. This performance finding is about the
  complete visible chat path, not a one-kernel microbenchmark.
- Same-file, same-seed interleaved A/B gave no stable benefit from combining
  compute and readback into one WebGPU submission, four-byte readback-buffer
  pooling, or worker notification batches of four/eight. Those candidate
  implementations were removed; the pool stays disabled. The adaptive render
  cadence is generic and has a unit test. Its final stamped build still needs
  WebGL product regression, so backend parity is not claimed complete.
- Final local checks: 76 JavaScript tests, 161 Python tests, TypeScript build,
  `git diff --check`. Worktree is intentionally uncommitted and dirty while the
  remaining seven-point gates are open. Next: close the 0.6B product-path speed
  gap without changing output semantics, then continue WebGL and Laya gates.
- The final stamped build was also loaded as the sole WebGL model through the
  native picker. Two fixed-seed complete chat replies matched 1–40 at 18.29 and
  18.68 tok/s; first-token times were 2.231 and 0.057 s. All 115 live updates
  in both runs took the original 60 ms render route, so the fast-stream UI
  change did not get imposed on WebGL's slower stream. Releasing WebGL returned
  WASM to about 42 MB and the page displayed 0 KB GPU buffers. The browser was
  then returned to WebGPU with the same local 0.6B file loaded; no other model
  was resident at the same time.
- A later ten-reply fixed-seed product run used `dispatch_flush=4096` on the
  same local 0.6B. All outputs remained correct, but settled throughput was
  135.4–137.0 tok/s, no better than the default build. The candidate URL was
  removed and the same local model restored on the original WebGPU URL.

## 2026-10-04 ▸ Decision-model layout detection and batched capture experiment

**Trigger:** The user clarified that question dependence must be determined by the
checkpoint's own contract, not by a Laya/model-name branch; independent questions must
batch, recording must work on that batch, and real-use tests must ask new questions.
The later requirement further restricts numerical work, control, and data movement to
JS/GPU (including pure CPU), with Python used only for orchestration.

### Changes and evidence

- Decision layout now prefers explicit `question_layout` metadata. For a legacy checkpoint
  without it, the complete config-plus-tensor schema identifies the per-question-row
  contract; otherwise loading fails rather than assuming independence. The local
  `convaiinnovations/laya-multilingual` configuration and its 170 safetensors names resolve
  to `per_question / checkpoint_structure`. The repository/model name is unused.
- A WebGPU encoder experiment records independent-row batches under a `(batch, length)`
  key, keeps per-row masks, bounds pinned mask memory, rewrites inputs on replay, and trims
  each row back to its actual length. Unit tests cover changed token IDs and masks.
- One local model was resident at a time. WebGL two-question browser output remained
  billing/yes, 764 ms hot for a 75-token request. WebGPU on the same short request was
  61 ms for a hot single question and 167 ms for a hot two-question uncaptured batch;
  this exposed the former single-only recording gate. On the new batch-capture build,
  three *different* two-question requests measured 184 ms, 156 ms (record), and 73 ms
  (replay); all returned billing/yes. No model was downloaded.
- Current automated checks after layout changes: 186 Python, 81 JavaScript and the
  TypeScript compiler passed. The capture changes have focused tests but still need
  complete-suite rerun and further browser correctness/performance coverage.

### Explicit limits

- Weights alone cannot prove cross-question training semantics. An explicit `joint`
  declaration currently fails with an adapter-needed error; no joint model is claimed
  supported by the per-question head.
- The 73 ms result remains above the 20 ms aspiration and does not prove fastest auto
  routing across shape/backends. WebGL's longer three-question batch took 2579 ms hot;
  batching must not be enabled merely because it is semantically valid.
- Python still performs numeric input staging, mask construction, operator calls and CPU
  NumPy execution. This fails the user's latest JS/GPU-only architecture requirement.
  The experiment is uncommitted; the JS CPU/WebGL/WebGPU execution migration is open.

### Unchanged

- LLM decode paths and model-list policy were not changed by the layout detector.
- The local decision model files were neither edited nor network-fetched.

## 2026-10-04 — decision batch routing and local-picker root cause

- Measured an already-loaded local Laya WebGPU request at 248 ms for 486 tokens:
  119.785 ms encoder and 121.08 ms head. The three sequences exceeded the old
  160-token batch eligibility gate, explaining the serial three-pass scaling.
- Removed that gate; a three-distinct-question 162–164-token regression proves
  one encoder and one head pass. Added pass/fallback diagnostics. Do not use
  alternate real user requests to calibrate scalar: auto defaults to batch
  until separately measured shape/backend evidence exists.
- Batched decision scores now cross GPU→host once rather than once per question;
  JS worker stages captured encoder embeddings/masks in shared memory, with
  release tests. This is not yet a complete JS/GPU-only decision pipeline.
- Diagnosed the local file chooser latch: the dropdown stayed selectable while
  its file input was disabled for a resident model; a suppressed `input.click()`
  left the awaited change/cancel Promise unresolved and `localPickActive` stuck.
  The dropdown and both native inputs now disable when a model is loaded and
  re-enable on release. The same dropdown and Load button are retained; no
  extra visible file chooser control was added. The chooser itself no longer
  holds a Promise/lock, so a suppressed picker can be retried.
- Verified the repaired single-dropdown UI in the in-app browser and ran
  190 Python / 86 JavaScript tests, TypeScript typecheck and JS syntax check.
  A local-disk CPU Laya three-question request reported one encoder/head pass,
  61 ms and 144 input tokens. The current WebGPU/WebGL real-model batch latency
  remains unverified because the browser automation did not open a native
  directory picker. No network model download was attempted and no model was
  co-loaded. No commit or push was made in this round.

## 2026-10-04 — live three-question fallback isolated

- User loaded the local Laya checkpoint in the in-app WebGPU page, without a
  network fetch. The 486-token, three-question hot response was 189 ms; the
  newly exposed diagnostic showed **encoder/head passes 3/3, route scalar**.
  The batch path failed before inference with `Javascript has no Float16
  support` when Pyodide tried to expose a NumPy float16 embedding table.
- Changed only the staging reference to a NumPy uint8 view sharing the exact
  same bytes; JS already decodes FP16 values through DataView into Number.
  This avoids a Float16Array bridge requirement without copying/requantizing
  the stored embeddings. A new regression proves the uint8 view shares memory.
- Full automated result after this fix: 191 Python, 87 JavaScript tests and
  `git diff --check` pass. The loaded browser instance is still on the prior
  worker, so no new 1/1-pass or latency claim is made yet. Model is kept
  resident until a deliberate rebuild/reload and native local re-selection.

## 2026-10-04 — byte-view bridge and cancelled-picker retry

- Standardized captured decision input references as zero-copy uint8 views for
  FP16 embedding bytes, int64 token IDs and int64 validity flags. JavaScript
  reads the original bytes with DataView and interprets them as Number; the
  typed views share their original NumPy memory. The JS test now passes only
  Uint8Array proxies, including a binary16 table and int64 ids.
- The directory-pick cancellation bug had a second independent cause: the
  select remained on its action option, so choosing the same option again did
  not dispatch `change`. The select now returns to its previous valid model
  before launching the native chooser, including the Load-button fallback.
  A separate in-app tab confirmed that choosing the directory action twice
  returned to the previous option both times, without another visible picker
  control. Native OS chooser automation itself is not available, so this is
  state-machine evidence, not a complete manual-dialog pass.
- Current source version stamps: SDK `9655bff4ef`, app `2eac7e3524`.
  The user's existing loaded-model tab remains on its older worker and has
  deliberately not been reloaded or network-refetched; real-model timing for
  this fix remains pending.

## 2026-10-04 — live one-pass evidence; batch final-head candidate

- User-loaded local WebGPU Laya run confirmed the byte-view bridge reaches
  `encoder/head 1/1, route batch` for three 486-token questions. Warm runs
  were 187 and 208 ms; a 160-token single question was 87 and 90 ms. Batch
  dispatch count fell but total latency still grows materially with the
  amount of question work. Cold/record runs ranged 240–389 ms and are kept
  separate from steady-state measurements.
- Found that scalar inference uses the correctness-gated `selected_q` final
  bidirectional head layer while batch always used `batched_full`. Added a
  batched selected-query final layer: every real K/V token remains available;
  only CLS and option marker Q/projection/MLP rows are computed. Distinct
  question lengths and option counts match the full head numerically on CPU.
  This is a candidate, not yet a WebGPU/WebGL speed conclusion. The user's
  loaded tab was not reloaded, and no second model was loaded.
- Profiled batch heads now expose prepare, layer queue, score queue, combined
  pending-GPU wait plus readback, and result times. The previous `head_ms`
  includes the first synchronization of outstanding encoder GPU work, so it
  cannot alone identify head arithmetic as the bottleneck.
- Replaced per-value FP16 exponent decode in the JS capture stage with a
  worker-local 65,536-value lookup over the same zero-copy byte view. A
  3x162x768 Node embedding-stage benchmark changed from ~2.7 to ~0.35 ms
  median; finite, subnormal, infinity and NaN regression values pass.
- The cancellation/reselect dropdown regression passes. Native OS-dialog
  reopening and end-to-end latency for this build still need browser evidence.
- Current source stamps are SDK `beb05a053a` and app `f8943b3fb2`.
  Automated checks: 192 Python / 89 JavaScript, TypeScript typecheck, JS
  syntax, version-stamp regression and clean diff. No commit or push yet.
# 2026-10-04 — multi-question scaling continuation

- User clarified the immediate gate: after batching, increasing question count must not
  leave roughly linear request growth. Absolute latency is secondary at this stage.
  The later gate is single-question performance; no 20–30 ms claim is established.
- Inspected the disk-backed Laya model in the existing in-app WebGPU tab. Browser native
  picker automation needed a CDP `Runtime.evaluate` user gesture on the *existing hidden
  input*, then the supported filechooser `setFiles` pointed at the existing local model
  directory. No network model read and no concurrent model load.
- Moved the decision-head padding-key mask construction out of Python and into worker JS.
  GPU paths stage one broadcast row per head through the shared upload arena; browser CPU
  writes the borrowed WASM byte view in place. Python supplies only short length metadata
  and buffer references. Unit regression checks mask semantics and proxy release.
- Real browser after this change: three distinct questions, 486 tokens, one encoder and
  one head pass, 174 ms warm. The previous ~120 ms `head prepare` appeared as ~140 ms
  `head layers queue`; mask staging by itself did not fix scaling.
- Added diagnostic per-layer/per-operator profile. In a warm 489-token three-question
  request, the first full head layer took 130.58 ms, of which norm1 measured 125.22 ms.
  Its generic WebGPU and WebGL fused LayerNorm used host-backed `_zeros` although each
  output lane is overwritten. Changed forward and fully-overwritten backward buffers to
  GPU-native `_empty` on both backends. Browser answers remained the same at displayed
  precision. Head-layer queue dropped to about 2–3 ms, but the GPU completion wait was
  ~125 ms and total three-question time ~155 ms.
- The WebGPU 3D batched matmul was a one-output-cell-per-thread scalar K loop. Added a
  16×16 workgroup-tiled kernel for B>=2, M/N>=32 and 32<=K<=256, retaining the scalar
  route for smaller shapes. It preserves strided input addressing. Local three-question
  browser answers remained unchanged at displayed precision, and changed-question warm
  timings became 64 ms / 163 tokens (one), 97 ms / 314 tokens (two), 130 ms / 489 tokens
  (three). These are positive absolute gains versus 71/112/155 ms on the prior kernel,
  but the roughly +33 ms/question slope still fails the stated non-linear scaling gate.
  WebGL performance and arbitrary-shape GPU numeric regression remain unverified.
- 192 Python unit tests, 92 JavaScript tests, TypeScript no-emit, source syntax/diff check
  pass. A `wgpy_test` batched strided/tail-shape test was added but browser harness did not
  yet complete it. The WebGPU wheel and SDK stamp were rebuilt. No commit or push.
- Extended the local single-model browser run to six distinct questions. Direct GPU
  row gather replaced the dense one-hot/matmul selected-row path on WebGPU and WebGL;
  the six-question selected-Q queue changed from 91/572 ms on first/record calls to
  0.45/0.52 ms. Six-question warm total was 219 ms at 941 tokens, still scaling
  roughly linearly with more distinct token rows.
- Tested an eight-row packed-FP16 matrix-kernel candidate to increase weight reuse.
  It made a three-question warm request slower (157 vs ~130 ms) and changed the
  billing answer to account, violating semantic correctness. The candidate was
  removed immediately; it is **not** an accepted optimisation. The browser still
  needs to reload the restored kernel before more correctness/performance claims.

## 2026-10-04 — corrected decision results and invalidated timings

- Found the actual cause of the uniform decision results: WebGPU `gather_rows`
  declared a WGSL binding named `meta`, a reserved identifier. The shader did
  not produce valid rows; a small six-by-four matrix returned stale data and
  the real model's marker logits were all zero. Renamed the binding and struct
  to `gather_meta`/`GatherMeta`, then verified the small row gather exactly.
  Earlier latency numbers using the broken shader are invalid and must not be
  cited as a performance improvement.
- Batched the scorer and action features for all decision rows in a single
  GPU path, with variable option counts masked. Real Laya browser logits on
  WebGPU are now nonzero and agree with scalar per-question inference within
  a few e-6. WebGL row gather and the same batch-vs-scalar comparison also
  pass. This proves internal numerical parity, not independent task accuracy.
- Corrected WebGPU changed-question warm requests measured about 43/72/102/179
  ms for 1/2/3/6 questions (163/319/495/954 tokens). The slope remains almost
  linear; one batch pass does not imply arithmetic parallelism or GPU saturation.
  WebGL one/two-question requests were roughly 1.5-1.9/3.75 s, so its speed
  needs further work even though logits matched.
- The existing model-structure inference for `question_layout` is not a
  universal semantic proof; explicit checkpoint metadata or a verified
  structural adapter is needed for any other architecture. It does not branch
  on a model name. No claim that arbitrary questions are independent.
- Automated source checks: 92 JS tests, TypeScript, and 194/195 host Python
  tests pass. The one failure is the older build-bound Qwen device-profile
  hash after common-kernel edits, not a numerical test failure. Do not change
  that profile hash without device remeasurement. No commit/push.
- Loaded the same local checkpoint on CPU only after releasing the WebGL
  instance. CPU's three example answers matched WebGPU/WebGL at displayed
  precision: `billing`, `true=0.9939`, `soon — today=0.5289`; the single batch
  needed about 50 s. This cross-check strengthens backend numerical evidence,
  but it does not supply independently labelled accuracy or satisfy CPU speed.
  Released CPU and restored one local WebGPU instance. The actual chat page
  displayed the same non-uniform answers in 115 ms for the first request after
  that load. No network weight fetch or simultaneous model load.
- With timestamp-query diagnostics enabled on the same WebGPU device, changed
  single-question hot input (165 tokens) recorded 441 encoder dispatches in
  one replayed pass at 42.27 ms GPU and about 72.4 ms end-to-end. A different
  three-question hot input (491 tokens) recorded 397 encoder dispatches in
  one replayed pass at 89.13 ms GPU and about 129.6 ms end-to-end; 17 passes
  summed to 97.12 ms GPU. The larger batch has ~3x token work but ~2.1x
  encoder GPU time, so there is some parallel benefit, yet the encoder remains
  the dominant critical path and request growth remains unacceptable. GPU
  timestamp mode adds query/readback overhead, so these are diagnostic ratios,
  not normal-page latency benchmarks. Do not call GPU compute or bandwidth
  saturation established from these counters alone.

## 2026-10-04 — locate added cost before optimising multi-question parallelism

- One disk-backed Laya model only; no weight download. Normal changed-input
  warm WebGPU runs at 1/2/3/6 questions (roughly 167/327/507/1018 input
  tokens) took ~43/72/103/193–205 ms. Each request used one encoder and one
  head pass. JS preparation and capture writes rose by only several ms;
  `head_ms` is not a pure head measurement because its first readback waits
  for encoder GPU work.
- Full-pass timestamp queries, with a known diagnostic overhead, measured the
  hot encoder replay at about 50/57/81/160 ms GPU for 1/2/3/6 questions.
  The head-related passes were much smaller. For B>=2 the encoder graph had
  397 dispatches at every count. The incremental critical path is therefore
  GPU work inside the batched encoder, not repeated encoder launches per
  question.
- Added an opt-in per-kernel timestamp mode and a small JS test. This mode
  force-flushes every dispatch and is too intrusive for quantitative
  cross-count comparisons: e.g. the six-question summed GPU time was lower
  than the three-question sum even though full-pass and normal runs show the
  opposite. It suggests packed-FP16 dense matmul (`mm_f16w`) and tiled
  attention (`batched_matmul_tiled16`) are the relevant families, but this is
  only candidate selection. No route has been enabled from these numbers.
- Next work: measure those families with a less intrusive design, then test
  shape/device-appropriate parallel and batch alternatives with output
  equivalence and changed-question end-to-end timings. Only after the
  multi-question slope is addressed may the single-question latency work
  resume. Browser profiling flags must be disabled for product runs.

## 2026-10-04 — lower-impact attribution and rejected shared-weight tile

- Replaced the force-flush-every-kernel diagnostic with opt-in selected-kernel
  timestamp passes inside one GPU command submission; the JavaScript test
  confirms two selected dispatches still use one submit. With the same sole
  local Laya model and changed state suffixes, B=2/3/6 encoder replays kept
  397 dispatches. `mm_f16w` (88 calls) measured about 30/45/86 ms GPU,
  `batched_matmul_tiled16` (44 calls) about 12/17/35 ms, and `ln_fwd`
  (45 calls) about 6/6/7 ms. The 2→6 increments of ~56 and ~23 ms explain
  most of the ~120 ms normal-page increase. Timestamp pass splitting still
  perturbs the run somewhat; use these for attribution, not a production
  throughput claim or proof of saturation.
- Implemented a temporary workgroup-shared packed-FP16 weight tile and
  validated its WGSL compiler diagnostics on the actual browser GPU. A
  separate browser-only synthetic test compared both shaders at M=33/192/
  384/576/1152, including G=1 and G=4. Outputs matched exactly, but the
  shared tile was 1.4–1.6× slower for representative M=192–1152 dense
  shapes. Removed the candidate, its switch, and the temporary test instead
  of enabling a negative route. No model-specific rule was introduced.
- The in-app browser's file chooser then stopped delivering any `change`
  event or selected files for the same local directory, including in a fresh
  isolated tab. No model was fetched from the network or co-loaded. This
  blocks real-checkpoint A/B for the next candidate until local selection
  works again; it does not invalidate the earlier attributions. No commit.

## 2026-10-04 — question-count scaling and fail-fast execution audit

- Selected-kernel GPU timestamps with one local Laya instance attribute the
  2→6-question increase to 88 packed-FP16 dense matmuls growing ~30→86 ms
  and 44 batched tiled-attention matmuls growing ~12→35 ms. LayerNorm's 45
  calls remain ~6–7 ms. The encoder is already one batch graph with 397
  dispatches for B≥2; each matrix dispatch covers all question rows. These
  are diagnostic, not proof of GPU compute or memory-bandwidth saturation.
- Audited the batch chain. `_bmm_raw` had a process-global exception latch:
  any 3D matmul error permanently changed every later attention BMM into a
  Python per-head loop. It now calls the backend's native 3D matmul directly
  and propagates failure. Decision batch encoding and scoring no longer fall
  back to host/scalar on exceptions or missing batch methods. Equal encoder
  tokens with distinct head metadata use separate batch rows, preserving
  answers without a scalar downgrade. The legacy per-question scorer loop in
  `_score_many` was removed. Unit tests cover failures and duplicate jobs.
- Made explicit WebGPU/WebGL boot fail rather than becoming CPU, checked the
  worker's actual Python backend against the requested one, and closed the
  runtime on mismatch. Direct `wgpy.initMain` also throws on an initialization
  error rather than trying a later backend. Decision warm-up, encoder capture
  allocation, missing SDK module manifest, stored-weight and shape-tuning
  candidate errors now propagate instead of choosing a slower default.
- The browser `/test/` worker was dying before pytest because it imported
  nonexistent `/lib/pyodide/pyodide.js`; then an unconditional install of a
  nonexistent Chainer wheel failed. It now uses the product's pinned Pyodide
  release, reports worker errors, and installs Chainer only for its own test.
  On the actual browser, focused native 3D BMM tests passed separately on
  WebGL and WebGPU, including transposed RHS views (1 test each). These test
  the low-level matrix primitive, not the full local decision model.
- Host tests: 200 passed with one preexisting stale Qwen profile-build hash
  failure; JavaScript tests: 94 passed. The profile cannot be stamped as
  current without remeasuring that Qwen route. The Mac's locked state is the
  concrete reason native file dialogs are not producing selection/change
  events; an attempted normal chooser timed out, and raw CDP file injection
  is unavailable in this browser. The new full-model slope, logits and WebGL
  decision performance have not been revalidated. No network weight load and
  no commit. Keep multi-question work open; do not start single-question work.

## 2026-10-04 — capture shape-drift and explicit GPU errors

- Found a distinct instability path: encoder capture retained four fixed shape
  slots and silently used the uncaptured path forever for all later shapes.
  Added per-name graph release and pin retirement on the WebGPU and WebGL
  interfaces, then LRU replacement of cold encoder captures. Unit tests cover
  slot replacement, recently used shapes, shared buffer pins and GPU release.
  This removes a known downgrade, but its magnitude on local Laya still needs
  a new-build browser A/B.
- WebGPU now checks asynchronous WGSL compilation diagnostics and scoped
  pipeline-validation errors before readback. A validation failure stays fatal
  for that context; an invalid shader cannot be interpreted as a valid zero
  output. WebGPU and WebGL readback failures carry the underlying error text
  to the worker through a small shared-memory error field, not a delayed RPC.
  Real-browser invalid-WGSL regression passed on WebGPU. The rebuilt WebGPU
  and WebGL wheels also each passed the native 3D batched-matmul browser test.
- The user has one local Laya model loaded in the existing WebGPU tab. That
  already-running tab uses the old code; the new source and bundles were built
  after it loaded. No second model was loaded and no model weights were fetched.
  The user's 56 ms best single-question result is not evidence of stable
  latency or a finished batch path. Host suite: 202 passed and one stale Qwen
  build-bound profile failure; JS suite: 100 passed. Full changed-input Laya
  correctness/slope/stability on the new build remains open. No commit.

## 2026-10-04 — new-build local-Laya slope and further fail-fast coverage

- Refreshed the existing in-app WebGPU page, releasing its old model, then the
  user loaded the sole local Laya checkpoint again; no other model was loaded
  or fetched over HTTP. The new build's first three-question UI run returned
  non-uniform answers in 132 ms at 486 tokens, with encoder/head passes 1/1.
  A second shorter input was 102 ms at 243 tokens and is **not** a comparable
  speed gain because the token count halved.
- Through the page's already-loaded SDK instance, sequentially ran distinct
  state variants at the same 164/321/498-token 1/2/3-question shapes. Once
  recorded, end-to-end results were about 47/75/101 ms with the batch route
  for 2/3 and one encoder/head pass each. Six further changed three-question
  states were 148/102/105/101/100/104 ms. The stable ~100 ms calls spent
  81-88 ms in the combined GPU/readback wait; one separate 299 ms outlier
  spent 274 ms there. A three-question result matched three separate calls
  on all displayed answer fields, but both compute and latency scalability
  remain open. No error-driven fallback was observed on those calls; this is
  narrower than proving none remains elsewhere.
- Additional candidate routes now propagate GQA, KV-pair, fused add-RMSNorm,
  and GGUF materialization failures instead of silently choosing a slower
  route. Selected GPU backend initialization and memory-release errors also
  surface. New host regressions for the candidate failures: 5 passed. Full
  host suite: 207 passed, one stale measured-profile hash failure that cannot
  be relabeled without remeasuring Qwen. JavaScript suite: 100 passed.
- Next: isolate and test positive dense/attention kernel or shape routes on
  the actual device, explain the GPU-wait outliers, and continue fail-fast
  audit. Only then revisit single-question latency. No commit.

## 2026-10-04 — operator slope and two measured latency candidates

- Repeated changed-input 2/3-question GPU timestamp profiles show the packed
  FP16 dense family increasing from about 30–33 ms to 44–45 ms (88 calls),
  with batched attention also increasing. The input grows by 183 tokens and
  warm request time by about 29 ms; LayerNorm stays nearly flat. Separate
  same-shader tests at 128/256/384/512/576/768 rows show approximately linear
  arithmetic time once 144 workgroups are available (0.20/0.33/0.46/0.59/
  0.66/0.92 ms for one representative dense operation). This is evidence that
  merely batching more rows is not an additional parallel-speed solution.
- A model-metadata-driven sliding-window QK kernel skips out-of-window tiles
  while fusing scale and additive mask. Its independent GPU timestamp is
  positive, and its full-model answers match the previous route. A similarly
  banded P@V primitive wins standalone but did not show an enclosing encoder
  gain, so the encoder continues using generic P@V. No model-name branch.
- Reducing the capture length bucket from 32 to 16 tokens cut the same
  two-question 327-token workload from roughly 71 to 66 ms. Enabling the
  fused QK for full-attention layers as well cut it to roughly 63–65 ms; the
  507-token three-question workload is still about 94–98 ms. The ten output
  sets compared against the earlier build were all identical at product
  precision; an independent GPU full/sliding QK numerical check gave maximum
  absolute difference 9.4e-9 with no incorrect cells. This is a narrower
  correctness claim than external-label model accuracy.
- A changed shape crossing a new 16-token capture bucket took 139 ms first,
  325 ms while recording on second use, then 132/112 ms. This cold-shape cost
  must be weighed against the steady-state bucket improvement; it is not
  closed. The large linear dense-math increment is also not closed. The sole
  local Laya checkpoint was released between builds and reloaded from the
  browser-native folder picker; no network model transfer or co-loading.
- Subsequent check of the encoder's existing 24-input mixed-length benchmark
  found the 16-token bucket's total latency (1345 ms) worse than 32 (1024 ms)
  because it fills and churns the four capture slots. Restored 32 as the
  general default; the 16-token local repeat win is not discarded as a
  candidate for a future workload-scoped selection. Isolated full-attention
  QK fusion at 32 tokens: 327-token two-question calls ~66–68 ms and
  507-token three-question calls ~92–93 ms, again ten matching answer sets.
- The captured WebGPU batch now stages one attention-mask plane per question,
  not twelve identical copies per question. The fused QK shader maps each
  head to its question's mask plane. At the same 32-token bucket, 2q calls
  were ~62–66 ms and 3q ~87–94 ms; all ten product answers matched the
  uncompressed-mask build. Independent GPU tests of compact masks across
  two question rows and three heads each, for both full/sliding attention,
  found no wrong cells and max absolute difference 9.4e-9. WebGL/CPU retain
  their full-mask equivalent path. Relevant host Python tests: 50 passed;
  JS capture/timestamp tests: 9 passed. This is a local latency reduction,
  not a proof of solved parallel scaling or completed project acceptance.
- An equal-input-token changed-state comparison measured three distinct
  single-question calls at 119.9–121.6 ms total and the same three questions
  in one batch at 88.4–90.3 ms (498 tokens in both cases), over five rounds.
  Every batch answer matched its separate answer at product precision;
  encoder/head passes were 1/1. This establishes actual ~1.34x batch speedup,
  not a claim that the GPU is fully utilized or that near-linear additional
  dense arithmetic is eliminated.
# 2026-10-04 WebGL memory and batch diagnostic

On the same local Laya WebGL checkpoint, repeated same-shape inference kept GPU
buffer accounting at ~503–504 MB. A reused-capture two-question call took
2342 ms at 376 MB GPU, while three questions took 3710 ms at 377 MB GPU.
Explicit Release moved GPU accounting to 0 KB
and WASM heap 994→50 MB; macOS reported 49% system memory free. The observed
slowness is not presently supported as a per-call memory leak. A two-question
visible-page request lasted 1822 ms and incurred a 1767-ms animation-frame
gap, matching a 1773-ms WebGL final readback wait. The readback currently runs
on the browser main thread, so this is proven UI blocking, separate from the
model's GPU work.
profile reported 234.8 ms encoder, 1786.1 ms head, with 1421.5 ms inside the
last GPU/readback wait (which includes pending encoder work). R16F dense weights
cut initial WebGL texture accounting ~479→269 MB and improved steady 2/3-question
runtime ~3.5/5.6 s to ~2.3/3.7 s, preserving five distinct-state product
answers to displayed precision. Batch was still slower than separate scalar
requests for some WebGL cases, so WebGL batch optimization is not complete.
An exploratory per-draw GPU timer-query instrument yielded a ~60 s aggregate
for a 2.38 s request and could not serve as credible operator attribution; it
was removed. Next: derive sound per-operator/shape costs or controlled ablations,
then address WebGL batch and CPU batch before commit/push and single-question
latency work. No model download and no commit.

Follow-up: WebGL2 in an OffscreenCanvas worker is available on this browser.
Moved context, commands and synchronous readback to a dedicated worker while
keeping payloads in SharedArrayBuffer and command signals small. Local Laya
loaded, inferred and released through the new path; after release the resource
panel reported GPU 0 KB/WASM 50 MB. On the final unflushed build, two/three
question answers matched the prior build and took 2353/3705 ms. Patching the
main-thread `WebGL2RenderingContext.readPixels` observed zero calls during an
inference, and the main-thread 10-ms interval had a max 11-ms gap, proving
event-loop blocking was removed. The compositor still had a 1.4–1.7 s
requestAnimationFrame gap, so GPU contention and model computation remain.
An experimental `gl.flush()` every 64 draws left ~1.65-s frame gaps and was
removed. The new worker/stamp code is not committed, and neither WebGL batch
nor remaining backend gates are complete.
As a separately isolated operator candidate, a scalar WebGL 334×768×2304
multiply took ~14–15 ms after warmup, while a two-output fragment shader took
~11–12 ms and produced the same 768 value in both output lanes on all-ones
inputs. At 507×768×2304 the comparison was ~18–20 versus ~15 ms; at
334×768×768 it was roughly tied. This has not passed full-matrix random-value
parity, its required unpack pass or end-to-end model timing, so it is not an
enabled optimization.

## 2026-10-04 requested remote performance checkpoint

WebGL now places eligible FP16 dense weights in row-aligned R16F textures, with
direct `(output column, reduction index)` addressing for a full contiguous RHS.
Wide or otherwise ineligible matrices keep the general layout/path. The
isolated browser WebGL matmul suite passed 4/4, including the new row-aligned
case. Host tests passed 211 with one skip; Node tests passed 100. On the sole
local Laya checkpoint, three changed-state questions took about 2.39 s versus
3.68 s before the row-aligned layout. The user observed 1.441–1.442 s on
repeated two-question, 327-token calls. Checked product answers matched the
prior build to four displayed decimals. The measured win is not a general
claim that WebGL batch parallelism, per-operator timing, CPU batching, or
single-question latency is solved. The user explicitly requested a commit and
remote push at this best-current-performance checkpoint before more tuning.

## 2026-10-04 ▸ xDecision GGUF recursion on published Pages

**Trigger:** Chrome's Pages tab loaded the Q8_0 GGUF, then showed
`RecursionError` inside WebGPU allocation while validating a materialized
candidate. The allocator's `__hash__` was only where the recursive stack ran
out: `ggml_dequant_ok` called `ggml_matmul` as its stored reference, and that
stored call unconditionally called `ggml_dequant_ok` again. ModelScope source
probe failures appeared in the console but were not this inference failure.

**Change:** Restrict alternate-candidate discovery to `auto` or the explicitly
requested candidate. A stored-format reference now runs without probing an
alternate route; this applies by execution mode and format capability, never
by model identity. A focused regression was red before the change and green
after it. Host tests: 212 passed, one skipped; JS tests: 100 passed.

**Browser evidence and limit:** Chrome's local updated build loaded the
existing disk-backed xDecision-Q8_0 GGUF, then answered three structured
questions (page-reported 152 ms). No model download was used. This proves the
recursion is removed on that route, not independent model accuracy or all
backend and performance gates. Published Pages must still receive the commit
and be checked after deployment.

Deployment check: commit `136647f` reached `origin/main`. The published Pages
HTML carries SDK version `cb903a1458`, and the served `_core.py` SHA-1
`a9c1b642ae9bd3013fb936297ca8ba6ff2ad5e9d` matches the local file.
No second model was loaded for a remote inference check while Chrome's local
test tab still held xDecision; deployment byte identity is confirmed, but
remote end-to-end inference remains a separate browser check.
