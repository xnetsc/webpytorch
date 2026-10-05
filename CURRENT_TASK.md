# Current task status

> Last updated: 2026-10-05

## Highest project principle

**Do not stop while either phase is incomplete.** A local commit, passing subset, status
report, individual-model result, or end of a conversation turn is not a completion point.
Phase one must finish correctness plus every applicable same-width optimisation and hardware
feature without converting the stored representation. Phase two must then finish the full
performance comparison and adopt whichever measured execution is fastest. Only both phases
passing their recorded acceptance gates completes this task.

## One-line status

**2026-10-05 WebGL decision latency 2230 → 1773 ms (three questions, 486 tokens):**
WebGL timer queries are not draw-granular on ANGLE-Metal (M5), so attribution is by
knockout. LayerNorm and softmax recomputed each row's statistics per element; both are now
two-pass above a measured rows×width² line and one-pass below it. Same answers; WebGPU
unchanged at ~97 ms. Dense projections are the next and largest item.

**2026-10-05 WebGPU F16-GGUF parity with the Laya reference:** On WebGPU the local
`xDecision-F16.gguf` and the Laya safetensors checkpoint hold identical resident weights
(169-tensor fingerprint) and dispatch the identical 422-kernel sequence; steady medians are
95.6 vs 96.1 ms for the three-question example. Load was 4.5 vs 1.6 s, entirely the GGUF
header (60 MB with an unused llama.cpp tokenizer, decoded element by element and re-walked
by a read ladder). The shared incremental, lazy header reader brings it to 1.7 vs 1.6 s with
unchanged answers and dispatches. The WebGL decision latency (~1.9 s) is still the open
performance item; Codex's `pair2` candidate is stashed, not committed.

**2026-10-05 WebGL operator attribution, not optimization completion:**
On the already-loaded local xDecision F16 GGUF, 417-token three-question
inference spends about 1.87–2.02 s in the 22-layer encoder before the two head
layers (~0.35 s) and tiny scorer. Within the encoder, diagnostic fenced stage
totals are attention ~1.15–1.22 s and MLP ~0.80–0.82 s. Across its 88 real
linear calls, QKV and MLP input projections each account for roughly 0.4–0.45
s under per-op barriers; output projections are smaller. The attention
QK/PV/softmax group also matters, but fine-grained barriers substantially
inflate whole-request time, so those estimates must not be added to the
unmodified ~2.3-s request. The user requested measurement before any further
candidate; no shader choice has been made. The loaded diagnostic wheel was
left in incumbent accumulator mode 1; on-disk experimental shader changes
were reverted. The broader performance and project gates remain open.

**2026-10-05 local F16 GGUF route corrected:** The actual local
`xDecision-F16.gguf` (not a safetensors checkpoint) has the same 170 tensor
names, shapes and dtypes as the Laya checkpoint, with semantically equal
encoder configuration and tokenizer. On Chrome WebGL, the prior generic GGML
byte-decoder route took 8011/7553 ms on the first two 417-token three-question
requests. Unquantized GGUF F16/F32 matrices now enter the same dense path as
other containers; Q8 and other quantized blocks remain stored-width GGML.
The four matched WebGL requests took 2386/2256/2260/2347 ms, with unchanged
first two checked answers. Serial WebGPU retest of that same local GGUF took
172/138/94/93 ms. Both used one batched encoder and one batched head pass.
This closes the F16-container-specific slowdown, **not** the shared WebGL
~2.3-second decision latency or the wider backend/performance gates.

**2026-10-05 checkpoint scope:** Decision reply-time device tuning now persists only
when new routes are measured, avoiding repeated calibration on reload. Completed
requests drop Python references and unpinned GPU scratch; explicit model release
was observed to return reported GPU buffers to zero. WebGPU capture-local
temporary reuse has a lifetime regression test, but its memory/latency benefit
has not yet been established in a controlled browser comparison. WebGL
capture-local reuse remains disabled after incorrect answers. The current
WebGL three-question F16 GGUF investigation and broader project acceptance
are **not** closed by this checkpoint.

**2026-10-05 WebGL capture and operator check:** The 2704-ms local Laya
three-question profile's final 1749-ms wait includes encoder GPU work. On a
comparable 423-token three-question diagnostic, a barrier after the encoder
measured 1799 ms pending there; 22 attention stages
totalled ~1106 ms and MLP stages ~748 ms under diagnostic fences. QKV and MLP
input projection segments were the largest (~670 and ~677 ms), though their
per-operation barriers add readback cost. WebGL capture-local buffer reuse
changed same-input answers and was removed. Without that alias, capture was
correct on six changed-input pairs but slower end to end: median 276.05 ms
versus 240.87 ms eager for a short one-question shape, despite reducing encoder
submission by ~38 ms. Therefore no WebGL capture speedup is enabled. Continue
isolated dense-projection measurement and three-question route optimization;
the multi-backend and project acceptance gates remain open.

**2026-10-05 xDecision/Laya comparison in progress:** Local files have the
same 170 tensor names/shapes, encoder config and tokenizer; sequential CPU
three-question runs agree on the first two checked answers and take 235.0 ms
(xDecision Q8 GGUF) versus 226.9 ms (Laya safetensors). Ignore the third
question's accuracy per user instruction. Browser timing now separates
decision weight-route tuning from the normal request path. On a fresh local
xDecision load, 825 ms of a 955-ms first request was weight tuning; saving
reply-time decision routes and reloading the same local file reused 37 entries
and reduced the next first request to 333 ms with 0 tuning calls. A 270-ms
second-use capture cost and hot GPU wait variance remain open. Five hot Laya
diagnostics spanned 129–213 ms (plus one 119-ms run); xDecision spanned
160–264 ms, so a stable Q8-specific hot regression is not established. The
final head readback fence includes encoder and head GPU work, not head alone.
The local xDecision GGUF is 402,546,752 bytes on disk; WebGPU buffers were
134 MB after load, 127 MB after the first three-question answer, then 1.09 GB
after the second answer recorded an encoder graph. Its sole `(3, 192)` capture
pinned 776 buffers / 1,033,578,476 bytes (including buffers it touched), while
the idle pool reported zero. Releasing the model returned GPU buffers to 0 KB.
This is capture residency, not a demonstrated unreleasable GGUF weight leak;
the memory/performance tradeoff and possible accumulation across up to four
capture shapes remain open. WASM heap capacity peaked at 2.38 GB, a separate
metric from GPU buffers, and fell to 50 MB after release.
No complete accuracy/performance acceptance is claimed by these observations.

**2026-10-04 xDecision GGUF browser regression fixed and deployed:**
Chrome on the published Pages build showed `RecursionError` while validating
Q8_0 materialization: `ggml_dequant_ok` called the stored `ggml_matmul`
reference, which unconditionally probed `ggml_dequant_ok` again. The stored
route now does not probe alternate executions. A regression failed before the
change and passed after it; host suite: 212 passed, one skipped; JS: 100 passed.
Chrome on the local updated build loaded the existing disk-backed xDecision
Q8_0 GGUF and answered three questions (page-reported 152 ms), with no model
download. Commit `136647f` was pushed to `origin/main`; the published Pages
script version and `_core.py` SHA-1 now match the checkout. A fresh remote
model-inference run remains unverified because the sole model is currently
loaded in the local Chrome tab. This fixes one load failure, not the remaining
project gates.

**2026-10-04 remote checkpoint requested; full project gates remain open:**
The latest local-Laya WebGL candidate uses row-aligned R16F dense-weight
textures and direct RHS texel addressing when the full weight matrix fits.
The new WebGL matmul browser regression passed 4/4, host Python tests passed
211 with one skip, and Node tests passed 100/100. On the same local decision
workload, WebGL three-question requests were about 2.39 s versus 3.68 s before
the row layout; the user's later steady two-question record was 1.441–1.442 s
at 327 tokens. Product answers matched the prior route to four displayed
decimals on the checked states. This is a measured performance checkpoint, not
proof of complete WebGL batch scaling, per-operator attribution, CPU batch
optimization, or the seven project acceptance gates. Commit and push this
checkpoint first, then resume those gates; do not treat the commit as completion.

**2026-10-04 WebGL memory/latency check; batch gate still open:** The local Laya
checkpoint on WebGL held about 503–504 MB GPU buffers over consecutive same-shape
requests without per-call growth. Releasing it reduced reported GPU buffers to
0 KB and WASM heap from 994 to 50 MB; system memory was 49% free. Therefore
the observed 2–4 second requests are not explained by a demonstrated leak or
system paging. On a reused capture, two questions took 2342 ms at 376 MB GPU,
while three took 3710 ms at 377 MB, further separating latency from live-byte
growth. A visible-page frame probe showed a 1767-ms frame gap during an
1822-ms two-question request, matching its 1773-ms WebGL readback wait; the
current main-thread `readPixels` therefore causes actual UI jank. A separate
two-question profile spent about 1.42 s at the final WebGL
GPU/readback fence, versus 0.28 s queueing head work; this fence includes the
encoder and head GPU work. The opt-in per-draw timer-query experiment returned
an impossible ~60 s sum for a 2.38 s request and was removed, so it is **not**
valid operator attribution. The WebGL R16F dense-weight candidate lowered the
initial weight texture ledger from ~479 to ~269 MB and 2/3-question steady time
from about 3.5/5.6 s to 2.3/3.7 s, with five distinct-state product-answer
comparisons matching to displayed precision. However batch remains slower than
separate scalar calls for some WebGL inputs; that and other backend gates remain
open. No commit or push.

**WebGL UI-stall follow-up, same date:** Moved the WebGL context/command replay
and synchronous readback to an OffscreenCanvas GPU worker, preserving shared
payload arenas and signal-only routing. The local model loaded and released
successfully (269 MB initial GPU, 0 KB after release); two/three-question
answers stayed equal with 2353/3705 ms, i.e. no throughput gain yet. Main-thread
`readPixels` calls fell to zero and a 10-ms JS timer stayed within about 11 ms,
but `requestAnimationFrame` still paused 1.4–1.7 s during GPU work. Thus the
remaining visible frame jank is GPU/compositor contention, not JS event-loop
blocking. Flushing every 64 draws did not improve it and was removed. The
worker refactor is uncommitted and needs full regression; batch performance
gate remains open.
An isolated WebGL matrix probe found that writing two adjacent outputs per
fragment (without reducing K precision) lowered steady 334×768×2304
multiply time from ~14–15 to ~11–12 ms, but the packed two-output texture
requires an unpack pass and end-to-end model validation before any selection.
No unmeasured route has been enabled.

**2026-10-04 measured multi-question operator increment; gate remains open:**
For distinct local-Laya questions, 2→3 inputs adds 183 tokens and about 25 ms
steady WebGPU request time on the latest candidate. Selected GPU timestamps
locate roughly 13–14 ms of the original increment in 88 packed-FP16 dense
matmuls and several ms in batched attention; normalization is nearly flat.
A standalone dense sweep increased workgroups 144→648 and work 0.45→2.04 GF
while GPU time rose 0.20→0.66 ms, so just adding question rows does not
increase throughput at these shapes. Fusing both sliding and full QK with the
additive mask/scale and staging just one mask plane per question (rather than
one per head) reduced the same 327/507-token workloads to roughly 64–66 / 87–94
ms, with ten matched answer sets. GPU full/sliding QK numeric parity had max
absolute error 9.4e-9 including shared masks. A 16-token capture bucket helped
repeated fixed shapes but historical mixed-shape tests and a 325-ms second-use
recording showed worse general total latency; **32 remains the default**.
At the same 498-token input, three separate single-question calls took
119.9–121.6 ms total versus 88.4–90.3 ms as one three-question batch;
all five answer sets matched. Thus batching gives a measured ~1.34x speedup,
but the remaining slope is real GPU work, not a hidden serial fallback.
The remaining near-linear dense workload, cold-shape costs, WebGL and other
project gates are **not solved**. The sole browser model is loaded from its
local directory on the latest WebGPU candidate. No download or commit.

**2026-10-04 new-build local-Laya concurrency check; gate remains open:** The sole
disk-backed `convaiinnovations_laya-multilingual` checkpoint was reloaded in the
existing WebGPU browser tab after building. A changed-state, equal-shape 1/2/3
question sequence warmed to about **47/75/101 ms** at 164/321/498 input tokens;
all multi-question calls used one encoder and one head pass and the batch route.
Further six distinct three-question states settled at 100-105 ms after one
148 ms outlier; capture replay stayed active and the GPU-readback wait accounted
for about 81-88 ms of the stable calls. One outlier reached 299 ms with 274 ms
in GPU/readback wait despite capture replay, so variability is **not** proved
fixed. Batched answers matched sequential single-question answers exactly at
the product's four-decimal result precision for one changed state; this does
not prove independent accuracy. Added fail-fast behavior for selected-backend
initialization, GQA/KV-pair/add-RMSNorm/GGUF candidate failures, and GPU memory
release errors; five new host regressions pass. Host suite: 207 passed, one
stale build-bound Qwen profile failure; JS: 100 passed. The warm 1→3 slope
still rises by ~54 ms, predominantly GPU work; no positive new concurrent
kernel candidate has been established yet. **Do not start single-question
optimization or claim fallback/concurrency complete. No commit.**

**2026-10-04 capture-eviction and error-propagation update; multi-question gate still open:**
The user has loaded one local Laya checkpoint in the existing WebGPU tab; that tab
still runs its already-loaded build and must not be mistaken for a validation of
subsequent source edits. The four-shape encoder capture cache had a confirmed
permanent performance downgrade: after filling, new repeated shapes ran
uncaptured forever. It now retires the least-recently-used graph by name on
WebGPU and releases its Python and JS pins, while preserving other graphs;
WebGL has the equivalent per-name capture API. WGSL compilation and WebGPU
pipeline validation errors now reach the next readback instead of returning
zero/stale results, with the original message carried to the waiting worker;
WebGL readback errors use the same shared error channel. Real-browser WebGPU
invalid-shader regression passed (1 test) and new-build 3D batched matmul
passed separately on WebGPU and WebGL (1 each). Host Python: 202 passed,
one stale build-bound Qwen profile hash failure; JS: 100 passed. The new-build
full Laya changed-input slope and shape-drift stability are **not yet tested**;
the batch GPU dense/attention work still grows with token rows. Do not call
the concurrency or fallback audit complete. Single-question optimisation is
deferred; no commit.

**2026-10-04 fail-fast batch/backend audit; browser model gate open:** The
measured 2→6-question added GPU time is concentrated in packed-FP16 dense
matmul (~56 ms) and tiled attention (~23 ms), not repeated model passes. The
88 dense and 44 attention dispatches already span all batch rows; a shared-
weight tile and native f16-load candidate failed positive device-speed tests
and are not enabled. A hidden global 3D-matmul failure latch that permanently
switched to a Python per-head loop is removed. Batched encoder/head errors now
propagate rather than falling back to host/scalar; equal token ids with distinct
head metadata receive separate batch rows. GPU boot, worker installation,
actual-backend mismatch, warm-up, capture allocation and execution-candidate
errors are no longer silently converted to slower routes. WebGPU and WebGL
browser tests each pass a real 3D batched matmul with strided-RHS coverage;
the browser harness itself was repaired after finding missing Pyodide and
Chainer paths. Host Python has 200 passes and one stale build-bound Qwen profile
hash failure; JavaScript has 94 passes. Full local-Laya changed-input A/B on
this new build is **not verified**: the Mac is locked and the browser's native
directory chooser does not deliver a file selection. No model was fetched over
HTTP or co-loaded. Multi-question and WebGL performance gates stay open;
single-question optimisation remains deferred. No commit.

**2026-10-04 precise encoder increment, candidate rejected:** A lower-impact
selected-kernel timestamp mode keeps each encoder graph in one GPU submission.
Across changed-input 2/3/6-question requests, its 88 packed-FP16 matmuls
cost ~30/45/86 ms GPU, 44 tiled-attention matmuls ~12/17/35 ms, and 45
LayerNorms ~6/6/7 ms. Thus 2→6 adds about 56 ms dense matmul and 23 ms
attention, explaining most of the observed ~120 ms request increase. The
batch graph already dispatches these operators across all question rows;
there is no per-question encoder loop to remove. A workgroup-shared-weight
candidate produced exact synthetic outputs but was 1.4–1.6× slower on
representative dense shapes and was removed. No positive parallel-path
change has yet passed real-model A/B or WebGL verification. The single-question
phase remains deferred. Browser model reload for further A/B is presently
blocked by the in-app file chooser not delivering a `change` event; do not
replace this with a network model load. The work remains uncommitted.

**2026-10-04 latest diagnostic (multi-question gate still open):** With one
local Laya checkpoint loaded, changed-input warm WebGPU requests measured
~43/72/103/193–205 ms for 1/2/3/6 questions (about 167/327/507/1018
tokens). All used one encoder and one head pass. In opt-in full-pass GPU
timestamps, the encoder pass rose from about 50 ms at one question to
~57/~81/~160 ms at 2/3/6; the action head and JS preparation are secondary.
The batch already executes as one 397-dispatch encoder graph for B>=2, so
the rising cost is primarily *work within those dispatches*, not repeated
Python calls or one model pass per question. An opt-in per-kernel profiler
points qualitatively to packed-FP16 dense matmul and tiled attention, but its
per-dispatch flush perturbs scheduling so strongly that cross-count kernel
times cannot be used to pick a route. Next: isolate those kernel families
with less-perturbing measurements, then test batching/parallel changes on
correct logits and changed questions. Single-question optimisation remains
strictly subsequent to resolving this scaling gate. No commit.

**2026-10-04 correctness correction (supersedes the decision timing claims below):**
The newly introduced WebGPU `gather_rows` shader used WGSL's reserved identifier
`meta`, so compilation failed silently and the destination contained stale or
zero data. The page's uniform 33.3%/50%/25% decision scores and all performance
claims made from that broken path were invalid, including the earlier 64/97/130
and 219 ms runs. The shader identifier is now `gather_meta`. Real local-model
browser tests on WebGPU and WebGL show nonzero, backend-matching logits, and
the batched output matches each backend's scalar output to a few e-6. This is
batch-vs-scalar equivalence, **not** an independent accuracy verdict. Corrected
WebGPU changed-question warm timings are roughly 43/72/102/179 ms at 1/2/3/6
questions and 163/319/495/954 tokens; the nearly linear growth remains open.
WebGL correctness matches but one/two questions took roughly 1.5-1.9/3.75 s;
performance parity is not established. The current full host suite has 194
passes and one stale build-bound Qwen profile hash failure after shader-source
changes. No performance profile should be restamped without Qwen revalidation.
The task is uncommitted and the seven-point gate remains open.
After releasing WebGL and loading the same local checkpoint alone on CPU,
the three example answers matched WebGPU/WebGL exactly at displayed precision:
`billing`, `true=0.9939`, `soon — today=0.5289`. CPU batch execution took
about 50 s, so this is a correctness cross-check, not a CPU performance pass.
CPU was released before restoring the sole local model on the WebGPU page;
the current product page again shows those three non-uniform answers (115 ms
for the first post-load request). Independent labels/reference accuracy and
non-linear multi-question scaling remain unverified.
Timestamp-query diagnostics on this WebGPU device put the hot encoder GPU pass
at ~42 ms for one 165-token question versus ~89 ms for three 491-token
questions; it dominates the ~72/~130 ms profiled request totals. Thus the
remaining growth is mostly real encoder GPU work, not merely Python/JS wait
or the action head. Query instrumentation changes wall time; this is not a
normal product-speed benchmark or proof of compute/bandwidth saturation.

**2026-10-04 active multi-question scaling investigation:** The current local
WebGPU browser has only the disk-backed `convaiinnovations_laya-multilingual`
loaded; no model network fetch or concurrent model load. One-/two-/three-question
warm, *changed-question* page runs on the tiled-bmm candidate read about
163/314/489 tokens in 64/97/130 ms respectively, with one encoder and one head
pass in every request. This is still approximately +33 ms per added question,
so the user's non-linear scaling requirement is **not yet met**. Earlier JS
key-mask staging removed a Python-built/uploaded broadcast mask, but merely
shifted the visible wait. Finer browser profiling found the first full head
LayerNorm spent ~125 ms in a host-backed zero allocation for an output fully
overwritten by the kernel. Replacing the zero allocation with GPU-native empty
allocation in WebGPU *and* WebGL cut that head-layer queue time to 2–3 ms and
preserved the displayed three answers, but exposed actual GPU completion time.
The tiled batched-attention candidate further changed the warm 1/2/3 results
from about 71/112/155 ms to 64/97/130 ms. This is one real browser device,
not proof of general benefit or WebGL parity. The first two requests include
compile/graph-record cost. A batched non-multiple-shape browser regression was
added but has not run in a working browser test harness. Current source checks:
192 Python and 92 JavaScript tests, TypeScript and diff check pass; work is
uncommitted. Continue exact-semantic GPU scaling optimisation before the
single-question latency target. Do not treat the seven-point project gate as done.
The later eight-row packed-FP16 candidate was rejected and removed after a
slower 157 ms three-question run changed the billing answer to account; the
browser must reload the restored kernel. Direct row gather has removed a
six-question selected-row one-hot-matmul hotspot (recorded 572 -> 0.52 ms),
but six-question warm total remains 219 ms, so scaling remains open. A full
Python run now also flags the older build-bound Qwen profile's hash mismatch;
do not relabel that profile as measured for the new common-kernel build.

**2026-10-04 live decision-batch evidence and next candidate:** The user reloaded
the local Laya model on WebGPU and confirmed three 486-token questions use one
encoder and one head pass. Stable repeated-request timings were 187/208 ms;
one 160-token question was 87/90 ms. This is still a substantial rise with
question count; the 1/1 pass count alone does not settle performance. Earlier
284/328 and 389/240 ms runs include compile/record cost and must not be
compared with steady state. The existing `head_ms` ends at the first GPU
readback, so it includes pending encoder work; a profile breakdown now names
queueing and the combined GPU wait separately. The batch path was doing a
full final bidirectional head layer while scalar used selected-query rows.
The new batched selected-query path preserves all real K/V rows, masks each
question's padding, and computes only CLS/option Q/projection/MLP rows. A CPU
full-layer numerical comparison with different lengths and option counts
passes; **the candidate still needs a real WebGPU/WebGL browser correctness
and latency comparison before claiming a win**. The FP16 byte-view stage now
uses a worker-local lookup for identical decode values; 3x162x768 Node
microbenchmark improved from about 2.7 to 0.35 ms, but this is not an
end-to-end product gain. Cancelled-directory retry has an automated state
regression; native chooser re-opening remains unverified by automation.
No model was downloaded or co-loaded; the user's current loaded tab is not
reloaded for this source update. Full phase and seven-point acceptance remain
open. The newly stamped SDK is `beb05a053a` and page app is `f8943b3fb2`;
automated checks pass (192 Python, 89 JavaScript, TypeScript, JS syntax,
versioning, and clean diff). These are source checks, not a browser-performance
pass for the new batch candidate.

**2026-10-04 current decision/picker correction:** The observed near-linear
three-question WebGPU request was 248 ms for 486 input tokens, split roughly
120 ms encoder + 121 ms head. The previous 160-token-per-question batch gate
forced three passes for this shape. That cutoff is removed; long distinct
questions have an automated one-encoder/one-head-pass regression. Product
requests no longer alternate between batch and scalar just to gather timings;
the default is batch until separate per-shape evidence establishes a winner.
Profiled responses now report encoder/head pass counts and any fallback.
The native local model picker also had a root-cause dead state: the file input
was disabled while a model was loaded, yet the active dropdown could click it;
the chooser Promise then never settled and prevented later attempts. The
dropdown and both native inputs are now disabled while a model is resident and
re-enabled on release, with no extra file button or pending chooser Promise.
The browser verifies the corrected single-dropdown UI; automated suites pass
(191 Python, 87 JavaScript). A local-disk CPU Laya run of three distinct
questions used one encoder and one head pass (61 ms, 144 tokens). The **new
WebGPU/WebGL real-model timing is not yet browser-verified**: browser picker
automation did not open the native chooser, and no model was fetched over the
network. A subsequent *loaded* browser profile identified the remaining
three-pass cause exactly: `Javascript has no Float16 support` during JS staging
of NumPy FP16 embeddings, followed by a silent scalar fallback (`3/3` passes,
189 ms hot for 486 tokens). The stage now passes a zero-copy uint8 view of
those original bytes and JS decodes FP16 with DataView; a regression checks
shared backing memory. The current browser worker has not reloaded this fix,
so `1/1` WebGPU evidence is still required. Do not claim final decision
performance, JS-only CPU/WebGL parity, or project completion on this evidence.
The byte-view bridge now also covers int64 IDs and validity flags without copying.
Canceling a directory pick had a second root cause: the dropdown stayed on its
action option, so selecting it again fired no `change`. It now restores the
previous valid option before opening the picker. A separate in-app page verified
two consecutive local-dir selections restore the option; native OS-dialog
opening remains unverified by automation. Source stamps are SDK `9655bff4ef`
and app `2eac7e3524`; the user's model-loaded page remains on the old worker.

**2026-10-04 active decision-runtime migration:** Decision question layout is now resolved
from explicit checkpoint metadata or a complete legacy config/tensor schema, never a
repository name; unknown semantics fail closed. A single locally selected model at a
time verified WebGL and WebGPU browser answers. WebGPU can batch independent question
rows and capture the batch by shape; three *different* two-question requests measured
184 ms, 156 ms (record), then 73 ms (replay), with billing/yes answers. This is not a
20 ms result and is not a compliant JS-only implementation: Python still prepares
tokens/embedding rows/masks, drives kernels, and CPU runs NumPy. The user's latest
constraint requires numerical work, control and movement in JS/GPU on all three
backends, leaving Python only orchestration. Do not call the decision optimisation
complete, do not commit the current experiment, and do not infer a real-use speedup
from repeating an identical question.

**2026-10-04 WebGL continuation:** The user provisionally accepted the current
WebGPU result and asked to move to WebGL. With one local model at a time,
WebGL Qwen3-0.6B produced the correct 1–40 answer at 18.2 tok/s with 2.4 s
first token; an explicit whole-decode calibration kept the already selected
`auto` route, and a second correct page answer measured 18.4 tok/s. After
release to GPU 0 KB / WASM 42 MB, the local 30B Q3_K_XL loaded alone;
`warming` was 4.0 s but `proving` was 16.1 s. Its first visible token took
roughly 20 s, and the severely paging run had reached just 13 tokens at about
0.3 tok/s when stopped to protect memory. A second release again verified GPU
0 KB / WASM 42 MB. The 30B WebGL latency and throughput gates remain failed;
no complete-answer accuracy claim is made for that stopped run. No model
network download, simultaneous model load, source optimisation or commit.

**2026-10-04 latest:** The historically recorded WebGPU Qwen3-0.6B Q4_K_M
decode route is preserved in `profiles/webgpu_apple_metal3_2026-10-04.json`
as a build- and device-bound profile, with a regression proving that the SDK
imports the exact route and GPU sampler without a model-name runtime branch.
The in-app page still has the sole native-file GGUF loaded and shows a correct
110-token answer at 136.9 tok/s. Six further actual-page replies with the same
automatically reused plan were all exact at 128.41/136.55/139.08/136.98/
134.71/136.02 tok/s; this still fails the no-frequent-below-135 condition.
Eight complete direct SDK replies previously
measured 137.09–143.39 tok/s (141.83 median), but 24 controlled actual-page
replies measured 126.88–138.90 tok/s. Thus the *SDK historical regime* is
reproduced and the route is recorded; the page's ordinary 140+ acceptance is
**not** proven. Current scoped checks: 173 Python, 79 JS, clean diff check.
WebGL, decision-model optimisation, the full seven-point matrix and final
commit remain open. Only one model is resident; no model download occurred.

**2026-10-04 active continuation:** The user clarified that the historical
0.6B target means ordinarily 140+ tok/s with only occasional drops to roughly
135+, not a 135 median. The sole local Qwen3-0.6B Q4_K_M was reloaded in the
in-app WebGPU browser; no model was downloaded or co-loaded. Current normal
page runs are below that standard even after runtime restoration of the
recorded full-fusion path. The old saved device profile was overwritten by a
newer v3 profile, so a route label alone cannot establish restoration. A
generic upper-level projection-shape composition and the historical fused
bundle are now in the correctness-gated offline tuner, compatible with old
profiles; 170 Python and 79 JS tests pass. SDK stamp `9ef7514292` was loaded.
The offline tune completed but selected an older slower plan; the recorded
full-fusion plan was saved to the same-build device-local profile and verified
to auto-reload from the same disk GGUF (`profile_reused=True`). Twelve exact
chat replies were only 115.01–133.53 tok/s, so historical performance is
**not restored**. A fresh origin's 60 tok/s result is invalid as a comparison
because reasoning and Python tools were enabled there, forcing JS sampling;
both settings must be aligned before repeating. The browser turn interruption
closed all tabs and released the model. WebGL, Laya, the seven-point matrix
and final commit are not complete.

**2026-10-04 controlled continuation:** An isolated local page was given the
same-build/device profile, native disk GGUF, reasoning off and tools off. It
auto-reused the recorded full-fusion/GPU-sampling plan. Twenty-four exact
231-context/110-token page replies ranged 126.88–138.90 tok/s (about 135
median), still below the product target. Eight fixed-seed complete SDK replies
were exact at 137.09–143.39 tok/s (141.83 median), reproducing the historical
SDK regime but **not** proving a 140+ chat-page result. The earlier 60 tok/s
fresh-origin run was invalid due JS sampling with tools on. A new regression
checks saved-plan roundtrip and full route application; 172 Python / 79 JS
tests and `git diff --check` pass. The one model remains in the isolated
in-app browser page; no other model is resident. Do not move on from 0.6B
product acceptance or claim WebGL/Laya/final seven-point completion.

**2026-10-04 latest in-app continuation:** Reconfirmed the sole locally selected
Qwen3-0.6B Q4_K_M model was ready on WebGPU. Five real chat-page 110-token
replies were semantically correct but only 104.92–128.55 tok/s; the 140+
gate remains open. Fixed a stale decode-plan diagnostic: after an unprofiled
greedy call, returning to the measured full/GPU-sampler route now clears the
old `mode_profile_missing` flag. The browser verified `full/gpu` with no false
missing marker. An experimental one-submission vocabulary sampler was tested
on the *same loaded model* and then rejected: all eight fixed-seed, full-SDK
paired runs were slower than the two-submission incumbent, while all full
texts matched; eight actual-page interleaved pairs also had no repeatable
positive result. The candidate code was removed and the incumbent retained.
The final stamped browser build (`380ffcae77`) was reloaded from the same
native disk file; four further correct replies measured 117.23/107.85/128.69/
128.27 tok/s, with `full/gpu` shown as the active measured route. The model
is left loaded, alone, on the normal in-app WebGPU page. A same-instance
231-position layer trace measured a 7.94 ms complete step: aggregate MLP/QKV/
attention-output/head stages were 2.079/1.1475/1.1355/1.0841 ms. After
the profiler reset its temporary captures, five more correct replies were
84.59–128.87 tok/s; GPU buffers returned from a transient 606 MB to 555 MB,
and capture pins stayed 984/~5 MB. macOS reports 64% free memory and no
thermal/performance warning, neither of which establishes GPU clock behavior.
Maintained suites:
168 Python and 79 JavaScript passed; `git diff --check` passed.
No other model or network download was used. This is neither a throughput
closure nor completion of the seven-point project matrix.

**2026-10-04 current loaded-model check (`bf9fed4c0b`):** The sole in-app
browser model is the local `Qwen3-0.6B-Q4_K_M.gguf`, selected through the
native file picker; no other model was loaded or fetched. An explicit offline
WebGPU v2 complete-decode tuning pass took 222.86 s and selected the
original-width Q4_K stored route with fused compact QKV, fused QK norm/RoPE,
separate gate/up, and diagnostic median 8.525 ms. Six subsequent **real chat
page** replies at 231 context / 110 generated tokens all gave the requested
1–40 sequence, at 112.35/113.81/123.76/122.28/113.95/86.93 tok/s: still
**not stable 140+**. A same-instance layer profile measured 6.565 ms for a
whole step at position 0 (diagnostic stages: MLP 1.991, head 1.059, QKV
1.005, attention output 0.653 ms net); this excludes actual sampling and is
not a product throughput claim. The sampled route's worker read/compute
diagnosis measured 111 selections: JS fetched 67,459,584 bytes, with 883.13
ms total read/wait plus 72.21 ms sampling; GPU fetched 444 bytes, with 878.25
ms read/wait, but per-call performance was variable. Five alternating real
page JS/GPU pairs all returned the same correct sequence and GPU won 5/5
(132.06–134.04 versus JS 72.12–123.05 tok/s); eight fixed-seed SDK pairs
preserved full text but GPU won only 5/8. The measured device profile was set
to GPU sampling for this vocabulary and confirmed by 4-byte-per-selection
readback, yet six later real-page replies were 118.70/111.41/133.24/133.67/
133.20/118.43 tok/s. A proposed combined fused-add/fused-KV/compact-head
upper plan preserved all fixed-seed answers but won only 6/10 SDK pairs, so
its runtime overrides were restored and no source change was made. The
140+ product gate and the seven-point matrix remain open; do not infer
completion from the diagnostic 8.525/6.565 ms figures.

**Subsequent current-source correction:** The v2 whole-decoder profile did
not include whether full-vocabulary sampling ran through JS or GPU, so its
upper-layer choice could be reused across materially different lower paths.
WebGPU's key is now `decode_plan_v3` and includes the *effective* sampler
route, including JS fallback for unsupported sampling options; WebGL's v1
topology is unchanged. The browser is on SDK stamp `72e61e0a7b`, with the
same native-file 0.6B loaded alone. Its prior GPU sampling profile was reused;
because no matching v3 upper profile existed, load exposed a truthful
original-width `budget_limited` plan. One baseline page answer was correct at
121.24 tok/s. The explicit GPU-sampler v3 whole-decoder search finished in
216.57 s, selected the original-width/separate-QKV/composed-QK route at a
diagnostic median 8.46 ms, and persisted the v3 profile. Eight actual page
answers all gave 1–40 at 118.09/119.98/119.76/125.43/111.10/86.79/128.52/
128.04 tok/s: still not stable 140+. The prior JS-tuned fused combination was
correct under GPU sampling but won only 5/8 interleaved page pairs, so the v3
plan was restored. A 12-run per-token worker profile linked 73–87 tok/s slow
replies to 10.9–12.9 ms GPU-completion/readback waits, versus 7.38–7.48 ms
for 127–130 tok/s replies. In a temporary `profile_gpu=1` run, hardware
timestamps over eight 110-token replies measured main-pass GPU medians of
5.90–6.42 ms on faster diagnostic samples versus 10.49–13.17 ms on slower
ones, with exactly 562 main dispatches and one sampler dispatch per token.
Instrumentation changes absolute rate, but the same-graph hardware-duration
split is real; its external cause is not identified. The browser was restored
to ordinary WebGPU and the same disk GGUF; its v3 GPU profile was reused, and
four correct page replies measured 119.71/120.37/128.02/128.78 tok/s.

**Further memory fix (`303f1177d3`):** The load-time greedy-chunk calibrator
had retained an unused greedy graph in a normally sampled model. Replacing
only that graph in the loaded browser reduced pinned buffers from 2,956 /
19.55 MB to 984 / 4.65 MB without changing a correct sampled answer. The SDK
now releases this graph after calibration when the active path is sampling,
and again when a caller switches back from greedy to sampling; the selected
chunk size remains available for lazy recapture. On the freshly loaded new
build, pinned buffers were zero before the first reply and 984 / 4.65 MB
after four correct replies, with no growth. A forced greedy chunk-2 test
matched scalar greedy's full 110-token answer, captured the chunk graph, and
returning to sampled chat reduced pins from 2,958 / 13.96 MB to 984 /
4.65 MB automatically while preserving the correct 1–40 answer. The forced
chunk route is restored to the measured auto choice (zero on that load).
Current maintained suites: 167 Python and 79 JavaScript passed. This is a
memory-lifetime fix, **not** evidence that 140+ throughput was reached.

The old JS-tuned fused composition was rechecked on the loaded GPU-sampling
model. A 16-pair alternating page test gave only 7/16 fused wins, all answers
correct. Inspection showed a route switch transient (about 3.45 MB of graph
buffers awaiting reaping), so that first-run A/B does not isolate steady-state
speed. Six paired two-run blocks, comparing only each route's second run,
split 3/6. Four uninterrupted eight-run A/B/A/B blocks also drifted heavily
within a block (settled medians about 125.8/130.4/110.9/90.9 tok/s). Thus
neither candidate has repeatable product-level superiority under this GPU
variability. The official v3 route was restored, and the local 0.6B remains
loaded in the ordinary in-app WebGPU page. `powermetrics` needs superuser here;
no GPU frequency/thermal cause has been established.

**2026-10-04 sampled-path upper-API validation:** Rebuilt/stamped SDK
`793af5ae3d`, released the previous 0.6B instance (browser reported GPU 0 KB,
WASM 42 MB), and selected the same complete GGUF directly from disk; no second
model or network model read. The WebGPU complete-decode profile is now versioned
`decode_plan_v2` because v1 timed full-logit readback/argmax while normal chat
actually sampled through JS/GPU. The offline v2 search used fixed-seed real
sampling for candidate timing and four-step token/logit semantic checks, took
217.4 s on the sole loaded model, and kept the original-width composition
(diagnostic median 9.6925 ms). A behavior regression rejects candidates with
equal logits but different sampled tokens, and verifies sampling state restore.
The first chat reply was correct at 127.10 tok/s before tuning; seven complete
post-tune page replies were also correct at 120.22, 123.66, 110.46, 127.51,
128.30, 124.08 and 94.58 tok/s (231 context, 110 generated tokens). This is
**not** a stable 140+ result, so the 0.6B product-speed gate stays open.
The post-search page reported 570 MB GPU buffers and 146 MB JS heap. Release
returned GPU buffers to 0 KB and WASM to 42 MB; a page reload lowered JS heap
from 172 MB immediately after release to 22 MB. Reselecting the same local file
reused the v2 profile (`profile_reused=True`), and six correct product replies
were 127.57/130.07/122.84/130.46/121.67/125.36 tok/s. The sole local 0.6B
is left loaded for further testing. Scoped suites:
163 Python, 79 JavaScript passed; the repository-wide legacy pytest collection
is not runnable under local CPython because it includes Pyodide-only top-level
`await` and unavailable legacy `wgpy_backends`, so use `test/test_*.py` for the
current source suite. The seven-point matrix, browser gates and final commit
remain open.

An initial runtime-only Q4_K shape A/B is **invalid and withdrawn**: its helper
used `webtorch._adam_kernel` instead of `webtorch._core._adam_kernel`, so it
changed the Python shape attribute but failed before resetting the captured
decode graph. The observed 16 equal outputs and rates therefore did not test
the proposed physical layouts. The helper must assert successful graph reset,
then the A/B must be repeated. After explicitly resetting the temporary graph,
capture pins were 0; one normal complete reply recaptured 984 pins / 5.58 MB,
and six further replies retained exactly that count. This distinguishes
temporary captures from a steady-state per-turn pin leak.

The corrected Q4_K A/B asserted 0 pins after every graph invalidation and
tested all 168 output-width>256 matrices. `balanced` won 10/18 fixed-seed SDK
pairs, not stable. `compact` initially won 8/8 SDK pairs but only 6/12 on the
next independent SDK series and 6/10 real chat-page pairs. All fixed-seed
SDK outputs matched in full; all page replies preserved the 1–40 sequence.
Thus a local candidate did not survive the required containing-API comparison;
no shape override was enabled. The default `auto` route was restored and a
final page reply remained correct at 126.88 tok/s (984 capture pins, 6 MB).
An existing WebGPU mapped-readback-buffer pool was likewise tested in 8
alternating product pairs. Both enabled and disabled routes had large drift
(79–131 tok/s) and no repeatable gain, so the opt-in pool was left disabled.
The `narrow` Q4_K layout, motivated by low output-row occupancy, was decisively
slower in 8/8 fixed-seed complete-SDK pairs (about 101–108 versus 128–133
tok/s), with identical full text. It too was restored, not enabled. The
adapter reports `shader-f16`, `subgroups`, and `timestamp-query`, but their
mere availability is not performance or semantic evidence for a new route.
A diagnostic same-width `tiny` (16×4) WebGPU Q4_K workgroup passed the
independent packed-weight selfcheck and produced the same complete fixed-seed
text, but won only 2/8 paired full-SDK runs (about 114–119 tok/s versus
default 114–120). It was removed from source and is not in `auto`.
Rebuilding the reverted wheel exposed nondeterministic ZIP timestamps that
changed the SDK version token despite identical source, discarding measured
profiles. All three wheel setup entrypoints now set a default
`SOURCE_DATE_EPOCH=946684800` while respecting an explicit override. Two
consecutive builds of WebGPU, WebGL and test wheels produced byte-identical
SHA-256 hashes; the stamp script was idempotent at `bf9fed4c0b`. This new
stamp still requires browser reloading and revalidation of the local 0.6B.

**2026-10-04 current 0.6B handoff:** The only in-app tab holds the sole local
`Qwen3-0.6B-Q4_K_M@ce11278f.gguf` on ordinary WebGPU. Five initial complete
231-context, 110-token page replies returned the 1–40 sequence at
118.56/124.44/106.84/135.82/123.26 tok/s. At position 231, the diagnostic
unsplit decode step was 6.785 ms; its stage trace was led by MLP 2.313 ms,
attention output 1.263 ms, head 1.257 ms and QKV 1.253 ms. Python input
preparation had a 0.260 ms median. An existing packed-Q6_K GPU embedding-row
candidate matched the host input, cosine and sine arrays exactly at three
token/position pairs, but its 3×3 page A–B–A (baseline 121.27/106.22/133.96,
candidate 139.60/137.31/106.33, returning baseline 130.84/136.94/137.31)
did not prove a sustained benefit; the instance method was restored. A JS
single-signal replay+sampling candidate passed its unit test and six complete
page replies (114.64–137.52 tok/s), then lost to a six-reply restored-path
median (candidate 127.97 versus restored 134.32 tok/s); its source and test
were reverted. The ordinary bundle was rebuilt/restamped to SDK hash
`4bbb2e651f`, released/reloaded from the same disk file, and left loaded.
Four interleaved SDK/page pairs showed no consistent page-only overhead.
Further same-instance whole-SDK shape checks left the incumbent `head_shape=compact`
and `gate_up=separate`: balanced/short-K head candidates were mixed, narrow was
clearly slower, and fused gate/up had no repeatable advantage. A 12-pair,
order-alternated, fixed-seed comparison of `qkv=fused:compact` against the
current `fused:default` produced identical complete text on all 24 runs but
won only 6/12 pairs (mean delta −1.46 tok/s), so it was not enabled. The
restored page again returned 1–40 at 132.19 tok/s. These tests reinforce
that isolated peaks of 140+ are not stable product acceptance.
One opt-in profiling script accidentally created a Python-backed JS diagnostic
object and caused one failed reply; that temporary flag was cleared and a
subsequent full reply passed. Current local checks: 79 JavaScript tests,
161 Python tests, TypeScript/webpack build and `git diff --check`. The product
140+ gate and the full seven-point matrix remain open; there is no final commit.

**2026-10-04 WebGL browser continuation:** On the in-app browser's only visible
tab, the runtime initially showed `ready (webgl)` with Release disabled; the
user-mentioned WebGPU `vocab-sample-candidate` tab was not present. The sole
local `models/Qwen3-0.6B-Q4_K_M.gguf` was therefore selected via the browser's
native file picker, with no network model transfer. It loaded 332.9 MB from
disk, retained the previously measured WebGL upper plan, and completed a
216-context, 110-token product reply at 18.76 tok/s with a 2.231 s first
token. The completed conversation contains exactly the requested 1–40 lines.
The WebGL page did not show the WebGPU-only slow-speed warning after the UI fix.
The WebGL model is currently the sole resident instance; GPU buffers read
448 MB. Local checks on this source: 161 Python tests, 77 JavaScript tests,
TypeScript build, version stamp and `git diff --check` passed. This does not
close the WebGPU product-speed gap or the seven-point acceptance matrix.

**2026-10-04 WebGPU 0.6B recheck:** After releasing WebGL (0 GPU buffers,
42 MB WASM), the same native GGUF was loaded alone on WebGPU. The saved
full decode plan and GPU vocabulary sampler were reused. Six default-1024
page replies measured 114.95/132.43/125.52/100.92/121.00/133.16 tok/s;
all returned the requested 1–40 sequence, though some used a code fence.
The replay-submission timing scope was about 0.3–0.4 ms/token; the mixed
selection/GPU-completion/readback scope varied from about 7.1 to 9.3 ms/token.
These scopes do not establish GPU arithmetic time. WebGL staged profiling gave
42.765 ms unsplit whole step and located its largest *diagnostic* stage costs
at QKV and MLP; WebGL SDK/page throughput was 19.01/18.76 tok/s. A sequential
1024→2048→1024 dispatch-flush A–B–A on the WebGPU product path gave six-run
medians of approximately 123.3/130.5/130.9 tok/s. The recovery after
returning to 1024 rules out a repeatable 2048 gain here; no threshold change
was made. The only current resident model is again Qwen3-0.6B on default
WebGPU. These observations do not establish stable 140+ product performance.

**Later 2026-10-04:** A further 512-dispatch six-run product series was
122.75/133.64/137.19/121.02/123.44/136.95 tok/s; the following default
series was 121.34/130.72/112.93/108.58/115.42/100.92. The large
same-route time drift prevents claiming a 512 win; the production default
remains unchanged. A fresh browser tab restored a 119.23–136.76 product
range, and direct SDK runs varied 112.03–142.50, so the 140+ product gate is
still open. One model was resident at a time. Model release repeatedly showed
GPU 0 KB and WASM 42 MB, while the shared browser GPU process RSS dropped
only about 1.81→1.78 GB immediately, then to about 1.65 GB after empty-page
reload/closing a temporary tab. This does not by itself distinguish driver
caching from a retained resource. The page's misleading `GPU + host` footer
has been corrected to `step + pick/readback`, because the latter includes
GPU sampling and completion waiting; the latest stamped browser showed
`step 0.34 ms + pick/readback 7.25 ms` at 129.85 tok/s and the requested
1–40 sequence. Current in-app tab is default WebGPU with only the local
Qwen3-0.6B loaded. Full JS suite now passes 78 tests; `git diff --check` passes.

**Hardware-clock correction:** The `profile_gpu=1` opt-in WebGPU timestamp
path was run sequentially on that same sole local model. Two warm three-token
runs each recorded three 366-dispatch compute passes: the first took about
15.0 ms and the next two about 6.29–6.42 ms. A 12-token diagnostic recorded
11 following main-pass times of 6.29–7.21 ms, plus roughly 2.03 ms total
across small auxiliary passes. The timestamp instrumentation itself perturbs
the short-run rate (111.89 tok/s for 12 tokens), so these are diagnostic
device durations, not an uninstrumented product speed. They prove that the
earlier 0.3–0.4 ms `gpu_ms` field was **submission**, not arithmetic, and that
GPU execution is a substantial part of the 7–9 ms completion wait. No
CPU-versus-GPU bottleneck conclusion should be drawn from the old footer.

**Product-format regression, latest stamp `76dfea11a1`:** The chat's old
system prompt ordered every reply to use Markdown and fenced code even when
the user explicitly asked for plain text. It now says the user's requested
format wins, while keeping Markdown as a default for unconstrained replies.
On the one local Qwen3-0.6B, three WebGPU product replies produced the 1–40
sequence without fences at 127.72/129.06/130.92 tok/s (231 context); two
sequential WebGL replies did the same at 18.36/18.47 tok/s, first-token
2.607/0.055 s. Some lines still have model-generated trailing spaces, so
this is **semantic/sequence parity, not byte-for-byte copying**. WebGL did
not show the false WebGPU slow warning. A shorter system prompt was explored
only as a direct-SDK candidate: five interleaved pairs showed mixed wins,
so it was not adopted as a measured page-level optimisation. The model was
released between backends; the original in-app tab is restored to normal
`?backend=webgpu` with only the local 0.6B loaded. Current local verification:
161 Python tests, 79 JavaScript tests, stamped source, `git diff --check`.

**Current priority, 2026-10-04:** The user supplied the single local
`Qwen3-0.6B-Q4_K_M.gguf` in the in-app browser. A proposed two-token GPU
sampling graph preserved the 1–40 output for three seeds but lost to the
one-token route in paired, same-plan measurements (candidate 134.04/135.10/
136.12 versus baseline 134.82/138.68/139.36 tok/s); it was removed from
source, stamped, and the model was released/reloaded from the same native file.
Release showed 0 GPU buffers and 42 MB WASM. The new build first loaded with
an explicit `budget_limited` original-width upper plan and no sampling profile:
one correct product reply was 110.8 tok/s. Reusing the previously correctness-
checked full plan and GPU sampler on this build improved the same prompt but
has **not** established a stable 140+ product result. Five interleaved direct-
API pairs (same seed, 216-token context, identical 1–40 text) gave GPU sampling
138.50–139.37 tok/s against JS sampling 115.12–127.50 tok/s; the GPU route is
the clear local winner. Three actual page replies on `auto` were correct at
122.3/135.4/132.4 tok/s. A direct measurement of the page token callback was
4.25 ms total over 115 tokens (0.01 ms median, 0.195 ms p95); throttling live
Markdown to 1000 ms did not help (129.0/121.2/128.6 tok/s) and was not kept.
The sampled speed is variable; do not call 0.6B accepted on the product path.
The new build's profile was verified on a fresh native-file reload: the full
plan and `vocab_sample_full=GPU` were automatically reused. The reloaded
page gave 114.6/130.2/135.7 tok/s (the first reply was cold), while five
warm direct-API replies without a fixed seed gave 140.24–141.30 tok/s in four
of five runs, plus one 126.45 outlier. A temporary 16 ms token-event batch
gave 119.8/137.1/136.3 tok/s and was not adopted; 50 ms batching was slower,
and one unseeded run ended after 4 tokens (not enough to attribute cause to
batching). The original page callback
was restored. Current browser model is still the sole resident 0.6B with
`auto` selected. A full offline upper-level composition search on this sole
instance completed in 295.9 s and chose the same prior plan after its
four-step semantic checks. Its diagnostic median was 10.215 ms; it found no
new correct positive upper combination. Following search, three direct API
replies were 138.03/140.50/133.74 tok/s and three product replies were
133.1/135.0/135.8 tok/s, all exact 1–40. Search temporarily raised the main
JS heap to about 153 MB; release returned GPU buffers to zero and WASM to
42 MB, then page/runtime reload lowered JS heap to about 23 MB. The model
was reselected from the same disk file and auto-reused the searched profile.
Clean-reload product replies were correct at 123.1/134.5/137.1 tok/s, so the
remaining product gap, WebGL, Laya, seven-point
acceptance and final commit remain open. Local checks: 161 Python tests,
76 JavaScript tests, JS build, version stamp and `git diff --check` pass.

**Current priority, 2026-10-03:** Qwen3-0.6B Q4_K_M is still not accepted as a
stable 140+ tok/s *product* result. The sole model was reselected repeatedly from
the same local 378 MB GGUF using the native file picker; no model bytes came over
HTTP. The SDK's device-local full-sampling profile now reuses a correct complete
decode plan and GPU vocabulary selection: fixed-seed, exact 1–40 direct API replies
settled around 139–141 tok/s, with 0.25–0.30 ms actual GPU execution per token
and about 6.7–6.9 ms host/GPU-completion time. A reloaded chat-page run with the
same stored profile and adaptive Markdown updates produced eight correct 1–40
replies; after one 127.85 tok/s opening run, seven were 136.41–137.54 tok/s.
The chat's extra system prompt is part of this comparison. The distinction
between direct API and visible chat remains open; do not report the product as
140+. Device-profile reuse has been verified, but it is stored in browser
IndexedDB for this source and device, not a universal route hardcoded by model
name. Same-seed logits agreed with the original-width reference at four steps
(max relative difference 0.0), and three complete seeded replies matched the
baseline text. WebGL's prior correct 1–40 product reply was 18.2 tok/s with a
2.4 s first token; its further optimisation remains open.

Interleaved full-output browser A/B tests found no reproducible advantage from
merging WebGPU readback submissions, pooling the four-byte readback buffer, or
batching worker token notifications. Those candidates are disabled/reverted.
The 4096-dispatch flush threshold was also tested in ten correct complete chat
replies (settled 135.4–137.0 tok/s) and did not beat the default; the original
URL and loaded WebGPU model were restored.
The actual per-token page callback averaged 0.051 ms, while asynchronous live
Markdown rendering consumed about 20 ms over 115 tokens. Reducing Markdown
refreshes to at most once per 200 ms *only when recent token gaps average under
12 ms* showed a small positive complete-page gain; slower streams retain the
60 ms cadence. This generic UI optimisation is in source and has a unit test.
On the final stamped build, WebGL reloaded the same local GGUF and gave two
correct 1–40 page replies at 18.29/18.68 tok/s; all 115 live updates in each
reply chose the original 60 ms interval (first-token 2.231/0.057 s). Browser
reload/release checks returned GPU buffers to 0 and WASM to 42 MB. Final local
checks on this snapshot: 76 JavaScript, 161 Python tests, TypeScript build,
and `git diff --check` pass. No final commit; seven-point acceptance remains open.
The user accepted prior 30B WebGPU throughput; 30B WebGL load/latency, Laya,
generic API/format parity, broader browser regression and clean final commit
remain open. Never co-load models or fetch their bytes from network.

The user accepted measured WebGPU 30B performance on 2026-10-03; WebGL 30B
remains active. Both backends now use shared-memory command/upload transport,
direct shared readback, JS-side unconstrained sampling with retained token
counts, and JS-side MoE host routing that writes output in place to shared
staging. Exact contiguous NumPy uploads are compared per physical shape on
repeats because browser timings showed direct access can be slower; a one-off
shape keeps the established path. WebGPU metadata no longer makes a Python
comm-buffer and transferred JS-array copy. Both backends' readback arenas now
allocate from 64 KiB and grow on demand instead of pinning 64 MiB immediately;
a synchronous WebGPU readback failure now wakes the waiting worker, matching
WebGL. Current local checks pass 160 Python/71 JavaScript tests, TypeScript
compilation and `git diff --check`. A new-session native file selection enabled a full
WebGL 30B retest: one final response correctly counted 1–40 at 114 tokens,
1.5 tok/s overall, 3.8 s first token, and roughly 180–190 ms/token in its
settled tail. Its cold beginning was much slower, and load `proving` took
20.9 s versus the strict ≤10 s gate. A previous same-build, longer-context
reply was correct at 3.8 tok/s but 27.1 s first token (240-token context),
so neither is a complete latency/throughput pass. The model was released;
WASM fell from 1.40 GB to 50 MB. Profiling then found one 7.0 s WebGL
`cat2_gl` compile/link stall and context loss during first-forward readback.
Failed-load recovery now closes and rebuilds the runtime; an invalid local
GGUF and an actual 30B context-loss run both returned to ready(WebGL) at
about 42 MB WASM instead of stranding 1.40 GB. Precompiling the generic
growing-KV shader before weight upload removed that local stall but did not
prevent context loss. An opt-in 64 MiB upload-flush candidate let the 30B
load succeed, yet proving took 33.4 s and its deliberately stopped reply only
reached 18 at about 0.4 tok/s. It remains disabled by default; this is not a
complete semantic or paired performance result. A later no-flush load also
succeeded but spent 35.4 s proving. WebGL GPU texture accounting now runs in
JS and reaches the product via shared memory: this model held 12.73 GB and
peaked at 13.33 GB declared texture storage, excluding driver copies. Its
independent 98-context reply correctly counted 1–40 at 114 tokens, 2.9 tok/s
overall, 2.8 s first token, and roughly 180–185 ms per final GPU step. Release
showed GPU 0 KB and WASM 42 MB. OS free memory fell to tens of MiB, but the
precise driver/memory split is not proven. No HTTP model
fallback was used. Other Python
NumPy/format paths and per-op scheduling are still outside the new JS-ownership
rule. The WebGL 30B performance gate, 0.6B, Laya, full seven-point acceptance
and final commit remain open. Never co-load models or fetch them from network.

Current production-parameter 30B retest after runtime hard-restart: the same native-file
GGUF loaded with `warm-step 9.3s` (first GPU synchronisation 9.025s), then answered
1–40 correctly at 34.1 tok/s with 3.6s first token. This one passing warm sample
does not supersede the 13–14s warm samples or the newly repeated 15.3s one-row
first-token stall (the first GPU compute pass itself was 31ms). A same-model repeat
immediately afterward was 1.2s to first token. The release/restart path again left
0KB GPU buffers and a 42MB WASM heap before reselecting the file. Exceptional
worker-termination/backend-disposal ordering is now hardened; 140 Python tests,
37 JavaScript tests, and TypeScript compilation pass. No commit; 30B latency and
top-level throughput gates remain open.

Newest resource result: `release()` alone made the GPU ledger zero but left
the 30B browser GPU process near 14 GB and the Pyodide heap at 987 MB.
The SDK/page now closes and rebuilds the whole runtime after release, with
matching WebGPU/WebGL device/context teardown. In a real WebGPU browser run,
the GPU process fell to 813 MB and WASM to 42 MB without page reload;
reselecting the same local GGUF worked, and its next correct 1–40 reply ran
at 38.34 tok/s with a 4.685 s first token. The latest cold warm steps were
14.1 and 13.7 s, still over the 10 s limit; one-row TTFT also previously
reached 18–20 s. This does not close the 30B performance gate or permit moving
to 0.6B/Laya. Browser WebGL release and semantic parity remain unverified.

Newest 30B evidence: the source-stamped native-file load reused the real-sampling
profile (25.37 ms median), and four correct 1–40 product replies ran at
34.53–36.96 tok/s, but first token ranged from 3.254 to 20.035 s. A later
load displayed `warm-step 10.2s`, beyond the 10 s limit. Main-thread GPU
timestamps measured only 30–33 ms of actual compute during 13–17 s one-row
replay stalls; both logits and timestamp readbacks waited together after
submission. System VM sampling during a stall showed intensive paging. This is
queue/residency delay under memory pressure, not a solved arithmetic bottleneck;
the precise driver/memory split and robust latency fix remain open. The 30B
remains the only loaded model and active priority. The temporary browser reply
cap and GPU hooks were restored, and a mistaken Worker-parameter diagnostic
patch was fully reverted after confirming WebGPU lives on the main thread.

Latest: the user-reloaded local 30B gave one correct cold 1–40 reply at 34.46 tok/s,
3.486 s first token, and a 10.0 s displayed warm step; its cached-prefix repeat was
35.55 tok/s and 0.212 s first token. Earlier same-route replays at 7–12 s and a
40.816 s full-prefill first token keep the latency gate open. A 12.3 s replay was
12.2256 s inside full-logit GPU synchronisation, not input setup; memory swapping
is substantial but not isolated as the sole cause. The old upper profile was measured
under greedy selection while the chat actually samples at temperature 0.6. A
sampling-route candidate did not beat the incumbent consistently (2/5 direct pairs),
so it was not adopted. Source now separates those profile modes and times all valid
offline candidates; the new-source browser reload, expanded 30B search, WebGL work,
0.6B/Laya sequence, and seven-point final acceptance remain open. One model at a time,
local native-file loading only. Local tests: 140 Python, 35 JavaScript, TypeScript pass.

The latest source-stamped browser run on the user-loaded local 30B proves a
generic cached-prefix one-row prefill improvement: correct 1–40 output at
34.9 tok/s and 0.194 s first token (new `decode_replay` route), versus repeated
12–16 s one-row stalls on general prefill. It is not a complete latency fix:
a later zero-dispatch replay still took 12.261 s to first token under system
paging. The cold turn on that build was
correct at 33.6 tok/s and 5.493 s first token; this cold warm-step was 8.6 s.
135 Python/35 JavaScript tests and TypeScript check pass. This is one fixed
latency path, **not** completion: 30B throughput is below the near-40 goal,
cold latency repeatability, WebGL optimisation and all other acceptance gates
remain open. Five latest paired product replies favoured a fused upper-level
candidate on settled throughput, but its first switch cost 14.163 s to first
token. An explicit full upper-composition search on the same resident 30B
completed in 52.166 s and chose fused add/RMS plus lower-layer auto routes;
the resulting product reply was correct at 35.63 tok/s and 0.245 s first token
after its prefix was warm. Cold paging still caused a separate 14.7 s first
token, and this measured plan is not yet persisted for automatic reuse on a
new load. WebGL throughput can be lower according to comparable industry
workloads and same-device measurement; its loading/first-token latency must
still remain seconds-level and semantic parity is not relaxed.

The latest stamped-browser retest on the same native-file 30B produced correct 1–40
replies at 33.2 and 33.3 tok/s, with 4.6 and 2.9 s first-token latency. A product-path
fusion pair gave only 33.6 versus 33.7 tok/s; the candidate was not enabled. Q3_K MoE
shape trials likewise found no reproducible gain beyond the existing route. The current
`auto` remains budget-limited with zero full-API candidates tested, so WebGPU throughput
and the two-stage acceptance remain open. WebGL follows WebGPU optimisation: its tok/s
target may be lower, calibrated against comparable published work and same-device tests,
but load and first-token latency must stay seconds-level and semantic parity remains
mandatory. Do not infer a fixed WebGL/WebGPU ratio across models, formats or devices.
The latest same-file cold run showed `warm-step 9.9s` (one pass, not a stable bound),
and a settled one-row-prefix reply still suffered 18.9s to first token. Repeated
loaded-model tests did not justify 512 or 1024 dispatch flushing over 2048, and an
exact Q3_K MoE SwiGLU/down fusion was not faster in a paired microbenchmark. The
temporary flush diagnostic was removed from source. Full-step profiling again points
to MoE MLP (12.47ms of 25.95ms). The next WebGPU work must attack this high-level
latency/throughput gap without sacrificing semantics; do not treat the near-40 or
seconds-level latency targets as met by an isolated sample.

**Not complete.** A 2026-10-03 browser cold-run found a concrete first-step stall:
48 MoE layers each lazily uploaded a four-to-eight-element zeroed routing-weight buffer,
forcing 49 WebGPU submissions in the first decode versus one in the second. An opt-in
timestamp trace measured exactly 1061 dispatches in each step and only 65.99/61.67 ms
of GPU execution, while the first host sync waited 16.99 s. The kernel overwrites every
routing slot before use, so the SDK now allocates the tiny weight buffer uninitialised on
device (WebGPU and WebGL equivalent API). With the same local GGUF and diagnostic settings,
the first step was one pass and 10.44 s sync; the second was one pass and 0.043 s sync.
The product reply remained correct (1–40) but ran at 33.5 tok/s and first token 3.0 s.
This is a causal improvement, **not** the ≤10 s or near-40 tok/s acceptance result; the
remaining cold wait and complete-API throughput are open. No model was downloaded.
Two uninstrumented follow-ups measured 13.085/12.942 s first sync. A queue probe found
`onSubmittedWorkDone()` at 12.395 s versus 12.351 s Python wait: the remaining stall is
before GPU queue completion, not `mapAsync` or UI. System VM sampling shows high swapping
and near-zero free physical pages while this 13.8 GB model is resident; a direct causal
separation of driver compilation and memory residency is still needed. The full local
Python/JS/TypeScript suites pass (123/33 respectively), but browser WebGL and final
acceptance do not. The 30B remains the active priority; do not move to 0.6B yet.

The GGUF loader now releases its large CPU/JS transfer window immediately
after weight upload, before warm-up. Three same-build, same-file 2048-dispatch
cold runs measured first GPU sync at 8.966, 15.899 and 13.789 s; only the
first had a 9.1 s `warm-step` and correct 1–40 product output at 34.6 tok/s.
The other two fail strict warm ≤10 s, so this is retained as memory hygiene,
not a proven warm fix. OS 1-second sampling during the third showed free pages
falling from ~807k to <4k and bursts of 20k–54k 16-KiB swap-out pages/sec;
memory pressure is a strong contributor candidate, not an isolated sole cause.
The third post-load reply was correct at 35.1 tok/s but first token still took
6.1 s for 98 prompt rows. On the loaded system, `top` showed about 13 GB in
the Codex service, 23 GB/24 GB physical memory used and an 11 GB compressor.
127 Python, 34 JavaScript, TypeScript, wheel consistency and diff checks pass;
the seven-point task and final browser/commit gates remain open.
The measured cold-start variance under sustained system swapping prevents a
defensible ≤10 s acceptance claim on this host. Do not silently move to 0.6B
or Laya, close unrelated user applications, change the model quantisation, or
commit the incomplete task. A reproducible low-pressure run on the same local
file is needed before deciding whether further SDK work alone can meet the
bound; current evidence does not prove either impossibility or completion.

Latest user-selected local 30B run on `dispatch_flush=2048` reported a
`warming 1.9s` + `warm-step 7.4s` stage pair (9.3 s for those stages) and
`114 tokens · 33.6 tok/s · GPU 28 ms + host 1 ms · 97 ctx · first token 5.4 s`.
The user confirmed this run's latency target was met. A second correct 1–40
reply on the same loaded model measured 35.1 tok/s with 1.9 s first token.
This is a successful individual warm run, not yet proof of repeatability or
the near-40 tok/s complete-API throughput target. The live plan remains
budget-limited `composed`/`full` with zero upper-composition candidates tested.
Layer profiling attributed about 12.7 ms to MoE MLP out of a 24.5 ms whole
decode step; next throughput work should target that path and re-evaluate its
containing API. The browser also left `Stopping…` visible after a completed
answer. Source now clears only that stale status at turn completion; a unit
regression passes, but the already-loaded tab predates the new stamped source
and has not been reloaded. Its stale text was cleared without releasing the model.
An additional runtime-only fused add/RMS + Q/K norm/RoPE comparison matched
four sequential full-logit outputs and won 8/9 complete-step pairs
(26.92→24.85 ms median), but actual chat was only 34.9–35.6 tok/s, not near
40. A diagnostic baseline recapture incurred 15.2 s first-token latency with
one prefill row. The candidate was restored to composed/full; selecting it as
top-level `auto` requires the product-path benefit and first-use cost to be
resolved, not merely its isolated graph timing.
At context positions 97 and 200, alternating same-graph medians were
25.748/25.825 ms; growing context alone is not the observed multi-millisecond
rate drift. A later correct baseline product reply was 35.3 tok/s but its
first token took 7.8 s with one new prefill row. A concurrent three-second
system VM sample recorded roughly 691 MB of swap-in and 88 MB of swap-out,
plus a transient wired-memory jump. This supports continued residency/queue
investigation but does not identify a sole process or prove an SDK-only fix.
Manually draining 131 MB of idle GPU pool after a reply kept all 1,833 capture
pins intact, but the following correct reply was 32.9 tok/s and 17.2 s to
first token for one new prefill row. That does not prove causation; it also
does not justify adding per-reply pool eviction as a performance fix.
Source now has a capability-checked cross-turn WebGPU decode-graph reuse path
for stable full-attention models, shared by streaming and non-streaming API
calls. Recurrent models keep recapture; WebGL keeps its eager equivalent.
132 Python/35 JavaScript tests, TypeScript and whitespace checks pass, and the
SDK version stamp changed. This is **not** yet a browser-verified TTFT fix:
the user's loaded 30B worker predates the new Python source, and its local
file was not re-picked or downloaded. Actual model correctness, latency and
memory lifetime must be tested on the new stamp before accepting this route.

Latest diagnostic `dispatch_flush=512` on the same sole local 30B took 18.166 s
for the first decode GPU sync (construction 16.7 s), although its correct
1–40 product reply reached 36.5 tok/s. A runtime-only fused/device-greedy
candidate reached 37.6 tok/s on a second reply but cost 18.5 s to first token
on first use, so it was rejected and the worker restored to composed/full.
The 512 and fused results do not pass warm ≤10 s or prove upper-API optimality.
With 30B resident, macOS reported ~23/24 GB physical memory used and 11 GB
compressed, a possible contributor to the cold GPU queue wait, not an isolated
cause. Continue 30B cold/warm diagnosis before 0.6B, then Laya.

Latest same-file cold baseline measured 10.503 s first sync and 34.4 tok/s on a correct
1–40 reply. A runtime-only checked fused candidate gave 36.4–36.6 tok/s; adding device
argmax gave 37.0 tok/s in a separate run. None establishes the upper-API optimum or
passes strict warm ≤10 s. An asynchronous duplicate-pipeline compilation trial measured
11.729 s first sync and was removed. Streaming generation was found to replay the
graph-recording step at the same token position, doubling a recurrent state transition;
source now consumes that captured result and avoids a final unneeded forward on both
WebGPU/capture and WebGL/non-capture paths. Three new regressions and the full local
126-Python/34-JavaScript suites pass. A same-file browser reload confirmed the worker
contains the fix and returned 1–40 correctly at 34.5 tok/s; cold first sync was still
10.393 s. A GQA single-kernel route won 25/30 graph pairs only at decoder position 0;
at the actual 98-token chat position it won just 19/30, barely changed median latency,
exceeded the strict logits tolerance and slowed the product reply to 33.8 tok/s. It was
rejected and runtime diagnostic overrides were restored. Do not report completion.

The currently loaded local 30B WebGPU GGUF runs on the bounded-staging
upload path. The latest same-build default-order load measured 12.09 s first decode GPU
synchronisation (above the 10 s limit) and 35.0 tok/s in a correct 1–40 product reply.
The reverse-upload experiment was slower at 14.17 s and has been removed. A fused
add+RMSNorm/QK-norm+RoPE complete-graph candidate won 17 of 20 paired rounds with exact
tested logits (24.71 versus 25.32 ms median), but is not yet a persisted `auto` route;
actual sampled replies were only about 35–38 tok/s. The next work remains cold GPU
residency and complete-API routing, with 30B/0.6B/Laya in that order. A local-file
progress-log reset regression has been fixed and unit-tested, but not yet verified on a
reloaded browser instance. The source-stamped wheels are rebuilt; current subset suites
pass 122 Python, 32 JavaScript and TypeScript, without final commit or acceptance. A
same-build native-file experiment now retains WebGPU MoE transpose sources in bounded
32-expert windows: model construction was 16.8 s versus 19.4 s for the old one-expert
sync path, while first GPU sync was still 16.20 versus 17.34 s. This is an upload gain,
not the required ≤10 s warm fix. WebGL keeps its equivalent whole-stack operator path.

Earlier baseline context: the same model runs on the bounded-staging
upload path. Earlier fresh local-file browser runs produced correct 1–40 output at 35.4 tok/s
(2048 dispatch flush) and 37.0 tok/s (512 flush); these are separate runs, not enough
paired evidence to select 512. The interactive load now skips the indivisible 16–19 s
whole-model semantic oracle and selects its exact original-width baseline. Its remaining
first decode GPU synchronisation is still above the user's ≤10 s warming limit: measured
12.74, 14.69 and 37.79 s on baseline-family runs, with host dispatch only 0.05–0.10 s.
The total warm-step stage was 13.3, 15.4 and 40.7 s. Full-weight upload pre-touch and
large `queue.writeBuffer` trials worsened this stage to 27.6 and 21.6 s; both were removed
from source and the JS bundle rebuilt. VM counters during a 30B load showed substantial
compression recovery and swap-in, a strong memory-pressure clue rather than proof of the
sole cause. **The 30B warming gate remains open.** Next: resolve cold GPU synchronisation
without sacrificing semantics, then 0.6B speed, then Laya BERT; 27B/full WebGL and all
seven-point product/final repository gates remain open.

A same-model upper-API candidate was measured after load: fused add+RMSNorm together with
fused Q/K norm+RoPE retained identical tokens and logits across four sequential decode
steps. Five interleaved complete-generation pairs at 90 tokens all favoured this combination
(median 40.99 versus 39.38 tok/s with greedy decoding). It was enabled only in the live
browser worker for that measurement; it is **not yet an automatically selected production
route**. Fused QKV was correct but about twice as slow and must not be selected here.

An upload-lifetime correction now waits for each large tensor's GPU copies before dropping
their mapped staging sources and waking the worker; byte-copy and async-error tests pass.
The first fresh local 30B load with that path still spent 16.06 s in first decode GPU sync,
so the ≤10 s gate is open. A second cold-load diagnostic synchronized every decode stage:
15.81 of 16.66 s fell in MoE MLP, especially layers 9–24 (roughly 0.7–0.97 s each);
QKV was 0.35 s total. The subsequent ordinary step's first/second GPU sync fell to
1.97/0.044 s. Pipeline creation was only 0.085 ms across 33 pipelines, so shader
compilation is not this wait. Cold expert-weight residency is the leading hypothesis,
not yet proven or resolved.
An additional bounded `queue.writeBuffer` upload trial made first GPU sync 28.38 s;
it was removed, leaving the mapped-staging lifetime correction as the production path.
A runtime-only selected-expert-page prefetch trial measured 14.41 s of first GPU sync,
versus two bounded-staging baseline readings of 16.06/16.41 s. That small unpaired
difference does not establish a repeatable win and still fails 10 s; the probe was removed
and its 96 tiny scratch buffers released.
Pre-warm release of completed scratch buffers reclaimed 280,951,332 GPU bytes in
0.0017 s on one fresh local-file load, without touching live weights or capture pins.
That run still spent 15.147 s in first decode GPU sync (second sync 1.865 s), so
the memory cleanup is retained as backend-neutral hygiene, not claimed as a warm
latency fix. A regression test covers the WebGPU and WebGL idle-pool contract.
Simply skipping the full decode warm shifted the cold work to the required full-forward
proof (11.5 s) and pinned 96 more buffers during first capture (not proof of extra
dispatches); a two-position proof followed by the
original warm step still took 12.855 s. Both runtime-only rearrangements were rejected.

The already-loaded 30B's warm 98-row prefill measured 2.858 s with explicit stage
synchronisation: MoE MLP 2.206 s, QKV 0.327 s, attention/output 0.211 s, other 0.115 s.
This is a *warm* diagnostic, not proof of cold first-token latency. Four native MoE
workgroup shapes were also checked on its real Q3_K expert weights; narrow was slower,
while default/balanced/compact differences were not repeatable enough to select one.

Layer profiling exposed a separate memory bug: `resetCaptures()` cleared JS captures but
left Python `_pinned_ids` intact. After profiling, pinned buffers climbed from about 1,929
to 15,675 (207 MB). Source now resets JS first, unpins Python IDs, destroys only orphaned
buffers, and preserves still-live Tensor ownership; an isolated regression test passes.
The already-running browser worker predates this fix, so browser memory verification is
still required on a rebuilt/reloaded instance.

## Completion evidence

- Phase one: all 28 accepted GGML formats pass stored-format GEMV, GEMV2, small/large GEMM,
  MoE decode and MoE batch checks on both backends; GPTQ INT4/INT8 matches an independent
  dequant reference. Same-width scalar/vec4 choices are independent by backend, format and
  batch mode, with no activation or weight width conversion.
- Phase two: WebGPU measures stored, activation-INT8 DP4A where available, and materialized
  execution per format, shape bucket and device profile. Nine interleaved paired rounds use
  an exact sign test, not a speedup-percentage cutoff: any repeatable positive latency result
  is retained, while an inconclusive result stays on the lower-memory candidate. M=1 and M=2
  are measured rather than hard-coded. WebGL explicitly retains its exact packed path because
  it has no equivalent packing/dot or materializer primitive.
- Parity: optimisation scope descends global → backend → format → operator mode → shape
  bucket → device profile, so a global loss never erases a local win. If a low-level WebGL
  primitive is unavailable, equivalence moves upward to the nearest efficient contract.
- End to end: local Qwen3-0.6B Q4_K_M returns deterministic `OK` on WebGPU and WebGL while
  retaining 168 Q4_K plus 28 Q6_K linears. The independent Qwen2 reference matches on both
  backends (maximum logits error `4.84e-8`, identical greedy tokens).
- Broader WebGL API gate: Conv2d/BatchNorm/MaxPool/Linear autograd training converges from
  loss `0.614` to `0.0` at 100% accuracy.
- Automated suites: 67 Python tests and 19 JavaScript tests pass; both checked-in backend
  wheels are rebuilt from source and current.

## Final no-cutoff phase-two audit

- DP4A is a real `QuantizedLinear(auto)` production candidate on WebGPU, not a benchmark-only
  experiment. It passes a per-input numerical gate before timing; unsupported or inaccurate
  candidates are removed without preventing other alternatives from competing.
- The target WebGPU selected stored for all measured DP4A buckets because DP4A was actually
  slower or inconsistent, not because it failed an arbitrary gain margin. Final speedups
  versus stored were INT4 `0.569/0.946/0.546/0.503×` and INT8
  `0.612/0.928/0.384/0.393×` at M=`1/2/32/128`. AutoGPTQ's zero-minus-one convention also
  passed, with maximum relative errors `0.00671` (INT4) and `0.00522` (INT8).
- The stored-versus-materialized matrix retained stable local materialized wins, including
  F16 at M=8/32/128 and several packed formats at M=128. Results with a faster median but
  insufficient paired evidence remain stored for lower memory and are reported as
  inconclusive rather than mislabeled as losses.
- Browser gates after the routing change: WebGPU 140/140 and WebGL 140/140 native-format
  GGML cases; WebGL GPTQ INT4/INT8 operator equivalence; local Qwen3 Q4_K_M returned `OK` on
  both backends while retaining 168 Q4_K and 28 Q6_K linears.

## Latest round — stored-weight execution and progress correctness

- All loader-supported GGML formats have native stored compute coverage; GPTQ INT4/INT8 uses
  the same model-agnostic interface and policy.
- Stored execution is the default. Stored/materialized auto-selection remains an explicit
  diagnostic opt-in until each original-width path has received its applicable hardware
  optimisation and passed numerical gates.
- Phase one never changes operand width as an optimisation. Q8_0 now vectorises reads and
  multiply-accumulate directly from its original signed-byte blocks, without requantising
  FP32 activations; packed INT8 dot with activation quantisation is deferred to phase two.
- Q4_0, Q4_1, and Q4_K now consume their original packed nibbles four bytes at a time and
  use exact vec4 multiply-accumulate. Two realistic-size interleaved runs showed 1.01–1.14×
  speedups without any alternate-width weight or activation path.
- Q5_0 now vectorises its original nibble plus high-bit plane for every shape. Q5_1 keeps
  scalar GEMV/GEMV2 because decode measurements were unstable, and uses the measured-positive
  exact vector path only for GEMM; neither path changes weight or activation width.
- Q6_K now reads its original nibble and two-bit high planes four source bytes at a time.
  Two independent runs showed 1.17–1.28× across M=1/32/128; the 140-case matrix and local
  Qwen3 Q4_K_M model (which contains 28 Q6_K linears) both pass with the production path.
- Browser verification after the exact Q8_0 vectorisation: xDecision returned the same
  non-uniform result on the duplicate-charge example — billing 89.7%, true 97.9%, urgency
  15.7% / 26.6% / 27.3% / 30.4% — in 304 ms for 486 input tokens.
- The browser correctness gate covers 140 stored GGML operator cases plus the device-side
  decision-feature reduction.
- The repository's complete `models/Qwen3-0.6B-Q4_K_M.gguf` now has a dedicated uncached
  end-to-end browser smoke test. It loaded in 1.2 s, answered `OK`, and reported 168 Q4_K
  plus 28 Q6_K native linears; no tensor was converted to another width.
- Automated verification: 66 Python tests and 19 JavaScript tests passed.
- The SDK cache token now covers the Python package and both worker bootstraps, not only the
  main-thread JavaScript file.
- Quantized ONNX is handled by the shared `OnnxModel` graph interpreter on both backends,
  with INT8/UINT8 operands and INT32 accumulation for the standard integer matmul and
  convolution operators.  The optional vision-decision adapter separately uses ORT WebGPU
  versus ORT WASM.  Neither path pretends ONNX tensors are GGML/GPTQ blocks.
