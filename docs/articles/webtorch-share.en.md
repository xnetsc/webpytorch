# Running a 30B model in a browser tab: the hard part isn't the compute

webtorch is a PyTorch-style SDK that runs in the browser, on WebGPU.

Results first (all of these run in the browser, nothing to install):

| Model | Size | Speed |
|---|---:|---|
| Qwen3-0.6B | 0.4 GB | ~175 tokens/s |
| Qwen3-30B-A3B (MoE) | 13.8 GB | ~42 tokens/s |
| Qwen3.8-27B (hybrid SSM) | 9.8 GB | ~7 tokens/s |
| 22-layer decision model | 0.7 GB | ~15 ms per question |

Machine: M5 MacBook, Chrome. Same below.

---

## 1. Why the browser

Running a model locally, the hard part is the environment, not the model. CUDA and driver versions have to match, AMD depends on whether ROCm supports the card, Macs go through Metal, many Windows laptops have only an Intel iGPU, and then there are Python versions and build flags. Plenty of people never get past installation.

WebGPU hands that problem to the browser vendors. Write WGSL once; the browser translates it to Metal, Vulkan or D3D12, and it runs on Apple, NVIDIA, AMD and Intel GPUs. Driver and vendor differences are the browser's problem.

Other benefits: open the page and it works; weights live in the browser's cache and prompts never leave the machine; the compute is local, so there is no server bill per inference; without WebGPU it falls back to WebGL, slow but working.

The Python layer exists because people who write models think in PyTorch. Pyodide compiles CPython to WebAssembly and runs it in the page; webtorch implements a PyTorch-style API on top: `Tensor`, autograd, `nn.Linear`, `optim.Adam`, and `import torch` can point at it. PyTorch code mostly runs unchanged.

```python
import webtorch
webtorch.use_default_io()

lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen3-30b-a3b.gguf")
print(lm.generate("Why is this surprising?", max_new=64))
```

This runs in the page.

The first plan was to fake `navigator.gpu` on WebGL2: translate WGSL to GLSL and compute in fragment shaders, so WebGPU-only libraries would run on older browsers. Vector addition worked, but WebGL2 has no compute shaders. A fragment shader writes one value per pixel: no scattered writes, no atomics, no shared memory, no barriers. The spec rules it out.

Every operator now has two implementations, one for WebGPU and one for WebGL. The base is WgPy, which already had the three hardest parts: two backends, Pyodide integration, and synchronous Python waiting on the GPU.

---

## 2. Structure and rules

```
your code (PyTorch style)
   ↓
webtorch: tensors, autograd, nn, LLM engine, quantization, ONNX
   ↓
WgPy array layer
   ↓
WebGPU (WGSL) / WebGL (GLSL)
```

Python runs in a Web Worker; the GPU device is on the main thread. Python waits on GPU results synchronously through `SharedArrayBuffer` and `Atomics.wait`, so the page must be cross-origin isolated (COOP/COEP headers). Without isolation, `SharedArrayBuffer` is unavailable and the model silently falls back to the CPU, thirty times slower. The SDK now reports the reason.

What actually shapes the design is a handful of rules:

1. **Python only schedules.** Computing, storage, copies and transfers happen in JS and on the GPU; Python holds buffer ids. Every crossing between Python and JS has a cost.
2. **Do in one call what could take ten.**
3. **Ask the device before using a capability.** GPUs differ from machine to machine: Intel, NVIDIA, AMD and so on.
4. **Compute in the format the weights are stored in.** Q4_K weights are decoded as Q4_K in the shader, not quietly converted to fp16. A different precision is allowed only as an explicit candidate that passes an accuracy gate and wins on measurement.
5. **Race, then choose.** Most operators have several implementations. The fastest on this device, for this format and this shape, is measured on the spot and remembered. No "5% faster or it doesn't count" threshold: a statistically solid win is a win.
6. **Interfaces follow the task, not the model's name.** A model's structure comes from its own config.

---

## 3. Pitfalls

### 3.1 The cost is between Python and JS

When training first worked, a tiny Transformer took seconds per step. Profile: of a 21 ms step, 17 ms went to Python walking the graph and issuing operators one by one; the GPU mostly waited. The decision model was worse: one forward pass spent 36 ms in Python issuing 550 GPU commands, and 0.7 ms computing.

Every call from Python into JS inside WebAssembly has a fixed cost. One call is cheap; hundreds or thousands per step are not.

The fix is record once, replay many times. The first time a step runs, JS records every command it issues (kernel, bound buffers, workgroup count). After that, new inputs go into the buffers the recording bound, and one call replays the whole step. Results are bit-identical; 17.8× faster.

Recording and replay had three problems:

- **Buffer ids get recycled.** The buffer pool hands freed ids to new allocations. A recording finds buffers by id, so a reused id makes the replay read unrelated data: NaNs everywhere. Buffers a recording uses are held back from the pool.
- **Held, never released.** On the 27B, every reply was slower than the last, and a page refresh made it twice as fast. Each generation re-recorded the decode step and held a fresh set of buffers; the previous set was never released. Now each recording owns its buffers and releases them when it is replaced.
- **The first recording caught extra work.** Also on the 27B: the first half of every reply ran three times slower, then sped up at token 447. The first run of a shape registers its kernel, and registration runs a numerical self-check. That first run happened inside the recording, so the self-check was recorded too: 1,728 extra commands replayed every token. Now one decode step runs outside any recording at load time, before recording starts.

### 3.2 The GPU is short of work, not bandwidth

A decision request has tens to hundreds of tokens, so its matrices are short and wide. Optimizations tried, and results:

- Fusing the activation into the previous step: a quarter fewer commands, almost no change.
- Padding K/V to an aligned size, to use a kernel twice as fast in isolation: slower overall.
- Quantizing weights to int8/int4: several times fewer bytes read, about half the speed.

What worked: there were two matmul kernels, and the fast one required the row count to be a multiple of 32. The row count is the token count, which the caller does not control, so nearly every matmul fell through to the slow kernel. Lifting the restriction made matmuls 6× faster.

At this scale the GPU is short of work: the same kernel on the same weights does 234 GFLOPS at 69 rows and 1,573 at 800. Dequantization is arithmetic, and arithmetic is what is scarce here, so quantization made it slower. Quantization is for fitting (a 27B does not fit otherwise), not for speed.

### 3.3 Don't make the GPU wait for the host

In decoding, the host waits for the GPU after every token, reads the result back, picks the next token and feeds it in. The logits for a 150k vocabulary are 600 KB, but on Apple's unified memory, reading 4 bytes and 608 KB measured the same, within 0.015 ms. No data crosses a bus; the time goes to the waiting itself.

What changed:

- Token selection runs on the GPU.
- The position counter and the rotary-embedding rows live on the GPU, advanced by the kernels.
- While the host reads chunk n, chunk n+1 is already queued, with one crossing per round.

The 0.6B went from about 140 to over 175 tokens a second, with identical text.

### 3.4 Against MLX

On the decision model, MLX measured 19 ms for three questions; WebGPU started at 100 ms. Merging the attention dispatches, folding the residual add into LayerNorm, and laying questions end to end instead of padding to the longest brought it to 44 ms.

The rest is the platform. Matmuls reach about 4 TFLOPS on this chip, against a 6.2 TFLOPS half-precision peak. MLX reaches 12 through Apple's matrix units, which WebGPU cannot use. Chrome's experimental subgroup matrices (behind a developer flag) measured 3.9: still the ordinary ALUs.

In everyday use it felt like MLX at twenty-something ms against 80 to 90 the first time and 50 to 60 after. The two numbers were measured differently: MLX back to back, webtorch with seconds between questions. Apple silicon clocks down when idle. After 200 ms of GPU idle the same work takes 2.6× longer; after 50 ms of CPU idle, 3 to 4×. A kernel that settles at 0.77 ms ran 3.6, 2.4, 2.0, 1.5, 1.2, 0.98 ms from idle; full clocks take fifteen to twenty milliseconds of continuous work. A question every few seconds always lands on a chip that just woke up.

MLX too:

| | back to back | every 2 seconds |
|---|---:|---:|
| MLX, one question | 7.6 ms | 35.8 ms |
| webtorch, one question | 15 ms | 32~38 ms |
| MLX, three questions | 19.0 ms | 48.2 ms |
| webtorch, three questions | 44 ms | 64~70 ms |

One question every two seconds: even. Three questions: about 1.4× apart, which is the matrix units.

Back to back or spaced out, peak or steady state, greedy or sampled: numbers from different conditions do not compare. Comparing a greedy 110 tokens/s with a sampled 84 meant chasing a nonexistent regression for several rounds, until the cause turned out to be a changed default temperature 😂

### 3.5 Slow the first time, slower the second, fast after that

A recording is only valid for the shape it saw, and question lengths vary: 24 real requests had 14 lengths. The old scheme rounded lengths up to a multiple of 32; a new length ran directly the first time (which also raced the implementations) and was recorded the second. So a new length was slow once, slower once more, then fast.

Replay is issued from JS one command at a time, not as an indivisible GPU command buffer, so each replay can recompute every command's workgroup count from the rows it actually has.

Recordings are now made per capacity: 128, 256 and 512 rows, recorded at load. Each row-wise kernel registers at record time how its workgroup count follows the row count: a matmul needs rows/32 rounded up, an elementwise op rows × width / 64, attention follows the longest sequence, the number of questions and the number of heads. A replay passes in the live row count, question count and longest length; each command launches only the workgroups it needs, never more than recorded.

Rows past the live ones hold a previous call's data, but every kernel computes a row from that row alone and attention reads only its own segment, so they are harmless. Every recording is also replayed once at a row count that is not a whole tile and compared bit for bit with the recording run; a mismatch fails the load, not an answer.

On the Q8 decision model in real use, the first two requests went from 70 and 78 ms to 23 and 29 ms, and no request records anything any more.

### 3.6 Measuring wrong

"Race and keep the fastest" assumes the measurement is right. Ways it goes wrong:

- **Garbage collection inside the timing.** A matmul ended up on an implementation 30% slower. The fast kernel normally took 0.52 ms, but two samples took 1.5 ms, each with a Python garbage collection plus a GPU buffer reap inside. The paired comparison came out 7 to 2, not significant, so the rule fell back to the default: the slow one. Collection is now paused during a timed sample and done between samples.
- **The browser's clock.** `performance.now()` is deliberately coarsened and jittered, and it includes the host issuing commands and waiting for the readback. Where the device allows it, timing uses GPU timestamps: a sample's commands go in without an early submit, and the compute pass durations are summed. That is pure GPU time.
- **Coarse GPU timestamps.** In Chrome without developer flags, timestamps come in 65.5 µs steps; a 2 µs kernel reads as 0 or 65.5. The test scripts launch Chrome with that flag and get nanoseconds, so the test environment is far more precise than an ordinary browser, and parameters tuned there are wrong in an ordinary browser. The runs per sample are not fixed: they are sized to about thirty timestamp steps, with the step read off the timestamps (the largest power of two dividing all of them).
- **A sleeping GPU needs a warm-up.** A race that starts from idle spends its first rounds on the clock ramp, so the two samples of a pair run at different frequencies. The race now runs warm-up rounds until one is no faster than the last, then times. Right after another race the GPU is already hot, and the warm-up is skipped.
- **An accuracy gate passed by luck.** The half-precision tiled kernel sums each run of 32 products in half precision, so its error depends on the input: on one 6144×5120 Q4_K weight of the 27B, 0.3% to 1.6% of the output scale from one input to the next, against a 1% bound. The load's random input passed; a remeasure's failed. An approximate route is now checked once more per weight on two fixed hard inputs (a few large values among small ones, a few very large ones among ordinary ones) and races only if it holds there. Summing in single precision brings the error down to 0.02% to 0.16% but gives back most of the speed, so the half sum stays, on the weights that pass.
- **In-session A/B tests can point the wrong way.** Changing a constant inside a live session showed a 31% speedup; a clean reload and the same measurement showed a 5% slowdown. The only protocol that counts: change the code, reload, load the model, measure.

### 3.7 Different GPUs, different capabilities

Checking the 178 kernels the three LLMs register against what the WebGPU spec guarantees found 36 that a minimum-spec device cannot run:

- twelve decode kernels used 17.4 KB of workgroup memory, where 16 KB is guaranteed;
- one bound nine storage buffers, where eight are guaranteed;
- the packed dot-product kernels need a specific WGSL language feature.

The M5 supports all of it, so none of this ever shows up there; on a GPU with only the spec minimum, the model fails to load. Every kernel is now checked at registration (features, language features, buffer count, workgroup memory) against what the device reports, or the spec minimum where it reports nothing. Kernels that cannot run are left out of the race instead of failing to compile later.

### 3.8 Errors that don't error

- **Shader compilation failures that don't fail.** Pipelines were created without checking compilation results, so a broken shader was silently skipped: the output buffer stayed untouched and read back as zeros, indistinguishable from a kernel that ran correctly on all-zero weights. One kernel stayed disabled for a long time because of a literal, `-3.4028235e38`, which is outside the 32-bit float range, so WGSL rejects it. Compilation results are now always checked.
- **Reusing the wrong conversation cache.** In a page left open for a long time, the model started repeating the prompt followed by garbage; a fresh page with the same code and model was fine. New rows were written into the KV cache during generation, but the record of which tokens the cache holds was updated only on normal completion. A stop, an error, or a stream the page stopped reading left the record on the previous turn, so the next turn reused rows from a different conversation. If the forward pass is correct and the output is garbage, clear the conversation cache first.
- **Memory ballooning.** The whole machine stuttered while the 27B generated: a 9.8 GB model held 36.9 GB across seventy-odd thousand buffers. Garbage in reference cycles was not collected in time, and the OS compressed and decompressed pages nonstop. Now there is budgeted incremental collection on the allocation path and a full collection at phase boundaries.

### 3.9 Remeasure on demand

Devices change: drivers update, laptops heat up and throttle, and measurements can be wrong. The `remeasure` API races the loaded model's choices again, at any time:

- **Only what the loaded model uses:** what its load, recordings and requests looked up, not everything the SDK supports.
- **The incumbent stays unless beaten.** A challenger replaces the choice in use only when paired measurements prove it faster; an inconclusive race changes nothing. The first version fell back to the static default whenever a race was inconclusive, replacing a proven choice with no evidence and rebuilding every recording that used it.
- **One change at a time, not every combination.** The decode-level composition used to be searched as the full product of every dimension, 147 combinations in one round, so it could only run offline. Now the search starts from the plan in use and changes one value of one dimension at a time. Each challenger must pass the accuracy check, then race the incumbent on separately recorded graphs replayed alternately, and is adopted only when proven faster. On the 30B one remeasure tried 18 challengers, adopted one, and took about 35 seconds.
- **Stoppable without losing work.** Each verdict takes effect when its race ends. A stop drops only the race in progress; everything already raced keeps its result. The time budget (60 s by default) stops it the same way, and what it did not reach continues when the page is idle. Each remeasure starts with the choices raced longest ago, and that order is saved with the device profile, so a budget that always runs out at the same point never strands the same choices.
- **Recordings that used a changed choice are rebuilt automatically.**
- **Two sets of routes, switchable.** The set a remeasure replaced is kept. When each of the last three replies with the set in use is slower than each of the other set's last three, a callback says so; switching is the caller's decision, and it can go back and forth. When the next remeasure changes routes, the set not in use is the one dropped.

After one remeasure, the 0.6B dropped from 175 to 152 tokens a second, and restoring all 20 changed choices did not bring it back, so the problem was probably not the choices. The cause: in interactive mode the decode-level composition does not search, it applies an exact reference plan, and that plan's token-picking route, combined with a pick mode derived from the last call's sampling settings, switched the greedy decode pipeline off. Every token went back to reading 600 KB of logits.

If reverting every change does not bring it back, it is not the changes. It is state.

The 27B once went from 7 to 4 tokens a second after a remeasure, and switching back to the old routes did not help either. A fixed kernel timed by the GPU's own timestamps told why: 1.06 ms cool, 1.48 to 1.80 ms right after the remeasure. The same kernel took 1.6 times as long: the chip had clocked down. A minute of ordinary replies without any remeasure did the same, a minute idle brought it back, and paging over that time was negligible. Run alternately in the same stretch of time, the old and new sets both decoded at 7.92 tokens a second. So speeds after a remeasure count only once the machine has cooled, and the only test of which set is faster is running them alternately.

After the fix, the 0.6B decodes at the same speed before and after a remeasure. Timing small kernels with the GPU's clock also cut load time: the 27B from 37 s to 31 s, the 30B from 41 s to 39 s, with byte-identical output.

---

## 4. Takeaways

- **Find what is actually holding things back before changing anything.** Of the decision-model optimizations that "had to help", half did nothing and half made it slower; the one that worked reduced nothing.
- **Align conditions before comparing.** Two numbers from different conditions often point the wrong way.
- **Verify the final output.** Read the answer, listen to the audio; an intermediate metric is not enough. Between "the forward pass matches" and "the answer is right" sit cache reuse, sampling state and recorded command sequences.
- **An error beats a silent fallback.** Drop what the device cannot run, fail on compile errors, say why something cannot be measured. A silent fallback only hides the problem.
- **Accept the platform's ceiling.** WebGPU cannot use Apple's matrix units; measure the gap and move on. Taking the first request from 80 ms to the low twenties matters more than peak FLOPS.

---

Try it (the first visit downloads the model; after that it loads from the browser cache): https://xnetsc.github.io/webpytorch/chat/

Different hardware gives different numbers; the reasoning holds.
