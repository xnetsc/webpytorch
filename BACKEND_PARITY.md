# WebGPU / WebGL parity contract

Backend parity is checked at the lowest *efficient* common layer, not mechanically at the
lowest implementable layer.  Every candidate first passes the same semantics/precision gate,
then competes on end-to-end latency, persistent and temporary memory, transfers/synchrony,
compilation and first-call cost.  If a lower-layer emulation loses overall, equivalence moves
up one layer at a time; the public API is the final mandatory boundary and may never be absent.

| Capability path | WebGPU path | WebGL path | Lowest common contract |
|---|---|---|---|
| `Tensor` / autograd | WGSL operators | GLSL operators | operator result and gradient |
| dense `Linear` | compute matmul, optional packed F16 fast path | fragment matmul | `Linear.forward` |
| GGML stored `Linear` (28 formats) | native packed WGSL decoders | native packed GLSL decoders | stored decoder arithmetic |
| GPTQ INT4/INT8 `Linear` | packed WGSL scalar/vec4 routes | packed GLSL scalar/vec4 routes | packed matmul operator |
| MoE | routed compute dispatch | routed fragment pass | routed `Linear.forward` |
| attention | flash/fused compute kernels | fused softmax plus fragment matmul | attention output |
| Q/K norm + rotary embedding | validated one-dispatch per-head layer path | equivalent norm + rotary composition | attention preprocessing layer |
| KV cache | measured separate/paired optional-F16 in-place scatter | growing texture cache behind the same paired-write layer interface | attention cache semantics |
| generation/chat | captured decode replay, with compatible full-attention graphs reusable across turns | eager decode | `generate` / `chat` API |
| GDN linear attention | scan when available, otherwise step | step | recurrent layer output/state |
| parallel projections | measured shared WGSL dispatch when it wins | measured equivalent projection operation | `parallel_linear` layer operation |
| parallel MLP activation | measured shared projection plus SwiGLU | measured combined packed fragment projection plus in-place SwiGLU | `parallel_swiglu` layer operation |
| Conv2d/Conv3d and training | WGSL forward/backward | GLSL forward/backward | operator output/gradient |
| decision features | WGSL device reduction | tensor reduction | `decision_features` result |
| ONNX | shared `OnnxModel` graph interpreter; ORT WebGPU for the optional vision adapter | shared `OnnxModel` graph interpreter; ORT WASM for the optional vision adapter | `OnnxModel` API |
| model I/O, cache, source choice, progress | shared transport | shared transport | public SDK events/state |
| tensor upload/readback | shared staging, direct shared readback, shape-measured upload route | shared staging, `readPixels` into shared memory, shape-measured upload route | buffer result and transfer completion/error |
| unconstrained token sampling and host MoE routing | JS operates on shared readback; selected outputs enter shared upload staging | same JS policy on WebGL readback and upload staging | selected token / expert indices and weights |

Hardware-only candidates do not become public gaps.  For example, WebGL has no compute
storage packing or packed INT8 dot instruction corresponding to WebGPU DP4A.  That candidate
therefore converges at `QuantizedLinear.forward`: WebGPU measures DP4A per format, shape and
device and WebGL explicitly selects its exact packed GLSL route; both expose the same input,
output, errors, model format, and application behavior.

The transfer and sampling rows describe implemented hot-path equivalence, not a
claim that Python is now only a scheduler: constrained sampling, model parsing
and some format conversions still run in Python. A native-file full-model
regression and per-layer/whole-API performance gate remain required before
calling this migration complete.

Cross-turn graph reuse is a WebGPU scheduling optimisation, not a different model
feature. It is valid only while the KV buffers, execution route, GPU split and
sampling-side device-pick choice remain compatible. Recurrent layers re-record
because prompt prefill can replace their device state. WebGL performs the same
generation semantics through its eager path; neither backend changes sampling
or the stored weight representation for this optimisation.

Quantized ONNX reaches parity at that API boundary: both backends preserve INT8/UINT8
operands and the ONNX-required INT32 accumulator for `MatMulInteger`, `ConvInteger`,
`QLinearMatMul` and `QLinearConv`.  This shared graph interpreter is distinct from the
optional vision-decision adapter, whose implementation really is ORT WebGPU versus ORT
WASM.  The contract does not claim that every ONNX operator is implemented.

Optimisation scope is resolved from broad to narrow, never as one global all-or-nothing
vote.  A candidate that wins every valid case is installed globally.  Otherwise its route
narrows successively by backend, stored format, operator mode, shape bucket and device
profile, keeping every correctness-gated local win.  A candidate is rejected only in the
buckets where it has no end-to-end speed/memory benefit; failure to win globally is not a
reason to erase a local win.  Selection has no minimum speedup percentage: interleaved paired
samples must establish that a candidate is repeatably faster.  An inconclusive result stays
on the earlier, lower-memory implementation; a stable small win is retained.

Acceptance requires code generation plus browser execution on both backends.  The GGML gate
runs 28 formats across GEMV, two-row verification, GEMM, MoE-GEMV and MoE-GEMM (140 labelled
cases) per backend; WebGL's GEMM cases additionally execute both its small- and large-batch
production variants.  GPTQ INT4/INT8 is checked against an independent NumPy dequant
reference on both backends.  Same-width performance selection is also measured independently
per backend.

WebGL does not inherit WebGPU's absolute tokens/second target. Its throughput
acceptance band is set from published **comparable** browser-inference workloads
and then checked against this project's same-device, same-model, same-format,
same-context runs. A ratio from a different model, batch size, quantisation or
hardware is context, not a pass mark; there is no universal WebGL/WebGPU
multiplier. Both backends must still take every repeatable positive optimisation
available at their respective lowest efficient layer. This throughput allowance
does not relax output semantics, model/type coverage, or the seconds-level load
and first-token latency requirements.
