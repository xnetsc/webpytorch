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
| KV cache | optional F16 in-place scatter | growing texture cache | attention cache semantics |
| generation/chat | captured decode replay | eager decode | `generate` / `chat` API |
| GDN linear attention | scan when available, otherwise step | step | recurrent layer output/state |
| Conv2d/Conv3d and training | WGSL forward/backward | GLSL forward/backward | operator output/gradient |
| decision features | WGSL device reduction | tensor reduction | `decision_features` result |
| ONNX | ORT WebGPU when the graph supports it | ORT WASM fallback | `OnnxModel` API |
| model I/O, cache, source choice, progress | shared transport | shared transport | public SDK events/state |

Hardware-only candidates do not become public gaps.  For example, WebGL has no compute
storage packing or packed INT8 dot instruction corresponding to WebGPU DP4A.  That candidate
therefore converges at `QuantizedLinear.forward`: WebGPU rejects the measured-negative DP4A
route and WebGL explicitly selects its exact packed GLSL route; both expose the same input,
output, errors, model format, and application behavior.

Optimisation scope is resolved from broad to narrow, never as one global all-or-nothing
vote.  A candidate that wins every valid case is installed globally.  Otherwise its route
narrows successively by backend, stored format, operator mode, shape bucket and device
profile, keeping every correctness-gated local win.  A candidate is rejected only in the
buckets where it has no end-to-end speed/memory benefit; failure to win globally is not a
reason to erase a local win.

Acceptance requires code generation plus browser execution on both backends.  The GGML gate
runs 28 formats across GEMV, two-row verification, GEMM, MoE-GEMV and MoE-GEMM (140 labelled
cases) per backend; WebGL's GEMM cases additionally execute both its small- and large-batch
production variants.  GPTQ INT4/INT8 is checked against an independent NumPy dequant
reference on both backends.  Same-width performance selection is also measured independently
per backend.
