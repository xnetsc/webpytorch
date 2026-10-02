# Current task status

> Last updated: 2026-10-02

## Highest project principle

**Do not stop while either phase is incomplete.** A local commit, passing subset, status
report, individual-model result, or end of a conversation turn is not a completion point.
Phase one must finish correctness plus every applicable same-width optimisation and hardware
feature without converting the stored representation. Phase two must then finish the full
performance comparison and adopt whichever measured execution is fastest. Only both phases
passing their recorded acceptance gates completes this task.

## One-line status

**Both phases are complete on WebGPU and WebGL: original-width execution is correct and
shape/backend optimised, and the measured second-stage alternatives are routed only where
they beat the lower-memory stored path. The final browser and automated gates pass.**

## Completion evidence

- Phase one: all 28 accepted GGML formats pass stored-format GEMV, GEMV2, small/large GEMM,
  MoE decode and MoE batch checks on both backends; GPTQ INT4/INT8 matches an independent
  dequant reference. Same-width scalar/vec4 choices are independent by backend, format and
  batch mode, with no activation or weight width conversion.
- Phase two: WebGPU stored-versus-materialized routing is measured per format, shape bucket
  and device profile with a 5% margin for the lower-memory stored representation. The
  activation-INT8 DP4A candidate passes accuracy but loses every profitable bucket and is
  rejected. WebGL explicitly retains its exact packed path because it has no equivalent
  packing/dot or materializer primitive.
- Parity: optimisation scope descends global → backend → format → operator mode → shape
  bucket → device profile, so a global loss never erases a local win. If a low-level WebGL
  primitive is unavailable, equivalence moves upward to the nearest efficient contract.
- End to end: local Qwen3-0.6B Q4_K_M returns deterministic `OK` on WebGPU and WebGL while
  retaining 168 Q4_K plus 28 Q6_K linears. The independent Qwen2 reference matches on both
  backends (maximum logits error `4.84e-8`, identical greedy tokens).
- Broader WebGL API gate: Conv2d/BatchNorm/MaxPool/Linear autograd training converges from
  loss `0.614` to `0.0` at 100% accuracy.
- Automated suites: 66 Python tests and 19 JavaScript tests pass; both checked-in backend
  wheels are rebuilt from source and current.

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
- Quantized ONNX is handled as its own `OnnxModel`/ORT graph path (WebGPU when supported,
  WASM fallback under WebGL), not by pretending its graph tensors are GGML/GPTQ blocks.
