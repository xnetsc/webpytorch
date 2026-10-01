# Current task status

> Last updated: 2026-10-01

## One-line status

**Stored-format correctness, xDecision Q8 browser inference, model-source selection, exact
load progress, and decision-output correctness are verified. Native-width hardware
optimisation is incomplete, so cross-width performance routing is disabled by default and
its earlier benchmark conclusions are provisional.**

## Latest round — stored-weight execution and progress correctness

- All loader-supported GGML formats have native stored compute coverage; GPTQ INT4/INT8 uses
  the same model-agnostic interface and policy.
- Stored execution is the default. Stored/materialized auto-selection remains an explicit
  diagnostic opt-in until each original-width path has received its applicable hardware
  optimisation and passed numerical gates.
- Phase one never changes operand width as an optimisation. Q8_0 now vectorises reads and
  multiply-accumulate directly from its original signed-byte blocks, without requantising
  FP32 activations; packed INT8 dot with activation quantisation is deferred to phase two.
- Browser verification after the exact Q8_0 vectorisation: xDecision returned the same
  non-uniform result on the duplicate-charge example — billing 89.7%, true 97.9%, urgency
  15.7% / 26.6% / 27.3% / 30.4% — in 304 ms for 486 input tokens.
- The browser correctness gate covers 140 stored GGML operator cases plus the device-side
  decision-feature reduction.
- Automated verification: 45 Python tests and 19 JavaScript tests passed.
- The SDK cache token now covers the Python package and both worker bootstraps, not only the
  main-thread JavaScript file.
- Quantized ONNX operator support remains a separate future task.
