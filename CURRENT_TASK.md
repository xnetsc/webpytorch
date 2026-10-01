# Current task status

> Last updated: 2026-10-01

## One-line status

**Native stored-format GGML and GPTQ execution, measured routing, xDecision Q8 browser
inference, model-source selection, exact load progress, and decision-output correctness are
implemented and verified.**

## Latest round — stored-weight execution and progress correctness

- All loader-supported GGML formats have native stored compute coverage; GPTQ INT4/INT8 uses
  the same model-agnostic interface and policy.
- Stored/materialized selection keys only on operator family, storage format, tensor shape,
  row bucket, and the active device's measurements.
- Browser verification: xDecision Q8 now returns non-uniform, input-dependent distributions;
  the duplicate-charge example scored billing 89.7% and true 97.9% in 595 ms.
- The browser correctness gate covers 140 stored GGML operator cases plus the device-side
  decision-feature reduction.
- Automated verification: 44 Python tests and 19 JavaScript tests passed.
- The SDK cache token now covers the Python package and both worker bootstraps, not only the
  main-thread JavaScript file.
- Quantized ONNX operator support remains a separate future task.
