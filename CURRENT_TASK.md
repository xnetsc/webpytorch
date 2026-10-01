# Current task status

> Last updated: 2026-10-01

## One-line status

**Native stored-format GGML and GPTQ execution, measured routing, xDecision Q8 browser
inference, model-source selection, and exact load progress are implemented and verified.**

## Latest round — stored-weight execution and progress correctness

- All loader-supported GGML formats have native stored compute coverage; GPTQ INT4/INT8 uses
  the same model-agnostic interface and policy.
- Stored/materialized selection keys only on operator family, storage format, tensor shape,
  row bucket, and the active device's measurements.
- Browser verification: xDecision Q8 loaded as 402.5 MB and ran a decision request in 668 ms.
- Automated verification: 44 Python tests and 17 JavaScript tests passed.
- Quantized ONNX operator support remains a separate future task.
