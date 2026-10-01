# Progress

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
