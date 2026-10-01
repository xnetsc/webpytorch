# Progress

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
