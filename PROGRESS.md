# Progress

## 2026-10-02 ▸ Local Qwen3 native-format end-to-end gate

- Corrected the distinction between browser-persistent model cache and repository-local
  model files: the cache settings page intentionally does not enumerate disk files.
- Added a repeatable browser smoke test for `models/Qwen3-0.6B-Q4_K_M.gguf`. It fetches the
  served local file with persistence disabled, explicitly requests native weights, checks
  the live model's packed Linear formats, and performs deterministic generation.
- Updated the old GGUF example and loader comments that still described every GGUF as being
  converted to INT4, which has not been true since stored-format kernels became the default.

### Evidence

- Chrome WebGPU: deterministic reply `OK`; 168 Q4_K + 28 Q6_K native Linear modules;
  load 1.2 s, TTFT 112 ms, captured decode enabled.
- Python suite: 49 passed. JavaScript suite: 19 passed.

## 2026-10-02 ▸ Phase-one exact Q5 vector kernels

- Added register-vector candidates that consume Q5_0/Q5_1's original four-bit nibble stream
  and separate fifth-bit plane directly. FP32 activations, block scales, minima, and output
  accumulation semantics are unchanged.
- Realistic K=4096, N=3072 interleaved runs showed Q5_0 at 1.00–1.13× and 1.01–1.11×
  across M=1/32/128, so its exact vector path is enabled globally.
- Q5_1 batch execution was consistently positive (M32 +7–8%, M128 +10–11%), while M1
  contradicted itself (+8%, then −12%). Production therefore keeps exact scalar GEMV/GEMV2
  and selects exact vector execution only for GEMM. This is same-width operator routing,
  not the deferred cross-width phase.

### Evidence

- Final Chrome WebGPU matrix: 140/140 stored-format operator cases; decision-feature CPU/GPU
  comparison passed.
- Python suite: 49 passed. JavaScript suite: 19 passed.

## 2026-10-02 ▸ Phase-one exact Q4 vector kernels

**Constraint:** Optimise within the source format first. No weight re-encoding, activation
requantisation, or materialised alternate-width comparison may decide this phase.

- Added register-local packed-nibble vector decoders for Q4_0, Q4_1, and Q4_K. Four source
  bytes are read together, split into their original low/high Q4 values, scaled with the
  format's exact block parameters, and accumulated against unchanged FP32 activations.
- Added an interleaved same-width scalar/vector benchmark. It labels itself phase one and
  cannot be used as evidence for cross-width routing.
- Rejected the first tiny-shape timing set: its 0.06–0.38 ms kernels were dominated by
  dispatch/reclamation noise and repeated runs contradicted one another.
- At realistic K=4096, N=3072 shapes, two independent interleaved runs showed Q4_0
  1.03–1.12× / 1.06–1.10×, Q4_1 1.02–1.09× / 1.03–1.10×, and Q4_K
  1.01–1.14× / 1.07–1.14× across M=1/32/128. Relative numerical differences stayed near
  1e-6 and came only from equivalent FP32 accumulation order.

### Evidence

- Final Chrome WebGPU matrix: 28 formats × 5 operator cases = 140/140, no failures; the
  decision-feature reduction also matched its CPU reference.
- Python suite: 47 passed. JavaScript suite: 19 passed.

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
