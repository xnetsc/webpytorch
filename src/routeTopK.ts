/** CPU-side MoE routing on JS views, with only small selected-index/weight
 * buffers returned to the GPU. Python chooses the route but never touches the
 * router logits or assembles its output data.
 */
export function routeTopK(logits: Float32Array, rows: number, experts: number,
                          k: number, renormalize: boolean) {
  if (rows < 1 || experts < 1 || k < 1 || k > experts || logits.length < rows * experts) {
    throw new Error('invalid MoE router shape');
  }
  const indices = new Int32Array(rows * k);
  const weights = new Float32Array(rows * k);
  routeTopKInto(logits, rows, experts, k, renormalize, indices, weights);
  return { indices, weights };
}

export function routeTopKInto(logits: Float32Array, rows: number, experts: number,
                              k: number, renormalize: boolean,
                              indices: Int32Array, weights: Float32Array): void {
  if (rows < 1 || experts < 1 || k < 1 || k > experts || logits.length < rows * experts ||
      indices.length < rows * k || weights.length < rows * k) {
    throw new Error('invalid MoE router shape');
  }
  const best = new Int32Array(k);
  const bestMass = new Float64Array(k);
  for (let row = 0; row < rows; row++) {
    const start = row * experts;
    let max = -Infinity;
    for (let e = 0; e < experts; e++) max = Math.max(max, logits[start + e]);
    if (!Number.isFinite(max)) throw new Error('MoE router logits are non-finite');
    let allMass = 0;
    let bestCount = 0;
    for (let e = 0; e < experts; e++) {
      const value = logits[start + e];
      if (!Number.isFinite(value)) throw new Error('MoE router logits are non-finite');
      allMass += Math.exp(value - max);
      let pos = bestCount;
      while (pos > 0 && value > logits[start + best[pos - 1]]) pos--;
      if (pos < k) {
        for (let j = Math.min(bestCount, k - 1); j > pos; j--) best[j] = best[j - 1];
        best[pos] = e;
        if (bestCount < k) bestCount++;
      }
    }
    let selectedMass = 0;
    for (let j = 0; j < k; j++) {
      bestMass[j] = Math.exp(logits[start + best[j]] - max);
      selectedMass += bestMass[j];
    }
    const divisor = renormalize ? selectedMass : allMass;
    for (let j = 0; j < k; j++) {
      indices[row * k + j] = best[j];
      weights[row * k + j] = Math.fround(bestMass[j] / divisor);
    }
  }
}
