/** Select one token without moving a vocabulary-sized logit vector into Python.
 * The input is a mutable JS view of a GPU readback and is reused in place.
 */
export interface SamplingOptions {
  doSample: boolean;
  execution?: 'auto' | 'js' | 'gpu';
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  random?: number;
  repetitionPenalty?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seen?: number[];
  seenCounts?: Map<number, number>;
  eosIds?: number[];
  blockEos?: boolean;
}

function topIndices(values: Float32Array, count: number): number[] {
  const heap: number[] = [];
  const worse = (a: number, b: number) =>
    values[a] < values[b] || (values[a] === values[b] && a > b);
  for (let i = 0; i < values.length; i++) {
    if (heap.length < count) {
      heap.push(i);
      let j = heap.length - 1;
      while (j > 0) {
        const p = (j - 1) >> 1;
        if (!worse(heap[j], heap[p])) break;
        [heap[j], heap[p]] = [heap[p], heap[j]];
        j = p;
      }
    } else if (worse(heap[0], i)) {
      heap[0] = i;
      for (let j = 0;;) {
        const l = 2 * j + 1;
        if (l >= heap.length) break;
        const r = l + 1;
        const c = r < heap.length && worse(heap[r], heap[l]) ? r : l;
        if (!worse(heap[c], heap[j])) break;
        [heap[c], heap[j]] = [heap[j], heap[c]];
        j = c;
      }
    }
  }
  heap.sort((a, b) => values[b] - values[a] || a - b);
  return heap;
}

export function sampleLogits(logits: Float32Array, options: SamplingOptions): number {
  const n = logits.length;
  if (!n) throw new Error('cannot sample an empty vocabulary');
  const counts = options.seenCounts || new Map<number, number>();
  if (!options.seenCounts) {
    for (const id of options.seen || []) {
      if (Number.isInteger(id) && id >= 0 && id < n) counts.set(id, (counts.get(id) || 0) + 1);
    }
  }
  const rp = options.repetitionPenalty || 1;
  const pp = options.presencePenalty || 0;
  const fp = options.frequencyPenalty || 0;
  if (rp !== 1 || pp || fp) {
    for (const [id, count] of counts) {
      let value = logits[id];
      if (rp !== 1) value = value > 0 ? value / rp : value * rp;
      logits[id] = value - pp - fp * count;
    }
  }
  if (options.blockEos) {
    for (const id of options.eosIds || []) if (id >= 0 && id < n) logits[id] = -Infinity;
  }

  let best = 0;
  for (let i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
  if (!options.doSample) return best;

  const max = logits[best];
  if (!Number.isFinite(max)) throw new Error('all sampled logits are non-finite');
  const minP = Math.max(0, options.minP || 0);
  const cutoff = minP > 0 ? max + Math.log(Math.max(minP, 1e-9)) : -Infinity;
  const temperature = Math.max(options.temperature || 1, 1e-5);
  const peak = max / temperature;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const value = logits[i] < cutoff ? 0 : Math.fround(Math.exp(logits[i] / temperature - peak));
    logits[i] = value;
    total += value;
  }
  if (!(total > 0)) throw new Error('sampled logits have no probability mass');
  const random = Math.min(1 - Number.EPSILON, Math.max(0, options.random || 0));
  const topP = Math.min(1, Math.max(0, options.topP ?? 1));
  const k = options.topK && options.topK > 0 ? Math.min(options.topK, n) : n;
  if (topP >= 1 && k >= n) {
    const target = random * total;
    let mass = 0;
    for (let i = 0; i < n; i++) {
      mass += logits[i];
      if (mass >= target) return i;
    }
    return n - 1;
  }

  let order = topIndices(logits, Math.min(k < n ? k : 256, n));
  let headMass = 0;
  for (const id of order) headMass += logits[id];
  if (k === n && headMass / total < topP && order.length < n) {
    order = [...Array(n).keys()].sort((a, b) => logits[b] - logits[a] || a - b);
  }
  let kept = 0;
  let keepMass = 0;
  for (const id of order) {
    keepMass += logits[id];
    kept++;
    if (kept >= k || keepMass / total >= topP) break;
  }
  const target = random * keepMass;
  let mass = 0;
  for (let j = 0; j < kept; j++) {
    mass += logits[order[j]];
    if (mass >= target) return order[j];
  }
  return order[kept - 1];
}
