import assert from 'node:assert/strict';
import test from 'node:test';
import { routeTopK, routeTopKInto } from '../src/routeTopK.ts';

test('JS MoE router selects top experts and renormalizes per row', () => {
  const logits = Float32Array.of(0, 2, 1, -1, 3, 1);
  const { indices, weights } = routeTopK(logits, 2, 3, 2, true);
  assert.deepEqual([...indices], [1, 2, 1, 2]);
  assert.ok(Math.abs(weights[0] + weights[1] - 1) < 1e-6);
  assert.ok(Math.abs(weights[2] + weights[3] - 1) < 1e-6);
  assert.ok(weights[0] > weights[1] && weights[2] > weights[3]);
  assert.deepEqual([...logits], [0, 2, 1, -1, 3, 1], 'GPU readback is not overwritten');
});

test('unnormalized weights retain full softmax mass for selected experts', () => {
  const { weights } = routeTopK(Float32Array.of(0, 1, 2), 1, 3, 1, false);
  assert.ok(Math.abs(weights[0] - Math.exp(2)/(1+Math.exp(1)+Math.exp(2))) < 1e-6);
});

test('MoE router writes directly into caller-provided shared output', () => {
  const memory = new SharedArrayBuffer(16);
  const indices = new Int32Array(memory, 0, 2);
  const weights = new Float32Array(memory, 8, 2);
  routeTopKInto(Float32Array.of(0, 2, 1), 1, 3, 2, true, indices, weights);
  assert.deepEqual([...indices], [1, 2]);
  assert.ok(Math.abs(weights[0] + weights[1] - 1) < 1e-6);
  assert.equal(indices.buffer, weights.buffer);
});

test('in-place routing preserves selection order and weights across rows and ties', () => {
  const rows = 37, experts = 8, k = 3;
  const logits = Float32Array.from({ length: rows * experts }, (_, i) =>
    (i * 19 % 11) - 5);
  for (const renormalize of [false, true]) {
    const got = routeTopK(logits, rows, experts, k, renormalize);
    for (let row = 0; row < rows; row++) {
      const start = row * experts;
      const selected = [...Array(experts).keys()]
        .sort((a, b) => logits[start + b] - logits[start + a] || a - b)
        .slice(0, k);
      const max = Math.max(...logits.subarray(start, start + experts));
      const mass = selected.map(e => Math.exp(logits[start + e] - max));
      const divisor = renormalize ? mass.reduce((a, b) => a + b, 0)
        : Array.from(logits.subarray(start, start + experts))
          .reduce((sum, value) => sum + Math.exp(value - max), 0);
      assert.deepEqual([...got.indices.subarray(row * k, (row + 1) * k)], selected);
      for (let j = 0; j < k; j++) {
        assert.ok(Math.abs(got.weights[row * k + j] - Math.fround(mass[j] / divisor)) < 1e-7);
      }
    }
  }
});
