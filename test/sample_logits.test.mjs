import assert from 'node:assert/strict';
import test from 'node:test';
import { sampleLogits } from '../src/sampleLogits.ts';

test('greedy token choice, penalties and EOS masking happen in JS', () => {
  assert.equal(sampleLogits(Float32Array.of(1, 3, 2), { doSample: false }), 1);
  assert.equal(sampleLogits(Float32Array.of(1, 3, 2), {
    doSample: false, seen: [1, 1], repetitionPenalty: 2,
  }), 2);
  assert.equal(sampleLogits(Float32Array.of(1, 3, 2), {
    doSample: false, blockEos: true, eosIds: [1],
  }), 2);
  assert.equal(sampleLogits(Float32Array.of(1, 3, 2), {
    doSample: false, seen: [1, 1], frequencyPenalty: 1,
  }), 2);
});

test('nucleus and top-k sampling operate on the JS readback in place', () => {
  const top = Float32Array.of(0, 1, 2);
  assert.equal(sampleLogits(top, {
    doSample: true, topK: 1, topP: 1, random: 0.5,
  }), 2);
  assert.ok(top[0] < top[1] && top[1] < top[2], 'logits were converted to weights in place');
  assert.equal(sampleLogits(Float32Array.of(0, 1, 2), {
    doSample: true, topK: 0, topP: 1, random: 0,
  }), 0);
  assert.equal(sampleLogits(Float32Array.of(0, 1, 2), {
    doSample: true, topK: 0, topP: 1, random: 0.99,
  }), 2);
  assert.equal(sampleLogits(Float32Array.of(0, 1, 2), {
    doSample: true, topK: 0, topP: 0.1, random: 0.99,
  }), 2);
});

test('incrementally retained JS token counts match rebuilding history each step', () => {
  const history = [1, 1, 2];
  const counts = new Map([[1, 2], [2, 1]]);
  for (let step = 0; step < 20; step++) {
    const source = Float32Array.of(0.5, 2.2, 1.8, 1.4);
    const options = { doSample: false, repetitionPenalty: 1.1,
      presencePenalty: 0.2, frequencyPenalty: 0.15 };
    const expected = sampleLogits(source.slice(), { ...options, seen: history });
    const actual = sampleLogits(source.slice(), { ...options, seenCounts: counts });
    assert.equal(actual, expected);
    history.push(actual);
    counts.set(actual, (counts.get(actual) || 0) + 1);
  }
});
