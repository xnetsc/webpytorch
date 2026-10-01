import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

await import('../webtorch/js/decision-vision.js');
const { normalizeDecisionState } = globalThis.webtorch;
const { answer, safeTemperature, toInternal, questionTypes } =
  await import('../webtorch/js/decision-vision-runtime.js');

test('vision answers expose calibrated answer confidence and guard unsafe temperatures once', () => {
  const cfg = { temperature: [2, 1, 1], temperature_by_options: {} };
  const q = toInternal(cfg, { type: 'choice', instructions: 'pick', criteria: { a: 'A', b: 'B' } });
  const out = answer(cfg, q, [{ order: [0, 1], logits: [4, 0], actProb: 0.8 }]);
  const expected = 1 / (1 + Math.exp(-2));
  assert.equal(out.answer_confidence, Number(expected.toFixed(4)));
  assert.notEqual(out.answer_confidence, Number((1 / (1 + Math.exp(-1))).toFixed(4)));
  assert.equal(safeTemperature(0.1006), 0.5);
  assert.equal(safeTemperature(Number.NaN), 1);
});

test('an export that names its own question types is read, not assumed', () => {
  // What the exports in circulation say: how many types, never what they are.
  assert.deepEqual(questionTypes({ temperature: [1, 1, 1] }).names, ['choice', 'score', 'noul']);
  assert.deepEqual(questionTypes({ temperature: [1, 1] }).names, ['choice', 'score']);
  // A fourth type is named positionally and takes the general shape rather than the last
  // one's -- it is not a two-outcome question just because it is unrecognised.
  assert.equal(questionTypes({ temperature: [1, 1, 1, 1] }).shapes.type3, 'named');

  // And one that declares them needs nothing added to this runtime.
  const cfg = { temperature: [1, 1, 1], temperature_by_options: {},
                question_types: [{ name: 'route', shape: 'named' },
                                 { name: 'severity', shape: 'ordered' },
                                 { name: 'holds', shape: 'fixed' }] };
  assert.deepEqual(questionTypes(cfg).names, ['route', 'severity', 'holds']);

  const sev = toInternal(cfg, { type: 'severity', instructions: 'how bad', criteria: ['low', 'high'] });
  assert.equal(sev.shape, 'ordered');
  assert.equal(sev.index, 1);                       // the row of the type embedding it is
  const scored = answer(cfg, sev, [{ order: [0, 1], logits: [0, 2], actProb: 0.5 }]);
  assert.equal(scored.type, 'severity');            // answered under ITS name
  assert.ok(scored.score !== undefined);            // and in its SHAPE

  const holds = toInternal(cfg, { type: 'holds', instructions: 'is it so' });
  assert.equal(holds.shape, 'fixed');
  assert.ok(answer(cfg, holds, [{ order: [0, 1], logits: [0, 2], actProb: 0.5 }]).noul !== undefined);

  assert.throws(() => toInternal(cfg, { type: 'choice', instructions: 'x', criteria: ['a', 'b'] }),
                /must be one of route, severity, holds/);
});

test('plain text and JSON states retain their existing shape', () => {
  assert.deepEqual(normalizeDecisionState('market is open'), {
    state: 'market is open', images: [],
  });
  assert.deepEqual(normalizeDecisionState({ market: 'open' }), {
    state: { market: 'open' }, images: [],
  });
});

test('typed image and multimodal states use one base64 wire shape', () => {
  const image = { type: 'image', media_type: 'image/png', data: 'AA==' };
  assert.deepEqual(normalizeDecisionState(image), { state: '', images: [image] });
  assert.deepEqual(normalizeDecisionState({
    type: 'multimodal', text: 'read this chart', images: [image],
  }), { state: 'read this chart', images: [image] });
});

test('an image field on an ordinary JSON state is separated from text state', () => {
  const image = { type: 'image', media_type: 'image/jpeg', data: 'AA==' };
  assert.deepEqual(normalizeDecisionState({ market: 'BTC', image }), {
    state: { market: 'BTC' }, images: [image],
  });
});

test('remote image URLs and ambiguous objects are rejected', () => {
  assert.throws(() => normalizeDecisionState({ type: 'multimodal', image: {
    type: 'image_url', url: 'https://example.test/x.png',
  } }), /decision image/);
});

test('vision model bytes are delegated to the supplied io_read bridge', async () => {
  const originalWorker = globalThis.Worker;
  let instance;
  class WorkerStub {
    constructor() { this.listeners = {}; this.messages = []; instance = this; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    postMessage(message) {
      this.messages.push(message);
      if (message.type === 'load') {
        queueMicrotask(() => this.listeners.message({ data: {
          type: 'read', id: 7, name: 'org/model/vision.onnx', offset: 16, length: 4,
        } }));
      } else if (message.type === 'read-result') {
        queueMicrotask(() => this.listeners.message({ data: {
          type: 'loaded', source: 'test', backend: 'webgpu', variant: 'fp16',
          qtypes: [], maxLen: 32, headMaxLen: 8,
        } }));
      }
    }
    terminate() { this.terminated = true; }
  }
  globalThis.Worker = WorkerStub;
  const calls = [];
  try {
    const model = await globalThis.webtorch.loadVisionDecision({
      model: 'org/model', workerURL: 'worker.js',
      read: async (...args) => { calls.push(args); return Uint8Array.of(1, 2, 3, 4); },
    });
    assert.deepEqual(calls, [['org/model/vision.onnx', 16, 4]]);
    const reply = instance.messages.find(message => message.type === 'read-result');
    assert.deepEqual([...reply.bytes], [1, 2, 3, 4]);
    model.release();
    assert.equal(instance.terminated, true);
  } finally {
    globalThis.Worker = originalWorker;
  }
});

test('vision worker owns neither a model transport nor a persistent model cache', async () => {
  const source = await readFile(new URL('../webtorch/js/decision-vision-worker.js', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /caches\.open|Cache Storage/);
  assert.doesNotMatch(source, /fetch\s*\(/);
  assert.match(source, /send\("read"/);
});
