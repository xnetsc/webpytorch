import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

await import('../webtorch/js/decision-vision.js');
const { normalizeDecisionState } = globalThis.webtorch;

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
