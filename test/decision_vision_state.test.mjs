import assert from 'node:assert/strict';
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
