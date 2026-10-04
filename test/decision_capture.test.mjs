import assert from 'node:assert/strict';
import test from 'node:test';
import { fillDecisionEmbeddings, fillDecisionMask, fillDecisionKeyMask,
  fillDecisionKeyMaskCpu, stageDecisionCapture, stageDecisionKeyMask } from '../src/decisionCapture.ts';

const view = a => new DataView(a.buffer, a.byteOffset, a.byteLength);

test('JS capture staging rewrites different question token rows in place', () => {
  // [1, 2], [3, 4], [5, 6], [7, 8] encoded as binary16.
  const table = Uint16Array.of(0x3c00, 0x4000, 0x4200, 0x4400,
    0x4500, 0x4600, 0x4700, 0x4800);
  const out = new Float32Array(2 * 4 * 2);
  const first = BigInt64Array.of(1n, 2n, 3n, 2n, 1n, 3n);
  fillDecisionEmbeddings(out, view(table), 'f16', view(first), 8, 2, 3, 4, 2, 4, 0);
  assert.deepEqual([...out], [3, 4, 5, 6, 7, 8, 1, 2,
    5, 6, 3, 4, 7, 8, 1, 2]);
  const next = BigInt64Array.of(2n, 1n, 3n, 1n, 2n, 1n);
  fillDecisionEmbeddings(out, view(table), 'f16', view(next), 8, 2, 3, 4, 2, 4, 0);
  assert.deepEqual([...out.slice(0, 8)], [5, 6, 3, 4, 7, 8, 1, 2]);
});

test('FP16 byte-view lookup preserves finite, subnormal and special values', () => {
  const bits = Uint16Array.of(0x0000, 0x8000, 0x0001, 0x03ff, 0x0400,
    0x3c00, 0xc000, 0x7bff, 0x7c00, 0xfc00, 0x7e00);
  const out = new Float32Array(bits.length);
  const ids = BigInt64Array.of(0n);
  fillDecisionEmbeddings(out, view(bits), 'f16', view(ids), 8,
    1, 1, 1, bits.length, 1, 0);
  assert.deepEqual([...out.slice(0, 8)], [0, -0, 2 ** -24, 1023 * 2 ** -24,
    2 ** -14, 1, -2, 65504]);
  assert.equal(out[8], Infinity);
  assert.equal(out[9], -Infinity);
  assert.ok(Number.isNaN(out[10]));
});

test('JS capture mask isolates independent rows, padding and sliding windows', () => {
  const valid = Int32Array.of(1, 1, 0, 1, 1, 1);
  const out = new Float32Array(2 * 2 * 4 * 4);
  fillDecisionMask(out, view(valid), 4, 2, 3, 4, 2, 'full_attention', 0);
  assert.deepEqual([...out.slice(0, 4)], [0, 0, -1e9, -1e9]);
  assert.deepEqual([...out.slice(2 * 16, 2 * 16 + 4)], [0, 0, 0, -1e9]);
  fillDecisionMask(out, view(valid), 4, 2, 3, 4, 2, 'sliding_attention', 1);
  assert.deepEqual([...out.slice(2 * 4, 2 * 4 + 4)], [-1e9, 0, -1e9, -1e9]);
  assert.deepEqual([...out.slice(2 * 16 + 2 * 4, 2 * 16 + 2 * 4 + 4)],
    [-1e9, 0, 0, -1e9]);
});

test('JS key mask broadcasts each question padding across every query and head', () => {
  const lengths = Int32Array.of(2, 4);
  const out = new Float32Array(2 * 3 * 4);
  fillDecisionKeyMask(out, view(lengths), 4, 2, 3, 4);
  for (let h = 0; h < 3; h++) {
    assert.deepEqual([...out.slice(h * 4, h * 4 + 4)], [0, 0, -1e9, -1e9]);
    assert.deepEqual([...out.slice((3 + h) * 4, (4 + h) * 4)], [0, 0, 0, 0]);
  }
  assert.throws(() => fillDecisionKeyMask(out, view(Int32Array.of(5, 4)), 4,
    2, 3, 4), /outside/);
});

test('WebGPU and WebGL stage only the broadcast key mask with signal-only uploads', () => {
  for (const backend of ['gpu', 'gl']) {
    const released = [];
    const lengths = Int32Array.of(2, 4);
    const source = new Uint8Array(lengths.buffer);
    const proxy = {
      getBuffer: () => ({ data: source, release: () => released.push('view') }),
      destroy: () => released.push('proxy'),
    };
    const arena = new Uint8Array(2 * 3 * 4 * 4);
    const uploads = [];
    const uploader = {
      prepare: bytes => { assert.equal(bytes, arena.byteLength); return arena; },
      uploadPrepared: (...args) => { uploads.push(args); return 1; },
      releasePrepared: () => released.push('arena'),
    };
    let flushes = 0;
    stageDecisionKeyMask(backend, () => { flushes++; }, uploader,
      17, proxy, 2, 3, 4);
    assert.equal(flushes, 1);
    assert.deepEqual([...new Float32Array(arena.buffer).slice(0, 4)],
      [0, 0, -1e9, -1e9]);
    assert.deepEqual(uploads, [[17, 0, arena.byteLength,
      backend === 'gl' ? 'Float32Array' : undefined]]);
    assert.deepEqual(released, ['arena', 'view', 'proxy']);
  }
});

test('browser CPU writes the key mask into its borrowed bytes without copying', () => {
  const target = new Float32Array(2 * 2 * 3);
  const lengths = Int32Array.of(1, 3);
  const released = [];
  const proxy = (name, typed) => ({
    getBuffer: () => ({ data: new Uint8Array(typed.buffer),
      release: () => released.push(`${name} view`) }),
    destroy: () => released.push(`${name} proxy`),
  });
  fillDecisionKeyMaskCpu(proxy('target', target), proxy('lengths', lengths), 2, 2, 3);
  assert.deepEqual([...target], [0, -1e9, -1e9, 0, -1e9, -1e9,
    0, 0, 0, 0, 0, 0]);
  assert.deepEqual(released, ['target view', 'lengths view',
    'target proxy', 'lengths proxy']);
});

test('one JS call stages both fresh questions and uploads every buffer, then releases views', () => {
  const released = [];
  const proxy = (name, data) => ({
    getBuffer: () => ({ data, release: () => released.push(`${name} view`) }),
    destroy: () => released.push(`${name} proxy`),
  });
  // The bridge lends only raw byte views; DataView interprets their layout without
  // allocating another typed copy, including the FP16 table and int64 token IDs.
  const bytes = typed => new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  const ids = proxy('ids', bytes(BigInt64Array.of(1n, 2n, 2n, 1n)));
  const valid = proxy('valid', bytes(BigInt64Array.of(1n, 1n, 1n, 1n)));
  const table = proxy('table', bytes(Uint16Array.of(0, 0x4200, 0x4700)));
  const arena = new Uint8Array(4 * (2 * 2 + 2 * 1 * 2 * 2));
  const uploads = [];
  const uploader = {
    prepare: size => { assert.equal(size, arena.byteLength); return arena; },
    uploadPrepared: (id, offset, bytes, ctor) => {
      uploads.push({ id, offset, bytes, ctor }); return 1;
    },
    releasePrepared: () => released.push('arena'),
  };
  let flushed = 0;
  stageDecisionCapture('gpu', () => { flushed++; }, uploader, 11,
    { full_attention: 12 }, ids, valid, table, 'f16', 2, 2, 2, 1, 3, 0, 1, 0);
  assert.equal(flushed, 1);
  assert.deepEqual([...new Float32Array(arena.buffer, 0, 4)], [3, 7, 7, 3]);
  assert.deepEqual(uploads, [
    { id: 11, offset: 0, bytes: 16, ctor: undefined },
    { id: 12, offset: 16, bytes: 32, ctor: undefined },
  ]);
  assert.deepEqual(released, ['arena', 'ids view', 'valid view', 'table view',
    'ids proxy', 'valid proxy', 'table proxy']);
});

test('failed JS capture staging releases every borrowed WASM view', () => {
  const released = [];
  const proxy = (name, data) => ({
    getBuffer: () => ({ data, release: () => released.push(`${name} view`) }),
    destroy: () => released.push(`${name} proxy`),
  });
  const ids = proxy('ids', Int32Array.of(99, 1));
  const valid = proxy('valid', Int32Array.of(1, 1));
  const table = proxy('table', Float32Array.of(0, 1, 2));
  const uploader = {
    prepare: bytes => new Uint8Array(bytes),
    uploadPrepared: () => { throw new Error('must not upload invalid token'); },
    releasePrepared: () => released.push('arena'),
  };
  assert.throws(() => stageDecisionCapture('gl', () => {}, uploader, 1, {},
    ids, valid, table, 'f32', 1, 2, 2, 1, 3, 0, 1, 0), /outside vocabulary/);
  assert.deepEqual(released, ['arena', 'ids view', 'valid view', 'table view',
    'ids proxy', 'valid proxy', 'table proxy']);
});
