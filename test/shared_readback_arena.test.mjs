import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function arena() {
  const source = await readFile(new URL('../src/sharedReadback.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports, SharedArrayBuffer, Int32Array, Atomics,
    Math, Number, Error, TextEncoder, TextDecoder,
  });
  return module.exports.sharedReadbackArena();
}

test('readback arena starts small, reuses memory, and rebinds after growth', async () => {
  const readback = await arena();
  const first = readback.begin(4);
  assert.equal(first.memory.byteLength, 65536);
  assert.equal(first.binding.data, first.memory);
  assert.equal(first.binding.notify.byteLength, 4);
  assert.equal(first.binding.error.byteLength, 4096);
  Atomics.store(first.status, 0, 1);
  const second = readback.begin(1024);
  assert.equal(second.memory, first.memory);
  assert.equal(second.status[0], 0);
  assert.equal(Object.keys(second.binding).length, 0);
  const grown = readback.begin(70000);
  assert.equal(grown.memory.byteLength, 131072);
  assert.notEqual(grown.memory, first.memory);
  assert.equal(grown.binding.data, grown.memory);
  assert.equal(grown.binding.notify, first.binding.notify);
  assert.equal(grown.binding.error, first.binding.error);
});

test('shared readback returns the real GPU failure without copying successful payloads', async () => {
  const readback = await arena();
  const first = readback.begin(4);
  const exports = await (async () => {
    const source = await readFile(new URL('../src/sharedReadback.ts', import.meta.url), 'utf8');
    const compiled = ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
    }).outputText;
    const module = { exports: {} };
    vm.runInNewContext(compiled, { module, exports: module.exports, TextEncoder, TextDecoder,
      Uint8Array, SharedArrayBuffer, Int32Array, Atomics, Math, Number, Error });
    return module.exports;
  })();
  exports.writeSharedReadbackError(first.binding.error, new Error('WGSL reserved identifier meta'));
  assert.equal(readback.errorMessage(), 'WGSL reserved identifier meta');
  readback.begin(4);
  assert.equal(readback.errorMessage(), '');
});

test('readback arena rejects invalid sizes', async () => {
  const readback = await arena();
  for (const size of [-1, NaN, Infinity, 1.5]) {
    assert.throws(() => readback.begin(size), /invalid shared readback length/);
  }
});
