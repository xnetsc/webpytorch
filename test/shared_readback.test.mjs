import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function loadTypeScript(path, imports) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: name => {
      assert.ok(name in imports, `unexpected import ${name}`);
      return imports[name];
    },
    Uint8Array, Uint16Array, Int32Array, Float32Array, SharedArrayBuffer,
    Atomics, console,
  });
  return module.exports;
}

test('WebGL readback passes shared memory to texture and wakes the worker', async () => {
  const { ComputeContextGL } = await loadTypeScript(
    '../src/webgl/webglComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webglContext': {},
    },
  );
  const context = new ComputeContextGL();
  const data = new SharedArrayBuffer(16);
  const notify = new SharedArrayBuffer(4);
  let targetWasShared = false;
  context.tensorBuffers.set(1, {
    getDataRaw(target) {
      targetWasShared = target.buffer === data;
      target.set([1, 2, 3, 4]);
      return { type: 'Float32Array', buffer: target.subarray(0, 4) };
    },
  });
  context.handleMessage({ method: 'gl.getData', id: 1, data, notify, ctorType: 'Float32Array' });
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(targetWasShared, true);
  assert.deepEqual(Array.from(new Float32Array(data)), [1, 2, 3, 4]);
  assert.equal(new Int32Array(notify)[0], 1);
});

test('WebGPU readback writes into caller shared memory', async () => {
  const { ComputeContextGPU } = await loadTypeScript(
    '../src/webgpu/webgpuComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webgpuContext': {},
      './webgpuTensorBuffer': {},
      './vocabSampler': { GPUVocabSampler: class {} },
    },
  );
  const context = new ComputeContextGPU();
  const data = new SharedArrayBuffer(8);
  let targetWasShared = false;
  context.tensorBuffers.set(1, {
    bufferShape: { byteLength: 8 },
    getDataInto(target) {
      targetWasShared = target.buffer === data;
      target.set([1, 2, 3, 4, 5, 6, 7, 8]);
      return Promise.resolve();
    },
  });
  await context.getDataInto(1, data);
  assert.equal(targetWasShared, true);
  assert.deepEqual(Array.from(new Uint8Array(data)), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('synchronous WebGPU readback failure wakes the worker', async () => {
  const { ComputeContextGPU } = await loadTypeScript(
    '../src/webgpu/webgpuComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webgpuContext': {},
      './webgpuTensorBuffer': {},
      './vocabSampler': { GPUVocabSampler: class {} },
    },
  );
  const context = new ComputeContextGPU();
  const data = new SharedArrayBuffer(4);
  const notify = new SharedArrayBuffer(4);
  context.tensorBuffers.set(1, {
    bufferShape: { byteLength: 8 },
    getDataInto() { throw new Error('lost device'); },
  });
  const originalError = console.error;
  console.error = () => {};
  try {
    context.handleMessage({ method: 'gpu.getData', id: 1, data, notify });
    await new Promise(setImmediate);
    assert.equal(new Int32Array(notify)[0], -1);
  } finally {
    console.error = originalError;
  }
});

test('WebGPU meta buffer consumes shared upload bytes without a transferred array', async () => {
  const { ComputeContextGPU } = await loadTypeScript(
    '../src/webgpu/webgpuComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webgpuContext': {},
      './webgpuTensorBuffer': {},
      './vocabSampler': { GPUVocabSampler: class {} },
    },
  );
  const context = new ComputeContextGPU();
  const memory = new SharedArrayBuffer(16);
  new Uint8Array(memory).set([0, 0, 4, 2, 9]);
  const notify = new SharedArrayBuffer(4);
  let usedSharedView = false;
  context.createMetaBuffer = (id, length, data) => {
    assert.equal(id, 7);
    assert.equal(length, 3);
    usedSharedView = data.buffer === memory;
    assert.deepEqual([...data], [4, 2, 9]);
  };
  context.handleMessage({ method: 'gpu.uploadMemory', memory, notify });
  context.handleMessage({ method: 'gpu.sharedMetaBuffer', id: 7,
    byteOffset: 2, byteLength: 3 });
  assert.equal(usedSharedView, true);
  assert.equal(new Int32Array(notify)[0], 1);
});
