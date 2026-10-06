import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function loadTypeScript(path, imports, globals = {}) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module,
    exports: module.exports,
    require: name => {
      if (name === '../sharedReadback') return { writeSharedReadbackError() {} };
      if (name === '../stagedRead') return { stagedReadTarget() { throw new Error('unused'); } };
      assert.ok(name in imports, `unexpected import ${name}`);
      return imports[name];
    },
    Uint8Array,
    Int32Array,
    SharedArrayBuffer,
    Atomics,
    console: { error() {} },
    ...globals,
  });
  return module.exports;
}

test('WebGPU upload copies every byte beyond the mapped-buffer limit', async () => {
  const limit = 32 * 1024 * 1024;
  const length = limit + 20;
  const destination = new Uint8Array(length);
  const uploads = [];
  const copies = [];
  const submitted = [];
  let flushes = 0;
  const device = {
    createBuffer(options) {
      if (!options.mappedAtCreation) {
        return { size: options.size, bytes: destination };
      }
      assert.ok(options.size <= limit, 'staging allocation exceeded the safe chunk');
      const buffer = {
        size: options.size,
        bytes: new Uint8Array(options.size),
        getMappedRange() { return this.bytes.buffer; },
        unmap() {},
        destroy() { this.destroyed = true; },
      };
      uploads.push(buffer);
      return buffer;
    },
    createCommandEncoder() {
      const pending = [];
      return {
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, byteLength) {
          const copy = { source, sourceOffset, target, targetOffset, byteLength };
          copies.push(copy);
          pending.push(copy);
        },
        finish() { return pending; },
      };
    },
    queue: {
      submit(commands) {
        submitted.push(...commands[0]);
      },
      onSubmittedWorkDone() {
        return Promise.resolve().then(() => {
          for (const copy of submitted.splice(0)) {
            assert.equal(copy.source.destroyed, undefined,
              'staging was destroyed before its GPU copy completed');
            copy.target.bytes.set(
              copy.source.bytes.subarray(copy.sourceOffset, copy.sourceOffset + copy.byteLength),
              copy.targetOffset,
            );
          }
        });
      },
    },
  };
  const ctx = { device, flush() { flushes++; } };
  const { WebGPUTensorBuffer } = await loadTypeScript(
    '../src/webgpu/webgpuTensorBuffer.ts',
    { './webgpuContext': { getNNWebGPUContext: () => ctx } },
    { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_WRITE: 8 } },
  );
  const data = new Uint8Array(length);
  for (let i = 0; i < length; i++) data[i] = (i * 37 + 11) & 255;
  const buffer = new WebGPUTensorBuffer({ byteLength: length }, false);
  const upload = buffer.setDataRaw(data);
  assert.ok(uploads.every(item => !item.destroyed), 'upload released in-flight staging');
  await upload;
  assert.equal(flushes, 1);
  assert.deepEqual(uploads.map(item => item.size), [limit, 20]);
  assert.deepEqual(copies.map(item => item.targetOffset), [0, limit]);
  assert.ok(uploads.every(item => item.destroyed));
  assert.ok(Buffer.from(destination).equals(Buffer.from(data)), 'upload changed weight bytes');
  assert.throws(() => buffer.setDataRaw(data.subarray(4)), /size mismatch/);
});

test('small WebGPU uploads preserve queue order without staging submissions', async () => {
  const steps = [];
  const dst = new Uint8Array(16);
  const device = {
    createBuffer(options) {
      assert.equal(options.mappedAtCreation, false);
      return { size: options.size, bytes: dst };
    },
    createCommandEncoder() { throw new Error('small upload created a copy command'); },
    queue: {
      writeBuffer(target, offset, data) {
        steps.push('write');
        target.bytes.set(data, offset);
      },
      submit() { throw new Error('small upload submitted a copy command'); },
    },
  };
  const ctx = { device, flush() { steps.push('flush'); } };
  const { WebGPUTensorBuffer } = await loadTypeScript(
    '../src/webgpu/webgpuTensorBuffer.ts',
    { './webgpuContext': { getNNWebGPUContext: () => ctx } },
    { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 } },
  );
  const data = Uint8Array.from({ length: 16 }, (_, i) => 17 * i);
  new WebGPUTensorBuffer({ byteLength: 16 }, false).setDataRaw(data);
  assert.deepEqual(steps, ['flush', 'write']);
  assert.ok(Buffer.from(dst).equals(Buffer.from(data)));
});

test('WebGPU readback destroys its staging buffer on success and map failure', async () => {
  const stages = [];
  let allocations = 0;
  const source = Uint8Array.from({ length: 16 }, (_, i) => i + 1);
  const device = {
    createBuffer(options) {
      if (allocations++ === 0) return { size: options.size };
      const fail = stages.length === 1;
      const stage = {
        mapAsync: () => fail ? Promise.reject(new Error('device lost')) : Promise.resolve(),
        getMappedRange: () => source.buffer,
        unmap() { this.unmapped = true; },
        destroy() { this.destroyed = true; },
      };
      stages.push(stage);
      return stage;
    },
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
    queue: { submit() {} },
  };
  const ctx = { device, flush() {} };
  const { WebGPUTensorBuffer } = await loadTypeScript(
    '../src/webgpu/webgpuTensorBuffer.ts',
    { './webgpuContext': { getNNWebGPUContext: () => ctx } },
    { GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 },
      GPUMapMode: { READ: 1 } },
  );
  const tensor = new WebGPUTensorBuffer({ byteLength: 16 }, false);
  assert.deepEqual(Array.from(await tensor.getDataRaw()), Array.from(source));
  assert.equal(stages[0].unmapped, true);
  assert.equal(stages[0].destroyed, true);
  await assert.rejects(tensor.getDataRaw(), /device lost/);
  assert.equal(stages[1].unmapped, undefined);
  assert.equal(stages[1].destroyed, true);
});

test('opt-in WebGPU readback reuses only an unmapped successful staging buffer', async () => {
  const source = Uint8Array.of(4, 3, 2, 1);
  const stages = [];
  let cached = null;
  const device = {
    createBuffer(options) {
      if (options.usage !== 12) return { size: options.size };
      const stage = {
        mapAsync: () => Promise.resolve(),
        getMappedRange: () => source.buffer,
        unmap() { this.mapped = false; },
        destroy() { this.destroyed = true; },
      };
      stages.push(stage);
      return stage;
    },
    createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish: () => ({}) }),
    queue: { submit() {} },
  };
  const ctx = {
    device, flush() {},
    rentReadback(size) { const b = cached; cached = null; return b || device.createBuffer({size, usage:12}); },
    returnReadback(buffer, _size, reusable) { if (reusable) cached = buffer; else buffer.destroy(); },
  };
  const { WebGPUTensorBuffer } = await loadTypeScript(
    '../src/webgpu/webgpuTensorBuffer.ts',
    { './webgpuContext': { getNNWebGPUContext: () => ctx } },
    { __wgpyReadbackPool: true,
      GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 },
      GPUMapMode: { READ: 1 } },
  );
  const tensor = new WebGPUTensorBuffer({ byteLength: 4 }, false);
  assert.deepEqual(Array.from(await tensor.getDataRaw()), [4, 3, 2, 1]);
  assert.deepEqual(Array.from(await tensor.getDataRaw()), [4, 3, 2, 1]);
  assert.equal(stages.length, 1);
  assert.equal(stages[0].mapped, false);
  assert.equal(stages[0].destroyed, undefined);
});

test('a failed WebGPU metadata write unmaps and releases its allocation', async () => {
  let unmapped = false, released = false;
  const gpuBuffer = {
    getMappedRange: () => new ArrayBuffer(2),
    unmap() { unmapped = true; },
  };
  const ctx = {
    device: { createBuffer: () => gpuBuffer },
    deferDispose(buffer) { released = buffer === gpuBuffer; },
  };
  const { WebGPUTensorBuffer } = await loadTypeScript(
    '../src/webgpu/webgpuTensorBuffer.ts',
    { './webgpuContext': { getNNWebGPUContext: () => ctx } },
    { GPUBufferUsage: { STORAGE: 1 } },
  );
  const tensor = new WebGPUTensorBuffer({ byteLength: 2 }, true);
  assert.throws(() => tensor.setMetaBufferContent(Uint8Array.of(1, 2, 3, 4)),
    RangeError);
  assert.equal(unmapped, true);
  tensor.dispose();
  assert.equal(released, true);
});

test('WebGPU upload failure wakes the worker with an error status', async () => {
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
  const notify = new SharedArrayBuffer(4);
  const status = new Int32Array(notify);
  context.setData = () => { throw new Error('mapped upload rejected'); };
  context.handleMessage({ method: 'gpu.setData', id: 1, data: new Uint8Array(4), notify });
  assert.equal(status[0], -1);
  context.setData = () => {};
  context.handleMessage({ method: 'gpu.setData', id: 1, data: new Uint8Array(4), notify });
  assert.equal(status[0], 1);
  let complete;
  context.setData = () => new Promise(resolve => { complete = resolve; });
  status[0] = 0;
  context.handleMessage({ method: 'gpu.setData', id: 1, data: new Uint8Array(4), notify });
  assert.equal(status[0], 0, 'worker woke before GPU upload completed');
  complete();
  await Promise.resolve();
  assert.equal(status[0], 1);
  context.setData = () => Promise.reject(new Error('GPU copy failed'));
  status[0] = 0;
  context.handleMessage({ method: 'gpu.setData', id: 1, data: new Uint8Array(4), notify });
  await Promise.resolve();
  assert.equal(status[0], -1, 'worker must wake on asynchronous upload failure');
});

test('GPU sampler reuses the shared readback binding and always wakes the worker', async () => {
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
  const data = new SharedArrayBuffer(65536);
  const notify = new SharedArrayBuffer(4);
  const status = new Int32Array(notify);
  let calls = 0;
  context.sampleLogitsDevice = async (_id, _count, _temperature, _random, target) => {
    assert.equal(target, data);
    calls++;
    return 7;
  };
  context.handleMessage({method:'gpu.sampleLogitsDevice', id:1, count:10,
    temperature:0.6, random:0.5, data, notify});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(status[0], 1);
  status[0] = 0;
  context.handleMessage({method:'gpu.sampleLogitsDevice', id:1, count:10,
    temperature:0.6, random:0.5});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(status[0], 1);
  assert.equal(calls, 2);
  status[0] = 0;
  context.sampleLogitsDevice = () => { throw new Error('device lost'); };
  context.handleMessage({method:'gpu.sampleLogitsDevice', id:1, count:10,
    temperature:0.6, random:0.5});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(status[0], -1);
});

test('re-recording a graph unpins orphaned GPU buffers without unpinning another graph', async () => {
  const { ComputeContextGPU } = await loadTypeScript(
    '../src/webgpu/webgpuComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webgpuContext': { getNNWebGPUContext: () => ({ runKernel() {} }) },
      './webgpuTensorBuffer': {},
      './vocabSampler': { GPUVocabSampler: class {} },
    },
  );
  const context = new ComputeContextGPU();
  const freed = [];
  for (const id of [1, 2, 3]) {
    context.tensorBuffers.set(id, { dispose() { freed.push(id); } });
  }
  const run = ids => context.runKernel({ name: 'probe', tensors: ids, workGroups: { x: 1, y: 1, z: 1 } });
  context.beginCapture('other'); run([3]); context.endCapture();
  context.beginCapture('decode'); run([1, 2]); context.endCapture();
  context.beginCapture('decode'); run([2]); context.endCapture();
  context.disposeBuffer(1);
  context.disposeBuffer(2);
  context.disposeBuffer(3);
  assert.deepEqual(freed, [1]);
  context.resetCaptures();
  context.disposeBuffer(2);
  context.disposeBuffer(3);
  assert.deepEqual(freed, [1, 2, 3]);
});

test('WebGL has the same graph replacement and release lifetime', async () => {
  const { ComputeContextGL } = await loadTypeScript(
    '../src/webgl/webglComputeContext.ts',
    {
      '../util': { nonNull: value => value },
      './webglContext': { getNNWebGLContext: () => ({ runKernel() {} }) },
    },
  );
  const context = new ComputeContextGL();
  const freed = [];
  for (const id of [1, 2]) {
    context.tensorBuffers.set(id, { dispose() { freed.push(id); } });
  }
  const run = id => context.runKernel({ name: 'probe', inputs: [], output: id, uniforms: [] });
  context.beginCapture('decode'); run(1); context.endCapture();
  context.beginCapture('decode'); run(2); context.endCapture();
  context.disposeBuffer(1);
  context.disposeBuffer(2);
  assert.deepEqual(freed, [1]);
  context.resetCaptures();
  context.disposeBuffer(2);
  assert.deepEqual(freed, [1, 2]);
});

for (const backend of ['WebGPU', 'WebGL']) {
  test(`${backend} retires one captured shape without unpinning another`, async () => {
    const gpu = backend === 'WebGPU';
    const imports = gpu ? {
      '../util': { nonNull: value => value },
      './webgpuContext': { getNNWebGPUContext: () => ({ runKernel() {} }) },
      './webgpuTensorBuffer': {},
      './vocabSampler': { GPUVocabSampler: class {} },
    } : {
      '../util': { nonNull: value => value },
      './webglContext': { getNNWebGLContext: () => ({ runKernel() {} }) },
    };
    const exports = await loadTypeScript(gpu
      ? '../src/webgpu/webgpuComputeContext.ts'
      : '../src/webgl/webglComputeContext.ts', imports);
    const context = gpu ? new exports.ComputeContextGPU() : new exports.ComputeContextGL();
    const freed = [];
    for (const id of [1, 2, 3]) context.tensorBuffers.set(id, {
      dispose() { freed.push(id); },
    });
    const run = ids => context.runKernel(gpu
      ? { name: 'probe', tensors: ids, workGroups: { x: 1, y: 1, z: 1 } }
      : { name: 'probe', inputs: ids.slice(0, -1).map(id => ({ name: 'x', id })),
          output: ids.at(-1), uniforms: [] });
    context.beginCapture('older'); run([1, 2]); context.endCapture();
    context.beginCapture('hot'); run([2, 3]); context.endCapture();
    context.releaseCapture('older');
    for (const id of [1, 2, 3]) context.disposeBuffer(id);
    assert.deepEqual(freed, [1]);
    context.releaseCapture('hot');
    context.disposeBuffer(2);
    context.disposeBuffer(3);
    assert.deepEqual(freed, [1, 2, 3]);
    assert.throws(() => context.replay('older'), /not found/);
  });
}
