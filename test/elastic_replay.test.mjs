// A recording made at a capacity, replayed for the quantities a call has
// (`GPUKernelRunDescriptor.dyn`, `ComputeContextGPU.replay(name, live)`).
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function loadContext(issued) {
  const source = await readFile(new URL('../src/webgpu/webgpuComputeContext.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  const imports = {
    '../util': { nonNull: value => value },
    './webgpuContext': { getNNWebGPUContext: () => ({ runKernel(r) { issued.push(r); } }) },
    './webgpuTensorBuffer': {},
    './vocabSampler': { GPUVocabSampler: class {} },
    '../sharedReadback': { writeSharedReadbackError() {} },
    '../stagedRead': { stagedReadTarget() { throw new Error('unused'); } },
  };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: name => { assert.ok(name in imports, `unexpected import ${name}`); return imports[name]; },
    Uint8Array, Int32Array, SharedArrayBuffer, Atomics, console: { error() {} },
  });
  const context = new module.exports.ComputeContextGPU();
  for (const id of [1, 2, 3]) context.tensorBuffers.set(id, { dispose() {} });
  return context;
}

test('a replay issues each ruled dispatch for the live quantities, never past its recording', async () => {
  const issued = [];
  const context = await loadContext(issued);
  context.beginCapture('pass');
  // 512 rows of a 32-row tiled matmul; an elementwise pass of 768 per row; attention over
  // 12 heads of 8 sequences; one dispatch with no rule at all.
  context.runKernel({ name: 'mm', tensors: [1, 2], workGroups: { x: 36, y: 16, z: 1 },
                      dyn: { y: ['rows', 1, 32] } });
  context.runKernel({ name: 'ew', tensors: [2, 3], workGroups: { x: 6144, y: 1, z: 1 },
                      dyn: { x: ['rows', 768, 64] } });
  context.runKernel({ name: 'attn', tensors: [1, 3], workGroups: { x: 16, y: 96, z: 1 },
                      dyn: { x: ['longest', 1, 32], y: ['segments', 12, 1] } });
  context.runKernel({ name: 'fixed', tensors: [3], workGroups: { x: 4, y: 1, z: 1 } });
  context.endCapture();
  issued.length = 0;
  context.replay('pass', { rows: 160, segments: 1, longest: 160 });
  assert.deepEqual(issued.map(r => [r.pipelineName, r.workGroups.x, r.workGroups.y]),
                   [['mm', 36, 5], ['ew', 1920, 1], ['attn', 5, 12], ['fixed', 4, 1]]);
  // Without live quantities it is the recording, as recorded.
  issued.length = 0;
  context.replay('pass');
  assert.deepEqual(issued.map(r => r.workGroups.y), [16, 1, 96, 1]);
  // Nothing live: a zero count issues nothing for that dispatch.
  issued.length = 0;
  context.replay('pass', { rows: 0, segments: 0, longest: 0 });
  assert.deepEqual(issued.map(r => r.pipelineName), ['fixed']);
  // Past the capacity, or a quantity the recording needs and the call did not give.
  assert.throws(() => context.replay('pass', { rows: 513, segments: 1, longest: 1 }),
                /past the capacity/);
  assert.throws(() => context.replay('pass', { rows: 10 }), /needs the live quantity '(longest|segments)'/);
});

test('a staged upload into a capacity writes the live prefix and leaves the rest', async () => {
  const source = await readFile(new URL('../src/webgpu/webgpuTensorBuffer.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  const writes = [];
  const ctx = {
    assertAlive() {}, flush() {},
    device: {
      createBuffer: (o) => ({ size: o.size, getMappedRange() { return new ArrayBuffer(o.size); }, unmap() {} }),
      queue: { writeBuffer(buffer, offset, data) { writes.push([offset, data.byteLength]); } },
    },
  };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: name => { assert.equal(name, './webgpuContext'); return { getNNWebGPUContext: () => ctx }; },
    Uint8Array, GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 },
  });
  const buffer = new module.exports.WebGPUTensorBuffer({ byteLength: 4096 }, false);
  buffer.setDataRaw(new Uint8Array(640), true);
  assert.deepEqual(writes, [[0, 640]]);
  assert.throws(() => buffer.setDataRaw(new Uint8Array(640)), /size mismatch/);
  assert.throws(() => buffer.setDataRaw(new Uint8Array(8192), true), /size mismatch/);
});
