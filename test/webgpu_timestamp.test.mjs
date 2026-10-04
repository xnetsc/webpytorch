import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

test('opt-in GPU timestamp uses the browser compute-pass descriptor and releases queries', async () => {
  const source = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const actions = [];
  const marker = { __wgpyProfileNextPass: true };
  const ticks = new BigUint64Array([0n, 393216n, 0n, 393216n]);
  const query = { destroy() { actions.push('query destroyed'); } };
  const readback = {
    mapAsync: async () => {},
    getMappedRange: () => ticks.buffer,
    unmap() { actions.push('read unmapped'); },
    destroy() { actions.push('read destroyed'); },
  };
  const resolve = { destroy() { actions.push('resolve destroyed'); } };
  let bufferCount = 0;
  let passDescriptor;
  const device = {
    features: new Set(['timestamp-query']),
    createQuerySet: () => query,
    createBuffer: () => bufferCount++ % 2 ? readback : resolve,
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass(descriptor) {
        passDescriptor = descriptor;
        return { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
      },
      resolveQuerySet() { actions.push('resolved'); },
      copyBufferToBuffer() { actions.push('copied'); },
      finish: () => ({}),
    }),
    queue: { submit() { actions.push('submitted'); } },
  };
  const adapter = {
    limits: {}, features: new Set(['timestamp-query']),
    requestDevice: async ({ requiredFeatures }) => {
      assert.deepEqual(Array.from(requiredFeatures), ['timestamp-query']);
      return device;
    },
  };
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: () => ({ WebGPUTensorBuffer: class {} }),
    globalThis: marker,
    location: { search: '?profile_gpu=1' },
    URLSearchParams,
    navigator: { gpu: { requestAdapter: async () => adapter } },
    GPUBufferUsage: { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 },
    GPUMapMode: { READ: 1 },
    console: { info() {} },
    BigUint64Array,
  });
  const ctx = new module.exports.NNWebGPUContext();
  await ctx.initialize();
  ctx.pipelines.set('probe', { bindGroupLayout: {}, pipeline: {} });
  ctx.runKernel({ pipelineName: 'probe', tensorBuffers: [],
                  workGroups: { x: 1, y: 1, z: 1 } });
  ctx.flush();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(marker.__wgpyProfileNextPass, false);
  assert.equal(passDescriptor.timestampWrites.querySet, query);
  assert.equal(passDescriptor.timestampWrites.beginningOfPassWriteIndex, 0);
  assert.equal(passDescriptor.timestampWrites.endOfPassWriteIndex, 1);
  assert.equal(marker.__wgpyGpuTiming.gpuMs, 0.393216);
  assert.deepEqual(actions, [
    'resolved', 'copied', 'submitted', 'read unmapped',
    'read destroyed', 'resolve destroyed', 'query destroyed',
  ]);

  marker.__wgpyProfileAllPasses = true;
  marker.__wgpyGpuPasses = [];
  for (let i = 0; i < 2; i++) {
    ctx.runKernel({ pipelineName: 'probe', tensorBuffers: [],
                    workGroups: { x: 1, y: 1, z: 1 } });
    ctx.flush();
  }
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(Array.from(marker.__wgpyGpuPasses, item =>
    [item.index, item.dispatches, item.gpuMs]), [
    [2, 1, 0.393216], [3, 1, 0.393216],
  ]);

  marker.__wgpyProfileAllPasses = false;
  marker.__wgpyProfileKernelNames = ['probe'];
  marker.__wgpySelectedKernelPasses = [];
  const submittedBefore = actions.filter(action => action === 'submitted').length;
  ctx.runKernel({ pipelineName: 'probe', tensorBuffers: [],
                  workGroups: { x: 1, y: 1, z: 1 } });
  ctx.runKernel({ pipelineName: 'probe', tensorBuffers: [],
                  workGroups: { x: 1, y: 1, z: 1 } });
  ctx.flush();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(actions.filter(action => action === 'submitted').length, submittedBefore + 1);
  assert.deepEqual(Array.from(marker.__wgpySelectedKernelPasses, item =>
    [item.dispatches, item.byName.probe.count, item.byName.probe.gpuMs]),
    [[2, 2, 0.786432]]);

  marker.__wgpyProfileKernelWorkgroups = true;
  marker.__wgpySelectedKernelPasses = [];
  ctx.runKernel({ pipelineName: 'probe', tensorBuffers: [],
                  workGroups: { x: 2, y: 3, z: 4 } });
  ctx.flush();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(marker.__wgpySelectedKernelPasses[0].byName['probe@2x3x4'].count, 1);
});
