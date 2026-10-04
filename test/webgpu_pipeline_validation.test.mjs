import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const source = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

function makeContext(messages = [], scopeError = null) {
  const device = {
    features: new Set(),
    pushErrorScope() {},
    popErrorScope: async () => scopeError,
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createShaderModule: () => ({ getCompilationInfo: async () => ({ messages }) }),
    createComputePipeline: () => ({}),
  };
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: () => ({ WebGPUTensorBuffer: class {} }),
    navigator: { gpu: { requestAdapter() {} } },
    GPUShaderStage: { COMPUTE: 1 },
    URLSearchParams,
    console,
  });
  const ctx = new module.exports.NNWebGPUContext();
  ctx.device = device;
  return ctx;
}

test('a WGSL compiler error fails the inference route before readback', async () => {
  const ctx = makeContext([{ type: 'error', message: "'meta' is reserved" }]);
  ctx.createPipeline('gather_rows', 'invalid wgsl', ['storage']);
  await assert.rejects(ctx.assertPipelinesReady(), /gather_rows.*meta.*reserved/);
  assert.throws(() => ctx.runKernel({ pipelineName: 'gather_rows', tensorBuffers: [],
                                     workGroups: { x: 1, y: 1, z: 1 } }), /gather_rows/);
  await assert.rejects(ctx.assertPipelinesReady(), /gather_rows/);
});

test('WebGPU pipeline validation errors also fail explicitly', async () => {
  const ctx = makeContext([], { message: 'binding layout mismatch' });
  ctx.createPipeline('dense', 'valid wgsl with invalid layout', ['storage']);
  await assert.rejects(ctx.assertPipelinesReady(), /dense.*binding layout mismatch/);
});

test('validated pipelines remain usable without rechecking old work', async () => {
  const ctx = makeContext();
  ctx.createPipeline('dense', 'valid wgsl', ['storage']);
  await ctx.assertPipelinesReady();
  await ctx.assertPipelinesReady();
  assert.equal(ctx.hasPipeline('dense'), true);
});
