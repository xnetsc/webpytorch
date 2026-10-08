import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

const source = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

function setup() {
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: () => ({ WebGPUTensorBuffer: class {} }),
    navigator: { gpu: { requestAdapter() {} } }, URLSearchParams, console,
  });
  const ctx = new module.exports.NNWebGPUContext();
  let submits = 0;
  ctx.device = {
    features: new Set(), createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({ setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }),
      finish: () => ({}), clearBuffer() {},
    }),
    queue: { submit() { submits++; } },
  };
  ctx.pipelines.set('test', { bindGroupLayout: {}, pipeline: {} });
  const dispatch = () => ctx.runKernel({ pipelineName: 'test', tensorBuffers: [], workGroups: { x: 1, y: 1, z: 1 } });
  return { ctx, dispatch, submits: () => submits };
}

test('a finite submission stays together across producer chunks and dispatch thresholds', () => {
  const { ctx, dispatch, submits } = setup();
  ctx.beginSubmission();
  for (let i = 0; i < 1100; i++) { dispatch(); ctx.kick(); }
  assert.equal(submits(), 0);
  ctx.endSubmission();
  assert.equal(submits(), 1);
  assert.equal(ctx.pendingCount, 0);
  dispatch(); ctx.kick();
  assert.equal(submits(), 2); // ordinary inference keeps submit-when-idle
});

test('nested scopes preserve explicit data barriers and release on the outer boundary', () => {
  const { ctx, dispatch, submits } = setup();
  ctx.beginSubmission(); ctx.beginSubmission();
  dispatch(); ctx.flush(); // upload/readback ordering must never be deferred
  assert.equal(submits(), 1);
  dispatch(); ctx.endSubmission(); ctx.kick();
  assert.equal(submits(), 1);
  ctx.endSubmission();
  assert.equal(submits(), 2);
  assert.throws(() => ctx.endSubmission(), /unbalanced/);
});

test('device clears obey the same finite submission boundary', () => {
  const { ctx, submits } = setup();
  ctx.beginSubmission();
  for (let i = 0; i < 1100; i++) ctx.clearBuffer({});
  assert.equal(submits(), 0);
  ctx.endSubmission();
  assert.equal(submits(), 1);
});
