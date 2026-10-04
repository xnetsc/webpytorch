import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function samplerWithMocks(compilationMessages = []) {
  const source = await readFile(new URL('../src/webgpu/vocabSampler.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const buffers = [];
  const dispatches = [];
  const steps = [];
  class BufferMock {
    constructor(shape) { this.bufferShape = shape; buffers.push(this); }
    setDataRaw(data) { steps.push('upload'); this.upload = Uint8Array.from(data); }
    async getDataInto(target) {
      steps.push('readback');
      new DataView(target.buffer, target.byteOffset, 4).setInt32(0, 7, true);
    }
    dispose() { this.disposed = true; }
  }
  const context = {
    device: { createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: compilationMessages }),
    }) },
    createPipeline(...args) { this.pipeline = args; },
    runKernel(request) { steps.push('dispatch'); dispatches.push(request); },
  };
  const module = { exports: {} };
  vm.runInNewContext(compiled, {
    module, exports: module.exports,
    require: name => name === './webgpuContext'
      ? { getNNWebGPUContext: () => context }
      : name === './webgpuTensorBuffer'
      ? { WebGPUTensorBuffer: BufferMock }
      : assert.fail(`unexpected import ${name}`),
    Uint8Array, ArrayBuffer, DataView, Number, Error,
  });
  return { GPUVocabSampler: module.exports.GPUVocabSampler,
    source: module.exports.VOCAB_SAMPLE_FULL_WGSL, buffers, dispatches, steps, context };
}

test('GPU vocabulary sampler keeps metadata and output in JS and reuses them', async () => {
  const { GPUVocabSampler, source, buffers, dispatches, steps, context } = await samplerWithMocks();
  assert.doesNotMatch(source, /\blet target\b/);
  const sampler = new GPUVocabSampler();
  const logits = { bufferShape: { byteLength: 1024 } };
  assert.equal(await sampler.sample(logits, 200, 0.6, 0.25, new Uint8Array(4)), 7);
  assert.equal(await sampler.sample(logits, 200, 0.6, 0.5, new Uint8Array(4)), 7);
  assert.equal(buffers.length, 2);
  assert.equal(dispatches.length, 2);
  assert.deepEqual(steps, ['upload', 'dispatch', 'readback',
    'upload', 'dispatch', 'readback']);
  assert.equal(dispatches[0].tensorBuffers[0], logits);
  assert.equal(context.pipeline[0], 'vocab_sample_full_js');
  const meta = new DataView(buffers[0].upload.buffer);
  assert.equal(meta.getUint32(0, true), 200);
  assert.equal(meta.getFloat32(8, true), 0.5);
  sampler.dispose();
  assert.equal(buffers[0].disposed, true);
  assert.equal(buffers[1].disposed, true);
});

test('GPU vocabulary sampler rejects invalid bounds and shader diagnostics', async () => {
  const { GPUVocabSampler } = await samplerWithMocks();
  const sampler = new GPUVocabSampler();
  const logits = { bufferShape: { byteLength: 16 } };
  await assert.rejects(sampler.sample(logits, 5, 1, 0.5, new Uint8Array(4)), /vocabulary/);
  await assert.rejects(sampler.sample(logits, 4, 0, 0.5, new Uint8Array(4)), /temperature/);
  const bad = await samplerWithMocks([{ type: 'error', message: 'reserved word' }]);
  await assert.rejects(new bad.GPUVocabSampler().sample(
    logits, 4, 1, 0.5, new Uint8Array(4)), /reserved word/);
});
