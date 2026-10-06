import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

// A buffer zeroed where it lives, in command order, with no host data: WebGPU's own clear in
// the open command buffer, WebGL's framebuffer clear typed to the texture.

async function method(path, signature) {
  const source = await readFile(new URL(path, import.meta.url), 'utf8');
  const start = source.indexOf(signature);
  assert.ok(start >= 0, signature);
  const body = source.slice(start, source.indexOf('\n  }\n', start) + 4);
  return ts.transpileModule('class X {\n' + body + '\n}', {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText;
}

test('WebGPU clears inside the open command buffer, after the pending dispatches', async () => {
  const cls = await method('../src/webgpu/webgpuContext.ts', '  clearBuffer(buffer: GPUBuffer): void {');
  const sandbox = vm.createContext({ order: [] });
  vm.runInContext(cls + `
    var C = Object.assign(Object.create(X.prototype), {
      pendingCount: 0, flushThreshold: 3, assertAlive() {},
      passEncoder: { end() { order.push('end pass'); } },
      commandEncoder: { clearBuffer(b) { order.push('clear ' + b.name); } },
      flush() { order.push('submit'); this.pendingCount = 0; },
    });`, sandbox);
  sandbox.C.clearBuffer({ name: 'S' });
  sandbox.C.clearBuffer({ name: 'conv' });
  assert.deepEqual(Array.from(sandbox.order), ['end pass', 'clear S', 'clear conv']);   // no submit
  sandbox.C.clearBuffer({ name: 'third' });
  assert.deepEqual(Array.from(sandbox.order).slice(-2), ['clear third', 'submit']);    // threshold
});

test('the WebGPU thread routes a clear to the tensor\'s own buffer', async () => {
  const source = await readFile(new URL('../src/webgpu/webgpuComputeContext.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const cleared = [];
  const module = { exports: {} };
  const imports = {
    '../util': { nonNull: v => v },
    './webgpuContext': { getNNWebGPUContext: () => ({ clearBuffer: b => cleared.push(b) }) },
    './webgpuTensorBuffer': {}, './vocabSampler': { GPUVocabSampler: class {} },
    '../sharedReadback': { writeSharedReadbackError() {} },
    '../stagedRead': { stagedReadTarget() { throw new Error('unused'); } },
  };
  vm.runInNewContext(compiled, { module, exports: module.exports, require: n => imports[n],
                                 Map, Set, Array, Error, console });
  const gpu = new module.exports.ComputeContextGPU();
  const buffer = { name: 'state' };
  gpu.tensorBuffers.set(3, { gpuBuffer: buffer });
  gpu.handleMessage({ method: 'gpu.clearBuffer', id: 3 });
  assert.deepEqual(cleared, [buffer]);
  assert.throws(() => gpu.handleMessage({ method: 'gpu.clearBuffer', id: 4 }), /was not created/);
});

test('WebGL clears every layer of the texture with the clear its format takes', async () => {
  const cls = await method('../src/webgl/webglContext.ts', '  clear(): void {');
  const gl = { COLOR: 0x1800, FLOAT: 0x1406, HALF_FLOAT: 0x140b, INT: 0x1404,
               UNSIGNED_INT: 0x1405, UNSIGNED_BYTE: 0x1401, calls: [] };
  for (const kind of ['fv', 'iv', 'uiv']) {
    gl['clearBuffer' + kind] = (buffer, index, values) => gl.calls.push(
      [kind, buffer, index, Array.from(values)]);
  }
  const sandbox = vm.createContext({ gl, Int32Array, Uint32Array, Float32Array });
  vm.runInContext(cls + `
    function getNNWebGLContext() { return { gl }; }
    function make(shape) {
      const t = Object.create(X.prototype);
      t.textureShape = shape; t.bound = [];
      t.bindToDrawTexture = (layer) => t.bound.push('bind ' + layer);
      t.unbindFromDrawTexture = () => t.bound.push('unbind');
      return t;
    }`, sandbox);
  const flat = sandbox.make({ dim: '2D', type: gl.FLOAT, width: 4, height: 2 });
  flat.clear();
  assert.deepEqual(Array.from(flat.bound), ['bind 0', 'unbind']);
  assert.deepEqual(gl.calls.splice(0), [['fv', gl.COLOR, 0, [0, 0, 0, 0]]]);
  const layered = sandbox.make({ dim: '2DArray', type: gl.FLOAT, width: 4, height: 2, depth: 3 });
  layered.clear();
  assert.deepEqual(Array.from(layered.bound),
                   ['bind 0', 'unbind', 'bind 1', 'unbind', 'bind 2', 'unbind']);
  assert.equal(gl.calls.splice(0).length, 3);
  sandbox.make({ dim: '2D', type: gl.INT, width: 1, height: 1 }).clear();
  assert.equal(gl.calls.splice(0)[0][0], 'iv');
});

test('the WebGL thread routes a clear to the tensor buffer, and the worker sends both', async () => {
  const compute = await readFile(new URL('../src/webgl/webglComputeContext.ts', import.meta.url), 'utf8');
  assert.match(compute, /case 'gl\.clearBuffer': \{[\s\S]*?tb\.clear\(\);/);
  const worker = await readFile(new URL('../src/worker.ts', import.meta.url), 'utf8');
  assert.match(worker, /clearBuffer: \(id: number\) => \{\s*commands\.enqueue\(\{ method: 'gl\.clearBuffer', id \}\);/);
  assert.match(worker, /clearBuffer: \(id: number\) => \{\s*commands\.enqueue\(\{ method: 'gpu\.clearBuffer', id \}\);/);
});
