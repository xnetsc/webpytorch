import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import ts from 'typescript';

async function loadTypeScript(path, imports = {}) {
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
    Uint8Array, Int32Array, SharedArrayBuffer, Atomics, Array, Number, Promise, String,
    TextEncoder, TextDecoder, Error, Map, console,
    GPUBufferUsage: { COPY_DST: 8, MAP_READ: 1 }, GPUMapMode: { READ: 1 },
  });
  return module.exports;
}

const staged = await loadTypeScript('../src/stagedRead.ts');

test('a staged read is collected from its slot once the GPU thread finishes it', () => {
  const arena = staged.stagedReadArena(4, 64);
  const bind = arena.binding();
  assert.ok(bind && bind.memory && bind.error);
  assert.equal(arena.binding(), null);                 // bound once
  const target = staged.stagedReadTarget(bind.memory, bind.error, bind.slots, bind.slotBytes);
  const seq = arena.stage(1, 8);
  target.finish(1, seq, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
  assert.deepEqual(Array.from(arena.collect(1, 8)), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('a late completion of an abandoned read cannot overwrite a newer one', () => {
  const arena = staged.stagedReadArena(4, 64);
  const bind = arena.binding();
  const target = staged.stagedReadTarget(bind.memory, bind.error, bind.slots, bind.slotBytes);
  const old = arena.stage(2, 4);                       // never collected
  const now = arena.stage(2, 4);
  target.finish(2, now, new Uint8Array([9, 9, 9, 9]));
  target.finish(2, old, new Uint8Array([1, 1, 1, 1]));
  assert.deepEqual(Array.from(arena.collect(2, 4)), [9, 9, 9, 9]);
});

test('a failed read wakes the collector with the GPU thread\'s reason', () => {
  const arena = staged.stagedReadArena(4, 64);
  const bind = arena.binding();
  const target = staged.stagedReadTarget(bind.memory, bind.error, bind.slots, bind.slotBytes);
  const seq = arena.stage(3, 4);
  target.finish(3, seq, null, new Error('device lost'));
  assert.throws(() => arena.collect(3, 4), /staged readback failed: device lost/);
});

test('slots and sizes are checked before anything is queued', () => {
  const arena = staged.stagedReadArena(4, 64);
  assert.throws(() => arena.stage(4, 4), /outside/);
  assert.throws(() => arena.stage(0, 6), /does not fit/);
  assert.throws(() => arena.stage(0, 68), /does not fit/);
  assert.throws(() => arena.collect(0, 4), /nothing was staged/);
});

test('the GPU thread copies behind queued work, maps, delivers, and reuses the copy', async () => {
  const calls = [];
  const src = { name: 'tokens' };
  let created = 0;
  const ctx = {
    device: {
      createBuffer: ({ size }) => {
        created++;
        return {
          size, mapAsync: async () => { calls.push('map'); },
          getMappedRange: (o, n) => new Uint8Array([4, 0, 0, 0, 5, 0, 0, 0]).buffer.slice(o, o + n),
          unmap: () => calls.push('unmap'), destroy: () => calls.push('destroy'),
        };
      },
    },
    copyAndSubmit: (from, to, n) => calls.push(['copy', from.name, n]),
    assertPipelinesReady: async () => {}, assertAlive: () => {},
  };
  const { ComputeContextGPU } = await loadTypeScript('../src/webgpu/webgpuComputeContext.ts', {
    '../util': { nonNull: v => v },
    './webgpuContext': { getNNWebGPUContext: () => ctx },
    './webgpuTensorBuffer': {},
    './vocabSampler': { GPUVocabSampler: class {} },
    '../sharedReadback': { writeSharedReadbackError() {} },
    '../stagedRead': staged,
  });
  const gpu = new ComputeContextGPU();
  const arena = staged.stagedReadArena(4, 64);
  gpu.handleMessage({ method: 'gpu.stageArena', ...arena.binding() });
  gpu.tensorBuffers.set(7, { gpuBuffer: src, bufferShape: { byteLength: 32 } });
  for (let round = 0; round < 2; round++) {
    const seq = arena.stage(0, 8);
    gpu.handleMessage({ method: 'gpu.stageRead', id: 7, byteLength: 8, slot: 0, seq });
    for (let i = 0; i < 4; i++) await Promise.resolve();
    assert.deepEqual(Array.from(new Int32Array(arena.collect(0, 8).buffer)), [4, 5]);
  }
  assert.equal(created, 1);                            // the mapped copy is reused
  assert.deepEqual(calls, [['copy', 'tokens', 8], 'map', 'unmap', ['copy', 'tokens', 8], 'map', 'unmap']);
  const bad = arena.stage(1, 8);
  gpu.handleMessage({ method: 'gpu.stageRead', id: 99, byteLength: 8, slot: 1, seq: bad });
  assert.throws(() => arena.collect(1, 8), /was not created/);
});

test('the copy goes into the open command buffer after the pending dispatches, then submits', async () => {
  const source = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
  const start = source.indexOf('  copyAndSubmit(src: GPUBuffer, dst: GPUBuffer, byteLength: number): void {');
  assert.ok(start >= 0);
  const body = source.slice(start, source.indexOf('\n  }\n', start) + 4)
    .replace('copyAndSubmit(src: GPUBuffer, dst: GPUBuffer, byteLength: number): void', 'copyAndSubmit(src, dst, byteLength)');
  const sandbox = vm.createContext({ order: [] });
  vm.runInContext('var C = { pendingCount: 2, assertAlive() {}, '
                  + 'passEncoder: { end() { order.push("end pass"); } }, '
                  + 'commandEncoder: { copyBufferToBuffer(s, so, d, dof, n) { order.push("copy " + n); } }, '
                  + 'flush() { order.push("submit"); }, ' + body + ' };', sandbox);
  sandbox.C.copyAndSubmit({}, {}, 20);
  assert.deepEqual(Array.from(sandbox.order), ['end pass', 'copy 20', 'submit']);
  assert.equal(sandbox.C.passEncoder, null);
});
