import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const ctx = await readFile(new URL('../src/webgpu/webgpuContext.ts', import.meta.url), 'utf8');
const compute = await readFile(new URL('../src/webgpu/webgpuComputeContext.ts', import.meta.url), 'utf8');
const main = await readFile(new URL('../src/main.ts', import.meta.url), 'utf8');

test('pending dispatches are submitted as soon as the GPU has nothing in flight', () => {
  const start = ctx.indexOf('  kick(): void {');
  assert.ok(start >= 0);
  const body = ctx.slice(start, ctx.indexOf('\n  }\n', start) + 4).replace('kick(): void', 'kick()');
  const sandbox = vm.createContext({});
  vm.runInContext('var K = { inflight: 0, pendingCount: 0, diagnosticQuery: null, flushes: 0, '
                  + 'flush() { this.flushes++; this.pendingCount = 0; this.inflight++; }, '
                  + body + ' };', sandbox);
  const K = sandbox.K;
  K.kick(); assert.equal(K.flushes, 0);                 // nothing pending
  K.pendingCount = 5; K.kick(); assert.equal(K.flushes, 1);
  K.pendingCount = 7; K.kick(); assert.equal(K.flushes, 1);   // GPU busy: keep accumulating
  K.inflight = 0; K.kick(); assert.equal(K.flushes, 2);       // it finished: submit at once
  K.inflight = 0; K.pendingCount = 3; K.diagnosticQuery = {}; K.kick();
  assert.equal(K.flushes, 2);                           // a timed pass is left whole
});

test('a submit counts in flight until done, then submits what accumulated', () => {
  assert.match(ctx, /queue\.submit\(\[this\.commandEncoder\.finish\(\)\]\);[\s\S]{0,200}onSubmittedWorkDone\?\.\(\);\s*if \(done && typeof done\.then === 'function'\) \{\s*this\.inflight\+\+;\s*done\.then\(\(\) => \{\s*this\.inflight--;\s*this\.kick\(\);/);
  assert.match(compute, /afterBatch\(\): void \{\s*try \{ getNNWebGPUContext\(\)\.kick\(\); \}/);
  assert.match(main, /releaseShared\(sharedGPU, e\.data\.slot\);\s*contextGPU\.afterBatch\(\);/);
});
