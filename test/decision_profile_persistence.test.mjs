import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../webtorch/js/webtorch-host.js', import.meta.url), 'utf8');
const begin = source.indexOf('    async decide(a) {');
const end = source.indexOf('    async calibrate(a) {', begin);
assert.ok(begin >= 0 && end > begin);
const method = source.slice(begin, end).trim().replace(/,\s*$/, '');

function harness(newTuning) {
  const writes = [];
  const pythonCalls = [];
  const cleanupCalls = [];
  const idleReleases = [];
  const answer = { answers: { billing: { choice: 'billing' } } };
  const context = vm.createContext({
    ready: true,
    remember: true,
    root: {},
    pyJSON: async code => {
      assert.match(code, /_tuning_after != _tuning_before/);
      return { answer, new_tuning: newTuning };
    },
    py: async code => {
      if (code.includes('_gpu_release_idle_pool()')) {
        cleanupCalls.push(code);
        return null;
      }
      pythonCalls.push(code);
      return JSON.stringify({ build: 'current', tuned: { route: 'stored' } });
    },
    kpKey: async () => 'this-device',
    kpPut: async (key, value) => { writes.push({ key, value }); },
    report: (_scope, message) => { throw new Error(message); },
    releaseDecisionScratchWhenIdle: () => { idleReleases.push(Date.now()); },
  });
  vm.runInContext(`const METHODS = { ${method} }; this.decide = METHODS.decide;`, context);
  return { context, answer, writes, pythonCalls, cleanupCalls, idleReleases };
}

test('decision replies keep newly measured device routes without changing answers', async () => {
  const h = harness(true);
  const result = await h.context.decide({ state: 'ticket', questions: {}, profile: true });
  assert.equal(result, h.answer);
  assert.equal(h.pythonCalls.length, 1);
  assert.equal(h.writes.length, 1);
  assert.equal(h.writes[0].key, 'this-device');
  assert.equal(h.writes[0].value.tuned.route, 'stored');
  // The scratch release is queued for idle time, never awaited before the answer.
  assert.equal(h.cleanupCalls.length, 0);
  assert.equal(h.idleReleases.length, 1);
  assert.equal(h.context.root.__decide_in, null);
});

test('hot decision replies trim scratch without exporting an unchanged profile', async () => {
  const h = harness(false);
  assert.equal(await h.context.decide({ state: 'ticket', questions: {} }), h.answer);
  assert.equal(h.pythonCalls.length, 0);
  assert.equal(h.writes.length, 0);
  assert.equal(h.cleanupCalls.length, 0);
  assert.equal(h.idleReleases.length, 1);
});

// The release itself: once, after the idle delay, and never while a call is in flight.
const fbegin = source.indexOf('  const DECISION_SCRATCH_IDLE_MS');
const fend = source.indexOf('\n  }\n', source.indexOf('function releaseDecisionScratchWhenIdle', fbegin)) + 4;
assert.ok(fbegin >= 0 && fend > fbegin);
const releaser = source.slice(fbegin, fend);

function idleHarness() {
  const ran = [];
  const context = vm.createContext({
    ready: true, inflight: 0, Date, setTimeout, Math,
    py: async (code) => { ran.push(code); return null; },
    report: (_scope, message) => { throw new Error(message); },
  });
  vm.runInContext(releaser + '\nthis.release = releaseDecisionScratchWhenIdle;', context);
  return { context, ran };
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

test('a burst of decisions releases scratch once, after it ends', async () => {
  const h = idleHarness();
  h.context.release();
  await sleep(120);
  h.context.release();                      // another answer inside the idle window
  await sleep(120);
  assert.equal(h.ran.length, 0);           // 240 ms since the first, 120 since the last
  await sleep(150);
  assert.equal(h.ran.length, 1);
  assert.match(h.ran[0], /_wt_core\._gpu_release_idle_pool\(\)/);
  assert.match(h.ran[0], /_wt_gc\.collect\(\)/);
});

test('scratch is not released while a call is running', async () => {
  const h = idleHarness();
  vm.runInContext('inflight = 1;', h.context);
  h.context.release();
  await sleep(300);
  assert.equal(h.ran.length, 0);
  vm.runInContext('inflight = 0;', h.context);
  await sleep(120);
  assert.equal(h.ran.length, 1);
});
