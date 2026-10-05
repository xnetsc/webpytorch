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
  });
  vm.runInContext(`const METHODS = { ${method} }; this.decide = METHODS.decide;`, context);
  return { context, answer, writes, pythonCalls, cleanupCalls };
}

test('decision replies keep newly measured device routes without changing answers', async () => {
  const h = harness(true);
  const result = await h.context.decide({ state: 'ticket', questions: {}, profile: true });
  assert.equal(result, h.answer);
  assert.equal(h.pythonCalls.length, 1);
  assert.equal(h.writes.length, 1);
  assert.equal(h.writes[0].key, 'this-device');
  assert.equal(h.writes[0].value.tuned.route, 'stored');
  assert.equal(h.cleanupCalls.length, 1);
  assert.match(h.cleanupCalls[0], /_wt_core\._gpu_release_idle_pool\(\)/);
  assert.equal(h.context.root.__decide_in, null);
});

test('hot decision replies trim scratch without exporting an unchanged profile', async () => {
  const h = harness(false);
  assert.equal(await h.context.decide({ state: 'ticket', questions: {} }), h.answer);
  assert.equal(h.pythonCalls.length, 0);
  assert.equal(h.writes.length, 0);
  assert.equal(h.cleanupCalls.length, 1);
});
