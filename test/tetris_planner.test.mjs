// The Tetris AI plays vanilla-js-tetris through the game's own functions. These tests run the
// vendored game file itself, unmodified, in a vm with a stub canvas, and hold the planner to
// what the game actually does: every planned placement, pressed through bridge.js, has to
// land on exactly the board the planner predicted.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const planner = require('../tetris/planner.js');
const tetrisSource = await readFile(new URL('../tetris/vendor/tetris.js', import.meta.url), 'utf8');
const bridgeSource = await readFile(new URL('../tetris/bridge.js', import.meta.url), 'utf8');
const aiSource = await readFile(new URL('../tetris/ai.js', import.meta.url), 'utf8');
const models = JSON.parse(await readFile(new URL('../chat/models.json', import.meta.url), 'utf8'));

function seeded(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Arrays made inside the vm have that realm's Array prototype, which strict deep equality
// counts as a difference; compare their contents.
const plain = (value) => JSON.parse(JSON.stringify(value));

// The game as the page runs it, minus the screen: tetris.js then bridge.js as two classic
// scripts in one realm, so the bridge reaches the game's globals the way it does in a page.
function game(seed) {
  const noop = () => {};
  const ctx2d = new Proxy({}, { get: () => noop, set: () => true });
  const elements = {};
  const document = {
    getElementById: (id) => elements[id] || (elements[id] = {
      getContext: () => ctx2d, width: 200, height: 400, textContent: '', addEventListener: noop,
    }),
    addEventListener: noop,
  };
  let onload = null;
  const window = { addEventListener: (type, fn) => { if (type === 'load') onload = fn; } };
  const context = vm.createContext({ document, window, requestAnimationFrame: noop,
                                     console: { log: noop } });
  vm.runInContext('Math.random = (' + seeded.toString() + ')(' + seed + ');', context);
  vm.runInContext(tetrisSource, context);
  vm.runInContext(bridgeSource, context);
  onload();
  return { api: window.TetrisGame, context };
}

test('every planned placement lands exactly where the planner said, through the real game', () => {
  let dropped = 0;
  for (const seed of [1, 2, 3, 4]) {
    const { api } = game(seed);
    const rotate = (m, dir) => api.rotateMatrix(m, dir);
    const pick = seeded(seed + 100);
    for (let n = 0; n < 60 && !api.over(); n++) {
      const s = api.snapshot();
      const list = planner.placements(s.board, s.matrix, s.pos, rotate);
      assert.ok(list.length > 0);
      // Mostly sensible moves so the game lasts and the board gets varied; sometimes any
      // reachable one, so awkward kicks and edge columns are exercised too.
      const ranked = planner.rank(s.board, s.matrix, s.pos, rotate);
      const plan = pick() < 0.35 ? list[Math.floor(pick() * list.length)] : ranked[0];
      const token = api.token();
      assert.equal(api.perform(plan), true);
      const after = api.snapshot();
      assert.deepEqual(plain(after.board), plain(plan.board));
      assert.equal(after.lines, s.lines + plan.cleared);
      assert.notEqual(api.token(), token, 'a new piece is a new matrix');
      dropped++;
    }
  }
  assert.ok(dropped > 100);
});

test('placements are distinct and include both walls', () => {
  const { api } = game(7);
  const s = api.snapshot();
  const list = planner.placements(s.board, s.matrix, s.pos, (m, d) => api.rotateMatrix(m, d));
  assert.equal(new Set(list.map(c => c.key)).size, list.length);
  const xs = list.flatMap(c => c.cells.map(cell => cell[0]));
  assert.equal(Math.min(...xs), 0);
  assert.equal(Math.max(...xs), s.board[0].length - 1);
});

test('a placement is found again from where the piece is after it has fallen', () => {
  const { api, context } = game(9);
  const rotate = (m, d) => api.rotateMatrix(m, d);
  const s = api.snapshot();
  const target = planner.rank(s.board, s.matrix, s.pos, rotate)[0];
  vm.runInContext('playerDrop(); playerDrop();', context);       // gravity, twice
  const now = api.snapshot();
  assert.equal(now.pos.y, s.pos.y + 2);
  const again = planner.find(now.board, now.matrix, now.pos, rotate, target.key);
  assert.ok(again);
  assert.equal(api.perform(again), true);
  assert.deepEqual(plain(api.snapshot().board), plain(target.board));
});

test('holding gravity stops the game loop from dropping the piece, and releasing it resumes', () => {
  const { api, context } = game(5);
  const y0 = api.snapshot().pos.y;
  api.holdGravity(true);
  vm.runInContext('update(5000); update(10000); update(15000);', context);
  assert.equal(api.snapshot().pos.y, y0);
  api.holdGravity(false);
  vm.runInContext('update(20000);', context);
  assert.equal(api.snapshot().pos.y, y0 + 1);
});

test('the question is one choice over the candidates, and the state carries the board', () => {
  const { api } = game(3);
  const rotate = (m, d) => api.rotateMatrix(m, d);
  for (let n = 0; n < 25; n++) api.perform(planner.rank(api.snapshot().board, api.snapshot().matrix,
                                                        api.snapshot().pos, rotate)[0]);
  const s = api.snapshot();
  const cands = planner.rank(s.board, s.matrix, s.pos, rotate).slice(0, 4);
  const q = planner.question(cands);
  assert.equal(q.type, 'choice');
  assert.deepEqual(Object.keys(q.criteria), ['A', 'B', 'C', 'D']);
  const state = planner.stateFor(s.board, s.matrix, s.next, s);
  assert.equal(state.piece, planner.pieceName(s.matrix));
  assert.equal(state.board.split('\n').length, planner.boardText(s.board).length || 1);
});

test('the page loads the xDecision GGUF the Pages chat app lists', () => {
  const entry = models.models.find(m => m.repo === 'mccoysc/xDecision');
  assert.ok(entry);
  assert.match(aiSource, new RegExp("repo: '" + entry.repo + "'"));
  assert.match(aiSource, new RegExp("file: '" + entry.file.replace(/[.]/g, '\\.') + "'"));
});
