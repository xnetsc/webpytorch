// The Tetris AI plays vanilla-js-tetris through the game's own functions. These tests run the
// vendored game file itself, unmodified, in a vm with a stub canvas, and hold the planner to
// what the game actually does: every planned placement, pressed through bridge.js, has to
// land on exactly the board the planner predicted.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';
import { headlessGame, seeded } from '../tetris/eval/headless.mjs';

const require = createRequire(import.meta.url);
const planner = require('../tetris/planner.js');
const aiSource = await readFile(new URL('../tetris/ai.js', import.meta.url), 'utf8');
const models = JSON.parse(await readFile(new URL('../chat/models.json', import.meta.url), 'utf8'));

// Arrays made inside the vm have that realm's Array prototype, which strict deep equality
// counts as a difference; compare their contents.
const plain = (value) => JSON.parse(JSON.stringify(value));

const game = headlessGame;

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

test('a turn puts the deciding facts in the state and the placements as options', () => {
  const { api } = game(3);
  const rotate = (m, d) => api.rotateMatrix(m, d);
  for (let n = 0; n < 25; n++) {
    const s = api.snapshot();
    api.perform(planner.rank(s.board, s.matrix, s.pos, rotate)[0]);
  }
  const s = api.snapshot();
  const ranked = planner.rank(s.board, s.matrix, s.pos, rotate, s.next, api.spawnOf);
  const asked = planner.ask(s, ranked, 4, 'choice');
  const q = asked.questions.move;
  assert.deepEqual(Object.keys(asked.questions), ['move']);
  assert.equal(q.type, 'choice');
  assert.equal(q.instructions, planner.INSTRUCTIONS);
  assert.ok(asked.shown.includes(ranked[0]), 'the planner\'s first choice is always offered');
  assert.ok(asked.shown.includes(asked.expected));
  // Options are the candidates' ids, A B C D, and nothing else.
  assert.deepEqual(asked.labels, asked.shown.map((_, i) => String.fromCharCode(65 + i)));
  assert.deepEqual(Object.keys(q.criteria), asked.labels);
  assert.ok(Object.values(q.criteria).every(v => v === null));
  // The state holds every candidate and what it does: everything the rule needs.
  assert.deepEqual(Object.keys(asked.state), ['game', 'piece', 'placements']);
  assert.equal(asked.state.piece, planner.pieceName(s.matrix));
  assert.deepEqual(asked.state.placements,
                   asked.shown.map((c, i) => ({ id: asked.labels[i], result: planner.facts(c) })));
  const results = asked.state.placements.map(p => p.result);
  assert.equal(new Set(results).size, results.length, 'one placement per kind of facts');
  const at = results.map(t => planner.ORDER.indexOf(t));
  assert.deepEqual(at, at.slice().sort((a, b) => a - b), 'listed in the verified order');
  const last = asked.labels.at(-1);
  assert.equal(planner.picked(asked, { move: { choice: last } }), asked.shown.at(-1));
});

test('the rule in code: hole-free line clear, then hole-free, then a line clear, then the rest', () => {
  const opt = (rowsCleared, newHoles) => ({ features: { rowsCleared, newHoles } });
  const first = opt(0, 2), clean = opt(0, 0), cleanLine = opt(1, 0), holedLine = opt(2, 1);
  assert.equal(planner.expected([first, clean, cleanLine, holedLine]), cleanLine);
  assert.equal(planner.expected([first, holedLine, clean]), clean);
  assert.equal(planner.expected([first, holedLine]), holedLine);
  assert.equal(planner.expected([first]), first);
  assert.deepEqual(planner.arrange([first, clean, cleanLine, holedLine]).map(planner.facts),
                   planner.ORDER);
  assert.equal(planner.facts(cleanLine), 'clears a line, hole-free');
  assert.equal(planner.facts(holedLine), 'clears a line, makes a new hole');
  assert.equal(planner.facts(clean), 'hole-free');
  assert.equal(planner.facts(first), 'makes a new hole');
});

test('the page loads the xDecision GGUF the Pages chat app lists', () => {
  const entry = models.models.find(m => m.repo === 'mccoysc/xDecision');
  assert.ok(entry);
  assert.match(aiSource, new RegExp("repo: '" + entry.repo + "'"));
  assert.match(aiSource, new RegExp("file: '" + entry.file.replace(/[.]/g, '\\.') + "'"));
});
