// Play whole games headless and report how far each one got, for one way of choosing:
//
//   --who model    xDecision answers the page's question (needs --model, runs decide.py)
//   --who random   a uniform pick among the same options the model is shown
//   --who rules    the planner's first choice ("只用规则" on the page)
//
//   node tetris/eval/eval.mjs --who model --model path/to/xDecision-Q8_0.gguf
//   node tetris/eval/eval.mjs --who random --games 5 --max-pieces 500 --k 4
//
// Every game is the real vendored tetris.js (headless.mjs) and every move goes through the
// bridge's perform(), as on the page; the options come from planner.ask(), as on the page.
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { headlessGame, seeded } from './headless.mjs';

const require = createRequire(import.meta.url);
const P = require('../planner.js');

const { values: o } = parseArgs({ options: {
  who: { type: 'string', default: 'model' },
  model: { type: 'string' },
  games: { type: 'string', default: '5' },
  'first-seed': { type: 'string', default: '1' },
  'max-pieces': { type: 'string', default: '500' },
  k: { type: 'string', default: '4' },
} });
const games = +o.games, first = +o['first-seed'], cap = +o['max-pieces'], k = +o.k;

let decide = null, stop = () => {};
if (o.who === 'model') {
  if (!o.model) throw new Error('--who model needs --model <path to the GGUF>');
  const server = spawn('python3', [new URL('./decide.py', import.meta.url).pathname, o.model],
                       { stdio: ['pipe', 'pipe', 'inherit'] });
  const waiting = [];
  createInterface({ input: server.stdout }).on('line', (line) => waiting.shift()(JSON.parse(line)));
  await new Promise((resolve) => waiting.push(resolve));
  decide = (state, questions) => new Promise((resolve) => {
    waiting.push(resolve);
    server.stdin.write(JSON.stringify({ state, questions }) + '\n');
  });
  stop = () => server.kill();
}

const results = [];
for (let seed = first; seed < first + games; seed++) {
  const { api } = headlessGame(seed);
  const rotate = (m, dir) => api.rotateMatrix(m, dir);
  const pick = seeded(seed * 7 + 1);
  let pieces = 0, top = 0, ms = 0, asked = 0;
  while (!api.over() && pieces < cap) {
    const s = api.snapshot();
    const ranked = P.rank(s.board, s.matrix, s.pos, rotate, s.next, api.spawnOf);
    if (!ranked.length) break;
    const q = P.ask(s, ranked, k, 'choice');
    let chosen = ranked[0];
    if (o.who === 'random') chosen = q.shown[Math.floor(pick() * q.shown.length)];
    else if (o.who === 'model' && q.shown.length > 1) {
      const res = await decide(q.state, q.questions);
      if (res.error) throw new Error(res.error);
      chosen = P.picked(q, res.answers);
      ms += res.ms; asked++;
    }
    if (chosen === ranked[0]) top++;
    if (!api.perform(chosen)) throw new Error('the game did not take a planned move');
    pieces++;
  }
  const s = api.snapshot();
  const r = { seed, pieces, lines: s.lines, score: s.score, ended: s.gameOver,
              rulesFirst: +(top / Math.max(1, pieces)).toFixed(3),
              msPerDecision: asked ? Math.round(ms / asked) : null };
  results.push(r);
  console.log(JSON.stringify(r));
}
stop();
const mean = (key, digits = 1) =>
  +(results.reduce((t, r) => t + r[key], 0) / results.length).toFixed(digits);
console.log(JSON.stringify({ who: o.who, k, games, cap, lines: mean('lines'), pieces: mean('pieces'),
                             ended: results.filter(r => r.ended).length,
                             rulesFirst: mean('rulesFirst', 3) }));
