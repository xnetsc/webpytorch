// Every input the page can send the model, asked. Candidates are one per kind of facts, under
// the letters A-D, so a question is one of 11 sets of two or more kinds for one of 7 pieces:
// 77 inputs in all. Each is built the way planner.ask() builds it and checked against the
// answer worked out in code from the same principles (planner.expected).
//
//   node tetris/eval/table.mjs path/to/xDecision-Q8_0.gguf
//
// Prints one line per input it got wrong and a total. The page relies on all 77 being right.
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';

const require = createRequire(import.meta.url);
const P = require('../planner.js');

const model = process.argv[2];
if (!model) throw new Error('usage: node tetris/eval/table.mjs path/to/xDecision-Q8_0.gguf');
const server = spawn('python3', [new URL('./decide.py', import.meta.url).pathname, model],
                     { stdio: ['pipe', 'pipe', 'inherit'] });
const waiting = [];
createInterface({ input: server.stdout }).on('line', (line) => waiting.shift()(JSON.parse(line)));
await new Promise((resolve) => waiting.push(resolve));
const decide = (state, questions) => new Promise((resolve) => {
  waiting.push(resolve);
  server.stdin.write(JSON.stringify({ state, questions }) + '\n');
});

// One stand-in candidate per kind: the facts the tags are made from.
const KINDS = [[1, 0], [0, 0], [1, 1], [0, 1]].map(([rowsCleared, newHoles]) =>
  ({ features: { rowsCleared, newHoles } }));

let right = 0, asked = 0;
for (const piece of P.NAMES) {
  for (let mask = 0; mask < 16; mask++) {
    const options = KINDS.filter((_, i) => mask & (1 << i));
    if (options.length < 2) continue;
    const shown = P.arrange(options);
    const labels = P.ids(shown);
    const state = P.stateOf(piece, shown, labels);
    const res = await decide(state, { move: P.question(labels, { type: 'choice' }) });
    if (res.error) throw new Error(res.error);
    const got = P.picked({ shown, labels }, res.answers), want = P.expected(options);
    asked++;
    if (got === want) right++;
    else {
      console.log('wrong: ' + JSON.stringify(state) + ' -> ' + res.answers.move.choice
                  + ', rule says ' + labels[shown.indexOf(want)]);
    }
  }
}
server.kill();
console.log(JSON.stringify({ asked, right }));
