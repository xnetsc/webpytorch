// vanilla-js-tetris without a screen: the vendored game file and bridge.js, run as two
// classic scripts in one vm realm -- the way the page runs them, so the bridge reaches the
// game's globals exactly as it does there -- with a stub canvas, no animation frames (so no
// gravity unless update() is called), and a seeded Math.random for repeatable piece orders.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const tetrisSource = readFileSync(new URL('../vendor/tetris.js', import.meta.url), 'utf8');
const bridgeSource = readFileSync(new URL('../bridge.js', import.meta.url), 'utf8');

/** mulberry32: a small seeded generator in [0, 1). */
export function seeded(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/** A started game: `api` is bridge.js's TetrisGame, `context` the realm it runs in. */
export function headlessGame(seed) {
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
