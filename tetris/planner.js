/* Where the current piece can go, what each spot leaves behind, and how to say it to a
 * decision model.
 *
 * Nothing here moves a piece. It reads a copy of the game's board and piece, lists every
 * placement the game's own controls can reach -- rotate presses, then sideways moves, then a
 * hard drop, the exact order `bridge.js` will press them in -- and describes what each one
 * leaves behind. The rotation itself is the game's `rotate()`, handed in by the bridge, so
 * a matrix here turns exactly as the piece on screen will.
 *
 * Collision and the rotate wall-kick copy tetris.js line for line rather than approximating
 * it. They have to: a plan that assumes a kick the game does not make lands somewhere else,
 * and `test/tetris_planner.test.mjs` plays planned moves through the real game to hold that.
 *
 * Loads as a plain script (`window.TetrisPlanner`) or as a CommonJS module, for the tests.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TetrisPlanner = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // tetris.js fills every cell of a piece with its colour index, 1..7, in this order
  // (its `colors` and `pieces` tables). There is no other record of which piece it is.
  const NAMES = 'IJLOSTZ';

  function pieceName(matrix) {
    for (const row of matrix || []) for (const v of row) if (v) return NAMES[v - 1] || '?';
    return '?';
  }

  function clone(m) { return m.map(r => r.slice()); }

  // tetris.js `checkCollision`, for any matrix and position rather than the player's.
  function collides(board, m, px, py) {
    for (let y = 0; y < m.length; ++y) {
      for (let x = 0; x < m[y].length; ++x) {
        if (m[y][x] !== 0) {
          const row = board[y + py];
          if (row === undefined || row[x + px] === undefined || row[x + px] !== 0) return true;
        }
      }
    }
    return false;
  }

  // tetris.js `playerRotate`: turn, then try shifting sideways (+1, -1, +2) until it fits,
  // and undo the turn if none does. Mutates `m` and `pos` as the game mutates the player.
  function rotateLikeGame(board, m, pos, rotate) {
    const x0 = pos.x;
    let offset = 1;
    rotate(m);
    while (collides(board, m, pos.x, pos.y)) {
      pos.x += offset;
      offset = -(offset + (offset > 0 ? 1 : -1));
      if (offset > m.length) {
        rotate(m, -1);
        pos.x = x0;
        return false;
      }
    }
    return true;
  }

  // tetris.js `merge` then `clearLines`, on a copy.
  function settle(board, m, px, py) {
    const out = clone(board);
    const cells = [];
    m.forEach((row, y) => row.forEach((v, x) => {
      if (v !== 0) { out[y + py][x + px] = v; cells.push([x + px, y + py]); }
    }));
    let cleared = 0;
    for (let y = out.length - 1; y >= 0; --y) {
      if (out[y].every(v => v !== 0)) {
        out.splice(y, 1);
        out.unshift(new Array(board[0].length).fill(0));
        ++y;
        ++cleared;
      }
    }
    return { board: out, cells, cleared };
  }

  function heights(board) {
    const H = board.length, W = board[0].length, h = new Array(W).fill(0);
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) if (board[y][x] !== 0) { h[x] = H - y; break; }
    }
    return h;
  }

  function holes(board) {
    const H = board.length, W = board[0].length;
    let n = 0;
    for (let x = 0; x < W; x++) {
      let roof = false;
      for (let y = 0; y < H; y++) {
        if (board[y][x] !== 0) roof = true;
        else if (roof) n++;
      }
    }
    return n;
  }

  // The board measures Dellacherie's evaluation uses, with the weights El-Tetris fitted for
  // them (Islam El-Ashi, 2011). They are not tuned for this game here; they are the
  // published values, used to put the candidates the model chooses between in order.
  const WEIGHTS = {
    landingHeight: -4.500158825082766,
    rowsCleared: 3.4181268101392694,
    rowTransitions: -3.2178882868487753,
    columnTransitions: -9.348695305445199,
    holes: -7.899265427351652,
    wells: -3.3855972247263626,
  };

  function measure(before, placed) {
    const b = placed.board, H = b.length, W = b[0].length;
    let rowT = 0, colT = 0, wells = 0;
    for (let y = 0; y < H; y++) {
      let prev = 1;                                // the walls count as filled
      for (let x = 0; x < W; x++) {
        const f = b[y][x] !== 0 ? 1 : 0;
        if (f !== prev) rowT++;
        prev = f;
      }
      if (prev === 0) rowT++;
    }
    for (let x = 0; x < W; x++) {
      let prev = 0;                                // open sky above, the floor below
      for (let y = 0; y < H; y++) {
        const f = b[y][x] !== 0 ? 1 : 0;
        if (f !== prev) colT++;
        prev = f;
      }
      if (prev === 0) colT++;
      let depth = 0;
      for (let y = 0; y < H; y++) {
        const left = x === 0 || b[y][x - 1] !== 0;
        const right = x === W - 1 || b[y][x + 1] !== 0;
        if (b[y][x] === 0 && left && right) { depth++; wells += depth; }
        else depth = 0;
      }
    }
    const ys = placed.cells.map(c => c[1]);
    const h = heights(b);
    let bump = 0;
    for (let x = 0; x + 1 < W; x++) bump += Math.abs(h[x] - h[x + 1]);
    const holesAfter = holes(b);
    const f = {
      rowsCleared: placed.cleared,
      landingHeight: H - (Math.min(...ys) + Math.max(...ys)) / 2,
      rowTransitions: rowT,
      columnTransitions: colT,
      holes: holesAfter,
      wells,
      newHoles: holesAfter - holes(before),
      maxHeight: Math.max(...h),
      bumpiness: bump,
    };
    f.score = Object.keys(WEIGHTS).reduce((s, k) => s + WEIGHTS[k] * f[k], 0);
    return f;
  }

  /**
   * Every distinct placement reachable from where the piece is now, by the game's controls
   * in the order the bridge presses them: `rotations` presses of rotate, sideways from
   * `fromX` to `x`, then a hard drop to row `y`. Two paths that end on the same cells are
   * one placement; the one with fewer presses is kept.
   */
  function placements(board, matrix, pos, rotate) {
    const out = [];
    const seen = new Set();
    const m = clone(matrix);
    const p = { x: pos.x, y: pos.y };
    for (let r = 0; r < 4; r++) {
      if (r > 0) rotateLikeGame(board, m, p, rotate);
      if (collides(board, m, p.x, p.y)) continue;
      let lo = p.x, hi = p.x;
      while (!collides(board, m, lo - 1, p.y)) lo--;
      while (!collides(board, m, hi + 1, p.y)) hi++;
      for (let x = lo; x <= hi; x++) {
        let y = p.y;
        while (!collides(board, m, x, y + 1)) y++;
        const placed = settle(board, m, x, y);
        const key = placed.cells.map(c => c[0] + ',' + c[1]).sort().join(' ');
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ rotations: r, fromX: p.x, x, y, matrix: clone(m), key,
                   cells: placed.cells, board: placed.board, cleared: placed.cleared,
                   features: measure(board, placed) });
      }
    }
    return out;
  }

  /**
   * Placements best first. With `next` (the game's preview piece) each one is also scored
   * by the best spot it leaves for that piece -- a one-piece lookahead, which is what keeps
   * a placement from looking good only because it ignores what comes next.
   */
  function rank(board, matrix, pos, rotate, next, spawnOf) {
    const list = placements(board, matrix, pos, rotate);
    for (const c of list) {
      c.value = c.features.score;
      if (next && spawnOf) {
        const follow = placements(c.board, next, spawnOf(next), rotate);
        c.followUp = follow.length ? Math.max(...follow.map(f => f.features.score)) : -1e9;
        c.value += c.followUp;
      }
    }
    return list.sort((a, b) => b.value - a.value);
  }

  // The same placement found again after the board moved on (gravity, or a person pressing
  // a key while the model was thinking): same cells, reached from where the piece is now.
  function find(board, matrix, pos, rotate, key) {
    return placements(board, matrix, pos, rotate).find(c => c.key === key) || null;
  }

  // ---- saying it to the model -------------------------------------------------------------
  //
  // xDecision answers from facts it is given; it does not reason, so it is used as a cheap
  // if-else. Each option carries a few fact tags, the question names the tags wanted, and
  // the model picks the option whose tags match. What it can and cannot do was measured with
  // xDecision Q8_0 on 240 positions from real games (tetris/README.md has the table):
  //
  //   picks the option whose text says what the question asks for     99-100% right
  //   "none of them" when nothing matches                              3% right
  //   true/false about "option B" whose facts are in the state         "true" every time
  //   sorting a fact sentence into categories that differ by a "no"    51% right
  //   weighing several relative facts ("lands lowest, flattest…")      planner's pick 63%
  //
  // So: positive tags, no "none" option, no option letters to look up, and no two options
  // with the same tags -- the rule then has exactly one best match.

  const LINE = 'clears a line', CLEAN = 'hole-free', HOLE = 'makes a new hole';

  /** An option's text: its fact tags. "clears a line" appears only when it does. */
  function facts(c) {
    const f = c.features;
    return (f.rowsCleared > 0 ? LINE + ', ' : '') + (f.newHoles > 0 ? HOLE : CLEAN);
  }

  /** The rule, in the tags' own words. */
  const INSTRUCTIONS = 'Choose the option that clears a line and is hole-free.';

  // What the rule prefers, best first: a hole-free line clear, then hole-free, then a line
  // clear that leaves a hole, then the rest.
  const PREFERENCE = [LINE + ', ' + CLEAN, CLEAN, LINE + ', ' + HOLE, HOLE];

  // The order options are listed in. There are four kinds of tags, so everything the model
  // can be asked is 11 sets of kinds x 7 pieces = 77 inputs, and all of them were tried
  // (tetris/eval/table.mjs). Listed left to right on the board it answered 364 of 420
  // orderings right; in this order it answers all 77 right, with the right answer first,
  // second or third depending on the set -- so it is the tags it reads, not a position.
  const ORDER = [LINE + ', ' + HOLE, CLEAN, LINE + ', ' + CLEAN, HOLE];

  /**
   * The options to put to the model: from the planner's best `k`, the best of each kind of
   * fact tags, in the planner's order. Options with the same tags read the same to the
   * model, so only the planner's preferred one of them is offered, and the model chooses
   * between kinds.
   */
  function candidates(ranked, k) {
    const out = [], seen = new Set();
    for (const c of ranked.slice(0, k)) {
      const tags = facts(c);
      if (seen.has(tags)) continue;
      seen.add(tags);
      out.push(c);
    }
    return out;
  }

  /** The options as they are listed to the model (label A first). */
  function arrange(options) {
    return options.slice().sort((a, b) => ORDER.indexOf(facts(a)) - ORDER.indexOf(facts(b)));
  }

  /**
   * The same rule evaluated in code: the answer the model should give. With the conditions
   * evaluated here, this policy did not lose a game in 5 x 500 pieces (tetris/eval).
   */
  function expected(options) {
    for (const tags of PREFERENCE) {
      const c = options.find(o => facts(o) === tags);
      if (c) return c;
    }
    return null;
  }

  /**
   * One question whose options are the candidates' fact tags, labelled A, B, C… `type` is
   * the model's name for its named-options question type, read from its `surface()`.
   */
  function question(shown, o) {
    o = o || {};
    const criteria = {};
    shown.forEach((c, i) => { criteria[String.fromCharCode(65 + i)] = facts(c); });
    return { type: o.type || 'choice', instructions: o.instructions || INSTRUCTIONS, criteria };
  }

  /**
   * Everything one turn puts to the model, from a bridge snapshot and the ranked placements:
   * the options as listed (`shown[i]` is label A+i), the answer the rule gives (`expected`), and
   * the `state` and `questions` for decide(). The page and tetris/eval both ask through
   * this, so what is measured is what the page asks.
   */
  function ask(snap, ranked, k, type) {
    const options = candidates(ranked, k);
    const shown = arrange(options);
    return {
      shown,
      expected: expected(options),
      state: stateOf(pieceName(snap.matrix)),
      questions: { move: question(shown, { type }) },
    };
  }

  /** What the model is told besides the options: only which piece it is. */
  function stateOf(piece) {
    return { game: 'Tetris', piece };
  }

  /** The option a decide() answer picked, from what `ask` returned. */
  function picked(asked, answers) {
    const labels = Object.keys(asked.questions.move.criteria);
    return asked.shown[labels.indexOf(answers.move.choice)];
  }

  return {
    NAMES, WEIGHTS, INSTRUCTIONS,
    pieceName, clone, collides, rotateLikeGame, settle, heights, holes, measure,
    PREFERENCE, ORDER,
    placements, rank, find, facts, candidates, arrange, expected, question, stateOf,
    ask, picked,
  };
}));
