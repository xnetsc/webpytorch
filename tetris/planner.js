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

  /**
   * One option's text, in words and relative to the other options.
   *
   * How it is worded decides whether the model can use it at all. Measured with xDecision
   * Q8_0 on 240 decision points from real games, four options each, against the planner's
   * own ranking (random picks its first choice 25% of the time, mean regret 29.8):
   *
   *   "columns 3-5: clears no lines, makes 1 new hole, stack 8 high, bumpiness 10"
   *       -- the model picked the LAST option 218 times in 240; first choice 33%
   *   "clears nothing, covers 1 empty cell (new holes, bad), lands 2 rows higher than the
   *    lowest option, surface 3 bumpier than the flattest"
   *       -- picked about evenly by position; first choice 60%, regret 10.6
   *   ... plus ", 4 more ragged edges than the tidiest"
   *       -- first choice 63-65%, regret 8.0-8.7
   *   ... plus ", leaves a worse spot for the next piece"
   *       -- worse: first choice 38%, regret 21.1; not used
   *
   * So the numbers are said as comparisons ("lowest", "2 rows higher than"), and a hole is
   * said to be bad: the model weighs what it is told, it does not infer Tetris.
   */
  function describe(c, set) {
    set = set || [c];
    const f = c.features, parts = [];
    const low = Math.min(...set.map(o => o.features.landingHeight));
    const flat = Math.min(...set.map(o => o.features.bumpiness));
    parts.push(f.rowsCleared === 0 ? 'clears nothing'
      : 'clears ' + f.rowsCleared + (f.rowsCleared === 1 ? ' line' : ' lines'));
    parts.push(f.newHoles > 0
      ? 'covers ' + f.newHoles + ' empty ' + (f.newHoles === 1 ? 'cell' : 'cells') + ' (new holes, bad)'
      : 'covers no empty cells');
    const up = f.landingHeight - low;
    const rows = +up.toFixed(1);
    parts.push(up < 0.01 ? 'lands lowest'
      : 'lands ' + rows + (rows === 1 ? ' row' : ' rows') + ' higher than the lowest option');
    parts.push(f.bumpiness <= flat ? 'flattest surface'
      : 'surface ' + (f.bumpiness - flat) + ' bumpier than the flattest');
    const tidy = Math.min(...set.map(edges));
    parts.push(edges(c) === tidy ? 'tidiest edges'
      : (edges(c) - tidy) + ' more ragged edges than the tidiest');
    return parts.join(', ');
  }

  // Filled/empty boundaries along rows and down columns: every overhang, gap and wall the
  // placement leaves. The evaluation's two transition counts, said as one thing.
  function edges(c) { return c.features.rowTransitions + c.features.columnTransitions; }

  // Everything describe() says about an option: two options with the same signature read
  // word for word the same to the model.
  function signature(c) {
    const f = c.features;
    return [f.rowsCleared, f.newHoles, f.landingHeight, f.bumpiness, edges(c)].join();
  }

  /**
   * The options to put to the model: the best `k` of `ranked`, less any it cannot tell
   * apart. Two placements described identically are one choice as far as the model can
   * see, and offering both only splits its answer between them; the planner's preferred
   * one stays.
   */
  function candidates(ranked, k) {
    const out = [], seen = new Set();
    for (const c of ranked.slice(0, k)) {
      const sig = signature(c);
      if (seen.has(sig)) continue;
      seen.add(sig);
      out.push(c);
    }
    return out;
  }

  /**
   * The board as the model reads it: a grid, filled rows only, bottom row last.
   * `#` is a filled cell and `.` an empty one.
   */
  function boardText(board) {
    const rows = [];
    let top = board.findIndex(r => r.some(v => v !== 0));
    if (top < 0) top = board.length;
    for (let y = top; y < board.length; y++) {
      rows.push(board[y].map(v => (v ? '#' : '.')).join(''));
    }
    return rows;
  }

  function stateFor(board, matrix, next, stats) {
    const h = heights(board);
    return {
      game: 'Tetris',
      board: boardText(board).join('\n') || '(empty)',
      columns: board[0].length,
      rows: board.length,
      stack_height: Math.max(0, ...h),
      holes: holes(board),
      piece: pieceName(matrix),
      next_piece: next ? pieceName(next) : null,
      lines_cleared: stats ? stats.lines : undefined,
    };
  }

  // The question type names are the checkpoint's own (`surface().takes.questions.types`);
  // `choice` is the conventional name of the general, named-options shape.
  const INSTRUCTIONS = 'Where should the current Tetris piece be dropped? Clear lines when '
    + 'possible, never leave holes under the stack, and keep the stack low and flat.';

  /**
   * One question whose options are the candidates, labelled A, B, C… `type` is the model's
   * name for its named-options question type, read from its `surface()`.
   */
  function question(candidates, o) {
    o = o || {};
    const criteria = {};
    candidates.forEach((c, i) => {
      criteria[String.fromCharCode(65 + i)] = describe(c, candidates);
    });
    return { type: o.type || 'choice', instructions: o.instructions || INSTRUCTIONS, criteria };
  }

  return {
    NAMES, WEIGHTS, INSTRUCTIONS,
    pieceName, clone, collides, rotateLikeGame, settle, heights, holes, measure,
    placements, rank, find, candidates, describe, boardText, stateFor, question,
  };
}));
