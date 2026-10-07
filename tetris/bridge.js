/* The one file that touches vanilla-js-tetris's internals.
 *
 * tetris.js is an ordinary script, not a module, and exports nothing. It does not need to:
 * its `function` declarations (playerRotate, playerMove, playerHardDrop, rotate, resetGame,
 * togglePause) are properties of the global object, and its top-level `let`/`const`
 * (board, player, score, lines, level, gameOver, isPaused, dropCounter) live in the global
 * lexical scope that every classic script on the page shares. A script loaded after it can
 * therefore read them and call them by name -- they are just not on `window`.
 *
 * So the AI plays the game through the game's own functions, the ones its keyboard handler
 * calls: no synthetic key events, no reading pixels back off the canvas. The game file is
 * vendored unmodified (vendor/tetris.js); everything it is asked to do goes through here.
 *
 * Load order: vendor/tetris.js, then this file.
 */
(function (root) {
  'use strict';

  // playerMove() writes a console line on every call (a debugging line left in upstream),
  // and the AI moves every piece several columns. Its moves are made with console.log muted
  // for the length of each call and nothing else.
  function quietly(fn) {
    const log = console.log;
    console.log = function () {};
    try { return fn(); } finally { console.log = log; }
  }

  function same(a, b) {
    return a.length === b.length && a.every((row, y) => row.every((v, x) => v === b[y][x]));
  }

  const game = {
    /** A copy of everything a decision reads. Mutating it changes nothing in the game. */
    snapshot() {
      return {
        board: board.map(r => r.slice()),
        matrix: player.matrix ? player.matrix.map(r => r.slice()) : null,
        pos: { x: player.pos.x, y: player.pos.y },
        next: player.next ? player.next.map(r => r.slice()) : null,
        score, lines, level, gameOver, paused: isPaused,
      };
    },

    /**
     * Which piece is in play, by identity: playerReset() replaces `player.matrix` with a new
     * array for every piece, while rotating turns the same array in place. A change in this
     * value is a new piece, and that is the whole of new-piece detection.
     */
    token() { return player.matrix; },
    over() { return gameOver; },
    paused() { return isPaused; },

    /** The game's own rotate(), for the planner to turn copies exactly as the game will. */
    rotateMatrix(m, dir) { rotate(m, dir); },

    /** Where playerReset() puts a new piece. */
    spawnOf(m) {
      return { x: Math.floor(board[0].length / 2) - Math.floor(m[0].length / 2), y: 0 };
    },

    // The three controls, exactly as the keyboard handler calls them.
    pressRotate() { playerRotate(); },
    pressMove(dir) { quietly(() => playerMove(dir)); },
    pressDrop() { playerHardDrop(); },

    /**
     * Hold the piece where it is while the model thinks. update() drops the piece once
     * `dropCounter` exceeds the level's interval; held at -Infinity it never does, and the
     * overlay togglePause() would paint stays off. playerHardDrop() sets it back to 0 itself,
     * so a played move releases it; `false` releases it without one.
     */
    holdGravity(on) { dropCounter = on ? -Infinity : 0; },

    reset() { resetGame(); },
    pause(on) { if (!gameOver && isPaused !== !!on) togglePause(); },

    /**
     * Play a planned placement from `planner.placements`: `rotations` presses of rotate,
     * sideways moves from `fromX` to `x`, then a hard drop. Each stage is checked against the
     * plan before the next, so a board that changed under the plan stops the move before
     * the drop instead of after it. Returns whether the piece was dropped.
     */
    perform(plan) {
      for (let i = 0; i < plan.rotations; i++) playerRotate();
      if (player.pos.x !== plan.fromX || !same(player.matrix, plan.matrix)) return false;
      const dir = Math.sign(plan.x - plan.fromX);
      while (player.pos.x !== plan.x) {
        const at = player.pos.x;
        quietly(() => playerMove(dir));
        if (player.pos.x === at) return false;
      }
      playerHardDrop();
      return true;
    },

    /**
     * One control press toward a plan, for watching it move: a rotation first, then one
     * column sideways, then the drop. Returns 'rotate' | 'move' | 'drop' | null (stuck).
     * The caller re-plans from the board as it is before every press.
     */
    step(plan) {
      if (plan.rotations > 0) { playerRotate(); return 'rotate'; }
      if (player.pos.x !== plan.x) {
        const at = player.pos.x;
        quietly(() => playerMove(Math.sign(plan.x - at)));
        return player.pos.x === at ? null : 'move';
      }
      playerHardDrop();
      return 'drop';
    },
  };

  root.TetrisGame = game;
}(typeof window !== 'undefined' ? window : globalThis));
