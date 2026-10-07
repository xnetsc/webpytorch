/* xDecision plays vanilla-js-tetris.
 *
 * Each new piece is one turn:
 *   1. read the game (bridge.js -> the game's own `board` and `player`),
 *   2. list every placement the game's controls can reach and what each leaves behind
 *      (planner.js),
 *   3. ask xDecision, through the webtorch SDK, one `choice` question whose options are the
 *      strongest of those placements -- the model's answer is the move,
 *   4. play it with the game's own controls (playerRotate / playerMove / playerHardDrop).
 *
 * The model is the one the Pages chat app lists (chat/models.json), loaded and asked the
 * way that app asks it: webtorch.start() -> wt.load(repo, {file}) -> wt.decide(state,
 * questions).
 */
(function () {
  'use strict';

  const G = window.TetrisGame;
  const P = window.TetrisPlanner;
  const $ = (s) => document.querySelector(s);

  // The xDecision entry of chat/models.json, which is where the Pages app takes it from.
  const MODEL = { repo: 'mccoysc/xDecision', file: 'models/gguf/xDecision-Q8_0.gguf',
                  size: 402546752 };

  // Where the bytes come from is this page's decision, not the SDK's (it installs no reader
  // of its own). xDecision is published on ModelScope; `modelscope_read()` tries both of
  // its origins, because .cn and .ai do not carry the same repos.
  const READERS = {
    modelscope: 'webtorch.set_io_read(webtorch.modelscope_read())',
    hf: 'webtorch.set_io_read(webtorch.hf_read())',
    // `?model=<url>`: the same GGUF from your own server (same origin, or one that sends
    // CORS headers), e.g. ?model=/models/gguf/xDecision-Q8_0.gguf for a local copy.
    url: 'webtorch.use_default_io()',
  };
  const ownURL = new URLSearchParams(location.search).get('model');

  const S = {
    wt: null,              // the SDK runtime, once started
    choiceType: null,      // the model's own name for its named-options question type
    on: false,             // AI playing
    busy: false,           // a turn in progress
    token: null,           // the piece the last turn was for (bridge `token()`)
    restartAt: 0,
    stats: { games: 0, pieces: 0, asked: 0, agree: 0, ms: 0, bestLines: 0, bestScore: 0 },
  };

  const opt = {
    mode: () => $('#mode').value,                       // 'model' | 'rules'
    k: () => Number($('#k').value),
    pace: () => $('#pace').value,                       // 'fast' | 'normal' | 'watch'
    restart: () => $('#restart').checked,
    hold: () => $('#hold').checked,
  };

  const say = (t) => { $('#status').textContent = t; };
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const rotate = (m, dir) => G.rotateMatrix(m, dir);

  // ---- the SDK ------------------------------------------------------------------------

  webtorch.onError((e) => { say(e.scope + ': ' + e.message); });

  const ready = webtorch.start({ baseURL: '../', onStatus: say }).then(async (wt) => {
    S.wt = wt;
    await wt.run('import webtorch\nwebtorch.set_io_write(webtorch.default_io_write)\n');
    say('运行时就绪（' + wt.backend + (wt.reason ? '：' + wt.reason : '') + '）。'
        + '点“加载 xDecision”下载模型，下载过的直接从浏览器缓存读取。');
    $('#load').disabled = false;
    return wt;
  }).catch((e) => { say('运行时启动失败：' + e.message); throw e; });

  async function loadModel() {
    $('#load').disabled = true;
    $('#source').disabled = true;
    try {
      const wt = await ready;
      const source = $('#source').value;
      await wt.run('import webtorch\n' + READERS[source] + '\n');
      const info = await wt.load(source === 'url' ? new URL(ownURL, location.href).href : MODEL.repo, {
        file: source === 'url' ? undefined : MODEL.file,
        onProgress: (p) => say('下载 xDecision：' + (p.bytes / 1e6).toFixed(0) + ' / '
                               + ((p.total || MODEL.size) / 1e6).toFixed(0) + ' MB'),
        onStage: (s) => say('加载中：' + s.stage + '…'),
      });
      if (info.kind !== 'decision') throw new Error('加载到的不是决策模型（kind=' + info.kind + '）');
      // The question types are the checkpoint's; the one we need is the one whose options
      // the caller names. Asked of the model rather than assumed.
      const types = ((info.surface || {}).takes || {}).questions || {};
      const named = Object.entries(types.types || {}).find(([, t]) => t.shape === 'named');
      if (!named) throw new Error('这个模型没有“从给定选项里选一个”的题型');
      S.choiceType = named[0];
      say('xDecision 已加载（' + S.wt.backend + '）。');
      $('#loadRow').hidden = true;
      $('#mode').value = 'model';
      syncButtons();
    } catch (e) {
      say('加载失败：' + e.message);
      $('#load').disabled = false;
      $('#source').disabled = false;
    }
  }

  // ---- one turn ---------------------------------------------------------------------

  async function turn() {
    const snap = G.snapshot();
    const ranked = P.rank(snap.board, snap.matrix, snap.pos, rotate, snap.next, G.spawnOf);
    if (!ranked.length) return;
    const useModel = opt.mode() === 'model' && S.choiceType;
    const asked = P.ask(snap, ranked, Math.max(1, opt.k()), S.choiceType);
    const shown = asked.shown;
    let chosen = ranked[0], answer = null;
    if (opt.hold()) G.holdGravity(true);
    if (useModel && shown.length > 1) {
      const t0 = performance.now();
      const res = await S.wt.decide(asked.state, asked.questions);
      const ms = performance.now() - t0;
      chosen = P.picked(asked, res.answers);
      answer = { labels: Object.keys(asked.questions.move.criteria),
                 probabilities: res.answers.move.probabilities, ms,
                 tokens: (res.usage || {}).input_tokens };
      S.stats.asked++;
      S.stats.ms += ms;
      if (chosen === ranked[0]) S.stats.agree++;
    }
    // Switched off, or the piece was dropped by a person or by gravity while the model was
    // thinking: this answer is for a piece that is no longer in play.
    if (!S.on || G.token() !== S.token) { G.holdGravity(false); return; }
    showTurn(snap, shown, chosen, ranked[0], answer, useModel);
    const pace = opt.pace();
    if (pace !== 'fast') { overlay(shown, chosen); await sleep(pace === 'watch' ? 450 : 250); }
    if (!S.on || G.token() !== S.token) { overlay(null); G.holdGravity(false); return; }
    await play(chosen.key, pace === 'watch');
    overlay(null);
    S.stats.pieces++;
    showStats();
  }

  // Play the chosen placement with the game's controls. It is looked up again from the
  // piece as it is NOW (same cells, fresh path), so a piece that fell a row while the model
  // thought still lands where the model chose; one that can no longer get there takes the
  // planner's best reachable spot instead.
  async function play(key, watch) {
    for (let presses = 0; presses < 24; presses++) {
      const s = G.snapshot();
      const plan = P.find(s.board, s.matrix, s.pos, rotate, key)
        || P.rank(s.board, s.matrix, s.pos, rotate)[0];
      if (!plan) break;
      if (!watch) { if (G.perform(plan)) return; continue; }
      const did = G.step(plan);
      if (did === 'drop') return;
      await sleep(90);
      if (!S.on) { G.holdGravity(false); return; }
    }
    G.pressDrop();
  }

  // ---- the loop -------------------------------------------------------------------------

  function frame(now) {
    requestAnimationFrame(frame);
    if (!S.on || S.busy) return;
    if (G.over()) {
      if (!opt.restart()) return;
      if (!S.restartAt) { S.restartAt = now + 1500; endGame(); return; }
      if (now < S.restartAt) return;
      S.restartAt = 0;
      S.stats.games++;
      G.reset();
      return;
    }
    if (G.paused()) return;
    const token = G.token();
    if (token === S.token) return;
    S.token = token;
    S.busy = true;
    turn().catch((e) => {
      say('这一步出错：' + e.message);
      G.holdGravity(false);
      setRunning(false);
    }).finally(() => { S.busy = false; });
  }

  function endGame() {
    const s = G.snapshot();
    S.stats.bestLines = Math.max(S.stats.bestLines, s.lines);
    S.stats.bestScore = Math.max(S.stats.bestScore, s.score);
    showStats();
  }

  function setRunning(on) {
    S.on = on;
    S.restartAt = 0;
    if (on) {
      if (G.over()) { G.reset(); S.stats.games++; }
      G.pause(false);
      S.token = null;
    } else {
      G.holdGravity(false);
      overlay(null);
    }
    syncButtons();
  }

  function syncButtons() {
    const modelReady = !!S.choiceType;
    $('#ai').disabled = opt.mode() === 'model' && !modelReady;
    $('#ai').textContent = S.on ? '停止 AI' : (opt.mode() === 'model' ? '让 xDecision 来玩' : '让规则来玩（对照）');
  }

  // ---- what the page shows ----------------------------------------------------------------

  const pct = (p) => (p * 100).toFixed(1) + '%';

  // The same facts as the option text the model reads (planner.describe), said for people.
  function summary(c, set) {
    const f = c.features;
    const low = Math.min(...set.map(o => o.features.landingHeight));
    const flat = Math.min(...set.map(o => o.features.bumpiness));
    const edge = (o) => o.features.rowTransitions + o.features.columnTransitions;
    const tidy = Math.min(...set.map(edge));
    const xs = c.cells.map(cell => cell[0] + 1);
    const lo = Math.min(...xs), hi = Math.max(...xs);
    return (lo === hi ? '第 ' + lo + ' 列' : '第 ' + lo + '–' + hi + ' 列') + '：'
      + [f.rowsCleared ? '消 ' + f.rowsCleared + ' 行' : '不消行',
         f.newHoles > 0 ? '新增 ' + f.newHoles + ' 个洞' : '不留洞',
         f.landingHeight - low < 0.01 ? '落点最低' : '比最低高 ' + +(f.landingHeight - low).toFixed(1) + ' 行',
         f.bumpiness <= flat ? '表面最平' : '不平度 +' + (f.bumpiness - flat),
         edge(c) <= tidy ? '边缘最整齐' : '参差 +' + (edge(c) - tidy)].join(' · ');
  }

  function showTurn(snap, shown, chosen, top, answer, useModel) {
    const box = $('#turn');
    box.textContent = '';
    const head = document.createElement('p');
    head.className = 'turnhead';
    head.textContent = '当前方块 ' + P.pieceName(snap.matrix) + '，下一个 '
      + (snap.next ? P.pieceName(snap.next) : '?') + ' · '
      + (useModel
        ? (answer ? 'xDecision ' + answer.ms.toFixed(0) + ' ms · 读了 ' + answer.tokens + ' 个 token'
                  : '前几名在模型看来都一样，只剩一个，不用问')
        : '规则直接取第一名（未调用模型）');
    box.appendChild(head);
    const list = document.createElement('ol');
    list.className = 'cands';
    shown.forEach((c, i) => {
      const li = document.createElement('li');
      if (c === chosen) li.className = 'chosen';
      const label = answer ? answer.labels[i] : String.fromCharCode(65 + i);
      const p = answer ? answer.probabilities[label] : null;
      const bar = document.createElement('span');
      bar.className = 'bar';
      bar.style.width = p == null ? '0' : (p * 100).toFixed(1) + '%';
      const text = document.createElement('span');
      text.className = 'ctext';
      text.textContent = label + '  ' + summary(c, shown) + (c === top ? '  ⚑规则首选' : '');
      // What the model actually read for this option, word for word.
      text.title = P.describe(c, shown);
      const prob = document.createElement('span');
      prob.className = 'prob';
      prob.textContent = p == null ? '' : pct(p);
      li.append(bar, text, prob);
      list.appendChild(li);
    });
    box.appendChild(list);
  }

  function showStats() {
    const s = G.snapshot(), st = S.stats;
    st.bestLines = Math.max(st.bestLines, s.lines);
    st.bestScore = Math.max(st.bestScore, s.score);
    $('#stats').textContent = [
      'AI 放下 ' + st.pieces + ' 块',
      '第 ' + (st.games + 1) + ' 局',
      '最多消行 ' + st.bestLines,
      '最高分 ' + st.bestScore,
      st.asked ? 'xDecision 平均 ' + (st.ms / st.asked).toFixed(0) + ' ms/步' : null,
      st.asked ? '与规则首选一致 ' + pct(st.agree / st.asked) : null,
    ].filter(Boolean).join(' · ');
  }

  // The candidates drawn over the board while a decision is shown: an outline per option,
  // the model's pick filled. A separate canvas on top of the game's, so the game's own
  // draw() -- which clears its canvas every frame -- is left alone.
  function overlay(shown, chosen) {
    const cv = $('#overlay'), ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!shown) return;
    const cell = cv.width / 10;
    shown.forEach((c, i) => {
      const mine = c === chosen;
      ctx.lineWidth = mine ? 3 : 1.5;
      ctx.strokeStyle = mine ? '#ffffff' : 'rgba(255,255,255,0.55)';
      ctx.fillStyle = 'rgba(255,255,255,0.28)';
      c.cells.forEach(([x, y]) => {
        if (mine) ctx.fillRect(x * cell, y * cell, cell, cell);
        ctx.strokeRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
      });
      const cx = c.cells.reduce((t, p) => t + p[0], 0) / c.cells.length;
      const cy = c.cells.reduce((t, p) => t + p[1], 0) / c.cells.length;
      ctx.fillStyle = mine ? '#ffe138' : '#ffffff';
      ctx.font = 'bold 14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String.fromCharCode(65 + i), (cx + 0.5) * cell, (cy + 0.5) * cell);
    });
  }

  // ---- wiring ---------------------------------------------------------------------------

  if (ownURL) {
    const o = document.createElement('option');
    o.value = 'url';
    o.textContent = '网址：' + ownURL;
    $('#source').appendChild(o);
    $('#source').value = 'url';
  }
  $('#load').onclick = loadModel;
  $('#ai').onclick = () => setRunning(!S.on);
  $('#mode').onchange = () => { if (S.on && opt.mode() === 'model' && !S.choiceType) setRunning(false); syncButtons(); };
  $('#k').oninput = () => { $('#kval').textContent = $('#k').value; };
  $('#hold').onchange = () => { if (!opt.hold()) G.holdGravity(false); };

  // tetris.js starts a game on `load`. Listeners run in the order they were added and this
  // one was added after the game's, so the game exists here; it waits paused until someone
  // presses Start (the game's own button) or hands it to the AI.
  window.addEventListener('load', () => {
    G.pause(true);
    syncButtons();
    showStats();
    requestAnimationFrame(frame);
  });
}());
