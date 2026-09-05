/* Chat UI: model selection (any ModelScope model), load/release with status, cache
   management, camera/file/URL tools, and zip export/import of the conversation. */
const $ = (s) => document.querySelector(s);

// Opened as a file rather than served. Nothing here can work: a file:// page cannot fetch
// its own modules, cannot start a worker on some browsers, and can never be cross-origin
// isolated, so there is no SharedArrayBuffer and no GPU backend. Say that once, plainly,
// with the command that fixes it -- the alternative is a page that looks alive and fails at
// every step with opaque CORS errors.
if (window.__coiFileMode) {
  document.body.innerHTML =
    '<div style="max-width:44rem;margin:12vh auto;padding:0 1.5rem;font:15px/1.6 system-ui,sans-serif">' +
    '<h2 style="margin:0 0 .6rem">This page has to be served over HTTP</h2>' +
    '<p style="margin:0 0 1rem;opacity:.8">Opening it straight from disk (<code>' +
    location.protocol + '//</code>) leaves the browser unable to load the runtime, and ' +
    'unable to grant the shared memory the GPU backend needs.</p>' +
    '<p style="margin:0 0 .4rem">From the folder that contains <code>chat/</code>:</p>' +
    '<pre style="background:#1113;padding:.8rem 1rem;border-radius:6px;overflow:auto">' +
    'node webtorch/serve-coi.mjs webtorch 8119</pre>' +
    '<p style="margin:.2rem 0 1rem;opacity:.8">then open <code>http://localhost:8119/chat/</code>. ' +
    'That server sends the two headers the runtime needs. A plain static server works too — ' +
    'the page falls back to a service worker that adds them.</p>' +
    '</div>';
  throw new Error('webtorch chat: must be served over HTTP, not opened from ' + location.protocol);
}

const worker = new Worker('worker.js?v=e8a0b03202');
// One SDK call brings up the GPU backend's main-thread half. Until it resolves the worker
// must not be spoken to, so `call` waits on it.
// `?backend=webgl` (or `webgpu`, or `cpu`) pins the order, for reproducing a report on the
// backend the reporter actually had. Without it, the best available wins.
const BACKEND_ORDER = (() => {
  const want = new URLSearchParams(location.search).get('backend');
  if (want === 'webgl') return ['webgl'];
  if (want === 'webgpu') return ['webgpu'];
  if (want === 'cpu') return [];
  return ['webgpu', 'webgl'];
})();
const gpuInit = webtorch.initMain(worker, { backendOrder: BACKEND_ORDER })
  .then(r => r.backend, () => 'cpu');
let seq = 0; const pending = new Map();
// Conversations live in IndexedDB so the sidebar survives a reload.
// Each: {id, title, messages:[{role, content, attachments:[{kind,name,text,dataUrl}]}], updated}
//
// NOT localStorage, which is where they used to be: it holds about 5 MB per origin, stores
// only strings, and every write is synchronous on the UI thread. One conversation with a
// couple of pasted images (attachments are data URLs) is enough to hit that ceiling, and a
// write that fails there fails silently -- the reply is on screen and gone after a reload.
// IndexedDB has room, takes structured values as they are, and writes off the main thread.
const STORE = 'webtorch-chat-convs';          // the old localStorage key, read once to migrate
const DB_NAME = 'webtorch-chat', DB_STORE = 'convs';
let convs = [];
let curId = null;
let attachments = [];
let modelLoaded = false;
// Does the loaded model see images? Decides whether camera / image attachments are offered —
// a text-only model can do nothing with pixels, so the controls say so instead of failing.
let modelImage = false;
// The reply currently being streamed, if any: which message it is and the DOM pieces being
// updated in place (see fillBody). Only one at a time — the worker decodes one token at a
// time anyway.
let streaming = null;

let dbPromise = null;
function db() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((res, rej) => {
    const rq = indexedDB.open(DB_NAME, 1);
    rq.onupgradeneeded = () => {
      const d = rq.result;
      if (!d.objectStoreNames.contains(DB_STORE)) d.createObjectStore(DB_STORE, { keyPath: 'id' });
    };
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error);
  });
  return dbPromise;
}
function tx(mode, fn) {
  return db().then(d => new Promise((res, rej) => {
    const t = d.transaction(DB_STORE, mode);
    const out = fn(t.objectStore(DB_STORE));
    t.oncomplete = () => res(out && out.result !== undefined ? out.result : out);
    t.onerror = () => rej(t.error);
    t.onabort = () => rej(t.error);
  }));
}

// Until the store has been read once, `convs` is not the truth and must not be written back
// as if it were -- `saveConvs` deletes every record that is not in it, so one early write
// would empty the store. The window is milliseconds, and the cost of losing it is every
// conversation the person has.
let convsReady = false;

async function loadConvs() {
  try {
    convs = await tx('readonly', st => st.getAll());
  } catch (e) { convs = []; }
  if (!Array.isArray(convs)) convs = [];
  // One-time move of anything the localStorage version left behind. The old key is cleared
  // only after the new store has the records, so an interrupted migration retries rather
  // than losing the conversations.
  let old = null;
  try { old = JSON.parse(localStorage.getItem(STORE) || 'null'); } catch (e) { old = null; }
  if (Array.isArray(old) && old.length) {
    const have = new Set(convs.map(c => c.id));
    const add = old.filter(c => c && c.id && !have.has(c.id));
    let moved = !add.length;                    // nothing to move is nothing to lose
    if (add.length) {
      try {
        await tx('readwrite', st => { add.forEach(c => st.put(c)); });
        convs = convs.concat(add);
        moved = true;
      } catch (e) { /* keep the localStorage copy for the next attempt */ }
    }
    // Only now, with the records confirmed in the new store.
    if (moved) { try { localStorage.removeItem(STORE); } catch (e) {} }
  }
  convs.sort((a, b) => (b.updated || 0) - (a.updated || 0));
  convsReady = true;
}

// Writes are fire-and-forget: every caller is a UI action that has already updated `convs`
// and re-rendered, and none of them can do anything useful with a storage failure. The
// whole set is written because that is the granularity every caller already works at --
// they mutate `convs` and call this -- and a handful of conversations is nothing to write.
function saveConvs() {
  if (!convsReady) return;                     // see convsReady
  const snapshot = convs.map(c => JSON.parse(JSON.stringify(c)));
  const keep = new Set(snapshot.map(c => c.id));
  tx('readwrite', st => {
    const all = st.getAllKeys();
    all.onsuccess = () => (all.result || []).forEach(k => { if (!keep.has(k)) st.delete(k); });
    snapshot.forEach(c => st.put(c));
  }).catch(() => { /* nothing the page can do about it */ });
}
function current() { return convs.find(c => c.id === curId) || null; }


// Presets are EXAMPLES ONLY — any ModelScope repo/file works via the two inputs.
// Examples only — any ModelScope repo/file works through the two inputs below.
// Full-size models at 3-bit-ish quantization, across several families and both dense and MoE,
// so the picker is not tied to one vendor or one architecture. `gb` drives the environment fit.
const PRESETS = [
  // Small first: a visitor on a modest machine should be able to reach a conversation in
  // under a minute rather than after a 13 GB download. Sizes are the real file sizes.
  { gb: 0.4,  label: 'Qwen3-0.6B · Q4_K_M · quickest start', repo: 'unsloth/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q4_K_M.gguf' },
  { gb: 1.0,  label: 'Qwen3-1.7B · Q4_K_M', repo: 'unsloth/Qwen3-1.7B-GGUF', file: 'Qwen3-1.7B-Q4_K_M.gguf' },
  { gb: 2.3,  label: 'Qwen3-4B-Instruct · Q4_K_M', repo: 'unsloth/Qwen3-4B-Instruct-2507-GGUF', file: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf' },
  { gb: 4.7,  label: 'Qwen3-8B · Q4_K_M', repo: 'unsloth/Qwen3-8B-GGUF', file: 'Qwen3-8B-Q4_K_M.gguf' },
  { gb: 8.4,  label: 'Qwen3-14B · Q4_K_M', repo: 'unsloth/Qwen3-14B-GGUF', file: 'Qwen3-14B-Q4_K_M.gguf' },
  { gb: 13.2, label: 'Qwen3.8-27B · 3-bit UD-Q3_K_XL · hybrid SSM + MTP', repo: 'unsloth/Qwen3.8-27B-GGUF', file: 'Qwen3.8-27B-UD-Q3_K_XL.gguf' },
  { gb: 10.9, label: 'Qwen3.8-27B · UD-IQ3_XXS', repo: 'unsloth/Qwen3.8-27B-GGUF', file: 'Qwen3.8-27B-UD-IQ3_XXS.gguf' },
  { gb: 13.0, label: 'Qwen3-32B · dense · UD-IQ3_XXS', repo: 'unsloth/Qwen3-32B-GGUF', file: 'Qwen3-32B-UD-IQ3_XXS.gguf' },
  { gb: 10.8, label: 'Gemma-3-27B-it · dense · UD-IQ3_XXS', repo: 'unsloth/gemma-3-27b-it-GGUF', file: 'gemma-3-27b-it-UD-IQ3_XXS.gguf' },
  { gb: 11.9, label: 'Mistral-Small-3.2-24B · dense · UD-Q3_K_XL', repo: 'unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF', file: 'Mistral-Small-3.2-24B-Instruct-2506-UD-Q3_K_XL.gguf' },
  { gb: 9.4,  label: 'Mistral-Small-3.2-24B · dense · UD-IQ3_XXS', repo: 'unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF', file: 'Mistral-Small-3.2-24B-Instruct-2506-UD-IQ3_XXS.gguf' },
  { gb: 11.5, label: 'gpt-oss-20B · MoE · Q3_K_M', repo: 'unsloth/gpt-oss-20b-GGUF', file: 'gpt-oss-20b-Q3_K_M.gguf' },
  { gb: 13.8, label: 'Qwen3-30B-A3B-Instruct · MoE · UD-Q3_K_XL', repo: 'unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF', file: 'Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf' },
  { gb: 13.8, label: 'Qwen3-Coder-30B-A3B · MoE · UD-Q3_K_XL', repo: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', file: 'Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf' },
  { gb: 0,    label: '— custom (type a repo/file below) —', repo: '', file: '' },
];

// What this machine can actually take: RAM, how much the browser will let us cache, and
// whether the GPU backend is available. Used to order the picker, never to hide anything.
let ENV = { ramGB: 0, quotaGB: 0, cores: 0, webgpu: false, persisted: false, runGB: Infinity };
const RUNTIME_GB = 1.5;            // interpreter + activations + KV cache, alongside the weights

// How big a model can run is decided by memory the GPU can reach, and that differs by
// topology:
//   * unified memory (Apple silicon, phones, integrated GPUs) — the GPU draws from system
//     RAM, so system RAM is the ceiling and there is no separate pool to account for;
//   * dedicated VRAM (discrete GPUs) — the ceiling is VRAM, which is typically far smaller
//     than system RAM.
// WebGPU does not report total GPU memory in either case (deliberately — it fingerprints),
// and it cannot be measured: buffers are allocated lazily, and even writing every byte
// keeps succeeding because the OS pages. Probing a 16 GB machine here handed out 25.8 GB
// lazily and 21.5 GB fully written, so an allocation probe measures the OS's willingness
// to swap, not usable memory. What is left is a derived estimate — so it is stated in the
// UI and can be overridden by someone who knows the actual VRAM.
const GPU_OVERRIDE_KEY = 'webtorch.gpuMemGB';
function gpuOverrideGB() {
  const v = parseFloat(localStorage.getItem(GPU_OVERRIDE_KEY) || '');
  return isFinite(v) && v > 0 ? v : 0;
}
async function detectEnv() {
  ENV.cores = navigator.hardwareConcurrency || 0;
  ENV.ramGB = navigator.deviceMemory || 0;          // 0 = not reported (Safari/Firefox)
  try {
    // Persistence only protects the cache from eviction. It does not decide what can run.
    if (navigator.storage && navigator.storage.persist) {
      try { ENV.persisted = await navigator.storage.persist(); } catch (e) { ENV.persisted = false; }
    }
    const est = await navigator.storage.estimate();
    ENV.quotaGB = (est.quota || 0) / 1e9;
  } catch (e) { ENV.quotaGB = 0; }
  try {
    if (navigator.gpu) {
      const ad = await navigator.gpu.requestAdapter();
      ENV.webgpu = !!ad;
      if (ad) {
        const L = ad.limits || {};
        // Real, spec'd caps — but per allocation, not totals. They bound the largest single
        // tensor; a value above the 4 GiB spec default is also evidence of a larger pool.
        ENV.perAllocGB = Math.max(L.maxBufferSize || 0, L.maxStorageBufferBindingSize || 0) / 1e9;
        ENV.gpuName = ad.info ? [ad.info.vendor, ad.info.architecture].filter(Boolean).join(' ') : '';
      }
    }
  } catch (e) { ENV.webgpu = false; }

  // Weights stay packed at their quantized width, so a 13 GB 3-bit file needs ~13 GB
  // resident. Disk quota is NOT part of this: it is a caching limit (~60% of free disk,
  // and it moves), and a model that exceeds it still streams and runs, just uncached.
  const ramCeil = ENV.ramGB ? Math.max(0, ENV.ramGB - RUNTIME_GB) : Infinity;
  const override = gpuOverrideGB();
  ENV.override = !!override;
  // Unified memory is the common case in a browser (phones, Macs, integrated GPUs) and is
  // also the safe assumption, since it ties the ceiling to RAM, which IS reported. A
  // discrete GPU with less VRAM than RAM cannot be detected, hence the override.
  ENV.gpuCeilGB = override || ramCeil;
  ENV.runGB = Math.min(ramCeil, ENV.gpuCeilGB);
  return ENV;
}
function envSummary() {
  const p = [];
  p.push(ENV.ramGB ? ENV.ramGB + ' GB RAM' : 'RAM not reported');
  if (ENV.cores) p.push(ENV.cores + ' cores');
  p.push(ENV.webgpu ? ('WebGPU' + (ENV.gpuName ? ' (' + ENV.gpuName + ')' : '')) : 'CPU only');
  p.push(ENV.quotaGB ? ENV.quotaGB.toFixed(1) + ' GB cache quota' : 'quota unknown');
  return p.join(' · ') + ' → can run up to '
       + (ENV.runGB === Infinity ? 'unknown' : '≈' + ENV.runGB.toFixed(1) + ' GB')
       + (ENV.override ? ' (your GPU memory setting)'
                       : ENV.webgpu ? ' (estimated — set GPU memory below if your GPU has its own)'
                                    : '');
}

function call(cmd, args) {
  const id = ++seq;
  const p = new Promise((res, rej) => pending.set(id, { res, rej }));
  gpuInit.then(() => worker.postMessage({ id, cmd, args }));
  return p;
}
// The worker's stop flag, shared memory rather than a message. See `stopFlagRaise` there:
// while a generation or a load is running, the worker's thread is inside one call and does
// not reach `onmessage` at all, so a stop that travels as a message cannot arrive until the
// thing it is stopping has finished. Storing into this is seen at the SDK's next checkpoint.
let stopFlag = null;
let intrBuf = null;
let escalateT = null;

// Two stops, in order of politeness.
//
// The flag is cooperative and is the one worth having: the work ends at its own next
// checkpoint, so a reply keeps the tokens it has already produced. But a checkpoint is only
// reached if the work is looking, and a load whose bytes are cached goes a long way between
// looks -- measured at 19 seconds of a worker that answers nothing.
//
// So if the flag has not been acted on shortly, interrupt the interpreter itself. Pyodide
// reads this buffer from its own eval loop and raises KeyboardInterrupt inside whatever is
// running, which does not depend on that code checking anything. `endStop` is called when
// the work does end, so the escalation only fires when it genuinely did not.
// Long enough for the polite stop to win where it can, short enough not to be a wait. A
// generation notices the flag between tokens and has been measured at 4-19ms; this is an
// order of magnitude above that, and everything past it was not going to notice at all.
const STOP_GRACE_MS = 120;
function askStop() {
  if (stopFlag) Atomics.store(stopFlag, 0, 1);
  if (!intrBuf) return;
  clearTimeout(escalateT);
  escalateT = setTimeout(() => { try { intrBuf[0] = 2; } catch (e) {} }, STOP_GRACE_MS);
}
function endStop() {
  clearTimeout(escalateT); escalateT = null;
  if (intrBuf) { try { intrBuf[0] = 0; } catch (e) {} }
}

// A click picks the message a browser belongs to, and moves that browser into place. Only
// a click does this: the passive reading of "current" must not scroll anything.
document.addEventListener('click', (e) => {
  const el = $('#messages');
  if (!el || !el.contains(e.target)) return;
  if (e.target.closest('.webtabs')) return;      // working inside a browser is not picking one
  const msg = e.target.closest('.msg');
  if (!msg || msg.dataset.idx == null) return;
  clickedMsgIdx = Number(msg.dataset.idx);
  alignCurrentWeb();
});

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'stopbuf') { stopFlag = new Int32Array(m.buf); return; }
  if (m.type === 'intrbuf') { intrBuf = new Uint8Array(m.buf); return; }
  if (m.type === 'result') {
    const p = pending.get(m.id); pending.delete(m.id);
    endStop();                                  // it ended; nothing left to escalate to
    if (p) (m.error ? p.rej(new Error(m.error)) : p.res(m.res));
  } else if (m.type === 'status') {
    $('#modelStatus').textContent = m.text;
    $('#miniStatus').textContent = m.text.split('\n')[0].slice(0, 60);
  }
  else if (m.type === 'exporting') {
    // Reported WHERE THE EXPORT WAS STARTED. `#progressText` lives in the Model panel and
    // the export button in the Storage one, so an export watched from Storage showed
    // nothing at all -- and since the picked file stays at zero bytes until close (the
    // browser writes a temporary and swaps it in), there was no sign anywhere that it was
    // working. Eighty seconds of that reads as a failure.
    const pct = m.total ? ' of ' + fmt(m.total) : '';
    storeStatus('Writing ' + fmt(m.bytes) + pct
              + ' — the file appears on disk when this finishes.');
  }
  else if (m.type === 'progress') { if (m.total) { expected = m.total; expectedIsReal = true; }
    if (m.bytes > 0) loadedGB = +(m.bytes / 1e9).toFixed(2);
    showProgress(m.bytes, m.rate, m.dlRate); }
  else if (m.type === 'stage') {
    if (m.stage === 'reading' || !stageLog.length && m.after === null) stageLog = [];
    // The byte meter has stopped and the load has not. Say which of the remaining steps is
    // running, in the words of what it is FOR rather than the function's name: someone
    // watching a 13 GB model wants to know it is still working, not which method is on the
    // stack.
    const say = {
      warming:  'measuring kernel shapes for this model',
      checking: 'checking the weights against the file',
      proving:  'running one forward pass to prove it works',
      ready:    '',
    }[m.stage];
    // Keep what the finished stage cost, so a slow load names its own cost instead of
    // being remembered as "it hung somewhere".
    if (m.after && m.elapsed >= 1) stageLog.push(m.after + ' ' + m.elapsed.toFixed(1) + 's');
    if (say) {
      const n = m.total ? ' (' + m.done + '/' + m.total + ')' : '';
      const past = stageLog.length ? '  [' + stageLog.join(' · ') + ']' : '';
      $('#progressText').textContent = say + n + ' …' + past;
      // An indeterminate stage: the bar stops pretending to measure and just moves, which
      // is the honest signal for "working, cannot say how far".
      setBarBusy(true);
    } else {
      setBarBusy(false);
    }
  }
  else if (m.type === 'loaded') {
    setBarBusy(false);
    if (stageLog.length) console.log('load stages: ' + stageLog.join(' · '));
    probeTools();            // asked once per model, before any reply needs the answer
    modelLoaded = true; modelImage = !!m.image;
    // Done is done: leaving the last mid-load fraction on screen reads as a load that
    // stalled just short of the end.
    if (lastLoadedBytes) $('#progressText').textContent = 'loaded ' + fmt(lastLoadedBytes);
    setBar(1); syncButtons(); refreshCache(); }
  else if (m.type === 'chunk') {
    // One decoded token. Append it to the message being streamed and update that message's
    // DOM in place — no full re-render per token, and the thinking box keeps whatever state
    // the person left it in.
    if (!streaming) return;
    const conv = convs.find(c => c.id === streaming.convId);
    const reply = conv && conv.messages[streaming.idx];
    if (!reply) { streaming = null; return; }
    // The SDK says which channel a piece belongs to. Reasoning is never appended to the
    // answer, not even for one frame: a model whose template opens the <think> span emits
    // only the closing tag, so anything guessing from the text alone shows the scratchpad
    // as the reply until that tag finally arrives.
    if (m.channel === 'thinking') reply.think = (reply.think || '') + m.text;
    else reply.content += m.text;
    const live = streaming.live;
    if (live) {
      stopDots(live);                                       // the wait is over
      live.tokens += 1;
      // The worker's stamp, not this thread's: timing the arrivals here counts the cost of
      // rendering the reply and of delivering the message, neither of which the model spent
      // on the token. It is the same quantity the final footer reports, so the number does
      // not change when the reply finishes. `at` is absent only if a build without it is
      // still in a cache -- then this falls back to the old reading rather than to nothing.
      const now = (m.at != null) ? m.at : performance.now();
      const wall = performance.now();
      if (!live.t0) live.t0 = now;
      resNoteStep(now);                    // the resource strip reads decode latency from here
      // decode speed = tokens after the first, over the time since the first; updated at
      // most twice a second so the number stays readable
      if (live.rate && live.tokens >= 2 && (live.tokens === 2 || wall - live.rateT > 500)) {
        live.rateT = wall; live.rate.hidden = false;
        const sps = (live.tokens - 1) / ((now - live.t0) / 1000);
        live.rate.textContent = live.tokens + ' tokens · ' + sps.toFixed(1) + ' tok/s';
      }
    }
    if (live && live.ans.isConnected) {
      const el = $('#messages');
      const follow = atBottom(el);          // asked before the reply grows under us
      fillBody(streaming.body, reply, streaming.live);
      keepAtBottom(el, follow);
    }
  }
  else if (m.type === 'statbuf') { res.buf = new Float64Array(m.buf); return; }
  else if (m.type === 'log') { console.log('[py]', m.text); }
  else if (m.type === 'storageFull') { offerDirectory(m.key); }
  else if (m.type === 'migrate') {
    $('#progressText').textContent = 'moving to disk · ' + fmt(m.bytes);
  }
  else if (m.type === 'backend') {
    ENV.backend = m.name;
    // The runtime is up: this is the point everything else was waiting for.
    envReady = true;
    $('#openSettings').disabled = false;
    syncButtons();
    // A CPU fallback is the difference between seconds and minutes per reply, so it is
    // stated rather than left for the user to infer from the wait.
    $('#envInfo').textContent = envSummary() + (m.name === 'cpu'
      ? ' — GPU backend unavailable, running on CPU (expect minutes per reply)'
      : m.name === 'webgl'
      ? ' — compute: WebGL (no WebGPU here; about ' + WEBGL_SLOWDOWN + ' slower)'
      : ' — compute: ' + m.name);
    if (m.name === 'webgl') warnWebglFallback(m.why || null);
    // Only for no GPU at all. WebGL is a GPU backend -- slower than WebGPU, but a dialog
    // headed "Running on the CPU" would be simply false, and the slow-reply note covers a
    // backend that is working and still not fast enough.
    if (m.name === 'cpu') warnCpuFallback(m.name, m.why);
  }
};
// No GPU backend. This is worth interrupting for: the difference is roughly three hundred
// times, and it is invisible until someone has waited out a reply. A line in Settings was
// not enough -- a report came in of 0.5 tok/s on an M4 Pro, a machine that should manage
// hundreds, and nobody involved knew the page had fallen back.
//
// The cause matters as much as the fact, because two of the three are fixable by the person
// reading this, so the page says which one it is from what it can actually observe.
// Above this, a CPU-only page cannot hold the weights: they are numpy arrays in the WASM
// heap rather than GPU buffers. Deliberately approximate -- the ceiling depends on what
// else the heap holds -- and it warns rather than blocks, because the number is a guide.
const CPU_MAX_GB = 2;

// What WebGL costs, measured on this codebase rather than estimated: same models, same
// prompt, same warm-up, each repeated three or four times.
//
//   Qwen3-0.6B Q4_K_M      108.0 -> 19.8 tok/s   5.5x
//   Qwen 3B Q4_K            35.7 -> 5.3          6.7x
//   Qwen3-30B-A3B MoE       34.8 -> 5.1          6.8x
//   Qwen3.8-27B hybrid       6.8 -> 0.76         8.9x   (i-quant, 48 of 64 layers recurrent)
//
// Stated inline rather than as a dialog: WebGL works and answers correctly, it is only
// slower, and the dialog is reserved for the case where the model will not run at all.
const WEBGL_SLOWDOWN = '5-9x';
// WebGL is a working backend, so this is not the CPU dialog's problem -- but the gap is
// large enough that a user who does not know which backend they got will read it as the
// model being slow rather than the browser lacking WebGPU. Same shape as the CPU dialog,
// once per session, and it does not block: "Continue" is the only outcome.
function warnWebglFallback(sdkWhy) {
  if (sessionStorage.getItem('webtorch.webglWarned')) return;
  sessionStorage.setItem('webtorch.webglWarned', '1');
  const dlg = document.createElement('dialog');
  dlg.className = 'cpuwarn';
  const h = document.createElement('h2'); h.textContent = 'Running on WebGL, not WebGPU';
  const p1 = document.createElement('p');
  // Which of the two it is, rather than assuming the first: WebGPU can be present and still
  // fail to start (the same distinction the CPU dialog makes), and telling someone their
  // browser has no WebGPU when the environment line above says it does is simply wrong.
  p1.textContent = (ENV.webgpu
      ? 'WebGPU is present but did not start, so the models run on WebGL. '
      : 'This browser has no WebGPU, so the models run on WebGL. ')
    + 'Replies are correct \u2014 every quantization format is checked against the reference '
    + 'decoder on this backend too \u2014 but they arrive about ' + WEBGL_SLOWDOWN + ' slower.';
  const p2 = document.createElement('p');
  p2.innerHTML = 'Measured here, same prompt and same warm-up: '
    + '<strong>Qwen3-0.6B</strong> 19.8 tok/s against 108.0 on WebGPU; '
    + '<strong>Qwen 3B</strong> 5.3 against 35.7; '
    + '<strong>Qwen3-30B-A3B</strong> (MoE) 5.1 against 34.8; '
    + '<strong>Qwen3.8-27B</strong> (hybrid, i-quant) 0.76 against 6.8 \u2014 the widest gap, '
    + 'because its quantization is the most arithmetic to decode.';
  const p3 = document.createElement('p'); p3.className = 'why';
  p3.textContent = 'WebGL2 has no compute stage: every kernel is a fragment shader, one '
    + 'invocation per output value, with no memory shared between invocations. A thread '
    + 'cannot stage the activations for its neighbours, so each one re-reads them \u2014 '
    + 'which costs more than reading the weights does.'
    + (ENV.webgpu ? '' : ' For WebGPU: Chrome or Edge 113+, or Safari 18+; Firefox does not '
                       + 'enable it by default yet.')
    + (sdkWhy ? ' The runtime reports: ' + sdkWhy + '.' : '');
  const diag = {
    backend: 'webgl',
    reason: sdkWhy || null,
    navigatorGPU: !!navigator.gpu,
    webgpuAdapter: ENV.webgpu ? (ENV.gpuName || 'yes') : false,
    crossOriginIsolated: !!window.crossOriginIsolated,
    protocol: location.protocol,
    cores: navigator.hardwareConcurrency || null,
    ua: navigator.userAgent,
  };
  const det = document.createElement('details');
  const sum = document.createElement('summary'); sum.textContent = 'Diagnostics';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(diag, null, 2);
  det.append(sum, pre);
  const bar = document.createElement('div'); bar.className = 'editbar';
  const ok = document.createElement('button'); ok.type = 'button';
  ok.className = 'primary'; ok.textContent = 'Continue';
  ok.onclick = () => dlg.close();
  const cp = document.createElement('button'); cp.type = 'button';
  cp.textContent = 'Copy diagnostics';
  cp.onclick = () => {
    navigator.clipboard.writeText(JSON.stringify(diag, null, 2))
      .then(() => { cp.textContent = 'Copied'; }, () => { cp.textContent = 'Copy failed'; });
  };
  bar.append(ok, cp);
  dlg.append(h, p1, p2, p3, det, bar);
  document.body.appendChild(dlg);
  dlg.showModal();
}

function warnCpuFallback(name, sdkWhy) {
  if (sessionStorage.getItem('webtorch.cpuWarned')) return;   // once per session
  sessionStorage.setItem('webtorch.cpuWarned', '1');
  const why = [];
  // What the SDK recorded at the point it gave up, which is the only account of the real
  // cause; everything after it is what the page can see for itself.
  if (sdkWhy) why.push('The runtime reports: ' + sdkWhy + '.');
  if (window.__coiFileMode) {
    why.push('This page was opened as a file. Serve the folder over HTTP instead — ' +
             'a file:// page cannot be cross-origin isolated, and without that the GPU ' +
             'backend cannot start.');
  } else if (!window.crossOriginIsolated) {
    why.push('This page is not cross-origin isolated, so SharedArrayBuffer is unavailable ' +
             'and the GPU backend cannot start. The server must send ' +
             'Cross-Origin-Opener-Policy: same-origin and Cross-Origin-Embedder-Policy, ' +
             'or the bundled service worker must be allowed to add them.');
    if (window.__coiNoSW) why.push('This browser has no service worker support to fall back on.');
    if (window.__coiSWFailed) why.push('The service worker failed to register: ' + window.__coiSWFailed);
  } else if (!ENV.webgpu) {
    why.push('This browser reports no WebGPU. Chrome or Edge 113+, or Safari 18+, ' +
             'have it; Firefox does not yet enable it by default.');
  } else {
    why.push('WebGPU is present but the backend did not start. The browser console will ' +
             'have the reason.');
  }
  const dlg = document.createElement('dialog');
  dlg.className = 'cpuwarn';
  const h = document.createElement('h2'); h.textContent = 'Running on the CPU';
  const p1 = document.createElement('p');
  // Speed is the smaller half of this. With a GPU backend the weights live in GPU buffers
  // and the WASM heap holds almost nothing; without one they are numpy arrays inside that
  // heap, which is a 32-bit address space -- about 4 GB, shared with the interpreter and
  // every intermediate. So most of the models in the picker do not run slowly on the CPU,
  // they fail to load at all, and saying only "slow" sets the wrong expectation.
  p1.textContent = 'No GPU backend is available. Two things follow, and the first is the '
    + 'one that decides whether a model runs at all: on the CPU the weights are held in '
    + 'the page\u2019s WASM heap, which caps out near 4 GB in total, so anything past '
    + 'roughly ' + CPU_MAX_GB + ' GB runs out of memory while loading rather than running '
    + 'slowly. Under that size it does run, at minutes per reply rather than seconds \u2014 '
    + 'a small model that would answer at a hundred tokens a second manages well under one.';
  const p2 = document.createElement('p'); p2.className = 'why'; p2.textContent = why.join(' ');
  const diag = {
    backend: name || 'unknown',
    reason: sdkWhy || null,
    crossOriginIsolated: !!window.crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    navigatorGPU: !!navigator.gpu,
    webgpuAdapter: ENV.webgpu ? (ENV.gpuName || 'yes') : false,
    serviceWorker: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
    protocol: location.protocol,
    coiFileMode: !!window.__coiFileMode,
    coiNoSW: !!window.__coiNoSW,
    coiSWFailed: window.__coiSWFailed || null,
    cores: navigator.hardwareConcurrency || null,
    ua: navigator.userAgent,
  };
  const det = document.createElement('details');
  const sum = document.createElement('summary'); sum.textContent = 'Diagnostics';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(diag, null, 2);
  det.append(sum, pre);
  const bar = document.createElement('div'); bar.className = 'editbar';
  const ok = document.createElement('button'); ok.type = 'button';
  ok.className = 'primary'; ok.textContent = 'Continue anyway';
  ok.onclick = () => dlg.close();
  const cp = document.createElement('button'); cp.type = 'button';
  cp.textContent = 'Copy diagnostics';
  cp.onclick = () => {
    navigator.clipboard.writeText(JSON.stringify(diag, null, 2))
      .then(() => { cp.textContent = 'Copied'; }, () => { cp.textContent = 'Copy failed'; });
  };
  bar.append(ok, cp);
  dlg.append(h, p1, p2, det, bar);
  document.body.appendChild(dlg);
  dlg.showModal();
}

// A reply that came back far slower than the machine should manage. The GPU dialog covers
// the case where there is no GPU at all; this covers the rest -- a GPU that is being used
// and is still slow, which is the harder one to notice because nothing is obviously wrong.
//
// The thresholds are per model size because tokens per second means nothing without it: a
// 0.6B and a 30B that both answer at 5 tok/s are one broken machine and one normal one.
// Deliberately well under what the same models actually reach here (0.6B at 149, a 30B MoE
// at 39), so this fires on something being wrong rather than on a slow afternoon.
const SLOW_LIMITS = [
  { maxGB: 1, floor: 40, label: 'under 1 GB' },
  { maxGB: 10, floor: 10, label: '1-10 GB' },
  { maxGB: Infinity, floor: 1, label: 'over 10 GB' },
];
// Recorded as the load runs, not read back from the progress line afterwards -- that line
// is cleared when the load finishes, so by the time a reply is slow there is nothing left
// in it to parse, and the check silently never fired.
let loadedGB = null;
function modelSizeGB() {
  if (loadedGB) return loadedGB;
  const chosen = PRESETS[+$('#preset').value];
  return (chosen && chosen.gb) || null;
}
// Held as state rather than as a node, because the note lives inside #messages and every
// render() rebuilds that list -- an appended node was being wiped by the render on the very
// next line, so the warning was computed correctly and then thrown away before anyone saw it.
let slowState = null;
function checkSlow(stats) {
  if (!stats || stats.tok_s == null) return;             // too short to have measured
  const gb = modelSizeGB();
  if (!gb) return;
  const lim = SLOW_LIMITS.find(l => gb <= l.maxGB);
  slowState = (!lim || stats.tok_s >= lim.floor) ? null : { rate: stats.tok_s, gb, lim };
  paintSlowNote();
}
function hideSlowNote() { slowState = null; const el = $('#slowNote'); if (el) el.remove(); }
function paintSlowNote() {
  const el = $('#slowNote'); if (el) el.remove();
  if (slowState) showSlowNote(slowState.rate, slowState.gb, slowState.lim);
}
function showSlowNote(rate, gb, lim) {
  const box = document.createElement('div');
  box.id = 'slowNote'; box.className = 'slownote';
  const t = document.createElement('div');
  t.innerHTML = '<strong>' + rate.toFixed(1) + ' tok/s</strong> for a ' + gb + ' GB model — ' +
    'a model this size (' + lim.label + ') should manage at least ' + lim.floor + ' here.';
  const det = document.createElement('details');
  const sum = document.createElement('summary'); sum.textContent = 'Why might that be';
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify({
    tok_s: rate, modelGB: gb, backend: ENV.backend || null,
    webgpuAdapter: ENV.webgpu ? (ENV.gpuName || 'yes') : false,
    crossOriginIsolated: !!window.crossOriginIsolated,
    contextTokens: lmaxValue() || 'auto',
    conversationMessages: (current() || { messages: [] }).messages.length,
    recentErrors: DBG_ERRS.slice(-3),
    diagnosticsTopic: DBG_TOPIC,
  }, null, 2);
  det.append(sum, pre);
  const x = document.createElement('button');
  x.type = 'button'; x.className = 'x'; x.textContent = '✕';
  x.onclick = hideSlowNote;
  box.append(t, det, x);
  const msgs = $('#messages');
  msgs.appendChild(box);
  msgs.scrollTop = msgs.scrollHeight;
}

// ---- remote diagnostics --------------------------------------------------------------
// A problem that only happens on someone else's machine is the hardest kind to fix, and the
// facts that would settle it -- which backend started, what is cached, what threw -- all
// live in their browser. This publishes those facts, on request, over MQTT: the page joins
// a random topic, the person hands that topic to whoever is helping, and the helper can ask.
//
// IT CANNOT RUN CODE, and that is deliberate rather than unfinished. The default broker is
// public: anyone who learns the topic can send to it. A request handler that evaluated what
// it received would be a remote shell into a stranger's browser, published to the open
// internet, in exchange for saving the trouble of adding a command. So there is a fixed
// list, each item returning facts the page already shows someone who knows where to look,
// and nothing that reaches a conversation, a file, or anything typed.
const DBG_ON_KEY = 'webtorch.dbgEnabled';
const DBG_URL_KEY = 'webtorch.dbgUrl';
const DBG_URL_DEFAULT = 'wss://broker.hivemq.com:8884/mqtt';
// 128 bits of it, because the topic is the only thing standing between a public broker and
// this page's diagnostics: it has to be unguessable, and it has to end.
//
// Kept for a day rather than for one page load. Diagnosing something usually outlives the
// tab it was noticed in -- a reload, a crash, coming back after lunch -- and a topic that
// changed each time meant the person had to re-send it every time, which in practice meant
// the helper was looking at a channel nobody was on any more. A day is long enough to cover
// that and short enough that a topic handed out once does not stay live indefinitely.
//
// The clock runs from when the topic was MINTED, not from last use: an expiry that slid
// forward on every visit would never arrive for the page that is open the most.
const DBG_TOPIC_KEY = 'webtorch.dbgTopic';
const DBG_TOPIC_TTL_MS = 24 * 60 * 60 * 1000;

function dbgMintTopic() {
  const t = 'webtorch/' + Array.from(crypto.getRandomValues(new Uint8Array(16)))
                               .map(b => b.toString(16).padStart(2, '0')).join('');
  try { localStorage.setItem(DBG_TOPIC_KEY, JSON.stringify({ topic: t, at: Date.now() })); }
  catch (e) { /* a topic that cannot be stored still works for this page */ }
  return t;
}
function dbgTopicBorn() {
  try {
    const v = JSON.parse(localStorage.getItem(DBG_TOPIC_KEY) || 'null');
    return v && typeof v.at === 'number' ? v.at : 0;
  } catch (e) { return 0; }
}
function dbgLoadTopic() {
  try {
    const v = JSON.parse(localStorage.getItem(DBG_TOPIC_KEY) || 'null');
    if (v && typeof v.topic === 'string' && /^webtorch\/[0-9a-f]{32}$/.test(v.topic)
        && typeof v.at === 'number' && Date.now() - v.at < DBG_TOPIC_TTL_MS) {
      return v.topic;
    }
  } catch (e) { /* unreadable is the same as absent */ }
  return dbgMintTopic();
}
let DBG_TOPIC = dbgLoadTopic();
let dbgClient = null;
let dbgWatchT = null;      // the periodic state check
let dbgDownSince = 0;      // when the channel was last known to be up, 0 while it is

function dbgEnabled() { return localStorage.getItem(DBG_ON_KEY) !== '0'; }
function dbgUrl() { return localStorage.getItem(DBG_URL_KEY) || DBG_URL_DEFAULT; }
function dbgSay(t) { const el = $('#dbgState'); if (el) el.textContent = t; }

// The recent errors a report needs. Kept in a ring so a page that has been open all day
// does not accumulate an unbounded log, and captured here rather than asked for later
// because by the time anyone asks, the console has usually been cleared.
const DBG_ERRS = [];
function dbgNoteError(kind, msg, extra) {
  DBG_ERRS.push({ t: new Date().toISOString(), kind: kind, msg: String(msg).slice(0, 400),
                  at: extra || null });
  if (DBG_ERRS.length > 40) DBG_ERRS.shift();
}
window.addEventListener('error', (e) => dbgNoteError('error', e.message,
  e.filename ? e.filename.split('/').pop() + ':' + e.lineno : null));
window.addEventListener('unhandledrejection', (e) =>
  dbgNoteError('unhandledrejection', (e.reason && e.reason.message) || e.reason));

async function dbgAnswer(cmd) {
  switch (cmd) {
    case 'ping':
      return { pong: true, at: new Date().toISOString(), topic: DBG_TOPIC };
    case 'diag':
      return { backend: ENV.backend || null, crossOriginIsolated: !!window.crossOriginIsolated,
               sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
               navigatorGPU: !!navigator.gpu, webgpuAdapter: ENV.webgpu ? (ENV.gpuName || 'yes') : false,
               serviceWorker: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
               protocol: location.protocol, coiFileMode: !!window.__coiFileMode,
               coiNoSW: !!window.__coiNoSW, coiSWFailed: window.__coiSWFailed || null,
               ramGB: ENV.ramGB, quotaGB: ENV.quotaGB, cores: ENV.cores, ua: navigator.userAgent };
    case 'state':
      return { modelLoaded: modelLoaded, modelId: ($('#modelId') || {}).value || null,
               status: ($('#modelStatus') || {}).textContent || null,
               streaming: !!streaming, python: pyState,
               conversations: convs.length, convsReady: convsReady };
    case 'caches': {
      const out = {};
      for (const name of await caches.keys()) {
        const c = await caches.open(name);
        const keys = await c.keys();
        let bytes = 0;
        for (const r of keys) {
          const m = await c.match(r);
          if (m) { try { bytes += (await m.clone().arrayBuffer()).byteLength; } catch (e) {} }
        }
        out[name] = { entries: keys.length, MB: +(bytes / 1e6).toFixed(1) };
      }
      return out;
    }
    case 'errors':
      return { count: DBG_ERRS.length, recent: DBG_ERRS.slice(-15) };
    case 'perf': {
      const conv = current();
      const last = conv && [...conv.messages].reverse().find(m => m.stats);
      return { last: last ? last.stats : null, backend: ENV.backend || null };
    }
    default:
      return { error: 'unknown request',
               understood: ['ping', 'diag', 'state', 'caches', 'errors', 'perf'] };
  }
}

async function dbgStart() {
  dbgStop();
  if (!dbgEnabled()) { dbgSay('off'); return; }
  dbgSay('connecting…');
  try {
    if (typeof mqtt === 'undefined') {
      await new Promise((res, rej) => {
        const sc = document.createElement('script');
        sc.src = 'https://cdn.jsdelivr.net/npm/mqtt@5.10.1/dist/mqtt.min.js';
        sc.crossOrigin = 'anonymous';
        sc.onload = res; sc.onerror = () => rej(new Error('client script blocked'));
        document.head.appendChild(sc);
      });
    }
    // `resubscribe` covers a reconnect the library handles itself; the `connect` handler
    // covers one it does not, and both are cheap. The keepalive is well under the minute a
    // background tab throttles timers to, so a tab that is merely hidden still pings.
    const c = mqtt.connect(dbgUrl(), { connectTimeout: 10000, reconnectPeriod: 4000,
                                       keepalive: 30, clean: true, resubscribe: true });
    dbgClient = c;
    c.on('connect', () => {
      dbgDownSince = 0;
      c.subscribe(DBG_TOPIC + '/ask', (e) => {
        dbgSay(e ? ('subscribe failed: ' + e.message) : 'listening on ' + DBG_TOPIC);
      });
    });
    c.on('reconnect', () => { if (dbgClient === c) dbgSay('reconnecting…'); });
    c.on('offline', () => { if (dbgClient === c) { dbgDown(); dbgSay('offline — retrying'); } });
    c.on('message', async (_t, payload) => {
      const cmd = payload.toString().trim().slice(0, 32) || 'ping';
      let body;
      try { body = await dbgAnswer(cmd); }
      catch (e) { body = { error: String(e && e.message || e) }; }
      try { c.publish(DBG_TOPIC + '/say', JSON.stringify({ cmd: cmd, body: body })); }
      catch (e) { /* nothing useful to do about a failed reply */ }
    });
    c.on('error', (e) => { dbgDown(); dbgSay('error: ' + String(e && e.message || e)); });
    c.on('close', () => { if (dbgClient === c) { dbgDown(); dbgSay('disconnected — retrying'); } });
    dbgWatchStart();
  } catch (e) {
    dbgSay('unavailable: ' + String(e && e.message || e));
  }
}

// Getting the channel back is not one mechanism but three, because the ways it goes down
// are not alike:
//
//  - The library's own retry handles a broker that dropped one connection.
//  - It does NOT handle a background tab, which is the common case for a page left open to
//    be diagnosed: timers there are throttled to about once a minute, the keepalive ping is
//    missed, the broker drops the connection, and the retry that would fix it is throttled
//    by the same rule. What gets that back is an EVENT -- the tab becoming visible, the
//    network returning -- so those reconnect directly.
//  - And neither handles a client that has stopped retrying at all, which a socket error at
//    the wrong moment can leave behind. Only asking "is it actually connected?" on a timer
//    finds that one, so the check runs whether or not anything was reported.
//
// The topic is deliberately NOT part of any of this. It is fixed for the life of the page,
// because whoever is helping was given it once; a reconnect that renamed the channel would
// be indistinguishable from the channel staying down.
const DBG_DOWN_LIMIT_MS = 45000;   // past this the library's own retry is not working

function dbgDown() { if (!dbgDownSince) dbgDownSince = Date.now(); }

function dbgCheck(why) {
  if (!dbgEnabled()) return;
  const c = dbgClient;
  if (!c) { dbgStart(); return; }                    // enabled with no client at all
  if (c.connected) { dbgDownSince = 0; return; }
  dbgDown();
  // Down long enough that reconnecting this client is not what is missing -- build another.
  // `end(true)` first, or the old one keeps its timers and its socket.
  if (Date.now() - dbgDownSince > DBG_DOWN_LIMIT_MS) {
    dbgSay('reconnecting from scratch' + (why ? ' (' + why + ')' : ''));
    dbgStart();
    return;
  }
  try { c.reconnect(); } catch (e) { dbgStart(); }
}

function dbgWatchStart() {
  clearInterval(dbgWatchT);
  dbgWatchT = setInterval(() => dbgCheck('periodic'), 15000);
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') dbgCheck('tab visible');
});
window.addEventListener('online', () => dbgCheck('network back'));

function dbgStop() {
  clearInterval(dbgWatchT); dbgWatchT = null;
  dbgDownSince = 0;
  if (dbgClient) { try { dbgClient.end(true); } catch (e) {} dbgClient = null; }
}

// How much of the day is left, said where the topic is shown -- a topic that has quietly
// expired and one that is simply not being asked about look identical otherwise.
function dbgTopicAge() {
  const el = $('#dbgTopic');
  if (!el) return;
  const left = DBG_TOPIC_TTL_MS - (Date.now() - dbgTopicBorn());
  const h = Math.floor(left / 3600000), m = Math.floor((left % 3600000) / 60000);
  el.title = left > 0 ? 'This topic stops working in ' + (h ? h + 'h ' : '') + m + 'm'
                      : 'Expired — a new topic is minted on the next reload';
}

function wireDebug() {
  const t = $('#dbgTopic'); if (!t) return;
  t.value = DBG_TOPIC;
  dbgTopicAge();
  $('#dbgUrl').value = dbgUrl();
  $('#dbgEnabled').checked = dbgEnabled();
  $('#dbgApply').onclick = () => {
    localStorage.setItem(DBG_ON_KEY, $('#dbgEnabled').checked ? '1' : '0');
    const u = $('#dbgUrl').value.trim();
    if (u) localStorage.setItem(DBG_URL_KEY, u); else localStorage.removeItem(DBG_URL_KEY);
    $('#dbgUrl').value = dbgUrl();
    dbgStart();
  };
  $('#dbgNew').onclick = () => {
    // Retiring a topic that has been handed out is the point of this control, so it takes
    // effect at once: the old one stops being listened on before the new one is announced.
    DBG_TOPIC = dbgMintTopic();
    t.value = DBG_TOPIC;
    dbgTopicAge();
    dbgStop();
    if (dbgEnabled()) dbgStart(); else dbgSay('off');
    note('New diagnostics topic — send this one instead: ' + DBG_TOPIC);
  };
  $('#dbgEnabled').onchange = () => {
    localStorage.setItem(DBG_ON_KEY, $('#dbgEnabled').checked ? '1' : '0');
    if ($('#dbgEnabled').checked) dbgStart(); else { dbgStop(); dbgSay('off'); }
  };
}

function setBar(f) { $('#bar').style.width = Math.min(100, f * 100) + '%'; }
// A model stays resident until it is released, so loading is blocked while one is held —
// otherwise a second multi-GB model would be pulled in on top of the first.
// Nothing works before the runtime has reported which backend it got. Until then every
// control is inert -- not just the ones that talk to the worker. Loading a model in that
// window is the costly mistake: cross-origin isolation may still be arriving (on a host that
// cannot send the headers, a service worker supplies them and only controls the NEXT load),
// and a model loaded before it lands runs on the CPU for its whole life -- minutes per reply
// instead of seconds, with nothing on screen to explain why.
let envReady = false;

function syncButtons() {
  const boot = !envReady;
  $('#loadBtn').disabled = boot || modelLoaded;
  $('#releaseBtn').disabled = boot || !modelLoaded;
  $('#loadBtn').title = boot ? 'Waiting for the compute backend…'
    : modelLoaded ? 'Release the current model first' : '';
  ['#preset', '#modelId', '#lmax', '#gpuMem', '#gpuMemClear',
   '#refreshCache', '#clearCache', '#exportBtn', '#importBtn'].forEach(sel => {
    const el = $(sel); if (el) el.disabled = boot;
  });
  // Nothing in the conversation area is usable until a model is actually ready to answer.
  const off = boot || !modelLoaded;
  ['#input', '#send', '#toolFile', '#toolCam', '#toolUrl', '#newChat'].forEach(sel => {
    const el = $(sel); if (el) el.disabled = off;
  });
  // While a reply is being written the send control is the stop control -- there is nothing
  // else to do with it, and a reply you cannot interrupt is a reply you have to wait out.
  const send = $('#send');
  send.classList.toggle('stopping', !!streaming);
  send.textContent = streaming ? 'Stop' : 'Send';
  send.title = streaming ? 'Stop generating' : '';
  if (streaming) send.disabled = false;
  // Image sources go away with a model that cannot see: the file picker stays (text files
  // still make sense), the camera does not.
  $('#toolCam').disabled = off || !modelImage;
  $('#toolCam').title = modelImage ? 'Capture from camera'
    : 'This model cannot see images';
  $('#toolFile').title = modelImage ? 'Attach a file'
    : 'Attach a text file (this model cannot see images)';
  $('#input').placeholder = boot ? 'Starting the compute runtime…'
    : off ? 'Load a model in ⚙ Settings to start chatting' : 'Send a message…';
  $('#composer').classList.toggle('locked', off);
}
function fmt(b) { return b > 1e9 ? (b/1e9).toFixed(2)+' GB' : b > 1e6 ? (b/1e6).toFixed(1)+' MB' : (b/1e3).toFixed(0)+' KB'; }
let expected = 0, expectedIsReal = false;
let lastLoadedBytes = 0;
let stageLog = [];

function setBarBusy(on) {
  const bar = $('#bar');
  if (!bar) return;
  bar.classList.toggle('busy', !!on);
  if (on) bar.style.width = '100%';
}

function showProgress(bytes, rate, dlRate) {
  setBarBusy(false);
  lastLoadedBytes = bytes;
  // "loaded", not "downloaded": the same meter covers a load served entirely from cache.
  // The download figure is shown only while bytes are actually coming over the wire --
  // it comes from the HTTP reader, and a cached load has none.
  //
  // "of" only against a total the LOADER reported. The preset's size is a rounded label for
  // a human choosing a model, not a byte count: comparing real bytes against it leaves a
  // finished load reading "397.2 MB of 400.0 MB", which says something false about a file
  // that arrived whole. It seeds the bar so there is something to watch, and the moment a
  // real total arrives it is replaced.
  const parts = ['loaded ' + fmt(bytes) + (expectedIsReal ? ' of ' + fmt(expected) : '')];
  if (rate) parts.push(fmt(rate) + '/s');
  if (dlRate) parts.push('↓ ' + fmt(dlRate) + '/s');
  $('#progressText').textContent = parts.join(' · ');
  if (expected) setBar(bytes / expected); else setBar(((bytes / 5e8) % 1));
}

// ---- model ----
// The estimate is the machine's, the correction is the user's — applying it re-runs the
// same detection path so the list, the pick and the summary all move together.
function wireGpuMem() {
  const box = $('#gpuMem'); if (!box) return;
  const cur = gpuOverrideGB(); if (cur) box.value = cur;
  const apply = async (v) => {
    if (v > 0) localStorage.setItem(GPU_OVERRIDE_KEY, String(v));
    else localStorage.removeItem(GPU_OVERRIDE_KEY);
    await detectEnv(); fillPresets();
  };
  box.onchange = () => apply(parseFloat(box.value));
  $('#gpuMemClear').onclick = () => { box.value = ''; apply(0); };
}

function fillPresets() {
  const sel = $('#preset'); sel.innerHTML = '';
  const custom = PRESETS[PRESETS.length - 1];
  const DEFAULT = PRESETS[0];                       // Qwen3.8-27B 3-bit
  const models = PRESETS.slice(0, -1);

  // The "from this device" entries are actions, not selectable models: choosing one opens
  // a picker, and cancelling restores whatever was selected before.
  let lastGood = null;
  sel.onchange = () => {
    const p = PRESETS[sel.value];
    if (!p) { localPick(sel.value, () => { sel.value = lastGood; }); return; }
    lastGood = sel.value;
    // one box, shown only for a model that is not in the list
    $('#modelId').value = p.repo ? (p.file ? p.repo + '/' + p.file : p.repo) : '';
    $('#customBox').hidden = !!p.repo;
  };

  // Hide only what this device genuinely cannot run: weights that exceed usable memory.
  // Exceeding the cache quota is not a reason to hide anything — such a model still runs,
  // it just re-downloads instead of being cached, so it gets a note rather than exclusion.
  const runnable = models.filter(m => m.gb <= ENV.runGB).sort((a, b) => b.gb - a.gb);
  const smallest = models.slice().sort((a, b) => a.gb - b.gb)[0];
  const noneRun  = runnable.length === 0;
  const shown    = noneRun ? [smallest] : runnable;  // never present an empty list
  const best     = runnable.includes(DEFAULT) ? DEFAULT : (runnable[0] || smallest);

  shown.concat([custom]).forEach((p) => {
    const o = document.createElement('option');
    o.value = PRESETS.indexOf(p);
    const note = !p.gb ? '' : noneRun ? ' ⚠ larger than this device\u2019s memory'
                                      : (p === best ? ' ✓ recommended' : '');
    o.textContent = p.label + (p.gb ? ' (' + p.gb.toFixed(1) + ' GB)' : '') + note;
    sel.appendChild(o);
  });

  // Loading a model from THIS device is the same decision as picking one to download, so
  // it lives in the same dropdown: a single .gguf file, or a whole model folder (GGUF
  // file(s), or an HF-format dir: config.json + *.safetensors, incl. GPTQ).
  [['local-file', 'Load a GGUF file from this device…'],
   ['local-dir',  'Load a model folder from this device (GGUF or HF safetensors)…'],
  ].forEach(([value, label]) => {
    const o = document.createElement('option');
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  });

  sel.value = PRESETS.indexOf(best); lastGood = sel.value; sel.onchange();
  // Said once, about the machine — not repeated on every row it happens to apply to.
  const cacheNote = ENV.quotaGB && best.gb > ENV.quotaGB
    ? ' Note: the cache quota is smaller than this model, so part of it re-downloads each session'
      + (ENV.persisted ? '.' : ' and the cache may be evicted (persistent storage was declined).')
    : '';
  $('#envInfo').textContent = envSummary() + (
      noneRun ? ' — no listed model fits in memory; showing the smallest, or enter another below'
    : best === DEFAULT ? ''
    : ' — the default needs more memory, so ' + best.label + ' is selected') + cacheNote;
}
// While a load is in flight the same button is the stop control — the only way out of a
// multi-GB download — and everything reverts to the initial state once it stops.
let loading = false;
$('#loadBtn').onclick = async () => {
  if (loading) {
    // Say so at once. The flag is set immediately now, but the load still has to reach its
    // next checkpoint, and a button that does not change reads as a button that did nothing.
    $('#loadBtn').textContent = 'Stopping…';
    $('#loadBtn').disabled = true;
    askStop();                                  // shared memory, so a busy worker still sees it
    call('stopLoad');
    return;
  }
  // a single identifier: "org/repo/file.gguf", or "org/repo" for a HF-format directory
  const id = $('#modelId').value.trim();
  if (!id) return alert('Enter a model as org/repo/file.gguf (or org/repo).');
  const repo = id, file = '';
  loading = true;
  $('#loadBtn').disabled = false;
  $('#loadBtn').textContent = 'Stop loading';
  $('#loadBtn').title = 'Stop this load — what is already loaded stays cached';
  setBar(0); expected = 0; expectedIsReal = false; lastLoadedBytes = 0;
  const chosen = PRESETS[$('#preset').value];
  if (chosen && chosen.gb) expected = chosen.gb * 1e9;   // a label, not a byte count
  // Checked BEFORE the download, not after it. Without a GPU the weights go into the WASM
  // heap and a large model runs out of memory at the end of a multi-gigabyte transfer --
  // which is a long way to travel to be told it was never going to work.
  if (ENV.backend === 'cpu' && chosen && chosen.gb > CPU_MAX_GB
      && !confirm('This page has no GPU backend, so the weights would go into the WASM '
                  + 'heap (about 4 GB in total, shared with everything else). At '
                  + chosen.gb + ' GB this model will almost certainly run out of memory '
                  + 'while loading, after downloading all of it.\n\nDownload it anyway?')) {
    loading = false; $('#loadBtn').textContent = 'Load'; $('#loadBtn').title = '';
    return;
  }
  try { await call('load', { repo, file, lmax: lmaxValue() }); note('Model ready. Large models take a while on first load; afterwards they come from the cache.'); }
  catch (e) {
    // The SDK raises one distinctive message for a stop the person asked for; that is a
    // normal ending, not a failure — say so, and put the meter back where it started.
    if (/load cancelled/.test(e.message)) {
      $('#modelStatus').textContent = 'load stopped';
      note('Load stopped. What was already loaded stays cached, so the next load resumes.');
      setBar(0); expected = 0; $('#progressText').textContent = '';
    } else {
      $('#modelStatus').textContent = 'load failed: ' + e.message; note(e.message);
    }
  }
  finally { loading = false; $('#loadBtn').textContent = 'Load model'; syncButtons(); }
};
$('#releaseBtn').onclick = async () => {
  await call('release'); modelLoaded = false; modelImage = false; setBar(0);
  $('#progressText').textContent = ''; syncButtons();
  note('Model released. Its files stay cached, so loading it again is fast.');
};

// ---- cache ----
async function refreshCache() {
  try {
    if (storeBusy) return;          // rebuilding the list mid-export would re-enable it
    const c = await call('cacheList');
    const el = $('#cacheList'); el.innerHTML = '';
    if (!(c.groups || []).length) { el.innerHTML = '<p class="hint">nothing cached yet</p>'; return; }
    const head = document.createElement('p'); head.className = 'hint';
    head.textContent = 'total ' + fmt(c.total);
    el.appendChild(head);
    // Listed per model rather than per file, because that is the unit you export, delete
    // or reuse -- a multi-file model is one thing, not five rows.
    c.groups.sort((a, b) => b.size - a.size).forEach(g => {
      const d = document.createElement('div'); d.className = 'item';
      const k = document.createElement('span'); k.className = 'k';
      k.title = g.keys.join('\n');
      // An incomplete model says so, and says how far it got -- otherwise a half-downloaded
      // entry is indistinguishable from a usable one.
      const partial = !g.complete;
      k.textContent = g.label + ' · ' + fmt(g.size)
        + (g.files > 1 ? ' · ' + g.files + ' files' : '')
        + (partial ? ' · ⚠ incomplete' + (g.total ? ' (' + Math.floor(100 * g.size / g.total)
                                                    + '% of ' + fmt(g.total) + ')' : '') : '');
      if (partial) k.classList.add('partial');
      const ex = document.createElement('button'); ex.textContent = 'export';
      ex.dataset.blocked = partial ? '1' : '0';            // its own reason, not the busy one
      ex.dataset.title0 = partial ? 'Finish downloading before exporting'
                                  : (g.files > 1 ? 'Save as a .zip' : 'Save the file itself');
      ex.onclick = () => exportModel(g);
      const del = document.createElement('button'); del.textContent = 'delete';
      del.dataset.blocked = '0';
      del.dataset.title0 = 'Remove these files from the cache';
      del.onclick = async () => {
        if (storeBusy) return;
        for (const key of g.keys) await call('cacheDelete', { key });
        refreshCache();
      };
      d.append(k, ex, del); el.appendChild(d);
    });
    applyStoreState();          // a freshly built list starts in the state that is true now
  } catch (e) { $('#cacheList').innerHTML = '<p class="hint">cache unavailable: ' + e.message + '</p>'; }
}

// A single-file model is saved as itself, so the exported file IS the model and any other
// tool reads it; several files become one .zip. The bytes stream from the SDK straight to
// disk through the picked handle, so a 12 GB model never becomes a Blob.
// While an export runs, nothing else in this panel may touch the cache: deleting the very
// bytes being read would leave a half-written file with no sign of why. `disabled` is set
// rather than the handlers removed, so the reason is visible -- a button that cannot be
// pressed says "not now", a button that does nothing says "broken".
let storeBusy = false;
const STORE_BUSY_WHY = 'An export is running — the cache cannot change until it finishes';

// Derived from what is true now, never restored from what was true before. A remembered
// "was it disabled" goes stale the moment the list re-renders, and then a button comes back
// enabled that should not be. Busy or not, an incomplete model still cannot be exported --
// that fact lives on the element, and this reads it every time.
function applyStoreState() {
  document.querySelectorAll('#cacheList button').forEach(b => {
    const own = b.dataset.blocked === '1';                 // its own reason, independent of us
    b.disabled = storeBusy || own;
    b.title = storeBusy ? STORE_BUSY_WHY : (b.dataset.title0 || '');
  });
  const clr = $('#clearCache');
  if (clr) { clr.disabled = storeBusy; clr.title = storeBusy ? STORE_BUSY_WHY : ''; }
}

function setStoreBusy(on) { storeBusy = !!on; applyStoreState(); }

// The Storage panel's own status line, for the things started from it.
function storeStatus(html, bad) {
  const el = $('#storeStatus');
  if (!el) return;
  if (html == null) { el.hidden = true; el.textContent = ''; return; }
  el.hidden = false; el.innerHTML = html;
  el.classList.toggle('bad', !!bad);
}

async function exportModel(g) {
  if (!window.showSaveFilePicker) {
    note('This browser cannot save large files directly (no File System Access API).');
    return;
  }
  const one = g.keys.length === 1;
  const name = one ? g.keys[0].split('/').pop() : g.label.split(' ')[0].replace(/\//g, '_') + '.zip';
  let handle;
  try {
    handle = await window.showSaveFilePicker({ suggestedName: name });
  } catch (e) { return; }                       // the person cancelled
  // Settle the write permission HERE, in the page, while the click that opened the picker
  // still counts as activation. The writing happens in the worker, and a worker can never
  // ask: `requestPermission` needs user activation and a worker has none, so a handle that
  // arrives there unpermitted fails at `createWritable` with nothing able to fix it.
  try {
    if (handle.queryPermission) {
      let p = await handle.queryPermission({ mode: 'readwrite' });
      if (p !== 'granted' && handle.requestPermission)
        p = await handle.requestPermission({ mode: 'readwrite' });
      if (p !== 'granted') {
        note('Export needs permission to write that file, and it was not granted.');
        return;
      }
    }
  } catch (e) { /* older browsers have no permission API; let the write speak for itself */ }
  note('Exporting ' + name + ' …');
  setStoreBusy(true);
  storeStatus('Preparing to write <code>' + name.replace(/[&<>]/g, '') + '</code>…');
  try {
    const n = await call('exportModel', { keys: g.keys, handle });
    storeStatus('Exported <code>' + name.replace(/[&<>]/g, '') + '</code> — ' + fmt(n)
              + ' written.');
    note('Exported ' + name + ' — ' + fmt(n) + ' written.');
  } catch (e) {
    storeStatus('<span class="miss">Export failed: '
              + String(e.message || e).replace(/[&<>]/g, '') + '</span>', true);
    note('Export failed: ' + e.message);
  } finally { setStoreBusy(false); }
  setBar(0); $('#progressText').textContent = ''; loadedGB = null;
}

// Browser storage is capped by policy, not by the disk -- a few GB here while hundreds are
// free. When a model runs past it the load keeps going by streaming, and this offers the one
// way out: a directory, which has no cap. Needs a gesture, so it has to happen in the page.
let offering = false;
async function offerDirectory(key) {
  if (offering || !window.showDirectoryPicker) return;
  offering = true;
  const name = String(key).split('/').pop();
  if (!confirm('Browser storage is full, so ' + name + ' cannot be kept.\n\n'
             + 'Choose a folder to keep models in instead? What is already stored moves '
             + 'there and is freed here; a folder has no size limit.')) {
    offering = false; note('Continuing without caching this model.'); return;
  }
  try {
    const dir = await window.showDirectoryPicker({ mode: 'readwrite' });
    note('Moving cached models to the folder…');
    await call('migrate', { dir });
    note('Done. Models are kept in that folder from now on.');
    refreshCache();
  } catch (e) { note('Not moved: ' + e.message); }
  offering = false;
}

// Local models are picked from the SAME model dropdown as downloaded ones (see
// fillPresets). Nothing is copied -- the files are read where they lie, no quota, no limit.
//
// Identity = the content for a single file, the folder for a folder model:
//   * a file is registered under a fingerprint of its size + first MB (SHA-256). The SAME
//     file then always maps to the SAME id, whichever door loads it -- so load(), pipeline()
//     and from_pretrained share ONE copy instead of loading it twice -- and two different
//     files that merely share a name ("model.gguf" is common) can never collide. A full-file
//     hash would mean reading all 13 GB before the load could even start; the fingerprint
//     decides "same model or not" just as well.
//   * a folder is registered under its own name, every file as "<dir>/<file>", so the
//     folder itself IS the model id and same-named files in different folders stay apart.
async function localFileId(handle) {
  const file = await handle.getFile();
  const head = await file.slice(0, 1 << 20).arrayBuffer();
  const sized = new Uint8Array(head.byteLength + 8);
  new DataView(sized.buffer).setBigUint64(0, BigInt(file.size));
  sized.set(new Uint8Array(head), 8);
  const digest = await crypto.subtle.digest('SHA-256', sized);
  const hex = [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('');
  const dot = handle.name.lastIndexOf('.');
  const hasExt = dot > 0 && /^\.[a-z0-9]+$/i.test(handle.name.slice(dot));
  const ext = hasExt ? handle.name.slice(dot) : '.gguf';
  // The person's own filename, then the fingerprint. The name is what they picked and what
  // they will look for in the box; the fingerprint is what makes the id an identity. It
  // used to be the fingerprint alone, which is correct and unreadable -- the box showed
  // "local-74b11e0d206b9cac6e45.gguf" for a file the person had just chosen by name.
  //
  // The FOLDER it came from cannot be shown: the File System Access API hands the page a
  // handle and a name, never a path, and that is deliberate on the browser's part.
  const base = hasExt ? handle.name.slice(0, dot) : handle.name;
  return (base || 'model') + '@' + hex.slice(0, 8) + ext;
}

// Runs one of the "from this device" dropdown entries: open the picker, register the model
// under its identity, start the load. The dropdown restores its previous pick on cancel.
async function localPick(which, onCancel) {
  const dir = which === 'local-dir';
  try {
    const handle = dir
      ? await window.showDirectoryPicker({ mode: 'read' })
      : (await window.showOpenFilePicker({
          multiple: false,
          types: [{ description: 'GGUF model', accept: { 'application/octet-stream': ['.gguf'] } }],
        }))[0];
    const name = dir ? handle.name : await localFileId(handle);
    const names = JSON.parse(await call('importModel', { handle, name }));
    if (!names.length) {
      note('No model files found in that ' + (dir ? 'folder' : 'file') + '.');
      onCancel(); return;
    }
    // A folder that ships an HF-format model loads BY THE FOLDER (config.json names the
    // rest), so its id is the folder; a GGUF loads by the file itself.
    let id = names[0];
    if (dir) {
      const gguf = names.find(n => n.endsWith('.gguf'));
      id = names.some(n => n.endsWith('/config.json') || n === 'config.json')
         ? handle.name : (gguf || names[0]);
    }
    note('Reading ' + names.length + ' file' + (names.length > 1 ? 's' : '')
         + ' straight from disk' + (dir ? ' (folder ' + handle.name + ')' : '')
         + ' — no download, no cache copy.');
    refreshCache();
    $('#preset').value = String(PRESETS.length - 1);   // "custom"
    $('#preset').onchange();
    $('#modelId').value = id;
    $('#loadBtn').click();
  } catch (e) {
    if (e.name !== 'AbortError') note('Import failed: ' + e.message);
    onCancel();
  }
}



// Remember the reply budget, the context budget and the thinking toggle the way the GPU
// override is remembered — these are the page's view of the SDK's generate/load parameters.
// Empty means "auto": run at the size the model itself declares.
const MAXNEW_KEY = 'webtorch.maxNew';
const LMAX_KEY = 'webtorch.lmax';
const THINK_KEY = 'webtorch.thinking';
function maxNewValue() { const v = parseInt($('#maxNew').value, 10); return v > 0 ? v : 0; }
function lmaxValue() { const v = parseInt($('#lmax').value, 10); return v >= 512 ? v : 0; }
function thinkingOn() { return !!$('#thinking').checked; }

// Every sampling control is the same shape: an input that is either empty (the model's own
// value wins) or a number the SDK takes under that exact name. Listing them once means the
// page stores, restores, resets and sends them without a branch per parameter.
const GEN_FIELDS = [
  ['#minNew', 'min_new_tokens', 'int'],
  ['#stopSeq', 'stop', 'list'],
  ['#temperature', 'temperature', 'num'],
  ['#topP', 'top_p', 'num'],
  ['#topK', 'top_k', 'int'],
  ['#minP', 'min_p', 'num'],
  ['#repPen', 'repetition_penalty', 'num'],
  ['#presPen', 'presence_penalty', 'num'],
  ['#freqPen', 'frequency_penalty', 'num'],
  ['#seed', 'seed', 'int'],
];
const genKey = sel => 'webtorch.gen' + sel.slice(1);

// Stop sequences are typed, so the escapes people type have to mean what they look like.
function parseStops(raw) {
  return String(raw || '').split(',').map(s => s.trim()).filter(Boolean)
    .map(s => s.replace(/\\n/g, '\n').replace(/\\t/g, '\t').replace(/\\r/g, '\r'));
}

// The generation options for one call: only what was actually set, so an untouched control
// never overrides what the model asks for.
function genOpts() {
  const o = { max_new: maxNewValue(), enable_thinking: thinkingOn() };
  for (const [sel, name, kind] of GEN_FIELDS) {
    const el = $(sel); if (!el) continue;
    const raw = String(el.value || '').trim();
    if (!raw) continue;
    if (kind === 'list') { const v = parseStops(raw); if (v.length) o[name] = v; continue; }
    const v = kind === 'int' ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isFinite(v)) o[name] = v;
  }
  return o;
}

(() => {
  $('#maxNew').value = localStorage.getItem(MAXNEW_KEY) || '';
  $('#lmax').value = localStorage.getItem(LMAX_KEY) || '';
  // On unless turned off: a model that reasons is worth watching reason, and the pane
  // folds itself once the answer starts. An explicit '0' still wins.
  $('#thinking').checked = localStorage.getItem(THINK_KEY) !== '0';
  $('#maxNew').onchange = () => {
    const v = maxNewValue();
    $('#maxNew').value = v ? String(v) : '';
    localStorage.setItem(MAXNEW_KEY, $('#maxNew').value);
  };
  $('#lmax').onchange = () => {
    const v = lmaxValue();
    $('#lmax').value = v ? String(v) : '';
    localStorage.setItem(LMAX_KEY, $('#lmax').value);
    note('Context length applies the next time a model loads.');
  };
  $('#thinking').onchange = () =>
    localStorage.setItem(THINK_KEY, $('#thinking').checked ? '1' : '0');
  for (const [sel] of GEN_FIELDS) {
    const el = $(sel); if (!el) continue;
    el.value = localStorage.getItem(genKey(sel)) || '';
    el.onchange = () => localStorage.setItem(genKey(sel), String(el.value || '').trim());
  }
  // Open the sampling card already expanded if anything in it is set, so a value that
  // silently shapes every reply is never hidden behind a fold.
  const adv = $('#advGrp');
  if (adv && GEN_FIELDS.slice(2).some(([sel]) => ($(sel) || {}).value)) adv.open = true;
  const reset = $('#genReset');
  if (reset) reset.onclick = () => {
    for (const [sel] of GEN_FIELDS) {
      const el = $(sel); if (!el) continue;
      el.value = ''; localStorage.removeItem(genKey(sel));
    }
    note('Generation options cleared — the model’s own values apply.');
  };
})();

// Settings come in one page per concern instead of one long scroll; the open tab is
// remembered.
const TAB_KEY = 'webtorch.settingsTab';
(() => {
  const show = (name) => {
    document.querySelectorAll('#tabs button').forEach(b =>
      b.classList.toggle('on', b.dataset.tab === name));
    document.querySelectorAll('.tab-panel').forEach(p => p.hidden = p.id !== 'tab-' + name);
    localStorage.setItem(TAB_KEY, name);
    if (name === 'store') refreshCache();
  };
  document.querySelectorAll('#tabs button').forEach(b => { b.onclick = () => show(b.dataset.tab); });
  show(localStorage.getItem(TAB_KEY) || 'model');
})();

$('#refreshCache').onclick = refreshCache;
$('#clearCache').onclick = async () => { if (confirm('Delete every cached model file?')) { await call('cacheClear'); refreshCache(); } };

// ---- attachments / tools ----
function addAttachment(a) { attachments.push(a); renderAttachments(); }
function renderAttachments() {
  const el = $('#attachments'); el.innerHTML = '';
  attachments.forEach((a, i) => {
    const c = document.createElement('span'); c.className = 'chip';
    c.textContent = a.kind + ': ' + a.name;
    const x = document.createElement('b'); x.textContent = '×';
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x); el.appendChild(c);
  });
}
$('#toolFile').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  if (f.type.startsWith('image/')) {
    if (!modelImage) {
      note('This model cannot see images — load a vision model to attach pictures.');
      e.target.value = ''; return;
    }
    const dataUrl = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(f); });
    addAttachment({ kind: 'image', name: f.name, dataUrl });
  } else {
    // Only text can be attached as text. Decoding a PDF or a zip with .text() yields
    // mojibake that looks like content and reaches the model as if it were the document --
    // worse than refusing, because nothing about the reply reveals it. Detect by decoding
    // strictly and by looking for the NUL bytes no text file contains.
    const buf = new Uint8Array(await f.arrayBuffer());
    let text = null;
    try {
      text = new TextDecoder('utf-8', { fatal: true }).decode(buf);
      if (text.indexOf('\u0000') >= 0) text = null;
    } catch (_) { text = null; }
    if (text === null) {
      note('“' + f.name + '” is not a text file — attach plain text (.txt, .md, .csv, code) '
           + 'or paste the part you want the model to read.');
      e.target.value = ''; return;
    }
    const clipped = text.length > 20000;
    addAttachment({ kind: 'file', name: f.name, text: text.slice(0, 20000),
                    clipped });
    if (clipped) note('“' + f.name + '” is long — the model gets the first '
                      + fmt(20000) + ' characters.');
  }
  e.target.value = '';
};
$('#toolUrl').onclick = async () => {
  const url = prompt('Fetch a URL and add its text to the message:');
  if (!url) return;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const t = await r.text();
    // Read the page as a document and keep only what a reader would: the title plus the
    // visible text. Scripts, styles and site chrome are dropped, and the raw HTML never
    // reaches the conversation — the bubble shows a chip, the model gets plain text.
    const doc = new DOMParser().parseFromString(t, 'text/html');
    doc.querySelectorAll('script,style,noscript,template,svg,iframe,nav,footer,header,form,aside')
       .forEach(n => n.remove());
    const title = ((doc.querySelector('title') || {}).textContent || '').trim();
    let text = (doc.body || doc.documentElement).textContent.replace(/\s+/g, ' ').trim();
    if (title) text = title + '\n\n' + text;
    if (!text) throw new Error('no readable text on that page');
    const clipped = text.length > 20000;
    addAttachment({ kind: 'url', name: url, text: text.slice(0, 20000), clipped });
    note('Added the page text (' + fmt(text.length) + (clipped ? ', first ' + fmt(20000)
         + ' characters go to the model' : '') + ') — it goes to the model, not into the bubble.');
  } catch (e) { alert('Fetch failed (the site may block cross-origin requests): ' + e.message); }
};
$('#toolCam').onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const dlg = $('#camDlg'), v = $('#cam');
    v.srcObject = stream; dlg.showModal();
    const stop = () => { stream.getTracks().forEach(t => t.stop()); dlg.close(); };
    $('#camCancel').onclick = stop;
    $('#camShot').onclick = () => {
      const c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      addAttachment({ kind: 'image', name: 'camera.png', dataUrl: c.toDataURL('image/png') });
      stop();
    };
  } catch (e) { alert('Camera unavailable: ' + e.message); }
};

// ---- chat ----
function newConv(activate = true) {
  const c = { id: 'c' + Date.now() + Math.random().toString(36).slice(2, 6),
              title: 'New chat', messages: [], updated: Date.now() };
  convs.unshift(c); if (activate) curId = c.id;
  saveConvs(); renderConvs(); render();
  return c;
}
function ensureConv() { return current() || newConv(); }
function titleFrom(text) {
  const t = (text || '').replace(/\s+/g, ' ').trim();
  return t ? (t.length > 38 ? t.slice(0, 38) + '…' : t) : 'New chat';
}
function renderConvs() {
  const el = $('#convList'); el.innerHTML = '';
  if (!convs.length) { el.innerHTML = '<p class="hint" style="padding:6px 10px">no conversations yet</p>'; return; }
  convs.forEach(c => {
    const d = document.createElement('div');
    d.className = 'conv' + (c.id === curId ? ' active' : '');
    const t = document.createElement('span'); t.className = 't'; t.textContent = c.title;
    const x = document.createElement('span'); x.className = 'x'; x.textContent = '✕';
    x.title = 'Delete conversation';
    x.onclick = (e) => {
      e.stopPropagation();
      convs = convs.filter(v => v.id !== c.id);
      if (curId === c.id) curId = convs.length ? convs[0].id : null;
      saveConvs(); renderConvs(); render();
    };
    d.onclick = () => { curId = c.id; renderConvs(); render(); };
    d.append(t, x); el.appendChild(d);
  });
}

// Edit a message in place — either role. Prose gets a plain box holding exactly what the
// message says; code blocks are edited in the block itself (see wireRunButtons), which is
// why this deliberately does NOT open the Markdown for a reply full of code.
function editMessage(m, body) {
  if (body.querySelector('textarea')) return;                 // already editing
  const prev = body.cloneNode(true);
  const ta = document.createElement('textarea');
  ta.className = 'editbox'; ta.value = m.content || '';
  ta.rows = Math.min(20, Math.max(3, (m.content || '').split('\n').length + 1));
  const bar = document.createElement('div'); bar.className = 'editbar';
  const ok = document.createElement('button'); ok.type = 'button';
  ok.className = 'primary'; ok.textContent = 'Save';
  const no = document.createElement('button'); no.type = 'button'; no.textContent = 'Cancel';
  bar.append(ok, no);
  body.textContent = ''; body.append(ta, bar);
  ta.focus();
  const restore = () => { body.replaceWith(prev); };
  no.onclick = restore;
  ok.onclick = () => {
    m.content = ta.value;
    if (m.role === 'assistant') m.think = m.think === undefined ? undefined : m.think;
    saveConvs(); renderConvs(); render();
  };
  ta.addEventListener('keydown', e => {
    if (e.key === 'Escape') restore();
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ok.click();
  });
}

// Remove one message. Emptying a conversation this way removes the conversation -- a
// conversation with nothing in it is not a thing anyone wants left in the list -- so that
// case asks first, and a cancelled ask leaves the message alone.
function deleteMessage(m) {
  const conv = current();
  if (!conv) return;
  const i = conv.messages.indexOf(m);
  if (i < 0) return;
  if (conv.messages.length === 1) {
    if (!confirm('This is the last message. Deleting it deletes the whole conversation.\n\nDelete the conversation?')) return;
    convs = convs.filter(c => c.id !== conv.id);
    curId = convs.length ? convs[0].id : null;
    saveConvs(); renderConvs(); render();
    return;
  }
  // A reply still being written is driven by an index into this array; deleting it (or
  // anything before it) would leave the stream appending to the wrong message. Say so
  // rather than corrupt the conversation -- it is over in a few seconds.
  if (streaming && streaming.convId === conv.id && streaming.idx >= i) {
    note('That reply is still being written — wait for it to finish.');
    return;
  }
  conv.messages.splice(i, 1);
  saveConvs(); renderConvs(); render();
}

// The sidebar is a column on a desktop and a drawer on a phone: the same markup, with the
// narrow layout in the stylesheet and this toggling one class. Tapping the backdrop or
// picking a conversation closes it, because on a phone it covers what you came to read.
function toggleSidebar(open) {
  const app = document.getElementById('app');
  const want = open === undefined ? !app.classList.contains('nav-open') : !!open;
  app.classList.toggle('nav-open', want);
}

function render() {
  const el = $('#messages'); el.innerHTML = '';
  const conv = current();
  const messages = conv ? conv.messages : [];
  if (!messages.length) {
    el.innerHTML = '<div class="msg bot"><div class="who">AI</div><div class="body">' +
      'Pick a model in ⚙ Settings, load it, and start chatting.</div></div>';
    return;
  }
  messages.forEach((m, i) => {
    const node = messageNode(m);
    node.dataset.idx = i;
    el.appendChild(node);
  });
  paintSlowNote();                       // lives in this list, so it is re-emitted with it
  el.scrollTop = el.scrollHeight;
}

// Which message is "current".
//
// A click names one. Without a click it is the LAST message still in view, which is what
// someone reading a conversation is looking at. Deliberately not recomputed into a scroll:
// letting a scroll change the current message and letting the current message drive a
// scroll is a loop, and this page already had one bug where following the reply took the
// view away from whoever was reading it.
let clickedMsgIdx = null;
function currentMsgIdx() {
  if (clickedMsgIdx != null) return clickedMsgIdx;
  const el = $('#messages');
  if (!el) return null;
  const bottom = el.getBoundingClientRect().bottom;
  let last = null;
  el.querySelectorAll('.msg').forEach(n => {
    if (n.getBoundingClientRect().top < bottom) last = n;
  });
  return last ? Number(last.dataset.idx) : null;
}

// The rule for where a browser sits: the current message's panel starts at the top of the
// view, and whatever does not fit runs on below it. Everything else stays in the flow, so
// the panels above and below are pushed by this one rather than competing with it.
function alignCurrentWeb() {
  const el = $('#messages');
  const idx = currentMsgIdx();
  if (!el || idx == null) return;
  const node = el.querySelector('.msg[data-idx="' + idx + '"]');
  const panel = node && node.querySelector('.webtabs');
  if (!panel) return;
  el.scrollTop += panel.getBoundingClientRect().top - el.getBoundingClientRect().top;
}

// One message as DOM. The body is either rendered fresh (finished message) or driven
// incrementally by `live` while a reply is being streamed.
// Following a reply is for someone who is AT the reply. Scrolling to the bottom on every
// token took the page back the moment anyone scrolled up to read something earlier -- during
// a long answer that is every 20ms, so it was not a jump but a wall. This asks first: within
// a screen's tail counts as following, and anywhere else counts as reading, which is left
// alone until the person comes back down on their own.
const FOLLOW_SLACK_PX = 80;
function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= FOLLOW_SLACK_PX;
}
function keepAtBottom(el, wasFollowing) {
  if (wasFollowing) el.scrollTop = el.scrollHeight;
}

function messageNode(m, live) {
  const d = document.createElement('div'); d.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');
  const w = document.createElement('div'); w.className = 'who'; w.textContent = m.role === 'user' ? 'You' : 'AI';
  const b = document.createElement('div'); b.className = 'body';
  const tools = document.createElement('div'); tools.className = 'mtools';
  const ed = document.createElement('button');
  ed.type = 'button'; ed.className = 'edit'; ed.textContent = '✎';
  ed.title = 'Edit this message';
  ed.onclick = () => editMessage(m, b);
  const x = document.createElement('button');
  x.type = 'button'; x.className = 'del'; x.textContent = '✕';
  x.title = 'Delete this message';
  x.onclick = () => deleteMessage(m);
  tools.append(ed, x);
  d.append(w, b, tools);
  if (m.role === 'user') {
    b.textContent = m.content;
  } else if (!live) {
    fillBody(b, m, live);
  }
  // `live` here is a FLAG, not the live object: a reply that is about to stream gets its
  // body from the caller, which owns the pieces it will keep updating. Calling fillBody
  // with the flag built a second answer element -- `live.det` on a boolean is undefined, so
  // it took the path that creates one -- and the reply then had two, the visible text going
  // into the one appended last.
  if (live) tools.style.display = 'none';        // nothing to edit or delete mid-stream
  // Double-click to edit -- but ONLY on a message that has no blocks of its own.
  //
  // A reply is rendered as blocks, and each of those handles its own double-click: a
  // paragraph opens as text, a table cell as whatever that cell holds. A handler here as
  // well would fire on the same double-click and replace the whole body with a textarea,
  // taking the block editor with it -- which is exactly what happened, and looked like the
  // cell editor's Save and Cancel doing nothing.
  //
  // What the person typed has no blocks: it is one piece of text, and this is its editor.
  else if (m.role === 'user') b.addEventListener('dblclick', () => editMessage(m, b));
  (m.attachments || []).forEach(a => {
    if (a.dataUrl) { const im = new Image(); im.src = a.dataUrl;
                     wireImage(im, a.name || 'image.png'); b.appendChild(im); }
    else { const p = document.createElement('div'); p.className = 'hint'; p.textContent = a.kind + ': ' + a.name; b.appendChild(p); }
  });
  // finished replies keep their final decode stats as a quiet footer (the same line that
  // shows live counts while streaming); user messages and failed replies have none
  if (m.role === 'assistant' && m.stats && m.stats.tok_s != null) {
    const f = document.createElement('div'); f.className = 'tokrate';
    // The split, when there is one: a reply slow because the device is busy and one slow
    // because the host is between steps read the same from outside and are different
    // problems. Only shown when a token cost enough to be worth explaining.
    let extra = '';
    const g = m.stats.gpu_ms, k = m.stats.pick_ms;
    if (g != null && k != null && (g + k) >= 20) {
      extra = '  ·  GPU ' + Number(g).toFixed(0) + ' ms + host ' + Number(k).toFixed(0) + ' ms';
      // The mean alone cannot say whether a reply was slow throughout or only until it warmed
      // up, and those need different answers. Show the first and last tenth when they differ
      // enough to mean something.
      const h = m.stats.gpu_ms_head, t = m.stats.gpu_ms_tail;
      if (h != null && t != null && Math.max(h, t) >= 1.25 * Math.min(h, t)) {
        extra += ' (' + Number(h).toFixed(0) + ' → ' + Number(t).toFixed(0) + ')';
      }
    }
    // Which decode loop ran. `replay` is the captured step; `grow` is the fallback that
    // re-runs a forward per token, and a model on it is slow for a reason no kernel change
    // will reach.
    if (m.stats.path && m.stats.path !== 'replay') extra += '  ·  ' + m.stats.path;
    f.textContent = (m.stats.n || 0) + ' tokens · ' + Number(m.stats.tok_s).toFixed(1)
                  + ' tok/s' + extra;
    b.appendChild(f);
  }
  return d;
}

// A reply may begin with reasoning the model wrote out ("<think>…</think>"). Render it folded
// away — present, expandable, but never mixed into the answer. Also recover replies saved
// by older builds that kept only part of the template text (a dangling "</think>" with no
// opener): everything before the closer WAS the thinking, so put the opener back.
// Messages saved before the stream carried channel labels keep reasoning inline in
// `content`. Parsing them back is fine — they are complete, so both tags are present or
// neither is. Live replies never come through here.
function splitThink(txt) {
  txt = typeof txt === 'string' ? txt : '';
  if (!txt.includes('<think>') && txt.includes('</think>')) txt = '<think>' + txt;
  const close = txt.indexOf('</think>');
  if (txt.startsWith('<think>') && close >= 7)
    return { think: txt.slice(7, close).trim(), rest: txt.slice(close + 8).replace(/^\s+/, ''), open: false };
  if (txt.startsWith('<think>'))
    return { think: txt.slice(7).trim(), rest: '', open: true };     // still thinking
  return { think: null, rest: txt, open: false };
}

// Fill a message body from raw reply text, incrementally when `live` is given: the same
// detail/text nodes are updated in place, so a person who opened the thinking box while the
// reply streams keeps it open — the DOM is never rebuilt under their cursor.
// `live` shape: {det, sum, pre, ans, touched, collapsed}
// ---- looking at an image properly ----------------------------------------------------
// A plot comes out at whatever size fits the conversation column, which for anything with
// axis labels is too small to read. Double-click opens it at full size; the overlay also
// offers to save it, because a chart someone just generated is usually wanted elsewhere.
function wireImage(img, name) {
  img.classList.add('zoomable');
  img.title = 'Double-click to open';
  img.addEventListener('dblclick', () => openLightbox(img.src, name));
}

function openLightbox(src, name) {
  const box = document.createElement('div');
  box.className = 'lightbox';
  const big = new Image(); big.src = src; big.className = 'lbimg';
  const bar = document.createElement('div'); bar.className = 'lbbar';
  const dl = document.createElement('a');
  dl.href = src; dl.download = name || ('plot-' + Date.now() + '.png');
  dl.textContent = '⤓ Save'; dl.className = 'lbbtn';
  const close = document.createElement('button');
  close.type = 'button'; close.className = 'lbbtn'; close.textContent = '✕ Close';
  bar.append(dl, close);
  box.append(bar, big);
  document.body.appendChild(box);
  const shut = () => { box.remove(); document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') shut(); };
  close.onclick = shut;
  // Clicking the backdrop closes; clicking the image or the bar does not.
  box.addEventListener('click', (e) => { if (e.target === box) shut(); });
  document.addEventListener('keydown', onKey);
}

// ---- running the Python in a reply ---------------------------------------------------
// A model that answers with code should let you press it. The code runs in its OWN Pyodide,
// in its own worker: the model's runtime is holding gigabytes of weights and is busy
// decoding, and someone's `while True` must not be able to take the chat down with it.
//
// It is booted at page load with the packages from Settings, not on the first Run. Loading
// pandas the first time someone presses the button is a ten-second pause with nothing to
// look at, and the person pressing Run has already decided they want this.
// What the page asks every model for, because the page is what renders the answer. Kept
// short and concrete: a long style guide costs context on every turn and models follow the
// specific instructions better than the general ones.
const UI_SYSTEM = [
  'Format every reply as Markdown.',
  'Put code in fenced blocks with a language tag (```python).',
  'Write mathematics as LaTeX: $inline$ and $$display$$.',
  'Use tables, lists and headings where they make the answer clearer.',
].join(' ');

const PYPKG_KEY = 'webtorch.pyPackages';
const PYON_KEY = 'webtorch.pyEnabled';
const PY_DEFAULT = 'numpy, pandas, matplotlib';
let pyWorker = null, pySeq = 0, pyState = 'off', pyVersion = null;
const pyPending = new Map();

function pyPackages() {
  const v = localStorage.getItem(PYPKG_KEY);
  return (v === null ? PY_DEFAULT : v).split(/[\s,]+/).filter(Boolean);
}
function pyEnabled() { return localStorage.getItem(PYON_KEY) !== '0'; }

function pyCall(cmd, args) {
  if (!pyWorker) return Promise.reject(new Error('the Python runtime is off'));
  const id = ++pySeq;
  return new Promise((res, rej) => {
    pyPending.set(id, { res, rej });
    pyWorker.postMessage({ id, cmd, args });
  });
}

function pyStart() {
  if (pyWorker || !pyEnabled()) return;
  pyWorker = new Worker('pyworker.js?v=1fea30ec14');
  pyWorker.onmessage = (e) => {
    const m = e.data || {};
    if (m.type === 'res') {
      const p = pyPending.get(m.id); pyPending.delete(m.id);
      if (p) (m.error ? p.rej(new Error(m.error)) : p.res(m.res));
    } else if (m.type === 'state') {
      pyState = m.state;
      if (m.version) pyVersion = m.version;
      const el = $('#pyState');
      if (el) el.textContent = m.state === 'ready'
        ? 'ready · ' + (m.packages || []).join(', ') : m.state + '…';
      // Only the Python buttons. JavaScript runs in a sandboxed frame that is always
      // there, so its blocks are runnable whether or not the interpreter is up.
      document.querySelectorAll('.runbtn[data-runner="python"]')
              .forEach(b => { b.disabled = pyState !== 'ready'; });
    }
  };
  pyCall('boot', { packages: pyPackages() }).catch(() => {});
}
function pyStop() {
  if (!pyWorker) return;
  pyWorker.terminate(); pyWorker = null; pyState = 'off';
  const el = $('#pyState'); if (el) el.textContent = 'off';
  document.querySelectorAll('.runbtn[data-runner="python"]')
          .forEach(b => { b.disabled = true; });
}

// Which environment a fenced block runs in, or null for one that does not run. THIS is
// where the language is decided -- not at the place the button is attached. It used to be
// the other way round, and `runBlock` fed whatever it was handed to Pyodide: widening that
// gate by one language would have sent JavaScript to Python and got back a SyntaxError.
// A block in any other language gets no button at all.
function langRunner(lang) {
  if (/^(py|python|python3)$/i.test(lang)) return 'python';
  if (/^(js|javascript|mjs|node)$/i.test(lang)) return 'javascript';
  return null;
}

// Attach a Run control to every runnable block in `root` that has not got one. Called after
// a reply is rendered rather than woven into the renderer, so the sanitiser still sees plain
// Markdown output and no button can arrive from the model's own text.
function wireRunButtons(root, msg) {
  let idx = -1;
  root.querySelectorAll('pre > code').forEach(code => {
    const pre = code.parentNode;
    idx += 1;                                   // every fence, so the index matches the source
    if (pre.dataset.run) return;
    pre.dataset.run = '1';
    const wrap = document.createElement('div'); wrap.className = 'codewrap';
    pre.parentNode.insertBefore(wrap, pre); wrap.appendChild(pre);

    // Editable in place. Not a textarea of Markdown: the person wants to fix a line of
    // Python, and being handed the backticks to edit around is a worse tool than the block
    // they are already looking at. `plaintext-only` keeps pasted rich text from arriving as
    // markup, and the edit is written back into the message's own fence on blur.
    if (msg) {
      code.setAttribute('contenteditable', 'plaintext-only');
      code.spellcheck = false;
      code.title = 'Click to edit';
      code.addEventListener('focus', () => wrap.classList.add('editing'));
      code.addEventListener('blur', () => {
        wrap.classList.remove('editing');
        const blk = wrap.closest('.blk');
        if (blk) {
          // The block IS the fence, so its new source is the fence rebuilt around the code.
          const src = mdBlocks(msg.content)[+blk.dataset.i] || '';
          const open = (src.match(/^\s*(?:`{3,}|~{3,})[^\n]*/) || [''])[0];
          const close = (open.match(/`{3,}|~{3,}/) || ['```'])[0];
          const body = code.textContent.replace(/\n+$/, '');
          if (setBlock(msg, +blk.dataset.i, open + '\n' + body + '\n' + close)) saveConvs();
        }
        rehighlight(code);
      });
      // Enter must not end the edit and Tab must indent -- this is a code box, not a form.
      code.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
          e.preventDefault();
          document.execCommand('insertText', false, '    ');
        } else if (e.key === 'Escape') { code.blur(); }
      });
    }

    const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || '';
    const runner = langRunner(lang);
    if (!runner) return;
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'runbtn';
    btn.dataset.runner = runner;
    btn.title = runner === 'python' ? 'Run this Python' : 'Run this JavaScript';
    btn.textContent = '▶';
    btn.disabled = runner === 'python' && pyState !== 'ready';
    btn.onclick = () => runBlock(code.textContent, wrap, btn, lang);
    wrap.appendChild(btn);
  });
}

// Colour the block again after an edit, keeping the language it was tagged with.
function rehighlight(code) {
  if (typeof hljs === 'undefined') return;
  const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || '';
  const text = code.textContent;
  try {
    code.innerHTML = lang && hljs.getLanguage(lang)
      ? hljs.highlight(text, { language: lang }).value
      : hljs.highlightAuto(text).value;
  } catch (e) { code.textContent = text; }
}

// Split Markdown into top-level blocks, source and all. This is the unit the person edits:
// one paragraph, one list, one table, one fenced block. A fence is a block whatever is
// inside it (blank lines included); everything else is separated by blank lines.
//
// Splitting the SOURCE rather than walking the rendered DOM is what lets an edit be written
// back: block n of the render is block n of the Markdown, so saving is an array assignment
// and a join, with no counting and no guessing which fence was which.
function mdBlocks(src) {
  const lines = String(src == null ? '' : src).split('\n');
  const out = [];
  let buf = [], fence = null;
  const flush = () => { if (buf.join('\n').trim()) out.push(buf.join('\n')); buf = []; };
  for (const line of lines) {
    const f = line.match(/^\s*(`{3,}|~{3,})/);
    if (fence !== null) {
      buf.push(line);
      if (f && line.trim().indexOf(fence) === 0) { out.push(buf.join('\n')); buf = []; fence = null; }
      continue;
    }
    if (f) { flush(); fence = f[1]; buf.push(line); continue; }
    if (!line.trim()) { flush(); continue; }
    buf.push(line);
  }
  // An unterminated fence is still a block -- a reply can be cut off mid-code.
  if (fence !== null) out.push(buf.join('\n')); else flush();
  return out;
}

// Write block `n` back and rebuild the message. Blocks are rejoined with a blank line,
// which is what separated them in the first place.
function setBlock(msg, n, text) {
  const bl = mdBlocks(msg.content);
  if (n < 0 || n >= bl.length) return false;
  if (bl[n] === text) return false;
  bl[n] = text;
  msg.content = bl.join('\n\n');
  return true;
}

// Output is something the reader chose to produce, so they get to take it back -- a long
// traceback or a figure sitting under the code is theirs to clear. A traceback lands in the
// same panel as ordinary output (it is a result too), so one control covers both.
function addRunoutClose(panel) {
  if (panel.querySelector(':scope > .x')) return;
  const x = document.createElement('button');
  x.type = 'button'; x.className = 'x'; x.textContent = '\u2715';
  x.title = 'Remove this output';
  x.onclick = (e) => { e.stopPropagation(); panel.remove(); };
  panel.appendChild(x);
}

// Pressing Run produces a panel and NOTHING else. What it printed, what it returned, what
// it drew: all of it lives in this one DOM node next to the block, and none of it is
// written to the message, to storage, or to the history the model is sent -- a redraw
// wipes it, which is the whole of its lifetime. It is the reader trying something out, not
// a turn in the conversation, and the model has its own tools for when it wants to run
// something and keep the answer. (Editing the block IS written back, but that is the code,
// not the run.) Anything added here goes in the panel; nothing goes in `msg`.
async function runBlock(code, wrap, btn, lang) {
  const runner = langRunner(lang) || 'python';
  let panel = wrap.nextElementSibling;
  if (!panel || !panel.classList.contains('runout')) {
    panel = document.createElement('div'); panel.className = 'runout';
    wrap.parentNode.insertBefore(panel, wrap.nextSibling);
  }
  panel.textContent = 'running…'; panel.className = 'runout';
  btn.disabled = true;
  try {
    if (runner === 'javascript') await showJsRun(code, panel);
    else await showPyRun(code, panel);
    if (!panel.childNodes.length) panel.textContent = '(no output)';
  } catch (e) {
    panel.classList.add('bad'); panel.textContent = String(e.message || e);
  } finally {
    btn.disabled = runner === 'python' && pyState !== 'ready';
    addRunoutClose(panel);
  }
}

// Python, in the second interpreter. Output first, then the value of the last expression,
// then whatever it drew. Errors are shown in the same place: a traceback is a result too.
async function showPyRun(code, panel) {
  const r = await pyCall('run', { code });
  panel.textContent = '';
  if (r.out) {
    const p = document.createElement('pre'); p.textContent = r.out.replace(/\n$/, '');
    panel.appendChild(p);
  }
  if (r.err) {
    const p = document.createElement('pre'); p.className = 'stderr';
    p.textContent = r.err.replace(/\n$/, ''); panel.appendChild(p);
  }
  if (r.value != null && r.value !== 'None') {
    const p = document.createElement('pre'); p.className = 'val'; p.textContent = r.value;
    panel.appendChild(p);
  }
  if ((r.images || []).length) panel.appendChild(imagesBlock(r.images));
  if (r.error) {
    panel.classList.add('bad');
    const p = document.createElement('pre'); p.textContent = r.error; panel.appendChild(p);
  }
}

// JavaScript, in the same sandboxed frame the tools use -- an opaque origin with no way
// back to this page. Nothing about it goes through Pyodide.
//
// One button has to serve both kinds of script, so the DOM is THERE and whether a page is
// shown is decided by what the code did with it: a calculation leaves the body empty and
// gets output and a value, a script that builds something gets the page as well. The model
// has two separate tools instead, because it has to say which it means before it runs.
async function showJsRun(code, panel) {
  const r = await runSandboxedJs(code, JS_TIMEOUT_MS, 'auto');
  panel.textContent = '';
  if ((r.logs || []).length) panel.appendChild(consoleBlock(r.logs));
  if (r.result != null) {
    const p = document.createElement('pre'); p.className = 'val'; p.textContent = r.result;
    panel.appendChild(p);
  }
  if (r.view) panel.appendChild(jsViewFrame({ name: 'page', html: r.view, height: 0 }));
  if (r.error) {
    panel.classList.add('bad');
    const p = document.createElement('pre'); p.textContent = r.error; panel.appendChild(p);
  }
}

// ---- reply rendering ----------------------------------------------------------------
// A model's answer is Markdown: fenced code, tables, lists, and — for anything technical —
// LaTeX. Rendering it as plain text showed people the backticks and the dollar signs.
//
// marked does the Markdown (GFM: tables, task lists, strikethrough), KaTeX the maths,
// highlight.js the code, and DOMPurify decides what is allowed to reach the DOM. That last
// one is not optional: marked deliberately does not sanitise, and the text being rendered
// comes out of a language model. Checked, not assumed -- `javascript:` and `data:text/html`
// hrefs and an `onerror` attribute all come back stripped, while `https:` links survive.
//
// Everything is loaded from a CDN, so all of it may simply be absent: offline, or blocked.
// `renderMarkdown` then returns null and the caller writes the text as it always did. The
// chat is usable either way; it is only prettier with them.
// Built on first use and kept. A MISSING library is not cached as a failure: these are
// `defer` scripts, so the first message rendered can easily arrive before they do, and
// latching then would leave the chat in plain text for the rest of the session. Only a
// library that is present and throws is given up on.
let MD = null, mdBroken = false;
function markdown() {
  if (MD || mdBroken) return MD;
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') return null;
  try {
    const m = new marked.Marked({ gfm: true, breaks: true });
    if (typeof markedKatex === 'function' && typeof katex !== 'undefined') {
      // `nonStandard` also accepts $…$ with no space around it, which is how models write it.
      m.use(markedKatex({ throwOnError: false, nonStandard: true }));
    }
    if (typeof hljs !== 'undefined') {
      m.use({ renderer: { code(token) {
        const src = token.text != null ? token.text : String(token);
        const lang = ((token.lang || '') + '').trim().split(/\s+/)[0];
        let body;
        try {
          body = lang && hljs.getLanguage(lang)
               ? hljs.highlight(src, { language: lang }).value
               : hljs.highlightAuto(src).value;
        } catch (e) { body = null; }
        if (body == null) return false;                      // let marked render it plainly
        return '<pre><code class="hljs' + (lang ? ' language-' + lang : '') + '">'
             + body + '</code></pre>';
      } } });
    }
    MD = m;
  } catch (e) { mdBroken = true; }
  return MD;
}
// KaTeX emits MathML and a pile of positioned spans; the allow-list has to cover them or the
// formula arrives as loose letters.
const MD_TAGS = ['math', 'semantics', 'annotation', 'mrow', 'mi', 'mo', 'mn', 'ms', 'mtext',
                 'msup', 'msub', 'msubsup', 'mfrac', 'msqrt', 'mroot', 'mover', 'munder',
                 'munderover', 'mtable', 'mtr', 'mtd', 'mspace', 'mpadded', 'mphantom',
                 'menclose', 'mstyle', 'svg', 'path', 'line'];
const MD_ATTR = ['aria-hidden', 'style', 'class', 'encoding', 'displaystyle', 'scriptlevel',
                 'mathvariant', 'stretchy', 'width', 'height', 'viewBox', 'preserveAspectRatio',
                 'd', 'x1', 'x2', 'y1', 'y2'];
// Two LaTeX forms the extension does not pick up, and models write both.
//
// `$$` on its own line with the formula under it is the common one -- and it fails BECAUSE
// of `breaks: true`: without a blank line the paragraph tokenizer swallows the whole run and
// turns the newlines into <br>, so the block-level `$$` rule never sees a block to match.
// Putting the delimiters and the formula on one line is a form that does render, and says
// the same thing.
//
// `\[ … \]` and `\( … \)` are not handled at all, and worse than not handled: marked
// treats the backslash as an escape and drops it, so `\[ a^2 \]` reads as `[ a^2 ]`.
//
// Code is left exactly as written -- a fenced block or a span of inline code that happens to
// contain `$$` means the characters, not a formula.
function normalizeMath(src) {
  const parts = String(src).split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  for (let i = 0; i < parts.length; i += 2) {          // odd indices are the code spans
    parts[i] = parts[i]
      .replace(/\\\[([\s\S]*?)\\\]/g, (_, f) => '$$' + f.trim() + '$$')
      .replace(/\\\(([\s\S]*?)\\\)/g, (_, f) => '$' + f.trim() + '$')
      // The body may not itself contain `$$`, and the closer may sit at the end of the
      // formula's own line. Both come from what models actually emit: one wrote
      //     $$\n<formula> $$\n$$\n<formula>\n$$
      // -- the first block closed on its own line's end -- and a pattern that only looked
      // for a newline before the closer ran straight THROUGH that boundary, swallowing the
      // `$$` between the two blocks. The first formula came out truncated and the second
      // was left as literal `$$` delimiters in the reply.
      .replace(/\$\$[ \t]*\n((?:(?!\$\$)[\s\S])*?)\s*\$\$/g,
               (_, f) => '$$' + f.trim() + '$$');
  }
  return parts.join('');
}

function renderMarkdown(text) {
  const m = markdown();
  if (!m) return null;
  try {
    return DOMPurify.sanitize(m.parse(normalizeMath(text)),
                              { ADD_TAGS: MD_TAGS, ADD_ATTR: MD_ATTR });
  } catch (e) { return null; }
}
// Put `text` into `el` as rendered Markdown, or as plain text when rendering is unavailable.
// A finished message is rendered BLOCK BY BLOCK, each in its own container that can be
// edited on its own. A reply still streaming is rendered as one piece: it changes every
// token, nothing in it is editable yet, and re-splitting it per token buys nothing.
// Re-parsing the whole reply for every token is O(n^2) over a reply, and on a fast model it
// is the largest cost in the loop: measured on a 0.6B, 109-112 tok/s through the API against
// 91-95 through the chat, for the same prompt and the same model. The tokens still arrive
// one at a time -- only the PARSE is throttled, so the screen is at most one interval behind,
// and the `render()` that ends a stream draws the final text whatever the last tick did.
//
// The counter and the thinking pane are not throttled with it: those are textContent writes,
// and they are what makes the reply feel live.
const STREAM_RENDER_MS = 60;
const liveRender = { t: 0, timer: 0, el: null, text: '' };
function flushLiveRender() {
  if (liveRender.timer) { clearTimeout(liveRender.timer); liveRender.timer = 0; }
  liveRender.t = performance.now();
  const el = liveRender.el;
  if (el && el.isConnected) setRendered(el, liveRender.text);
}
// Only what changed. A reply grows by appending, so each tick alters the LAST block and
// leaves every earlier one final -- re-rendering the whole reply meant re-parsing, re-
// sanitising, re-typesetting and re-highlighting text that had not moved, and doing it again
// for every token after that. Splitting into blocks is a line scan; rendering one block is a
// fraction of rendering the reply, and the fraction shrinks as the reply grows.
//
// The structure is the one a FINISHED message uses (`.blocks` > `.blk`), so nothing shifts
// on screen when the stream ends; `.live` only suppresses the hover that marks a block as
// editable, which it is not until then.
//
// An unterminated fence changes the split as it arrives -- `mdBlocks` yields one block while
// it is open, and the same text may resolve differently once the closing fence lands. That
// is why this compares SOURCES rather than trusting positions: a block whose source changed
// is re-rendered wherever it sits.
function setRenderedStream(el, text) {
  const src = mdBlocks(text);
  const prev = el.__blocks || null;
  if (!prev) { el.textContent = ''; el.classList.add('blocks', 'live'); }
  for (let i = 0; i < src.length; i++) {
    if (prev && prev[i] === src[i] && el.childNodes[i]) continue;
    const node = document.createElement('div');
    node.className = 'blk';
    node.innerHTML = renderMarkdown(src[i]) || '';
    wireRunButtons(node, null);
    if (el.childNodes[i]) el.replaceChild(node, el.childNodes[i]);
    else el.appendChild(node);
  }
  while (el.childNodes.length > src.length) el.removeChild(el.lastChild);
  el.__blocks = src;
}

function setRenderedLive(el, text) {
  liveRender.el = el; liveRender.text = text;
  const due = performance.now() - liveRender.t;
  if (due >= STREAM_RENDER_MS) { flushLiveRender(); return; }
  if (!liveRender.timer) liveRender.timer = setTimeout(flushLiveRender, STREAM_RENDER_MS - due);
}
function resetLiveRender() {
  if (liveRender.timer) { clearTimeout(liveRender.timer); liveRender.timer = 0; }
  if (liveRender.el) { delete liveRender.el.__blocks; liveRender.el.classList.remove('live'); }
  liveRender.t = 0; liveRender.el = null; liveRender.text = '';
}

function setRendered(el, text, msg) {
  if (renderMarkdown('') == null) {            // no renderer: plain text, as before
    el.textContent = text; el.classList.remove('md'); el.classList.remove('blocks');
    return;
  }
  el.classList.add('md');
  if (!msg) {                                  // streaming
    setRenderedStream(el, text);
    return;
  }
  el.classList.remove('live');
  el.classList.add('blocks');
  el.textContent = '';
  mdBlocks(text).forEach((src, i) => el.appendChild(blockNode(msg, i, src)));
  el.querySelectorAll('img').forEach(im => wireImage(im, 'image.png'));
  if (!el.childNodes.length) el.appendChild(blockNode(msg, 0, ''));
}

// One block: its rendering, and the means to change it. Code is edited in the block itself
// (see wireRunButtons); everything else opens as the few lines of Markdown that produced
// it -- which is the block's own source, not the whole message.
function blockNode(msg, i, src) {
  const d = document.createElement('div');
  d.className = 'blk'; d.dataset.i = String(i);
  d.innerHTML = renderMarkdown(src) || '';
  wireRunButtons(d, msg);
  const isCode = !!d.querySelector('pre > code');
  const table = d.querySelector('table');
  if (table) {
    wireTable(msg, i, d, table);
  } else if (!isCode) {
    d.title = 'Double-click to edit';
    d.addEventListener('dblclick', (e) => {
      if (e.target.closest('.runout, .blockedit, .cellpop')) return;
      editBlock(msg, i, d);
    });
  }
  return d;
}

// A table is edited AS A TABLE. Double-clicking one and being handed a row of pipes is not
// editing a table, it is editing a description of one -- so the cells themselves take the
// text, and the Markdown is rebuilt from them.
//
// Structure is the exception: adding a column, removing a row, changing the alignment. There
// is no good direct gesture for those, so the block carries a button to the source, which is
// the right tool for exactly that job and the wrong one for fixing a number.
function wireTable(msg, i, node, table) {
  const grid = tableCells(mdBlocks(msg.content)[i] || '');
  const rows = [table.querySelector('thead tr')].concat([...table.querySelectorAll('tbody tr')])
                                                .filter(Boolean);
  rows.forEach((tr, r) => [...tr.children].forEach((cell, c) => {
    // The cell's SOURCE, kept on the cell. Rebuilding the Markdown from what the cell
    // displays cannot work: a cell holding $\phi^n$ renders to KaTeX, whose textContent is
    // the MathML and the HTML one after the other, and writing that back destroys the row.
    cell.dataset.src = (grid[r] && grid[r][c] != null) ? grid[r][c] : cell.textContent.trim();
    wireCell(msg, i, table, cell);
  }));
  const src = document.createElement('button');
  src.type = 'button'; src.className = 'srcbtn';
  src.textContent = '⌗'; src.title = 'Edit the source (add or remove rows and columns)';
  src.onclick = () => editBlock(msg, i, node);
  node.classList.add('tblblk');
  node.appendChild(src);
}

// What a cell holds decides how it is edited. Plain words are edited where they are; a
// formula or an image is not something you can type over in place -- its rendering is not
// its source -- so those get a small editor for that one cell, with the result shown as you
// type. The kind is read from what was rendered, which is the honest signal.
function cellKind(cell) {
  if (cell.querySelector('.katex')) return 'math';
  if (cell.querySelector('img')) return 'image';
  if (cell.querySelector('code')) return 'code';
  if (cell.querySelector('a')) return 'link';
  return 'text';
}

function wireCell(msg, i, table, cell) {
  const kind = cellKind(cell);
  cell.dataset.kind = kind;
  if (kind === 'text') {
    cell.setAttribute('contenteditable', 'plaintext-only');
    cell.spellcheck = false;
    cell.title = 'Click to edit';
    cell.addEventListener('blur', () => {
      cell.dataset.src = cell.textContent.replace(/\s+/g, ' ').trim();
      commitTable(msg, i, table);
    });
    cell.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') cell.blur();
      if (e.key === 'Enter') { e.preventDefault(); cell.blur(); }   // a row is one line
    });
    return;
  }
  cell.title = 'Click to edit this ' + kind;
  cell.classList.add('cell-' + kind);
  cell.addEventListener('click', (e) => {
    // A click from inside the editor is the editor's, not an invitation to open another.
    // Without this, Save re-opened it: the handler tears the editor down and re-wires the
    // cell, and the very same click then reaches the fresh listener with nothing to stop it.
    if (e.target.closest('.cellpop')) return;
    if (cell.querySelector('.celledit')) return;
    e.preventDefault();
    openCellEditor(msg, i, table, cell, kind);
  });
}

// One cell, its source, and a preview of what that source becomes.
function openCellEditor(msg, i, table, cell, kind) {
  const was = cell.innerHTML;
  const box = document.createElement('div'); box.className = 'cellpop';
  const inp = document.createElement('input');
  inp.type = 'text'; inp.className = 'celledit'; inp.value = cell.dataset.src || '';
  inp.spellcheck = false;
  const hint = document.createElement('div'); hint.className = 'cellhint';
  hint.textContent = kind === 'math' ? 'LaTeX between $ … $'
                   : kind === 'image' ? '![alt](url)'
                   : kind === 'code' ? 'code between ` … `' : 'Markdown';
  const prev = document.createElement('div'); prev.className = 'cellprev';
  const draw = () => { prev.innerHTML = renderInline(inp.value); };
  draw();
  inp.addEventListener('input', draw);
  const bar = document.createElement('div'); bar.className = 'editbar';
  const ok = document.createElement('button'); ok.type = 'button';
  ok.className = 'primary'; ok.textContent = 'Save';
  const no = document.createElement('button'); no.type = 'button'; no.textContent = 'Cancel';
  bar.append(ok, no);
  box.append(hint, inp, prev, bar);
  cell.textContent = ''; cell.appendChild(box);
  inp.focus(); inp.select();
  const close = (html) => {
    cell.innerHTML = html;
    wireCell(msg, i, table, cell);          // its kind may have changed
  };
  no.onclick = (e) => { e.stopPropagation(); close(was); };
  ok.onclick = (e) => {
    e.stopPropagation();
    cell.dataset.src = inp.value.trim();
    close(renderInline(cell.dataset.src));
    commitTable(msg, i, table);
  };
  box.addEventListener('click', (e) => e.stopPropagation());
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close(was);
    if (e.key === 'Enter') ok.click();
  });
}

// Markdown for the inside of a cell: no paragraph wrapper, same sanitising as everything
// else. Falls back to the text when the renderer is not here.
function renderInline(src) {
  const m = markdown();
  if (!m) return String(src).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  try {
    return DOMPurify.sanitize(m.parseInline(String(src)),
                              { ADD_TAGS: MD_TAGS, ADD_ATTR: MD_ATTR });
  } catch (e) { return String(src); }
}

function commitTable(msg, i, table) {
  const md = tableToMd(table, mdBlocks(msg.content)[i] || '');
  if (md && setBlock(msg, i, md)) saveConvs();
}

// The cells of a Markdown table, as source, row by row. The delimiter row is skipped: it is
// alignment, not content.
function tableCells(src) {
  const lines = String(src).split('\n').map(l => l.trim()).filter(l => l.indexOf('|') >= 0);
  return lines.filter(l => !/^\|?\s*:?-{1,}/.test(l)).map(splitRow);
}

// Split one row on its pipes, honouring \| inside a cell.
function splitRow(line) {
  const s = line.trim().replace(/^\|/, '').replace(/\|$/, '');
  const out = []; let cur = '';
  for (let k = 0; k < s.length; k++) {
    if (s[k] === '\\' && s[k + 1] === '|') { cur += '|'; k++; continue; }
    if (s[k] === '|') { out.push(cur.trim()); cur = ''; continue; }
    cur += s[k];
  }
  out.push(cur.trim());
  return out;
}

// Rebuild the Markdown table from the cells' SOURCE. The alignment row is carried over from
// the original when the column count still matches -- alignment is not visible in a cell,
// so regenerating it would quietly discard it.
function tableToMd(table, src) {
  const rowOf = (tr) => [...tr.children].map(c =>
    String(c.dataset.src == null ? c.textContent : c.dataset.src)
      .replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim());
  const head = table.querySelector('thead tr');
  const body = [...table.querySelectorAll('tbody tr')];
  if (!head) return null;
  const cols = head.children.length;
  const srcDelim = (String(src).split('\n').find(l => /^\s*\|?\s*:?-{1,}/.test(l)) || '').trim();
  const delimOk = srcDelim && srcDelim.split('|').filter(x => x.trim()).length === cols;
  const delim = delimOk ? srcDelim : '| ' + Array(cols).fill('---').join(' | ') + ' |';
  const line = (cells) => '| ' + cells.join(' | ') + ' |';
  return [line(rowOf(head)), delim].concat(body.map(tr => line(rowOf(tr)))).join('\n');
}

// Edit one block in place. Saving rebuilds only this block, so the rest of the message --
// including any code output already on screen -- is left where it is.
function editBlock(msg, i, node) {
  if (node.querySelector('.blockedit')) return;
  const src = mdBlocks(msg.content)[i] || '';
  const ta = document.createElement('textarea');
  ta.className = 'blockedit'; ta.value = src;
  ta.rows = Math.min(18, Math.max(2, src.split('\n').length + 1));
  const bar = document.createElement('div'); bar.className = 'editbar';
  const ok = document.createElement('button'); ok.type = 'button';
  ok.className = 'primary'; ok.textContent = 'Save';
  const no = document.createElement('button'); no.type = 'button'; no.textContent = 'Cancel';
  const del = document.createElement('button'); del.type = 'button';
  del.className = 'danger'; del.textContent = 'Delete block';
  bar.append(ok, no, del);
  node.textContent = ''; node.append(ta, bar);
  ta.focus();
  const redraw = () => {
    const fresh = blockNode(msg, i, mdBlocks(msg.content)[i] || '');
    node.replaceWith(fresh);
  };
  no.onclick = redraw;
  ok.onclick = () => { if (setBlock(msg, i, ta.value)) saveConvs(); redraw(); };
  del.onclick = () => {
    const bl = mdBlocks(msg.content);
    bl.splice(i, 1);
    msg.content = bl.join('\n\n');
    saveConvs(); render();
  };
  ta.addEventListener('keydown', e => {
    if (e.key === 'Escape') redraw();
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ok.click();
  });
}

function fillBody(b, msg, live) {
  // `msg.think` is present once anything arrived on the thinking channel; `undefined` means
  // a stored message that predates channels, which still needs its tags parsed.
  const has = msg && msg.think !== undefined;
  const { think, rest, open } = has
    ? { think: msg.think === null ? null : msg.think,
        rest: msg.content || '',
        open: !(msg.content || '') }
    : splitThink((msg && msg.content) || '');
  if (live && live.det) {                                            // in-place update
    if (think !== null && !live.det.parentNode) {
      b.prepend(live.det);                                           // thinking appeared mid-stream
      live.det.open = true;
    }
    if (live.det.parentNode) {
      setRenderedLive(live.pre, think === null ? '' : think);
      live.det.classList.toggle('live', rest === '');
      live.sum.textContent = rest === '' ? 'Thinking…' : 'Thought before answering';
      if (rest !== '' && !live.collapsed && !live.touched) {
        live.det.open = false; live.collapsed = true;                // auto-fold once the answer starts
      }
    }
    if (msg.toollog && msg.toollog.length) setToollogLive(live, msg.toollog);
    setRenderedLive(live.ans, rest);      // no msg: a reply still arriving is not editable
    // A page built mid-reply appears in the reply as it is written, after the answer so far.
    // Rebuilt only when what it should SHOW changes -- how many pages, and whether the
    // reader has opened them. Counting `.webtab` cannot tell the icon from an empty panel,
    // so the icon was being rebuilt on every token.
    const sig = webPagesOf(msg.toollog).length + '/' + webPagesOf(msg.toollog, true).length
              + (openWebPanels.has(msg.toollog) ? ':open' : ':icon');
    if (sig !== live.webSig) {
      if (live.web) live.web.remove();
      live.web = webPanelFor(msg);
      live.webSig = sig;
      if (live.web) b.appendChild(live.web);
    }
    return;
  }
  if (think !== null) {
    const det = document.createElement('details'); det.className = 'think';
    const sum = document.createElement('summary');
    sum.textContent = open ? 'Thinking…' : 'Thought before answering';
    const pre = document.createElement('div'); pre.className = 'ans';
    setRendered(pre, think, null);          // reasoning is prose too: lists, code, formulae
    det.append(sum, pre); b.appendChild(det);
    det.open = open;                                                 // mid-think: watch it; done: folded
  }
  if (msg && msg.toollog && msg.toollog.length) b.appendChild(toollogNode(msg.toollog));
  const ans = document.createElement('div');
  ans.className = 'ans';
  setRendered(ans, rest, live ? null : msg);
  b.appendChild(ans);
  const web = webPanelFor(msg);
  if (web) b.appendChild(web);
}
// The waiting dots: prefill can take seconds before the first token exists, and an empty
// bubble reads as a frozen page. They live in the reply body from submit until the first
// chunk lands (or the reply ends in error), pulsing · → ·· → ···.
function startDots(live, body) {
  const dots = document.createElement('span');
  dots.className = 'dots'; dots.textContent = '·';
  live.dots = dots;
  live.dotsTimer = setInterval(() => {
    dots.textContent = dots.textContent.length >= 3 ? '·' : dots.textContent + '·';
  }, 450);
  body.appendChild(dots);
}
function stopDots(live) {
  if (!live || !live.dots) return;
  clearInterval(live.dotsTimer); live.dots.remove(); live.dots = null;
}
function note(t) { $('#hintbar').textContent = t || ''; }
function promptFor(m) {
  let p = '';
  (m.attachments || []).forEach(a => {
    if (a.text) p += `[${a.kind}: ${a.name}${a.clipped ? ', truncated' : ''}]\n${a.text}\n\n`;
    else if (a.dataUrl) p += `[image attached: ${a.name}]\n`;
  });
  return p + m.content;
}
$('#composer').onsubmit = async (e) => {
  e.preventDefault();
  const text = $('#input').value.trim();
  if (!text && !attachments.length) return;
  if (!modelLoaded) return note('Load a model first (left panel).');
  if (streaming) return note('Wait for the current reply to finish.');
  const conv = ensureConv();
  const msg = { role: 'user', content: text, attachments };
  conv.messages.push(msg); attachments = []; renderAttachments();
  if (conv.messages.length === 1) conv.title = titleFrom(text);
  $('#input').value = ''; render(); renderConvs();

  // The whole conversation goes to the model, not just the last line — that is what makes
  // it a chat. Assistant turns go back WITHOUT their thinking: the reasoning was this
  // model's scratch work for its last answer, not context for the next one.
  // A system turn the page adds, not the SDK: this is about how the ANSWER IS DISPLAYED
  // here — the chat renders Markdown, highlights code and typesets LaTeX, and a model that
  // replies in prose gets none of that. It belongs to the client that does the rendering,
  // so the SDK stays neutral about presentation.
  const msgs = [{ role: 'system', content: UI_SYSTEM }];
  conv.messages.slice(0, -1).forEach(m => msgs.push({
    role: m.role,
    content: m.role === 'user' ? promptFor(m)
           : (m.think !== undefined ? (m.content || '') : splitThink(m.content).rest),
  }));
  msgs.push({ role: 'user', content: promptFor(msg) });

  for (let i = msgs.length - 1; i > 0; i--) if (!msgs[i].content) msgs.splice(i, 1);

  const reply = { role: 'assistant', content: '' };
  conv.messages.push(reply);
  // Add the reply bubble directly (not a full render) and wire its live pieces.
  const el = $('#messages');
  const node = messageNode(reply, true);
  el.appendChild(node); el.scrollTop = el.scrollHeight;
  const body = node.querySelector('.body');
  const det = document.createElement('details'); det.className = 'think';
  const sum = document.createElement('summary'); sum.textContent = 'Thinking…';
  const pre = document.createElement('div'); pre.className = 'ans';
  det.append(sum, pre);
  det.ontoggle = () => { if (streaming) streaming.live.touched = true; };
  const ansEl = document.createElement('div'); ansEl.className = 'ans';
  const live = { det, sum, pre, ans: ansEl,
                 touched: false, collapsed: false, tokens: 0, t0: 0, rateT: 0 };
  startDots(live, body);                              // until the first token lands
  body.appendChild(live.ans);
  const rate = document.createElement('div');
  rate.className = 'tokrate'; rate.hidden = true;
  body.appendChild(rate); live.rate = rate;
  resetLiveRender();                    // the first token of a reply renders immediately
  streaming = { convId: conv.id, idx: conv.messages.length - 1, live, body };
  resStartRun();                       // this reply's latency, not the previous one's
  syncButtons();                                      // Send becomes Stop

  try {
    // Images ride alongside the conversation: the worker turns each data URL into pixels
    // and gives them to the model as media. Without this the model only ever saw the text
    // note that a picture existed.
    const imgs = (msg.attachments || []).filter(a => a.kind === 'image' && a.dataUrl)
                                        .map(a => a.dataUrl);
    // A reply may ask to run something, and then be asked again with what came back. The
    // loop is here rather than in the SDK because the tools live here: the sandbox is the
    // page's, and so is the decision to let a model reach it.
    //
    // Every round reads only the tokens THAT round produced. Scanning everything streamed
    // so far finds round 1's call again in round 2's reply -- the same call re-run every
    // round, and its trace appended to what the reader sees once per round.
    let r = null;
    let streamedLen = reply.content.length;
    for (let round = 0; ; round++) {
      const opts = Object.assign({ messages: msgs, images: imgs, prompt: promptFor(msg) },
                                 genOpts());
      const withTools = toolsEnabled();
      if (withTools) {
        opts.tools = toolDefs();
        // This page's decision, not the SDK's: hold the model to names that exist. A wrong
        // name here cannot be run and the reader sees a failed round; a model that calls
        // nothing instead just answers, which is the better of the two failures for a chat.
        opts.require_known_tools = true;
      }
      try {
        r = await call('generate', opts);
      } catch (err) {
        // The probe says this model takes tools; if sending them fails anyway, that is the
        // template disagreeing with the probe. Drop them and answer -- a turn lost to a
        // tool definition is worse than a turn without tools -- and stop offering them to
        // this model rather than failing the same way on every message after it.
        if (!withTools) throw err;
        modelTakesTools = false;
        delete opts.tools;
        console.warn('webtorch: this model rejected tool definitions, continuing without:',
                     err && err.message);
        r = await call('generate', opts);
      }
      const raw = reply.content.slice(streamedLen);      // this round's reply, nothing else
      streamedLen = reply.content.length;
      const prefix = reply.content.slice(0, streamedLen - raw.length);
      // ONE scan, and it happens in the SDK: what the reader is shown and what the loop
      // runs have to come from the same reading of the text, or a call the loop missed gets
      // printed as prose. Reading a model's own call format is the SDK's job, not this
      // page's -- see `toolcall.py` and the tokenizer's `tool_call_format`.
      const scan = await call('toolScan', { text: raw, tools: toolDefs() });
      const shown = scan.shown;                          // prose kept, protocol removed
      if (!toolsEnabled() || round >= MAX_TOOL_ROUNDS) {
        // No round comes back around to tidy this one, so it tidies itself.
        reply.content = prefix + shown;
        break;
      }
      const calls = scan.calls;
      if (!calls.length) { reply.content = prefix + shown; break; }
      // An id the template can answer a result with. The model's own call may already carry
      // one -- keep it; making another would tie the result to an id the model never said.
      calls.forEach((c, i) => { if (!c.id) c.id = 'wt_' + round + '_' + i; });
      // Each call is recorded the moment it returns, not at the end of the round. Two
      // reasons, both found by watching it: a later call in the SAME round that asks what
      // pages are open has to see the ones already built (`list_web_pages` answered "none"
      // immediately after two `render_web_page` calls beside it), and the reader should
      // see a page as soon as it exists rather than when the whole round finishes.
      const results = [];
      for (const c of calls) {
        const res = await runToolCall(c);
        results.push(res);
        reply.toollog = (reply.toollog || [])
          .concat([{ name: c.name, args: c.args, result: res,
                     logs: c.logs || null, images: c.images || null,
                     view: c.view || null, viewHeight: c.viewHeight || 0,
                     tab: c.tab || null }]);
        setToollogLive(live, reply.toollog);
      }
      // The conversation the model sees next: what it said, then what each call returned.
      // Whether the calls travel as structured `tool_calls` or as the model's own text is
      // a property of its template, so the SDK builds these turns -- this page only says
      // what happened.
      (await call('toolRound', { text: shown, calls, results })).forEach(m => msgs.push(m));
      // The call and its result go to a folded side panel, NOT into the answer text.
      // Putting the trace in the answer taught the model to imitate it as content: with a
      // real trace in its history it next "called" a tool by writing a forged one --
      // invalid JSON and all -- and answered from the forgery. The answer stays the
      // answer; the record of what ran stays beside it, and out of the model's context.
      reply.content = prefix + shown;
      streamedLen = reply.content.length;
      setRendered(body.querySelector('.ans') || body, reply.content);
      setToollogLive(live, reply.toollog);
      if (_stopRequested()) break;
    }
    if (live.tdet) live.tdet.open = false;          // the run is over; the panel folds
    if (r && r.truncated) {
      reply.content += '\n\n[stopped at ' + r.n + ' tokens — raise “Max reply length”'
                     + ' in ⚙ Settings, or leave it empty for no limit]';
    }
    if (!reply.content.trim()) reply.content = '(empty reply)';
    reply.stats = r || null;                    // final n / tok_s for the footer line
    checkSlow(r);
  } catch (err) {
    reply.content = (reply.content ? reply.content + '\n\n' : '') + 'Error: ' + err.message;
  } finally { stopDots(live); resetLiveRender(); }
  resEndRun();          // the gap to the next reply is not a decode step
  streaming = null;
  syncButtons();                                      // Stop becomes Send again
  conv.updated = Date.now(); saveConvs(); render();
};
// The form's submit is the send path; while streaming, the same button stops instead. The
// worker sets the SDK's cancel flag and `generate` returns the part of the reply that
// exists, so the stopped answer stays in the conversation.
$('#send').addEventListener('click', e => {
  if (!streaming) return;                       // idle: let the form submit as usual
  e.preventDefault();
  askStop();                                    // takes effect now, not when the worker is free
  call('stopGen').catch(() => {});
  note('Stopping…');
});
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#newChat').onclick = () => { attachments = []; renderAttachments(); newConv();
                                toggleSidebar(false); };
// ---- tools the model can call ---------------------------------------------------------
//
// A registry, not a list of Python functions. What a tool IS here is a definition the model
// is shown and something to run when it asks -- so a second kind of tool (a fetch, a
// calculator, whatever a host wants) is a `registerTool` call rather than an edit to the
// generate loop. The Python runtime is the first client of that, not the shape of it.
//
// The definitions go to the model through the SDK's `tools=`, which hands them to the
// MODEL'S OWN chat template. So which models can be told about tools is a question about
// their template, and nothing here needs to know the answer.
const TOOLS = [];
function registerTool(def, run) { TOOLS.push({ def, run }); }
// Written in the shape THIS model's template was shown to read. The registry holds one
// canonical form; which one goes over the wire is a fact about the model, discovered by
// `toolsSupported`, not a convention picked here.
function toolDefs() {
  return TOOLS.map(t => t.def);         // one canonical form; the SDK reshapes for the model
}
// A plain lookup: the SDK reads what the model wrote and reports the REGISTERED name, so
// noise ("run_ Python" for "python") has already been resolved by the time a call gets
// here. See `toolcall.match_name`.
function findTool(name) {
  return TOOLS.find(t => t.def.function.name === name);
}

const TOOLS_KEY = 'webtorch.toolsOn';
// Whether the LOADED model can be told about tools -- asked once per model, never guessed.
// null = not asked yet, so the first reply after a load does not race the probe.
let modelTakesTools = null;      // null = not asked yet
// One question, in the page's own terms: can this model use tools at all? How its template
// writes a call, which shape of definition it reads, how a result gets back to it -- those
// are the SDK's business and no longer travel up here.
async function probeTools() {
  modelTakesTools = null;
  try {
    const r = await call('toolsSupported');
    modelTakesTools = !!(r && r.ok);
  } catch (e) { modelTakesTools = false; }
  return modelTakesTools;
}

// A tool result as a message THIS model will actually receive.
//
// Not assumed to be an ordinary turn: a template may define its own structure for one, and
// a template that does not know the `tool` role drops it silently -- the model then answers
// without ever having seen what the tool returned. The probe says which is the case; an
// ordinary user turn is the fallback, never the assumption.

// Keep the result an object when it is one, so folding the name in does not bury JSON
// inside a JSON string.
function safeJson(v) {
  if (typeof v !== 'string') return v;
  try { return JSON.parse(v); } catch (e) { return v; }
}
function toolsEnabled() {
  // On unless turned off. The reason it used to default off was that a model which can run
  // code on its own initiative is a different thing from one that only answers -- but the
  // runtime it reaches is already sandboxed away from the conversation, and the failure the
  // default was avoiding is smaller than the one it caused: asked for a product of two large
  // numbers, a model with no tool has to invent an answer, and does. The switch is still
  // there, and an explicit '0' still wins; only "never touched it" changed sides.
  return localStorage.getItem(TOOLS_KEY) !== '0' && pyEnabled() && TOOLS.length > 0
         && modelTakesTools === true;
}

// How many times a reply may call a tool and be asked again. A bound, because a model that
// answers every result with another call would otherwise never finish.
const MAX_TOOL_ROUNDS = 4;


async function runToolCall(c) {
  const t = findTool(c.name);
  if (!t) {
    // The correction is the CALLER'S OWN call with the name fixed -- not a specimen call of
    // mine. A generic example gets copied literally, arguments and all: a small model asked
    // to multiply two numbers called a tool that does not exist, was shown `print(2 + 2)` as
    // the shape of a valid call, ran exactly that, and reported 4 as its answer. The example
    // has to be the thing the model actually wanted, so copying it does what it meant.
    const listed = TOOLS.map(x => {
      const f = x.def.function;
      const req = (f.parameters && f.parameters.required) || [];
      const props = Object.keys((f.parameters && f.parameters.properties) || {});
      return '  ' + f.name + '(' + props.map(k => k + (req.includes(k) ? '' : '?')).join(', ') + ')';
    }).join('\n');
    // Which real tool it most likely meant. A call with the wrong name is usually a real
    // name misspelled, so name similarity is the first evidence -- whitespace and
    // punctuation carry no meaning ("run_ Python" is "runpython"); matching argument names
    // decide where the name is no resemblance at all. It is only used to address the
    // correction -- nothing runs on a guess.
    // The evidence comes from the SDK (`suggest_tool`); the threshold and the fallback
    // order are THIS page's policy, because what to do about a call that named nothing is
    // a product question, not a fact about the model.
    const ranked = await call('toolSuggest',
                              { name: c.name, args: c.args || {}, tools: toolDefs() });
    const pick = (ranked[0] && ranked[0].name_score >= 0.5 && ranked[0].name)
              || (ranked.find(r => r.args_match) || {}).name
              || (TOOLS[0] && TOOLS[0].def.function.name);
    const guess = pick ? TOOLS.find(x => x.def.function.name === pick) : null;
    // The corrected call is shown in the form this model's template actually writes -- with
    // the model's OWN arguments. Handing back JSON to a model whose format is the XML one
    // would correct the name and break the call.
    const fixed = guess
      ? await call('toolRender', { name: guess.def.function.name, args: c.args || {},
                                   tools: toolDefs() })
      : '';
    return 'there is no tool called "' + c.name + '". These exist, with their arguments:\n'
         + listed + '\n\nYour call again, with the name corrected — send this:\n' + fixed;
  }
  try {
    const args = typeof c.args === 'string' ? JSON.parse(c.args) : (c.args || {});
    const r = await t.run(args);
    // A tool may produce something for the READER as well as an answer for the model. That
    // part rides on the call rather than in the result: it goes to the panel beside the
    // reply, and never into the context, where a page of markup would cost more than the
    // whole answer and tell the model nothing it did not just write.
    if (r && typeof r === 'object'
        && (r.__view != null || r.__logs != null || r.__images != null)) {
      if (r.__view != null) { c.view = r.__view; c.viewHeight = r.__viewHeight || 0;
                              c.tab = r.__tab || null; }
      if (r.__logs != null) c.logs = r.__logs;
      if (r.__images != null) c.images = r.__images;
      const forModel = Object.assign({}, r);
      delete forModel.__view; delete forModel.__viewHeight; delete forModel.__tab;
      delete forModel.__logs; delete forModel.__images;
      return JSON.stringify(forModel);
    }
    return typeof r === 'string' ? r : JSON.stringify(r);
  } catch (e) {
    // The failure goes BACK to the model rather than ending the turn: "that raised, here is
    // what it said" is something it can act on, and is usually what it needs.
    return 'the tool failed: ' + String((e && e.message) || e);
  }
}

// What the reader sees of a tool round. The model's own reply already says what it means to
// do; this is the record of what actually ran and what came back, which is the part the
// model could otherwise report inaccurately.
function _stopRequested() { return streaming === null; }

function toolTrace(calls, results) {
  return calls.map((c, i) => {
    const a = c.args || {};
    const body = a.code != null ? String(a.code)
               : Object.keys(a).length ? JSON.stringify(a, null, 2) : '';
    return '**' + c.name + '**\n\n'
         + (body ? '```' + (a.code != null ? 'python' : 'json') + '\n' + body + '\n```\n\n' : '')
         + '```\n' + String(results[i]).slice(0, 4000) + '\n```';
  }).join('\n\n');
}

// The folded panel that records a reply's tool calls: what was asked for, what came back.
// Kept OUT of the message content on purpose -- the answer is the answer, and a trace
// sitting in the answer text gets imitated by the model as content on the next turn.
// Whether a reply's tool panel is open, kept across redraws for the same reason the web
// panels are: opening a page rebuilds the list, and a panel that collapsed itself every
// time anything was clicked inside it would be unusable.
const openToolLogs = new WeakSet();

function toollogNode(log) {
  const det = document.createElement('details'); det.className = 'toollog';
  if (log && openToolLogs.has(log)) det.open = true;
  det.addEventListener('toggle', () => {
    if (!log) return;
    if (det.open) openToolLogs.add(log); else openToolLogs.delete(log);
  });
  const sum = document.createElement('summary');
  sum.textContent = log.length === 1 ? 'Used a tool' : 'Used ' + log.length + ' tool calls';
  // `toolTrace` writes markdown -- the call's name in bold, its arguments as a fenced
  // python or json block, what came back as another fence -- and it was being put on the
  // page as plain text, so the reader saw the backticks instead of the code.
  const body = document.createElement('div'); body.className = 'ans';
  setRendered(body, toolTrace(log, log.map(e => e.result)), null);
  det.append(sum, body);
  // What it printed, then what it drew: the order the call itself produced them in.
  // What it printed and what it drew, in the order a run produces them -- the order the run
  // button under a code block already shows them in.
  log.forEach(e => {
    if (e && e.logs && e.logs.length) det.appendChild(consoleBlock(e.logs));
    if (e && e.images && e.images.length) det.appendChild(imagesBlock(e.images));
  });
  return det;
}

// The pages a reply built, in the reply itself.
//
// Not inside the tool record: that panel is the trace of what ran, folded away by default
// because most of the time nobody needs it, and a page built FOR the reader does not belong
// behind it. What the reply produced sits in the reply.
function webPanelFor(msg) {
  const log = msg && msg.toollog;
  const built = webPagesOf(log, true);
  if (!built.length) return null;      // nothing was built, or the model took it all back
  return webTabsNode(webPagesOf(log), (name) => {
    // The reader's control puts a page AWAY; it does not destroy it. Closing the last tab
    // used to leave the reply with no route back to what it had produced -- and because the
    // flag is written to storage with the conversation, not even a reload brought it back.
    // Only `close_web_page` is a real close: the author retracting its own page.
    log.forEach(e => { if (e && e.view && e.tab === name) e.hidden = true; });
    saveConvs(); render();
  }, log, built.length);
}

// The pages a tool log holds, newest name wins if a tab was reopened. `all` counts the ones
// the reader has put away too -- what the reply produced, rather than what is on show.
function webPagesOf(log, all) {
  const out = [];
  (log || []).forEach(e => {
    if (!e || !e.view || e.closed) return;
    if (!all && e.hidden) return;
    const at = out.findIndex(p => p.name === e.tab);
    const page = { name: e.tab || 'page', html: e.view, height: e.viewHeight || 0 };
    if (at >= 0) out[at] = page; else out.push(page);
  });
  return out;
}
// Live update while the reply is still streaming: the panel appears as soon as the first
// call returns, open so the work is visible; the loop folds it when the reply finishes.
function setToollogLive(live, log) {
  if (!live) return;
  if (!live.tdet) {
    live.tdet = toollogNode(log);
    live.tdet.open = true;
    live.ans.parentNode && live.ans.parentNode.insertBefore(live.tdet, live.ans);
  } else if ((webPagesOf(log).length
              && live.tdet.querySelectorAll('.webtab').length !== webPagesOf(log).length)
             || (log.some(e => e && e.logs && e.logs.length)
                 && live.tdet.querySelectorAll('.jslog').length
                    !== log.filter(e => e && e.logs && e.logs.length).length)) {
    const fresh = toollogNode(log);                 // a new view to show: rebuild the panel
    fresh.open = live.tdet.open;
    live.tdet.replaceWith(fresh);
    live.tdet = fresh;
  } else {
    setRenderedLive(live.tdet.querySelector('.ans'),
                    toolTrace(log, log.map(e => e.result)));
  }
}

// ---- the Python runtime, as tools -----------------------------------------------------
// Three tools, not one with a mode. Folding them into a single `python(action=…)` was
// tried and measured: on the same template and tokenizer it took the definitions from 483
// tokens to 459 -- 24 tokens, 5%. The cost is the three return contracts, not the frame
// repeated around them, and those contracts are what stop a model quoting a traceback as
// an answer. Against that saving, a name like `run_python` says on its own what the tool
// is for, while `python` plus an action asks a small model for one more decision before it
// can use anything at all -- and a 0.6B asked to multiply two large numbers talked itself
// out of calling the merged tool. Not worth 5%.
//
// They are named `python` and `javascript`, not `run_python` and `run_javascript`. That is
// a different question from the merge above -- still three tools, still no `action` to
// decide -- and it is the one that decided whether a call worked at all. A `run_` prefix
// reads as the start of a FILENAME, and a 0.6B completes it as one: it would name
// `run_python` correctly while reading the definitions, then write "调用 run_2.py" a
// sentence later and call that. Measured over 10 attempts at one arithmetic question,
// with `run_` prefixes: 4 correct, 4 wrong names (`run_2`, `run_2_python`, `run_0_0`,
// `run_20230920`, `run_ordinary_python`), 2 no call at all. Without them, over 12: 11
// correct, 0 wrong, 1 no call.
//
// Sampling is not what fixes this and neither is temperature. By the time the call is
// written the model has already committed to the wrong name in its own reasoning, and at
// that point it is CERTAIN of it: measured on a real failing generation, the next token
// after `{"name": "run_` was `2` at p=0.99978 (T=0.6) against `python` at 0.00015, so
// greedy decoding inside the call would lock the mistake in rather than repair it, and
// top_p=0.95 measured no better than none. The name has to not invite the mistake.
registerTool({
  type: 'function',
  function: {
    name: 'python',
    // Terse on purpose. A definition is re-sent with EVERY request, so its prose is paid
    // for on every prompt and every decode step after it: written out at length these three
    // cost 781 tokens, which took a 0.6B from 110 tok/s to 64 and its first token from 0.3 s
    // to 3.4. The contract still has to be exact -- it just has to be a signature, not an
    // essay. What the model must NOT confuse is stated; the rest is left out.
    description: 'Run Python, get the output back. Separate interpreter from the model; '
      + 'state persists across calls. Use it instead of computing in your head.\n'
      + 'Returns {"stdout":str,"stderr":str,"result":str|null,"traceback":str|null,'
      + '"figures":int}. stderr is library noise, not the answer. traceback non-null means it '
      + 'failed and result is null. figures are shown to the reader, not to you.',
    parameters: {
      type: 'object',
      properties: { code: { type: 'string', description: 'The Python source to run.' } },
      required: ['code'],
    },
  },
}, async ({ code }) => {
  if (!code) return { stdout: '', stderr: '', result: null,
                      traceback: 'no code was given', figures: 0 };
  const r = await pyCall('run', { code: String(code) });
  const clean = (x) => (x ? String(x).replace(/\n+$/, '') : '');
  const out = {
    stdout: clean(r.out),
    stderr: clean(r.err),
    // A run that raised HAS no result. Reporting the last value alongside a traceback invites
    // it to be quoted as the answer.
    result: r.error ? null : (r.value != null && r.value !== 'None' ? String(r.value) : null),
    traceback: r.error ? String(r.error) : null,
    figures: (r.images || []).length,
  };
  // The definition says figures are shown to the reader, and until now they were not: this
  // counted them and dropped the pictures. They travel the same side channel the JavaScript
  // tool's do -- to the panel, never into the model's context, where a base64 PNG would cost
  // more than every reply in the conversation.
  if ((r.images || []).length) out.__images = r.images;
  // And the two streams as lines, so a call's output reads the same whichever tool made it.
  const lines = [];
  clean(r.out).split('\n').forEach(t => { if (t) lines.push({ level: 'log', text: t }); });
  clean(r.err).split('\n').forEach(t => { if (t) lines.push({ level: 'warn', text: t }); });
  if (r.error) lines.push({ level: 'error', text: String(r.error).replace(/\n+$/, '') });
  if (lines.length) out.__logs = lines;
  return out;
});

registerTool({
  type: 'function',
  function: {
    name: 'install_python_packages',
    description: 'Make modules importable by name (distribution first, else a pure-Python '
      + 'wheel from PyPI). No names asks what is loaded.\n'
      + 'Returns {"ready":[str],"installed_from_pypi":[str],'
      + '"unavailable":[{"name":str,"why":str}]}. Do not re-request an unavailable name.',
    parameters: {
      type: 'object',
      properties: {
        names: { type: 'array', items: { type: 'string' },
                 description: 'Module names, e.g. ["sympy", "networkx"].' },
      },
    },
  },
}, async ({ names }) => {
  const want = Array.isArray(names) ? names.map(String).filter(Boolean) : [];
  const r = await pyCall('packages', { packages: want.length ? pyPackages().concat(want)
                                                             : pyPackages() });
  return {
    ready: r.loaded || [],
    installed_from_pypi: r.fromPyPI || [],
    unavailable: (r.unavailable || []).map(u => (typeof u === 'string'
      ? { name: u, why: '' } : { name: u.name, why: u.why || '' })),
  };
});

registerTool({
  type: 'function',
  function: {
    name: 'restart_python',
    description: 'Discard the interpreter and start a fresh one; everything defined so far '
      + 'is lost. For a wedged runtime, not for an ordinary traceback.\n'
      + 'Returns {"restarted":true,"state":str,"ready":[str]}. state "ready" means usable.',
      parameters: { type: 'object', properties: {} },
  },
}, async () => {
  try { await pyCall('reset'); } catch (e) { /* the interpreter is going away regardless */ }
  pyStop(); pyStart();
  for (let i = 0; i < 240 && pyState !== 'ready'; i++)
    await new Promise(r => setTimeout(r, 250));
  return { restarted: true, state: pyState,
           ready: pyState === 'ready' ? pyPackages() : [] };
});


// JavaScript, run where it cannot reach anything of ours.
//
// `eval` on this page would hand a model's script the conversation, the stored models and
// the DOM. A sandboxed iframe with no `allow-same-origin` gets an opaque origin instead,
// and that is not a claim -- measured from inside one: listing IndexedDB, opening
// `webtorch-chunks` (where the model weights are) and reading Cache Storage all raise
// SecurityError, while the page itself sees both databases.
//
// It is a document rather than a Worker because a Worker is same-origin: no DOM, but the
// stored models are still one `indexedDB.open` away.
const JS_TIMEOUT_MS = 5000;

// The DOM names a computation has no business touching. Shadowing them is the CONTRACT --
// "this tool has no page" -- not the security boundary; the sandbox is that, and it holds
// whatever the code does with these. What shadowing buys is that a script written for the
// computing tool cannot half-work by drawing something nobody will see.
const JS_NO_DOM = ['document', 'window', 'top', 'parent', 'frames', 'location', 'history'];

function runSandboxedJs(code, timeoutMs, wantDom) {
  return new Promise((resolve) => {
    const f = document.createElement('iframe');
    f.setAttribute('sandbox', 'allow-scripts');
    // Off the screen, but VISIBLE and laid out. `display:none` skips layout entirely, and
    // `visibility:hidden` skips it inside an iframe too -- under either, every element in
    // the page reports a zero-height box and the frame it is shown in cannot be sized to
    // its contents. Moving it out of view is the only one of the three that still measures.
    f.style.cssText = 'position:absolute;left:-10000px;top:0;width:760px;height:600px;'
                    + 'border:0;opacity:0;pointer-events:none';
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      try { f.remove(); } catch (e) {}
      window.removeEventListener('message', onMsg);
      resolve(v);
    };
    const onMsg = (e) => { if (e.source === f.contentWindow) done(e.data); };
    window.addEventListener('message', onMsg);
    // The script goes in the HEAD. Put after <body> it is parsed INTO the body, so the
    // document always held something and "did this draw anything?" was always yes.
    f.srcdoc = '<!doctype html><html><head><meta charset="utf-8">'
      + '<scr' + 'ipt data-wt-harness>'
      + '(async () => {'
      // The script sits in the HEAD -- put after <body> it is parsed INTO the body, and the
      // captured page then contains its own source. So wait for the body to exist before
      // running anything that expects one.
      + '  if (document.readyState === "loading") {'
      + '    await new Promise(r => document.addEventListener("DOMContentLoaded", r));'
      + '  }'
      + '  const out = [];'
      + '  const w = (level) => (...a) => out.push({ level: level, text: a.map(x => {'
      + '    try { return typeof x === "string" ? x : JSON.stringify(x); }'
      + '    catch (e) { return String(x); } }).join(" ") });'
      + '  console.log = w("log"); console.info = w("info");'
      + '  console.warn = w("warn"); console.error = w("error");'
      + '  console.debug = w("log");'
      + '  let value = null, error = null;'
      + '  const src = ' + JSON.stringify(code) + ';'
      + '  const hide = ' + JSON.stringify(wantDom ? [] : JS_NO_DOM) + ';'
      // Compiled before it is run, and in four shapes, the first that COMPILES winning.
      // `eval` cannot hold a top-level await -- awaiting its RESULT is not the same thing,
      // and a script that used one came back null -- so the body goes inside an async
      // function, which can. But an async BLOCK has no completion value, so a script whose
      // last line is the answer would come back null the other way. Hence the middle
      // shapes: the last statement returned and the rest run before it, split at the last
      // newline or, for a one-liner, the last semicolon. `new Function` raising at
      // construction is what makes trying them safe -- a split landing inside a string
      // simply fails to compile and the next shape is used.
      + '  const tail = (i) => i < 0 ? null : "return (async () => {" + src.slice(0, i + 1)'
      + '                + "\\nreturn (" + src.slice(i + 1) + "\\n); })()";'
      + '  const shapes = ['
      + '    "return (async () => (" + src + "\\n))()",'
      + '    tail(src.lastIndexOf("\\n") - 1),'
      + '    tail(src.lastIndexOf(";")),'
      + '    "return (async () => {" + src + "\\n})()"'
      + '  ];'
      + '  let fn = null;'
      + '  for (const shape of shapes) {'
      + '    if (!shape) continue;'
      + '    try { fn = new Function(...hide, shape); break; } catch (e) { fn = null; }'
      + '  }'
      + '  if (!fn) { error = "could not compile that as JavaScript"; }'
      + '  else { try { value = await fn(...hide.map(() => undefined)); }'
      + '         catch (e) { error = String((e && (e.stack || e.message)) || e); } }'
      + '  let shown = null;'
      + '  if (value !== undefined && value !== null) {'
      + '    try { shown = typeof value === "string" ? value : JSON.stringify(value); }'
      + '    catch (e) { shown = String(value); }'
      + '    if (shown === undefined) shown = String(value);'
      + '  }'
      + '  let view = null;'
      // three modes, not two. `always` is the render tool, which was asked for a page and
      // gets one even if it is blank; `auto` is the run button, where one control serves
      // both a calculation and a drawing and only the code knows which it was; `never` is
      // the computing tool, which has no document to serialise.
      + '  const viewMode = ' + JSON.stringify(wantDom === 'auto' ? 'auto'
                                             : wantDom ? 'always' : 'never') + ';'
      + '  const drew = !!(document.body && document.body.innerHTML.trim());'
      + '  if (viewMode === "always" || (viewMode === "auto" && drew)) {'
      // Serialised WITHOUT this harness. `documentElement.outerHTML` includes the script in
      // the head that is running right now, so a stored page would re-run it on every
      // render -- executing the model's code again and posting to whatever framed it.
      + '    const copy = document.documentElement.cloneNode(true);'
      + '    copy.querySelectorAll("script[data-wt-harness]").forEach(n => n.remove());'
      + '    view = copy.outerHTML;'
      // Height is NOT measured here. This frame is parked off-screen to run in, and what
      // it reports depends on how it was hidden rather than on the page: display:none and
      // visibility:hidden skip layout outright, and even moved off-screen every element
      // came back with a zero box. The frame that shows the page measures itself instead --
      // see `jsViewFrame`, which is the one actually laid out.
      + '  }'
      + '  parent.postMessage({ logs: out, result: shown, error: error,'
      + '                       view: view }, "*");'
      + '})();'
      + '</scr' + 'ipt></head><body></body></html>';
    document.body.appendChild(f);
    // A script that never finishes -- an endless loop, an await that never settles -- blocks
    // only its own frame. Dropping the frame is what ends it.
    setTimeout(() => done({ logs: [], result: null, view: null,
      error: 'timed out after ' + timeoutMs + 'ms and was stopped' }), timeoutMs);
  });
}

// What a call printed, as its own block.
//
// Not an iframe, which is what a rendered page gets. That one has to be a frame: it is a
// foreign document and the sandbox is the only thing making it safe to show. These are
// lines of text we captured -- putting them in a frame would cost a second document, make
// them unselectable, and leave the height to be guessed, for nothing. They stay in our own
// DOM, as text.
//
// The level is kept because it is the one thing the model cannot use and a reader can: a
// warning and a result look identical once they are both "output".
function consoleBlock(logs) {
  const box = document.createElement('div');
  box.className = 'jslog';
  logs.forEach(l => {
    const line = document.createElement('div');
    line.className = 'jsline ' + (l.level === 'error' ? 'err'
                                : l.level === 'warn' ? 'warn' : 'out');
    line.textContent = l.text;
    box.appendChild(line);
  });
  return box;
}

// Figures a call produced. The same treatment they get from the run button under a code
// block -- inline, and double-click to open -- because they are the same thing arriving by
// a different route.
function imagesBlock(images) {
  const box = document.createElement('div');
  box.className = 'jsfigs';
  images.forEach((b64, k) => {
    const im = new Image();
    im.src = 'data:image/png;base64,' + b64;
    wireImage(im, 'figure-' + (k + 1) + '.png');
    box.appendChild(im);
  });
  return box;
}

// The pages a reply built, in one panel with a tab per page -- a browser's shape, because
// that is what a model driving several pages is doing.
//
// Each page is rebuilt from its own source into a fresh sandbox rather than kept as a live
// frame: the source is what the conversation stores, so it survives a redraw and a reload,
// and it is shown under exactly the restrictions it ran under. Only the SELECTED tab is
// built, so ten pages cost one frame.
//
// The height a page reported for itself sizes its frame, bounded: a page is embedded in a
// reply, and a runaway one must not take the reply over.
const JS_VIEW_MIN = 80, JS_VIEW_MAX = 640;

function jsViewFrame(page) {
  const f = document.createElement('iframe');
  f.className = 'jsview';
  f.setAttribute('sandbox', 'allow-scripts');
  f.style.height = JS_VIEW_MIN + 'px';
  // The page measures ITSELF, in the frame that is actually on screen. Measuring where it
  // ran does not work: that frame is parked off-screen, and under every way of hiding it
  // the contents report a zero-height box. A stored page is a page and nothing else, so the
  // reporter is added HERE, at display time, and never becomes part of what is stored.
  const reporter = '<scr' + 'ipt data-wt-fit>(() => {'
    + 'const post = () => {'
    + '  let bottom = 0;'
    + '  for (const el of document.body ? document.body.children : []) {'
    + '    bottom = Math.max(bottom, el.getBoundingClientRect().bottom); }'
    + '  const pad = document.body'
    + '    ? parseFloat(getComputedStyle(document.body).paddingBottom) || 0 : 0;'
    + '  parent.postMessage({ __wtFit: Math.ceil(bottom + pad) }, "*"); };'
    + 'addEventListener("load", post); setTimeout(post, 0); setTimeout(post, 250);'
    + 'if (window.ResizeObserver && document.body) {'
    + '  new ResizeObserver(post).observe(document.body); }'
    + '})();</scr' + 'ipt>';
  f.srcdoc = String(page.html || '').replace(/<\/body>/i, reporter + '</body>')
             + (/<\/body>/i.test(page.html || '') ? '' : reporter);
  const onFit = (e) => {
    if (e.source !== f.contentWindow || !e.data || typeof e.data.__wtFit !== 'number') return;
    const h = Math.min(Math.max(e.data.__wtFit, JS_VIEW_MIN), JS_VIEW_MAX);
    f.style.height = h + 'px';
  };
  window.addEventListener('message', onFit);
  // The listener outlives nothing: a frame removed from the page stops posting, and the
  // panel is rebuilt wholesale on every redraw.
  f.addEventListener('DOMNodeRemovedFromDocument',
                     () => window.removeEventListener('message', onFit));
  return f;
}

// Which reply's panel is open. Nothing is open by default: a page is a page -- it runs
// scripts, loads fonts, animates -- and a conversation that quietly starts eight of them
// every time it is drawn is a conversation that gets slower the longer it goes. A message
// that has pages says so with a button, and the pages exist when someone asks for them.
//
// Keyed on the log array, which is the message's own object, so the choice survives a
// redraw and dies with the conversation rather than being written to storage.
const openWebPanels = new WeakSet();

// `pages` are {name, html, height} and `onClose(name)` removes one for good. Hiding is a
// different thing and deliberately so: it keeps the pages, because "I do not want to look
// at this right now" is not "throw it away".
//
// Two states, one control. There is the icon and there is the browser, and `hide` is the
// way back -- the earlier design had a global "hide pages" AND a per-panel collapse, which
// are the same intent expressed twice and could disagree about what was showing.
function webTabsNode(pages, onClose, log, built) {
  // An empty tab bar is not a state: putting the last tab away IS going back to the icon.
  if (log && !pages.length) openWebPanels.delete(log);
  if (log && !openWebPanels.has(log)) {
    const n = pages.length || built || 0;
    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'webopen';
    open.innerHTML = '<span class="webopenicon">◱</span> '
      + (n === 1 ? '1 page' : n + ' pages');
    open.title = pages.length ? 'Show the pages this reply built'
                              : 'Bring back the pages this reply built';
    open.onclick = () => {
      if (!pages.length) {                      // everything was put away -- bring it back
        (log || []).forEach(e => { if (e) e.hidden = false; });
        saveConvs();
      }
      openWebPanels.add(log); render();
    };
    return open;
  }
  const box = document.createElement('div');
  box.className = 'webtabs';
  const bar = document.createElement('div');
  bar.className = 'webtabbar';
  const stage = document.createElement('div');
  stage.className = 'webstage';
  let active = 0;

  const draw = () => {
    bar.textContent = ''; stage.textContent = '';
    pages.forEach((p, i) => {
      const t = document.createElement('span');
      t.className = 'webtab' + (i === active ? ' on' : '');
      const label = document.createElement('span');
      label.textContent = p.name;
      label.onclick = () => { active = i; draw(); };
      t.appendChild(label);
      if (onClose) {
        const x = document.createElement('button');
        x.type = 'button'; x.className = 'webtabx'; x.textContent = '×';
        x.title = 'Put this page away';
        x.onclick = (e) => { e.stopPropagation(); onClose(p.name); };
        t.appendChild(x);
      }
      bar.appendChild(t);
    });
    // One control, because there are only two states: the icon, and the pages. Hiding is
    // not closing -- the pages are still there, and the icon says how many.
    const eye = document.createElement('button');
    eye.type = 'button'; eye.className = 'webtabhide';
    eye.textContent = 'hide';
    eye.title = 'Put these pages back behind their icon';
    eye.onclick = () => { if (log) openWebPanels.delete(log); render(); };
    bar.appendChild(eye);
    if (pages[active]) stage.appendChild(jsViewFrame(pages[active]));
  };
  draw();
  box.append(bar, stage);
  return box;
}

registerTool({
  type: 'function',
  function: {
    name: 'javascript',
    description: 'Run JavaScript and get the output back. A sandbox with no page in it: '
      + 'document, window and location are not defined, and nothing is displayed. Each call '
      + 'starts empty, so nothing persists between calls. Top-level await works.\n'
      + 'Returns {"stdout":str,"stderr":str,"result":str|null,"error":str|null}. stdout is '
      + 'console.log/info, stderr is console.warn/error, result is the last expression. '
      + 'error non-null means it threw or timed out and result is null.\n'
      + 'To build something for the reader to look at, use render_web_page instead.',
    parameters: {
      type: 'object',
      properties: { code: { type: 'string', description: 'The JavaScript source to run.' } },
      required: ['code'],
    },
  },
}, async ({ code }) => {
  if (!code) return { stdout: '', stderr: '', result: null, error: 'no code was given' };
  const r = await runSandboxedJs(String(code), JS_TIMEOUT_MS, false);
  return jsOut(r, false);
});

// Each reply gets its OWN browser. Pages a reply builds are tabs of one panel; the reply
// before it has a different panel with different tabs, because they are different pieces of
// work and nothing said the second should inherit the first's windows.
//
// A name is a tab within that reply: opening one that exists replaces its contents, the way
// reloading a tab does; a new name opens another tab beside it.
let webTabSeq = 0;
function replyBeingWritten() {
  if (!streaming) return null;
  const conv = convs.find(c => c.id === streaming.convId);
  return conv ? conv.messages[streaming.idx] : null;
}
function openWebTabs() {
  const m = replyBeingWritten();
  return webPagesOf(m ? m.toollog : []).map(p => p.name);
}

registerTool({
  type: 'function',
  function: {
    name: 'render_web_page',
    description: 'Run JavaScript that builds a web page, and show that page to the reader in '
      + 'a tab. A sandbox with its own empty document: build in it with the DOM, or set '
      + 'document.body.innerHTML. Styles, images, SVG and canvas all work. Top-level await '
      + 'works. The reader sees the page; you do not.\n'
      + 'tab names the tab. Reusing a name replaces that tab, a new name opens another '
      + 'beside it; omit it and one is named for you. Each call still starts from an empty '
      + 'document -- a tab is where a page is shown, not a session.\n'
      + 'Returns {"stdout":str,"stderr":str,"result":str|null,"error":str|null,'
      + '"rendered":bool,"tab":str}.',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'JavaScript that builds the page.' },
        tab: { type: 'string', description: 'Which tab to show it in.' },
      },
      required: ['code'],
    },
  },
}, async ({ code, tab }) => {
  const m0 = replyBeingWritten();
  if (m0 && !m0.toollog) webTabSeq = 0;          // a fresh reply starts its numbering over
  const name = String(tab || '').trim() || ('page ' + (++webTabSeq));
  if (!code) return { stdout: '', stderr: '', result: null,
                      error: 'no code was given', rendered: false, tab: name };
  const r = await runSandboxedJs(String(code), JS_TIMEOUT_MS, true);
  const out = jsOut(r, true);
  out.tab = name;
  if (out.__view != null) out.__tab = name;
  return out;
});

registerTool({
  type: 'function',
  function: {
    name: 'list_web_pages',
    description: 'The tabs of pages currently open in this conversation.\n'
      + 'Returns {"tabs":[str]}.',
    parameters: { type: 'object', properties: {} },
  },
}, async () => ({ tabs: openWebTabs() }));

registerTool({
  type: 'function',
  function: {
    name: 'close_web_page',
    description: 'Close one open page by its tab name. Gone, not hidden -- the reader has '
      + 'their own control for hiding.\n'
      + 'Returns {"closed":bool,"tabs":[str]}; closed false means no tab had that name.',
    parameters: {
      type: 'object',
      properties: { tab: { type: 'string', description: 'Which tab to close.' } },
      required: ['tab'],
    },
  },
}, async ({ tab }) => {
  const name = String(tab || '').trim();
  let closed = false;
  const m = replyBeingWritten();
  (m && m.toollog || []).forEach(e => {
    if (e && e.view && !e.closed && e.tab === name) { e.closed = true; closed = true; }
  });
  if (closed) { saveConvs(); render(); }
  return { closed, tabs: openWebTabs() };
});

// Both tools report the same way -- and the same way `python` does, so a call's output
// reads alike whichever tool made it.
function jsOut(r, withView) {
  const logs = r.logs || [];
  const pick = (levels) => logs.filter(l => levels.includes(l.level))
                               .map(l => l.text).join('\n');
  const out = { stdout: pick(['log', 'info']), stderr: pick(['warn', 'error']),
                result: r.result == null ? null : r.result, error: r.error || null };
  if (withView) out.rendered = !!r.view;
  // These go to the reader, never into the model's context. The page because it is usually
  // larger than the answer and says nothing the model does not know, having just written
  // it; the log records because the model already has both streams above, and what it does
  // not have -- which line was a warning -- is a question for someone reading.
  if (withView && r.view) { out.__view = r.view; out.__viewHeight = r.height || 0; }
  if (logs.length) out.__logs = logs;
  return out;
}


// ---- the resource strip ---------------------------------------------------------------
//
// What a page can and cannot know about the machine it is on, and why this is shaped the
// way it is.
//
// It CAN know, exactly: how many bytes of GPU buffer this backend holds (every one was
// allocated through wgpy's own createBuffer, so the ledger is not an estimate), how large
// the WASM heap Python lives in is, how large this thread's JS heap is, and how much
// storage the origin is using.
//
// It CANNOT know, at all: process CPU, GPU utilisation, or anything about paging. No Web
// API exposes them. So those are NOT shown as numbers -- a percentage nobody measured is
// worse than no percentage. They are shown as a LEVEL, derived from two things a page can
// measure and which the system-side numbers were used to calibrate against:
//
//   frame interval  -- how long the main thread goes between animation frames. When the
//                      kernel is busy moving memory, this stretches from 17ms to hundreds.
//   decode latency  -- milliseconds per token, against the best this model has managed in
//                      this session. The GPU stalling on compressed pages shows up here and
//                      nowhere else a page can see.
//
// The thresholds come from a measured run rather than from taste. On a 24GB machine running
// a 12.2GB model, sampling the accelerator's own utilisation beside the kernel's
// decompression counters gave: under 20k decompressions per sample the GPU ran at 77-86%;
// at 20-60k it fell to 60%; past 60k it averaged 58% and touched 2-3% while kernel_task
// took 99-125% of a core. 86 -> 60 is a 1.4x slowdown, which is why "tight" starts below
// that and "severe" is set well past it.
const RES_KEY = 'webtorch.resOpen';
// Dispersion within one reply, not speed. A big model on a small GPU is uniformly slow and
// has nothing wrong with it; a model whose pages keep being decompressed is slow in bursts,
// and it is the bursts that this is for. p90 against the MEDIAN of the same window says
// exactly that and needs no historical floor -- an earlier version compared against the
// fastest step ever seen and called a perfectly healthy 0.6B "severe", because the first
// tokens of a reply arrive at 13ms and the rest at 27ms and the minimum is not the floor.
//
// Calibration: that healthy run measures 1.23. The pressure we are looking for was sampled
// on the system side at a 1.4x average throughput loss with individual stalls of seconds,
// so tight sits above healthy jitter and severe well inside the stalling regime.
const RES_STEP_TIGHT = 1.5, RES_STEP_SEVERE = 3.0;    // p90 / median of one reply's steps
const RES_FRAME_TIGHT = 34, RES_FRAME_SEVERE = 100;   // ms; 34 is two dropped frames
const RES_LEVELS = ['normal', 'tight', 'severe'];
const RES_LABEL = { normal: 'normal', tight: 'tight', severe: 'severe' };

const res = { steps: [], stepsAt: 0, lastAt: 0, frames: [], lastFrame: 0, buf: null,
              stats: null, statsAt: 0, pending: 0, storage: null, timer: null, poll: 0 };
// A hidden tab does not run animation frames at all, so the buffer stops filling and the
// last thing in it is from whenever the tab was last looked at. Dropped rather than kept:
// "no reading" is true, and a level from before the tab was hidden is not.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') { res.frames.length = 0; res.lastFrame = 0; }
});

// Decode latency, from the worker's own clock. Only gaps between consecutive tokens of one
// reply count: the gap across a tool call, or from the last reply to this one, is not a
// decode step and would read as a stall that never happened.
function resNoteStep(at) {
  if (res.lastAt && at > res.lastAt) {
    const dt = at - res.lastAt;
    if (dt < 60000) {
      res.steps.push(dt);
      res.stepsAt = Date.now();
      if (res.steps.length > 40) res.steps.shift();
    }
  }
  res.lastAt = at;
}
function resEndRun() { res.lastAt = 0; }
// A new reply measures itself, not the one before it.
function resStartRun() { res.steps.length = 0; res.lastAt = 0; }

// Frame interval. Sampled continuously, because the pressure being looked for arrives
// between generations as often as during one.
function resFrames() {
  const tick = (t) => {
    if (res.lastFrame) {
      res.frames.push(t - res.lastFrame);
      if (res.frames.length > 90) res.frames.shift();
    }
    res.lastFrame = t;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// The 90th percentile, not the mean: a stall is a tail event, and averaging it with the
// frames on either side hides exactly the thing being looked for.
function resPct(a, p) {
  if (!a.length) return 0;
  const b = [...a].sort((x, y) => x - y);
  return b[Math.min(b.length - 1, Math.floor(b.length * p))];
}
const resP90 = (a) => resPct(a, 0.9);
function resLevel(v, tight, severe) {
  if (!v) return null;
  return v >= severe ? 'severe' : v >= tight ? 'tight' : 'normal';
}
const resWorst = (...ls) => RES_LEVELS[Math.max(...ls.map(l => RES_LEVELS.indexOf(l)).filter(i => i >= 0), 0)];

const resBytes = (n) => n == null ? '—'
  : n >= 1073741824 ? (n / 1073741824).toFixed(2) + ' GB'
  : n >= 1048576 ? Math.round(n / 1048576) + ' MB'
  : Math.round(n / 1024) + ' KB';

function resRead() {
  // Straight out of shared memory when that is the newer of the two. The polled reply
  // carries its own time, and shared memory carries the moment it was written, because
  // neither is always the fresher one: nothing can ask the worker for these while it is
  // busy, and nothing writes the shared array while it is idle. Taking the array
  // unconditionally is how a reply's peak stayed on screen after the memory behind it had
  // been handed back.
  if (res.buf && res.buf[0] && res.buf[4] >= (res.statsAt || 0)) {
    res.stats = { gpuBytes: res.buf[0], gpuPeak: res.buf[1], gpuBuffers: res.buf[2],
                  wasmBytes: res.buf[3] || (res.stats && res.stats.wasmBytes) || null };
  }
  // Not expired on a clock. A reply that just finished slowly is exactly what someone is
  // looking at the strip about, and blanking the reading a minute later takes the answer
  // away at the moment it is wanted. It stands until the next reply replaces it, and says
  // which reply it is about.
  const stepP90 = resP90(res.steps), stepMid = resPct(res.steps, 0.5);
  const ratio = (res.steps.length >= 8 && stepMid) ? stepP90 / stepMid : 0;
  const frame = resP90(res.frames);
  const gpu = resLevel(ratio, RES_STEP_TIGHT, RES_STEP_SEVERE);
  const cpu = resLevel(frame, RES_FRAME_TIGHT, RES_FRAME_SEVERE);
  // Paging is the one pressure with a signature rather than a measurement: the GPU stalls
  // AND the main thread stutters at the same time, because the kernel is doing the work
  // both are waiting on. Either alone is something else -- a big model is slow without
  // stuttering, a busy tab stutters without slowing decode -- so this takes the LOWER of
  // the two and claims nothing when only one is raised.
  const paging = (gpu && cpu)
    ? RES_LEVELS[Math.min(RES_LEVELS.indexOf(gpu), RES_LEVELS.indexOf(cpu))] : null;
  const s = res.stats || {};
  let js = null;
  try { js = performance.memory ? performance.memory.usedJSHeapSize : null; } catch (e) {}
  return { gpuBytes: s.gpuBytes, gpuPeak: s.gpuPeak, wasmBytes: s.wasmBytes, jsBytes: js,
           storage: res.storage, gpu, cpu, paging, ratio, frame };
}

function resRender() {
  const bar = $('#resbar');
  if (!bar) return;
  const r = resRead();
  const open = localStorage.getItem(RES_KEY) !== '0';
  bar.hidden = false;
  $('#resCaret').textContent = open ? '▾' : '▸';
  $('#resToggle').setAttribute('aria-expanded', open ? 'true' : 'false');
  $('#resBody').hidden = !open;

  const worst = resWorst(r.paging, r.gpu, r.cpu);
  const worstName = r.paging === worst ? 'paging' : r.gpu === worst ? 'GPU' : 'CPU';
  $('#resSummary').innerHTML = open
    ? 'Resources'
    : 'Resources · ' + resBytes(r.gpuBytes) + ' on the GPU · '
      + (worst && worst !== 'normal'
          ? '<b class="res-' + worst + '">' + worstName + ' ' + RES_LABEL[worst] + '</b>'
          : '<span class="res-normal">no pressure</span>');
  if (!open) { $('#resBody').innerHTML = ''; return; }

  // Measured quantities carry their number. Pressures carry a level and nothing else --
  // there is no honest number behind them, and inventing one would be the whole mistake.
  const num = (label, v, extra) =>
    '<div class="rescell"><span class="reslab">' + label + '</span>'
    + '<span class="resval">' + v + '</span>'
    + (extra ? '<span class="resnote">' + extra + '</span>' : '') + '</div>';
  const lvl = (label, l, why) =>
    '<div class="rescell"><span class="reslab">' + label + '</span>'
    + '<span class="resval res-' + (l || 'unknown') + '">'
    + (l ? RES_LABEL[l] : 'no reading yet') + '</span>'
    + '<span class="resnote">' + why + '</span></div>';

  $('#resBody').innerHTML =
      num('GPU buffers', resBytes(r.gpuBytes),
          r.gpuPeak ? 'peak ' + resBytes(r.gpuPeak) : '')
    + num('WASM heap', resBytes(r.wasmBytes), 'Python')
    + num('JS heap', resBytes(r.jsBytes), 'this tab')
    + num('Stored', r.storage ? resBytes(r.storage.usage) + ' / ' + resBytes(r.storage.quota) : '—',
          'models and chats')
    + lvl('GPU pressure', r.gpu,
          r.ratio ? 'decode spread ' + r.ratio.toFixed(1) + '×'
                    + (streaming ? '' : ', last reply') : 'needs a reply to measure')
    + lvl('CPU pressure', r.cpu,
          r.frame ? 'frames ' + Math.round(r.frame) + ' ms' : 'measuring')
    + lvl('Paging pressure', r.paging, 'GPU and CPU together');
}

async function resTick() {
  if (worker) {
    // The worker is single-threaded, so this request queues behind whatever it is doing: a
    // reply in progress can hold it for twenty seconds, and until it answers the numbers on
    // screen are from before that reply started. Showing them anyway is how a leak that
    // climbed to 20GB read as "flat" -- so the panel says the figure is held rather than
    // presenting a stale one as current.
    if (res.pending && Date.now() - res.pending < 30000) return;   // one outstanding is enough
    const asked = Date.now();
    res.pending = asked;
    try {
      const v = await call('stats');
      if (res.pending === asked) { res.stats = v; res.statsAt = Date.now(); res.pending = 0; }
    } catch (e) { res.pending = 0; /* gone: keep the last, still marked by age */ }
  }
  // Storage is the expensive one and the slowest to change; asked for about once a minute.
  if (!res.poll || Date.now() - res.poll > 60000) {
    res.poll = Date.now();
    try {
      if (navigator.storage && navigator.storage.estimate) {
        const e = await navigator.storage.estimate();
        res.storage = { usage: e.usage || 0, quota: e.quota || 0 };
      }
    } catch (e) { /* a browser that will not say leaves this blank */ }
  }
  resRender();
}

function wireResources() {
  const bar = $('#resbar');
  if (!bar) return;
  $('#resToggle').onclick = () => {
    localStorage.setItem(RES_KEY, localStorage.getItem(RES_KEY) === '0' ? '1' : '0');
    resRender();
  };
  resFrames();
  resRender();
  // Faster while a reply is being written, because that is when the numbers move and when
  // someone is looking; slow otherwise, since each tick costs a message to the worker.
  // Two clocks, deliberately. Drawing runs on its own and never waits for the runtime,
  // because the runtime cannot answer during a reply -- and the first version put the draw
  // AFTER the await, so a request left outstanding for 51 seconds took the whole panel with
  // it: the figures sat at whatever they were before the reply began, with nothing on screen
  // saying they were stale. Asking is the slow half and is not allowed to hold the fast one.
  setInterval(resRender, 1000);
  const loop = () => {
    clearTimeout(res.timer);
    res.timer = setTimeout(() => { resTick().finally(loop); }, streaming ? 1000 : 4000);
  };
  resTick().finally(loop);
}

// ---- the Python-environment settings -------------------------------------------------
function wirePython() {
  const box = $('#pyPackages'), on = $('#pyEnabled'), st = $('#pyState');
  if (!box) return;
  const saved = localStorage.getItem(PYPKG_KEY);
  box.value = saved === null ? PY_DEFAULT : saved;
  on.checked = pyEnabled();
  const tools = $('#pyTools');
  if (tools) {
    tools.checked = localStorage.getItem(TOOLS_KEY) !== '0';
    tools.onchange = () => localStorage.setItem(TOOLS_KEY, tools.checked ? '1' : '0');
  }
  st.textContent = pyEnabled() ? 'starting…' : 'off';
  // What "Apply & load" did, said in the panel rather than on the hint bar under the
  // composer -- the settings dialog covers that, so a name that could not be loaded was
  // reported to a place nobody pressing this button can see. Silence read as success.
  const pyResult = (html, bad) => {
    const el = $('#pyResult');
    if (!el) return;
    el.hidden = false; el.innerHTML = html;
    el.classList.toggle('bad', !!bad);
  };
  const esc = (t) => String(t).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
  const nameList = (a) => a.map(n => '<code>' + esc(n) + '</code>').join(', ');

  $('#pyApply').onclick = async () => {
    localStorage.setItem(PYPKG_KEY, box.value);
    if (!on.checked) {
      localStorage.setItem(PYON_KEY, '0'); pyStop();
      pyResult('Running code is off. Nothing is loaded.', false);
      return;
    }
    localStorage.setItem(PYON_KEY, '1');
    const want = pyPackages();
    // Applying LOADS. That is the whole point of the setting existing.
    if (!pyWorker) {
      pyResult('Starting the runtime and loading ' + nameList(want) + '…', false);
      pyStart(); return;
    }
    st.textContent = 'loading…';
    pyResult('Loading ' + nameList(want) + '…', false);
    try {
      const r = await pyCall('packages', { packages: want });
      const missing = (r && r.unavailable) || [];
      const got = (r && r.loaded) || [];
      const pypi = (r && r.fromPyPI) || [];
      // Three outcomes, because they mean different things to whoever typed the name: it is
      // here, it had to be fetched from PyPI, or nothing can supply it -- and in that last
      // case the reason the runtime gave is worth more than any wording of mine.
      const parts = ['Ready: ' + (got.length ? nameList(got) : 'nothing')];
      if (pypi.length) parts.push('Installed from PyPI: ' + nameList(pypi));
      missing.forEach(m => {
        const nm = typeof m === 'string' ? m : m.name;
        const why = (typeof m === 'string' ? '' : m.why) || '';
        parts.push('<span class="miss">Could not load <code>' + esc(nm) + '</code>'
                 + (why ? ' — ' + esc(why.replace(/[.\s]+$/, '')) : '')
                 + '. Not in the Pyodide '
                 + (pyVersion ? esc(pyVersion) + ' ' : '')
                 + 'distribution, and no wheel on PyPI built for it. Which packages the '
                 + 'distribution carries changes between Pyodide releases, so a name missing '
                 + 'here may exist in another one. A wheel built for Pyodide can be added '
                 + 'from this device below.</span>');
      });
      pyResult(parts.join('<br>'), missing.length > 0);
    } catch (e) {
      st.textContent = 'failed';
      pyResult('<span class="miss">Loading failed: ' + esc(e.message || e) + '</span>', true);
    }
  };
  const install = async (args, label) => {
    st.textContent = 'installing ' + label + '…';
    pyResult('Installing <code>' + esc(label) + '</code>…', false);
    if (!pyWorker) pyStart();
    try {
      const r = await pyCall('install', args);
      pyResult('Installed <code>' + esc((r && r.installed) || label) + '</code>.', false);
    } catch (e) {
      st.textContent = 'install failed';
      pyResult('<span class="miss">Could not install <code>' + esc(label) + '</code>: '
             + esc(e.message || e) + '</span>', true);
    }
  };
  $('#pyWheelPick').onclick = () => $('#pyWheelFile').click();
  $('#pyWheelFile').onchange = async (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    // Sent as bytes: the worker has no access to the page's File objects, and a wheel is
    // small enough that copying it is not worth a stream.
    const bytes = await f.arrayBuffer();
    install({ bytes, name: f.name }, f.name);
  };
  $('#pyReset').onclick = async () => {
    if (!pyWorker) { pyStart(); return; }
    try { await pyCall('reset'); } catch (e) {}
    pyStop(); pyStart();
  };
  on.onchange = () => {
    localStorage.setItem(PYON_KEY, on.checked ? '1' : '0');
    if (on.checked) pyStart(); else pyStop();
  };
}

$('#navBtn').onclick = () => toggleSidebar();
$('#navScrim').onclick = () => toggleSidebar(false);
// A conversation picked from the drawer is a conversation you want to read, so get out of
// the way. Delegated, because the list is rebuilt on every change.
$('#convList').addEventListener('click', () => toggleSidebar(false));
$('#openSettings').onclick = () => { $('#settingsDlg').showModal(); refreshCache(); };
$('#closeSettings').onclick = () => $('#settingsDlg').close();

// ---- export / import (real .zip containing JSON) ----
$('#exportBtn').onclick = async () => {
  const payload = { version: 2, exported: new Date().toISOString(), conversations: convs };
  const blob = await WTZip.zip([{ name: 'chat.json', data: JSON.stringify(payload, null, 2) }]);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'webtorch-chat-' + Date.now() + '.zip';
  a.click(); URL.revokeObjectURL(a.href);
};
$('#importBtn').onclick = () => $('#importFile').click();
$('#importFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try {
    const entries = await WTZip.unzip(await f.arrayBuffer());
    const name = Object.keys(entries).find(k => k.endsWith('.json'));
    if (!name) throw new Error('no .json inside the zip');
    const data = JSON.parse(entries[name]);
    let added;
    if (Array.isArray(data.conversations)) added = data.conversations;
    else if (Array.isArray(data.messages))            // v1 export: a single conversation
      added = [{ id: 'c' + Date.now(), title: titleFrom((data.messages[0] || {}).content),
                 messages: data.messages, updated: Date.now() }];
    else throw new Error('not a webtorch chat export');
    convs = added.concat(convs); curId = convs[0] && convs[0].id;
    saveConvs(); renderConvs(); render();
    note('Imported ' + added.length + ' conversation(s).');
  } catch (err) { alert('Import failed: ' + err.message); }
  e.target.value = '';
};

// The conversation store is async now, so the first paint happens without it and the
// sidebar fills in when it arrives -- a few milliseconds, and nothing else waits on it.
render(); syncButtons();
loadConvs().then(() => {
  if (!convs.length) newConv(); else curId = convs[0].id;
  renderConvs(); render();
});
detectEnv().then(() => { fillPresets(); wireGpuMem(); });
// Ask the SDK to tell us when origin storage runs out, so the page can offer a folder.
call('armStorage', {}).catch(() => {});
wirePython();
wireDebug();
wireResources();
// After the model runtime, which is what the person is waiting for.
setTimeout(dbgStart, 2500);
// After the first paint: booting a second Python is a few seconds of CPU, and the model
// runtime starting up is what the person is actually waiting for.
setTimeout(pyStart, 1200);
note('Pick a model and press Load. Downloads come from ModelScope and are cached, so the next load is instant.');
call('boot').then(refreshCache).catch(e => {
  $('#modelStatus').textContent = 'runtime failed: ' + e.message;
  $('#openSettings').disabled = false;        // a dead runtime must not lock the UI shut
});
