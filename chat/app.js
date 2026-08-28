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

const worker = new Worker('worker.js');
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
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'result') {
    const p = pending.get(m.id); pending.delete(m.id);
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
  else if (m.type === 'loaded') {
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
      // decode speed = tokens after the first, over the time since the first; updated at
      // most twice a second so the number stays readable
      if (live.rate && live.tokens >= 2 && (live.tokens === 2 || wall - live.rateT > 500)) {
        live.rateT = wall; live.rate.hidden = false;
        const sps = (live.tokens - 1) / ((now - live.t0) / 1000);
        live.rate.textContent = live.tokens + ' tokens · ' + sps.toFixed(1) + ' tok/s';
      }
    }
    if (live && live.ans.isConnected) {
      fillBody(streaming.body, reply, streaming.live);
      const el = $('#messages'); el.scrollTop = el.scrollHeight;
    }
  }
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
//   Qwen3-0.6B Q4_K_M      118.6 -> 19.8 tok/s   6.0x
//   Qwen 3B Q4_K            35.7 -> 5.3          6.7x
//   Qwen3-30B-A3B MoE       34.8 -> 5.1          6.8x
//   Qwen3.8-27B hybrid       6.8 -> 0.76         8.9x   (i-quant, 48 of 64 layers recurrent)
//
// Stated inline rather than as a dialog: WebGL works and answers correctly, it is only
// slower, and the dialog is reserved for the case where the model will not run at all.
const WEBGL_SLOWDOWN = '6-9x';
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
    + '<strong>Qwen3-0.6B</strong> 19.8 tok/s against 118.6 on WebGPU; '
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
// New each time the page starts, and 128 bits of it: the topic is the only thing standing
// between a public broker and this page's diagnostics, so it is not guessable and it does
// not outlive the session that had the problem.
const DBG_TOPIC = 'webtorch/' + Array.from(crypto.getRandomValues(new Uint8Array(16)))
                                     .map(b => b.toString(16).padStart(2, '0')).join('');
let dbgClient = null;

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
    const c = mqtt.connect(dbgUrl(), { connectTimeout: 10000, reconnectPeriod: 15000 });
    dbgClient = c;
    c.on('connect', () => {
      c.subscribe(DBG_TOPIC + '/ask', (e) => {
        dbgSay(e ? ('subscribe failed: ' + e.message) : 'listening on ' + DBG_TOPIC);
      });
    });
    c.on('message', async (_t, payload) => {
      const cmd = payload.toString().trim().slice(0, 32) || 'ping';
      let body;
      try { body = await dbgAnswer(cmd); }
      catch (e) { body = { error: String(e && e.message || e) }; }
      try { c.publish(DBG_TOPIC + '/say', JSON.stringify({ cmd: cmd, body: body })); }
      catch (e) { /* nothing useful to do about a failed reply */ }
    });
    c.on('error', (e) => { dbgSay('error: ' + String(e && e.message || e)); });
    c.on('close', () => { if (dbgClient === c) dbgSay('disconnected'); });
  } catch (e) {
    dbgSay('unavailable: ' + String(e && e.message || e));
  }
}
function dbgStop() {
  if (dbgClient) { try { dbgClient.end(true); } catch (e) {} dbgClient = null; }
}

function wireDebug() {
  const t = $('#dbgTopic'); if (!t) return;
  t.value = DBG_TOPIC;
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
    // A topic is per-session by design; changing it means reloading into a new one.
    note('The topic is new every time the page starts — reload to get another.');
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
function showProgress(bytes, rate, dlRate) {
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
  $('#thinking').checked = localStorage.getItem(THINK_KEY) === '1';
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
  messages.forEach(m => el.appendChild(messageNode(m)));
  paintSlowNote();                       // lives in this list, so it is re-emitted with it
  el.scrollTop = el.scrollHeight;
}

// One message as DOM. The body is either rendered fresh (finished message) or driven
// incrementally by `live` while a reply is being streamed.
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
  } else {
    fillBody(b, m, live);
  }
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
    f.textContent = (m.stats.n || 0) + ' tokens · ' + Number(m.stats.tok_s).toFixed(1) + ' tok/s';
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
  pyWorker = new Worker('pyworker.js');
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
      document.querySelectorAll('.runbtn').forEach(b => { b.disabled = pyState !== 'ready'; });
    }
  };
  pyCall('boot', { packages: pyPackages() }).catch(() => {});
}
function pyStop() {
  if (!pyWorker) return;
  pyWorker.terminate(); pyWorker = null; pyState = 'off';
  const el = $('#pyState'); if (el) el.textContent = 'off';
  document.querySelectorAll('.runbtn').forEach(b => { b.disabled = true; });
}

// Attach a Run control to every Python block in `root` that has not got one. Called after a
// reply is rendered rather than woven into the renderer, so the sanitiser still sees plain
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
    if (!/^(py|python|python3)$/i.test(lang)) return;
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'runbtn'; btn.title = 'Run this code';
    btn.textContent = '▶';
    btn.disabled = pyState !== 'ready';
    btn.onclick = () => runBlock(code.textContent, wrap, btn);
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

async function runBlock(code, wrap, btn) {
  let panel = wrap.nextElementSibling;
  if (!panel || !panel.classList.contains('runout')) {
    panel = document.createElement('div'); panel.className = 'runout';
    wrap.parentNode.insertBefore(panel, wrap.nextSibling);
  }
  panel.textContent = 'running…'; panel.className = 'runout';
  btn.disabled = true;
  try {
    const r = await pyCall('run', { code });
    panel.textContent = '';
    // Output first, then the value of the last expression, then whatever it drew. Errors
    // are shown in the same place: a traceback is a result too.
    if (r.out) { const p = document.createElement('pre'); p.textContent = r.out.replace(/\n$/, ''); panel.appendChild(p); }
    if (r.err) {
      const p = document.createElement('pre'); p.className = 'stderr';
      p.textContent = r.err.replace(/\n$/, ''); panel.appendChild(p);
    }
    if (r.value != null && r.value !== 'None') {
      const p = document.createElement('pre'); p.className = 'val'; p.textContent = r.value; panel.appendChild(p);
    }
    (r.images || []).forEach((b64, k) => {
      const im = new Image(); im.src = 'data:image/png;base64,' + b64;
      wireImage(im, 'figure-' + (k + 1) + '.png');
      panel.appendChild(im);
    });
    if (r.error) {
      panel.classList.add('bad');
      const p = document.createElement('pre'); p.textContent = r.error; panel.appendChild(p);
    }
    if (!panel.childNodes.length) panel.textContent = '(no output)';
  } catch (e) {
    panel.classList.add('bad'); panel.textContent = String(e.message || e);
  } finally {
    btn.disabled = pyState !== 'ready';
    addRunoutClose(panel);
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
      .replace(/\$\$[ \t]*\n([\s\S]*?)\n[ \t]*\$\$/g, (_, f) => '$$' + f.trim() + '$$');
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
      live.pre.textContent = think === null ? '' : think;
      live.det.classList.toggle('live', rest === '');
      live.sum.textContent = rest === '' ? 'Thinking…' : 'Thought before answering';
      if (rest !== '' && !live.collapsed && !live.touched) {
        live.det.open = false; live.collapsed = true;                // auto-fold once the answer starts
      }
    }
    setRenderedLive(live.ans, rest);      // no msg: a reply still arriving is not editable
    return;
  }
  if (think !== null) {
    const det = document.createElement('details'); det.className = 'think';
    const sum = document.createElement('summary');
    sum.textContent = open ? 'Thinking…' : 'Thought before answering';
    const pre = document.createElement('pre'); pre.textContent = think;
    det.append(sum, pre); b.appendChild(det);
    det.open = open;                                                 // mid-think: watch it; done: folded
  }
  const ans = document.createElement('div');
  ans.className = 'ans';
  setRendered(ans, rest, live ? null : msg);
  b.appendChild(ans);
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
  const pre = document.createElement('pre');
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
    let r = null;
    for (let round = 0; ; round++) {
      const opts = Object.assign({ messages: msgs, images: imgs, prompt: promptFor(msg) },
                                 genOpts());
      const withTools = toolsEnabled();
      if (withTools) opts.tools = toolDefs();
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
      if (!toolsEnabled() || round >= MAX_TOOL_ROUNDS) break;
      const calls = parseToolCalls(reply.content);
      if (!calls.length) break;
      const results = [];
      for (const c of calls) results.push(await runToolCall(c));
      // The conversation the model sees next: what it said, then what each call returned.
      // It keeps its own markup here -- that is the format its template speaks.
      msgs.push({ role: 'assistant', content: reply.content });
      calls.forEach((c, i) => msgs.push({ role: 'tool', name: c.name, content: results[i] }));
      // What the reader sees keeps the prose and loses the protocol.
      reply.content = stripToolCalls(reply.content);
      // And what the READER sees: the call and its result, in the reply itself. A tool round
      // that only the model could see would leave the answer resting on something invisible.
      reply.content += '\n\n' + toolTrace(calls, results) + '\n\n';
      setRendered(body.querySelector('.ans') || body, reply.content);
      if (_stopRequested()) break;
    }
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
  return TOOLS.map(t => {
    const f = t.def.function;
    if (modelToolShape === 'flat') {
      return { name: f.name, description: f.description, parameters: f.parameters };
    }
    return t.def;                       // nested {type:"function", function:{…}}
  });
}
function findTool(name) { return TOOLS.find(t => t.def.function.name === name); }

const TOOLS_KEY = 'webtorch.toolsOn';
// Whether the LOADED model can be told about tools -- asked once per model, never guessed.
// null = not asked yet, so the first reply after a load does not race the probe.
let modelTakesTools = null;      // null = not asked yet
let modelToolShape = null;       // which definition shape its template actually reads
async function probeTools() {
  modelTakesTools = null; modelToolShape = null;
  try {
    const r = await call('toolsSupported');
    modelTakesTools = !!(r && r.ok);
    modelToolShape = (r && r.shape) || null;
  } catch (e) { modelTakesTools = false; }
  return modelTakesTools;
}
function toolsEnabled() {
  // Off unless asked for: a model that can run code on its own initiative is a different
  // thing from one that answers, and that should be a decision rather than a default.
  return localStorage.getItem(TOOLS_KEY) === '1' && pyEnabled() && TOOLS.length > 0
         && modelTakesTools === true;
}

// How many times a reply may call a tool and be asked again. A bound, because a model that
// answers every result with another call would otherwise never finish.
const MAX_TOOL_ROUNDS = 4;

// Tool calls out of a finished reply.
//
// `<tool_call>{…}</tool_call>` is what a ChatML template emits, and a bare JSON object is
// what a model without one tends to produce. Both are read, and anything else is simply not
// a tool call -- a reply that merely talks about JSON is left alone, because the object has
// to carry a name that is actually registered before it is run.
const TOOL_CALL_RE = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

// ONE scan, used for both running the calls and removing them from what is shown. Two
// separate readers drift: the reply ends up with a call the loop did not run, printed raw as
// prose, which is exactly what happened -- a bare call at the end of a message was neither
// executed nor hidden.
//
// A span counts as a call when it is inside <tool_call> markup (that is protocol, whatever
// name it carries) or when it is a bare JSON object naming a tool that ACTUALLY EXISTS here.
// Bare JSON with an unknown name is left alone: a reply may legitimately show a JSON object,
// and guessing would eat the answer.
function scanToolCalls(text) {
  const src = String(text);
  const found = [];
  const asCall = (o, wrapped) => {
    if (!o || typeof o.name !== 'string') return null;
    const known = !!findTool(o.name);
    if (!known && !wrapped) return null;
    return { name: o.name, args: o.arguments || o.parameters || {}, known };
  };
  TOOL_CALL_RE.lastIndex = 0;
  let m;
  while ((m = TOOL_CALL_RE.exec(src))) {
    let c = null;
    try { c = asCall(JSON.parse(m[1]), true); } catch (e) {}
    if (c) found.push({ start: m.index, end: m.index + m[0].length, call: c });
  }
  // Bare objects, wherever they sit. Brace-matched rather than regexed: the arguments are
  // themselves an object, and a pattern stopping at the first `}` would cut a call in half.
  for (let k = 0; k < src.length; k++) {
    if (src[k] !== '{') continue;
    if (found.some(f => k >= f.start && k < f.end)) continue;
    let depth = 0, inStr = false, esc = false, end = -1;
    for (let q = k; q < src.length; q++) {
      const ch = src[q];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === '{') depth++;
      else if (ch === '}') { depth--; if (depth === 0) { end = q + 1; break; } }
    }
    if (end < 0) break;
    let c = null;
    try { c = asCall(JSON.parse(src.slice(k, end)), false); } catch (e) {}
    if (c) { found.push({ start: k, end, call: c }); k = end - 1; }
  }
  found.sort((a, b) => a.start - b.start);
  return found;
}

function parseToolCalls(text) { return scanToolCalls(text).map(f => f.call); }

// The markup is protocol, not prose, and so is a bare call to a tool that exists here.
function stripToolCalls(text) {
  const src = String(text);
  const spans = scanToolCalls(src);
  if (!spans.length) return src.trim();
  let out = '', at = 0;
  spans.forEach(f => { out += src.slice(at, f.start); at = f.end; });
  out += src.slice(at);
  return out.replace(/\n{3,}/g, '\n\n').trim();
}

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
    // Which real tool it most likely meant: the one whose argument names its call already
    // fits. That is a fact about the call, not a guess at intent -- and it is only used to
    // address the correction to it, never to run anything.
    const given = Object.keys((typeof c.args === 'object' && c.args) || {});
    const guess = TOOLS.find(x => {
      const props = Object.keys((x.def.function.parameters || {}).properties || {});
      return given.length && given.every(k => props.includes(k));
    }) || TOOLS[0];
    const fixed = guess
      ? '<tool_call>\n' + JSON.stringify({ name: guess.def.function.name,
                                           arguments: c.args || {} }) + '\n</tool_call>'
      : '';
    return 'there is no tool called "' + c.name + '". These exist, with their arguments:\n'
         + listed + '\n\nYour call again, with the name corrected — send this:\n' + fixed;
  }
  try {
    const args = typeof c.args === 'string' ? JSON.parse(c.args) : (c.args || {});
    const r = await t.run(args);
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

// ---- the Python runtime, as tools -----------------------------------------------------
registerTool({
  type: 'function',
  function: {
    name: 'run_python',
    // The RETURN SHAPE is part of the contract, so it is stated here rather than left to be
    // inferred from whatever came back. JSON with named fields rather than prose: a model
    // that cannot tell stderr from stdout reports one as the other, and the first import of
    // a plotting library writes a notice to stderr that has been answered with as the result.
    description: 'Run Python. The interpreter is separate from the one holding the model and '
      + 'keeps its state between calls, so a name defined in one call is still there in the '
      + 'next. Use it to compute rather than to guess: arithmetic, parsing, checking that a '
      + 'library does what you are about to claim.\n\n'
      + 'Returns a JSON object:\n'
      + '  {"stdout": string, "stderr": string, "result": string|null, '
      + '"traceback": string|null, "figures": number}\n'
      + '  stdout    - what the code printed.\n'
      + '  stderr    - notices written by LIBRARIES, not by the code (a plotting package '
      + 'building its font cache, say). Never the answer.\n'
      + '  result    - the value of the last expression, or null if it had none. This is what '
      + 'to report when the task was to compute something.\n'
      + '  traceback - non-null means the code raised and there is NO result; fix it and call '
      + 'again.\n'
      + '  figures   - how many plots were drawn. They are shown to the reader; you cannot '
      + 'see them, so do not describe them.',
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
  return {
    stdout: clean(r.out),
    stderr: clean(r.err),
    // A run that raised HAS no result. Reporting the last value alongside a traceback invites
    // it to be quoted as the answer.
    result: r.error ? null : (r.value != null && r.value !== 'None' ? String(r.value) : null),
    traceback: r.error ? String(r.error) : null,
    figures: (r.images || []).length,
  };
});

registerTool({
  type: 'function',
  function: {
    name: 'install_python_packages',
    description: 'Make Python modules importable, by name. Anything the Pyodide distribution '
      + 'ships comes from there; anything else is fetched from PyPI if it has a pure-Python '
      + 'wheel. A package with compiled code cannot be added this way. Call with no names to '
      + 'ask what is already loaded.\n\n'
      + 'Returns a JSON object:\n'
      + '  {"ready": string[], "installed_from_pypi": string[], '
      + '"unavailable": [{"name": string, "why": string}]}\n'
      + '  ready               - every module importable now.\n'
      + '  installed_from_pypi - the ones that had to be fetched.\n'
      + '  unavailable         - could not be supplied, with the reason. Do not ask again '
      + 'for the same name; use a different library or say it cannot be done.',
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
    description: 'Throw away the Python interpreter and start a fresh one. Everything defined '
      + 'so far is lost and the configured modules are loaded again. Use it when the state '
      + 'has become confusing or something is wedged - not to recover from an ordinary '
      + 'traceback, which loses work for nothing.\n\n'
      + 'Returns a JSON object:\n'
      + '  {"restarted": true, "state": string, "ready": string[]}\n'
      + '  state - "ready" when the new interpreter is usable; anything else means it is not '
      + 'yet, and running code will fail.\n'
      + '  ready - the modules importable in the fresh interpreter.',
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


// ---- the Python-environment settings -------------------------------------------------
function wirePython() {
  const box = $('#pyPackages'), on = $('#pyEnabled'), st = $('#pyState');
  if (!box) return;
  const saved = localStorage.getItem(PYPKG_KEY);
  box.value = saved === null ? PY_DEFAULT : saved;
  on.checked = pyEnabled();
  const tools = $('#pyTools');
  if (tools) {
    tools.checked = localStorage.getItem(TOOLS_KEY) === '1';
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
