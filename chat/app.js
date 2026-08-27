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
const gpuInit = webtorch.initMain(worker, { backendOrder: ['webgpu', 'webgl'] })
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
  else if (m.type === 'progress') { if (m.total) expected = m.total;
    showProgress(m.bytes, m.rate, m.dlRate); }
  else if (m.type === 'loaded') { modelLoaded = true; modelImage = !!m.image;
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
      const now = performance.now();
      if (!live.t0) live.t0 = now;
      // decode speed = tokens after the first, over the time since the first; updated at
      // most twice a second so the number stays readable
      if (live.rate && live.tokens >= 2 && (live.tokens === 2 || now - live.rateT > 500)) {
        live.rateT = now; live.rate.hidden = false;
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
      : ' — compute: ' + m.name);
  }
};
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
let expected = 0;
function showProgress(bytes, rate, dlRate) {
  // "loaded", not "downloaded": the same meter covers a load served entirely from cache.
  // The download figure is shown only while bytes are actually coming over the wire --
  // it comes from the HTTP reader, and a cached load has none.
  const parts = ['loaded ' + fmt(bytes) + (expected ? ' of ' + fmt(expected) : '')];
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
  if (loading) { call('stopLoad'); return; }
  // a single identifier: "org/repo/file.gguf", or "org/repo" for a HF-format directory
  const id = $('#modelId').value.trim();
  if (!id) return alert('Enter a model as org/repo/file.gguf (or org/repo).');
  const repo = id, file = '';
  loading = true;
  $('#loadBtn').textContent = 'Stop loading';
  $('#loadBtn').title = 'Stop this load — what is already loaded stays cached';
  setBar(0); expected = 0;
  const chosen = PRESETS[$('#preset').value];
  if (chosen && chosen.gb) expected = chosen.gb * 1e9;
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
      ex.disabled = partial;
      ex.title = partial ? 'Finish downloading before exporting'
                         : (g.files > 1 ? 'Save as a .zip' : 'Save the file itself');
      ex.onclick = () => exportModel(g);
      const del = document.createElement('button'); del.textContent = 'delete';
      del.onclick = async () => {
        for (const key of g.keys) await call('cacheDelete', { key });
        refreshCache();
      };
      d.append(k, ex, del); el.appendChild(d);
    });
  } catch (e) { $('#cacheList').innerHTML = '<p class="hint">cache unavailable: ' + e.message + '</p>'; }
}

// A single-file model is saved as itself, so the exported file IS the model and any other
// tool reads it; several files become one .zip. The bytes stream from the SDK straight to
// disk through the picked handle, so a 12 GB model never becomes a Blob.
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
  note('Exporting ' + name + ' …');
  try {
    const n = await call('exportModel', { keys: g.keys, handle });
    note('Exported ' + name + ' (' + fmt(n) + ').');
  } catch (e) { note('Export failed: ' + e.message); }
  setBar(0); $('#progressText').textContent = '';
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
  // Double-click the message to edit it — except inside a code block, which is its own
  // editor already and where a double-click is how you select a word.
  else b.addEventListener('dblclick', (e) => {
    if (e.target.closest('pre, .runout, .editbox')) return;
    editMessage(m, b);
  });
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
let pyWorker = null, pySeq = 0, pyState = 'off';
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
  } finally { btn.disabled = pyState !== 'ready'; }
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
function renderMarkdown(text) {
  const m = markdown();
  if (!m) return null;
  try {
    return DOMPurify.sanitize(m.parse(String(text)),
                              { ADD_TAGS: MD_TAGS, ADD_ATTR: MD_ATTR });
  } catch (e) { return null; }
}
// Put `text` into `el` as rendered Markdown, or as plain text when rendering is unavailable.
// A finished message is rendered BLOCK BY BLOCK, each in its own container that can be
// edited on its own. A reply still streaming is rendered as one piece: it changes every
// token, nothing in it is editable yet, and re-splitting it per token buys nothing.
function setRendered(el, text, msg) {
  if (renderMarkdown('') == null) {            // no renderer: plain text, as before
    el.textContent = text; el.classList.remove('md'); el.classList.remove('blocks');
    return;
  }
  el.classList.add('md');
  if (!msg) {                                  // streaming
    el.classList.remove('blocks');
    el.innerHTML = renderMarkdown(text) || '';
    wireRunButtons(el, null);
    return;
  }
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
      if (e.target.closest('.runout, .blockedit')) return;
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
  table.querySelectorAll('th, td').forEach(cell => {
    cell.setAttribute('contenteditable', 'plaintext-only');
    cell.spellcheck = false;
    cell.addEventListener('blur', () => {
      const md = tableToMd(table, mdBlocks(msg.content)[i] || '');
      if (md && setBlock(msg, i, md)) saveConvs();
    });
    cell.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') cell.blur();
      // Enter inside a cell would add a line the Markdown cannot hold: a row is one line.
      if (e.key === 'Enter') { e.preventDefault(); cell.blur(); }
    });
  });
  const src = document.createElement('button');
  src.type = 'button'; src.className = 'srcbtn';
  src.textContent = '⌗'; src.title = 'Edit the source (add or remove rows and columns)';
  src.onclick = () => editBlock(msg, i, node);
  node.classList.add('tblblk');
  node.appendChild(src);
}

// Rebuild a Markdown table from the DOM. The alignment row is taken from the original
// source when the column count still matches -- alignment is not visible in the cells, and
// regenerating it from scratch would quietly discard it.
function tableToMd(table, src) {
  const rowOf = (tr) => [...tr.children].map(c =>
    c.textContent.replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim());
  const head = table.querySelector('thead tr');
  const body = [...table.querySelectorAll('tbody tr')];
  if (!head) return null;
  const cols = head.children.length;
  const srcDelim = (src.split('\n').find(l => /^\s*\|?\s*:?-{1,}/.test(l)) || '').trim();
  const delimOk = srcDelim && srcDelim.split('|').filter(x => x.trim()).length === cols;
  const delim = delimOk ? srcDelim
                        : '| ' + Array(cols).fill('---').join(' | ') + ' |';
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
    setRendered(live.ans, rest);          // no msg: a reply still arriving is not editable
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
  streaming = { convId: conv.id, idx: conv.messages.length - 1, live, body };
  syncButtons();                                      // Send becomes Stop

  try {
    // Images ride alongside the conversation: the worker turns each data URL into pixels
    // and gives them to the model as media. Without this the model only ever saw the text
    // note that a picture existed.
    const imgs = (msg.attachments || []).filter(a => a.kind === 'image' && a.dataUrl)
                                        .map(a => a.dataUrl);
    const r = await call('generate', Object.assign(
      { messages: msgs, images: imgs, prompt: promptFor(msg) }, genOpts()));
    if (r && r.truncated) {
      reply.content += '\n\n[stopped at ' + r.n + ' tokens — raise “Max reply length”'
                     + ' in ⚙ Settings, or leave it empty for no limit]';
    }
    if (!reply.content.trim()) reply.content = '(empty reply)';
    reply.stats = r || null;                    // final n / tok_s for the footer line
  } catch (err) {
    reply.content = (reply.content ? reply.content + '\n\n' : '') + 'Error: ' + err.message;
  } finally { stopDots(live); }
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
// ---- the Python-environment settings -------------------------------------------------
function wirePython() {
  const box = $('#pyPackages'), on = $('#pyEnabled'), st = $('#pyState');
  if (!box) return;
  const saved = localStorage.getItem(PYPKG_KEY);
  box.value = saved === null ? PY_DEFAULT : saved;
  on.checked = pyEnabled();
  st.textContent = pyEnabled() ? 'starting…' : 'off';
  $('#pyApply').onclick = async () => {
    localStorage.setItem(PYPKG_KEY, box.value);
    if (!on.checked) { localStorage.setItem(PYON_KEY, '0'); pyStop(); return; }
    localStorage.setItem(PYON_KEY, '1');
    // Applying LOADS. That is the whole point of the setting existing.
    if (!pyWorker) { pyStart(); return; }
    st.textContent = 'loading…';
    try {
      const r = await pyCall('packages', { packages: pyPackages() });
      if (r.unavailable && r.unavailable.length) {
        note('Not in this Pyodide build: ' + r.unavailable.join(', '));
      }
    } catch (e) { st.textContent = 'failed: ' + e.message; }
  };
  const install = async (args, label) => {
    st.textContent = 'installing ' + label + '…';
    if (!pyWorker) pyStart();
    try {
      const r = await pyCall('install', args);
      note('Installed ' + (r && r.installed ? r.installed : label) + '.');
    } catch (e) {
      st.textContent = 'install failed';
      note('Could not install ' + label + ': ' + e.message);
    }
  };
  $('#pyInstallUrl').onclick = () => {
    const u = $('#pyWheelUrl').value.trim();
    if (!u) return;
    install({ url: u }, u.split('/').pop());
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
// After the first paint: booting a second Python is a few seconds of CPU, and the model
// runtime starting up is what the person is actually waiting for.
setTimeout(pyStart, 1200);
note('Pick a model and press Load. Downloads come from ModelScope and are cached, so the next load is instant.');
call('boot').then(refreshCache).catch(e => {
  $('#modelStatus').textContent = 'runtime failed: ' + e.message;
  $('#openSettings').disabled = false;        // a dead runtime must not lock the UI shut
});
