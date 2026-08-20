/* Chat UI: model selection (any ModelScope model), load/release with status, cache
   management, camera/file/URL tools, and zip export/import of the conversation. */
const $ = (s) => document.querySelector(s);
const worker = new Worker('worker.js');
// One SDK call brings up the GPU backend's main-thread half. Until it resolves the worker
// must not be spoken to, so `call` waits on it.
const gpuInit = webtorch.initMain(worker, { backendOrder: ['webgpu', 'webgl'] })
  .then(r => r.backend, () => 'cpu');
let seq = 0; const pending = new Map();
// Conversations live in localStorage so the sidebar survives a reload.
// Each: {id, title, messages:[{role, content, attachments:[{kind,name,text,dataUrl}]}], updated}
const STORE = 'webtorch-chat-convs';
let convs = [];
let curId = null;
let attachments = [];
let modelLoaded = false;

function loadConvs() {
  try { convs = JSON.parse(localStorage.getItem(STORE) || '[]'); } catch (e) { convs = []; }
  if (!Array.isArray(convs)) convs = [];
}
function saveConvs() {
  try { localStorage.setItem(STORE, JSON.stringify(convs)); } catch (e) { /* quota */ }
}
function current() { return convs.find(c => c.id === curId) || null; }


// Presets are EXAMPLES ONLY — any ModelScope repo/file works via the two inputs.
// Examples only — any ModelScope repo/file works through the two inputs below.
// Full-size models at 3-bit-ish quantization, across several families and both dense and MoE,
// so the picker is not tied to one vendor or one architecture. `gb` drives the environment fit.
const PRESETS = [
  { gb: 13.2, label: 'Qwen3.8-27B · 3-bit UD-Q3_K_XL', repo: 'unsloth/Qwen3.8-27B-GGUF', file: 'Qwen3.8-27B-UD-Q3_K_XL.gguf' },
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
  else if (m.type === 'progress') { showProgress(m.bytes); }
  else if (m.type === 'loaded') { modelLoaded = true; setBar(1); syncButtons(); refreshCache(); }
  else if (m.type === 'log') { console.log('[py]', m.text); }
  else if (m.type === 'backend') {
    ENV.backend = m.name;
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
function syncButtons() {
  $('#loadBtn').disabled = modelLoaded;
  $('#releaseBtn').disabled = !modelLoaded;
  $('#loadBtn').title = modelLoaded ? 'Release the current model first' : '';
  // Nothing in the conversation area is usable until a model is actually ready to answer.
  const off = !modelLoaded;
  ['#input', '#send', '#toolFile', '#toolCam', '#toolUrl', '#newChat'].forEach(sel => {
    const el = $(sel); if (el) el.disabled = off;
  });
  $('#input').placeholder = off ? 'Load a model in ⚙ Settings to start chatting'
                                : 'Send a message…';
  $('#composer').classList.toggle('locked', off);
}
function fmt(b) { return b > 1e9 ? (b/1e9).toFixed(2)+' GB' : b > 1e6 ? (b/1e6).toFixed(1)+' MB' : (b/1e3).toFixed(0)+' KB'; }
let expected = 0;
function showProgress(bytes) {
  $('#progressText').textContent = 'downloaded ' + fmt(bytes) + (expected ? ' of ~' + fmt(expected) : '');
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

  sel.onchange = () => {
    const p = PRESETS[sel.value];
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

  sel.value = PRESETS.indexOf(best); sel.onchange();
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
$('#loadBtn').onclick = async () => {
  // a single identifier: "org/repo/file.gguf", or "org/repo" for a HF-format directory
  const id = $('#modelId').value.trim();
  if (!id) return alert('Enter a model as org/repo/file.gguf (or org/repo).');
  const repo = id, file = '';
  $('#loadBtn').disabled = true; setBar(0); expected = 0;
  const chosen = PRESETS[$('#preset').value];
  if (chosen && chosen.gb) expected = chosen.gb * 1e9;
  try { await call('load', { repo, file }); note('Model ready. Large models take a while on first load; afterwards they come from the cache.'); }
  catch (e) { $('#modelStatus').textContent = 'load failed: ' + e.message; note(e.message); }
  finally { syncButtons(); }
};
$('#releaseBtn').onclick = async () => {
  await call('release'); modelLoaded = false; setBar(0);
  $('#progressText').textContent = ''; syncButtons();
  note('Model released. Its files stay cached, so loading it again is fast.');
};

// ---- cache ----
async function refreshCache() {
  try {
    const c = await call('cacheList');
    const el = $('#cacheList'); el.innerHTML = '';
    if (!c.items.length) { el.innerHTML = '<p class="hint">nothing cached yet</p>'; return; }
    const head = document.createElement('p'); head.className = 'hint';
    head.textContent = 'total ' + fmt(c.total);
    el.appendChild(head);
    c.items.sort((a, b) => b.size - a.size).forEach(it => {
      const d = document.createElement('div'); d.className = 'item';
      const k = document.createElement('span'); k.className = 'k';
      k.title = it.key; k.textContent = it.key.split('/').pop() + ' · ' + fmt(it.size);
      const b = document.createElement('button'); b.textContent = 'delete';
      b.onclick = async () => { await call('cacheDelete', { key: it.key }); refreshCache(); };
      d.append(k, b); el.appendChild(d);
    });
  } catch (e) { $('#cacheList').innerHTML = '<p class="hint">cache unavailable: ' + e.message + '</p>'; }
}
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
    const dataUrl = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(f); });
    addAttachment({ kind: 'image', name: f.name, dataUrl });
  } else {
    const text = await f.text();
    addAttachment({ kind: 'file', name: f.name, text: text.slice(0, 20000) });
  }
  e.target.value = '';
};
$('#toolUrl').onclick = async () => {
  const url = prompt('Fetch a URL and add its text to the message:');
  if (!url) return;
  try {
    const r = await fetch(url); const t = await r.text();
    const text = t.replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<[^>]+>/g, ' ')
                  .replace(/\s+/g, ' ').trim().slice(0, 20000);
    addAttachment({ kind: 'url', name: url, text });
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

function render() {
  const el = $('#messages'); el.innerHTML = '';
  const conv = current();
  const messages = conv ? conv.messages : [];
  if (!messages.length) {
    el.innerHTML = '<div class="msg bot"><div class="who">AI</div><div class="body">' +
      'Pick a model in ⚙ Settings, load it, and start chatting.</div></div>';
    return;
  }
  messages.forEach(m => {
    const d = document.createElement('div'); d.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');
    const w = document.createElement('div'); w.className = 'who'; w.textContent = m.role === 'user' ? 'You' : 'AI';
    const b = document.createElement('div'); b.className = 'body';
    let txt = m.content;
    const think = /<think>([\s\S]*?)<\/think>/.exec(txt);
    if (think) {
      const t = document.createElement('div'); t.className = 'think'; t.textContent = think[1].trim();
      b.appendChild(t); txt = txt.replace(think[0], '').trim();
    }
    b.appendChild(document.createTextNode(txt));
    (m.attachments || []).forEach(a => {
      if (a.dataUrl) { const im = new Image(); im.src = a.dataUrl; b.appendChild(im); }
      else { const p = document.createElement('div'); p.className = 'hint'; p.textContent = a.kind + ': ' + a.name; b.appendChild(p); }
    });
    d.append(w, b); el.appendChild(d);
  });
  el.scrollTop = el.scrollHeight;
}
function note(t) { $('#hintbar').textContent = t || ''; }
function promptFor(m) {
  let p = '';
  (m.attachments || []).forEach(a => {
    if (a.text) p += `[${a.kind}: ${a.name}]\n${a.text}\n\n`;
    else if (a.dataUrl) p += `[image attached: ${a.name}]\n`;
  });
  return p + m.content;
}
$('#composer').onsubmit = async (e) => {
  e.preventDefault();
  const text = $('#input').value.trim();
  if (!text && !attachments.length) return;
  if (!modelLoaded) return note('Load a model first (left panel).');
  const conv = ensureConv();
  const msg = { role: 'user', content: text, attachments };
  conv.messages.push(msg); attachments = []; renderAttachments();
  if (conv.messages.length === 1) conv.title = titleFrom(text);
  $('#input').value = ''; render(); renderConvs();
  conv.messages.push({ role: 'assistant', content: '…' }); render();
  try {
    const out = await call('generate', { prompt: promptFor(msg), max_new: 256 });
    conv.messages[conv.messages.length - 1].content = out;
  } catch (err) { conv.messages[conv.messages.length - 1].content = 'Error: ' + err.message; }
  conv.updated = Date.now(); saveConvs(); render();
};
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#newChat').onclick = () => { attachments = []; renderAttachments(); newConv(); };
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

loadConvs(); if (!convs.length) newConv(); else curId = convs[0].id;
renderConvs(); render(); syncButtons();
detectEnv().then(() => { fillPresets(); wireGpuMem(); });
note('Pick a model and press Load. Downloads come from ModelScope and are cached, so the next load is instant.');
call('boot').then(refreshCache).catch(e => $('#modelStatus').textContent = 'runtime failed: ' + e.message);
