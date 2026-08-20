/* Chat UI: model selection (any ModelScope model), load/release with status, cache
   management, camera/file/URL tools, and zip export/import of the conversation. */
const $ = (s) => document.querySelector(s);
const worker = new Worker('worker.js');
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
let ENV = { ramGB: 0, quotaGB: 0, cores: 0, webgpu: false, budgetGB: 0 };
async function detectEnv() {
  ENV.cores = navigator.hardwareConcurrency || 0;
  ENV.ramGB = navigator.deviceMemory || 0;            // coarse, Chrome-only; 0 = unknown
  try {
    // Persistent storage lifts the eviction risk and usually the quota, which decides how big
    // a model can be cached at all.
    if (navigator.storage && navigator.storage.persist) {
      try { await navigator.storage.persist(); } catch (e) { /* user may decline */ }
    }
    const est = await navigator.storage.estimate();
    ENV.quotaGB = (est.quota || 0) / 1e9;
  } catch (e) { ENV.quotaGB = 0; }
  try {
    if (navigator.gpu) {
      const ad = await navigator.gpu.requestAdapter();
      ENV.webgpu = !!ad;
      if (ad && ad.limits) ENV.maxBufferGB = (ad.limits.maxBufferSize || 0) / 1e9;
    }
  } catch (e) { ENV.webgpu = false; }
  // A model must both fit in memory and fit in the cache. Leave room for the runtime.
  const mem = ENV.ramGB ? ENV.ramGB * 0.55 : 12;      // unknown RAM -> assume a typical laptop
  const disk = ENV.quotaGB ? ENV.quotaGB * 0.8 : 999;
  ENV.budgetGB = Math.min(mem, disk);
  return ENV;
}
function envSummary() {
  const p = [];
  p.push(ENV.ramGB ? ENV.ramGB + ' GB RAM' : 'RAM unknown');
  if (ENV.cores) p.push(ENV.cores + ' cores');
  p.push(ENV.quotaGB ? ENV.quotaGB.toFixed(0) + ' GB cache quota' : 'quota unknown');
  p.push(ENV.webgpu ? 'WebGPU' : 'CPU only');
  return p.join(' · ') + ' → fits ≈' + ENV.budgetGB.toFixed(1) + ' GB';
}

function call(cmd, args) {
  const id = ++seq;
  return new Promise((res, rej) => { pending.set(id, { res, rej }); worker.postMessage({ id, cmd, args }); });
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
function fillPresets() {
  const sel = $('#preset'); sel.innerHTML = '';
  const custom = PRESETS[PRESETS.length - 1];
  const DEFAULT = PRESETS[0];                       // Qwen3.8-27B 3-bit: always the default
  const models = PRESETS.slice(0, -1).filter(m => m !== DEFAULT);
  // The default stays pinned at the top; the rest are ordered by what this machine can hold
  // (largest that fits first), with anything over budget kept but flagged.
  const fits = models.filter(m => m.gb <= ENV.budgetGB).sort((a, b) => b.gb - a.gb);
  const over = models.filter(m => m.gb > ENV.budgetGB).sort((a, b) => a.gb - b.gb);
  const ordered = [DEFAULT].concat(fits, over, [custom]);
  // Default to Qwen3.8-27B 3-bit when this machine can hold it; otherwise the largest that
  // fits; and when nothing fits (a phone, say) the smallest, so the pick is at least plausible.
  const smallest = models.concat([DEFAULT]).slice().sort((a, b) => a.gb - b.gb)[0];
  const noneFit = fits.length === 0 && DEFAULT.gb > ENV.budgetGB;
  const best = (DEFAULT.gb <= ENV.budgetGB) ? DEFAULT : (fits[0] || smallest);
  ordered.forEach((p) => {
    const o = document.createElement('option');
    o.value = PRESETS.indexOf(p);
    const tag = p.gb ? ' (' + p.gb.toFixed(1) + ' GB)' : '';
    // one verdict per entry — fits, or does not
    const fit = !p.gb ? ''
      : p.gb > ENV.budgetGB ? ' ⚠ too big for this device'
      : (p === best ? ' ✓ best fit here' : '');
    const dflt = (p === DEFAULT) ? ' — default' : '';
    o.textContent = p.label + tag + dflt + fit;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    const p = PRESETS[sel.value];
    // one box, and only for a model that is not in the list
    $('#modelId').value = p.repo ? (p.file ? p.repo + '/' + p.file : p.repo) : '';
    $('#customBox').hidden = !!p.repo;
  };
  // Default to Qwen3.8-27B 3-bit; if this machine cannot hold it, default to the best that fits.
  sel.value = PRESETS.indexOf(best); sel.onchange();
  $('#envInfo').textContent = envSummary() + (
      noneFit ? ' — no preset fits this device; the smallest is selected, and you can enter any '
                + 'other model below'
    : best === DEFAULT ? ''
    : ' — the default is too large here, so ' + best.label + ' is selected');
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
detectEnv().then(fillPresets);
note('Pick a model and press Load. Downloads come from ModelScope and are cached, so the next load is instant.');
call('boot').then(refreshCache).catch(e => $('#modelStatus').textContent = 'runtime failed: ' + e.message);
