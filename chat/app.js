/* Chat UI: model selection (any ModelScope model), load/release with status, cache
   management, camera/file/URL tools, and zip export/import of the conversation. */
const $ = (s) => document.querySelector(s);
const worker = new Worker('worker.js');
let seq = 0; const pending = new Map();
let messages = [];           // {role, content, attachments:[{kind,name,text,dataUrl}]}
let attachments = [];
let modelLoaded = false;

// Presets are EXAMPLES ONLY — any ModelScope repo/file works via the two inputs.
const PRESETS = [
  { label: 'Qwen3.8-27B · 3-bit UD-Q3_K_XL (13.2 GB) — default', repo: 'unsloth/Qwen3.8-27B-GGUF', file: 'Qwen3.8-27B-UD-Q3_K_XL.gguf' },
  { label: 'Qwen3.8-27B · UD-IQ3_XXS (10.9 GB)', repo: 'unsloth/Qwen3.8-27B-GGUF', file: 'Qwen3.8-27B-UD-IQ3_XXS.gguf' },
  { label: 'Qwen3-4B-Instruct · Q4_K_M (2.5 GB) — smallest practical', repo: 'unsloth/Qwen3-4B-Instruct-2507-GGUF', file: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf' },
  { label: 'Qwen3-4B-Instruct · UD-Q3_K_XL (2.1 GB)', repo: 'unsloth/Qwen3-4B-Instruct-2507-GGUF', file: 'Qwen3-4B-Instruct-2507-UD-Q3_K_XL.gguf' },
  { label: 'Qwen3-8B · Q4_K_M (5.0 GB)', repo: 'unsloth/Qwen3-8B-GGUF', file: 'Qwen3-8B-Q4_K_M.gguf' },
  { label: 'Qwen3-30B-A3B-Instruct · MoE · UD-Q3_K_XL (13.8 GB)', repo: 'unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF', file: 'Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf' },
  { label: 'Qwen3-Coder-30B-A3B · MoE · UD-Q3_K_XL (13.8 GB)', repo: 'unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF', file: 'Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf' },
  { label: '— custom (type a repo/file below) —', repo: '', file: '' },
];

function call(cmd, args) {
  const id = ++seq;
  return new Promise((res, rej) => { pending.set(id, { res, rej }); worker.postMessage({ id, cmd, args }); });
}
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'result') {
    const p = pending.get(m.id); pending.delete(m.id);
    if (p) (m.error ? p.rej(new Error(m.error)) : p.res(m.res));
  } else if (m.type === 'status') { $('#modelStatus').textContent = m.text; }
  else if (m.type === 'progress') { showProgress(m.bytes); }
  else if (m.type === 'loaded') { modelLoaded = true; setBar(1); refreshCache(); }
  else if (m.type === 'log') { console.log('[py]', m.text); }
};
function setBar(f) { $('#bar').style.width = Math.min(100, f * 100) + '%'; }
function fmt(b) { return b > 1e9 ? (b/1e9).toFixed(2)+' GB' : b > 1e6 ? (b/1e6).toFixed(1)+' MB' : (b/1e3).toFixed(0)+' KB'; }
let expected = 0;
function showProgress(bytes) {
  $('#progressText').textContent = 'downloaded ' + fmt(bytes) + (expected ? ' of ~' + fmt(expected) : '');
  if (expected) setBar(bytes / expected); else setBar(((bytes / 5e8) % 1));
}

// ---- model ----
function fillPresets() {
  const sel = $('#preset');
  PRESETS.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p.label; sel.appendChild(o); });
  sel.onchange = () => { const p = PRESETS[sel.value]; $('#repo').value = p.repo; $('#file').value = p.file; };
  sel.value = 0; sel.onchange();
}
$('#loadBtn').onclick = async () => {
  const repo = $('#repo').value.trim(), file = $('#file').value.trim();
  if (!repo) return alert('Enter a ModelScope repo (org/repo).');
  $('#loadBtn').disabled = true; setBar(0); expected = 0;
  const m = /\((\d+(?:\.\d+)?)\s*GB\)/.exec(PRESETS[$('#preset').value]?.label || '');
  if (m) expected = parseFloat(m[1]) * 1e9;
  try { await call('load', { repo, file }); note('Model ready. Large models take a while on first load; afterwards they come from the cache.'); }
  catch (e) { $('#modelStatus').textContent = 'load failed: ' + e.message; note(e.message); }
  finally { $('#loadBtn').disabled = false; }
};
$('#releaseBtn').onclick = async () => { await call('release'); modelLoaded = false; setBar(0); $('#progressText').textContent=''; };

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
function render() {
  const el = $('#messages'); el.innerHTML = '';
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
  const msg = { role: 'user', content: text, attachments };
  messages.push(msg); attachments = []; renderAttachments();
  $('#input').value = ''; render();
  messages.push({ role: 'assistant', content: '…' }); render();
  try {
    const out = await call('generate', { prompt: promptFor(msg), max_new: 256 });
    messages[messages.length - 1].content = out;
  } catch (err) { messages[messages.length - 1].content = 'Error: ' + err.message; }
  render();
};
$('#input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
});
$('#newChat').onclick = () => { messages = []; attachments = []; render(); renderAttachments(); };

// ---- export / import (real .zip containing JSON) ----
$('#exportBtn').onclick = async () => {
  const payload = { version: 1, exported: new Date().toISOString(), messages };
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
    if (!Array.isArray(data.messages)) throw new Error('not a webtorch chat export');
    messages = data.messages; render(); note('Imported ' + messages.length + ' messages.');
  } catch (err) { alert('Import failed: ' + err.message); }
  e.target.value = '';
};

fillPresets(); render();
note('Pick a model and press Load. Downloads come from ModelScope and are cached, so the next load is instant.');
call('boot').then(refreshCache).catch(e => $('#modelStatus').textContent = 'runtime failed: ' + e.message);
