/* Chat worker: boots Pyodide + the webtorch package, loads models from ModelScope with
   progress, generates, and exposes the SDK's cache-management calls. */
importScripts('../lib/pyodide/pyodide.js');
importScripts('../dist/wgpy-worker.js');

const PKG = 'webtorch';
const PKG_MODULES = ['__init__.py', '_core.py', '_sdk.py', 'torchshim.py', 'ggufload.py',
  'hfcompat.py', 'webenv.py', 'llm.py', 'vl.py', 'detection.py', 'tts.py', 'webio.py',
  'onnxrt.py', 'lm_engine.py', 'quantize.py', 'audiofe.py', 'cosyvoice.py',
  'multimodal.py', 'iqtables.py'];

let pyodide = null, ready = false;
const send = (m) => postMessage(m);
const log = (t) => send({ type: 'log', text: t });

async function fetchText(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + ': ' + r.status); return r.text(); }

async function boot() {
  if (ready) return;
  send({ type: 'status', text: 'starting Python runtime…' });
  pyodide = await loadPyodide({ indexURL: '../lib/pyodide/', stdout: log, stderr: log });
  self.pyodide = pyodide;
  send({ type: 'status', text: 'loading numpy…' });
  await pyodide.loadPackage(['micropip', 'numpy']);
  send({ type: 'status', text: 'installing GPU backend…' });
  try {
    const mp = pyodide.pyimport('micropip');
    await mp.install('../dist/wgpy_webgpu-1.0.0-py3-none-any.whl');
  } catch (e) { log('WebGPU backend unavailable (' + e.message + '), using CPU'); }
  send({ type: 'status', text: 'loading webtorch…' });
  try { pyodide.FS.mkdir(PKG); } catch (e) {}
  for (const m of PKG_MODULES) {
    try { pyodide.FS.writeFile(`${PKG}/${m}`, await fetchText(`../${PKG}/${m}`)); }
    catch (e) { log('skip ' + m + ': ' + e.message); }
  }
  await pyodide.runPythonAsync(`
import sys, json
sys.path.insert(0, "/")
import webtorch
# Downloads always come from ModelScope, with the SDK's persistent (IndexedDB-backed) cache
# so a model is fetched once and reused across reloads.
webtorch.set_io_read(webtorch.modelscope_read())
webtorch.set_io_write(webtorch.default_io_write)
_MODEL = {"m": None, "id": None}
`);
  ready = true;
  send({ type: 'status', text: 'ready' });
  send({ type: 'ready' });
}

async function loadModel(repo, file) {
  await boot();
  const src = file ? `${repo}/${file}` : repo;
  send({ type: 'status', text: `loading ${src} …` });
  pyodide.globals.set('_src', src);
  await pyodide.runPythonAsync(`
import js, time, webtorch
from webtorch import webio
src = _src
# progress: count the bytes the IO layer actually pulls
_orig = webio._fetch_once
_seen = {"n": 0, "t": time.time()}
async def _counting(url, rng, headers):
    b = await _orig(url, rng, headers)
    _seen["n"] += len(b)
    now = time.time()
    if now - _seen["t"] > 0.4:
        _seen["t"] = now
        js.postMessage(js.Object.fromEntries(js.Object.entries(
            js.JSON.parse(json.dumps({"type":"progress","bytes":_seen["n"]})))))
    return b
webio._fetch_once = _counting
try:
    if _MODEL["m"] is not None:
        webtorch.release(_MODEL["m"]); _MODEL["m"] = None
    m = await webtorch.load(src)
    _MODEL["m"] = m; _MODEL["id"] = src
finally:
    webio._fetch_once = _orig
`);
  send({ type: 'status', text: `ready: ${src}` });
  send({ type: 'loaded', id: src });
}

async function generate(prompt, opts) {
  if (!ready || !pyodide) throw new Error('no runtime');
  pyodide.globals.set('_prompt', prompt);
  pyodide.globals.set('_maxnew', (opts && opts.max_new) || 256);
  const out = await pyodide.runPythonAsync(`
m = _MODEL["m"]
if m is None:
    raise RuntimeError("no model loaded — pick one and press Load model")
r = m.generate(_prompt, max_new=int(_maxnew))
getattr(r, "text", str(r))
`);
  return out;
}

async function cacheList() {
  await boot();
  const out = await pyodide.runPythonAsync(`
import json, webtorch
items = await webtorch.list_cache()
hosts = await webtorch.cache_hosts()
json.dumps({"items":[{"key":e["key"],"host":e["host"],"size":e["size"],
                      "complete":bool(e["complete"])} for e in items],
            "hosts":hosts, "total": await webtorch.cache_size()})
`);
  return JSON.parse(out);
}
async function cacheDelete(key) {
  await boot(); pyodide.globals.set('_k', key);
  await pyodide.runPythonAsync(`import webtorch; await webtorch.delete_cache(_k)`);
}
async function cacheClear() {
  await boot();
  await pyodide.runPythonAsync(`import webtorch; await webtorch.clear_cache()`);
}
async function releaseModel() {
  if (!ready) return;
  await pyodide.runPythonAsync(`
import webtorch
if _MODEL["m"] is not None:
    webtorch.release(_MODEL["m"]); _MODEL["m"] = None; _MODEL["id"] = None
`);
  send({ type: 'status', text: 'model released' });
}

onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    let res = null;
    if (cmd === 'boot') await boot();
    else if (cmd === 'load') await loadModel(args.repo, args.file);
    else if (cmd === 'generate') res = await generate(args.prompt, args);
    else if (cmd === 'cacheList') res = await cacheList();
    else if (cmd === 'cacheDelete') await cacheDelete(args.key);
    else if (cmd === 'cacheClear') await cacheClear();
    else if (cmd === 'release') await releaseModel();
    send({ type: 'result', id, res });
  } catch (err) {
    send({ type: 'result', id, error: (err && err.message) || String(err) });
    send({ type: 'status', text: 'error: ' + ((err && err.message) || err) });
  }
};
