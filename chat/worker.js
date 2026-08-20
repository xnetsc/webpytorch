/* Chat worker: boots Pyodide + the webtorch package, loads models from ModelScope with
   progress, generates, and exposes the SDK's cache-management calls. */
importScripts('../lib/pyodide/pyodide.js');
importScripts('../dist/wgpy-worker.js');
importScripts('../webtorch/js/webtorch-worker.js');

let pyodide = null, ready = false;
const send = (m) => postMessage(m);
const log = (t) => send({ type: 'log', text: t });


async function boot() {
  if (ready) return;
  // The SDK brings up the backend, Pyodide and the package in the order they require.
  const r = await webtorch.initWorker({
    baseURL: '../', stdout: log, stderr: log,
    onStatus: (t) => send({ type: 'status', text: t }),
  });
  pyodide = r.pyodide;
  // Model files come from ModelScope, fetched directly: its `resolve` route and the CDN it
  // redirects to send `Access-Control-Allow-Origin: *` and allow Range, so a cross-origin
  // isolated page can stream them. The SDK does the ranged reads and its own persistent cache.
  await pyodide.runPythonAsync(`
import json, webtorch
webtorch.set_io_read(webtorch.modelscope_read())
webtorch.set_io_write(webtorch.default_io_write)
_MODEL = {"m": None, "id": None}
`);
  send({ type: 'backend', name: r.backend });
  ready = true;
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
