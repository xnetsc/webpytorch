/* Chat worker: boots Pyodide + the webtorch package, loads models from ModelScope with
   progress, generates, and exposes the SDK's cache-management calls. */
// Pyodide from a CDN unless told otherwise, so this page works on a plain static host.
// Where Python comes from. The local distribution if this checkout has one, the CDN if not.
//
// Pyodide has NO package cache of its own: `loadPackage` fetches from `indexURL` every time,
// and what saves a reload from re-downloading is only the browser's HTTP cache, which is
// evictable. A copy served from this origin is not a cache, it is the source -- no network,
// no eviction, and it works with the machine offline.
//
importScripts('./pyodide-version.js');
const PYODIDE_URL = self.PYODIDE_URL || self.PYODIDE_CDN;
importScripts(PYODIDE_URL + 'pyodide.js');
importScripts('../dist/wgpy-worker.js');
importScripts('../webtorch/js/webtorch-worker.js');

let pyodide = null, ready = false;
const send = (m) => postMessage(m);
const log = (t) => send({ type: 'log', text: t });


async function boot() {
  if (ready) return;
  // The SDK brings up the backend, Pyodide and the package in the order they require.
  const r = await webtorch.initWorker({
    baseURL: '../', pyodideIndexURL: PYODIDE_URL, stdout: log, stderr: log,
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
  // When it is not the GPU, ask the SDK why. The reason is recorded at the point of failure
  // rather than inferred here, which is the only way to tell a missing WebGPU from a page
  // that is simply not cross-origin isolated.
  let why = null;
  if (r.backend !== 'webgpu') {
    try {
      why = await pyodide.runPythonAsync('import webtorch; webtorch.backend_reason()');
    } catch (e) { why = null; }
  }
  send({ type: 'backend', name: r.backend, why: why });
  ready = true;
  send({ type: 'ready' });
}

async function loadModel(repo, file, lmax) {
  await boot();
  const src = file ? `${repo}/${file}` : repo;
  send({ type: 'status', text: `loading ${src} …` });
  pyodide.globals.set('_src', src);
  pyodide.globals.set('_lmax', lmax || 0);
  // Progress comes from the SDK's own read hook, which counts bytes SERVED. A model
  // already cached does no network at all, so counting fetches would leave that load
  // looking frozen -- this reports either way, and needs no knowledge of SDK internals.
  // Two different things from two different owners: how far the LOAD has got comes from
  // the SDK, and the download speed comes from the HTTP reader we installed -- the SDK has
  // no transport and cannot know one.
  self.__dlRate = 0;
  self.__prog = (done, total, rate) => {
    const now = Date.now();
    if (now - (self.__progT || 0) < 250 && done !== total) return;
    self.__progT = now;
    send({ type: 'progress', bytes: done, total: total || 0,
           rate, dlRate: self.__dlRate });
  };
  self.__dl = (rate) => { self.__dlRate = rate; };
  const out = await pyodide.runPythonAsync(`
import js, webtorch
src = _src
lmax = int(_lmax)
webtorch.cancel(False)      # a stale stop request must not hit this load
webtorch.set_read_progress(lambda i: js.self.__prog(i["done"], i["total"] or 0, i["rate"]))
webtorch.set_download_progress(lambda i: js.self.__dl(i["rate"]))
try:
    if _MODEL["m"] is not None:
        webtorch.release(_MODEL["m"]); _MODEL["m"] = None
    m = await webtorch.load(src, **({"lmax": lmax} if lmax else {}))
    _MODEL["m"] = m; _MODEL["id"] = src
finally:
    webtorch.set_read_progress(None)
    webtorch.set_download_progress(None)
getattr(_MODEL["m"], "kind", "")
`);
  // The page uses the model kind to offer (or hide) image inputs before the user tries one.
  // It comes back as this block's value: a top-level name assigned inside an awaited block
  // is local to the coroutine Pyodide wraps it in, so a second runPython could not see it.
  send({ type: 'status', text: `ready: ${src}` });
  send({ type: 'loaded', id: src, image: out === 'multimodal' });
}

// Attached images become real pixels for the model. A data URL means nothing to the
// decoder, so decode here -- the worker has OffscreenCanvas -- and hand Python raw RGB plus
// its shape. Sending the bytes rather than the URL also keeps the Python side free of any
// image format handling.
async function decodeImages(urls) {
  const out = [];
  for (const u of urls || []) {
    try {
      const bmp = await createImageBitmap(await (await fetch(u)).blob());
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const g = c.getContext('2d');
      g.drawImage(bmp, 0, 0);
      const px = g.getImageData(0, 0, bmp.width, bmp.height).data;   // RGBA
      const rgb = new Uint8Array(bmp.width * bmp.height * 3);
      for (let i = 0, j = 0; i < px.length; i += 4, j += 3) {
        rgb[j] = px[i]; rgb[j + 1] = px[i + 1]; rgb[j + 2] = px[i + 2];
      }
      out.push({ w: bmp.width, h: bmp.height, rgb });
      bmp.close();
    } catch (e) { send({ type: 'log', text: 'image decode failed: ' + e.message }); }
  }
  return out;
}

// Can this model be told about tools at all?
//
// Rendered twice, with and without. A template that does not take tools either raises or --
// worse, because it is silent -- produces the very same prompt, and in both cases telling
// the model about tools is at best pointless and at worst a broken prompt. Comparing the
// two outputs is the only answer that does not depend on knowing the model.
async function toolsSupported() {
  if (!ready || !pyodide) return false;
  try {
    // The result must be the last TOP-LEVEL expression: a value produced inside an `if`
    // body is not what Pyodide hands back, so this assigns and ends with the name.
    return await pyodide.runPythonAsync(`
_ok = False
_m = _MODEL["m"]
_t = getattr(getattr(_m, "impl", _m), "tok", None) if _m is not None else None
if _t is not None:
    _msgs = [{"role": "user", "content": "hi"}]
    _probe = [{"type": "function", "function": {
        "name": "_probe_tool", "description": "probe",
        "parameters": {"type": "object", "properties": {}}}}]
    try:
        _a = _t.encode_chat(None, None, messages=_msgs)
        _b = _t.encode_chat(None, None, messages=_msgs, tools=_probe)
        _ok = bool(_a != _b)
    except Exception:
        _ok = False
_ok
`);
  } catch (e) { return false; }
}

async function generate(prompt, opts) {
  // A stop asked for during the LAST reply must not end this one before it starts.
  await pyodide.runPythonAsync('import webtorch; webtorch.cancel(False)');
  if (!ready || !pyodide) throw new Error('no runtime');
  const imgs = await decodeImages((opts || {}).images);
  pyodide.globals.set('_prompt', prompt || '');
  pyodide.globals.set('_imgs', imgs.map(i => ({ w: i.w, h: i.h, rgb: i.rgb })));
  pyodide.globals.set('_opts', JSON.stringify(opts || {}));
  // Streaming is the default here: each token is pushed to the page the moment it is
  // decoded, so the reply is readable while it is still being written. The SDK's
  // `stream=True` does the decode; this layer only ferries token -> message.
  // Stamped on the WORKER's clock, at the moment the token leaves the decode loop.
  // The page used to time these on its own clock, which put the cost of rendering the
  // reply -- and the delivery latency of the message itself -- inside the tok/s it was
  // reporting for the model. Only differences between these stamps are ever used, so
  // the worker having its own time origin does not matter.
  self.__chunk = (ch, t) => send({ type: 'chunk', channel: ch, text: t,
                                   at: performance.now() });
  try {
    const out = await pyodide.runPythonAsync(`
import json, js
m = _MODEL["m"]
if m is None:
    raise RuntimeError("no model loaded — pick one and press Load model")
_o = json.loads(_opts)
_n = int(_o.get("max_new") or 0)          # 0 = no budget: run until the model stops itself
_think = bool(_o.get("enable_thinking"))
_msgs = _o.get("messages") or None        # full conversation; falls back to the one prompt
# Every generation option the SDK takes is forwarded by name, so adding a control to the
# page needs no change here -- and an option the page does not set stays at the model's own
# default rather than being overridden with a guess.
# "tools" rides here too: the SDK hands it to the model's own chat template, so which
# models can be told about tools is a question about their template, not about this list.
# (No backticks in this block -- it is inside a JS template literal.)
_PASS = ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
         "presence_penalty", "frequency_penalty", "min_new_tokens", "max_length", "stop",
         "tools")
_kw = dict(max_new=_n or None, stream=True, channels=True, enable_thinking=_think)
for _k in _PASS:
    _v = _o.get(_k)
    if _v is not None and _v != "" and _v != []:
        _kw[_k] = _v
# Images go to the model as media. That path builds input embeddings, which the streaming
# decode cannot take, so a reply with a picture is produced in one piece and delivered as a
# single chunk -- the same message shape either way.
_media = None
_lst = _imgs.to_py() if hasattr(_imgs, "to_py") else (list(_imgs) if _imgs else [])
if _lst:
    import numpy as _np
    _media = []
    for _im in _lst:
        _w = int(_im["w"]); _h = int(_im["h"])
        _buf = _im["rgb"]
        _buf = _buf.to_py() if hasattr(_buf, "to_py") else _buf
        _media.append(_np.frombuffer(bytes(_buf), dtype=_np.uint8).reshape(_h, _w, 3))
    _media = _media[0] if len(_media) == 1 else _media

if _media is not None:
    # The model itself says whether it can see, through the same "kind" the page uses to
    # offer the controls -- not through whichever attribute a particular implementation
    # happens to keep its encoder in.
    _im = getattr(m, "impl", m)
    if (getattr(m, "kind", "") != "multimodal"
            and not hasattr(_im, "encoder") and not hasattr(_im, "vision")):
        raise RuntimeError("this model cannot see images — load a vision model to send one")
    _kw.pop("stream", None); _kw.pop("channels", None)
    _r = m.generate(_prompt, media=_media, **_kw)
    _txt = _r.text if hasattr(_r, "text") else str(_r)
    js.self.__chunk("content", _txt)
else:
    _gen = m.generate(messages=_msgs, **_kw) if _msgs else m.generate(_prompt, **_kw)
    for _c in _gen:
        js.self.__chunk(_c["channel"], _c["text"])
_s = getattr(getattr(m, "impl", m), "last_stream", None) or {}
json.dumps({"n": int(_s.get("n") or 0), "truncated": bool(_s.get("truncated")),
            "ttft_s": _s.get("ttft_s"), "tok_s": _s.get("tok_s")})
`);
    return JSON.parse(out);
  } finally { self.__chunk = null; }
}

async function cacheList() {
  await boot();
  const out = await pyodide.runPythonAsync(`
import json, webtorch
items = await webtorch.list_cache()
hosts = await webtorch.cache_hosts()
groups = await webtorch.model_groups()
json.dumps({"items":[{"key":e["key"],"host":e["host"],"size":e["size"],
                      "complete":bool(e["complete"])} for e in items],
            "groups":[{"name":g["name"],"label":g["label"],"keys":g["keys"],
                       "size":g["size"],"total":g["total"],"files":g["files"],
                       "complete":bool(g["complete"]),"partial":g["partial"]} for g in groups],
            "hosts":hosts, "total": await webtorch.cache_size()})
`);
  return JSON.parse(out);
}

// Export and import stream through the file itself. Functions cannot cross postMessage,
// but a FileSystemFileHandle and a File can, so the page picks the file (which needs a user
// gesture) and all the streaming happens here: the SDK hands over one chunk at a time and
// each goes straight to disk. A 12 GB model is never a Blob and never a wasm allocation.
// 配额满时告诉页面，由它去要一个目录；SDK 不弹窗，页面才有用户手势。
async function armStorageWatch() {
  await boot();
  self.__full = (key) => send({ type: 'storageFull', key });
  await pyodide.runPythonAsync(`
import js, webtorch
webtorch.set_storage_full(lambda i: js.self.__full(i["key"]))
`);
}

// 把已经存在 IndexedDB 里的搬到用户选的目录，搬一条删一条，之后改用该目录。
async function migrateToDirectory(dir) {
  await boot();
  pyodide.globals.set('_dir', dir);
  self.__mig = (n, key) => send({ type: 'migrate', bytes: n, key });
  return await pyodide.runPythonAsync(`
import js, webtorch
await webtorch.migrate_cache(_dir, on_progress=lambda n, k: js.self.__mig(n, k))
`);
}

// 导入 = 登记本地文件/文件夹，就地读，不复制、不入 IndexedDB。
// name = 页面算好的身份：单文件传内容指纹（同一文件永远是同一 id），目录传目录名。
async function importModel(handle, name) {
  await boot();
  pyodide.globals.set('_h', handle);
  pyodide.globals.set('_n', name || '');
  return await pyodide.runPythonAsync(`
import json, webtorch
json.dumps(await webtorch.import_model(_h, _n or None))
`);
}

let exportTotal = 0;
async function exportModel(keys, handle) {
  await boot();
  // How much there is to write, so the meter is a fraction rather than a rising number.
  try {
    pyodide.globals.set('_ekeys', keys);
    exportTotal = Number(await pyodide.runPythonAsync(`
import webtorch
_items = await webtorch.list_cache()
_want = set(_ekeys)
sum(int(i.get("size") or 0) for i in _items if i.get("key") in _want)
`)) || 0;
  } catch (e) { exportTotal = 0; }
  let w;
  try {
    w = await handle.createWritable();
  } catch (e) {
    // Say which failure this was. "Failed to execute 'createWritable'" on its own sends
    // whoever reads it looking in the wrong place -- the usual cause is the write permission
    // on the picked file, which only the page can obtain.
    throw new Error('cannot write to the chosen file (' + (e && e.name || 'error') + ': '
      + String((e && e.message) || e).slice(0, 160) + '). If the browser asked for permission '
      + 'and it was dismissed, choose the file again and allow it.');
  }
  let done = 0;
  self.__sink = async (bytes) => {
    const u8 = bytes.toJs ? bytes.toJs() : bytes;
    await w.write(u8);
    done += u8.length;
    // Its own message type. Reported as `progress` it was rendered as "loaded …", which
    // reads as a model being loaded -- and an export of a few hundred megabytes takes long
    // enough (81 s measured for 400 MB) that the only other thing to look at is the file
    // itself, which stays at ZERO until `close()`: the browser writes a temporary file and
    // swaps it in at the end. Nothing wrong, nothing to see, and no way to tell.
    send({ type: 'exporting', bytes: done, total: exportTotal });
  };
  try {
    pyodide.globals.set('_keys', keys);
    await pyodide.runPythonAsync(`
import js, webtorch
from pyodide.ffi import to_js
async def _w(b):
    await js.self.__sink(to_js(b))
await webtorch.export_model(list(_keys), _w)
`);
  } finally {
    await w.close();
    self.__sink = null;
  }
  return done;
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

// Sets the SDK's one-shot stop flag; the load in flight raises at its next IO checkpoint
// and its own error path reports the cancellation. Nothing to stop before boot finishes.
// Setting a flag must not queue behind the thing it is meant to stop.
//
// This used to be `runPythonAsync('webtorch.cancel()')`, which is scheduled like any other
// Python job -- so the cancel waited for the load it was cancelling to yield, and on a slow
// network that is many seconds. Pressing Stop did nothing visible, then everything at once.
//
// Calling the function directly is not the same thing: while the load is suspended on a
// fetch, control is in the JS event loop and the interpreter is free, so this sets the flag
// now and the load sees it at its next checkpoint.
function stopLoad() {
  if (!ready || !pyodide) return;
  try {
    const wt = pyodide.pyimport('webtorch');
    try { wt.cancel(); } finally { if (wt.destroy) wt.destroy(); }
  } catch (e) {
    // Falls back to the queued form rather than not cancelling at all.
    try { pyodide.runPythonAsync('import webtorch; webtorch.cancel()'); } catch (e2) {}
  }
}

onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    let res = null;
    if (cmd === 'boot') await boot();
    else if (cmd === 'load') await loadModel(args.repo, args.file, args.lmax);
    else if (cmd === 'stopLoad') stopLoad();
    // Stop a reply mid-flight. The same flag the loader uses, read (not raised on) between
    // tokens, so `generate` returns the part of the answer that already exists.
    else if (cmd === 'stopGen') {
      await pyodide.runPythonAsync('import webtorch; webtorch.cancel()');
      res = true;
    }
    else if (cmd === 'generate') res = await generate(args.prompt, args);
  else if (cmd === 'toolsSupported') res = await toolsSupported();
    else if (cmd === 'py') { await boot(); res = await pyodide.runPythonAsync(args.code); }
    else if (cmd === 'cacheList') res = await cacheList();
    else if (cmd === 'cacheDelete') await cacheDelete(args.key);
    else if (cmd === 'cacheClear') await cacheClear();
    else if (cmd === 'exportModel') res = await exportModel(args.keys, args.handle);
    else if (cmd === 'importModel') res = await importModel(args.handle, args.name);
    else if (cmd === 'migrate') res = await migrateToDirectory(args.dir);
    else if (cmd === 'armStorage') await armStorageWatch();
    else if (cmd === 'release') await releaseModel();
    send({ type: 'result', id, res });
  } catch (err) {
    const msg = (err && err.message) || String(err);
    send({ type: 'result', id, error: msg });
    // A stop the person asked for is a normal ending, not an error: the page's own handler
    // already says "load stopped", so do not overwrite it with an error line. Any other
    // failure shows the LAST line of a Python traceback (the actual error), not the whole
    // stack — the full trace still rides on the result above for the page to surface.
    if (!/load cancelled/.test(msg)) {
      const last = msg.split('\n').filter(l => l.trim()).pop() || msg;
      send({ type: 'status', text: 'error: ' + last });
    }
  }
};
