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
importScripts('../dist/wgpy-worker.js');
importScripts('../webtorch/js/webtorch-worker.js');

let pyodide = null, ready = false;
const send = (m) => postMessage(m);

// A stop that does not queue.
//
// A generation is a plain Python loop inside ONE runPythonAsync call, so this worker's
// thread is inside that call from the first token to the last. A stop sent as a message is
// not queued behind the work -- `onmessage` never runs at all until the work ends -- and the
// button reads as dead. Same for a load: its checkpoints are between reads, not between
// messages. So the stop travels through memory instead: the page stores a 1 here and the
// SDK sees it at its very next checkpoint, with no message, no queue and no interpreter.
//
// SharedArrayBuffer needs cross-origin isolation, which this page already requires for
// everything else. Where it is missing the message path still works as it did.
// Stopping, and the runtime's figures, both live in the SDK -- see `tasks` in
// webtorch/js/webtorch-worker.js. What used to be here was the shared flag, the interrupt
// buffer, the stats array, the stray-interrupt absorber and the run epoch: about ninety
// lines that every caller of this SDK would have had to write, and get right, before a Stop
// button worked at all. `tasks` is obtained from `initWorker` below.
let tasks = null;

const log = (t) => send({ type: 'log', text: t });


async function boot() {
  if (ready) return;
  // The SDK brings up the backend, Pyodide and the package in the order they require.
  const r = await webtorch.initWorker({
    baseURL: '../', pyodideIndexURL: PYODIDE_URL, stdout: log, stderr: log,
    onStatus: (t) => send({ type: 'status', text: t }),
  });
  pyodide = r.pyodide;
  tasks = r.tasks;
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

// What this DEVICE decided about our kernels: which thread shape is fastest here, and
// whether each one computes the right thing here. Neither depends on the model or the
// conversation, and deriving them costs 36 shader compiles and 36 numerical checks -- the
// whole of the warming phase, measured at 107s on a 27B before it was cut down.
//
// So they are kept. The SDK offers them as data and takes them back (`kernel_profile` /
// `use_kernel_profile`); where they live is this app's decision, which is why the storage is
// here and not there. Keyed by the adapter, because a different GPU is a different answer --
// and stamped by build inside the profile, because a different shader is too.
const KP_DB = 'webtorch-kernel-profile';

async function kpOpen() {
  return new Promise((res, rej) => {
    const rq = indexedDB.open(KP_DB, 1);
    rq.onupgradeneeded = () => rq.result.createObjectStore('p');
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error);
  });
}

async function kpKey() {
  try {
    const a = await navigator.gpu.requestAdapter();
    const i = (a && (a.info || {})) || {};
    return [i.vendor, i.architecture, i.device, i.description].join('/') || 'unknown';
  } catch (e) { return null; }        // no adapter, nothing to key on, nothing to keep
}

async function kpGet(key) {
  try {
    const db = await kpOpen();
    return await new Promise((res) => {
      const rq = db.transaction('p').objectStore('p').get(key);
      rq.onsuccess = () => res(rq.result || null);
      rq.onerror = () => res(null);
    });
  } catch (e) { return null; }
}

async function kpPut(key, val) {
  try {
    const db = await kpOpen();
    await new Promise((res) => {
      const tx = db.transaction('p', 'readwrite');
      tx.objectStore('p').put(val, key);
      tx.oncomplete = tx.onerror = () => res();
    });
  } catch (e) { /* keeping it is an optimisation; failing to is not an error */ }
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
  // Where the load IS, as opposed to how many bytes it has read. Reading stops well before
  // the load does -- what follows measures kernel shapes, checks the weights against the
  // file and runs one forward -- and on a 13GB model that tail is minutes with the byte
  // meter frozen at the end, which reads as a hang.
  self.__stage = (stage, done, total, after, elapsed) => {
    send({ type: 'stage', stage, done: done || 0, total: total || 0,
           after: after || null, elapsed: elapsed || 0 });
  };
  const kpk = await kpKey();
  self.__kp = kpk ? await kpGet(kpk) : null;
  const run = tasks.begin();
  const out = await run.until(pyodide.runPythonAsync(`
import js, webtorch
src = _src
lmax = int(_lmax)
webtorch.cancel(False)      # a stale stop request must not hit this load
webtorch.set_read_progress(lambda i: js.self.__prog(i["done"], i["total"] or 0, i["rate"]))
webtorch.set_download_progress(lambda i: js.self.__dl(i["rate"]))
webtorch.set_load_progress(
    lambda i: js.self.__stage(i["stage"], i.get("done"), i.get("total"),
                              i.get("after"), i.get("elapsed")))
# What this device worked out last time, if this app kept it.
try:
    _kp = js.self.__kp
    if _kp is not None:
        _n = webtorch.use_kernel_profile(_kp.to_py() if hasattr(_kp, "to_py") else _kp)
        js.console.log("kernel profile: reused %d entries" % _n)
except Exception as _e:
    pass
try:
    if _MODEL["m"] is not None:
        webtorch.release(_MODEL["m"]); _MODEL["m"] = None
    m = await webtorch.load(src, **({"lmax": lmax} if lmax else {}))
    _MODEL["m"] = m; _MODEL["id"] = src
except (webtorch.Cancelled, KeyboardInterrupt) as _e:
    # Both are the same event seen from two places: the polite flag reached a checkpoint, or
    # the interpreter was interrupted because none was reached in time. Either way the load
    # is over and leaves whole chunks plus the one it landed in the middle of; drop the
    # ragged edge here, once it has actually unwound, so what stays is usable and resumes.
    _freed = await webtorch.trim_stopped()
    if _freed:
        js.console.log("stopped load: dropped " + str(_freed) + " partial bytes")
    raise webtorch.Cancelled("load cancelled") from None
finally:
    webtorch.set_read_progress(None)
    webtorch.set_download_progress(None)
    webtorch.set_load_progress(None)
import json as _json
_json.dumps({"kind": getattr(_MODEL["m"], "kind", ""),
             "surface": _MODEL["m"].surface()})
`));
  // Abandoned by a stop while it was still running: it may well have gone on to finish, but
  // the person asked for it to end and something newer may already have started.
  if (!run.current()) throw new Error('load cancelled');
  // The page uses the model kind to offer (or hide) image inputs before the user tries one.
  // It comes back as this block's value: a top-level name assigned inside an awaited block
  // is local to the coroutine Pyodide wraps it in, so a second runPython could not see it.
  // Keep what this load worked out, so the next one does not work it out again.
  if (kpk) {
    try {
      const prof = JSON.parse(await pyodide.runPythonAsync(
        'import json, webtorch\njson.dumps(webtorch.kernel_profile())'));
      await kpPut(kpk, prof);
    } catch (e) { /* an optimisation, not a step */ }
  }
  send({ type: 'status', text: `ready: ${src}` });
  // What the model says it takes and returns. The page builds itself from this instead of
  // keeping its own table of model kind -> interface, which was one boolean wide and had no
  // room for a model that is not a chat at all.
  const info = JSON.parse(out);
  send({ type: 'loaded', id: src, kind: info.kind, surface: info.surface,
         image: !!(info.surface && info.surface.takes && info.surface.takes.images) });
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

// Can this model be given tools at all -- the one question the page has. Which definition
// shape its template reads, how it writes a call, how a result reaches it: the SDK settles
// all of that inside `generate(tools=...)` and the tool methods, and none of it comes up
// here. A page that branched on any of it would be making decisions about a chat template.
// Ask a decision model. One pass, no stream: there is nothing to show as it arrives, so the
// answer comes back whole. The questions travel as they were written -- this does not decide
// what to ask, how many options to offer or what any probability means.
async function decide(state, questions) {
  if (!ready || !pyodide) throw new Error('no runtime');
  self.__decide_in = JSON.stringify({ state: state, questions: questions });
  const out = await pyodide.runPythonAsync(`
import js, json, webtorch
_req = json.loads(js.self.__decide_in)
_m = _MODEL["m"]
if _m is None:
    raise RuntimeError("load a model first")
if not hasattr(_m, "decide"):
    raise RuntimeError("this model answers by writing text, not by scoring questions")
json.dumps(_m.decide(_req["state"], _req["questions"]))
`);
  return JSON.parse(out);
}

async function toolsSupported() {
  if (!ready || !pyodide) return { ok: false };
  try {
    const r = await pyodide.runPythonAsync(`
import json
_m = _MODEL["m"]
json.dumps({"ok": bool(_m is not None and _m.tools_supported())})`);
    return JSON.parse(r);
  } catch (e) {
    // The reason travels with the answer. A probe that fails silently is indistinguishable
    // from a model that takes no tools, and that ambiguity cost an hour.
    return { ok: false, error: String((e && e.message) || e).slice(-400) };
  }
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
  const run = tasks.begin();
  // Gated on the run: a generation abandoned by a stop keeps decoding until its own next
  // checkpoint, and those tokens belong to a reply the page has already closed.
  // `n` is the SDK's own token count at the moment this piece was produced, not a count of
  // pieces: a token that completes no character, and one whose text is being held back to
  // decide its channel, both yield nothing. Counting arrivals undercounts the reply and, with
  // it, the live rate -- badly on CJK text, where the two diverge most.
  self.__chunk = (ch, t, n) => { if (run.current())
                                send({ type: 'chunk', channel: ch, text: t,
                                       n: (n == null ? null : Number(n)),
                                       at: performance.now() }); };
  try {
    const out = await run.until(pyodide.runPythonAsync(`
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
         "tools", "constraint", "require_known_tools")
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
    _live = getattr(m, "impl", m)
    for _c in _gen:
        js.self.__chunk(_c["channel"], _c["text"], getattr(_live, "stream_n", None))
_s = getattr(getattr(m, "impl", m), "last_stream", None) or {}
json.dumps({"n": int(_s.get("n") or 0), "truncated": bool(_s.get("truncated")),
            "ttft_s": _s.get("ttft_s"), "tok_s": _s.get("tok_s"),
            "context": _s.get("context"),
            "gpu_ms": _s.get("gpu_ms"), "pick_ms": _s.get("pick_ms"),
            "gpu_ms_head": _s.get("gpu_ms_head"), "gpu_ms_tail": _s.get("gpu_ms_tail"),
            "gpu_ms_curve": _s.get("gpu_ms_curve"),
            "recaptured_at": _s.get("recaptured_at"), "pins": _s.get("pins"),
            "prefilled": _s.get("prefilled"), "prefill_d": _s.get("prefill_d"),
            "path": _s.get("path")})
`));
    return JSON.parse(out);
  } finally {
    if (run.current()) self.__chunk = null;
    // The boundary where the memory actually comes back. A collect can only free what
    // nothing refers to, and while a reply is being written the frames on the stack still
    // refer to most of it -- the same collect frees far more here, with the call graph
    // unwound, than it does from inside the allocation path. Measured: 20.78GB held falls
    // to 11.43GB, and to 9.2GB once the pool is trimmed with it.
    try {
      await pyodide.runPythonAsync(
        'import wgpy_backends.webgpu.webgpu_buffer as _b\n'
        + 'if hasattr(_b, "reap_now"): _b.reap_now()');
    } catch (e) { /* WebGL, or a runtime already gone: nothing to reap */ }
  }
}

// ---- tool calling: the SDK does it, this only carries the question across -------------
//
// Reading a model's calls, shaping a result for it and knowing what its template can carry
// are facts about the model, so they live in the SDK (webtorch/toolcall.py and the
// tokenizer's three probes). Nothing here decides any of it.

// One round trip, not two: the reply the reader sees and the calls to run come from the
// SAME scan, and two scans drift -- a call the loop did not run gets printed as prose.
async function toolScan(text, tools) {
  if (!ready || !pyodide) return { shown: String(text || ''), calls: [] };
  pyodide.globals.set('_ts_text', String(text || ''));
  pyodide.globals.set('_ts_tools', JSON.stringify(tools || []));
  const out = await pyodide.runPythonAsync(`
import json
_m = _MODEL["m"]
_tl = json.loads(_ts_tools)
if _m is None:
    _r = {"shown": _ts_text, "calls": []}
else:
    _r = {"shown": _m.strip_tool_calls(_ts_text, _tl),
          "calls": _m.tool_calls(_ts_text, _tl)}
json.dumps(_r, ensure_ascii=False)`);
  return JSON.parse(out);
}

async function toolRound(text, calls, results) {
  pyodide.globals.set('_tro_text', String(text || ''));
  pyodide.globals.set('_tro_calls', JSON.stringify(calls || []));
  pyodide.globals.set('_tro_res', JSON.stringify(results || []));
  const out = await pyodide.runPythonAsync(`
import json
json.dumps(_MODEL["m"].tool_round_messages(_tro_text, json.loads(_tro_calls),
                                           json.loads(_tro_res)), ensure_ascii=False)`);
  return JSON.parse(out);
}

async function toolSuggest(name, args, tools) {
  pyodide.globals.set('_sg_name', String(name || ''));
  pyodide.globals.set('_sg_args', JSON.stringify(args || {}));
  pyodide.globals.set('_sg_tools', JSON.stringify(tools || []));
  const out = await pyodide.runPythonAsync(`
import json
json.dumps(_MODEL["m"].suggest_tool(_sg_name, json.loads(_sg_tools), json.loads(_sg_args)))`);
  return JSON.parse(out);
}

async function toolResult(callObj, content) {
  pyodide.globals.set('_tr_call', JSON.stringify(callObj || {}));
  pyodide.globals.set('_tr_body', String(content == null ? '' : content));
  const out = await pyodide.runPythonAsync(`
import json
json.dumps(_MODEL["m"].tool_result_message(json.loads(_tr_call), _tr_body),
           ensure_ascii=False)`);
  return JSON.parse(out);
}

async function toolRender(name, args, tools) {
  pyodide.globals.set('_rc_name', String(name || ''));
  pyodide.globals.set('_rc_args', JSON.stringify(args || {}));
  pyodide.globals.set('_rc_tools', JSON.stringify(tools || []));
  return await pyodide.runPythonAsync(`
import json
_MODEL["m"].render_tool_call(_rc_name, json.loads(_rc_args), json.loads(_rc_tools))`);
}

async function splitReasoning(text) {
  if (!ready || !pyodide) return { reasoning: null, answer: String(text || ''), open: false };
  pyodide.globals.set('_sr_text', String(text || ''));
  const out = await pyodide.runPythonAsync(`
import json
_m = _MODEL["m"]
json.dumps(_m.split_reasoning(_sr_text) if _m is not None
           else {"reasoning": None, "answer": _sr_text, "open": False}, ensure_ascii=False)`);
  return JSON.parse(out);
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
  if (tasks) tasks.cancel();             // lands at the next checkpoint, message or not
  if (!ready || !pyodide) return;
  try {
    const wt = pyodide.pyimport('webtorch');
    try { wt.cancel(); } finally { if (wt.destroy) wt.destroy(); }
  } catch (e) {
    // Falls back to the queued form rather than not cancelling at all.
    try { pyodide.runPythonAsync('import webtorch; webtorch.cancel()'); } catch (e2) {}
  }
}

// What the runtime can say about itself, for the resource strip on the page.
//
// Only numbers we actually hold. A page cannot read the device's GPU utilisation, the
// process's CPU, or anything about paging -- those have no Web API -- so nothing here
// pretends to: the GPU figure is the backend's own ledger of buffers it asked for and has
// not returned, and the heap figure is the WASM memory Python is living in. Cheap enough
// to call on a timer: two property reads and a tuple.
function runtimeStats() {
  const out = { gpuBytes: null, gpuPeak: null, gpuBuffers: null, wasmBytes: null,
                loaded: !!ready };
  if (!pyodide) return out;
  try {
    // `pyodide._module.HEAP8`, not `pyodide.HEAP8`: the heap views live on the emscripten
    // module, and the top-level object has never carried them. Reading the wrong one gives
    // undefined rather than an error, so it reported "unknown" forever.
    const m = pyodide._module && pyodide._module.HEAP8;
    if (m && m.byteLength) out.wasmBytes = m.byteLength;
  } catch (e) { /* a runtime that will not say is reported as unknown, not as zero */ }
  try {
    const v = pyodide.runPython(
      'from wgpy_backends.webgpu.platform import get_platform as _gp\n'
      + '_gp().gpuBytes() if hasattr(_gp(), "gpuBytes") else (0, 0, 0)');
    const a = v && v.toJs ? v.toJs() : v;
    if (v && v.destroy) v.destroy();
    if (a && a.length === 3) {
      out.gpuBytes = Number(a[0]); out.gpuPeak = Number(a[1]); out.gpuBuffers = Number(a[2]);
    }
  } catch (e) { /* WebGL backend, or no platform yet: unknown rather than zero */ }
  return out;
}

onmessage = async (e) => {
  // The SDK's two halves talk over this same port. Those messages are not commands and
  // carry no id, so answering them would post a result nobody asked for.
  if (e.data && e.data.__webtorch) return;
  const { id, cmd, args } = e.data;
  // Before `boot` there is nothing to enter: `tasks` arrives with the SDK.
  if (tasks) tasks.enter();
  try {
    let res = null;
    if (cmd === 'boot') await boot();
    else if (cmd === 'load') await loadModel(args.repo, args.file, args.lmax);
    else if (cmd === 'stopLoad') stopLoad();
    else if (cmd === 'generate') res = await generate(args.prompt, args);
    else if (cmd === 'decide') res = await decide(args.state, args.questions);
  else if (cmd === 'toolsSupported') res = await toolsSupported();
    else if (cmd === 'py') { await boot(); res = await pyodide.runPythonAsync(args.code); }
    else if (cmd === 'stats') res = runtimeStats();
    else if (cmd === 'toolScan') res = await toolScan(args.text, args.tools);
    else if (cmd === 'toolResult') res = await toolResult(args.call, args.content);
    else if (cmd === 'toolSuggest') res = await toolSuggest(args.name, args.args, args.tools);
    else if (cmd === 'toolRound') res = await toolRound(args.text, args.calls, args.results);
    else if (cmd === 'toolRender') res = await toolRender(args.name, args.args, args.tools);
    else if (cmd === 'splitReasoning') res = await splitReasoning(args.text);
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
  } finally {
    // Whatever this command was, nothing is running now, so an interrupt still armed here
    // has no work left to land on -- only the interpreter's own plumbing.
    if (tasks) tasks.leave();
  }
};
