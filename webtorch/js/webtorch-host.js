/* webtorch — the worker, so callers do not write one.
 *
 * The SDK is Python, it runs in a worker because it must (Pyodide on the main thread stops
 * the page), and everything a caller wants to do with it therefore crosses a message
 * boundary. Every host that ever used this SDK wrote the same several hundred lines to
 * cross it: marshal the arguments, `runPythonAsync` a template, parse the JSON back, and
 * invent a message shape for every callback the SDK offers. None of that is about any
 * particular app, and getting it wrong is silent -- an option spelled differently is simply
 * not passed.
 *
 * So the crossing lives here, and the page gets functions instead. This file is not loaded
 * by hand: `webtorch.start()` on the main thread creates a Worker from it and returns the
 * other side. See `webtorch-main.js`.
 *
 * The one thing a host still supplies is its own Python, through `run(code, vars)` -- which
 * is how policy that is genuinely the host's, such as where models are fetched from, stays
 * the host's.
 */
(function (root) {
  const Q = new URLSearchParams(root.location.search);
  const BASE = Q.get('base') || '../../';
  // The host's own cache-busting token, if it has one, so the package files are fetched the
  // same way the host serves everything else. The SDK has no policy of its own here.
  const VERSION = Q.get('v') || null;
  importScripts(BASE + 'dist/wgpy-worker.js');
  importScripts(BASE + 'webtorch/js/webtorch-worker.js');

  let pyodide = null, tasks = null, ready = false;

  // ---- the wire ------------------------------------------------------------------------
  //
  // `__wt` rather than `__webtorch`: the bootstrap's two halves already talk over that one,
  // and these are different messages with a different shape. A host never sees either.
  function reply(id, ok, value) { root.postMessage({ __wt: 'reply', id: id, ok: ok, value: value }); }
  function emit(id, name, data) { root.postMessage({ __wt: 'event', id: id, name: name, data: data }); }

  // The call being served, so the SDK's own progress hooks -- which are installed once and
  // know nothing about calls -- can address their events at whoever asked.
  let current = 0;

  // ---- what this device worked out about itself -----------------------------------------
  //
  // Measuring the fastest shader shape for a model costs real time on every load, and the
  // answer does not change between loads on the same GPU. Keeping it is worth a lot; the
  // SDK still does not decide to.
  //
  // Writing to a browser's storage is the HOST's call -- it knows its quota, its privacy
  // promises and whether this page should leave anything behind at all -- so this is off
  // unless `start({rememberTuning: true})` asked for it. The shape of what gets stored
  // stays in here, because that is the SDK's and a host has no business knowing it.
  let remember = false;
  const KP_DB = 'webtorch-kernel-profile';
  function kpOpen() {
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
    } catch (e) { return null; }      // no adapter, nothing to key on, nothing to keep
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

  // ---- images --------------------------------------------------------------------------
  //
  // The SDK takes pixels, not URLs: `media=` wants height x width x 3 bytes. Turning one
  // into the other is the same loop in every host, so it is done here and a caller passes
  // whatever a browser can fetch.
  async function pixels(urls) {
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
        out.push({ w: bmp.width, h: bmp.height, rgb: rgb });
        bmp.close();
      } catch (e) {
        // The person attached this and it is not going to the model. That is not a log line.
        report('image', 'could not read an attached image, so it was not sent: ' + e.message);
      }
    }
    return out;
  }

  /** Everything this file catches, to the page. Nothing here is allowed to just vanish. */
  function report(scope, message) {
    emit(0, 'error', { scope: scope, message: String((message && message.message) || message) });
  }

  async function py(code) { return await pyodide.runPythonAsync(code); }
  async function pyJSON(code) { return JSON.parse(await py(code)); }

  // ---- what a caller can ask for -------------------------------------------------------
  const METHODS = {

    async start(a) {
      if (ready) return { backend: await py('import webtorch; webtorch.backend()') };
      const r = await root.webtorch.initWorker({
        baseURL: BASE,
        version: VERSION,
        pyodideIndexURL: a && a.pyodideIndexURL,
        onError: (e) => emit(0, 'error', e),
        stdout: (t) => emit(0, 'log', t),
        stderr: (t) => emit(0, 'log', t),
        onStatus: (t) => emit(0, 'status', t),
      });
      pyodide = r.pyodide; tasks = r.tasks;
      remember = !!(a && a.rememberTuning);
      await py('import json, webtorch\n_MODEL = {"m": None, "id": None}\n');
      ready = true;
      // When it is not the GPU, the SDK says why. Recorded where it failed, which is the
      // only way to tell a browser without WebGPU from a page that is not isolated.
      let why = null;
      if (r.backend !== 'webgpu') {
        try { why = await py('import webtorch; webtorch.backend_reason()'); }
        catch (e) { why = null; }
      }
      return { backend: r.backend, reason: why };
    },

    /** Any Python, with values from the page bound as globals. The host's own escape hatch. */
    async run(a) {
      await METHODS.start({});
      for (const k of Object.keys((a && a.vars) || {})) pyodide.globals.set(k, a.vars[k]);
      const out = await py(a.code);
      return (out && out.toJs) ? out.toJs() : out;
    },

    async load(a) {
      await METHODS.start({});
      const src = a.file ? (a.source + '/' + a.file) : a.source;
      emit(current, 'status', 'loading ' + src + ' …');
      pyodide.globals.set('_src', src);
      pyodide.globals.set('_lmax', a.maxContext || 0);
      // Two different things from two different owners: how far the LOAD has got comes
      // from the SDK, and the download rate comes from the transport the host installed --
      // the SDK has no transport and cannot know one.
      const who = current;
      let dlRate = 0, lastAt = 0;
      root.__dl = (rate) => { dlRate = rate; };
      root.__prog = (done, total, rate) => {
        const now = Date.now();
        if (now - lastAt < 250 && done !== total) return;      // a meter, not a firehose
        lastAt = now;
        emit(who, 'progress', { bytes: done, total: total || 0, rate: rate, dlRate: dlRate });
      };
      // Where the load IS, as opposed to how many bytes it has read. Reading stops well
      // before the load does -- what follows measures kernel shapes, checks the weights and
      // runs one forward -- and on a 13GB model that tail is minutes with the byte meter
      // frozen at the end, which reads as a hang.
      root.__stage = (stage, done, total, after, elapsed) =>
        emit(who, 'stage', { stage: stage, done: done || 0, total: total || 0,
                             after: after || null, elapsed: elapsed || 0 });
      const kpk = remember ? await kpKey() : null;
      root.__kp = kpk ? await kpGet(kpk) : null;
      const task = tasks.begin();
      const out = await task.until(py(`
import js, webtorch
src = _src
lmax = int(_lmax)
webtorch.cancel(False)      # a stale stop request must not hit this load
webtorch.set_read_progress(lambda i: js.self.__prog(i["done"], i["total"] or 0, i["rate"]))
webtorch.set_download_progress(lambda i: js.self.__dl(i["rate"]))
webtorch.set_load_progress(
    lambda i: js.self.__stage(i["stage"], i.get("done"), i.get("total"),
                              i.get("after"), i.get("elapsed")))
# What this device worked out last time.
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
    # Both are the same event seen from two places: the cooperative flag reached a
    # checkpoint, or the interpreter was interrupted because none was reached in time.
    # Either way the load is over and leaves whole chunks plus the one it landed in the
    # middle of; drop the ragged edge here, once it has actually unwound, so what stays is
    # usable and resumes.
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
      // Abandoned by a stop while it was still running: it may well have gone on to finish,
      // but the caller asked for it to end and something newer may already have started.
      if (!task.current()) throw new Error('load cancelled');
      // Keep what this load worked out, so the next one does not work it out again.
      if (kpk) {
        try {
          await kpPut(kpk, JSON.parse(await py(
            'import json, webtorch\njson.dumps(webtorch.kernel_profile())')));
        } catch (e) {
          report('tuning', 'could not keep what this load measured, so the next load will '
                 + 'measure it again: ' + ((e && e.message) || e));
        }
      }
      emit(who, 'status', 'ready: ' + src);
      const info = JSON.parse(out);
      // Said to everyone, not just to whoever awaited this call: a model arriving or going
      // away changes what a whole interface may offer, and the part that has to react is
      // rarely the part that asked.
      emit(0, 'model', { state: 'loaded', id: src, kind: info.kind, surface: info.surface });
      // What the model says it takes and returns, so a caller builds itself from this
      // rather than keeping its own table of model kind -> interface.
      return { id: src, kind: info.kind, surface: info.surface };
    },

    async release() {
      if (!ready) return null;
      await py(`
import webtorch
if _MODEL["m"] is not None:
    webtorch.release(_MODEL["m"]); _MODEL["m"] = None; _MODEL["id"] = None
`);
      emit(current, 'status', 'model released');
      emit(0, 'model', { state: 'released' });
      return null;
    },

    async generate(a) {
      if (!ready) throw new Error('no runtime');
      // A stop asked for during the LAST reply must not end this one before it starts.
      await py('import webtorch; webtorch.cancel(False)');
      const opts = a.options || {};
      const imgs = await pixels(a.images);
      pyodide.globals.set('_prompt', a.prompt || '');
      pyodide.globals.set('_imgs', imgs.map((i) => ({ w: i.w, h: i.h, rgb: i.rgb })));
      pyodide.globals.set('_opts', JSON.stringify(opts));
      const who = current;
      const task = tasks.begin();
      // Stamped on THIS clock, at the moment the piece leaves the decode loop. A caller
      // timing these on its own puts the cost of rendering the reply, and the delivery
      // latency of the message, inside the rate it reports for the model. Only differences
      // between stamps are ever used, so a separate time origin does not matter.
      //
      // Gated on the task: a generation abandoned by a stop keeps decoding until its own
      // next checkpoint, and those pieces belong to a reply the caller has already closed.
      //
      // `n` is the SDK's own token count at the moment the piece was produced, not a count
      // of pieces: a token that completes no character, and one whose text is held back to
      // decide its channel, both yield nothing. Counting arrivals undercounts the reply and,
      // with it, the live rate -- badly on CJK text, where the two diverge most.
      root.__chunk = (ch, t, n) => {
        if (task.current()) {
          emit(who, 'token', { channel: ch, text: t,
                               n: (n == null ? null : Number(n)), at: performance.now() });
        }
      };
      try {
        return await pyJSONTask(task, `
import json, js
m = _MODEL["m"]
if m is None:
    raise RuntimeError("no model loaded")
_o = json.loads(_opts)
_n = int(_o.get("max_new") or 0)          # 0 = no budget: run until the model stops itself
_think = bool(_o.get("enable_thinking"))
_msgs = _o.get("messages") or None        # full conversation; falls back to the one prompt
# Every generation option the SDK takes is forwarded by name, so adding a control needs no
# change here -- and an option the caller does not set stays at the model's own default
# rather than being overridden with a guess.
# "tools" rides here too: the SDK hands it to the model's own chat template, so which models
# can be told about tools is a question about their template, not about this list.
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
# single token event -- the same shape either way.
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
    # The model itself says whether it can see, through the same "kind" a caller uses to
    # offer the controls -- not through whichever attribute an implementation happens to
    # keep its encoder in.
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
`);
      } finally {
        if (task.current()) root.__chunk = null;
        // The boundary where the memory actually comes back. A collect can only free what
        // nothing refers to, and while a reply is being written the frames on the stack
        // still refer to most of it -- the same collect frees far more here, with the call
        // graph unwound, than it does from inside the allocation path. Measured: 20.78GB
        // held falls to 11.43GB, and to 9.2GB once the pool is trimmed with it.
        try {
          await py('import wgpy_backends.webgpu.webgpu_buffer as _b\n'
                 + 'if hasattr(_b, "reap_now"): _b.reap_now()');
        } catch (e) {
          // This is where a reply's memory actually comes back. Failing here is the
          // difference between a session that stays usable and one that climbs until the
          // tab dies, so it does not get to be silent.
          report('memory', 'could not release what this reply held: ' + ((e && e.message) || e));
        }
      }
    },

    async decide(a) {
      if (!ready) throw new Error('no runtime');
      root.__decide_in = JSON.stringify({ state: a.state, questions: a.questions });
      return await pyJSON(`
import js, json
_req = json.loads(js.self.__decide_in)
_m = _MODEL["m"]
if _m is None:
    raise RuntimeError("load a model first")
if not hasattr(_m, "decide"):
    raise RuntimeError("this model answers by writing text, not by scoring questions")
json.dumps(_m.decide(_req["state"], _req["questions"]))
`);
    },

    // ---- what the model can be asked about its own output ------------------------------
    async toolsSupported() {
      if (!ready) return { ok: false };
      try {
        return await pyJSON(`
import json
_m = _MODEL["m"]
json.dumps({"ok": bool(_m is not None and _m.tools_supported())})`);
      } catch (e) {
        // The reason travels with the answer. A probe that fails silently is
        // indistinguishable from a model that takes no tools.
        return { ok: false, error: String((e && e.message) || e).slice(-400) };
      }
    },

    async toolCalls(a) {
      if (!ready) return { shown: String(a.text || ''), calls: [] };
      pyodide.globals.set('_ts_text', String(a.text || ''));
      pyodide.globals.set('_ts_tools', JSON.stringify(a.tools || []));
      return await pyJSON(`
import json
_m = _MODEL["m"]
_tl = json.loads(_ts_tools)
if _m is None:
    _r = {"shown": _ts_text, "calls": []}
else:
    _r = {"shown": _m.strip_tool_calls(_ts_text, _tl),
          "calls": _m.tool_calls(_ts_text, _tl)}
json.dumps(_r, ensure_ascii=False)`);
    },

    async toolResult(a) {
      pyodide.globals.set('_tr', JSON.stringify({ call: a.call, content: a.content }));
      return await pyJSON(`
import json
_d = json.loads(_tr); _m = _MODEL["m"]
json.dumps(_m.tool_result_message(_d["call"], _d["content"]), ensure_ascii=False)`);
    },

    async toolSuggest(a) {
      pyodide.globals.set('_tg', JSON.stringify({ name: a.name, args: a.args, tools: a.tools || [] }));
      return await pyJSON(`
import json
_d = json.loads(_tg); _m = _MODEL["m"]
json.dumps(_m.suggest_tool(_d["name"], _d["args"], _d["tools"]), ensure_ascii=False)`);
    },

    async toolRound(a) {
      pyodide.globals.set('_tro', JSON.stringify(
        { text: a.text, calls: a.calls || [], results: a.results || [] }));
      return await pyJSON(`
import json
_d = json.loads(_tro); _m = _MODEL["m"]
json.dumps(_m.tool_round_messages(_d["text"], _d["calls"], _d["results"]), ensure_ascii=False)`);
    },

    async toolRender(a) {
      pyodide.globals.set('_trd', JSON.stringify({ name: a.name, args: a.args, tools: a.tools || [] }));
      return await pyJSON(`
import json
_d = json.loads(_trd); _m = _MODEL["m"]
json.dumps(_m.render_tool_call(_d["name"], _d["args"], _d["tools"]), ensure_ascii=False)`);
    },

    async splitReasoning(a) {
      if (!ready) return { reasoning: null, answer: String(a.text || ''), open: false };
      pyodide.globals.set('_sr_text', String(a.text || ''));
      return await pyJSON(`
import json
_m = _MODEL["m"]
json.dumps(_m.split_reasoning(_sr_text) if _m is not None
           else {"reasoning": None, "answer": _sr_text, "open": False}, ensure_ascii=False)`);
    },

    // ---- the cache ---------------------------------------------------------------------
    async cacheList() {
      await METHODS.start({});
      return await pyJSON(`
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
    },

    async cacheDelete(a) {
      await METHODS.start({});
      pyodide.globals.set('_k', a.key);
      await py('import webtorch; await webtorch.delete_cache(_k)');
      return null;
    },

    async cacheClear() {
      await METHODS.start({});
      await py('import webtorch; await webtorch.clear_cache()');
      return null;
    },

    /**
     * Write cached files into a file the caller picked. The handle comes from the page,
     * because only a page can ask for one; everything after it is the SDK's own format.
     */
    async cacheExport(a) {
      await METHODS.start({});
      const who = current;
      let total = 0;
      try {
        pyodide.globals.set('_ekeys', a.keys);
        total = Number(await py(`
import webtorch
_items = await webtorch.list_cache()
_want = set(_ekeys)
sum(int(i.get("size") or 0) for i in _items if i.get("key") in _want)
`)) || 0;
      } catch (e) { total = 0; }
      let w;
      try {
        w = await a.handle.createWritable();
      } catch (e) {
        // Say which failure this was. "Failed to execute 'createWritable'" on its own sends
        // whoever reads it looking in the wrong place -- the usual cause is the write
        // permission on the picked file, which only the page can obtain.
        throw new Error('cannot write to the chosen file (' + (e && e.name || 'error') + ': '
          + String((e && e.message) || e).slice(0, 160) + '). If the browser asked for '
          + 'permission and it was dismissed, choose the file again and allow it.');
      }
      let done = 0;
      root.__sink = async (bytes) => {
        const u8 = bytes.toJs ? bytes.toJs() : bytes;
        await w.write(u8);
        done += u8.length;
        // The file itself stays at ZERO until `close()` -- the browser writes a temporary
        // and swaps it in at the end -- so for the 81 seconds a 400MB export takes there is
        // otherwise nothing at all to look at.
        emit(who, 'exporting', { bytes: done, total: total });
      };
      try {
        pyodide.globals.set('_keys', a.keys);
        await py(`
import js, webtorch
from pyodide.ffi import to_js
async def _w(b):
    await js.self.__sink(to_js(b))
await webtorch.export_model(list(_keys), _w)
`);
      } finally {
        await w.close();
        root.__sink = null;
      }
      return done;
    },

    async cacheImport(a) {
      await METHODS.start({});
      pyodide.globals.set('_h', a.handle);
      pyodide.globals.set('_n', a.name || '');
      return await pyJSON('import json, webtorch\njson.dumps(await webtorch.import_model(_h, _n or None))');
    },

    async cacheMigrate(a) {
      await METHODS.start({});
      const who = current;
      pyodide.globals.set('_dir', a.directory);
      root.__mig = (n, key) => emit(who, 'migrating', { bytes: n, key: key });
      return await py(`
import js, webtorch
await webtorch.migrate_cache(_dir, on_progress=lambda n, k: js.self.__mig(n, k))
`);
    },

    /** Ask to be told when the browser refuses to keep any more. */
    async watchStorage() {
      await METHODS.start({});
      const who = current;
      root.__full = (key) => emit(who, 'storageFull', { key: key });
      await py('import js, webtorch\nwebtorch.set_storage_full(lambda i: js.self.__full(i["key"]))');
      return null;
    },

    /**
     * What the runtime can say about itself when it is idle. `resources()` on the page is
     * the same figures while it is busy; this one also answers before any matmul has run.
     */
    async stats() {
      const out = { gpuBytes: null, gpuPeak: null, gpuBuffers: null, wasmBytes: null,
                    loaded: !!ready };
      if (!pyodide) return out;
      try {
        // `pyodide._module.HEAP8`, not `pyodide.HEAP8`: the heap views live on the
        // emscripten module, and the top-level object has never carried them. Reading the
        // wrong one gives undefined rather than an error, so it reported "unknown" forever.
        const m = pyodide._module && pyodide._module.HEAP8;
        if (m && m.byteLength) out.wasmBytes = m.byteLength;
      } catch (e) { /* a runtime that will not say is unknown, not zero */ }
      try {
        const v = pyodide.runPython(
          'from wgpy_backends.webgpu.platform import get_platform as _gp\n'
          + '_gp().gpuBytes() if hasattr(_gp(), "gpuBytes") else (0, 0, 0)');
        const a = v && v.toJs ? v.toJs() : v;
        if (v && v.destroy) v.destroy();
        if (a && a.length === 3) {
          out.gpuBytes = Number(a[0]); out.gpuPeak = Number(a[1]); out.gpuBuffers = Number(a[2]);
        }
      } catch (e) { /* WebGL, or no platform yet: unknown rather than zero */ }
      return out;
    },

    /**
     * Stop a load specifically. The flag alone reaches it, but a load suspended on a fetch
     * leaves the interpreter free -- so calling `cancel` directly sets the SDK's own flag
     * now instead of waiting for the load it is cancelling to yield. Queued, that wait was
     * many seconds on a slow network: Stop did nothing visible, then everything at once.
     */
    async stopLoad() {
      if (tasks) tasks.cancel();
      if (!ready || !pyodide) return null;
      try {
        const wt = pyodide.pyimport('webtorch');
        try { wt.cancel(); } finally { if (wt.destroy) wt.destroy(); }
      } catch (e) {
        try { py('import webtorch; webtorch.cancel()'); }
        catch (e2) { report('cancel', 'the stop could not be delivered: ' + ((e2 && e2.message) || e2)); }
      }
      return null;
    },
  };

  // `until` around a call that returns JSON, kept separate so the template above reads as
  // Python rather than as plumbing.
  async function pyJSONTask(task, code) { return JSON.parse(await task.until(py(code))); }

  root.onmessage = async (e) => {
    const d = e.data;
    // The bootstrap's own half talks over `__webtorch`; those are not calls.
    if (!d || d.__wt !== 'call') return;
    const fn = METHODS[d.method];
    if (tasks) tasks.enter();
    const prev = current;
    current = d.id;
    try {
      if (!fn) throw new Error('webtorch: no such call "' + d.method + '"');
      reply(d.id, true, await fn(d.args || {}));
    } catch (err) {
      const message = String((err && err.message) || err);
      reply(d.id, false, message);
      // And as an event, because a caller that is awaiting this gets the rejection but a
      // host that wants one place to notice everything going wrong has nowhere else to look.
      // A host awaiting the call will see both; that is the caller's to sort out, and it is
      // better than a failure with no general way to hear about it.
      emit(0, 'error', { scope: d.method, message: message });
    } finally {
      current = prev;
      if (tasks) tasks.leave();
    }
  };
})(self);
