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
let STOP = null;
try { STOP = new Int32Array(new SharedArrayBuffer(4)); } catch (e) { STOP = null; }

// The runtime's own figures, in shared memory rather than in messages.
//
// The worker is single-threaded, so while a reply is being written it cannot answer a
// request for them -- one was measured outstanding for 51 seconds, with the panel showing
// numbers from before the reply began. Messages would work, but they tie how fresh the
// display is to how often this side decides to send; a shared array leaves that to the
// reader, which is where it belongs. Float64 because bytes held pass what an f32 counts
// exactly, and the values are independent scalars, so a torn read costs one stale frame and
// nothing more -- no atomics needed for that.
//   [0] bytes held   [1] peak   [2] buffer count   [3] wasm heap
let STAT = null;
try { STAT = new Float64Array(new SharedArrayBuffer(32)); } catch (e) { STAT = null; }

// The escalation, for work that will not stop because it is not looking.
//
// The flag below is cooperative: it ends a decode loop between tokens and a load at its IO
// checkpoints, and it is the right stop where it lands, because the work gets to finish
// tidily -- a generation keeps the tokens it has. But between checkpoints nothing is
// looking, and a load whose bytes are already cached goes a long way between them.
//
// This is what stops that. Pyodide checks this buffer from the interpreter's own eval loop,
// so writing SIGINT here raises KeyboardInterrupt inside whatever Python is running,
// without that code having to poll anything. The page raises it only after the polite flag
// has had its moment.
let INTR = null;
try { INTR = new Uint8Array(new SharedArrayBuffer(1)); } catch (e) { INTR = null; }
function intrClear() { if (INTR) INTR[0] = 0; }

// A stop that already worked leaves a loaded gun behind. The page escalates by writing
// SIGINT into this buffer 120ms after the polite flag, and the polite flag usually wins --
// so the byte is often still set when there is no longer anything to interrupt. Pyodide
// raises at its NEXT checkpoint whatever that happens to be, and the next thing to run is
// the event loop's own scheduling, where nothing is awaiting anything: it surfaces as an
// uncaught PythonError whose traceback is entirely webloop.py, describing the machinery
// rather than anything the person did.
//
// Cleared below the moment a command ends, which closes the common case, and absorbed here
// for the window that clear cannot cover -- the page's timer can fire a microsecond after
// it. A KeyboardInterrupt with no command running IS the stop that already succeeded, and
// reporting it as a failure would be reporting the thing working.
function isStrayInterrupt(reason) {
  if (cmdDepth > 0) return false;
  const m = String((reason && (reason.message || reason.toString())) || '');
  return /KeyboardInterrupt/.test(m);
}
let cmdDepth = 0;
self.addEventListener('unhandledrejection', (e) => {
  if (!isStrayInterrupt(e.reason)) return;
  intrClear();
  e.preventDefault();
});

// Ending the WAIT, which is not the same as ending the WORK.
//
// The flag above ends the work, but only where the work looks at it: a decode loop between
// tokens, a load at its IO checkpoints. Between two checkpoints nothing looks, and a load
// whose bytes are already cached does not reach one for a long time -- measured at 19
// seconds. Racing the work against a promise that a stop rejects gives the answer back
// immediately: the page is free to act on the stop while the abandoned work winds itself
// down at its own next checkpoint.
//
// What that costs is having to ignore the abandoned run. It is still running, it will still
// stream tokens and may still finish a load, and none of that may be reported as the
// current one -- hence the epoch: only the newest run may speak.
let runEpoch = 0;
const stopWaiters = new Set();
function newRun() {
  const mine = ++runEpoch;
  let rejector;
  const stopped = new Promise((_, rej) => { rejector = rej; stopWaiters.add(rej); });
  stopped.catch(() => {});                 // raced, so this promise's rejection is expected
  return {
    epoch: mine,
    current: () => runEpoch === mine,
    race: (work) => Promise.race([
      work.finally(() => stopWaiters.delete(rejector)),
      stopped,
    ]),
  };
}
function stopWaitersFire() {
  for (const rej of stopWaiters) { try { rej(new Error('stopped')); } catch (e) {} }
  stopWaiters.clear();
}
function stopFlagRaise() { if (STOP) Atomics.store(STOP, 0, 1); stopWaitersFire(); }
function stopFlagClear() { if (STOP) Atomics.store(STOP, 0, 0); }
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
  // Point the SDK at the shared flag, so a stop is seen at the next checkpoint rather than
  // at the next message. Reading it is one index into shared memory, which is what lets this
  // sit in a per-token loop.
  if (STOP) {
    pyodide.globals.set('_STOPFLAG', STOP);
    await pyodide.runPythonAsync(`
import webtorch
webtorch.set_cancel_probe(lambda: _STOPFLAG[0] != 0)
`);
    send({ type: 'stopbuf', buf: STOP.buffer });
  }
  if (STAT) send({ type: 'statbuf', buf: STAT.buffer });
  if (INTR) {
    try { pyodide.setInterruptBuffer(INTR); send({ type: 'intrbuf', buf: INTR.buffer }); }
    catch (e) { INTR = null; }           // an older Pyodide: the polite flag is all there is
  }
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
  stopFlagClear(); intrClear();          // the Python flag is cleared below; these are its twins
  const run = newRun();
  const out = await run.race(pyodide.runPythonAsync(`
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
getattr(_MODEL["m"], "kind", "")
`));
  // Abandoned by a stop while it was still running: it may well have gone on to finish, but
  // the person asked for it to end and something newer may already have started.
  if (!run.current()) throw new Error('load cancelled');
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

// Which shape of tool definition does THIS model's template actually consume?
//
// There is no universal one. The nested {type:"function", function:{...}} form is what most
// templates written against the OpenAI convention expect; others take the flat
// {name, description, parameters}. Handing over the wrong one is not an error -- a Jinja
// template quietly renders nothing for a field it does not know, so the model is told about
// a tool it never sees, and then never calls it. Silent, and indistinguishable from a model
// that simply chose not to.
//
// So the shapes are TRIED, not assumed: render the same chat with each and keep the one whose
// rendering actually contains the tool's name AND its parameter's name. That is evidence the
// template read the definition rather than skipped it. If none does, the model cannot be told
// about tools in any form this knows how to write, and it is not offered any.
async function toolsSupported() {
  if (!ready || !pyodide) return { ok: false, shape: null };
  try {
    const r = await pyodide.runPythonAsync(`
import json
_shape = None
_m = _MODEL["m"]
_t = getattr(getattr(_m, "impl", _m), "tok", None) if _m is not None else None
if _t is not None:
    _msgs = [{"role": "user", "content": "hi"}]
    _NAME = "zzprobetoolzz"
    _ARG = "zzprobeargzz"
    _fn = {"name": _NAME, "description": "probe",
           "parameters": {"type": "object",
                          "properties": {_ARG: {"type": "string", "description": "probe"}},
                          "required": [_ARG]}}
    _cands = [("nested", [{"type": "function", "function": _fn}]),
              ("flat", [dict(_fn)])]
    try:
        _plain = _t.encode_chat(None, None, messages=_msgs)
    except Exception:
        _plain = None
    if _plain is not None:
        for _label, _tools in _cands:
            try:
                _ids = _t.encode_chat(None, None, messages=_msgs, tools=_tools)
            except Exception:
                continue
            if _ids == _plain:
                continue                      # the template ignored them entirely
            try:
                _txt = _t.decode(_ids)
            except Exception:
                _txt = ""
            # The NAME alone is not enough: a template can mention a tool and drop its
            # arguments, which produces calls with no parameters.
            if _NAME in _txt and _ARG in _txt:
                _shape = _label
                break
# How a tool RESULT reaches this model, discovered the same way. A template may define its
# own structure for one, may render it as an ordinary turn, or may drop a role it does not
# know -- and a dropped result is silent: the model answers without ever seeing what the
# tool returned.
_result_via = None
_keeps_name = False
if _t is not None:
    _MK = "zzresultmarkzz"; _TN = "zztoolnamezz"
    _base = [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "calling"}]
    def _renders(_msgs):
        try:
            return _t.decode(_t.encode_chat(None, None, messages=_msgs))
        except Exception:
            return ""
    _as_tool = _renders(_base + [{"role": "tool", "name": _TN, "content": _MK}])
    if _MK in _as_tool:
        _result_via = "tool"
        _keeps_name = _TN in _as_tool
    else:
        _as_user = _renders(_base + [{"role": "user", "content": _MK}])
        if _MK in _as_user:
            _result_via = "user"
# How a RESULT is tied to the CALL it answers, discovered the same way. Some templates
# carry an id both ways -- one rendered on the assistant's call, one read back on the
# result -- and then a reply with several calls needs no other hint about which answer
# belongs to which. Two probes, because the two halves are independent facts: does the
# template RENDER an id given on the call (and in which of the shapes it takes there),
# and does it READ an id field on the result message (and which name does it read).
# A template that does neither ties results to calls by order alone.
_call_id_shape = None
_result_id_field = None
if _t is not None and _result_via == "tool":
    _CID = "zzcallidzz"
    _amsg = {"role": "assistant", "content": "calling"}
    for _s, _calls in (("nested", [{"id": _CID, "type": "function",
                                    "function": {"name": _TN, "arguments": {}}}]),
                       ("flat", [{"id": _CID, "name": _TN, "arguments": {}}])):
        _c = dict(_amsg); _c["tool_calls"] = _calls
        if _CID in _renders(_base[:-1] + [_c]):
            _call_id_shape = _s
            break
    if _call_id_shape is not None:
        _ids = (("tool_call_id", "zzidtczz"), ("call_id", "zzidcizz"), ("id", "zzididzz"))
        _rm = {"role": "tool", "content": _MK}
        for _f, _v in _ids:
            _rm[_f] = _v
        _r = _renders(_base + [_rm])
        for _f, _v in _ids:
            if _v in _r:
                _result_id_field = _f
                break
# How this model WRITES a call, taken from the template rather than assumed. The template
# is what turns "tool_calls" into text, so rendering one with known markers and reading back
# what surrounds it gives the exact delimiters this model emits -- no guessing that everyone
# uses <tool_call>, which is one family's convention and not a standard.
_call_open = None
_call_close = None
_call_payload = None
if _t is not None and _shape is not None:
    _N = "zzprobenamezz"; _A = "zzprobeargzz"; _V = "zzprobevalzz"
    _rc = _renders([{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"id": "zzprobeidzz", "type": "function",
                                     "function": {"name": _N, "arguments": {_A: _V}}}]}])
    _i = _rc.find(_N)
    if _i >= 0:
        # The payload is the object that CONTAINS the name. Walking back to the nearest
        # brace and matching it forward is not that: on a template whose tool definitions
        # are rendered as JSON earlier in the prompt, the nearest brace before the call is
        # inside those definitions, and it closes before the call even begins. That parsed,
        # so the probe reported the definition's shape as the call's and handed back a
        # fragment of the tools block as the delimiters -- after which nothing the model
        # wrote could ever match, and its calls were neither run nor removed.
        #
        # So: walk candidate braces outward from the name and keep the WIDEST one that both
        # contains the name and parses as JSON on its own. Widest, not nearest, because the
        # nested form wraps the call -- {"type":"function","function":{"name":...}} -- and
        # stopping at the first container reports the inner object as the payload and the
        # wrapper text as the delimiter. Parsing is what stops it going too far: a span that
        # reached back into the tools block would have prose in it and would not parse.
        _st = -1; _en = -1
        _cand = _rc.rfind("{", 0, _i)
        while _cand >= 0:
            _d = 0; _e = -1
            for _q in range(_cand, len(_rc)):
                if _rc[_q] == "{": _d += 1
                elif _rc[_q] == "}":
                    _d -= 1
                    if _d == 0: _e = _q + 1; break
            if _e > _i:
                try:
                    json.loads(_rc[_cand:_e])
                    _st = _cand; _en = _e
                except Exception:
                    pass
            _cand = _rc.rfind("{", 0, _cand)
        if _en > 0:
            _pre = _rc[:_st]
            _post = _rc[_en:]
            # chr(10), never a backslash-n literal: this whole block lives inside a JS
            # template literal, so JS turns the escape into a real newline before Python ever
            # sees it and the string is left unterminated. It compiled fine when read from
            # the FILE and failed only when run -- which is why testing the file's text is
            # not testing what runs.
            _NL = chr(10)
            # The LAST NON-EMPTY line before the payload is the opener. A template that puts
            # its tag on its own line leaves an empty remainder after the final newline, and
            # taking that reports "this model uses no delimiters" about one that does.
            _lines = [x for x in _pre.split(_NL) if x.strip()]
            _call_open = _lines[-1] if _lines else ""
            _after = [x for x in _post.split(_NL) if x.strip()]
            _call_close = _after[0] if _after else ""
            try:
                _obj = json.loads(_rc[_st:_en])
                _call_payload = "nested" if ("function" in _obj and "name" not in _obj) else "flat"
            except Exception:
                _call_payload = None
    # No JSON payload found: several templates write the call as XML instead --
    #   <tool_call>{nl}<function=NAME>{nl}<parameter=KEY>{nl}value{nl}</parameter>{nl}</function>
    # -- and looking only for a brace reported "this model has no call format" about a model
    # whose format is spelled out in its own system prompt. The name IS present in that form,
    # so this cannot hang off "name not found"; it hangs off "no payload was read".
    if _call_payload is None:
        _fx = _rc.find("<function=")
        if _fx >= 0:
            _NL = chr(10)
            _lines = [x for x in _rc[:_fx].split(_NL) if x.strip()]
            _call_open = _lines[-1] if _lines else ""
            _fe = _rc.find("</function>", _fx)
            _post = _rc[_fe + len("</function>"):] if _fe >= 0 else ""
            _after = [x for x in _post.split(_NL) if x.strip()]
            _call_close = _after[0] if _after else ""
            _call_payload = "xml"
json.dumps({"ok": _shape is not None, "shape": _shape,
            "result_via": _result_via, "result_keeps_name": bool(_keeps_name),
            "call_id_shape": _call_id_shape, "result_id_field": _result_id_field,
            "call_open": _call_open, "call_close": _call_close,
            "call_payload": _call_payload})
`);
    return JSON.parse(r);
  } catch (e) {
    // The reason travels with the answer. A probe that fails silently is indistinguishable
    // from a model that takes no tools, and that ambiguity cost an hour.
    return { ok: false, shape: null, result_via: null, result_keeps_name: false,
             call_id_shape: null, result_id_field: null,
             call_open: null, call_close: null, call_payload: null,
             error: String((e && e.message) || e).slice(-400) };
  }
}

async function generate(prompt, opts) {
  // A stop asked for during the LAST reply must not end this one before it starts.
  stopFlagClear(); intrClear();
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
  const run = newRun();
  // Gated on the run: a generation abandoned by a stop keeps decoding until its own next
  // checkpoint, and those tokens belong to a reply the page has already closed.
  self.__chunk = (ch, t) => { if (run.current())
                                send({ type: 'chunk', channel: ch, text: t,
                                       at: performance.now() }); };
  try {
    const out = await run.race(pyodide.runPythonAsync(`
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
  stopFlagRaise();                       // lands at the next checkpoint, message or not
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
// Three stores. Called from the matmul, so it runs many times a layer, and costs little
// enough there that it does not need to be rationed.
self.__gpustat = (held, peak, n) => {
  if (!STAT) return;
  STAT[0] = held; STAT[1] = peak; STAT[2] = n;
  try {
    const m = pyodide && pyodide._module && pyodide._module.HEAP8;
    if (m) STAT[3] = m.byteLength;
  } catch (e) { /* leave the last value */ }
};

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
  const { id, cmd, args } = e.data;
  cmdDepth++;
  try {
    let res = null;
    if (cmd === 'boot') await boot();
    else if (cmd === 'load') await loadModel(args.repo, args.file, args.lmax);
    else if (cmd === 'stopLoad') stopLoad();
    // Stop a reply mid-flight. The same flag the loader uses, read (not raised on) between
    // tokens, so `generate` returns the part of the answer that already exists.
    //
    // The shared flag is the stop. This message cannot be what does it -- while a generation
    // runs, this handler does not get to run at all -- so by the time we are here the page
    // has already stored into shared memory and the decode loop has already seen it. The
    // Python call below is only for a page with no SharedArrayBuffer, where the message is
    // all there is.
    else if (cmd === 'stopGen') {
      stopFlagRaise();
      if (!STOP) await pyodide.runPythonAsync('import webtorch; webtorch.cancel()');
      res = true;
    }
    else if (cmd === 'generate') res = await generate(args.prompt, args);
  else if (cmd === 'toolsSupported') res = await toolsSupported();
    else if (cmd === 'py') { await boot(); res = await pyodide.runPythonAsync(args.code); }
    else if (cmd === 'stats') res = runtimeStats();
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
    cmdDepth--;
    if (cmdDepth === 0) intrClear();
  }
};
