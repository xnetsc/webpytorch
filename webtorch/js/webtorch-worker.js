/* webtorch — worker-side bootstrap.
 *
 * Brings up the GPU backend, Pyodide and the webtorch package in the one order that
 * works: the worker half of wgpy must be initialised after the main thread's half and
 * before Pyodide loads. Getting that order wrong does not raise — it silently drops every
 * tensor op onto numpy inside wasm — so the backend that actually came up is probed and
 * returned rather than assumed.
 *
 * Load after dist/wgpy-worker.js, then:
 *     const { pyodide, backend, tasks } = await webtorch.initWorker({ baseURL: '../' });
 *
 * `tasks` is how a page stops work that is already running. See the block above it for why
 * that cannot be a message. The caller wraps what it dispatches and what it cancels; the
 * shared memory, the interrupt buffer, the cancel probe and the runtime's figures are set
 * up here, because every caller needs all of them and the order they go up in is not
 * obvious from the outside.
 *
 * This module talks to its other half over messages carrying `__webtorch`. A host with its
 * own `onmessage` should ignore those:
 *     onmessage = (e) => { if (e.data && e.data.__webtorch) return; ... }
 */
(function (root) {
  const wt = root.webtorch || (root.webtorch = {});

  // Where Pyodide itself comes from. Overridable before initWorker for an offline copy.
  // The SDK's own default, for a host that sets nothing. The chat app overrides it from
  // chat/pyodide-version.js, which is the single place this project pins a release.
  wt.PYODIDE_URL = wt.PYODIDE_URL || 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';

  // The package's own inventory, read from a manifest generated with it, so adding a
  // module never needs an edit here. The inline list is the fallback for a tree served
  // without the manifest, and is only ever a floor.
  const FALLBACK = ["__init__.py", "_core.py", "_sdk.py", "audiofe.py", "backend.py", "cosyvoice.py", "detection.py", "ggufload.py", "hfcompat.py", "iqtables.py", "linear_attn.py", "llm.py", "lm_engine.py", "multimodal.py", "onnxrt.py", "portable.py", "quantize.py", "torchshim.py", "tts.py", "vl.py", "webenv.py", "webio.py"];

  async function moduleList(base, version) {
    try {
      const r = await fetch(base + 'webtorch/modules.json'
                            + (version ? '?v=' + encodeURIComponent(version) : ''));
      if (r.ok) {
        const m = (await r.json()).modules;
        if (Array.isArray(m) && m.length) return m;
      }
    } catch (e) { /* fall through */ }
    return FALLBACK;
  }

  // Set by the main thread before it sends anything else (see webtorch-main.js).
  let announced = null;
  const announcedBackend = new Promise((resolve) => { announced = resolve; });
  root.addEventListener('message', function (e) {
    if (e.data && e.data.__webtorch === 'backend') announced(e.data.backend);
  });

  // How the package's own files are fetched: ordinarily, so whatever the host serves them
  // with decides whether they are cached.
  //
  // This used to append `v=Date.now()` and `cache: 'no-store'` to every one of them, which
  // began as a development convenience -- an edited module that the browser answers from
  // cache is indistinguishable from a bug, and much harder to find -- and shipped as a
  // policy: 27 files and 1.3MB re-fetched on every page load of every host, forever, with
  // no way to turn it off. Caching and cache-busting belong to whoever serves the files.
  // A host that wants a URL which changes when the bytes do passes `version` (this project
  // has `scripts/stamp.sh`, which is exactly that); a host that sets cache headers instead
  // passes nothing.
  function text(u, version) {
    const url = version ? (u + (u.includes('?') ? '&' : '?') + 'v=' + encodeURIComponent(version)) : u;
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(u + ': ' + r.status);
      return r.text();
    });
  }

  /**
   * Was this the SDK's own cancellation, rather than something going wrong?
   *
   * A stop is not an error, and a host that cannot tell them apart reports one as the
   * other -- which is how a stopped reply used to end with an "Error:" line under the
   * half-written answer the stop had just preserved. The rejection crosses between the two
   * contexts as a message string, so the test is on the text; hosts ask this instead of
   * matching it themselves.
   */
  wt.isCancelled = function (e) {
    return /webtorch: cancelled by request/.test(String((e && e.message) || e || ''));
  };

  // ---- stopping work, and reading what the runtime holds ------------------------------
  //
  // Both of these are shared memory rather than messages, for the same reason: while Python
  // is running, this worker's thread is inside that one call and `onmessage` does not run at
  // all. A stop sent as a message arrives after the thing it was stopping has finished, and
  // a request for the runtime's figures was measured outstanding for 51 seconds while the
  // display showed numbers from before the reply began. A shared array is read by the other
  // side whenever it likes, with nothing queued behind the work.
  //
  // All of it degrades. Without cross-origin isolation there is no SharedArrayBuffer, the
  // allocations fail, and a cancel falls back to the message path -- which works, and only
  // arrives late.
  function shared(Ctor, bytes) {
    try { return new Ctor(new SharedArrayBuffer(bytes)); } catch (e) { return null; }
  }
  // The cooperative flag. Read, not raised on, at the SDK's own checkpoints -- between
  // tokens, between reads -- so work that ends on it keeps what it has already produced.
  const STOP = shared(Int32Array, 4);
  // The escalation, for work that will not stop because it is not looking. Pyodide checks
  // this from the interpreter's eval loop, so a SIGINT here raises KeyboardInterrupt inside
  // whatever is running without that code polling anything. The other half writes it only
  // after the cooperative flag has had its moment.
  const INTR = shared(Uint8Array, 1);
  // [0] bytes held  [1] peak  [2] buffer count  [3] wasm heap  [4] when it was written.
  //
  // Float64 because bytes held pass what an f32 counts exactly, and these are independent
  // scalars -- a torn read costs one stale frame and nothing more, so no atomics. [4] exists
  // because the writer is not always running: these are written from the matmul, so when a
  // reply ends they stop, and the last value written is that reply's peak. It would stand as
  // if it were current while several gigabytes are handed back, so the reader compares this
  // stamp against its own and takes whichever is newer.
  const STAT = shared(Float64Array, 40);

  // Only what is actually held. A page cannot read the device's GPU utilisation, the
  // process's CPU, or anything about paging -- there is no Web API for any of it -- so
  // nothing here pretends to. Called from the matmul, many times a layer, so it is three
  // stores and a property read.
  root.__gpustat = function (held, peak, n) {
    if (!STAT) return;
    STAT[0] = held; STAT[1] = peak; STAT[2] = n; STAT[4] = Date.now();
    try {
      const m = root.pyodide && root.pyodide._module && root.pyodide._module.HEAP8;
      if (m) STAT[3] = m.byteLength;
    } catch (e) { /* leave the last value */ }
  };

  // A stop that already worked leaves a loaded gun behind. The other half escalates a short
  // time after the cooperative flag, and the cooperative flag usually wins -- so the byte is
  // often still set when there is no longer anything to interrupt. Pyodide raises at its
  // NEXT checkpoint whatever that happens to be, and the next thing to run is the event
  // loop's own scheduling, where nothing is awaiting anything: it surfaces as an uncaught
  // PythonError whose traceback is entirely webloop.py, describing the machinery rather than
  // anything the person did.
  //
  // `leave` closes the common case; this absorbs the window that clearing cannot cover,
  // because the other half's timer can fire a microsecond after it. A KeyboardInterrupt with
  // no command running IS the stop that already succeeded, and reporting it as a failure
  // would be reporting the thing working.
  //
  // Narrow on purpose. This is a global handler on someone else's context, and swallowing
  // an error the host wanted to see is worse than showing one it did not: so it fires only
  // when the interrupt byte is STILL SET, which means the raise came from the escalation
  // this file wrote and nothing has consumed it. A KeyboardInterrupt from anywhere else --
  // the host's own, a library's -- passes straight through.
  let depth = 0;
  root.addEventListener('unhandledrejection', function (e) {
    if (depth > 0 || !INTR) return;
    let armed = false;
    try { armed = INTR[0] !== 0; } catch (err) { return; }
    if (!armed) return;
    const m = String((e.reason && (e.reason.message || e.reason.toString())) || '');
    if (!/KeyboardInterrupt/.test(m)) return;
    try { INTR[0] = 0; } catch (err) {}
    e.preventDefault();
  });

  // Ending the WAIT, which is not the same as ending the WORK.
  //
  // The flag ends the work, but only where the work looks at it, and a load whose bytes are
  // already cached does not reach a checkpoint for a long time -- measured at 19 seconds.
  // Racing the work against a promise that a cancel rejects gives the answer back at once:
  // the caller is free to act while the abandoned work winds itself down on its own.
  //
  // What that costs is having to ignore the abandoned run, which is what `current` is for:
  // it is still running, it may still finish, and none of that may be reported as the
  // current one.
  let epoch = 0;
  const waiting = new Set();
  function fireCancel() {
    for (const rej of waiting) { try { rej(new Error('webtorch: cancelled by request')); } catch (e) {} }
    waiting.clear();
  }
  const tasks = {
    /** Buffers the page half needs. Sent for it; hosts do not touch these. */
    _channels: function () {
      return { stop: STOP && STOP.buffer, intr: INTR && INTR.buffer,
               stat: STAT && STAT.buffer };
    },
    /**
     * A command is running: while one is, a KeyboardInterrupt belongs to it.
     *
     * The other half is told, because the escalation must only ever be aimed at work that
     * exists. Cancelling while nothing runs used to arm it anyway, and the interrupt then
     * landed in whatever came NEXT -- measured directly: a cancel with an idle worker, then
     * a plain `sum(range(200000))`, and that sum was the thing that died.
     */
    enter: function () {
      if (depth++ === 0) root.postMessage({ __webtorch: 'busy' });
    },
    /**
     * That command is over. Clearing the interrupt byte here is not enough on its own: the
     * other half arms a TIMER when it cancels, and a command that ends inside the grace
     * period leaves that timer to fire into whatever runs next. A cancelled generation
     * returns in about 14ms against a 120ms grace, so the timer landed in the abandoned
     * work every time -- as a KeyboardInterrupt raised inside the tokenizer, reported to
     * the person as a traceback where their half-written answer should have been.
     *
     * So the other half is told, and disarms. The escalation then only ever fires for work
     * that really did not end, which is what it is for.
     */
    leave: function () {
      depth = Math.max(0, depth - 1);
      if (depth > 0) return;
      if (INTR) { try { INTR[0] = 0; } catch (e) {} }
      root.postMessage({ __webtorch: 'idle' });
    },
    /**
     * Start a cancellable operation. Clears any stop left over from the last one -- a stale
     * cancel must not land on work that has only just begun.
     *
     *     const task = tasks.begin();
     *     const out = await task.until(pyodide.runPythonAsync(...));
     *     if (!task.current()) return;        // a newer operation has taken over
     */
    begin: function () {
      if (STOP) Atomics.store(STOP, 0, 0);
      if (INTR) { try { INTR[0] = 0; } catch (e) {} }
      const mine = ++epoch;
      let rejector;
      const cancelled = new Promise(function (_, rej) { rejector = rej; waiting.add(rej); });
      cancelled.catch(function () {});     // raced, so this rejection is expected
      return {
        current: function () { return epoch === mine; },
        cancelled: function () { return !!(STOP && Atomics.load(STOP, 0)); },
        until: function (work) {
          return Promise.race([
            work.finally(function () { waiting.delete(rejector); }),
            cancelled,
          ]);
        },
      };
    },
    /**
     * Cancel from inside this context. The page half calls this over the private channel,
     * so a host only needs it for a cancel of its own making.
     */
    cancel: function () {
      if (STOP) Atomics.store(STOP, 0, 1);
      fireCancel();
      // Where there is no shared memory the message IS the cancel, and by the time it
      // arrives the interpreter is free -- so the SDK's own flag has to be set in Python.
      if (!STOP && root.pyodide) {
        try { root.pyodide.runPythonAsync('import webtorch; webtorch.cancel()'); }
        catch (e) { /* nothing better to try */ }
      }
    },
  };
  wt.tasks = tasks;
  root.addEventListener('message', function (e) {
    if (e.data && e.data.__webtorch === 'cancel') tasks.cancel();
  });

  /**
   * Boot everything. Options (all optional):
   *   baseURL        prefix for dist/ and webtorch/ (default '../')
   *   pyodideIndexURL  where to load Pyodide from (default: the CDN, see PYODIDE_URL)
   *   version        appended to the package's own file URLs, for a host whose cache
   *                  policy is a URL that changes with the bytes. Omitted, they are
   *                  fetched plainly and the host's cache headers decide.
   *   onStatus       (text) => void, progress for the UI
   * Resolves to { pyodide, backend, tasks } where backend is what actually came up:
   * 'webgpu' | 'webgl' | 'cpu'.
   */
  wt.initWorker = async function (opts) {
    opts = opts || {};
    const base = opts.baseURL || '../';
    // A CDN by default: the full Pyodide distribution is ~900 MB, which no static host
    // wants to carry. Point `pyodideIndexURL` at a local copy to run without a network.
    const idx = opts.pyodideIndexURL || wt.PYODIDE_URL;
    const say = opts.onStatus || function () {};

    // Wait for the main thread's choice; it has the device, this context does not.
    const wanted = await announcedBackend;

    if (wanted !== 'cpu' && typeof root.wgpy !== 'undefined') {
      say('connecting to the GPU…');
      try { await root.wgpy.initWorker(); }
      catch (e) { console.warn('webtorch: wgpy.initWorker failed, running on CPU:', e); }
    }

    say('starting Python…');
    // The loader too, if the host has not brought it: it has to come from the same place
    // Pyodide itself does, and a host that gets that pair out of step gets a mismatch it
    // cannot read from the error. A host that loaded it already is left alone.
    if (typeof loadPyodide === 'undefined') importScripts(idx + 'pyodide.js');
    const pyodide = await loadPyodide({ indexURL: idx, stdout: opts.stdout, stderr: opts.stderr });
    root.pyodide = pyodide;
    await pyodide.loadPackage(['micropip', 'numpy']);

    if (wanted !== 'cpu') {
      say('installing the ' + wanted + ' backend…');
      try {
        const mp = pyodide.pyimport('micropip');
        await mp.install(base + 'dist/wgpy_' + wanted + '-1.0.0-py3-none-any.whl');
      } catch (e) { console.warn('webtorch: backend wheel install failed:', e); }
    }

    say('loading webtorch…');
    try { pyodide.FS.mkdir('webtorch'); } catch (e) { /* already there */ }
    // A module that does not arrive is fatal, not a warning to step over.
    //
    // Skipping leaves the directory there with modules missing from it, and Python treats a
    // directory with no `__init__` as a NAMESPACE PACKAGE: `import webtorch` then succeeds
    // and the module has nothing on it. The failure surfaces dozens of steps later as
    // "module 'webtorch' has no attribute 'set_io_read'", which says nothing about the
    // fetch that actually failed -- and if even one module is missing the SDK is not
    // whatever the caller thinks it is anyway.
    const missing = [];
    for (const m of await moduleList(base, opts.version)) {
      try { pyodide.FS.writeFile('webtorch/' + m, await text(base + 'webtorch/' + m, opts.version)); }
      catch (e) { missing.push(m + ' (' + (e && e.message || e) + ')'); }
    }
    if (missing.length) {
      throw new Error('webtorch: ' + missing.length + ' module(s) could not be loaded from '
        + base + 'webtorch/ — ' + missing.slice(0, 5).join(', ')
        + (missing.length > 5 ? ', …' : '')
        + '. The SDK is incomplete, so this stops here rather than importing a package with '
        + 'nothing in it.');
    }
    await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/")');

    // Point the SDK's own cancellation at the shared flag. Reading it is one index into
    // shared memory, which is what lets the check sit in a per-token loop; without it the
    // SDK has no way to be told anything while it is running.
    if (STOP) {
      pyodide.globals.set('__webtorch_stop', STOP);
      await pyodide.runPythonAsync(
        'import webtorch\nwebtorch.set_cancel_probe(lambda: __webtorch_stop[0] != 0)\n');
    }
    if (INTR) {
      // An older Pyodide has no interrupt buffer; the cooperative flag is then all there is.
      try { pyodide.setInterruptBuffer(INTR); } catch (e) { /* cooperative only */ }
    }
    // Hand the page half what it reads and writes. One message, once, before anything the
    // host sends -- a cancel that arrives before this has nothing to store into.
    root.postMessage({ __webtorch: 'channels', channels: tasks._channels() });

    // What is actually live, not what was requested.
    let backend = 'cpu';
    try {
      backend = await pyodide.runPythonAsync('import webtorch; webtorch.backend()');
    } catch (e) { console.warn('webtorch: backend probe failed:', e); }
    say('ready (' + backend + ')');
    return { pyodide: pyodide, backend: backend, tasks: tasks };
  };
})(self);
