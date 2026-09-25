/* webtorch — main-thread bootstrap.
 *
 * The GPU backend spans two JavaScript contexts: the main thread owns the device, the
 * worker reaches it over shared memory. Both halves must be initialised, in order, before
 * Pyodide starts in the worker — and when they are not, nothing fails. Every tensor op
 * quietly runs on numpy inside wasm instead, which is correct and about two orders of
 * magnitude slower. Wiring that by hand is the kind of thing every caller gets wrong once,
 * so it lives here.
 *
 * Load after dist/wgpy-main.js, then:
 *     const worker = new Worker('worker.js');
 *     const { backend, tasks } = await webtorch.initMain(worker);
 *     // hold every message to the worker until this resolves
 *
 * `tasks` is the page's side of stopping work and of reading what the runtime holds. Both
 * travel through shared memory rather than messages, because a worker that is running
 * Python does not reach `onmessage` at all until it finishes -- so a stop sent as a message
 * arrives after the thing it was stopping, and a request for the figures was measured
 * outstanding for 51 seconds. None of that is the caller's to arrange; this listens for the
 * worker half's private messages and hands back two calls.
 *
 * A host with its own `worker.onmessage` should ignore messages carrying `__webtorch`:
 *     worker.onmessage = (e) => { if (e.data && e.data.__webtorch) return; ... }
 */
(function (root) {
  const wt = root.webtorch || (root.webtorch = {});

  /** First backend in `order` this browser can actually provide. */
  function pick(order) {
    for (const b of order) {
      if (b === 'webgpu' && root.navigator && root.navigator.gpu) return b;
      if (b === 'webgl') {
        try {
          const c = document.createElement('canvas');
          if (c.getContext('webgl2') || c.getContext('webgl')) return b;
        } catch (e) { /* no document, or blocked */ }
      }
    }
    return null;
  }

  /**
   * Initialise the main-thread half and tell the worker which backend to set up.
   * Resolves to {backend}: 'webgpu' | 'webgl' | 'cpu'. Never rejects unless
   * `requireGpu` is set — a CPU fallback is reported, not thrown, so a caller can
   * decide whether it is acceptable.
   */
  wt.initMain = async function (worker, opts) {
    opts = opts || {};
    const order = opts.backendOrder || ['webgpu', 'webgl'];
    let backend = null;
    if (typeof root.wgpy === 'undefined') {
      if (opts.requireGpu) throw new Error('load dist/wgpy-main.js before webtorch-main.js');
      console.warn('webtorch: wgpy-main.js not loaded, running on CPU');
    } else if ((backend = pick(order))) {
      try {
        await root.wgpy.initMain(worker, { backendOrder: [backend] });
      } catch (e) {
        if (opts.requireGpu) throw e;
        console.warn('webtorch: GPU backend init failed, running on CPU:', e);
        backend = null;
      }
    } else if (opts.requireGpu) {
      throw new Error('no GPU backend available (tried: ' + order.join(', ') + ')');
    }
    backend = backend || 'cpu';
    // The worker cannot detect this for itself: it has no device and no Python yet.
    worker.postMessage({ __webtorch: 'backend', backend: backend });
    return { backend: backend, tasks: tasksFor(worker) };
  };

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

  // ---- everything that goes wrong, in one place -------------------------------------------
  //
  // Not console.warn. A host cannot read a console, and a failure only the console knows
  // about is a failure the person using the page finds out about by noticing that something
  // is slow or missing. Everything the SDK catches ends up here instead: a call that threw,
  // the runtime dying, the backend falling back to the CPU, the service worker's handler
  // refusing to load. Each carries a `scope` so a host can decide what deserves saying out
  // loud and what only deserves a log.
  //
  // Module level, and registered by its own call, because errors arrive before any of this
  // is set up -- `installServiceWorker` runs before `start`, and both can fail.
  const sinks = [];
  /** Hear about everything that goes wrong, from any part of the SDK. */
  wt.onError = function (fn) { if (fn) sinks.push(fn); return wt; };
  function report(scope, message, extra) {
    const e = Object.assign({ scope: scope, message: String(message) }, extra || {});
    if (!sinks.length) console.warn('webtorch [' + scope + ']: ' + e.message);
    for (const fn of sinks) { try { fn(e); } catch (err) { /* a sink that throws is its own */ } }
  }
  wt._report = report;                 // for the other half of this package, not for hosts

  // ---- the service worker the SDK needs ---------------------------------------------------
  //
  // Call this as early on the page as you can, before anything that wants the GPU:
  //
  //     webtorch.installServiceWorker({ handler: 'cache-sw.js' });
  //
  // It registers the worker that shipped with this package and, on a first visit, reloads
  // once so the document is served through it. Everything about WHY -- which headers, the
  // claim/controller race, the reload that must happen at most once -- is in here. A host
  // that sets the isolation headers on its own server does not need to call it at all.
  //
  // `handler` is optional and is the host's own service-worker-environment code: caching,
  // offline, whatever a host would have written its own worker for. A client has exactly one
  // controller, so there is only one worker to have; this is how a host gets into it. See
  // `webtorch.handleFetch` in webtorch-sw.js for what that code looks like and for what it
  // is and is not allowed to do.
  //
  // `onMessage` is the page's end of the one channel that handler may talk over. Both ends
  // are declared here, with the worker -- the handler's end when its file loads, the page's
  // end in this call -- and the raw `message` events are sealed off inside the worker so
  // there is no second way.
  //
  // Resolves to what happened, rather than throwing: 'isolated' (nothing needed),
  // 'reloading' (a reload is on its way), 'registered' (worker is in place, isolation lands
  // next load), or a reason it cannot work -- 'not-secure', 'no-service-worker', 'not-served',
  // 'still-not-isolated', or 'failed: …'.
  const RELOAD_ONCE = 'webtorch.sw.reloaded';

  /**
   * Send to the host's half inside the service worker. It arrives at whatever that half
   * passed to `webtorch.onPageMessage`.
   *
   * The pair exists because a service worker's `message` port is sealed to the host's
   * handler: a worker that outlives the page and is shared by every tab, and that is the
   * only thing keeping the document isolated, is not a place for a second protocol nobody
   * can see. One typed channel, both ends declared up front.
   */
  wt.sendToHandler = async function (data) {
    if (!navigator.serviceWorker) throw new Error('webtorch: no service worker here');
    const reg = await navigator.serviceWorker.ready;
    const target = navigator.serviceWorker.controller || reg.active;
    if (!target) throw new Error('webtorch: no active service worker to send to');
    target.postMessage({ __wtsw: 1, data: data });
  };

  wt.installServiceWorker = async function (opts) {
    opts = opts || {};
    wt.onError(opts.onError);
    // The worker's own problems -- a handler that would not load, a handler that threw --
    // reach the page over its own envelope, separate from the one the host's two halves use.
    if (navigator.serviceWorker && !wt._swListening) {
      wt._swListening = true;
      navigator.serviceWorker.addEventListener('message', function (e) {
        const d = e.data;
        if (d && d.__wtsw === 2) report('service-worker', d.message, { detail: d.detail });
      });
    }
    // The page's end of that channel, declared here and not later, so both halves of a
    // host's service-worker code are fixed at the moment the worker is created.
    if (opts.onMessage && navigator.serviceWorker) {
      navigator.serviceWorker.addEventListener('message', function (e) {
        const d = e.data;
        if (d && d.__wtsw === 1) opts.onMessage(d.data);
      });
    }
    const base = new URL(opts.baseURL || '../', location.href).href;
    const url = base + 'webtorch-sw.js'
              + (opts.handler ? '?handler=' + encodeURIComponent(
                  new URL(opts.handler, location.href).pathname) : '');

    // Opened as a file rather than served. There is no response to add headers to and no
    // service worker to add them with, so SharedArrayBuffer cannot exist -- and the module
    // fetches would be blocked by the origin anyway. Said plainly, because the symptom
    // otherwise is a silent fall back to the CPU.
    if (location.protocol === 'file:' || location.protocol === 'data:') {
      report('service-worker', 'opened from ' + location.protocol + ' rather than served, so '
             + 'the page cannot be cross-origin isolated and the model will run on the CPU');
      return 'not-served';
    }
    // Almost always this is the page not being a SECURE CONTEXT rather than the browser
    // lacking the feature, and saying the wrong one sends whoever reads it looking in the
    // wrong place. A service worker needs https, or localhost / 127.0.0.1, which browsers
    // treat as trustworthy; a LAN address over plain http is not, and neither is file://.
    // Cross-origin isolation needs the same thing, so on such an origin SharedArrayBuffer
    // is gone even if the server does send the headers itself -- there is no workaround
    // here, only a different URL.
    if (!self.isSecureContext) {
      report('service-worker', 'this page is not a secure context (' + location.origin
             + '), so there is no service worker and no SharedArrayBuffer, and the model '
             + 'will run on the CPU. Serve it over https, or from localhost.');
      return 'not-secure';
    }
    if (!navigator.serviceWorker) {
      report('service-worker', 'this browser has no service worker, so a static host cannot '
             + 'be cross-origin isolated and the model will run on the CPU');
      return 'no-service-worker';
    }

    if (self.crossOriginIsolated) {
      // Already isolated, by this worker on an earlier load or by the server's own headers.
      // Nothing is urgent, and no reload is needed -- but register anyway, because the host's
      // handler wants to be in place for the next load.
      try { sessionStorage.removeItem(RELOAD_ONCE); } catch (e) { /* private mode */ }
      try { await navigator.serviceWorker.register(url); } catch (e) { /* not urgent */ }
      return 'isolated';
    }

    // Read BEFORE registering. The moment the worker activates, `clients.claim()` sets
    // `navigator.serviceWorker.controller` -- so reading it afterwards says only that a
    // worker EXISTS, not that it served THIS document. That race decides everything: this
    // load's document response already went out without the headers, and no amount of
    // claiming changes it. If claim wins and the check reads the controller, the reload is
    // skipped and the first visit stays on the CPU until somebody refreshes by hand. (Seen
    // for real: a phone reporting a service worker and `crossOriginIsolated: false` at once.)
    const hadController = !!navigator.serviceWorker.controller;
    let reg;
    try { reg = await navigator.serviceWorker.register(url); }
    catch (err) {
      report('service-worker', 'could not register: ' + String((err && err.message) || err));
      return 'failed: ' + String((err && err.message) || err);
    }
    if (hadController) return 'registered';   // this load was already served by a worker

    // A reload helps at most once. If it did not produce isolation, reloading again never
    // will, and without this guard a broken setup reloads for ever.
    let reloaded = false;
    try { reloaded = !!sessionStorage.getItem(RELOAD_ONCE); } catch (e) { /* private mode */ }
    if (self.crossOriginIsolated) return 'isolated';        // meanwhile, nothing left to do
    if (reloaded) {
      report('service-worker', 'the worker is active but the page is still not isolated '
             + 'after a reload, so the model will run on the CPU');
      return 'still-not-isolated';
    }
    try { sessionStorage.setItem(RELOAD_ONCE, '1'); } catch (e) { /* private mode */ }
    if (reg.active) { location.reload(); return 'reloading'; }
    const sw = reg.installing || reg.waiting;
    if (sw) sw.addEventListener('statechange', function (e) {
      if (e.target.state === 'activated') location.reload();
      // Redundant without activating means the install threw -- a syntax error in the
      // worker, or an importScripts that 404ed. Nothing reloads, and without this nothing
      // says why.
      if (e.target.state === 'redundant') {
        report('service-worker', 'the worker failed to install, so the page will not be '
               + 'cross-origin isolated and the model will run on the CPU');
      }
    });
    return 'reloading';
  };

  // ---- the whole SDK, as functions ------------------------------------------------------
  //
  // `initMain`/`initWorker` above are the low-level pair: they hand back a Pyodide and leave
  // the crossing to the caller. `start` is the one to reach for. It creates the worker from
  // a script this package ships, so a host writes no worker at all, and returns an object
  // whose methods are the SDK -- arguments marshalled, Python run, results parsed, and the
  // SDK's own progress hooks delivered as ordinary callbacks.
  //
  // Options, all optional: `baseURL`, `backendOrder`/`requireGpu` (as `initMain`),
  // `pyodideIndexURL`, `version` (passed through to the files this package serves, for a
  // host whose caching is a URL that changes with the bytes), `onStatus`, `onLog`,
  // `onModel` and `onError` -- which must be given here rather than through `on()` if a host
  // wants to see the boot, since all of it happens before this returns -- and
  // `rememberTuning`, which
  // lets the SDK keep what it measured about this GPU so the next load does not measure it
  // again. That last one is off by default: leaving something behind in a browser's storage
  // is the host's call, not the SDK's, and so is caching a model (install a writer with
  // `run("webtorch.set_io_write(webtorch.default_io_write)")` if that is what you want).
  //
  //     const wt = await webtorch.start({ baseURL: '../' });
  //     const m  = await wt.load('org/repo/file.gguf', { onProgress: p => … });
  //     const r  = await wt.generate('hello', { onToken: t => … });
  //
  // Anything this does not wrap is still reachable: `wt.run(code, vars)` executes Python in
  // the same runtime with values from the page bound as globals, which is where policy that
  // is genuinely the host's -- where models are fetched from, for one -- belongs.
  let nextId = 1;

  wt.start = async function (opts) {
    opts = opts || {};
    // Made absolute against the PAGE before it is handed over. `baseURL` is written the way
    // a caller thinks of it -- from the page -- but the worker script resolves its own
    // `importScripts` against where IT sits, which is inside this package. '../' then means
    // two different directories on the two sides, and the worker silently failed to load
    // the backend from a path one level too deep.
    const base = new URL(opts.baseURL || '../', location.href).href;
    // Passed through to the worker script and to the package files, and used for nothing
    // else. Caching is the host's: this is only how a host that versions its URLs says so.
    const v = opts.version ? ('&v=' + encodeURIComponent(opts.version)) : '';
    const worker = new Worker(base + 'webtorch/js/webtorch-host.js?base='
                              + encodeURIComponent(base) + v);
    const pending = new Map();          // call id -> {resolve, reject, on}
    const listeners = {};               // name -> [fn], for events not tied to a call

    // Registered HERE, before the boot call, because the most useful status a host can show
    // is the boot's own -- connecting to the GPU, starting Python, installing the backend --
    // and all of it happens before this function returns. Left to `api.on` afterwards, those
    // messages are emitted into an empty list and lost, and a host has no way to catch them
    // because the object carrying `on` does not exist yet. (It shipped that way once: the
    // page sat on "no model loaded" through the whole boot and only came alive at the first
    // load.)
    function listen(name, fn) { if (fn) (listeners[name] || (listeners[name] = [])).push(fn); }
    listen('status', opts.onStatus);
    listen('log', opts.onLog);
    // A model arriving or going away: {state:'loaded'|'released', id, kind, surface}.
    listen('model', opts.onModel);
    // Errors do not go through `listeners` -- they go to the one sink, so that what the
    // service worker reports and what the runtime reports arrive at the same place.
    wt.onError(opts.onError);

    // The runtime dying, which until now was silent in the worst possible way: every call in
    // flight simply never settled, so a page that had asked for anything waited for ever
    // with nothing to show and no way to find out. A 27B that will not fit is exactly this
    // -- the worker goes down on an allocation and takes the conversation's turn with it.
    worker.addEventListener('error', function (e) {
      const message = 'the runtime stopped: ' + ((e && e.message) || 'worker error');
      report('runtime', message);
      for (const [, p] of pending) { try { p.reject(new Error(message)); } catch (err) {} }
      pending.clear();
    });
    worker.addEventListener('messageerror', function () {
      report('runtime', 'a message could not be delivered to the runtime');
    });

    worker.addEventListener('message', function (e) {
      const d = e.data;
      if (!d || (d.__wt !== 'reply' && d.__wt !== 'event')) return;
      const p = pending.get(d.id);
      if (d.__wt === 'event') {
        // Errors never belong to a call: one sink, whatever they were doing at the time.
        if (d.name === 'error') {
          report((d.data && d.data.scope) || 'runtime', (d.data && d.data.message) || d.data);
          return;
        }
        // A call's own callback first; otherwise whoever is listening for that name. An
        // event with id 0 belongs to no call -- status and log during boot.
        const own = p && p.on && p.on[d.name];
        if (own) { own(d.data); return; }
        for (const fn of listeners[d.name] || []) fn(d.data);
        return;
      }
      pending.delete(d.id);
      if (!p) return;
      d.ok ? p.resolve(d.value) : p.reject(new Error(d.value));
    });

    const { backend, tasks } = await wt.initMain(worker, opts);

    function call(method, args, on) {
      const id = nextId++;
      const p = new Promise(function (resolve, reject) {
        pending.set(id, { resolve: resolve, reject: reject, on: on || null });
      });
      worker.postMessage({ __wt: 'call', id: id, method: method, args: args || {} });
      return p;
    }
    // Callbacks are passed inline with the other options, because that is where a caller
    // thinks of them; they are pulled out here rather than crossing as messages, which they
    // cannot do.
    function split(o, names) {
      const on = {}, rest = {};
      for (const k of Object.keys(o || {})) {
        const ev = names[k];
        if (ev) { on[ev] = o[k]; } else { rest[k] = o[k]; }
      }
      return [rest, on];
    }

    const api = {
      /** 'webgpu' | 'webgl' | 'cpu' -- what ops will really run on. */
      backend: backend,
      /** Why it is not the GPU, recorded where it failed. Null when it is. */
      reason: null,

      /** Stop whatever is running, now, while the worker is busy. See `tasks.cancel`. */
      cancel: function () { tasks.cancel(); },
      /** What the runtime holds right now, straight out of shared memory. */
      resources: function () { return tasks.resources(); },
      /** Was this the stop, rather than something going wrong? */
      isCancelled: wt.isCancelled,

      /** Listen for events that belong to no particular call: 'status', 'log'. */
      on: function (name, fn) {
        (listeners[name] || (listeners[name] = [])).push(fn);
        return api;
      },

      /** Python in the same runtime, with `vars` bound as globals. The escape hatch. */
      run: function (code, vars) { return call('run', { code: code, vars: vars || {} }); },

      /**
       * The page-side end of the SAME global byte callbacks used by every Python loader.
       * This is deliberately only a bridge: source selection, URL mapping and persistence
       * all remain properties of whichever callbacks the application installed.
       */
      io: {
        start: function () { return call('ioStart'); },
        read: function (name, offset, length) {
          return call('ioRead', { name: name, offset: offset || 0,
                                  length: length == null ? null : length });
        },
        write: function (name, data, offset) {
          return call('ioWrite', { name: name, data: data, offset: offset || 0 });
        },
      },

      /**
       * Load a model. `source` is whatever the installed reader takes; `file` names one
       * inside a repo. Returns {id, kind, surface} -- `surface` is the model's own account
       * of what it takes and returns, so a caller builds itself from that rather than from
       * a table of its own.
       */
      load: function (source, o) {
        const [rest, on] = split(o, { onProgress: 'progress', onStage: 'stage',
                                      onStatus: 'status' });
        return call('load', { source: source, file: rest.file,
                              maxContext: rest.maxContext }, on);
      },
      release: function () { return call('release'); },
      /** Stop a load. Separate from `cancel` only because a suspended load can be told directly. */
      stopLoading: function () { return call('stopLoad'); },

      /**
       * Generate. Every option the SDK takes is forwarded by name -- temperature, top_p,
       * stop, tools, and the rest -- and one left out keeps the model's own default.
       * `onToken({channel, text, n, at})` arrives as the reply is written.
       */
      generate: function (prompt, o) {
        const [rest, on] = split(o, { onToken: 'token' });
        const images = rest.images; delete rest.images;
        return call('generate', { prompt: prompt, options: rest, images: images }, on);
      },
      /** Score structured questions against a structured state, for a model that decides. */
      decide: function (state, questions) { return call('decide', { state: state, questions: questions }); },
      /** Fit decision probabilities on separate labelled held-out examples. */
      calibrate: function (examples, o) {
        o = o || {};
        return call('calibrate', { examples: examples, byOptions: o.byOptions,
                                   minSamples: o.minSamples });
      },

      /** What the model can be asked about its own output. */
      tools: {
        supported: function () { return call('toolsSupported'); },
        calls: function (text, list) { return call('toolCalls', { text: text, tools: list }); },
        result: function (c, content) { return call('toolResult', { call: c, content: content }); },
        suggest: function (name, args, list) { return call('toolSuggest', { name: name, args: args, tools: list }); },
        round: function (text, calls, results) { return call('toolRound', { text: text, calls: calls, results: results }); },
        render: function (name, args, list) { return call('toolRender', { name: name, args: args, tools: list }); },
      },
      splitReasoning: function (text) { return call('splitReasoning', { text: text }); },

      /** Model files this browser is keeping. */
      cache: {
        list: function () { return call('cacheList'); },
        delete: function (key) { return call('cacheDelete', { key: key }); },
        clear: function () { return call('cacheClear'); },
        /** Into a file the caller picked; the handle can only come from a page. */
        export: function (keys, handle, o) {
          const [, on] = split(o, { onProgress: 'exporting' });
          return call('cacheExport', { keys: keys, handle: handle }, on);
        },
        import: function (handle, name) { return call('cacheImport', { handle: handle, name: name }); },
        migrate: function (directory, o) {
          const [, on] = split(o, { onProgress: 'migrating' });
          return call('cacheMigrate', { directory: directory }, on);
        },
        /** Be told when the browser refuses to keep any more. */
        watch: function (fn) { return call('watchStorage', {}, { storageFull: fn }); },
      },

      /** The runtime's figures when it is idle; `resources()` is the same while it is busy. */
      stats: function () { return call('stats'); },
    };

    // Boot now, so `backend` and `reason` are answers rather than promises by the time this
    // returns -- a caller that has to ask twice will forget once.
    const started = await call('start', { pyodideIndexURL: opts.pyodideIndexURL,
                                          rememberTuning: !!opts.rememberTuning });
    api.backend = started.backend;
    api.reason = started.reason || null;
    return api;
  };

  // How long the cooperative stop gets before the interpreter is interrupted.
  //
  // Long enough for it to win where it can, short enough not to be a wait: a generation
  // notices the flag between tokens and has been measured at 4-19ms, this is an order of
  // magnitude above that, and everything past it was not going to notice at all.
  const GRACE_MS = 120;

  /** The page's half. One per worker; `initMain` returns it. */
  function tasksFor(worker) {
    let stop = null, intr = null, stat = null, escalate = null, busy = false;
    worker.addEventListener('message', function (e) {
      const d = e.data;
      if (!d) return;
      if (d.__webtorch === 'channels') {
        const c = d.channels || {};
        if (c.stop) stop = new Int32Array(c.stop);
        if (c.intr) intr = new Uint8Array(c.intr);
        if (c.stat) stat = new Float64Array(c.stat);
        return;
      }
      if (d.__webtorch === 'busy') { busy = true; return; }
      // The worker has nothing running. Disarm: a cancelled command comes back well inside
      // the grace period, and a timer left armed fires into whatever is running by then.
      if (d.__webtorch === 'idle') {
        busy = false;
        clearTimeout(escalate); escalate = null;
        if (intr) { try { intr[0] = 0; } catch (e) { /* detached */ } }
      }
    });
    return {
      /**
       * Stop whatever the worker is doing. Two stops, in order of politeness: the
       * cooperative one first, which lets the work end at its own next checkpoint and keep
       * what it has already produced, and the interpreter interrupt a moment later for work
       * that is not looking -- a load whose bytes are cached was measured going 19 seconds
       * between checkpoints. The second only ever fires when the first did not land, and
       * the worker clears it as soon as a command ends.
       *
       * Takes effect while the worker is busy, which is the whole point: it is a store into
       * shared memory, not a message waiting behind the work.
       */
      cancel: function () {
        if (stop) Atomics.store(stop, 0, 1);
        // Also as a message, which is the only path when there is no shared memory. It
        // arrives when the worker is next free; where the store above worked, it is already
        // over by then and this is a no-op.
        worker.postMessage({ __webtorch: 'cancel' });
        // Only at work that exists. An interrupt has no target when the worker is idle,
        // so arming then just leaves it loaded for whatever runs next.
        if (!intr || !busy) return;
        clearTimeout(escalate);
        escalate = setTimeout(function () {
          try { intr[0] = 2; } catch (e) { /* detached */ }
        }, GRACE_MS);
      },
      /**
       * What the runtime is holding right now: {gpuBytes, gpuPeak, gpuBuffers, wasmBytes,
       * at}, or null before the worker has reported any. `at` is when the worker last wrote
       * them -- compare it against the last one you displayed, because the writer runs from
       * the matmul and stops when a reply does.
       *
       * Only figures that are actually held. A page cannot read the device's GPU
       * utilisation, the process's CPU, or anything about paging: there is no Web API for
       * any of them, so nothing is reported for them.
       */
      resources: function () {
        if (!stat || !stat[0]) return null;
        return { gpuBytes: stat[0], gpuPeak: stat[1], gpuBuffers: stat[2],
                 wasmBytes: stat[3] || null, at: stat[4] };
      },
    };
  }
})(self);
