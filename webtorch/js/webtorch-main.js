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
  // Resolves to what happened, rather than throwing: 'isolated' (nothing needed),
  // 'reloading' (a reload is on its way), 'registered' (worker is in place, isolation lands
  // next load), or a reason it cannot work -- 'no-service-worker', 'not-served',
  // 'still-not-isolated', or 'failed: …'.
  const RELOAD_ONCE = 'webtorch.sw.reloaded';

  wt.installServiceWorker = async function (opts) {
    opts = opts || {};
    const base = new URL(opts.baseURL || '../', location.href).href;
    const url = base + 'webtorch-sw.js'
              + (opts.handler ? '?handler=' + encodeURIComponent(
                  new URL(opts.handler, location.href).pathname) : '');

    // Opened as a file rather than served. There is no response to add headers to and no
    // service worker to add them with, so SharedArrayBuffer cannot exist -- and the module
    // fetches would be blocked by the origin anyway. Said plainly, because the symptom
    // otherwise is a silent fall back to the CPU.
    if (location.protocol === 'file:' || location.protocol === 'data:') return 'not-served';
    if (!navigator.serviceWorker) return 'no-service-worker';

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
    catch (err) { return 'failed: ' + String((err && err.message) || err); }
    if (hadController) return 'registered';   // this load was already served by a worker

    // A reload helps at most once. If it did not produce isolation, reloading again never
    // will, and without this guard a broken setup reloads for ever.
    let reloaded = false;
    try { reloaded = !!sessionStorage.getItem(RELOAD_ONCE); } catch (e) { /* private mode */ }
    if (self.crossOriginIsolated) return 'isolated';        // meanwhile, nothing left to do
    if (reloaded) return 'still-not-isolated';
    try { sessionStorage.setItem(RELOAD_ONCE, '1'); } catch (e) { /* private mode */ }
    if (reg.active) { location.reload(); return 'reloading'; }
    const sw = reg.installing || reg.waiting;
    if (sw) sw.addEventListener('statechange', function (e) {
      if (e.target.state === 'activated') location.reload();
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
  // host whose caching is a URL that changes with the bytes), and `rememberTuning` -- which
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

    worker.addEventListener('message', function (e) {
      const d = e.data;
      if (!d || (d.__wt !== 'reply' && d.__wt !== 'event')) return;
      const p = pending.get(d.id);
      if (d.__wt === 'event') {
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
