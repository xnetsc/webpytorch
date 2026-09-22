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
