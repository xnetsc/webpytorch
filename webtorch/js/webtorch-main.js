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
 *     const { backend } = await webtorch.initMain(worker);
 *     // hold every message to the worker until this resolves
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
    return { backend: backend };
  };
})(self);
