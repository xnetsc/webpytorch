/* webtorch — worker-side bootstrap.
 *
 * Brings up the GPU backend, Pyodide and the webtorch package in the one order that
 * works: the worker half of wgpy must be initialised after the main thread's half and
 * before Pyodide loads. Getting that order wrong does not raise — it silently drops every
 * tensor op onto numpy inside wasm — so the backend that actually came up is probed and
 * returned rather than assumed.
 *
 * Load after dist/wgpy-worker.js and the Pyodide loader, then:
 *     const { pyodide, backend } = await webtorch.initWorker({ baseURL: '../' });
 */
(function (root) {
  const wt = root.webtorch || (root.webtorch = {});

  // The package's own module list: SDK knowledge, so callers do not have to track it.
  const MODULES = ['__init__.py', '_core.py', '_sdk.py', 'backend.py', 'torchshim.py',
    'ggufload.py', 'hfcompat.py', 'webenv.py', 'llm.py', 'linear_attn.py', 'vl.py',
    'detection.py', 'tts.py', 'webio.py', 'onnxrt.py', 'lm_engine.py', 'quantize.py',
    'audiofe.py', 'cosyvoice.py', 'multimodal.py', 'iqtables.py'];

  // Set by the main thread before it sends anything else (see webtorch-main.js).
  let announced = null;
  const announcedBackend = new Promise((resolve) => { announced = resolve; });
  root.addEventListener('message', function (e) {
    if (e.data && e.data.__webtorch === 'backend') announced(e.data.backend);
  });

  async function text(u) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(u + ': ' + r.status);
    return r.text();
  }

  /**
   * Boot everything. Options (all optional):
   *   baseURL        prefix for dist/ and webtorch/ (default '../')
   *   pyodideIndexURL  default `${baseURL}lib/pyodide/`
   *   onStatus       (text) => void, progress for the UI
   * Resolves to { pyodide, backend } where backend is what actually came up:
   * 'webgpu' | 'webgl' | 'cpu'.
   */
  wt.initWorker = async function (opts) {
    opts = opts || {};
    const base = opts.baseURL || '../';
    const idx = opts.pyodideIndexURL || (base + 'lib/pyodide/');
    const say = opts.onStatus || function () {};

    // Wait for the main thread's choice; it has the device, this context does not.
    const wanted = await announcedBackend;

    if (wanted !== 'cpu' && typeof root.wgpy !== 'undefined') {
      say('connecting to the GPU…');
      try { await root.wgpy.initWorker(); }
      catch (e) { console.warn('webtorch: wgpy.initWorker failed, running on CPU:', e); }
    }

    say('starting Python…');
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
    for (const m of MODULES) {
      try { pyodide.FS.writeFile('webtorch/' + m, await text(base + 'webtorch/' + m)); }
      catch (e) { console.warn('webtorch: skipped ' + m + ': ' + e.message); }
    }
    await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/")');

    // What is actually live, not what was requested.
    let backend = 'cpu';
    try {
      backend = await pyodide.runPythonAsync('import webtorch; webtorch.backend()');
    } catch (e) { console.warn('webtorch: backend probe failed:', e); }
    say('ready (' + backend + ')');
    return { pyodide: pyodide, backend: backend };
  };
})(self);
