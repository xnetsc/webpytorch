// Pyodide worker: boots WgPy (WebGPU/WebGL backend), writes the webtorch library
// modules into the Pyodide FS, then runs the selected script (run.py).
importScripts('../lib/pyodide/pyodide.js');
importScripts('../dist/wgpy-worker.js');

let pyodide;
function log(message) { postMessage({ namespace: 'app', method: 'log', message }); }
function result(json) { postMessage({ namespace: 'app', method: 'result', json }); }
function stdout(line) { log(line.replace(/\x1b\[[0-9;]*[A-Za-z]/, '')); }

async function fetchText(url) {
  const f = await fetch(url + (url.includes('?') ? '&' : '?') + 'cb=' + Date.now(), { cache: 'no-store' });
  if (!f.ok) throw new Error(`${url}: ${f.statusText}`);
  return f.text();
}

// the `webtorch` SDK package, loaded into the Pyodide FS as a package dir so
// `import webtorch` works (public API in __init__).
const PKG = 'webtorch';
const PKG_MODULES = ['__init__.py', '_core.py', '_sdk.py', 'torchshim.py', 'ggufload.py',
  'hfcompat.py', 'webenv.py', 'llm.py', 'vl.py', 'detection.py', 'tts.py', 'webio.py',
  'onnxrt.py', 'lm_engine.py', 'quantize.py', 'audiofe.py', 'cosyvoice.py'];

async function start(config) {
  log('init wgpy worker interface');
  let initWorkerResult = null;
  try {
    initWorkerResult = await wgpy.initWorker();
  } catch (e) {
    log(`initWorker failed: ${e.message}`);
  }

  log('loading pyodide');
  pyodide = await loadPyodide({ indexURL: '../lib/pyodide/', stdout, stderr: stdout });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  if (initWorkerResult) {
    await pyodide.loadPackage(`../dist/wgpy_${initWorkerResult.backend}-1.0.0-py3-none-any.whl`);
    log(`backend: ${initWorkerResult.backend}`);
  } else {
    log('no GPU backend — running on numpy CPU');
  }

  try { pyodide.FS.mkdir(PKG); } catch (e) { /* exists */ }
  for (const m of PKG_MODULES) {
    pyodide.FS.writeFile(`${PKG}/${m}`, await fetchText(`../${PKG}/${m}`));
  }

  self.pythonIO = { config, result: null };
  self.pyodide = pyodide;          // expose to Python (from js import pyodide) for the FS env module
  log('running python');
  await pyodide.runPythonAsync(await fetchText(config.script || 'run.py'));
  if (self.pythonIO.result) result(self.pythonIO.result);
  if (self.pythonIO.audio) {
    const a = self.pythonIO.audio;             // { samples: number[], sr: number }
    const buf = Float32Array.from(a.samples);
    postMessage({ namespace: 'app', method: 'audio', samples: buf, sr: a.sr }, [buf.buffer]);
  }
  log('done');
}

addEventListener('message', (ev) => {
  if (ev.data.namespace !== 'app') return;
  if (ev.data.method === 'start') start(ev.data.config).catch((e) => log(`Worker error: ${e.message}\n${e.stack || ''}`));
});
