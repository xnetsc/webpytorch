/* A Python runtime for the code blocks in a reply — separate from the model's.
 *
 * Deliberately its own worker with its own Pyodide. The model's runtime holds gigabytes of
 * weights and is busy decoding; running someone's `while True` in it would take the chat
 * down with it. This one can be reset, and its memory is its own.
 *
 * Packages come from the same Pyodide distribution as the runtime itself, and are loaded
 * when the settings say so rather than when the person presses Run — the first `import
 * pandas` otherwise costs a ten-second pause with nothing to look at.
 */
const PYODIDE_URL = self.PYODIDE_URL || 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/';
importScripts(PYODIDE_URL + 'pyodide.js');

let py = null;                 // the interpreter, once booted
let booting = null;            // the boot in flight, so concurrent asks share one
const loaded = new Set();      // packages already in this interpreter

const send = (m) => self.postMessage(m);

async function boot(packages) {
  if (py) return py;
  if (booting) return booting;
  booting = (async () => {
    send({ type: 'state', state: 'booting' });
    py = await loadPyodide({ indexURL: PYODIDE_URL });
    // matplotlib in a worker has no screen to draw on; AGG renders to a buffer, which is
    // what the figure capture below reads.
    py.runPython('import os; os.environ.setdefault("MPLBACKEND", "AGG")');
    await add(packages || []);
    send({ type: 'state', state: 'ready', packages: [...loaded] });
    return py;
  })();
  return booting;
}

// Load whatever is asked for and not already here. A name Pyodide does not ship is reported
// and skipped rather than failing the batch -- one bad entry in the settings box should not
// leave the others unloaded.
async function add(names) {
  const want = (names || []).map(s => String(s).trim()).filter(Boolean)
                            .filter(n => !loaded.has(n));
  const bad = [];
  for (const n of want) {
    try { await py.loadPackage(n); loaded.add(n); }
    catch (e) { bad.push(n); }
  }
  return bad;
}

// Run one block. Returns everything the person needs to see: what it printed, what it
// raised, the value of the last expression, and any figures it drew.
async function run(code) {
  await boot();
  let out = '';
  py.setStdout({ batched: (s) => { out += s + '\n'; } });
  py.setStderr({ batched: (s) => { out += s + '\n'; } });
  // Imports the code needs that are not loaded yet — Pyodide can find them from the import
  // statements, so `import numpy` works on the first run without anyone configuring it.
  try { await py.loadPackagesFromImports(code); } catch (e) { /* reported by the run itself */ }
  let value = null, error = null, images = [];
  try {
    const r = await py.runPythonAsync(code);
    if (r !== undefined && r !== null) {
      try { value = String(r.toString ? r.toString() : r); } catch (e) { value = null; }
      if (r && typeof r.destroy === 'function') { try { r.destroy(); } catch (e) {} }
    }
  } catch (e) {
    error = String(e.message || e);
  }
  // Any figure the code left open, as a PNG. Done after the run so a script that both
  // prints and plots shows both.
  try {
    if (loaded.has('matplotlib')) {
      const b64 = py.runPython(`
def __wt_figs():
    import base64, io
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out = []
    for n in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(n).savefig(buf, format="png", bbox_inches="tight", dpi=110)
        out.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return out
__wt_figs()
`);
      images = b64 ? b64.toJs() : [];
      if (b64 && b64.destroy) b64.destroy();
    }
  } catch (e) { /* a broken plot must not hide the output */ }
  py.setStdout({}); py.setStderr({});
  return { out, value, error, images };
}

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data || {};
  try {
    let res = null;
    if (cmd === 'boot') { await boot(args && args.packages); res = [...loaded]; }
    else if (cmd === 'packages') {
      await boot(args && args.packages);
      const bad = await add(args && args.packages);
      send({ type: 'state', state: 'ready', packages: [...loaded] });
      res = { loaded: [...loaded], unavailable: bad };
    }
    else if (cmd === 'run') res = await run(args && args.code);
    else if (cmd === 'reset') {                       // drop the interpreter and start over
      py = null; booting = null; loaded.clear();
      send({ type: 'state', state: 'idle' });
      res = true;
    }
    send({ type: 'res', id, res });
  } catch (err) {
    send({ type: 'res', id, error: String(err && err.message || err) });
  }
};
