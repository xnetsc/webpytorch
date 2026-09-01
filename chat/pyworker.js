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
    // Nothing here has a screen to draw on, and each library has to be told in its own way.
    //
    // matplotlib: AGG renders to a buffer, which is what the figure capture below reads.
    //
    // SDL: this matters more than a default, because without it pygame does not fail, it
    // HANGS. `pygame.display.set_mode()` looks for the canvas the emscripten driver needs,
    // a worker has no document to find one in, and the call never returns -- taking the
    // interpreter with it, so every later call queues behind it forever (measured: a
    // following `1+1` never came back either). Naming a driver this build does not have
    // turns that into a plain error at the point of the call. Neither `dummy` nor
    // `offscreen` is compiled into the SDL 2.28.4 here, and an OffscreenCanvas installed
    // through `specialHTMLTargets` does not satisfy the emscripten driver either -- both
    // tried, both still hung. Drawing itself is unaffected: Surfaces, draw, font and
    // image.save need no video driver at all.
    py.runPython(`import os
os.environ.setdefault("MPLBACKEND", "AGG")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")`);
    await add(packages || []);
    send({ type: 'state', state: 'ready', packages: [...loaded],
           version: (py && py.version) || null });
    return py;
  })();
  return booting;
}

// Load whatever is asked for and not already here.
//
// Two sources, tried in that order, because they are not the same thing. `loadPackage` reads
// the distribution that shipped with this Pyodide -- 260-odd prebuilt packages, fetched from
// `indexURL`, which the service worker has cached, so it works with no network. micropip
// reaches PyPI, which covers everything else that is a pure-Python wheel.
//
// The order is loadPackage first, not because it is faster (measured: comparable, 111 ms
// against 86 ms for a package of similar size) but because it is the offline one. micropip
// would serve both -- it resolves distribution names from the lockfile before going out --
// but it would put a network dependency in front of packages that do not need one.
//
// A name is not a failure until BOTH have refused it. What comes back distinguishes the
// three outcomes, because they mean different things to whoever typed the name: it is here,
// it had to be fetched from PyPI, or nothing can supply it (typically a package with
// compiled code, which has no pure-Python wheel to install).
async function add(names) {
  const want = (names || []).map(s => String(s).trim()).filter(Boolean)
                            .filter(n => !loaded.has(n));
  const bad = [], fromPyPI = [];
  for (const n of want) {
    try { await py.loadPackage(n); loaded.add(n); continue; }
    catch (e) { /* not in the distribution -- try PyPI */ }
    try {
      if (!micropipReady) {
        await py.loadPackage('micropip'); loaded.add('micropip'); micropipReady = true;
      }
      const mp = py.pyimport('micropip');
      try { await mp.install(n); } finally { if (mp.destroy) { try { mp.destroy(); } catch (e2) {} } }
      loaded.add(n); fromPyPI.push(n);
    } catch (e) {
      // The LAST meaningful line, not the first: a Python exception arrives here as a
      // traceback, whose first line is the word "Traceback" and tells nobody anything. The
      // reason is the exception line at the end.
      const lines = String((e && e.message) || e).split('\n').map(x => x.trim()).filter(Boolean);
      const msg = [...lines].reverse().find(x => /^[A-Za-z_.]+(Error|Exception):/.test(x))
                || lines[lines.length - 1] || 'unknown error';
      bad.push({ name: n, why: msg.slice(0, 220) });
    }
  }
  return { unavailable: bad, fromPyPI };
}
let micropipReady = false;

// Run one block. Returns everything the person needs to see: what it printed, what it
// raised, the value of the last expression, and any figures it drew.
async function run(code) {
  await boot();
  let out = '', err = '';
  // Imports the code needs that are not loaded yet — Pyodide can find them from the import
  // statements, so `import numpy` works on the first run without anyone configuring it.
  //
  // BEFORE the redirect, not after: the package loader narrates what it is doing
  // ("matplotlib already loaded from default channel"), and captured as stdout that reads as
  // something the code printed. It is the runtime talking about itself.
  try { await py.loadPackagesFromImports(code); } catch (e) { /* reported by the run itself */ }
  // Kept apart. A library that writes a notice on first use -- matplotlib building its font
  // cache is the one everybody meets -- lands in stderr, and mixed into stdout it reads as
  // part of what the code produced. The model then reports it as such.
  py.setStdout({ batched: (s) => { out += s + '\n'; } });
  py.setStderr({ batched: (s) => { err += s + '\n'; } });
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
  // Anything the code left drawn, as a PNG. Done after the run so a script that both prints
  // and plots shows both.
  //
  // pygame draws too, and its pictures reach the reader the same way -- one channel for
  // "what this run produced to look at", whichever library made it. Its display surface is
  // the closest thing it has to matplotlib's open figures; a Surface that was never given
  // to `set_mode` is a value the code holds, and only the code knows which of them was the
  // point.
  try {
    if (loaded.has('pygame-ce') || loaded.has('pygame')) {
      const b64 = py.runPython(`
def __wt_pygame():
    import base64, io
    try:
        import pygame
        if not pygame.display.get_init():
            return []
        s = pygame.display.get_surface()
        if s is None:
            return []
        buf = io.BytesIO()
        pygame.image.save(s, buf, "PNG")
        return [base64.b64encode(buf.getvalue()).decode()]
    except Exception:
        return []
__wt_pygame()
`);
      const got = b64 ? b64.toJs() : [];
      if (b64 && b64.destroy) b64.destroy();
      images = images.concat(got);
    }
  } catch (e) { /* a broken surface must not hide the output */ }
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
  return { out, err, value, error, images };
}

// Install a wheel: from a URL, or from a file the person picked. micropip handles both --
// a local one is written into the interpreter's filesystem first and installed from there,
// because micropip installs from a location and a File is not one.
//
// A URL is fetched by the page, not by micropip: this page runs under
// Cross-Origin-Embedder-Policy, so a host that does not send CORS headers cannot be read at
// all, and saying which host refused is more use than a Python traceback about it.
async function install(spec) {
  await boot();
  if (!loaded.has('micropip')) { await py.loadPackage('micropip'); loaded.add('micropip'); }
  const micropip = py.pyimport('micropip');
  let where = spec.url;
  try {
    if (spec.bytes) {
      const dir = '/wheels';
      try { py.FS.mkdir(dir); } catch (e) { /* already there */ }
      const path = dir + '/' + (spec.name || 'package.whl');
      py.FS.writeFile(path, new Uint8Array(spec.bytes));
      where = 'emfs:' + path;
    } else {
      // Fetched here so a CORS failure is reported as one.
      const r = await fetch(spec.url);
      if (!r.ok) throw new Error('HTTP ' + r.status + ' from ' + spec.url);
      const buf = new Uint8Array(await r.arrayBuffer());
      const dir = '/wheels';
      try { py.FS.mkdir(dir); } catch (e) {}
      const name = (spec.url.split('/').pop() || 'package.whl').split('?')[0];
      const path = dir + '/' + name;
      py.FS.writeFile(path, buf);
      where = 'emfs:' + path;
    }
    await micropip.install(where);
    const mod = (spec.name || where).split('/').pop().split('-')[0].replace(/_/g, '-');
    loaded.add(mod);
    send({ type: 'state', state: 'ready', packages: [...loaded] });
    return { installed: mod };
  } finally {
    if (micropip && micropip.destroy) { try { micropip.destroy(); } catch (e) {} }
  }
}

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data || {};
  try {
    let res = null;
    if (cmd === 'boot') { await boot(args && args.packages); res = [...loaded]; }
    else if (cmd === 'packages') {
      await boot(args && args.packages);
      const r = await add(args && args.packages);
      send({ type: 'state', state: 'ready', packages: [...loaded] });
      res = { loaded: [...loaded], unavailable: r.unavailable, fromPyPI: r.fromPyPI };
    }
    else if (cmd === 'run') res = await run(args && args.code);
    else if (cmd === 'install') res = await install(args || {});
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
