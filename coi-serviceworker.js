/* Cross-origin isolation for hosts that cannot set headers, and a cache for Python.
 *
 * Two jobs, one worker, because a page can only be controlled by one and both need to sit
 * in front of the same requests.
 *
 * The SDK needs SharedArrayBuffer, which needs the document to be cross-origin isolated,
 * which normally means two response headers. A static host like GitHub Pages cannot send
 * them, so this adds them to every response it proxies; without it the page silently falls
 * back to running the whole model on the CPU.
 *
 * WHICH headers is not decided here -- see `webtorch/js/webtorch-coi.js`, which is the SDK's
 * and is imported below. The split is deliberate: a host can build a service worker, a scope
 * and a caching policy on its own, and cannot work out from first principles that COEP has
 * to be `require-corp` rather than `credentialless` (it is a measurement on WebKit, and
 * getting it wrong is invisible except on iPhones). Whatever the host can build fully stays
 * here; the rest is asked for.
 *
 * The cache is for Python. Pyodide has none of its own: `loadPackage` fetches the wheel from
 * `indexURL` every time, and the only thing between a reload and downloading numpy again is
 * the HTTP cache, which the browser evicts whenever it likes. It also does not go through
 * `fetch` -- the Emscripten runtime reads packages with XMLHttpRequest -- so wrapping fetch
 * in the worker that loads them catches nothing, which is measurably what happened. A
 * service worker sits below all of it: XHR, fetch, importScripts and streaming compilation
 * alike. Every wheel is cached, whoever asked for it and wherever it came from.
 */
// Two caches, because there are two kinds of file and only one of them can be trusted
// forever.
//
//   FIXED  — a wheel, the wasm, a versioned CDN script. The version is IN the name, so the
//            bytes behind a given URL never change. Cache-first, and the network is never
//            touched again. This is what makes the page work with no network at all.
//   APP    — this page's own html/js/css and the workers. These change whenever the project
//            does, so they are NETWORK-FIRST: the cache is a fallback for when the network
//            is gone, never a reason to keep serving yesterday's code.
//
// Bump the suffix to abandon both; `activate` deletes anything that is not current. That is
// for a change to the caching itself, not for shipping new code — APP is network-first, so
// new code lands without it.
//
// FIXED's rule is "the version is in the name", and the wheels THIS PROJECT ships break it:
// wgpy_webgpu-1.0.0-py3-none-any.whl and wgpy_webgl-1.0.0-py3-none-any.whl keep that name
// whatever their bytes say. They used to be cached as fixed anyway, with a note to bump
// CACHE_V whenever their contents changed. That is a rule someone has to remember, in a
// place far from the file they edited, and it was forgotten the first time it mattered: a
// backend fix shipped, and every client that had ever loaded the page went on installing
// the wheel from before it. Nothing looked wrong — the fix simply was not there.
//
// So they are not treated as fixed. A wheel served from THIS origin is one of ours and its
// name says nothing about its contents, so it goes in APP and is fetched network-first like
// the rest of our code: a change lands on the next load, and the cached copy still answers
// when the network is gone. A wheel from anywhere else (PyPI, a CDN) does carry its version
// in its name and stays fixed. The cost is one conditional request for ~50KB per load,
// against a Pyodide distribution measured in megabytes.
var CACHE_V = 'v4';
var FIXED = 'webtorch-fixed-' + CACHE_V;
var APP = 'webtorch-app-' + CACHE_V;

// Named by version, so a hit is always the right bytes.
var FIXED_EXT = /\.(whl|wasm|zip|tar|data|tgz|gz|woff2?|ttf)$/i;
// The wheels this repo builds and serves: a fixed name over changing bytes, so they are not
// fixed no matter what the extension says.
var OUR_WHEEL = /\/dist\/[^/]+\.whl$/i;
function isFixed(url) {
  try {
    var u = new URL(url);
    if (u.origin === self.location.origin && OUR_WHEEL.test(u.pathname)) return false;
    if (FIXED_EXT.test(u.pathname)) return true;
    if (/pyodide-lock\.json$/.test(u.pathname)) return true;
    // A CDN path that pins a version: /npm/marked@18.0.11/…, /pyodide/v0.27.7/…
    if (u.origin !== self.location.origin && /@[\w.+-]+\/|\/v\d[\w.]*\//.test(u.pathname)) return true;
    return false;
  } catch (e) { return false; }
}
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  // (activate is below, with the cache cleanup)

  // The headers come from the SDK, because which response makes SharedArrayBuffer work is
  // the SDK's knowledge and not this file's: the choice of `require-corp` over
  // `credentialless` is a measurement on WebKit, and a host writing this from first
  // principles gets it wrong in a way that shows up only on iPhones, as a silent CPU
  // fallback. What is OURS is the worker, the scope, and everything below about what to
  // cache and how -- none of which the SDK can see.
  importScripts('webtorch/js/webtorch-coi.js');
  const isolate = self.webtorch.isolate;

  // Response from `res.body` on the way out, and by the time `caches.open` resolves the body
  // is already spoken for, so the clone throws and the entry is silently never written. That
  // is exactly what happened -- the caches stayed empty while everything looked fine.
  //
  // `waitUntil` keeps the worker alive until the write lands; without it a worker that goes
  // idle takes the pending put with it.
  function keep(event, cacheName, req, res) {
    if (!res || !res.ok || res.type === 'opaque') return res;   // opaque: unreadable, poison
    var copy = res.clone();
    event.waitUntil(caches.open(cacheName).then(function (c) {
      return c.put(req.url, copy);
    }).catch(function () { /* over quota, or not storable */ }));
    return res;
  }

  // Fixed: cache, then network. Once it is here it is here.
  function fromCache(event, req) {
    return caches.match(req.url).then(function (hit) {
      if (hit) return isolate(hit);
      return fetch(req).then(function (res) { return isolate(keep(event, FIXED, req, res)); });
    });
  }

  // App: network, then cache. A running network always wins, so an update lands the moment
  // it exists; the cache only answers when the network cannot.
  //
  // `no-cache` is what makes that true. A plain `fetch(req)` carries the request's own cache
  // mode, so the browser's HTTP cache answers it without going to the network at all -- the
  // "network first" rule then quietly serves a stale file, and the update lands only on the
  // SECOND reload, once the first has revalidated it. That is the "have to refresh twice"
  // behaviour, and it is not cosmetic: it served a worker.js old enough to be missing a
  // command the page had already been updated to call. `no-cache` revalidates rather than
  // re-downloads, so an unchanged file still costs one 304.
  function fromNetwork(event, req) {
    return fetch(new Request(req, { cache: 'no-cache' }))
      .then(function (res) { return isolate(keep(event, APP, req, res)); })
      .catch(function (err) {
        return caches.match(req.url).then(function (hit) {
          if (hit) return isolate(hit);
          throw err;
        });
      });
  }

  self.addEventListener('fetch', function (event) {
    const req = event.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    if (req.method !== 'GET') {
      event.respondWith(fetch(req).then(isolate));
      return;
    }
    event.respondWith(isFixed(req.url) ? fromCache(event, req) : fromNetwork(event, req));
  });

  // Anything from an older cache version is dead weight.
  // Take over the open pages, and drop anything from an older cache version.
  // Anything FIXED holds that this worker would no longer call fixed. Bumping CACHE_V is not
  // what removes it: the name is current, so the generation survives, and the entry sits
  // there unreachable -- `fromCache` is never consulted for that URL any more. It is only
  // wasted bytes, but it is also the kind of leftover that makes a cache read as evidence of
  // something that is not happening. A rule that changes should take its old entries with it.
  function dropMisfiled() {
    return caches.open(FIXED).then(function (c) {
      return c.keys().then(function (reqs) {
        return Promise.all(reqs.map(function (r) {
          if (!isFixed(r.url)) return c.delete(r);
        }));
      });
    }).catch(function () { /* nothing here is worth failing activation over */ });
  }

  self.addEventListener('activate', function (e) {
    e.waitUntil(Promise.all([
      self.clients.claim(),
      caches.keys().then(function (names) {
        return Promise.all(names.map(function (n) {
          if (/^webtorch-(fixed|app)-/.test(n) && n !== FIXED && n !== APP) return caches.delete(n);
        }));
      }).then(dropMisfiled)
    ]));
  });

  // Let the page empty the cache from Settings.
  self.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'clear-python-cache') {
      e.waitUntil(Promise.all([caches.delete(FIXED), caches.delete(APP)]).then(function () {
        if (e.source) e.source.postMessage({ type: 'python-cache-cleared' });
      }));
    }
  });
} else {
  (function () {
    var src = document.currentScript.src;

    // The worker has a second job now -- caching Python -- so it registers whether or not
    // the page is already isolated. What used to happen here was a stand-down: on a server
    // that sends the headers itself the worker was unregistered and never installed, which
    // is most correctly-configured deployments, and would have left them with no cache.
    //
    // Standing down is no longer necessary because the worker no longer overrides anything:
    // `isolate` leaves a response that already carries COEP exactly as the server sent it.
    // A FOREIGN worker from some earlier version is still cleared out, since that one does
    // override.
    if (window.crossOriginIsolated) {
      // Isolated -- but by whom? That is the whole question, and getting it wrong breaks the
      // page either way.
      //
      // If THIS worker is the one supplying the headers, unregistering it un-isolates the
      // next load, which registers it again, which isolates the load after that: the page
      // alternates between GPU and CPU on every refresh. (Seen on GitHub Pages, which cannot
      // send the headers, so the worker is the only thing providing them.)
      //
      // If the SERVER supplies them, a worker left over from somewhere that did not keeps
      // proxying -- which once mattered: an older version of this worker answered with
      // `credentialless`, an overwrite that breaks isolation in WebKit outright and
      // weakens the policy elsewhere. (Seen for real: a fall back to the CPU on a server
      // whose headers were correct.) Today `isolate` refuses to touch a response that
      // already carries COEP, and the value it would add is the strict one anyway.
      //
      // So stand down only when the isolation is not ours to hold.
      // Isolated already -- so no reload is needed and nothing is urgent. Register anyway,
      // for the cache; it will control the next load.
      try { sessionStorage.removeItem('__coi_reloaded'); } catch (e) { /* private mode */ }
      if (navigator.serviceWorker) navigator.serviceWorker.register(src).then(null, function () {});
      return;
    }

    // Opened as a file, not served. There is no response to add headers to and no service
    // worker API to add them with, so SharedArrayBuffer cannot exist -- and the module
    // fetches this page depends on are blocked by the file:// origin anyway. Say so plainly
    // instead of leaving a half-working page: without this the symptom is a silent fall back
    // to CPU, or a pile of opaque CORS errors.
    if (location.protocol === 'file:' || location.protocol === 'data:') {
      window.__coiFileMode = true;
      console.warn('coi: opened from ' + location.protocol +
                   ' -- serve this folder over HTTP instead (see the banner on the page)');
      return;
    }

    if (!navigator.serviceWorker) {
      window.__coiNoSW = true;
      console.warn('coi: no service worker support; SharedArrayBuffer stays unavailable');
      return;
    }
    // Read BEFORE registering. The moment the worker activates, its `clients.claim()` sets
    // `navigator.serviceWorker.controller` -- so reading it after registration says only that
    // a worker EXISTS, not whether it served THIS document. That race decides everything:
    // the document response of this load already went out without isolation headers, and no
    // amount of claiming changes that. If claim wins, a check that reads the controller
    // skips the reload, and the first visit stays on the CPU until somebody refreshes by
    // hand. (Seen for real: a phone reporting `serviceWorker: true` and
    // `crossOriginIsolated: false` at the same time.)
    var hadController = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.register(src).then(
      function (reg) {
        if (hadController) return;   // this load was served by a worker already
        // A reload helps at most once. If it did not produce isolation, reloading again
        // never will, and without this guard a broken setup reloads forever.
        var reloaded = false;
        try { reloaded = !!sessionStorage.getItem('__coi_reloaded'); } catch (e) { /* private mode */ }
        if (window.crossOriginIsolated) return;      // meanwhile, nothing left to do
        if (reloaded) {
          window.__coiStillNotIsolated = true;
          console.warn('coi: worker active but the page is still not isolated after a reload');
          return;
        }
        try { sessionStorage.setItem('__coi_reloaded', '1'); } catch (e) { /* private mode */ }
        if (reg.active) { window.location.reload(); return; }
        // Still installing: wait for activation rather than racing it.
        var sw = reg.installing || reg.waiting;
        if (sw) sw.addEventListener('statechange', function (e) {
          if (e.target.state === 'activated') window.location.reload();
        });
      },
      function (err) { window.__coiSWFailed = String(err); console.warn('coi: registration failed:', err); }
    );
  })();
}
