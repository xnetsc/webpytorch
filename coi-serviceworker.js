/* Cross-origin isolation for hosts that cannot set headers, and a cache for Python.
 *
 * Two jobs, one worker, because a page can only be controlled by one and both need to sit
 * in front of the same requests.
 *
 * wgpy needs SharedArrayBuffer, and SharedArrayBuffer needs the document to be
 * cross-origin isolated -- which normally means two response headers. A static host like
 * GitHub Pages cannot send them, so a service worker adds them to every response it
 * proxies. Without this the page silently falls back to running the whole model on the CPU.
 *
 * COEP is `credentialless` rather than `require-corp`: model weights come from ModelScope
 * and Pyodide from a CDN, and neither sets Cross-Origin-Resource-Policy. `credentialless`
 * fetches those without credentials instead of blocking them outright. Headers are only
 * ADDED when the response has none -- a server that sets its own is the authority, and
 * overriding a stricter `require-corp` with `credentialless` would weaken it.
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
// Bump the suffix to abandon both; `activate` deletes anything that is not current.
var CACHE_V = 'v2';
var FIXED = 'webtorch-fixed-' + CACHE_V;
var APP = 'webtorch-app-' + CACHE_V;

// Named by version, so a hit is always the right bytes.
var FIXED_EXT = /\.(whl|wasm|zip|tar|data|tgz|gz|woff2?|ttf)$/i;
function isFixed(url) {
  try {
    var u = new URL(url);
    if (FIXED_EXT.test(u.pathname)) return true;
    if (/pyodide-lock\.json$/.test(u.pathname)) return true;
    // A CDN path that pins a version: /npm/marked@18.0.11/…, /pyodide/v0.25.0/…
    if (u.origin !== self.location.origin && /@[\w.+-]+\/|\/v\d[\w.]*\//.test(u.pathname)) return true;
    return false;
  } catch (e) { return false; }
}
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  // (activate is below, with the cache cleanup)

  // Add the isolation headers, but only to a response that has none of its own.
  function isolate(res) {
    if (res.status === 0) return res;                    // opaque: nothing to re-wrap
    if (res.headers.get('Cross-Origin-Embedder-Policy')) return res;   // the server decided
    const headers = new Headers(res.headers);
    headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, {
      status: res.status, statusText: res.statusText, headers: headers,
    });
  }

  // The clone is taken RIGHT HERE, synchronously, before anything reads the body.
  // Cloning inside the promise instead looks equivalent and is not: `isolate` builds a new
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
  function fromNetwork(event, req) {
    return fetch(req)
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
  self.addEventListener('activate', function (e) {
    e.waitUntil(Promise.all([
      self.clients.claim(),
      caches.keys().then(function (names) {
        return Promise.all(names.map(function (n) {
          if (/^webtorch-(fixed|app)-/.test(n) && n !== FIXED && n !== APP) return caches.delete(n);
        }));
      })
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
      // proxying, and its `credentialless` overrides the server's stricter `require-corp` --
      // breaking the isolation it was added to provide. (Seen for real: a fall back to the
      // CPU on a server whose headers were correct.)
      //
      // So stand down only when the isolation is not ours to hold.
      // Isolated already -- so no reload is needed and nothing is urgent. Register anyway,
      // for the cache; it will control the next load.
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
    navigator.serviceWorker.register(src).then(
      function (reg) {
        // The worker only controls pages loaded after it took over, so the first visit
        // needs one reload before SharedArrayBuffer exists.
        if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
      },
      function (err) { window.__coiSWFailed = String(err); console.warn('coi: registration failed:', err); }
    );
  })();
}
