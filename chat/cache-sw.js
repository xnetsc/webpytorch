/* This page's caching, in the service worker.
 *
 * A client has exactly one controlling service worker, and on a static host that one has to
 * be the SDK's -- only a worker that controls the document can put the isolation headers on
 * the document's own response, and without them the whole model runs on the CPU. So this is
 * not a worker of its own: `webtorch.installServiceWorker({handler: 'cache-sw.js'})` in
 * index.html points the SDK's worker at this file, and it is loaded there.
 *
 * What is here is entirely this page's business and the SDK knows none of it: which files
 * are worth keeping, for how long, and which may be trusted forever. The SDK holds the
 * isolation above whatever this returns, so a mistake in here can make the page slow or
 * stale -- it cannot make it silently fall back to the CPU.
 *
 * The cache is for Python more than for us. Pyodide has none of its own: `loadPackage`
 * fetches the wheel from `indexURL` every time, and the only thing between a reload and
 * downloading numpy again is the HTTP cache, which the browser evicts whenever it likes. It
 * also does not go through `fetch` -- the Emscripten runtime reads packages with
 * XMLHttpRequest -- so wrapping fetch in the worker that loads them catches nothing, which
 * is measurably what happened. A service worker sits below all of it: XHR, fetch,
 * importScripts and streaming compilation alike.
 *
 * Two caches, because there are two kinds of file and only one of them can be trusted
 * forever.
 *
 *   FIXED  — a wheel, the wasm, a versioned CDN script. The version is IN the name, so the
 *            bytes behind a given URL never change. Cache-first, and the network is never
 *            touched again. This is what makes the page work with no network at all.
 *   APP    — this page's own html/js/css, and the SDK's Python modules. These change
 *            whenever the project does, so they are NETWORK-FIRST: the cache is a fallback
 *            for when the network is gone, never a reason to keep serving yesterday's code.
 *
 * Bump the suffix to abandon both; `activate` deletes anything that is not current. That is
 * for a change to the caching itself, not for shipping new code — APP is network-first, so
 * new code lands without it.
 *
 * FIXED's rule is "the version is in the name", and the wheels THIS PROJECT ships break it:
 * wgpy_webgpu-1.0.0-py3-none-any.whl and wgpy_webgl-1.0.0-py3-none-any.whl keep that name
 * whatever their bytes say. They used to be cached as fixed anyway, with a note to bump
 * CACHE_V whenever their contents changed. That is a rule someone has to remember, in a
 * place far from the file they edited, and it was forgotten the first time it mattered: a
 * backend fix shipped, and every client that had ever loaded the page went on installing the
 * wheel from before it. Nothing looked wrong — the fix simply was not there.
 *
 * So they are not treated as fixed. A wheel served from THIS origin is one of ours and its
 * name says nothing about its contents, so it goes in APP and is fetched network-first like
 * the rest of our code. A wheel from anywhere else (PyPI, a CDN) does carry its version in
 * its name and stays fixed.
 */
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

// `res.body` is spoken for the moment the response is returned, so the copy has to be taken
// before the write is even scheduled: by the time `caches.open` resolves, cloning throws and
// the entry is silently never written. That is exactly what happened once -- the caches
// stayed empty while everything looked fine. `keepUntil` keeps the worker alive until the
// write lands; without it a worker that goes idle takes the pending put with it.
function keep(ctx, cacheName, url, res) {
  if (!res || !res.ok || res.type === 'opaque') return res;   // opaque: unreadable, poison
  var copy = res.clone();
  ctx.keepUntil(caches.open(cacheName).then(function (c) {
    return c.put(url, copy);
  }).catch(function () { /* over quota, or not storable */ }));
  return res;
}

// Fixed: cache, then network. Once it is here it is here.
async function fromCache(req, ctx) {
  var hit = await caches.match(req.url);
  if (hit) return hit;
  return keep(ctx, FIXED, req.url, await fetch(req));
}

// App: network, then cache. A running network always wins, so an update lands the moment it
// exists; the cache only answers when the network cannot.
//
// `no-cache` is what makes that true. A plain `fetch(req)` carries the request's own cache
// mode, so the browser's HTTP cache answers it without going to the network at all -- the
// "network first" rule then quietly serves a stale file, and the update lands only on the
// SECOND reload, once the first has revalidated it. That is the "have to refresh twice"
// behaviour, and it is not cosmetic: it once served a worker old enough to be missing a
// command the page had already been updated to call. `no-cache` revalidates rather than
// re-downloads, so an unchanged file still costs one 304.
async function fromNetwork(req, ctx) {
  try {
    return keep(ctx, APP, req.url, await fetch(new Request(req, { cache: 'no-cache' })));
  } catch (err) {
    var hit = await caches.match(req.url);
    if (hit) return hit;
    throw err;
  }
}

webtorch.handleFetch(function (req, ctx) {
  return isFixed(req.url) ? fromCache(req, ctx) : fromNetwork(req, ctx);
});

// Old generations, and anything filed under FIXED that no longer qualifies as fixed.
self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (names) {
      return Promise.all(names.map(function (n) {
        if (/^webtorch-(fixed|app)-/.test(n) && n !== FIXED && n !== APP) return caches.delete(n);
      }));
    }).then(function () {
      return caches.open(FIXED).then(function (c) {
        return c.keys().then(function (reqs) {
          return Promise.all(reqs.map(function (r) {
            if (!isFixed(r.url)) return c.delete(r);
          }));
        });
      });
    }).catch(function () { /* nothing here is worth failing activation over */ })
  );
});
