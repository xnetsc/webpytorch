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

// ---- cleaning up, which is the other half of caching ------------------------------------
//
// Every file this page serves carries a content hash in its query (`app.js?v=327c024b76`),
// and a cache is keyed by the WHOLE url. So each deploy writes a new entry and leaves the
// old one behind for ever: measured on the deployed page before this existed, 52 entries for
// the SDK's 27 Python modules, none of which could ever be matched again. A cache that only
// grows is not a cache, it is a slow leak with a hit rate.
//
// So a write evicts its own older versions. Same origin, same path, different query means an
// earlier build of the same file, and there is no reason to keep one: this bucket is
// network-first, so the only thing a stale entry can do is answer when the network is gone,
// and answering with last month's code is worse than not answering.
async function dropOlderVersions(c, url) {
  const me = new URL(url);
  const olds = (await c.keys()).filter(function (r) {
    const u = new URL(r.url);
    return u.origin === me.origin && u.pathname === me.pathname && u.search !== me.search;
  });
  for (const r of olds) await c.delete(r);
  return olds.length;
}

// The same rule applied to what is already there, once per worker update. A path with more
// than one entry is a path whose versions have piled up; drop them all rather than guess
// which is current, because network-first will put the right one back on the next load and
// the only thing lost in between is offline coverage for those files.
async function sweepVersions(c) {
  const byPath = {};
  for (const r of await c.keys()) {
    const u = new URL(r.url);
    (byPath[u.origin + u.pathname] || (byPath[u.origin + u.pathname] = [])).push(r);
  }
  let n = 0;
  for (const k of Object.keys(byPath)) {
    if (byPath[k].length < 2) continue;
    for (const r of byPath[k]) { await c.delete(r); n++; }
  }
  return n;
}

// Code that does not exist any more.
//
// A write only ever evicts its OWN path, so a file deleted from the project is never reached
// by it: nothing requests that path again, so nothing writes it, so the copy sits there for
// ever. This page has deleted two service-worker-adjacent files and a whole worker in the
// last day, and every browser that ever loaded them still had them -- and would still have
// SERVED them, because this bucket falls back to the cache when the network says no, and a
// 404 is the network saying no.
//
// A worker activating IS a new deploy, so that is when to ask. One conditional request per
// same-origin entry, a few dozen, mostly 304s; anything the server no longer has goes. Only
// our own origin, because what a CDN chooses to keep is not ours to police.
async function sweepDeleted(c) {
  const mine = (await c.keys()).filter(function (r) {
    return new URL(r.url).origin === self.location.origin;
  });
  let gone = 0;
  const LANES = 6;                        // enough to not be slow, few enough to not be rude
  await Promise.all(Array.from({ length: LANES }, async function (_, lane) {
    for (let i = lane; i < mine.length; i += LANES) {
      const r = mine[i];
      try {
        const res = await fetch(new Request(r.url, { cache: 'no-cache' }));
        if (res.status === 404 || res.status === 410) { await c.delete(r); gone++; }
      } catch (e) { /* offline: a file we cannot ask about is not a file we may delete */ }
    }
  }));
  return gone;
}

// `res.body` is spoken for the moment the response is returned, so the copy has to be taken
// before the write is even scheduled: by the time `caches.open` resolves, cloning throws and
// the entry is silently never written. That is exactly what happened once -- the caches
// stayed empty while everything looked fine. `keepUntil` keeps the worker alive until the
// write lands; without it a worker that goes idle takes the pending put with it.
function keep(ctx, cacheName, url, res) {
  if (!res || !res.ok || res.type === 'opaque') return res;   // opaque: unreadable, poison
  var copy = res.clone();
  ctx.keepUntil(caches.open(cacheName).then(async function (c) {
    try {
      await c.put(url, copy);
    } catch (err) {
      // Out of room, most likely. Swallowing it means the cache quietly stops working and
      // the page gets slower with no way to find out, so the page is told -- it is the only
      // thing here that can say anything to anyone.
      webtorch.sendToPage({ kind: 'cache-full', url: url,
                            error: String((err && err.name) || err) });
      return;
    }
    await dropOlderVersions(c, url);
  }).catch(function () { /* the cache itself is unavailable; nothing to do about it here */ }));
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

// Everything that has to go: old generations, anything filed under FIXED that no longer
// qualifies as fixed, and versions that piled up before a write was evicting its own.
self.addEventListener('activate', function (e) {
  e.waitUntil((async function () {
    try {
      for (const n of await caches.keys()) {
        if (/^webtorch-(fixed|app)-/.test(n) && n !== FIXED && n !== APP) await caches.delete(n);
      }
      const fixed = await caches.open(FIXED);
      for (const r of await fixed.keys()) { if (!isFixed(r.url)) await fixed.delete(r); }
      const app = await caches.open(APP);
      const stale = await sweepVersions(app);
      const gone = await sweepDeleted(app) + await sweepDeleted(fixed);
      if (stale || gone) {
        console.log('cache-sw: dropped ' + stale + ' stale versions and '
                    + gone + ' files that no longer exist');
        webtorch.sendToPage({ kind: 'cache-swept', staleVersions: stale, deleted: gone });
      }
    } catch (err) { /* nothing here is worth failing activation over */ }
  })());
});

// What the page may ask. Only what the page cannot find out for itself: the names are ours.
webtorch.onPageMessage(async function (msg, reply) {
  if (!msg || msg.ask !== 'caches') return;
  const out = { buckets: {}, total: 0 };
  for (const name of [FIXED, APP]) {
    const c = await caches.open(name);
    const n = (await c.keys()).length;
    out.buckets[name] = n; out.total += n;
  }
  reply(out);
});
