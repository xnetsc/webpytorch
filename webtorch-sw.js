/* webtorch — the SDK's service worker.
 *
 * The SDK needs the document to be cross-origin isolated, or SharedArrayBuffer is gone and
 * the whole model runs on the CPU. On a host that can set response headers that is two lines
 * of server config and none of this. On a static host it is a service worker, because only
 * a worker that CONTROLS the document can put headers on the document's own response.
 *
 * A client has exactly one controller, so there is one service worker to have, and it has to
 * be this one. That is not a land grab: it is the only arrangement in which the isolation can
 * be guaranteed rather than hoped for. What a host would otherwise have put in its own worker
 * goes in through `webtorch.handleFetch` instead, and runs with the isolation held above it.
 *
 * Not loaded by hand. `webtorch.installServiceWorker()` on the page registers it, points it
 * at the host's handler, and deals with the reload that a first visit needs.
 *
 * WHY IT SITS AT THE DISTRIBUTION ROOT, next to dist/ and webtorch/: a worker's scope cannot
 * reach above the directory its script is served from, unless the server sends
 * `Service-Worker-Allowed`, which static hosts do not. From webtorch/js/ this could only ever
 * control webtorch/js/, and the host's pages -- the documents that need isolating -- are not
 * in there.
 */
const HANDLER = new URLSearchParams(self.location.search).get('handler') || null;
const BASE = new URL('.', self.location.href).href;

importScripts(BASE + 'webtorch/js/webtorch-coi.js');
const isolate = self.webtorch.isolate;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

// ---- what a host plugs in ---------------------------------------------------------------
//
// One handler, set by the host's own script when this worker loads it below. It is given a
// Request and whatever it returns is served; returning nothing means "not mine, go to the
// network". It never sees the FetchEvent, and that is the point -- see the note on the fetch
// listener for what that buys.
let handler = null;
self.webtorch.handleFetch = function (fn) {
  if (typeof fn !== 'function') throw new TypeError('webtorch.handleFetch wants a function');
  handler = fn;
};

// ---- the isolation, held above whatever the host does -----------------------------------
//
// This listener is added BEFORE the host's script is loaded, and that ordering is the
// guarantee. Fetch listeners run in the order they were added and the FIRST call to
// `respondWith` wins -- a later one throws -- so a host script that adds its own listener,
// deliberately or by pasting something in, cannot take the response away from here. Every
// answer this worker gives goes out through `isolate`, including the ones the host produced.
//
// It is precedence, not a sandbox: the host's code shares this global and could still break
// things it has no business touching. What it cannot do is quietly serve a document without
// the headers, which is the failure that matters, because that one is silent -- the page
// works, and everything on it is thirty times slower.
self.addEventListener('fetch', (event) => {
  const req = event.request;
  // Range requests and the like: `only-if-cached` outside same-origin mode must not be
  // answered here at all.
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  event.respondWith(serve(event, req));
});

async function serve(event, req) {
  if (handler && req.method === 'GET') {
    try {
      const res = await handler(req, {
        // For work that must outlive the response -- writing to a cache, mostly.
        keepUntil: (p) => { try { event.waitUntil(p); } catch (e) { /* too late */ } },
      });
      if (res) return isolate(res);
    } catch (err) {
      // A handler that throws is a handler that is having a bad day, not a reason to fail
      // the request: the network still works, and isolation still has to happen.
      console.warn('webtorch: the host fetch handler threw, going to the network:', err);
    }
  }
  return isolate(await fetch(req));
}

// Loaded LAST, so everything above is already in place when it runs.
if (HANDLER) {
  try { importScripts(new URL(HANDLER, self.location.href).href); }
  catch (err) {
    // The host's handler is the host's business; isolation is not, and it must survive a
    // handler that will not load. Requests then go straight to the network, isolated.
    console.warn('webtorch: could not load the host fetch handler ' + HANDLER + ':', err);
  }
}
