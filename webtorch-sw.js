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

// What goes wrong in here, to every page that can hear it. A service worker's console is a
// different console from the page's, opened from a different place, and nobody looks at it;
// a handler that will not load or that throws on every request would otherwise be invisible
// while the page just quietly stopped caching. `__wtsw: 2` is this, separate from the `1`
// the host's two halves talk over.
async function report(message, detail) {
  try {
    const cs = await self.clients.matchAll({ includeUncontrolled: true, type: 'window' });
    for (const c of cs) { try { c.postMessage({ __wtsw: 2, message: message, detail: detail }); } catch (e) {} }
  } catch (e) { /* no clients yet; the console line below is all there is */ }
  console.warn('webtorch service worker: ' + message, detail || '');
}

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

// ---- what a host plugs in ---------------------------------------------------------------
//
// One handler, set by the host's own script when this worker loads it below. It is given a
// Request and whatever it returns is served; returning nothing means "not mine, go to the
// network". It never sees the FetchEvent, and that is the point -- see the note on the fetch
// listener for what that buys.
let handler = null;
/** The host's fetch handler. Registered when its file loads, which is at install. */
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
      // the request: the network still works, and isolation still has to happen. It is said
      // out loud, though -- a handler that throws on everything is a page with no caching
      // at all, and that is worth knowing before someone wonders why loads got slower.
      report('the page fetch handler threw, going to the network',
             String((err && err.message) || err));
    }
  }
  return isolate(await fetch(req));
}

// ---- the one way the host's two halves talk to each other -------------------------------
//
// A service worker outlives the page, is shared by every tab, and is the only thing standing
// between the document and the headers that keep the model on the GPU. Letting a host's
// handler take raw `message` events off it would mean a second, unseen protocol on the same
// port as whatever this worker needs it for, in a context where "unseen" is how the bad
// failures here have always looked. So there is one channel, it is typed, and the events it
// rides on are sealed below before the host's file is loaded.
//
// The host's half registers its side when its file loads -- that is, at install -- and the
// page's side is given to `webtorch.installServiceWorker`. Neither can be added later.
let onPage = null;
self.webtorch.onPageMessage = function (fn) {
  if (typeof fn !== 'function') throw new TypeError('webtorch.onPageMessage wants a function');
  onPage = fn;
};
/** To every window this worker knows about, controlled or not. */
self.webtorch.sendToPage = async function (data) {
  const cs = await self.clients.matchAll({ includeUncontrolled: true, type: 'window' });
  for (const c of cs) { try { c.postMessage({ __wtsw: 1, data: data }); } catch (e) {} }
};
self.addEventListener('message', (event) => {
  const d = event.data;
  if (!d || d.__wtsw !== 1 || !onPage) return;
  try {
    onPage(d.data, function (answer) {
      if (event.source) { try { event.source.postMessage({ __wtsw: 1, data: answer }); } catch (e) {} }
    });
  } catch (err) {
    report('the page message handler threw', String((err && err.message) || err));
  }
});

// Sealed. `message` and `messageerror` are this channel's, and a handler that reaches for
// them gets told where the door is rather than quietly getting a second protocol. Everything
// else a service worker can listen for is still the host's -- `activate` for cache cleanup,
// for one -- because none of it can take a response away from the fetch listener above.
(function seal() {
  const CLOSED = { message: 1, messageerror: 1 };
  const why = function (type) {
    return new Error('webtorch: a service worker handler cannot listen for "' + type
      + '". Use webtorch.onPageMessage(fn) and webtorch.sendToPage(data), and give the page '
      + 'its side as `onMessage` when you call webtorch.installServiceWorker().');
  };
  const add = self.addEventListener.bind(self);
  self.addEventListener = function (type) {
    if (CLOSED[type]) throw why(type);
    return add.apply(null, arguments);
  };
  for (const name of ['onmessage', 'onmessageerror']) {
    Object.defineProperty(self, name, {
      get: function () { return null; },
      set: function () { throw why(name.slice(2)); },
      configurable: false,
    });
  }
})();

// Loaded LAST, so everything above is already in place when it runs.
if (HANDLER) {
  try { importScripts(new URL(HANDLER, self.location.href).href); }
  catch (err) {
    // The host's handler is the host's business; isolation is not, and it must survive a
    // handler that will not load. Requests then go straight to the network, isolated -- and
    // the page is told, because otherwise its caching is simply gone with nothing said.
    report('could not load the page handler ' + HANDLER, String((err && err.message) || err));
  }
}
