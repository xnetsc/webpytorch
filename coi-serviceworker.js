/* Cross-origin isolation for hosts that cannot set headers.
 *
 * wgpy needs SharedArrayBuffer, and SharedArrayBuffer needs the document to be
 * cross-origin isolated -- which normally means two response headers. A static host like
 * GitHub Pages cannot send them, so a service worker adds them to every response it
 * proxies. Without this the page silently falls back to running the whole model on the CPU.
 *
 * COEP is `credentialless` rather than `require-corp`: model weights come from ModelScope
 * and Pyodide from a CDN, and neither sets Cross-Origin-Resource-Policy. `credentialless`
 * fetches those without credentials instead of blocking them outright.
 */
if (typeof window === 'undefined') {
  self.addEventListener('install', () => self.skipWaiting());
  self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

  self.addEventListener('fetch', function (event) {
    const req = event.request;
    if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
    event.respondWith(
      fetch(req)
        .then(function (res) {
          if (res.status === 0) return res;               // opaque: nothing to re-wrap
          const headers = new Headers(res.headers);
          headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          return new Response(res.body, {
            status: res.status, statusText: res.statusText, headers: headers,
          });
        })
        .catch(function (e) { console.error(e); throw e; })
    );
  });
} else {
  (function () {
    if (window.crossOriginIsolated) return;              // already isolated: nothing to do
    if (!navigator.serviceWorker) {
      console.warn('coi: no service worker support; SharedArrayBuffer stays unavailable');
      return;
    }
    navigator.serviceWorker.register(document.currentScript.src).then(
      function (reg) {
        // The worker only controls pages loaded after it took over, so the first visit
        // needs one reload before SharedArrayBuffer exists.
        if (reg.active && !navigator.serviceWorker.controller) window.location.reload();
      },
      function (err) { console.warn('coi: registration failed:', err); }
    );
  })();
}
