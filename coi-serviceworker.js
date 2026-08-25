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
    var src = document.currentScript.src;

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
      var sw = navigator.serviceWorker;
      var ctrl = sw && sw.controller;
      if (sw && !(ctrl && ctrl.scriptURL === src)) {
        sw.getRegistrations().then(function (regs) {
          regs.forEach(function (r) {
            if (r.active && r.active.scriptURL === src) r.unregister();
          });
        }, function () {});
      }
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
