/* webtorch — the response that makes SharedArrayBuffer work.
 *
 * The SDK's GPU backend reaches the device from a worker over shared memory, so it needs
 * SharedArrayBuffer, which needs the document to be cross-origin isolated. That is not a
 * thing a page can arrange at run time: `crossOriginIsolated` is a getter with no setter,
 * decided from the document's own response headers before any script runs. So it cannot be
 * passed to the SDK as a parameter -- it has to already be true.
 *
 * A host that controls its server sets the two headers there and needs none of this. A host
 * on a static site (GitHub Pages and friends) cannot, and its only remaining option is a
 * service worker that adds them to what it proxies.
 *
 * That service worker is the HOST'S: it is registered at the host's scope, it sits in front
 * of the host's own files, and what it caches and how is the host's policy. The SDK has no
 * business owning it. But the one thing in it that is the SDK's -- exactly which response
 * makes SharedArrayBuffer work -- is also the one thing a host cannot write correctly from
 * first principles, so it is here:
 *
 *     importScripts('/webtorch/js/webtorch-coi.js');
 *     self.addEventListener('fetch', (e) => {
 *       e.respondWith(myOwnCachePolicy(e).then(webtorch.isolate));
 *     });
 *
 * The host brings the worker, the scope and the caching. This brings the headers.
 */
(function (root) {
  const wt = root.webtorch || (root.webtorch = {});

  /**
   * The same response, carrying the headers that make SharedArrayBuffer available.
   *
   * Only ADDED when the response has none of its own: a server that sets them is the
   * authority, and re-writing its answer would override a decision that was already made.
   */
  wt.isolate = function (res) {
    if (!res || res.status === 0) return res;              // opaque: nothing to re-wrap
    if (res.headers.get('Cross-Origin-Embedder-Policy')) return res;    // the server decided
    const headers = new Headers(res.headers);
    // require-corp, not credentialless. The two isolate identically where both are known,
    // and credentialless sounds better -- it asks for the isolation without requiring every
    // cross-origin fetch to opt in, which matters when weights come from one host and
    // Pyodide from another. But WebKit does not implement it, so on every browser on an
    // iPhone the header is simply ignored: the document is NOT isolated, SharedArrayBuffer
    // is gone, and the model falls back to the CPU with nothing said. Measured, WebKit,
    // served straight and through a service worker both:
    //   Cross-Origin-Embedder-Policy: require-corp    -> crossOriginIsolated = true
    //   Cross-Origin-Embedder-Policy: credentialless  -> crossOriginIsolated = false
    // What require-corp costs is that every cross-origin response must carry CORS or CORP.
    // For this SDK's own dependencies that is already paid: jsdelivr answers with both,
    // model hosts with CORS, and a CDN script or stylesheet tag asks with `crossorigin`.
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, {
      status: res.status, statusText: res.statusText, headers: headers,
    });
  };

  /** The two headers, for a host that is configuring a server rather than a worker. */
  wt.ISOLATION_HEADERS = {
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Opener-Policy': 'same-origin',
  };
})(typeof self !== 'undefined' ? self : this);
