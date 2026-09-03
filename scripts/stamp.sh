#!/bin/sh
# Stamp a content hash onto every locally-served script URL.
#
# Without it the browser runs a stale copy after a deploy and only picks up the new one on a
# SECOND reload. That is not the service worker's doing -- it revalidates -- it is the
# browser's own memory cache answering a `<script src>` on a reload without ever consulting
# the worker. Measured directly: `fetch('app.js')` returned the new file while the running
# `normalizeMath` was still the old one.
#
# A URL that changes when the bytes change has no stale copy to serve. The service worker
# itself is NOT stamped: it has its own update path, and a changing URL would re-register it
# on every deploy.
#
# Idempotent -- an existing ?v= is replaced, not appended to. Run before committing a change
# to any of these files.
set -e
cd "$(dirname "$0")/.."
h() { shasum -a 1 "$1" | cut -c1-10; }

stamp_html() {                     # file, path-as-written, real-path
  v=$(h "$3")
  perl -0pi -e "s{src=\"\Q$2\E(\?v=[0-9a-f]+)?\"}{src=\"$2?v=$v\"}g" "$1"
}
stamp_worker() {                   # file, worker-file-name, real-path
  v=$(h "$3")
  perl -0pi -e "s{new Worker\('\Q$2\E(\?v=[0-9a-f]+)?'\)}{new Worker('$2?v=$v')}g" "$1"
}

stamp_html chat/index.html "../dist/wgpy-main.js"          dist/wgpy-main.js
stamp_html chat/index.html "../webtorch/js/webtorch-main.js" webtorch/js/webtorch-main.js
stamp_html chat/index.html "zip.js"                        chat/zip.js
stamp_html chat/index.html "app.js"                        chat/app.js
stamp_worker chat/app.js   "worker.js"                     chat/worker.js
stamp_worker chat/app.js   "pyworker.js"                   chat/pyworker.js
# app.js changed if a worker stamp moved, so its own stamp is taken last
stamp_html chat/index.html "app.js"                        chat/app.js

echo "stamped:"
grep -o 'src="[^"]*?v=[0-9a-f]*"' chat/index.html | sed 's/^/  /'
grep -o "new Worker('[^']*')" chat/app.js | sed 's/^/  /'
