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

# The script tag is also the version token for the worker-side SDK package.  Hashing only
# webtorch-main.js leaves a changed Python module at the old URL, so a normal reload can run
# a stale kernel even though the checkout is current.  One digest covers every file loaded
# by webtorch.start(): both worker bootstraps, the module manifest, and all Python modules.
sdk_hash() {
  {
    for f in dist/wgpy-main.js dist/wgpy-worker.js \
             dist/wgpy_webgpu-1.0.0-py3-none-any.whl \
             dist/wgpy_webgl-1.0.0-py3-none-any.whl \
             webtorch/js/webtorch-main.js webtorch/js/webtorch-host.js \
             webtorch/js/webtorch-worker.js webtorch/modules.json; do
      shasum -a 1 "$f"
    done
    find webtorch -maxdepth 1 -type f -name '*.py' -print | LC_ALL=C sort | while IFS= read -r f; do
      shasum -a 1 "$f"
    done
  } | shasum -a 1 | cut -c1-10
}

stamp_html() {                     # file, path-as-written, real-path
  v=$(h "$3")
  perl -0pi -e "s{src=\"\Q$2\E(\?v=[0-9a-f]+)?\"}{src=\"$2?v=$v\"}g" "$1"
}
stamp_html_version() {             # file, path-as-written, already-computed-version
  perl -0pi -e "s{src=\"\Q$2\E(\?v=[0-9a-f]+)?\"}{src=\"$2?v=$3\"}g" "$1"
}
stamp_worker() {                   # file, worker-file-name, real-path
  v=$(h "$3")
  perl -0pi -e "s{new Worker\('\Q$2\E(\?v=[0-9a-f]+)?'\)}{new Worker('$2?v=$v')}g" "$1"
}
# A stylesheet is cached exactly as a script is, and a page whose CSS is one version behind
# is a page whose layout is one version behind. It arrives through `href`, not `src`.
stamp_href() {                     # file, path-as-written, real-path
  v=$(h "$3")
  perl -0pi -e "s{href=\"\Q$2\E(\?v=[0-9a-f]+)?\"}{href=\"$2?v=$v\"}g" "$1"
}

stamp_href chat/index.html "style.css"                     chat/style.css
stamp_html chat/index.html "../dist/wgpy-main.js"          dist/wgpy-main.js
stamp_html_version chat/index.html "../webtorch/js/webtorch-main.js" "$(sdk_hash)"
stamp_html chat/index.html "zip.js"                        chat/zip.js
stamp_html chat/index.html "app.js"                        chat/app.js
stamp_worker chat/app.js   "pyworker.js"                   chat/pyworker.js
# app.js changed if a worker stamp moved, so its own stamp is taken last
stamp_html chat/index.html "app.js"                        chat/app.js

echo "stamped:"
grep -oE '(src|href)="[^"]*\?v=[0-9a-f]*"' chat/index.html | sed 's/^/  /'
grep -o "new Worker('[^']*')" chat/app.js | sed 's/^/  /'
