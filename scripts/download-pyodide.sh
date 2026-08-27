#!/bin/sh
# Not needed to run this project: the workers load Pyodide from the CDN, and the service
# worker caches it (the version is in the URL, so a bump fetches and caches the new one and
# the old entries are dropped). Keeping a copy in the tree was how the local copy and the
# pinned version came to disagree -- a mismatch that only broke for whoever had no copy.
#
# Use this only to serve the whole thing from your own host, offline from the first load.
# The version lives in chat/pyodide-version.js; keep them the same.
set -e
VERSION="${1:-0.27.7}"
mkdir -p lib
cd lib
curl -OL "https://github.com/pyodide/pyodide/releases/download/${VERSION}/pyodide-${VERSION}.tar.bz2"
tar jxf "pyodide-${VERSION}.tar.bz2"
echo "extracted to lib/pyodide -- point the workers at it with self.PYODIDE_URL"
