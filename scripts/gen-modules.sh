#!/bin/sh
# Regenerate webtorch/modules.json -- the list the browser bootstrap fetches, since a page
# cannot list a directory. Run after adding or removing a module; a missing entry shows up
# only at runtime, as an ImportError from inside Pyodide.
set -e
cd "$(dirname "$0")/.."
{
  printf '{\n "modules": [\n'
  ls webtorch/*.py | sed 's|webtorch/||' | sort | sed 's|.*|  "&",|' | sed '$ s|,$||'
  printf ' ]\n}\n'
} > webtorch/modules.json
echo "webtorch/modules.json: $(grep -c '\.py' webtorch/modules.json) modules"
