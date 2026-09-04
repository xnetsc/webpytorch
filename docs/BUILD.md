# Build & run

webtorch runs in the browser. You need to (1) build the WgPy backend wheels, (2) fetch the
Pyodide runtime, (3) serve with cross-origin isolation, and (4) provide model weights for the
examples that use them.

## Prerequisites
- Node.js (for the TS build + dev server) and Python 3.11.
- A browser with WebGPU (Chrome/Edge 113+) — WebGL works as a fallback.

## 1. Backend wheels (WgPy fork)
Homebrew/system Python is often PEP-668 "externally managed", so build wheels in a venv:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install wheel setuptools numpy
# the npm build scripts call `python`; run the setup scripts with the venv python directly:
python setup_webgpu.py bdist_wheel      # -> dist/wgpy_webgpu-1.0.0-py3-none-any.whl
python setup_webgl.py  bdist_wheel      # -> dist/wgpy_webgl-1.0.0-py3-none-any.whl
```

Rebuild the JS bundle after TypeScript changes (`src/`):

```bash
npm install
npm run build            # -> dist/wgpy-worker.js etc.
```

> The Python backend kernels (`webgl/`, `webgpu/`) are WGSL/GLSL strings added at runtime,
> so **Python-only** kernel edits need only the wheel rebuilt — no JS/webpack rebuild.

For a Python-only edit there is a shortcut that needs no venv and no build deps: it swaps
the changed files into the existing wheels and recomputes `RECORD`, leaving everything else
byte for byte as `bdist_wheel` produced it. A wheel whose sources have not changed comes out
identical, so it shows as no diff.

```bash
python3 scripts/pack-wheels.py            # rebuild both from webgpu/ and webgl/
python3 scripts/pack-wheels.py --check    # report staleness only, exit 1 if stale
```

Use the venv build above when anything about the packaging itself changes — a new module,
metadata, dependencies. The shortcut only replaces files the wheel already contains.

### The wheels are checked in, so they can disagree with the tree

`dist/*.whl` are build artifacts kept in the repo, because the page installs them with
micropip and a static host has nothing to build with. Editing `webgpu/` and committing
therefore ships nothing: the tree says one thing and the artifact the browser runs says
another, silently. A hook refuses that commit —

```bash
git config core.hooksPath .githooks       # once per clone
```

— and `git commit --no-verify` gets past it when the mismatch is deliberate.

`.githooks/` holds one check per file and runs them all: `check-wheels.py` is the above,
`check-docs.py` refuses a commit whose `docs/API.md` contradicts the signatures it
documents. See [../CONTRIBUTING.md](../CONTRIBUTING.md).

### Two generated files that are easy to forget

Neither is built by npm or by the wheel scripts, and skipping either fails at runtime rather
than at build time:

```bash
scripts/gen-modules.sh    # after adding/removing a webtorch/*.py
scripts/stamp.sh          # after changing anything the page loads from chat/
```

`webtorch/modules.json` is the list the browser bootstrap fetches, because a page cannot list
a directory — a module missing from it surfaces as an ImportError from inside Pyodide.
`stamp.sh` puts a content hash on each local script URL (`…/webtorch-main.js?v=01f205aa85`):
without it the browser's own memory cache answers `<script src>` with the previous file on
the first reload after a deploy, and the service worker never sees the request.

## 2. Pyodide runtime
Pyodide is NOT vendored: the workers load it from the CDN, pinned in one place
(`chat/pyodide-version.js`), and the service worker caches it — the version is in the
URL, so changing that file is all an upgrade takes. `scripts/download-pyodide.sh` is
only for serving it from your own host.

```bash
bash scripts/download-pyodide.sh        # optional: serve Pyodide from your own host
```

## 3. Serve (COOP/COEP required)
WgPy's sync bridge needs `SharedArrayBuffer`, which needs cross-origin isolation. Use the
included server (also does HTTP Range, needed for streaming model loads):

```bash
node serve-coi.mjs . 8119
# open http://localhost:8119/chat/     → the chat client: pick a model, wait for the load
# or  http://localhost:8119/webapp/   → the example runner: tick WebGPU, pick an example, Run
```

## 4. Model weights (`models/`, git-ignored)
Examples that run real models expect served weight files under `models/`. They are large and
not committed. Prepare them with the scripts in `tools/` (which download source models from
Hugging Face / ModelScope and pack served `.npz`/`.onnx`), e.g.:

- `tools/prep_vits.py`, `tools/prep_detr.py` — pack the VITS / DETR web weights.
- CosyVoice2 / Qwen packing is documented inline in the corresponding `tools/` and example
  scripts; quantize any fp16 HF LLM to int4 with `webtorch.Quantizer.quantize(...)`.

Point an example at your files, or drop them at the paths the example expects (`/models/...`).

## Off-browser smoke tests
Pure-Python SDK logic can be exercised on the host (numpy fallback), e.g.:

```bash
python -c "import webtorch; print(webtorch.__version__, webtorch.__all__)"
```
GPU-only paths (int4 kernels, capture) only run in the browser.
