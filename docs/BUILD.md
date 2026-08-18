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

## 2. Pyodide runtime
Put Pyodide 0.25.0 under `lib/pyodide/` (the demo loads `../lib/pyodide/pyodide.js`):

```bash
bash scripts/download-pyodide.sh        # or download pyodide-0.25.0 and extract into lib/pyodide/
```

## 3. Serve (COOP/COEP required)
WgPy's sync bridge needs `SharedArrayBuffer`, which needs cross-origin isolation. Use the
included server (also does HTTP Range, needed for streaming model loads):

```bash
node serve-coi.mjs . 8119
# open http://localhost:8119/webapp/  → tick WebGPU, pick an example, Run
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
