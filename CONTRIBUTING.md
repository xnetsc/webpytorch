# Contributing

Thanks for your interest! webtorch is a PyTorch-compatible SDK that runs in the
browser on WebGPU/WebGL (via Pyodide + a modified WgPy backend).

## Getting set up
- Build the backend + fetch the Pyodide runtime: see [docs/BUILD.md](docs/BUILD.md).
- Serve with the required cross-origin isolation headers: `node serve-coi.mjs . 8119`
  and open `http://localhost:8119/webapp/`.
- Pure-Python SDK changes can be smoke-tested off-browser with numpy
  (`import webtorch` uses a numpy fallback when no GPU backend is present).

## Where things live
- SDK package: `webtorch/` (public API in `__init__.py`; keep the public surface
  **generic / task-level** — model-specific classes stay internal under `webtorch.models.*`).
- Backend kernels (WgPy fork): `src/`, `webgl/`, `webgpu/` — rebuild wheels after Python
  changes (`setup_webgpu.py`/`setup_webgl.py`); rebuild JS (`npm run build`) after TS changes.

## Guidelines
- Public interfaces must be generic or series/task-level (`pipeline`, `AutoModelFor*`,
  `Quantizer`), never a one-off `FooModel` on the public API.
- The library core performs **no IO** — inject IO through the async callbacks in `webio`.
- Verify behavior by exercising it end-to-end (run the model, check the output), not just
  by "it imported". For audio/text, check the actual output (e.g. ASR the waveform).

## PRs
Keep changes focused, describe how you verified them, and update `docs/` when you change
a public interface.
