# Contributing

Thanks for your interest! webtorch is a PyTorch-compatible SDK that runs in the
browser on WebGPU/WebGL (via Pyodide + a modified WgPy backend).

## Getting set up
- Build the backend + fetch the Pyodide runtime: see [docs/BUILD.md](docs/BUILD.md).
- Serve with the required cross-origin isolation headers: `node serve-coi.mjs . 8119`
  and open `http://localhost:8119/chat/` (the chat client) or `/webapp/` (the example runner).
- Pure-Python SDK changes can be smoke-tested off-browser with numpy
  (`import webtorch` uses a numpy fallback when no GPU backend is present).

## Where things live
- SDK package: `webtorch/` (public API in `__init__.py`; keep the public surface
  **generic / task-level** — model-specific classes stay internal under `webtorch.models.*`).
- Backend kernels (WgPy fork): `src/`, `webgl/`, `webgpu/` — rebuild wheels after Python
  changes (`setup_webgpu.py`/`setup_webgl.py`); rebuild JS (`npm run build`) after TS changes.
- **Adding or removing a `webtorch/*.py` — run `scripts/gen-modules.sh`.** A page cannot list
  a directory, so the browser bootstrap fetches `webtorch/modules.json`. Forget it and the
  module simply is not there, and you learn about it as an ImportError from inside Pyodide.
- **Changing anything under `chat/` — run `scripts/stamp.sh`.** It puts a content hash on
  each local script URL. Without it the browser's own memory cache answers `<script src>`
  with the old file on the first reload after a deploy; the service worker never sees the
  request, so it cannot help.

## Guidelines
- Public interfaces must be generic or series/task-level (`pipeline`, `AutoModelFor*`,
  `Quantizer`), never a one-off `FooModel` on the public API.
- **All functional code belongs in `webtorch/`.** Outside it there is UI and orchestration
  only. The chat app is a *user* of this SDK: it decides which tools to offer and implements
  them, while parsing, constraining and rendering them is the SDK's. If you find yourself
  writing model logic in `chat/`, the API it needs is the missing piece.
- **A caller must not need to know model internals.** Probing a template, rendering a call,
  constraining a decode — each becomes an API or a parameter to one. Parameters are named in
  the caller's language, not the implementation's: `require_known_tools=True`, not
  `constraint="tool_names"`.
- The library core performs **no IO** — inject IO through the async callbacks in `webio`.
- Verify behavior by exercising it end-to-end (run the model, check the output), not just
  by "it imported". For audio/text, check the actual output (e.g. ASR the waveform).

## If you touch a kernel

Three failure modes here are silent, and all three have cost real days:

- **A WGSL compile error does not raise.** The kernel returns zeros and everything downstream
  looks alive. Gate every new kernel on a numerical self-check against the reference
  (`_ggml_selfcheck` is the pattern) *before* you time it.
- **Timing one dispatch measures the readback.** A sync costs 1–2 ms on this stack and a
  decode matmul tens of microseconds. Batch ~24 dispatches per sync, or you rank your
  candidates by noise — that mistake once took a 0.6B from 104 tok/s to 64.
- **Runs drift within a session.** The same configuration measured 7.61 ms and 4.49 ms when
  candidates were timed in blocks. Interleave them and take medians. For an end-to-end
  number, reload the page: changing a constant in a live session has produced a 31% "speedup"
  that was really a 5% slowdown.

If a knob measures as no help, say so *next to the constant* and leave it fixed. Making it
dynamic anyway costs a tuning pass at every load and buys nothing.

## Documentation ships with the change

A document does not go wrong by getting old. It goes wrong when it **asserts something
about the code that is no longer true** — and a reader who follows it is then worse off
than if it had said nothing. So a change and the document it makes false belong in the
same commit.

"Touched code, therefore touch a document" is not that rule: it is satisfied by editing a
space, and it stops commits that genuinely need no documentation change. The rule is
narrower and harder — *no document may state a fact about the code that your commit makes
false* — and two parts of it are checked for you by `.githooks/check-docs.py`:

- a signature written in `docs/API.md` as a definition entry must agree with the real one
  (usage examples mid-sentence are not definitions, and are ignored);
- a name your commit **adds** to `webtorch.__all__` must appear in `docs/API.md`.

Those are refused. Everything else — what a backend does, how fast something is, which
formats are supported — cannot be checked against source, so when the public surface moves
and no document does, the hook names what changed and lets you through. That judgement is
yours. If a number you are changing appears in a document, re-measure it under the method
that document states rather than adjusting it to taste.

## PRs
Keep changes focused and describe how you verified them. Enable the hooks once per clone —
`git config core.hooksPath .githooks` — and `--no-verify` when a mismatch is deliberate.
