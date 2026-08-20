# webtorch chat

A ChatGPT-style chat UI that runs models **in your browser** via webtorch (Pyodide + WebGPU).

- **Any model, not a fixed list.** The dropdown is only a set of examples — type any
  ModelScope `org/repo` and file to load it. Downloads always come from ModelScope.
- **Cached.** Model files are stored by the SDK's persistent cache (IndexedDB-backed), so the
  second load is instant. The sidebar lists what is cached and can delete entries.
- **Status.** Loading and download progress are shown while the model streams in.
- **Tools.** Attach a file, capture from the camera, or pull the text of a URL into a message.
- **Export / import.** Conversations save as a real `.zip` containing `chat.json`.
- **Release.** Frees the model's memory so another can be loaded.

## Running it
The page needs the built runtime assets, which are not committed (they are large):
`lib/pyodide/` and `dist/` — see [../docs/BUILD.md](../docs/BUILD.md). Then serve the repo with
cross-origin isolation and open `/chat/`:

```bash
node serve-coi.mjs . 8119     # COOP/COEP + HTTP Range
# http://localhost:8119/chat/
```

## What actually fits in a browser
A browser tab is not a workstation. Practical guidance:

| model | 4-bit size | in a browser |
|---|---|---|
| Qwen3-4B-Instruct | ~2.5 GB | works |
| Qwen3-8B | ~5 GB | works on a machine with headroom |
| Qwen3-30B-A3B (MoE) | ~14 GB | download and memory are a real obstacle |
| Qwen3.8-27B | ~13 GB | **not yet runnable** — its blocks are state-space /
  gated-linear-attention, which this engine does not implement yet; the loader says so
  up front instead of downloading first |

Everything runs on the CPU path unless WebGPU is available, in which case the int4/int8
kernels are used.
