# webtorch chat

A ChatGPT-style chat UI that runs models **in your browser** via webtorch (Pyodide + WebGPU).

- **Any model, not a fixed list.** The dropdown is only a set of examples — type any
  ModelScope `org/repo` and file to load it. Downloads always come from ModelScope.
- **Conversations** live in the sidebar (stored locally); model and cache settings are behind
  the ⚙ Settings button. The composer stays locked until a model is ready.
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
Every preset is a full-size model (11-14 GB at 3-4 bit) — small models are not worth the
plumbing. The first load of one is a long download; afterwards it comes from the cache.
Split multi-part GGUFs (`...-00001-of-00002.gguf`) are not supported yet, so the presets are
all single-file builds.

Everything runs on the CPU path unless WebGPU is available, in which case the int4/int8
kernels are used.
