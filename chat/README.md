# webtorch chat

A ChatGPT-style chat UI that runs models **in your browser** via webtorch (Pyodide + WebGPU).

- **Any model, not a fixed list.** The dropdown is only a set of examples — type any
  ModelScope `org/repo` and file to load it. Downloads always come from ModelScope.
- **Conversations** live in the sidebar (stored locally); model and cache settings are behind
  the ⚙ Settings button. The composer stays locked until a model is ready.
- **Cached.** Model files are stored by the SDK's persistent cache (IndexedDB-backed), so the
  second load is instant. The sidebar lists what is cached and can delete entries.
- **Status.** Loading and download progress are shown while the model streams in.
- **Attachments.** Attach a file, capture from the camera, or pull the text of a URL into a
  message.
- **Tools the model can call.** Python and JavaScript are offered to the model; it writes the
  call, the page runs it, the result goes back for the next turn. The app decides which tools
  exist and implements them — parsing them out of the reply, and constraining the model to
  names that exist, is the SDK's job (`tools=`, `require_known_tools=True`).
- **Code you can run.** A Python block gets a ▶ and runs in a second Pyodide, separate from
  the one holding the model, with output, tracebacks and matplotlib figures inline.
- **Replies that render.** Markdown, highlighted code, LaTeX via KaTeX, tables — sanitised
  before they reach the DOM — and each block is editable in place.
- **Offline.** A service worker keeps the wheels, the wasm and the runtime; the app's own
  files stay network-first, so an update lands as soon as there is a network.
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
The presets run from 0.4 GB to 13.8 GB — a 0.6B for a quick first run, up to full-size 27B
and 30B builds at 3-4 bit. The first load of a large one is a long download; afterwards it
comes from the cache. Split multi-part GGUFs (`...-00001-of-00002.gguf`) are not supported,
so the presets are all single-file builds.

WebGPU is the fast backend. **WebGL is a working fallback, not a broken state** — every
kernel exists there and every quantization format is checked against the reference decoder
on it too; replies are correct and arrive about 5-9× slower, and the page says so. Only when
neither backend is available do the weights fall back to the page's WASM heap, which is
roughly 2 GB of usable room and is the case the page warns about.
