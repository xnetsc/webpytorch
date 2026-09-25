# webtorch chat

A ChatGPT-style chat UI that runs models **in your browser** via webtorch (Pyodide + WebGPU).

- **Repository-owned model list, not a hard-coded picker.** The page reads
  [`models.json`](models.json) and creates the dropdown from it. Each entry has a display name,
  an optional byte size, an optional complete HTTP URL, and/or a hub repository plus file.
  A custom `org/repo/file` still works without a list entry.
- **Automatic application-side source choice.** When an entry has a complete URL, the page
  checks it first so it is available as a fallback, then measures Hugging Face, ModelScope and
  `hf-mirror.com`. The fastest reachable source wins; if all hubs are absent, the complete URL
  still works. Without a complete URL, no reachable hub means the model cannot be downloaded.
  This is page policy: it installs one of the SDK's existing readers and does not alter SDK
  download behavior. One selected source is pinned for the whole load.
- **Optional metadata stays optional.** An omitted size is not shown. Selecting that entry
  probes its source; when a file server reports a total byte count, the option is updated with
  the measured size. `hash` may be supplied for an immutable publication but is not required.
- **Image decisions when declared.** A decision preset whose `surface()` includes image input
  reveals an image picker. The page sends a typed base64 multimodal state through the same
  `decide(state, questions)` call and reports when the image feature cache was reused.
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
- **A tool round is its own block.** A reply built over several tool calls keeps a blank line
  between the rounds, so a round that ends in a code fence and one that opens with another
  cannot merge into a token that is neither. That newline is the only thing the app puts
  between two generations; neither round's own text is touched.
- **Rendering never edits the record.** A reply that is nothing but one ```` ```markdown ````
  fence is displayed unwrapped — a model handing back "the markdown" inside a code block meant
  the markdown, not a picture of it — but the message keeps exactly what the model produced.
  When what is shown differs from what is stored, per-block editing is not offered: those
  blocks are addressed by index into the stored text, and offering an editor whose indices
  point somewhere else is worse than offering none.
- **Enter belongs to the input method first.** Typing Chinese, Japanese or Korean assembles
  characters in the field and Enter is how the input method accepts them — so Enter does not
  send while a composition is open, and Escape does not close an editor around one.
- **Read-only until it can act.** While the runtime is starting, no model is loaded, or a
  reply is being written, the conversation stops offering to be changed — the same states in
  which the composer is already flat. A question rewritten with no way to answer it leaves a
  reply to something the question no longer says.
- **Change a question, get a new answer.** Editing one of your own messages answers it again,
  replacing the reply that was under it — an answer to a question that no longer says the same
  thing is worse than no answer. The old one leaves the screen at once, but it is not written
  over until a token of the new one actually arrives: a run that fails, or a page that is
  closed mid-answer, leaves the answer you already had. Later turns are left alone.
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

## Model list

The minimum useful entry is a `name` plus either `repo` or `url`. `file` identifies a
single-file hub model; without it the page probes `probe` (default `config.json`). A complete
`url` points to that same file or probe document, not merely to a website:

```json
{
  "name": "Example Q4",
  "repo": "group/example-GGUF",
  "file": "example-Q4_K_M.gguf",
  "url": "https://downloads.example.net/example-Q4_K_M.gguf",
  "size": 123456789
}
```

`size` and `url` are optional. The current Vision entry points to a same-origin Pages path.
The Pages deployment obtains the intact ONNX files from this repository's Release and places
them at that path; the weights are not committed to Git and are not split.

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
