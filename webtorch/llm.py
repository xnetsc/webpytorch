"""webtorch.llm -- LLM inference engine with graph-capture acceleration.

High-level API for running LLMs in the browser on top of webtorch/WgPy — AutoGPTQ
safetensors or llama.cpp GGUF, at int4 / int8 / fp16 — with a WebGPU graph-capture fast
path for decode.

    import llm
    model = await llm.GPTQModel.load("/models/qwen7b-gptq")
    out = model.generate("Give me three tips for staying focused.")
    print(out.text, "|", out.decode_tok_s, "tok/s")

What it does:
  * streams an AutoGPTQ safetensors model (single- or multi-shard) tensor-by-
    tensor via HTTP Range, quantizes embed/lm_head to int4 on the GPU (so the
    >1GB vocab matrices never sit in the WASM heap);
  * ChatML prompt formatting + a byte-level BPE tokenizer;
  * a fixed-capacity in-place KV cache written by a scatter kernel, so the whole
    decode step is shape-static and can be graph-captured;
  * WebGPU: captures the decode step once and replays it per token (removes the
    per-op Python dispatch overhead -- ~20x faster decode). WebGL falls back to a
    correct, un-captured growing-cache path (WebGL can't do in-place capture).

Nothing here is tied to a model family. Shapes, head dims, rope, MoE and hybrid
(state-space / linear-attention) blocks all come from the model's own config or GGUF
metadata, optional pieces (QK-norm, fused QKV, fused Q+output-gate, sparse MoE, recurrent
blocks) are detected from the tensors the file actually contains, and the tokenizer's special
tokens and chat layout are read from the vocabulary the model ships.
"""
import json, math, time
import numpy as np
from . import _core as wt

xp = wt.xp




# ----------------------------- tokenizer ------------------------------------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# Chat formats, keyed by a marker token the MODEL's own vocabulary declares. This is discovery,
# not a model whitelist: whichever marker a tokenizer actually contains selects the format, so
# a model family the SDK has never heard of still works as long as it uses one of these
# conventions (and falls back to a plain transcript otherwise).
_CHAT_FORMATS = (
    ("<|im_start|>", "chatml"),            # Qwen, Yi, and many others
    ("<|start_header_id|>", "llama3"),     # Llama 3.x
    ("<start_of_turn>", "gemma"),          # Gemma
    ("[INST]", "mistral"),                 # Mistral / Mixtral
)
# Tokens that end a turn, in the order we prefer them. Again: whichever the model HAS.
_EOT_CANDIDATES = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "</s>",
                   "<|endoftext|>", "<|end_of_text|>")


def _tpl_raise(msg):
    raise ValueError(msg)


# "not asked yet", distinct from "asked, and this model has no call format".
def _matmul_row_align():
    """Rows the batched matmul wants its input to be a multiple of, or 1 when the fast path
    is not in use. Asked of the backend rather than written down twice."""
    try:
        return wt._matmul_row_align()
    except Exception:
        return 1


_UNSET = object()


def pretok_style(tj):
    """How a `tokenizer.json` cuts text up, when it does not do it with a regex.

    Two families of BPE are in circulation and they are not interchangeable. Byte-level BPE
    splits with a regex and maps bytes through a printable alphabet. The other -- what
    SentencePiece-trained vocabularies use -- replaces every space with one marker character,
    prepends it, and splits there; its merges are over CHARACTERS, not mapped bytes, and a
    character outside the vocabulary falls back to its bytes as `<0xAB>` tokens.

    Reading a Metaspace file as byte-level does not fail. It returns different tokens: on a
    256k multilingual vocabulary, measured, "Hello world" came out as three tokens instead of
    two and eight Chinese characters as twenty-four instead of six. That is a different input
    to the model, silently.

    Returns `{"mark", "prepend"}` for that family, or None for byte-level.
    """
    def walk(node):
        if not isinstance(node, dict):
            return None
        if node.get("type") == "Sequence":
            for sub_node in (node.get("pretokenizers") or []):
                got = walk(sub_node)
                if got is not None:
                    return got
            return None
        if node.get("type") == "Metaspace":
            return {"mark": node.get("replacement") or "\u2581",
                    "prepend": node.get("prepend_scheme", "always") != "never"}
        return None
    got = walk((tj or {}).get("pre_tokenizer"))
    if got is not None:
        return got
    # Some files carry no Metaspace pre-tokenizer and do the same job in the normalizer.
    n = (tj or {}).get("normalizer") or {}
    if n.get("type") == "Replace" and (n.get("pattern") or {}).get("String") == " ":
        return {"mark": n.get("content") or "\u2581", "prepend": True}
    return None


def pretok_pattern(tj):
    """The pre-tokenization pattern a `tokenizer.json` asks for, or None when it says nothing.

    Two shapes appear. A `Split` pre-tokenizer carries the pattern as a literal string, which
    is what the later GPT-4/Qwen-style files do -- that string is the answer, whatever it says.
    A bare `ByteLevel` with `use_regex` carries no string: in the `tokenizers` library that
    node IS the GPT-2 pattern, so a file that names it is asking for GPT-2 even though it
    never spells it out. `use_regex: false` means no splitting at all.

    Read rather than inferred from the model: two models from the same family can ship
    different tokenizers, and the file is the only thing that knows which."""
    def walk(node):
        if not isinstance(node, dict):
            return None
        t = node.get("type")
        if t == "Sequence":
            for sub_node in (node.get("pretokenizers") or []):
                got = walk(sub_node)
                if got is not None:
                    return got
            return None
        if t == "Split":
            pat = (node.get("pattern") or {})
            return pat.get("Regex") or pat.get("String") or None
        if t == "ByteLevel":
            return BPETokenizer.PRETOK_GPT2 if node.get("use_regex", True) else None
        return None
    return walk((tj or {}).get("pre_tokenizer"))


class BPETokenizer:
    """Byte-level BPE (vocab + merges), with the special tokens and chat format discovered
    from the model itself.

    Nothing here is tied to a model family: the special-token ids come from the vocabulary the
    model ships, the end-of-turn id from the model's own metadata (or the first end-of-turn
    marker its vocabulary defines), and the chat layout from whichever marker token the
    vocabulary contains. A model using none of the known conventions still works through a
    plain `role: content` transcript."""

    # How the text is cut up BEFORE merges are applied. Two are in circulation and they
    # disagree about digits and about a space before one: GPT-2 takes " 3" as one piece
    # (` ?\p{N}+`), the later GPT-4/Qwen pattern takes digits one at a time and leaves the
    # space to its own rule. Picking the wrong one does not fail -- it silently produces a
    # different, longer token sequence, which is a different input to the model.
    #
    # Which one a model wants is stated in its `tokenizer.json`, so it is read from there
    # (see `pretok_pattern`) rather than assumed.
    PRETOK_GPT2 = (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"
                   r"|\s+(?!\S)|\s+")
    PRETOK_GPT4 = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                   r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

    def __init__(self, vocab, merges, eos_ids=None, chat_format=None,
                 chat_template=None, control=None, pattern=None, style=None):
        self.enc = vocab
        self._callfmt = _UNSET          # see `tool_call_format`; derived once, on demand
        self._toolshape = _UNSET        # see `tools_shape`
        self._resfmt = _UNSET           # see `tool_result_format`
        # A model ships its own prompt layout. Probing the vocabulary for a known marker
        # gets the family right but not the details -- Qwen3 opens its turn with <think>,
        # which no amount of "this looks like ChatML" will tell you, and without it the
        # model answers by completing the template and stopping. So when the file carries a
        # chat template, that template is the source of truth; the probe is the fallback.
        self.chat_template = chat_template
        self._tpl = None
        # Tokens BPE must never split. From the file's own token_type where it has one --
        # anything else is guesswork about what "looks special".
        self.control = sorted((t for t in (control or ()) if t in vocab), key=len, reverse=True)
        self.dec = {i: t for t, i in vocab.items()}
        # discover specials that this vocabulary actually defines
        self.SPECIALS = {t: vocab[t] for t in
                         ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|eot_id|>",
                          "<|start_header_id|>", "<|end_header_id|>", "<start_of_turn>",
                          "<end_of_turn>", "<|end_of_text|>", "<|end|>", "<s>", "</s>")
                         if t in vocab}
        self.chat_format = chat_format or next(
            (f for tok, f in _CHAT_FORMATS if tok in vocab), "plain")
        eos = [int(i) for i in (eos_ids or []) if i is not None]
        eos += [vocab[t] for t in _EOT_CANDIDATES if t in vocab]
        # de-duplicate, keep order; may be empty -> generation then stops at max_new
        self.eos_ids = list(dict.fromkeys(eos))
        self.eot = self.eos_ids[0] if self.eos_ids else -1
        self.b2u = _bytes_to_unicode(); self.u2b = {v: k for k, v in self.b2u.items()}
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        try:
            import regex as _re
            pat = pattern or self.PRETOK_GPT4
        except Exception:
            # `re` has no \p{...}; this is the ASCII reading of whichever pattern was asked
            # for, and it differs from it outside ASCII.
            import re as _re
            pat = (r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
                   if (pattern or self.PRETOK_GPT4) == self.PRETOK_GPT2 else
                   r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+|\s+")
        self.re = _re; self.pat = _re.compile(pat)
        # `style` is the Metaspace description from `pretok_style`, or None for byte-level.
        # It is read from the model's own file, never assumed: the two families produce
        # different token sequences for the same text and nothing about the vocabulary says
        # which one it was trained with.
        self.style = style or None
        self.mark = (style or {}).get("mark", "\u2581")
        self.prepend = bool((style or {}).get("prepend", True))
        for t, i in self.SPECIALS.items():
            self.dec[i] = t

    def _bpe(self, tok):
        word = list(tok)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]): i for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks:
                break
            i = pairs[best]; word = word[:i] + [best[0] + best[1]] + word[i + 2:]
        return word

    def _encode_metaspace(self, text):
        """Space becomes the marker, the marker starts every piece, merges run over
        characters, and anything the vocabulary does not have falls back to its bytes."""
        t = str(text).replace(" ", self.mark)
        if self.prepend and not t.startswith(self.mark):
            t = self.mark + t
        pieces, cur = [], ""
        for ch in t:
            if ch == self.mark and cur:
                pieces.append(cur); cur = ch
            else:
                cur += ch
        if cur:
            pieces.append(cur)
        ids = []
        for piece in pieces:
            for sym in self._bpe(piece):
                i = self.enc.get(sym)
                if i is not None:
                    ids.append(i)
                    continue
                for b in sym.encode("utf-8"):          # byte fallback, as the file spells it
                    j = self.enc.get("<0x%02X>" % b)
                    if j is not None:
                        ids.append(j)
        return ids

    def encode(self, text):
        if self.style:
            return self._encode_metaspace(text)
        ids = []
        for chunk in self.re.findall(self.pat, text):
            s = "".join(self.b2u[b] for b in chunk.encode("utf-8"))
            for p in self._bpe(s):
                i = self.enc.get(p)
                if i is None:                     # incomplete vocab: fall back to its bytes
                    for ch in p:
                        j = self.enc.get(ch)
                        if j is not None: ids.append(j)
                else:
                    ids.append(i)
        return ids

    def _sp(self, name):
        return self.SPECIALS.get(name)

    async def prepare_template(self):
        """Compile the model's chat template, if it has one. Async because jinja2 may still be
        arriving: the worker starts fetching it at boot (`__webtorch_jinja2`), in the
        background, and this waits for that fetch rather than starting one. Resolving it here
        with micropip instead -- what this did before -- was a network round trip inside the
        load: 3.4 s to 20 s of a 0.6B's load. Rendering itself stays sync."""
        if self._tpl is not None or not self.chat_template:
            return self._tpl
        try:
            try:
                import jinja2
            except ImportError:
                pending = None
                try:
                    import js
                    pending = getattr(js.self, "__webtorch_jinja2", None)
                except Exception:
                    pending = None
                if pending is not None:
                    await pending
                try:
                    import jinja2
                except ImportError:
                    import micropip
                    await micropip.install("jinja2")
                    import jinja2
            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = _tpl_raise
            env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
            self._tpl = env.from_string(self.chat_template)
        except Exception as e:                     # no jinja2, or a template we cannot parse
            self._tpl = False
            print("webtorch: chat template unavailable (%s); using the detected format" % e)
        return self._tpl

    def encode_special(self, text):
        """Encode text, emitting each control token as its own id instead of BPE-ing it."""
        if not self.control:
            return self.encode(text)
        parts = self.re.split("(" + "|".join(self.re.escape(t) for t in self.control) + ")", text)
        out = []
        for p in parts:
            if not p:
                continue
            i = self.enc.get(p)
            out.append(i) if (i is not None and p in self.control) else out.extend(self.encode(p))
        return out

    def render_chat(self, messages, add_generation_prompt=True, **kw):
        """The model's own template applied to a message list -> token ids, or None if the
        model has no usable template."""
        if not self._tpl:
            return None
        try:
            txt = self._tpl.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                   bos_token=self._tok_str(self.SPECIALS.get("<s>")),
                                   eos_token=self._tok_str(self.eot), **kw)
        except Exception as e:
            print("webtorch: chat template failed to render (%s); using the detected format" % e)
            self._tpl = False
            return None
        return self.encode_special(txt)

    def render_chat_text(self, messages, add_generation_prompt=True, **kw):
        """The model's own template applied to a message list -> TEXT, or None if it has
        no usable template. `render_chat` is this plus tokenisation."""
        if not self._tpl:
            return None
        try:
            return self._tpl.render(
                messages=messages, add_generation_prompt=add_generation_prompt,
                bos_token=self._tok_str(self.SPECIALS.get("<s>")),
                eos_token=self._tok_str(self.eot), **kw)
        except Exception:
            return None

    def tool_call_format(self, tools=None):
        """How THIS model writes a tool call: {"open", "close", "payload"}, or None.

        Read from the template rather than assumed. The template is what turns `tool_calls`
        into text, so rendering one whose name and arguments are known markers and reading
        back what surrounds them gives the exact delimiters this model emits -- guessing that
        everyone writes <tool_call> would be one family's convention stated as a standard.

        `payload` is "flat" or "nested" for a JSON call and "xml" for the
        <function=NAME><parameter=K> form some templates use instead.
        """
        if self._callfmt is not _UNSET:
            return self._callfmt
        self._callfmt = None
        N, A, V = "zzprobenamezz", "zzprobeargzz", "zzprobevalzz"
        rc = self.render_chat_text(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"id": "zzprobeidzz", "type": "function",
                              "function": {"name": N, "arguments": {A: V}}}]}],
            tools=tools)
        if not rc:
            return None
        i = rc.find(N)
        if i < 0:
            return None
        NL = chr(10)

        def _lines_before(upto):
            xs = [x for x in rc[:upto].split(NL) if x.strip()]
            return xs[-1] if xs else ""

        def _line_after(frm):
            xs = [x for x in rc[frm:].split(NL) if x.strip()]
            return _drop_eot(xs[0] if xs else "")

        def _drop_eot(line):
            """The closing delimiter, without the end-of-turn marker that follows it.

            A template ends the assistant turn on the same line it closes the call, so the
            raw text gives `</tool_call><|im_end|>` -- but the model's OUTPUT never contains
            the turn marker, so a delimiter carrying it matches nothing and the markup is
            left in the reply for a reader to see. (Reading this back through decode() hid
            the problem by dropping special tokens, which is why it did not show until the
            text was read directly.)"""
            for t in (self._tok_str(self.eot), self._tok_str(self.SPECIALS.get("<s>"))):
                if t and line.endswith(t):
                    line = line[:-len(t)]
            return line.rstrip()

        # The payload is the object that CONTAINS the name, and it is the WIDEST such object
        # that parses on its own -- not the nearest brace. A template that renders its tool
        # DEFINITIONS as JSON earlier in the prompt puts a brace before the call that closes
        # before the call begins; taking that one reported a fragment of the definitions as
        # the delimiters, after which nothing the model wrote ever matched. Parsing is what
        # stops the search going too far: a span reaching back into the definitions has prose
        # in it and does not parse.
        st = en = -1
        cand = rc.rfind("{", 0, i)
        while cand >= 0:
            d, e = 0, -1
            for q in range(cand, len(rc)):
                if rc[q] == "{":
                    d += 1
                elif rc[q] == "}":
                    d -= 1
                    if d == 0:
                        e = q + 1
                        break
            if e > i:
                try:
                    json.loads(rc[cand:e])
                    st, en = cand, e
                except Exception:
                    pass
            cand = rc.rfind("{", 0, cand)
        if en > 0:
            try:
                obj = json.loads(rc[st:en])
                payload = "nested" if ("function" in obj and "name" not in obj) else "flat"
            except Exception:
                payload = None
            self._callfmt = {"open": _lines_before(st), "close": _line_after(en),
                             "payload": payload}
            return self._callfmt
        # No JSON payload: several templates write the call as XML instead. The name is
        # present in that form too, so this cannot hang off "name not found" -- it hangs off
        # "no payload was read".
        fx = rc.rfind("<function=", 0, i + len(N))
        if fx >= 0:
            fe = rc.find("</function>", fx)
            self._callfmt = {"open": _lines_before(fx),
                             "close": _line_after(fe + len("</function>")) if fe >= 0 else "",
                             "payload": "xml"}
        return self._callfmt

    def tools_shape(self):
        """Which shape of tool DEFINITION this model's template actually renders --
        "nested" ({"type":"function","function":{...}}), "flat" ({...}), or None when the
        template ignores tools altogether. Asked of the template, never assumed.

        The name alone is not enough to say a shape worked: a template can mention a tool
        and drop its arguments, which produces calls with no parameters.
        """
        if self._toolshape is not _UNSET:
            return self._toolshape
        self._toolshape = None
        msgs = [{"role": "user", "content": "hi"}]
        plain = self.render_chat_text(msgs)
        if plain is None:
            return None
        NAME, ARG = "zzprobetoolzz", "zzprobeargzz"
        fn = {"name": NAME, "description": "probe",
              "parameters": {"type": "object",
                             "properties": {ARG: {"type": "string", "description": "probe"}},
                             "required": [ARG]}}
        for label, tools in (("nested", [{"type": "function", "function": fn}]),
                             ("flat", [dict(fn)])):
            txt = self.render_chat_text(msgs, tools=tools)
            if not txt or txt == plain:
                continue                       # the template ignored them entirely
            if NAME in txt and ARG in txt:
                self._toolshape = label
                break
        return self._toolshape

    def tool_result_format(self):
        """How a tool RESULT reaches this model: {"via", "keeps_name", "call_id_shape",
        "id_field"}. Discovered from the template, because a dropped result is SILENT -- the
        model answers without ever seeing what the tool returned.

        A template may define its own structure for a result, may render it as an ordinary
        turn, or may drop a role it does not know. `call_id_shape` and `id_field` are
        independent facts: whether the template RENDERS an id given on a call, and whether it
        READS one back on the result. A template that does neither ties results to calls by
        order alone.
        """
        if self._resfmt is not _UNSET:
            return self._resfmt
        out = {"via": None, "keeps_name": False, "call_id_shape": None, "id_field": None}
        self._resfmt = out
        if not self._tpl:
            return out
        MK, TN = "zzresultmarkzz", "zztoolnamezz"
        base = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "calling"}]

        def r(msgs):
            return self.render_chat_text(msgs) or ""

        as_tool = r(base + [{"role": "tool", "name": TN, "content": MK}])
        if MK in as_tool:
            out["via"] = "tool"
            out["keeps_name"] = TN in as_tool
        elif MK in r(base + [{"role": "user", "content": MK}]):
            out["via"] = "user"
        if out["via"] != "tool":
            return out
        CID = "zzcallidzz"
        for shape, tcalls in (
                ("nested", [{"id": CID, "type": "function",
                             "function": {"name": TN, "arguments": {}}}]),
                ("flat", [{"id": CID, "name": TN, "arguments": {}}])):
            amsg = dict(base[-1]); amsg["tool_calls"] = tcalls
            if CID in r(base[:-1] + [amsg]):
                out["call_id_shape"] = shape
                break
        if out["call_id_shape"] is not None:
            for field, val in (("tool_call_id", "zzidtczz"), ("call_id", "zzidcizz"),
                               ("id", "zzididzz")):
                m = {"role": "tool", "content": MK, field: val}
                if val in r(base + [m]):
                    out["id_field"] = field
                    break
        return out

    def _tok_str(self, i):
        return self.dec.get(int(i), "") if i is not None and int(i) >= 0 else ""

    def encode_chat(self, user=None, system="You are a helpful assistant.", messages=None,
                    tools=None, **tpl_kw):
        """Render a chat prompt: the model's own template when it has one, else whatever
        format its vocabulary indicates.

        `messages` is a full conversation ([{role, content}, ...]); `user`/`system` are the
        one-turn shorthand for it. `tools` and any other keyword are handed to the template
        unchanged, which is what makes tool calling, thinking toggles and the rest generic:
        a model that documents them in its template gets them, and one that does not
        ignores them. Nothing here knows what any particular model calls its options."""
        msgs = messages if messages is not None else (
            ([{"role": "system", "content": system}] if system else [])
            + ([{"role": "user", "content": user}] if user is not None else []))
        if tools:
            from . import toolcall
            tools = toolcall.as_shape(tools, self.tools_shape())
        got = self.render_chat(msgs, tools=tools, **tpl_kw)
        if got is not None:
            return got
        # No template: fall back to the layout the vocabulary implies. Tools cannot be
        # expressed without one, so say so rather than dropping them silently.
        if tools:
            raise ValueError("this model ships no chat template, so tool definitions cannot "
                             "be rendered; pass them inside a message instead")
        f = self.chat_format
        sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
        turns = [m for m in msgs if m["role"] != "system"]
        if f == "chatml":
            ims, ime = self._sp("<|im_start|>"), self._sp("<|im_end|>")
            seq = []
            for m in msgs:
                seq += ([ims] + self.encode(m["role"] + "\n" + m["content"]) + [ime]
                        + self.encode("\n"))
            return seq + [ims] + self.encode("assistant\n")
        if f == "llama3":
            sh, eh = self._sp("<|start_header_id|>"), self._sp("<|end_header_id|>")
            eot = self._sp("<|eot_id|>")
            seq = []
            bos = self._sp("<|begin_of_text|>")
            if bos is not None: seq.append(bos)
            for m in msgs:
                seq += ([sh] + self.encode(m["role"]) + [eh]
                        + self.encode("\n\n" + m["content"]) + [eot])
            return seq + [sh] + self.encode("assistant") + [eh] + self.encode("\n\n")
        if f == "gemma":
            sot, eot = self._sp("<start_of_turn>"), self._sp("<end_of_turn>")
            seq = []
            for i, m in enumerate(turns):     # Gemma has no system role: fold it into turn 1
                body = m["content"]
                if i == 0 and sys_txt:
                    body = sys_txt + "\n\n" + body
                role = "model" if m["role"] == "assistant" else "user"
                seq += [sot] + self.encode(role + "\n" + body) + [eot] + self.encode("\n")
            return seq + [sot] + self.encode("model\n")
        if f == "mistral":
            bos = self._sp("<s>")
            seq = [bos] if bos is not None else []
            for i, m in enumerate(turns):
                body = m["content"]
                if i == 0 and sys_txt:
                    body = sys_txt + "\n\n" + body
                seq += (self.encode("[INST] " + body + " [/INST]") if m["role"] != "assistant"
                        else self.encode(body))
            return seq
        # plain transcript -- works for a base model or an unknown convention
        pre = (sys_txt + "\n\n") if sys_txt else ""
        body = "".join("%s: %s\n" % ("User" if m["role"] != "assistant" else "Assistant",
                                      m["content"]) for m in turns)
        return self.encode(pre + body + "Assistant:")

    def to_bytes(self, ids):
        """Token ids -> the raw bytes they stand for, specials dropped."""
        buf = bytearray()
        for i in ids:
            t = self.dec.get(int(i), "")
            if t in self.SPECIALS:
                continue
            for ch in t:
                buf.append(self.u2b.get(ch, 32))
        return buf

    def decode(self, ids):
        return self.to_bytes(ids).decode("utf-8", "replace")

    def stream_decoder(self):
        """A decoder that can be fed one token at a time without mangling the text.

        A BPE token is a run of BYTES, not characters. A Chinese character is three bytes and
        routinely straddles two tokens, so decoding each token on its own turns the first
        part into U+FFFD -- the ids are right, the decode is not, and the replacement
        character is what reaches the screen. This holds back a trailing partial sequence
        until the bytes that finish it arrive."""
        return _ByteStream(self)


def _utf8_split(buf):
    """(complete, tail) where `tail` is an unfinished multi-byte sequence at the end."""
    n = len(buf)
    for back in range(1, min(4, n) + 1):
        b = buf[n - back]
        if b < 0x80:                       # ASCII: nothing pending
            break
        if b >= 0xC0:                      # a lead byte: does it have all its continuations?
            need = 2 if b < 0xE0 else (3 if b < 0xF0 else 4)
            if back < need:
                return buf[:n - back], buf[n - back:]
            break
    return buf, bytearray()


class _ByteStream:
    """Incremental token -> text decoding that never splits a character."""

    def __init__(self, tok):
        self.tok = tok
        self.buf = bytearray()

    def push(self, ids):
        self.buf += self.tok.to_bytes(ids)
        done, self.buf = _utf8_split(self.buf)
        return done.decode("utf-8", "replace") if done else ""

    def flush(self):
        """Whatever is still held when the stream ends. A sequence that never completed is
        genuinely broken, so it is decoded with replacement rather than dropped."""
        out = self.buf.decode("utf-8", "replace") if self.buf else ""
        self.buf = bytearray()
        return out


# ----------------------------- result --------------------------------------
# Reasoning models emit their scratchpad inside a tagged span, and several of them OPEN that
# span in the chat template rather than in the generated text -- Qwen3 ends its prompt with
# "<think>\n", so the model itself only ever emits the CLOSING tag. Anything that pairs tags
# by looking at the output alone therefore renders raw reasoning as if it were the answer,
# right up until the closing tag finally arrives. Only the prompt settles which channel a
# stream starts in, so resolve it here, once, and hand callers text that is already labelled.
# Pairs are matched against the vocabulary and the rendered prompt, never assumed per model.
_THINK_TAGS = (("<think>", "</think>"), ("<thinking>", "</thinking>"),
               ("<reasoning>", "</reasoning>"), ("<reason>", "</reason>"),
               ("<|thinking|>", "<|/thinking|>"), ("<|think|>", "<|/think|>"))


class _Channels:
    """Split a decoded token stream into 'thinking' and 'content'.

    `opened` is the closing tag the prompt already committed us to, if the template opened a
    reasoning span itself. Tags survive token boundaries: a tag can arrive split across any
    number of chunks, so text that is still a possible prefix of one is held back rather than
    emitted into the wrong channel."""

    def __init__(self, opened=None):
        self.close = opened
        self.ch = "thinking" if opened else "content"
        self.buf = ""

    @staticmethod
    def _prefix_len(s, tags):
        """How many trailing chars of `s` could still grow into one of `tags`."""
        for n in range(min(len(s), max(len(t) for t in tags) - 1), 0, -1):
            tail = s[-n:]
            if any(t.startswith(tail) for t in tags):
                return n
        return 0

    def feed(self, text):
        """Consume a chunk, return a list of (channel, text) with empty pieces dropped."""
        self.buf += text
        out = []
        while True:
            tags = [self.close] if self.close else [o for o, _ in _THINK_TAGS]
            hit = min(((self.buf.find(t), t) for t in tags if self.buf.find(t) >= 0),
                      default=None)
            if hit is None:
                break
            i, tag = hit
            if i:
                out.append((self.ch, self.buf[:i]))
            self.buf = self.buf[i + len(tag):]
            if self.close:                       # closing a span: back to the answer
                self.close = None; self.ch = "content"
            else:                                # the model opened one itself
                self.close = dict(_THINK_TAGS)[tag]; self.ch = "thinking"
        keep = self._prefix_len(self.buf, [self.close] if self.close
                                else [o for o, _ in _THINK_TAGS])
        if len(self.buf) > keep:
            out.append((self.ch, self.buf[:len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep:]
        return [(c, t) for c, t in out if t]

    def flush(self):
        """Whatever is still held back when the stream ends -- a partial tag that never
        completed is just text."""
        out = [(self.ch, self.buf)] if self.buf else []
        self.buf = ""
        return out


def _pin_stats():
    """(buffers, pinned bytes, pool bytes, dispatches so far), or zeros off the WebGPU path.

    The dispatch counter is what settles whether two recordings of the same graph recorded
    the same graph. Sampled either side of a recording, its difference is that recording's
    command list, and a difference there is not a subtlety about buffers -- it is a different
    set of commands from source that cannot produce one."""
    try:
        from wgpy_backends.webgpu.webgpu_buffer import capture_pin_stats
        from wgpy_backends.webgpu.platform import WebGPUPlatform
        return tuple(capture_pin_stats()) + (int(WebGPUPlatform.dispatches),)
    except Exception:
        return (0, 0, 0, 0)


def _decode_rate(t_first, steps):
    """Tokens per second of the decode loop, or None when there is nothing to measure.

    Timed from the moment the FIRST token existed, over the steps after it. Two costs are
    deliberately outside this window:

      * prefill, which is reported separately as `ttft_s` -- it is paid once and scales with
        the prompt, so folding it in makes a long prompt look like a slow model.
      * the first decode step, which on the GPU path also builds the pipeline for the
        captured command list. It made the first reply after a load read 10.6 tok/s against
        31.4 for the next one on the same model.

    Below two steps there is no interval to divide by. Reporting `1226.99 tok/s` for a
    two-token reply, which is what dividing by an almost-zero denominator produced, is worse
    than reporting nothing.
    """
    if steps < 2 or t_first is None:
        return None
    dt = time.perf_counter() - t_first
    if dt <= 0:
        return None
    return round((steps - 1) / dt, 2)


def _load_stage(stage, done=None, total=None):
    """Tell whoever installed `set_load_progress` where this load has got to."""
    try:
        from . import webio
        webio.load_stage(stage, done, total)
    except Exception:
        pass


class GenResult:
    def __init__(self, text, tokens, ttft_s, decode_tok_s):
        self.text = text; self.tokens = tokens
        self.ttft_s = ttft_s; self.decode_tok_s = decode_tok_s

    def __repr__(self):
        return "GenResult(ttft=%.2fs, %.1f tok/s)\n%s" % (self.ttft_s, self.decode_tok_s, self.text)


# ----------------------------- model ---------------------------------------
# One range request spans several of the IO layer's 16 MB chunks so they are fetched in
# parallel; the fp32 expansion is done in _HEAD_BLK-row sub-blocks so it stays bounded
# whatever the read size.
_READ_BYTES = 64 << 20
_HEAD_BLK = 4096
# On the native path the head is never expanded to fp32, so the block size is not bounded by
# a working set. 61 small matmuls cost far more than a few large ones -- the head is the
# biggest tensor in the model and is multiplied every token -- but a block is one GPU buffer,
# so budget it in BYTES rather than rows: the same row count is 115 MB for one quantization
# and 230 MB for another, and the large end runs into what the backend will stage.
_HEAD_BYTES_NATIVE = 128 << 20


def _source_bits(infos, G):
    """Int width that preserves what a GGUF actually stores.

    The engine's kernels read one packed layout, so GGUF weights are converted into it --
    which means a width has to be chosen. Deriving it from the file's own bits-per-weight
    keeps `dtype="auto"` meaning the same thing it means for an AutoGPTQ directory: run at
    the precision that was downloaded, rather than silently taking an 8-bit file to 4.
    Measured from the block type holding the most weights, via its own byte count, so it
    needs no per-format table and no per-model knowledge.
    """
    weight = {}
    for t in infos:
        if not t["name"].startswith("blk."):
            continue
        n = 1
        for d in t["dims"]:
            n *= int(d)
        weight[t["type"]] = weight.get(t["type"], 0) + n
    if not weight:
        return 4
    # Weighted by element count, not tensor count: a block holds a handful of huge weight
    # matrices and many tiny F32 norms, so counting tensors would let the norms decide.
    dom = max(weight, key=weight.get)
    bpw = G.tensor_nbytes(dom, 4096) * 8.0 / 4096
    return 8 if bpw > 4.5 else 4


# Widening a BF16 buffer costs several times its own size in temporaries: it is the top
# half of a float32, so it goes through a uint32 array, a shifted copy of that, and a float32
# view before it lands as fp16. On a 64MB read that is over 300MB of peaks, and 32-bit WASM
# refuses to allocate it -- which is what stopped a 3B safetensors model from loading at all,
# on a machine with plenty of room for the model itself. Converting in bounded slices keeps
# the peak fixed no matter how large a block the caller reads.
_F16_SLICE = 1 << 22                                   # elements per conversion step


def _bytes_to_f16(raw, dt):
    """Raw safetensors bytes of dtype `dt` -> a flat float16 array."""
    if dt == "F16":
        return np.frombuffer(raw, np.float16).copy()
    if dt == "BF16":
        src = np.frombuffer(raw, np.uint16)
        out = np.empty(src.size, np.float16)
        for i in range(0, src.size, _F16_SLICE):
            j = min(src.size, i + _F16_SLICE)
            out[i:j] = (src[i:j].astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        return out
    src = np.frombuffer(raw, np.float32)
    out = np.empty(src.size, np.float16)
    for i in range(0, src.size, _F16_SLICE):
        j = min(src.size, i + _F16_SLICE)
        out[i:j] = src[i:j].astype(np.float16)
    return out


class _RowTable:
    """A 2-D weight that is only ever read by row, exposed with ndarray-style indexing.

    Embedding and lm_head tables are indexed by token id -- a handful of rows per step, or
    one block at a time when quantizing the head. Materializing the whole thing as a dense
    (vocab, hidden) fp16 array is ~1.5 GB on a 27B, which 32-bit WASM refuses to allocate
    as a single object. Subclasses supply `_rows`; indexing semantics match ndarray, so a
    scalar gives (H,), a sequence gives (n, H) and a slice gives (n, H).
    """

    def __init__(self, nrows, H):
        self.shape = (nrows, H)
        self.dtype = np.float16

    def __len__(self):
        return self.shape[0]

    def _rows(self, idx):
        """Dense (len(idx), H) fp16 for the given row numbers."""
        raise NotImplementedError

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._rows(np.arange(*idx.indices(self.shape[0])))
        ids = np.asarray(idx, np.int64)
        flat = np.atleast_1d(ids).ravel()
        # Repeated ids are common in a prompt; fetch each distinct row once.
        uniq, inv = np.unique(flat, return_inverse=True)
        out = self._rows(uniq)[inv]
        return out.reshape(ids.shape + (self.shape[1],)) if ids.ndim else out[0]


class _QuantRows(_RowTable):
    """Rows kept in their GGUF-quantized form and dequantized on lookup."""

    dtype = np.float32                                         # rows arrive as fp32

    def __init__(self, buf, gtype, H, row_nbytes, dequant):
        raw = np.frombuffer(buf, np.uint8)
        _RowTable.__init__(self, raw.size // row_nbytes, H)
        self.raw = raw.reshape(self.shape[0], row_nbytes)      # a view, not a copy
        self.gtype, self._dequant = gtype, dequant

    def _rows(self, idx):
        idx = np.asarray(idx, np.int64)
        # Each row is an independent, block-aligned run of bytes, so gathering the wanted
        # rows and dequantizing the result in ONE call is equivalent to a call per row --
        # and avoids paying numpy's per-call overhead once per token, which dominates
        # everything else at prompt length (measured 3.1s vs 12ms for 320 ids).
        sel = self.raw[idx]                                    # (n, row_nbytes), contiguous
        n, H = idx.size, self.shape[1]
        # `sel` is handed to the dequantizer directly (it reads through the buffer
        # protocol) and the fp32 it returns is handed on as-is: the caller wants fp32, so
        # a detour through fp16 would be two conversions and two copies per token.
        return self._dequant(self.gtype, sel, n * H).reshape(n, H)


class _ChunkRows(_RowTable):
    """Rows held as fp16 in several bounded blocks rather than one oversized array."""

    def __init__(self, chunks, rpc, nrows, H):
        _RowTable.__init__(self, nrows, H)
        self.chunks, self.rpc = chunks, rpc

    def _rows(self, idx):
        idx = np.asarray(idx, np.int64)
        out = np.empty((idx.size, self.shape[1]), np.float16)
        # One vectorized gather per source block rather than a copy per row.
        ch = idx // self.rpc
        for c in np.unique(ch):
            m = ch == c
            out[m] = self.chunks[int(c)][idx[m] - int(c) * self.rpc]
        return out


# Components a config can declare that this engine does not implement. Keyed off what the
# config SAYS THE MODEL COMPUTES, never off a model name or an allowlist of architectures --
# a name tells you nothing about whether the maths is covered, and an allowlist is wrong the
# day after it is written.
#
# The failure this prevents is the quiet one. A missing quantization type raises on its own;
# an unimplemented COMPONENT does not. The loader reads the weights it recognises, ignores
# the ones it does not, and produces a model that runs and answers -- wrongly. Better to
# refuse and say which piece is missing.
#
# This is necessarily incomplete: a component invented after this was written passes through
# unnoticed. It covers what is known to exist and to matter.
_UNIMPLEMENTED = [
    ("indexer_budget",  lambda c: c.get("indexer_budget"),
     "a sparse-attention indexer (a learned scorer that picks which cached keys to attend "
     "to, under a token budget) -- this engine attends densely"),
    ("ple_layer_ids",   lambda c: c.get("ple_layer_ids"),
     "per-layer embeddings (an extra embedding table injected at specific layers)"),
    ("ngram_size",      lambda c: c.get("ngram_size") and c.get("ngram_vocab_size_base"),
     "n-gram input embeddings (a second, much larger embedding table indexed by token "
     "n-grams)"),
    ("hc_count",        lambda c: c.get("hc_count"),
     "an `hc_*` low-rank component this engine has no implementation of"),
    ("deepstack_visual_indexes",
     lambda c: (c.get("vision_config") or {}).get("deepstack_visual_indexes"),
     "deepstack visual injection (several vision-tower layers fed into the decoder at "
     "different depths, not just the final one)"),
]


def unsupported_components(cfg):
    """Which declared components this engine cannot run. `cfg` may be a nested config; the
    text sub-config is searched too, since that is where a multimodal file puts them."""
    seen = {}
    for c in (cfg, cfg.get("text_config") or {}):
        if not isinstance(c, dict):
            continue
        for key, probe, why in _UNIMPLEMENTED:
            try:
                if probe(c):
                    seen[key] = why
            except Exception:
                pass
    return [(k, seen[k]) for k in sorted(seen)]


def check_supported(cfg, where="this model"):
    """Raise before any weights are read, naming every component that is missing."""
    bad = unsupported_components(cfg)
    if not bad:
        return
    raise NotImplementedError(
        "%s declares components this engine does not implement, so it would run and "
        "produce wrong output rather than fail:\n%s\n"
        "Nothing is silently skipped: each of these changes what the model computes."
        % (where, "\n".join("  * %s: %s" % (k, why) for k, why in bad)))


class CausalLM:
    """Quantized causal LM (qwen2 family) with capture-accelerated decode.

    Load from either format -- the inference engine (KV cache, capture/replay,
    generate) is shared; only loading differs:
        CausalLM.from_gptq("/models/qwen7b-gptq")     # AutoGPTQ int4 safetensors
        CausalLM.from_gguf("/models/qwen3b-gguf/model.gguf")   # llama.cpp GGUF
    """

    def __init__(self, base):
        self.base = (base.rstrip("/") + "/") if base else ""
        self.capture_ready = False
        self._greedy_chunk_capture_ready = False
        # Each layer-level API owns an independently usable auto route.  The complete
        # decoder tuner may override either one when a combination that is not locally
        # fastest wins at the higher layer.
        self._qkv_execution = "auto"
        self._gate_up_execution = "auto"
        self._qk_norm_rope_execution = "auto"
        self._kv_write_execution = "auto"
        self._embedding_row_execution = "auto"
        self._head_shape_execution = "auto"
        # Sampling state, so a forward called outside generate() still works.
        self.mtp = None             # multi-token-prediction head, when the file has one
        self._seen = []
        self._gen_start = 0
        self._con_text = ""
        self.rope_style = "hf"          # qwen2 = HF rotate_half (both GPTQ and GGUF)
        self.gs = 128; self.bits = 4    # fallback-kernel params; native GGUF keeps its format

    def _apply_cfg(self, cfg):
        """Read an HF `config.json` into the engine's shape parameters. Purely config-driven —
        no model-name special cases — so it covers the whole Llama-style decoder family
        (Llama / Mistral / Qwen2 / Qwen3 / …), dense or MoE:
          * `head_dim` is honoured when present (Qwen3 has head_dim * num_heads != hidden_size);
            it falls back to hidden_size // num_heads for models that omit it.
          * QK-norm (Qwen3) is detected from the weights at load time, not from the model name.
          * MoE fields (num_experts / num_experts_per_tok / …) drive the sparse path."""
        # Multimodal/composite configs nest the decoder under `text_config` (and the vision
        # tower under `vision_config`); use the text sub-config when present.
        if "text_config" in cfg and "hidden_size" in (cfg.get("text_config") or {}):
            self.full_cfg = cfg
            cfg = dict(cfg["text_config"])
        else:
            self.full_cfg = cfg
        self.H = cfg["hidden_size"]; self.L = cfg["num_hidden_layers"]
        self.NH = cfg["num_attention_heads"]
        self.NKV = cfg.get("num_key_value_heads", self.NH)
        self.HD = int(cfg.get("head_dim") or (self.H // self.NH))
        self.VOCAB = cfg["vocab_size"]
        self.eps = cfg.get("rms_norm_eps", 1e-6); self.theta = cfg.get("rope_theta", 10000.0)
        # MoE (0 => dense); mirrors lm_engine.build_lm's config contract
        self.n_experts = int(cfg.get("num_experts", cfg.get("num_local_experts", 0)) or 0)
        self.top_k = int(cfg.get("num_experts_per_tok", 0) or 0)
        self.norm_topk = bool(cfg.get("norm_topk_prob", True))
        self.sparse_step = int(cfg.get("decoder_sparse_step", 1) or 1)
        self.mlp_only = set(cfg.get("mlp_only_layers", []) or [])
        self.has_shared_expert = bool(cfg.get("shared_expert_intermediate_size", 0))
        # ---- hybrid linear-attention models (Gated DeltaNet family) ----
        # `layer_types` marks each layer "full_attention" or "linear_attention"; when absent,
        # `full_attention_interval` (every Nth layer is full) is used; otherwise all-full.
        types = cfg.get("layer_types") or []
        interval = int(cfg.get("full_attention_interval", 0) or 0)
        if types:
            self.layer_types = list(types)
        elif interval > 1:
            self.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                for i in range(self.L)]
        else:
            self.layer_types = ["full_attention"] * self.L
        self.is_hybrid = any(t != "full_attention" for t in self.layer_types)
        self.linear_cfg = {
            "n_k_heads": int(cfg.get("linear_num_key_heads", 0) or 0),
            "n_v_heads": int(cfg.get("linear_num_value_heads", 0) or 0),
            "k_head_dim": int(cfg.get("linear_key_head_dim", 0) or 0),
            "v_head_dim": int(cfg.get("linear_value_head_dim", 0) or 0),
            "conv_kernel_dim": int(cfg.get("linear_conv_kernel_dim", 0) or 0),
            "hidden": self.H,
        }
        # partial rope: only the first `partial_rotary_factor` of head_dim is rotated.
        # Two spellings are in circulation for the same block; which one a file uses is a
        # question about its vintage, not about which model it is, so both are read.
        rp = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        self.theta = float(rp.get("rope_theta", self.theta))
        self.partial_rotary = float(cfg.get("partial_rotary_factor",
                                            rp.get("partial_rotary_factor", 1.0)) or 1.0)
        self.rope_dim = int(self.HD * self.partial_rotary) // 2 * 2 or self.HD
        # Multi-axis rope, if the config asks for it.
        _sec = (rp.get("mrope_section") if isinstance(rp, dict) else None)
        if _sec:
            self.rope_sections = [int(x) for x in _sec]
        self.attn_output_gate = bool(cfg.get("attn_output_gate", False))
        # resolved from the text sub-config (nested configs put it there, not at the top level)
        self.tie_embeddings = bool(cfg.get("tie_word_embeddings", False))

    def _is_moe_layer(self, i):
        return (self.n_experts > 0 and i not in self.mlp_only
                and ((i + 1) % self.sparse_step == 0))

    async def _opt_norm(self, name):
        """Load an optional norm weight (e.g. Qwen3's q_norm/k_norm); None when absent."""
        return wt.Tensor(await self._np(name)) if name in self._idx else None

    async def _build_linear_attn(self, i, p, lin):
        """Build one linear-attention (Gated DeltaNet) layer. Weight names follow the
        `linear_attn.*` convention of this model family; every piece is optional so variants
        with/without conv, gate or output norm all load."""
        from . import linear_attn as la
        async def opt_lin(nm):
            return await lin(p + nm) if (p + nm + ".weight") in self._idx else None
        async def opt_arr(nm):
            return await self._np(p + nm) if (p + nm) in self._idx else None
        base = "linear_attn."
        w = {"q": await opt_lin(base + "q_proj"), "k": await opt_lin(base + "k_proj"),
             "v": await opt_lin(base + "v_proj"), "o": await opt_lin(base + "out_proj"),
             "a": await opt_lin(base + "a_proj"), "dt": await opt_lin(base + "dt_proj"),
             "g": await opt_lin(base + "g_proj"),
             "conv_w": await opt_arr(base + "conv1d.weight"),
             "conv_b": await opt_arr(base + "conv1d.bias"),
             "norm": await opt_arr(base + "norm.weight")}
        if w["conv_w"] is not None and w["conv_w"].ndim == 3:   # (C,1,W) -> (C,W)
            w["conv_w"] = w["conv_w"].reshape(w["conv_w"].shape[0], -1)
        return la.LinearAttention(dict(self.linear_cfg), w, eps=self.eps)

    async def _build_layer(self, i, lin):
        """Build one decoder layer generically from the served weights. `lin(prefix)` is the
        format's linear factory (int4 for AutoGPTQ, fp16/quantized for plain HF), so this is
        shared by every loader. Optional pieces are detected from the weights, not the model
        name: QK-norm (`self_attn.{q,k}_norm.weight`, Qwen3) and a sparse-MoE block."""
        p = "model.layers.%d." % i
        lay = {"in_ln": wt.Tensor(await self._np(p + "input_layernorm.weight")),
               "post_ln": wt.Tensor(await self._np(p + "post_attention_layernorm.weight"))}
        if self._is_linear_layer(i):                    # recurrent linear-attention layer
            lay["linear"] = await self._build_linear_attn(i, p, lin)
        else:                                           # softmax attention layer
            lay.update({
                "q": await lin(p + "self_attn.q_proj"), "k": await lin(p + "self_attn.k_proj"),
                "v": await lin(p + "self_attn.v_proj"), "o": await lin(p + "self_attn.o_proj"),
                "qn": await self._opt_norm(p + "self_attn.q_norm.weight"),
                "kn": await self._opt_norm(p + "self_attn.k_norm.weight")})
        if self._is_moe_layer(i):                       # sparse layer: router + experts
            moe = {"gate": await lin(p + "mlp.gate"), "top_k": self.top_k,
                   "norm_topk": self.norm_topk,
                   "experts": [{"gate": await lin(p + "mlp.experts.%d.gate_proj" % e),
                                "up": await lin(p + "mlp.experts.%d.up_proj" % e),
                                "down": await lin(p + "mlp.experts.%d.down_proj" % e)}
                               for e in range(self.n_experts)]}
            if self.has_shared_expert:
                moe["shared"] = {"gate": await lin(p + "mlp.shared_expert.gate_proj"),
                                 "up": await lin(p + "mlp.shared_expert.up_proj"),
                                 "down": await lin(p + "mlp.shared_expert.down_proj")}
                moe["shared_gate"] = await lin(p + "mlp.shared_expert_gate")
            lay["moe"] = moe
        else:                                           # dense SwiGLU
            lay["gate"] = await lin(p + "mlp.gate_proj")
            lay["up"] = await lin(p + "mlp.up_proj")
            lay["down"] = await lin(p + "mlp.down_proj")
        return lay

    # ---- range helpers (all reads go through the global IO callback) ----
    async def _rng(self, fn, a, b):
        from . import webio
        return await webio.io_read(self.base + fn, a, b - a + 1)     # a..b inclusive

    async def _hdr(self, shard):
        if shard not in self._shard_hdr:
            n = int.from_bytes(await self._rng(shard, 0, 7), "little")
            h = json.loads((await self._rng(shard, 8, 8 + n - 1)).decode())
            h.pop("__metadata__", None)
            self._shard_hdr[shard] = (h, 8 + n)
        return self._shard_hdr[shard]

    async def _np(self, name):
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]
        raw = await self._rng(shard, base + a, base + z - 1)
        dt = info["dtype"]
        from . import webio
        arr = (np.frombuffer(raw, np.int32) if dt == "I32" else webio.to_f32(raw, dt))
        return arr.reshape(info["shape"])

    async def _qlin(self, prefix):
        b = await self._np(prefix + ".bias") if (prefix + ".bias") in self._idx else None
        return wt.QuantizedLinear.from_autogptq(
            await self._np(prefix + ".qweight"), await self._np(prefix + ".qzeros"),
            await self._np(prefix + ".scales"), b, self.gs, self.bits)

    async def _f16_chunked(self, name, chunk_bytes=64 << 20):
        """Stream a big 2-D weight (embed / lm_head) in row blocks -> host fp16, so the full
        fp32 tensor never materializes. dtype-aware: handles F16 / BF16 / F32 sources."""
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]; V, Hh = info["shape"]; dt = info["dtype"]
        esz = 2 if dt in ("F16", "BF16") else 4; row = Hh * esz
        rpc = max(1, chunk_bytes // row); chunks = []
        for v0 in range(0, V, rpc):
            v1 = min(V, v0 + rpc)
            raw = await self._rng(shard, base + a + v0 * row, base + a + v1 * row - 1)
            chunks.append(_bytes_to_f16(raw, dt).reshape(v1 - v0, Hh))
        return _ChunkRows(chunks, rpc, V, Hh)

    def _fp16_head(self, WVH, blk=_HEAD_BLK):
        """The output head as a list of row-blocks, one Linear each.

        `UnquantizedLinear` transposes its weight into a single dense fp32 buffer, so handing
        it the whole head means (H, V) floats at once -- 1.16GB on a 152k vocabulary, which
        32-bit WASM will not allocate however much memory the machine has. The engine already
        treats `self.head` as a list and concatenates the pieces, which is how the quantized
        and GGUF paths have always loaded it; this makes the fp16 path agree."""
        V = int(WVH.shape[0]); out = []
        for v0 in range(0, V, blk):
            v1 = min(V, v0 + blk)
            out.append(wt.UnquantizedLinear(np.asarray(WVH[v0:v1], np.float16)))
        return out

    def _quantize_head(self, WVH, blk=4096):
        V, Hh = WVH.shape; blocks = []
        for v0 in range(0, V, blk):
            v1 = min(V, v0 + blk)
            WbT = np.ascontiguousarray(WVH[v0:v1].astype(np.float32).T)
            qw, qz, sc, _, _ = wt._gptq_quantize(WbT, self.gs, self.bits)
            blocks.append(wt.QuantizedLinear(qw, qz, sc, np.zeros((v1 - v0,), np.float32),
                                             Hh, v1 - v0, Hh, v1 - v0, self.gs, self.bits)); del WbT
        return blocks

    async def _head_int4(self, name, V, Hh, blk=4096):
        shard = self._idx[name]; h, base = await self._hdr(shard)
        info = h[name]; a, z = info["data_offsets"]; row = Hh * 2; blocks = []
        for v0 in range(0, V, blk):
            v1 = min(V, v0 + blk)
            raw = await self._rng(shard, base + a + v0 * row, base + a + v1 * row - 1)
            Wb = np.frombuffer(raw, np.float16).astype(np.float32).reshape(v1 - v0, Hh)
            blocks.extend(self._quantize_head(Wb, blk=blk)); del Wb
        return blocks

    @classmethod
    async def from_gptq(cls, base, lmax=None):
        """Stream + build from a served AutoGPTQ int4 dir (e.g. '/models/qwen7b-gptq')."""
        self = cls(base)
        try:
            return await self._from_gptq(lmax)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_gptq(self, lmax=None):
        from . import webio
        self.lmax = lmax
        self._shard_hdr = {}
        cfg = await webio.read_json(self.base + "config.json")
        check_supported(cfg, "this model's config.json")
        self._apply_cfg(cfg)
        self.lmax = self._auto_lmax(lmax, cfg.get("max_position_embeddings"))
        self.gs = cfg["quantization_config"]["group_size"]; self.bits = cfg["quantization_config"]["bits"]
        tie = self.tie_embeddings

        self.tok = await self._hf_tokenizer(cfg)

        imeta = await webio.read_json(self.base + "model.safetensors.index.json")
        self._idx = imeta.get("weight_map")
        if not self._idx:
            h, _ = await self._hdr("model.safetensors")
            self._idx = {k: "model.safetensors" for k in h}

        t0 = time.perf_counter()
        if not tie:
            self.head = await self._head_int4("lm_head.weight", self.VOCAB, self.H)
            self.embed = await self._f16_chunked("model.embed_tokens.weight")
        else:
            self.embed = await self._f16_chunked("model.embed_tokens.weight")
            self.head = self._quantize_head(self.embed)
        self.layers = [await self._build_layer(i, self._qlin) for i in range(self.L)]
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        # Two different questions, and they had one flag between them. `_gpu` is "can a
        # decode step be captured and replayed", which needs the whole step on the device
        # including the scatter KV cache -- WebGPU only, today. `_fused` is "does this
        # backend have the small fused kernels", which WebGL now does. Sharing the flag
        # left WebGL running rmsnorm and rope as expressions: about 400 dispatches a token
        # that the fused forms do in two, on the backend where a dispatch costs the most.
        self._gpu = wt._adam_backend_ready()        # webgpu -> capture path available
        self._fused = wt._adam_backend_ready() or wt._webgl_ready()
        self._init_state()
        self._audit()               # weights must agree with the file
        self._smoke()               # and the first forward must be usable
        return self

    # ---- GGUF (llama.cpp) loading: stored-format kernels, conversion only as fallback ----
    async def _grng(self, a, b):
        from . import webio
        return await webio.io_read(self._gguf, a, b - a + 1)     # a..b inclusive

    async def _gload(self, name):
        """Dequant a GGUF tensor to fp32, shape (out,in) [ggml dims are reversed]."""
        from . import ggufload as G
        t = self._ginfo[name]; n = 1
        for d in t["dims"]:
            n *= int(d)
        nb = G.tensor_nbytes(t["type"], n)
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb - 1)
        return G.dequant(t["type"], raw, n).reshape(tuple(reversed(t["dims"])))

    def _head_blk(self, ttype, H):
        """Rows per head block: a byte budget when nothing is expanded, a row count when it is."""
        from . import ggufload as G
        nm = G.GGML_NAMES.get(ttype)
        native = (getattr(self, "_weights", "native") == "native"
                  and wt.ggml_native_supported(nm) and H % wt._GGML_TYPES[nm][2] == 0)
        if not native:
            return _HEAD_BLK
        row = G.tensor_nbytes(ttype, H)
        rows = max(1, _HEAD_BYTES_NATIVE // max(row, 1) // _HEAD_BLK) * _HEAD_BLK
        return rows

    def _head_block(self, ttype, chunk, rows, H):
        """One block of the LM head as a Linear. Native when the type allows it, which keeps
        the head off the conversion path too -- it is (vocab, H), the largest single tensor
        in most models, and expanding it to fp32 is what forces the block-at-a-time walk."""
        from . import ggufload as G
        nm = G.GGML_NAMES.get(ttype)
        if (getattr(self, "_weights", "native") == "native"
                and wt.ggml_native_supported(nm) and H % wt._GGML_TYPES[nm][2] == 0):
            return wt.GGMLWeight(chunk, nm, (rows, H), type_id=ttype).as_linear()
        return self._gquant(G.dequant(ttype, chunk, rows * H).reshape(rows, H))

    async def _tok_files(self):
        """(vocab, merges, added) from whichever tokenizer layout the model ships.

        Two are in circulation and a model may have either: the GPT-2 pair (vocab.json +
        merges.txt) and the single tokenizer.json that HF's `tokenizers` writes. Which one a
        model has says nothing about the model, so requiring one of them rules out families
        arbitrarily. tokenizer.json also keeps the special tokens in `added_tokens`, OUTSIDE
        the vocabulary proper -- miss those and the end-of-turn id is wrong, which is not a
        crash but a model that never stops where it should."""
        from . import webio
        try:
            vocab = await webio.read_json(self.base + "vocab.json")
            lines = (await webio.read_text(self.base + "merges.txt")).split("\n")
            return vocab, [m for m in lines[1:] if m and not m.startswith("#")], []
        except Exception:
            pass
        tj = await webio.read_json(self.base + "tokenizer.json")
        self._pretok = pretok_pattern(tj)
        self._pretok_style = pretok_style(tj)
        mdl = tj.get("model") or {}
        vocab = dict(mdl.get("vocab") or {})
        added = []
        for a in (tj.get("added_tokens") or []):
            if isinstance(a, dict) and a.get("content") is not None and a.get("id") is not None:
                vocab[a["content"]] = int(a["id"]); added.append(a["content"])
        # merges are ["a b", ...] in newer files and [["a", "b"], ...] in older ones
        merges = [" ".join(m) if isinstance(m, (list, tuple)) else m
                  for m in (mdl.get("merges") or [])]
        return vocab, merges, added

    async def _hf_tokenizer(self, cfg):
        """Tokenizer for an HF directory: vocab, merges, and the model's own chat template.

        tokenizer_config.json is where a served model states its prompt layout and its
        added tokens; reading it is what makes an unfamiliar model's turns come out right
        instead of approximately right."""
        from . import webio
        vocab, merges, extra = await self._tok_files()
        _e = cfg.get("eos_token_id")
        eos = _e if isinstance(_e, list) else ([_e] if _e is not None else [])
        tc = {}
        try:
            tc = await webio.read_json(self.base + "tokenizer_config.json")
        except Exception:
            pass                                   # optional -- fall back to the format probe
        # vocab.json holds only the trained vocabulary: the special tokens live outside it,
        # in tokenizer_config.json's `added_tokens_decoder` (or tokenizer.json's
        # `added_tokens`). Fold them back in, because anything that looks a marker up by name
        # -- end-of-turn, the chat markers, a vision placeholder -- searches the vocabulary,
        # and without them the lookup either fails or silently matches the wrong token.
        added = tc.get("added_tokens_decoder") or {}
        ctrl = []
        for tid, v in added.items():
            if isinstance(v, dict) and v.get("content"):
                ctrl.append(v["content"])
                try:
                    vocab.setdefault(v["content"], int(tid))
                except (TypeError, ValueError):
                    pass
        ctrl = ctrl or extra
        tok = BPETokenizer(vocab, merges, eos_ids=eos,
                           chat_template=tc.get("chat_template"), control=ctrl,
                           pattern=getattr(self, "_pretok", None),
                           style=getattr(self, "_pretok_style", None))
        await tok.prepare_template()
        await self._load_gen_defaults()
        return tok

    async def _load_gen_defaults(self):
        """Sampling settings the model ships with (generation_config.json).

        A model states what it should be sampled at -- Qwen3 asks for temperature 0.6 /
        top_p 0.95 / top_k 20 -- and ignoring that makes it look worse than it is. These
        are defaults: anything passed to generate() still wins. Read by name, so a field
        the sampler does not implement is simply not picked up."""
        from . import webio
        try:
            gc = await webio.read_json(self.base + "generation_config.json")
        except Exception:
            return                                  # optional file
        keys = ("temperature", "top_p", "top_k", "do_sample", "repetition_penalty", "min_p",
                "max_new_tokens", "max_length", "min_new_tokens", "presence_penalty",
                "frequency_penalty", "stop")
        got = {k: gc[k] for k in keys if gc.get(k) is not None}
        if got:
            self.gen_defaults = dict(getattr(self, "gen_defaults", {}) or {}, **got)

    async def _gexperts_stacked(self, name, also=None):
        """A layer's experts for one projection, as a single stacked weight -- or None when
        this build cannot keep them in their native encoding.

        Stacking is what makes the layer capturable: with the experts in one buffer the
        shader selects one by index, so the dispatch does not depend on which experts the
        router chose. The requantizing path has to produce a Linear per expert and stays on
        `_gexperts`.
        """
        from . import ggufload as G
        t = self._ginfo[name]
        ne, out_d, in_d = (int(d) for d in reversed(t["dims"]))
        nm = G.GGML_NAMES.get(t["type"])
        if not (getattr(self, "_weights", "native") == "native"
                and nm is not None and wt.ggml_native_supported(nm)
                and in_d % wt._GGML_TYPES[nm][2] == 0):
            return None
        per = out_d * in_d
        nb = G.tensor_nbytes(t["type"], per)
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb * ne - 1)
        # Keep views of the file bytes.  Copying every expert here, then copying the
        # joined gate/up bytes again, kept several whole-layer copies in the WASM heap
        # while the GPU upload also needed its own bounded staging window.
        chunks = [memoryview(raw)[e * nb:(e + 1) * nb] for e in range(ne)]
        also_chunks = None
        if also is not None:
            t2 = self._ginfo.get(also)
            if t2 is None or t2["type"] != t["type"]:
                return None
            ne2, out2, in2 = (int(d) for d in reversed(t2["dims"]))
            if (ne2, in2) != (ne, in_d):
                return None
            nb2 = G.tensor_nbytes(t2["type"], out2 * in2)
            off2 = self._gds + t2["offset"]
            raw2 = await self._grng(off2, off2 + nb2 * ne - 1)
            # One weight per expert holding both: the first out_d rows are this tensor's, the
            # rest the other's, so a single matmul produces both halves at once.
            also_chunks = [memoryview(raw2)[e * nb2:(e + 1) * nb2]
                           for e in range(ne)]
            out_d = out_d + out2
        return wt.GGMLMoELinear(chunks, nm, in_d, out_d,
                               also_chunks=also_chunks)

    async def _gexperts(self, name):
        """Yield each expert of a stacked MoE tensor as a ready Linear, one at a time.

        llama.cpp packs a layer's experts into a single `ffn_*_exps` tensor. The packed
        bytes are fetched in ONE range request -- a read per expert would turn three reads
        per layer into hundreds -- and then split, which is exact: quantization blocks run
        along `in` and out*in is a whole number of blocks, so every per-expert offset is
        block-aligned.

        On the native path each expert's slice goes straight to the GPU in its own encoding.
        Otherwise it is dequantized and requantized one expert at a time, because the whole
        stack as fp32 is ~800 MB on a 128-expert layer; peak stays at one expert.
        """
        from . import ggufload as G
        t = self._ginfo[name]
        ne, out_d, in_d = (int(d) for d in reversed(t["dims"]))
        per = out_d * in_d
        nb = G.tensor_nbytes(t["type"], per)                  # bytes per expert
        off = self._gds + t["offset"]
        raw = await self._grng(off, off + nb * ne - 1)
        nm = G.GGML_NAMES.get(t["type"])
        native = (getattr(self, "_weights", "native") == "native"
                  and wt.ggml_native_supported(nm)
                  and in_d % wt._GGML_TYPES[nm][2] == 0)
        for e in range(ne):
            chunk = raw[e * nb:(e + 1) * nb]
            if native:
                yield wt.GGMLWeight(chunk, nm, (out_d, in_d),
                                    type_id=t["type"]).as_linear()
            else:
                yield self._gquant(G.dequant(t["type"], chunk, per).reshape(out_d, in_d))

    async def _gload_native(self, name, bias=None):
        """Upload a GGUF weight unchanged, for a kernel that decodes it while multiplying.

        No dequantize, no requantize, no fp32 intermediate: the packed tensor is a few tens
        of MB where its fp32 expansion would be hundreds, so it goes up in one piece.
        Returns None when the tensor is not something the native kernels handle -- a type
        without a decode fragment, a non-2D tensor, or a K that is not block-aligned -- and
        the caller falls back to converting it."""
        from . import ggufload as G
        t = self._ginfo[name]
        dims = [int(d) for d in reversed(t["dims"])]
        if len(dims) != 2:
            return None
        nm = G.GGML_NAMES.get(t["type"])
        if not wt.ggml_native_supported(nm):
            return None
        N, K = dims
        if K % wt._GGML_TYPES[nm][2]:
            return None
        row = G.tensor_nbytes(t["type"], K)
        off = self._gds + t["offset"]
        stored = wt.GGMLWeight(await self._grng(off, off + N * row - 1), nm, (N, K),
                               type_id=t["type"])
        return stored.as_linear(bias)

    async def _gload_quant(self, name, bias=None):
        """Read a GGUF weight and quantize it without ever holding it whole.

        A 27B feed-forward weight is 89M values: 356 MB as fp32, and the i-quant
        dequantizers build an intermediate of the same size again, which a 32-bit heap will
        not give. Nothing needs the whole thing -- each output column is quantized
        independently, and GGUF blocks run along the input dimension, so a band of output
        rows is both self-contained and block-aligned. This walks those bands: read, expand,
        quantize, keep only the packed result. Peak is one band, whatever the tensor's size.
        """
        from . import ggufload as G
        if getattr(self, "_weights", "native") == "native":
            got = await self._gload_native(name, bias)
            if got is not None:
                return got
        if getattr(self, "_qbits", None) == 0:                # fp16 path wants it whole
            return self._gquant(await self._gload(name), bias)
        t = self._ginfo[name]
        dims = [int(d) for d in reversed(t["dims"])]
        if len(dims) != 2:
            return self._gquant(await self._gload(name), bias)
        N, K = dims                                            # (out, in)
        per = 32 // self.bits
        kmul = self.gs if self.gs % per == 0 else self.gs * per
        Kp = K + (-K) % kmul
        Np = N + (-N) % per
        row = G.tensor_nbytes(t["type"], K)                    # bytes per output row
        off = self._gds + t["offset"]
        band = max(per, (_READ_BYTES // max(K * 4, 1)) // per * per) or per
        qws, qzs, scs = [], [], []
        for r0 in range(0, Np, band):
            r1 = min(Np, r0 + band)
            live = max(0, min(N, r1) - r0)                     # real rows in this band
            if live:
                raw = await self._grng(off + r0 * row, off + (r0 + live) * row - 1)
                W = G.dequant(t["type"], raw, live * K).reshape(live, K)
                del raw
            else:
                W = np.zeros((0, K), np.float32)
            if live < r1 - r0 or Kp != K:                      # pad to the packing multiple
                W = np.pad(W, ((0, (r1 - r0) - live), (0, Kp - K)))
            qw, qz, sc, _, _ = wt._gptq_quantize(W, self.gs, self.bits, from_out_in=True)
            del W
            qws.append(qw); qzs.append(qz); scs.append(sc)
        qw = np.concatenate(qws, axis=1); del qws
        qz = np.concatenate(qzs, axis=1); del qzs
        sc = np.concatenate(scs, axis=1); del scs
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, self.gs, self.bits)

    def _gquant(self, W_out_in, bias=None):
        """fp32 (out,in) weight -> a linear layer. Quantized to int`bits` by default; with
        `self._qbits == 0` (dtype="fp16") it stays unquantized, which also makes GGUF models
        runnable without a GPU (the int kernel is GPU-only)."""
        if getattr(self, "_qbits", None) == 0:
            return wt.UnquantizedLinear(W_out_in, bias)
        # Handed over in its (out, in) layout: the quantizer transposes one column block at
        # a time, so the full (in, out) copy -- 356 MB on a 27B feed-forward weight, on top
        # of the 356 MB the tensor already occupies -- is never built.
        N, K = W_out_in.shape; per = 32 // self.bits
        kmul = self.gs if self.gs % per == 0 else self.gs * per
        Kp = K + (-K) % kmul; Np = N + (-N) % per
        src = W_out_in
        if Kp != K or Np != N:
            src = np.pad(src, ((0, Np - N), (0, Kp - K)))
        qw, qz, sc, _, _ = wt._gptq_quantize(src, self.gs, self.bits, from_out_in=True)
        del src
        b = np.zeros((N,), np.float32) if bias is None else np.asarray(bias, np.float32)
        return wt.QuantizedLinear(qw, qz, sc, b, K, N, Kp, Np, self.gs, self.bits)

    @classmethod
    async def from_gguf(cls, url, lmax=None, bits=None, quantize=True, weights="native",
                        expert_weights_norm=None):
        """Load a llama.cpp GGUF. `url` is the served .gguf file.

        `weights="native"` (the default) uploads each tensor in its own encoding and
        multiplies it packed -- no dequantize, no requantize, no fp32 intermediate.
        `weights="requant"` takes the older path: dequantize, requantize to int`bits`
        (4 or 8), run the ordinary int kernel. A type with no native decode falls back to
        that path per tensor regardless.

        `lmax=None` runs at the context length the file itself declares (`{arch}.context_length`),
        trimmed only if the KV cache for it would exceed the memory budget; pass a number to
        force a specific size.

        `bits=None` (the default) takes the width from the file itself, so an 8-bit GGUF
        runs at int8 instead of being quietly reduced to int4. Note that the conversion
        itself is a kernel-format limitation, not a property of the format: the engine
        reads one packed layout, so k-quant super-blocks are re-approximated by the
        kernel's group scale/zero scheme. Running GGUF blocks natively would need a
        kernel per quantization type.

        **Architecture-generic**: shape/rope/eps are read from GGUF's own `{arch}.*` metadata
        keys (the format's convention), and optional pieces are detected from the tensors —
        QK-norm (`attn_q_norm`/`attn_k_norm`, e.g. Qwen3) and sparse-MoE (`ffn_gate_inp` +
        `ffn_*_exps`). So any Llama-style decoder GGUF (llama / qwen2 / qwen3 / mistral / …)
        loads without a per-model branch. A GGUF whose weights use a quantization type this
        SDK cannot dequantize (e.g. the IQ i-quants) is rejected with a clear error."""
        self = cls(None)
        try:
            return await self._from_gguf(url, lmax, bits, quantize, weights,
                                         expert_weights_norm)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_gguf(self, url, lmax=None, bits=None, quantize=True, weights="native",
                         expert_weights_norm=None):
        from . import ggufload as G
        self._expert_norm_default = expert_weights_norm
        self.lmax = lmax; self._gguf = url
        _, meta, infos, ds = await G.read_header(self._grng)
        # `bits=None` follows the file (see _source_bits); an explicit width still wins.
        self.bits = int(bits) if bits else _source_bits(infos, G)
        # `quantize=False` (dtype="fp16") keeps the dequantized weights unquantized, so a GGUF
        # runs on the numpy/CPU path too — the int kernel is GPU-only.
        self._qbits = self.bits if quantize else 0
        # "native" multiplies straight out of the file's own encoding; "requant" converts
        # every weight to the int4/int8 kernel format first, which is what this did before
        # the native kernels existed. Native is the default: it skips the conversion (the
        # bulk of a load) and the second rounding that came with it. Types without a native
        # decode fall back to conversion per tensor either way.
        if weights not in ("native", "requant"):
            raise ValueError("weights must be 'native' or 'requant', got %r" % (weights,))
        self._weights = weights
        arch = meta.get("general.architecture")
        if not arch:
            raise ValueError("GGUF has no general.architecture")
        A = arch + "."                                        # GGUF namespaces its keys by arch
        def m(key, *alts, default=None, required=True):
            for k in (key,) + alts:
                if A + k in meta: return meta[A + k]
            if default is not None or not required: return default
            raise NotImplementedError(
                "GGUF arch %r is missing %r — unsupported architecture for from_gguf" % (arch, key))
        self.H = int(m("embedding_length"))
        # `block_count` counts multi-token-prediction blocks too, but those are a separate
        # head (their tensors carry a `nextn.` prefix), not part of the decoder stack --
        # running one as a normal layer would silently corrupt every generation. The count
        # is declared in the file, so this needs no per-model knowledge.
        self.L = int(m("block_count")) - int(m("nextn_predict_layers", default=0,
                                               required=False) or 0)
        self.NH = int(m("attention.head_count"))
        self.NKV = int(m("attention.head_count_kv", default=self.NH, required=False) or self.NH)
        self.HD = int(m("attention.key_length", default=0, required=False) or (self.H // self.NH))
        self.VOCAB = len(meta["tokenizer.ggml.tokens"])
        self.eps = float(m("attention.layer_norm_rms_epsilon", default=1e-6, required=False))
        self.theta = float(m("rope.freq_base", default=10000.0, required=False))
        # MoE (0 => dense), same contract as the safetensors path
        self.n_experts = int(m("expert_count", default=0, required=False) or 0)
        self.top_k = int(m("expert_used_count", default=0, required=False) or 0)
        self._arch = arch
        self.norm_topk = self._expert_norm(arch, m)
        self.sparse_step = 1; self.mlp_only = set()
        self.has_shared_expert = False
        # ---- hybrid / state-space blocks, read from the file's own metadata ----
        # A GGUF declares these under its arch namespace whenever some blocks are recurrent
        # (linear attention / SSM) instead of softmax attention. Nothing here names a model:
        # a file that declares them gets the recurrent path, a file that does not gets the
        # plain decoder path.
        interval = int(m("full_attention_interval", default=0, required=False) or 0)
        recurrent = m("attention.recurrent_layers", default=None, required=False)
        if recurrent:
            self.layer_types = ["linear_attention" if bool(r) else "full_attention"
                                for r in list(recurrent)[:self.L]]
        elif interval > 1:
            self.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                                for i in range(self.L)]
        else:
            self.layer_types = ["full_attention"] * self.L
        self.is_hybrid = any(t != "full_attention" for t in self.layer_types)
        # The size the model itself declares (`{arch}.context_length`), capped only by the
        # KV-cache memory budget; an explicit `lmax` wins over both.
        self.lmax = self._auto_lmax(lmax, m("context_length", default=None, required=False))
        d_state = int(m("ssm.state_size", default=0, required=False) or 0)
        d_inner = int(m("ssm.inner_size", default=0, required=False) or 0)
        n_group = int(m("ssm.group_count", default=0, required=False) or 0)
        dt_rank = int(m("ssm.time_step_rank", default=0, required=False) or 0)
        self.linear_cfg = {
            "n_k_heads": n_group, "n_v_heads": dt_rank,
            "k_head_dim": d_state,
            "v_head_dim": (d_inner // dt_rank) if dt_rank else 0,
            "conv_kernel_dim": int(m("ssm.conv_kernel", default=0, required=False) or 0),
            "hidden": self.H}
        # partial rope: the file says how many dims are rotated
        n_rot = int(m("rope.dimension_count", default=0, required=False) or 0)
        self.rope_dim = n_rot if n_rot else self.HD
        self.partial_rotary = (self.rope_dim / self.HD) if self.HD else 1.0
        self.attn_output_gate = False        # set below if the weights show a gate
        _eos = [meta.get("tokenizer.ggml.eos_token_id"), meta.get("tokenizer.ggml.eot_token_id")]
        _toks = meta["tokenizer.ggml.tokens"]
        _tt = meta.get("tokenizer.ggml.token_type") or []
        # GGML_TOKEN_TYPE_CONTROL / _USER_DEFINED: the file says which tokens are markers,
        # so BPE never has to guess from what a token looks like.
        _ctrl = [_toks[i] for i, t in enumerate(_tt) if int(t) in (3, 4)]
        self.tok = BPETokenizer({t: i for i, t in enumerate(_toks)},
                                meta.get("tokenizer.ggml.merges", []), eos_ids=_eos,
                                chat_template=meta.get("tokenizer.chat_template"), control=_ctrl)
        # The template is compiled after the weights, not here: jinja2 is still arriving
        # from the worker's boot-time fetch, and waiting for it before reading a single
        # weight put that whole fetch -- a network round trip, 2-14 s measured -- in front of
        # the load instead of beside it.
        self._ginfo = {t["name"]: t for t in infos}; self._gds = ds

        # Preflight: report every unsupported quantization up front, from the header alone,
        # instead of failing part-way through a multi-GB download. Mixed ("dynamic") files
        # combine many types, so list all of them with how many tensors each covers.
        from collections import Counter
        bad = Counter(G.GGML_NAMES.get(t["type"], "type%d" % t["type"])
                      for t in infos if not G.is_supported(t["type"]))
        if bad:
            raise NotImplementedError(
                "this GGUF uses quantization types webtorch cannot dequantize yet: %s. "
                "Supported: %s. Pick a build using only supported types (e.g. Q4_K_M, Q5_K_M, "
                "Q6_K, Q8_0, IQ4_XS)." % (
                    ", ".join("%s (%d tensors)" % (k, v) for k, v in sorted(bad.items())),
                    ", ".join(sorted(G.SUPPORTED_NAMES))))

        # Preflight the architecture from the header: a block must be something this engine
        # can run — softmax attention (separate q/k/v or a fused qkv) or a recurrent
        # linear-attention / state-space block (ssm_* tensors). Report a mismatch now rather
        # than after downloading gigabytes.
        names = set(self._ginfo)
        if "blk.0.attn_norm.weight" in names:
            post = any(("blk.0." + n + ".weight") in names
                       for n in ("ffn_norm", "post_attention_norm"))
            attn = (all(("blk.0.attn_%s.weight" % x) in names for x in ("q", "k", "v"))
                    or "blk.0.attn_qkv.weight" in names)
            recurrent = any(n.startswith("blk.0.ssm_") for n in names)
            if not post or not (attn or recurrent):
                missing = []
                if not post: missing.append("a post-attention norm (ffn_norm/post_attention_norm)")
                if not (attn or recurrent):
                    missing.append("attention (attn_q/k/v or attn_qkv) or recurrent (ssm_*) tensors")
                raise NotImplementedError(
                    "this GGUF's %r blocks are not a shape this engine implements: layer 0 "
                    "lacks %s." % (arch, "; ".join(missing)))
        bad = sorted({G.GGML_NAMES.get(t["type"], "type%d" % t["type"])
                      for t in infos if t["type"] not in G.SUPPORTED_TYPES})
        if bad:                                               # e.g. IQ4_XS / IQ2_M i-quants
            raise NotImplementedError(
                "GGUF uses quantization type(s) %s which this SDK cannot dequantize "
                "(supported: %s). Use a supported quant (Q4_K/Q6_K/Q8_0/Q4_0/Q4_1/F16/F32)."
                % (", ".join(bad), ", ".join(sorted(G.SUPPORTED_NAMES))))

        # WebGL's growing KV cache uses this append kernel on every token. Compiling it
        # after 13+ GB of weights have been uploaded can synchronise the driver at the
        # load-time proof; compile while memory is still free instead. The same generic
        # kernel serves any model whose execution uses this cache, not a model-name case.
        wt._webgl_prepare_growing_cache()

        t0 = time.perf_counter()
        # token_embd -> host f16 (streamed in row-blocks so the full 1.2GB fp32
        # never materializes); if tied, build the int4 head from the same blocks.
        et = self._ginfo["token_embd.weight"]
        H = int(et["dims"][0]); V = int(et["dims"][1])
        erow = G.tensor_nbytes(et["type"], H); eoff = self._gds + et["offset"]
        # The table is kept PACKED and read a row at a time (see _PackedRows): it is only
        # ever indexed by token id, and a dense (V,H) fp16 copy is ~1.5 GB on a 27B --
        # more than 32-bit WASM will hand out for a single array.
        ebuf = np.empty(V * erow, np.uint8)
        tied = "output.weight" not in self._ginfo
        self._tied_embed_head = bool(tied)
        self.head = [] if tied else None
        # Read in big spans, not row counts: the IO layer splits a request into 16 MB
        # pieces and runs several at once, so one large read keeps every fetch slot busy
        # where a 4096-row read (a few MB) leaves them idle and pays the round trip per
        # chunk. Head quantization still walks the span in bounded sub-blocks, because it
        # is the fp32 expansion -- not the read -- that has to stay small.
        _ehb = self._head_blk(et["type"], H)
        rpr = max(_ehb, (_READ_BYTES // erow) // _ehb * _ehb)
        for v0 in range(0, V, rpr):
            v1 = min(V, v0 + rpr)
            raw = await self._grng(eoff + v0 * erow, eoff + v1 * erow - 1)
            ebuf[v0 * erow:v1 * erow] = np.frombuffer(raw, np.uint8)
            if tied:                                          # (blk,H) = (out,in)
                hb = self._head_blk(et["type"], H)
                for b0 in range(v0, v1, hb):
                    b1 = min(v1, b0 + hb)
                    self.head.append(self._head_block(
                        et["type"], raw[(b0 - v0) * erow:(b1 - v0) * erow], b1 - b0, H))
        self.embed = _QuantRows(memoryview(ebuf.data).cast("B"), et["type"], H, erow, G.dequant)
        if not tied:                                          # separate lm_head, also streamed
            ot = self._ginfo["output.weight"]
            orow = G.tensor_nbytes(ot["type"], H); ooff = self._gds + ot["offset"]
            self.head = []
            _ohb = self._head_blk(ot["type"], H)
            orpr = max(_ohb, (_READ_BYTES // orow) // _ohb * _ohb)
            for v0 in range(0, V, orpr):                      # large reads, small expansions
                v1 = min(V, v0 + orpr)
                raw = await self._grng(ooff + v0 * orow, ooff + v1 * orow - 1)
                hb = self._head_blk(ot["type"], H)
                for b0 in range(v0, v1, hb):
                    b1 = min(v1, b0 + hb)
                    self.head.append(self._head_block(
                        ot["type"], raw[(b0 - v0) * orow:(b1 - v0) * orow], b1 - b0, H))
        # A single tied native head means the complete compressed embedding table is already
        # bounded by `_HEAD_BYTES_NATIVE`.  Preserve its original compact row layout on
        # WebGPU as a second-stage candidate: the projection's transposed layout is ideal for
        # scanning every vocabulary row, while embedding lookup wants one contiguous row.
        # This is format/shape capability detection, never a repository or model-name rule.
        self._embed_packed_rows = None
        self._embed_row_words = 0
        if (tied and wt._adam_backend_ready() and len(self.head) == 1
                and isinstance(self.head[0], wt.GGMLLinear)
                and self.head[0].type_name == "Q6_K" and erow % 4 == 0):
            self._embed_packed_rows = xp.asarray(ebuf.view(np.int32))
            self._embed_row_words = erow // 4
        self.layers = []
        from . import webio as _wio
        for i in range(self.L):
            # A stop has to be able to land HERE. Building the layers is the phase that runs
            # without awaiting anything the caller can interrupt -- bytes already in the cache
            # are dequantized and uploaded straight through -- and on a 1.7B that was measured
            # as a 19-second window in which the worker answered nothing at all and a stop
            # waited for the whole phase to end. One checkpoint per layer bounds the wait by
            # one layer instead.
            _wio._check_cancel()
            p = "blk.%d." % i
            async def qb(nm, _p=p):                            # weight + optional bias -> QuantizedLinear
                b = (await self._gload(_p + nm + ".bias")) if (_p + nm + ".bias") in self._ginfo else None
                return await self._gload_quant(_p + nm + ".weight", b)
            async def opt_norm(nm, _p=p):                      # QK-norm (qwen3-style) if present
                n = _p + nm + ".weight"
                return wt.Tensor(await self._gload(n)) if n in self._ginfo else None
            # The post-attention norm is called `ffn_norm` by most converters and
            # `post_attention_norm` by others; take whichever this file has.
            post_name = next((n for n in ("ffn_norm", "post_attention_norm")
                              if (p + n + ".weight") in self._ginfo), None)
            if post_name is None:
                raise NotImplementedError(
                    "GGUF block %d has no post-attention norm (looked for ffn_norm / "
                    "post_attention_norm)" % i)
            lay = {"in_ln": wt.Tensor(await self._gload(p + "attn_norm.weight")),
                   "post_ln": wt.Tensor(await self._gload(p + post_name + ".weight"))}
            has_ssm = any(n.startswith(p + "ssm_") for n in self._ginfo)
            if has_ssm or self._is_linear_layer(i):
                # Recurrent (linear-attention / state-space) block. Every piece is optional and
                # located by name, so any converter's layout loads as long as the tensors exist.
                async def opt_g(nm, _p=p):
                    n = _p + nm
                    return await self._gload(n) if n in self._ginfo else None
                async def opt_qb(nm, _p=p):
                    return (await qb(nm, _p)) if (_p + nm + ".weight") in self._ginfo else None
                w = {"qkv": await opt_qb("attn_qkv"), "g": await opt_qb("attn_gate"),
                     "q": await opt_qb("attn_q"), "k": await opt_qb("attn_k"),
                     "v": await opt_qb("attn_v"),
                     "beta": await opt_qb("ssm_beta"), "alpha": await opt_qb("ssm_alpha"),
                     "o": await opt_qb("ssm_out"),
                     "A": await opt_g("ssm_a"), "dt_bias": await opt_g("ssm_dt.bias"),
                     "norm": await opt_g("ssm_norm.weight"),
                     "conv_w": await opt_g("ssm_conv1d.weight"),
                     "conv_b": await opt_g("ssm_conv1d.bias")}
                if w["conv_w"] is not None and w["conv_w"].ndim > 2:
                    w["conv_w"] = w["conv_w"].reshape(w["conv_w"].shape[0], -1)
                from . import linear_attn as la
                lay["linear"] = la.LinearAttention(dict(self.linear_cfg), w, eps=self.eps)
                self.layer_types[i] = "linear_attention"
            else:
                lay.update({
                    "q": await qb("attn_q"), "k": await qb("attn_k"), "v": await qb("attn_v"),
                    "o": await qb("attn_output"),
                    "qn": await opt_norm("attn_q_norm"), "kn": await opt_norm("attn_k_norm")})
            if (p + "ffn_gate_inp.weight") in self._ginfo:     # sparse-MoE block
                # llama.cpp stacks experts: ffn_{gate,up,down}_exps = (n_experts, out, in).
                # Taken one expert at a time -- see _gload_expert; the full stack does not
                # fit, and only the packed result is kept.
                ne = int(self._ginfo[p + "ffn_gate_exps.weight"]["dims"][-1])
                stacked, experts = {}, None
                # gate and up take the same input and the same experts, so they ride in one
                # weight of twice the output width and one dispatch. Not for speed: it was
                # meant to be, on the theory that a routed layer's matmuls are small enough
                # that the command count limits the step, and the step did not move (35.57ms
                # against 35.62ms). Kept because one weight and one dispatch is less to hold
                # than two, and it halves the routed layer's descriptor and capture work.
                gu = await self._gexperts_stacked(p + "ffn_gate_exps.weight",
                                                  also=p + "ffn_up_exps.weight")
                dn = await self._gexperts_stacked(p + "ffn_down_exps.weight")
                if gu is not None and dn is not None:
                    stacked = {"gate_up": gu, "down": dn}
                else:                                      # a path that cannot stack them
                    stacked = {}
                    experts = [{} for _ in range(ne)]
                    for key, nm in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"),
                                    ("down", "ffn_down_exps")):
                        e = 0
                        async for lin in self._gexperts(p + nm + ".weight"):
                            experts[e][key] = lin; e += 1
                lay["moe"] = {"gate": await self._gload_quant(p + "ffn_gate_inp.weight"),
                              "top_k": self.top_k or 2, "norm_topk": self.norm_topk,
                              "experts": experts, "stacked": stacked or None,
                              "n_experts": ne}
            else:
                lay["gate"] = await qb("ffn_gate"); lay["up"] = await qb("ffn_up")
                lay["down"] = await qb("ffn_down")
            self.layers.append(lay)
        self.final_norm = wt.Tensor(await self._gload("output_norm.weight"))
        self.mtp = await self._gload_mtp()
        await self.tok.prepare_template()
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._fused = wt._adam_backend_ready() or wt._webgl_ready()
        # Reading is done; everything from here reads nothing, so the byte meter has stopped
        # and only these say the load is still alive. On a 13GB model they are minutes.
        # The largest CPU/JS transfer window belongs to weight upload, not inference.
        # Releasing it here (rather than only in the host's load-finally block) leaves
        # more physical memory for the first whole-model GPU step. The following small
        # input/readback buffers may allocate a fresh bounded window as needed.
        wt._release_transfer_memory()
        _load_stage("warming")
        self._init_state()          # kernel shapes measured for this model
        _load_stage("checking")
        self._audit()               # weights must agree with the file
        _load_stage("proving")
        self._smoke()               # and the first forward must be usable
        _load_stage("ready")
        return self

    async def _gload_mtp(self):
        """Load a multi-token-prediction head, if the file ships one. None otherwise.

        Found by tensor name rather than by architecture: a NextN/MTP block is an extra
        decoder block past the trunk whose distinguishing tensor is `nextn.eh_proj`, and
        DeepSeek-V3, Qwen3.5 and GLM all lay it out that way. Everything else in the block
        is an ordinary attention layer, and the pieces it does not carry -- its own
        embedding table, its own LM head -- fall back to the trunk's.
        """
        pre = next(("blk.%d." % i for i in range(self.L, self.L + 8)
                    if ("blk.%d.nextn.eh_proj.weight" % i) in self._ginfo), None)
        if pre is None:
            return None

        async def qb(nm):
            b = ((await self._gload(pre + nm + ".bias"))
                 if (pre + nm + ".bias") in self._ginfo else None)
            return await self._gload_quant(pre + nm + ".weight", b)

        async def opt_qb(nm):
            return (await qb(nm)) if (pre + nm + ".weight") in self._ginfo else None

        async def opt_t(nm):
            n = pre + nm + ".weight"
            return wt.Tensor(await self._gload(n)) if n in self._ginfo else None

        post = next((n for n in ("ffn_norm", "post_attention_norm")
                     if (pre + n + ".weight") in self._ginfo), None)
        m = {"eh": await qb("nextn.eh_proj"),
             "enorm": await opt_t("nextn.enorm"), "hnorm": await opt_t("nextn.hnorm"),
             "in_ln": await opt_t("attn_norm"),
             "post_ln": (await opt_t(post)) if post else None,
             "qkv": await opt_qb("attn_qkv"),
             "q": await opt_qb("attn_q"), "k": await opt_qb("attn_k"),
             "v": await opt_qb("attn_v"), "o": await opt_qb("attn_output"),
             "qn": await opt_t("attn_q_norm"), "kn": await opt_t("attn_k_norm"),
             "gate": await opt_qb("ffn_gate"), "up": await opt_qb("ffn_up"),
             "down": await opt_qb("ffn_down"),
             "head_norm": await opt_t("nextn.shared_head_norm"),
             "head": await opt_qb("nextn.shared_head_head")}
        if m["eh"] is None or m["q"] is None or m["o"] is None:
            return None                       # incomplete; run without it
        return m

    def mtp_logits(self, h_last, token, pos, cache):
        """Predict the token AFTER `token`, from the trunk's last hidden state.

        `h_last` is the trunk's final hidden for the position that produced `token`; the
        head combines it with that token's embedding and runs one more decoder block. This
        is the draft step of speculative decoding -- one layer against the trunk's many.
        """
        m = self.mtp
        if m is None:
            return None
        e = wt.Tensor(np.asarray(self.embed[int(token)], np.float32).reshape(1, self.H))
        pair = wt.cat([self._rms(e, m["enorm"]) if m["enorm"] is not None else e,
                       self._rms(h_last, m["hnorm"]) if m["hnorm"] is not None else h_last],
                      axis=-1)
        h = m["eh"](pair)
        lay = {"q": m["q"], "k": m["k"], "v": m["v"], "o": m["o"],
               "qn": m["qn"], "kn": m["kn"], "gate": m["gate"], "up": m["up"],
               "down": m["down"]}
        x = self._rms(h, m["in_ln"]) if m["in_ln"] is not None else h
        c, sn = self._rope_np(pos, 1)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(sn)
        q, k, v = self._qkv(lay, x, 1)
        q = self._rope_qk(q, cos_t, sin_t, 1)
        k = self._rope_qk(k, cos_t, sin_t, 1)
        sc = 1.0 / math.sqrt(self.HD)
        h = h + self._attn_out(lay, cache.attn(0, q, k, v, pos, scale=sc), 1)
        if m["post_ln"] is not None:
            x = self._rms(h, m["post_ln"])
            h = h + self._mlp(lay, x)
        hn = m["head_norm"] if m["head_norm"] is not None else self.final_norm
        fin = self._rms(h, hn)
        if m["head"] is not None:
            return np.asarray(m["head"](fin).numpy()).reshape(-1)
        return self._logits(wt.Tensor(wt._contig(fin.data[-1:])))

    # ---- fp16 (plain HF safetensors) loading: run UNquantized, or quantize on load ----
    def _mklin(self, W_out_in, bias=None):
        """One linear at the chosen precision: fp16 (UnquantizedLinear) or int4/int8."""
        if self._qbits:
            return self._gquant(W_out_in, bias)            # quantize-on-load -> QuantizedLinear
        return wt.UnquantizedLinear(W_out_in, bias)        # fp16 weights, fp32 compute

    async def _lin(self, prefix):
        b = await self._np(prefix + ".bias") if (prefix + ".bias") in self._idx else None
        return self._mklin(await self._np(prefix + ".weight"), b)

    @classmethod
    async def from_fp16(cls, base, lmax=None, quantize=None):
        """Load a plain fp16/bf16 HF safetensors dir (no `quantization_config`). Runs the
        model UNquantized by default (fp16 weights, fp32 compute); pass `quantize=4` or `8`
        to instead quantize every linear to int on load (same served fp16 model → int
        inference). Same engine (KV cache, capture/replay, generate) as the int loaders."""
        self = cls(base)
        try:
            return await self._from_fp16(lmax, quantize)
        except BaseException:
            self._abort_build()      # a dead load must not strand half a model in GPU memory
            raise

    async def _from_fp16(self, lmax=None, quantize=None):
        from . import webio
        self.lmax = lmax; self._shard_hdr = {}
        self._qbits = int(quantize) if quantize else 0
        if self._qbits:
            self.bits = self._qbits                        # gs keeps the default 128
        cfg = await webio.read_json(self.base + "config.json")
        self._apply_cfg(cfg)
        self.lmax = self._auto_lmax(lmax, cfg.get("max_position_embeddings"))
        tie = self.tie_embeddings

        self.tok = await self._hf_tokenizer(cfg)

        try:                                                   # sharded models have an index
            imeta = await webio.read_json(self.base + "model.safetensors.index.json")
            self._idx = imeta.get("weight_map")
        except Exception:                                      # single-file model: no index.json
            self._idx = None
        if not self._idx:
            h, _ = await self._hdr("model.safetensors")
            self._idx = {k: "model.safetensors" for k in h}

        t0 = time.perf_counter()
        self.embed = await self._f16_chunked("model.embed_tokens.weight")     # (V,H) fp16
        headW = self.embed if tie else await self._f16_chunked("lm_head.weight")
        self.head = (self._quantize_head(headW) if self._qbits
                     else self._fp16_head(headW))
        self.layers = [await self._build_layer(i, self._lin) for i in range(self.L)]
        self.final_norm = wt.Tensor(await self._np("model.norm.weight"))
        self.load_s = round(time.perf_counter() - t0, 1)
        self._gpu = wt._adam_backend_ready()
        self._fused = wt._adam_backend_ready() or wt._webgl_ready()
        self._init_state()
        self._audit()               # weights must agree with the file
        self._smoke()               # and the first forward must be usable
        return self

    # ---- math ----
    def _rms(self, x, w):
        # One fused dispatch where the backend has it. Written as an expression this is six
        # kernels, and on this stack a dispatch costs about the same whatever its size, so
        # the two norms in every layer add up to a large share of a decode step.
        if getattr(self, "_fused", False) and wt._RMS_FUSED:  # unset while building
            r = wt.rmsnorm(x, w, self.eps)
            if r is not None:
                return r
        return (x / ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()) * w

    def _add_rms(self, residual, update, w):
        """One layer-boundary contract on both GPU backends.

        WebGPU can write the residual sum and its RMS-normalised view in one compute
        dispatch. WebGL reaches the same pair through its fastest portable two-pass path.
        Keeping that difference below this layer avoids either backend changing the model
        graph merely because its atomic primitive set differs.
        """
        # The decode plan's choice is about one row. For more rows "fused" does a strict
        # subset of "composed"'s work -- one pass that reads the pair once and writes both,
        # against an add pass and a norm pass over its result -- so it is not raced there;
        # the plan's "composed" (its exact reference when the plan ran out of time) was
        # costing a prefill two extra dispatches a layer, each ~250 us of host time.
        rows = 1
        for d in tuple(getattr(getattr(residual, "data", residual), "shape", (1,)))[:-1]:
            rows *= int(d)
        route = getattr(self, "_add_rms_execution", "auto")
        if rows > 1 and route == "composed" and wt._adam_backend_ready():
            route = "fused"
        return wt.add_rmsnorm(residual, update, w, self.eps, execution=route)

    def _audit(self):
        """Refuse a model whose weights do not agree with what its own file declares.

        An architecture this has not seen before does not announce itself. It loads, decodes
        at full speed, and answers fluent nonsense -- OLMoE did exactly that through two
        separate assumptions, and neither raised anything. What CAN be established is
        agreement: a norm weight is either one head wide or the whole projection and nothing
        else; a router emits one score per expert; a projection's width is what the head
        counts say. Where the numbers disagree the load is wrong, so it stops here rather
        than at the point someone notices the answers are strange.

        This checks facts, never conventions. Whether routed weights are renormalised cannot
        be read off any weight -- see `_expert_norm`, which reports what it assumed instead.
        """
        H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        bad = []

        def width(lin, attr="Nt"):
            v = getattr(lin, attr, None)
            return int(v) if v is not None else None

        for i, lay in enumerate(getattr(self, "layers", [])):
            if self._is_linear_layer(i):
                continue
            for nm, w, nheads in (("attn_q_norm", lay.get("qn"), NH),
                                  ("attn_k_norm", lay.get("kn"), NKV)):
                if w is None:
                    continue
                n = int(getattr(getattr(w, "data", w), "size", 0))
                if n not in (HD, nheads * HD):
                    bad.append("blk.%d.%s is %d wide, which is neither one head (%d) nor the "
                               "whole projection (%d)" % (i, nm, n, HD, nheads * HD))
            q = width(lay.get("q"))
            if q is not None and q not in (NH * HD, 2 * NH * HD):
                bad.append("blk.%d.attn_q emits %d, not %d (or %d with a fused gate)"
                           % (i, q, NH * HD, 2 * NH * HD))
            for nm in ("k", "v"):
                got = width(lay.get(nm))
                if got is not None and got != NKV * HD:
                    bad.append("blk.%d.attn_%s emits %d, not %d"
                               % (i, nm, got, NKV * HD))
            o_in = width(lay.get("o"), "Kt")
            if o_in is not None and o_in != NH * HD:
                bad.append("blk.%d.attn_output takes %d, not %d" % (i, o_in, NH * HD))
            moe = lay.get("moe")
            if moe:
                ne = int(moe.get("n_experts") or 0)
                g = width(moe.get("gate"))
                if g is not None and ne and g != ne:
                    bad.append("blk.%d router emits %d scores for %d experts" % (i, g, ne))
                k = int(moe.get("top_k") or 0)
                if ne and k > ne:
                    bad.append("blk.%d routes to %d of %d experts" % (i, k, ne))
                st = moe.get("stacked")
                if st and ne:
                    held = int(getattr(st.get("gate_up"), "n_experts", ne) or ne)
                    if held != ne:
                        bad.append("blk.%d holds %d stacked experts, not %d" % (i, held, ne))
            if bad and len(bad) >= 6:
                break                                   # enough to name the problem
        if bad:
            raise ValueError(
                "this model's weights do not match what its file declares, so it would run "
                "and answer wrongly rather than fail:\n  " + "\n  ".join(bad[:6])
                + "\nThis is a gap in webtorch's support for architecture %r, not a broken "
                  "file." % getattr(self, "_arch", "?"))

    def _smoke(self):
        """One forward pass, required to produce usable logits.

        A shader that fails to compile is not an exception here: the platform runs nothing
        and the output buffer is left as it was, which reads as a numerically wrong model
        rather than a broken one. So a freshly loaded model answers once before anyone can
        ask it anything, and that answer has to be finite and to distinguish between tokens
        at all -- the two things every real forward pass does and a kernel that never ran
        does not."""
        capturable = self._capturable()
        # WebGL's growing-cache decoder is the public one-row execution path.
        # Prove a complete decode there without forcing a four-row prefill and
        # its layer-by-layer tuning onto the load critical path. Batched MoE
        # correctness is covered separately and must still pass product prefill
        # regression; this is a load-time smoke test, not that regression.
        ids = [1, 2, 3, 4] if capturable else [1]
        vsz = int(getattr(self.tok, "vocab_size", 0) or 0)
        if vsz:
            ids = [i % vsz for i in ids]
        if capturable:
            self._prefill(ids)
            hidden = self._last_prefill_hidden
        else:
            # WebGL and the other non-capturable paths deliberately have no fixed Kc/Vc
            # arrays.  Their equivalent model-level implementation is the growing cache;
            # forcing the WebGPU prefill here made a perfectly valid packed WebGL model
            # fail its load-time proof with ``CausalLM has no attribute Kc`` before a user
            # could generate a token.  The smoke gate must exercise the same complete path
            # the public generation API will use on this backend.
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            self._kv_forward(ids, 0, cache)
            hidden = self._last_hidden
        lg = np.asarray(self._logits(hidden), np.float32)
        if lg.size == 0:
            raise ValueError("this model produced no logits on its first forward pass")
        if not np.all(np.isfinite(lg)):
            raise ValueError("this model produced non-finite logits on its first forward "
                             "pass, so the load is wrong rather than the prompt")
        if float(lg.max() - lg.min()) == 0.0:
            raise ValueError("this model gave every token in the vocabulary the same score "
                             "on its first forward pass, which is what a kernel that did "
                             "not run looks like, not a model")
        self._kv_drop()               # those rows were the test's, not a conversation's
        return True

    def _expert_norm(self, arch, m):
        """Whether this file's routed layers renormalise their top-k weights.

        The file first: `{arch}.expert_weights_norm` is the answer whenever it is written.
        Then what the architecture is known to do (`_EXPERT_WEIGHTS_NORM`). Only if neither
        says anything is a value assumed, and that is reported rather than taken silently --
        the wrong choice here does not fail, it degrades the answer."""
        said = m("expert_weights_norm", default=None, required=False)
        if said is not None:
            return bool(said)
        if arch in self._EXPERT_WEIGHTS_NORM:
            return self._EXPERT_WEIGHTS_NORM[arch]
        caller = self.__dict__.get("_expert_norm_default")
        if caller is not None:
            return bool(caller)
        print("webtorch: %r does not say whether routed weights are renormalised and this "
              "build has no record for it; assuming they are. Pass "
              "`expert_weights_norm=False` to load() if the replies are fluent but wrong."
              % arch)
        return True

    def _qk_norm(self, t, w, T, nheads):
        """Apply a QK-norm whose width decides how it is applied. `t` is (T, nheads, HD)."""
        if w is None:
            return t
        n = int(getattr(getattr(w, "data", w), "size", 0))
        if n == nheads * self.HD and n != self.HD:
            # Full-width: normalise across the flattened projection, then split heads again.
            return self._rms(t.reshape(T, nheads * self.HD), w).reshape(T, nheads, self.HD)
        return self._rms(t, w)

    def _rot(self, x):
        """rotate_half. With partial rope the swap happens WITHIN the rotated prefix
        (`rope_dim`); the pass-through tail is appended unchanged (it is multiplied by sin=0,
        so its value is irrelevant — only the layout must line up)."""
        hd = self.HD; rd = getattr(self, "rope_dim", hd)
        half = rd // 2
        parts = [wt._slice_last(x, half, rd) * (-1.0), wt._slice_last(x, 0, half)]
        if rd < hd:
            parts.append(wt._slice_last(x, rd, hd))
        return wt.cat(parts, axis=-1)

    def _rope_qk(self, t, cos_t, sin_t, T):
        """Rope for a (heads, T, HD) tensor against (T, HD) cos/sin.

        Every rope site goes through here. They used to write `t * cos + self._rot(t) * sin`
        inline -- three of the four did -- so the fused kernel was reachable from exactly one
        of them, and the backend that needed it most never took it.
        """
        if getattr(self, "_fused", False) and wt._ROPE_FUSED:
            r = wt.rope_decode(t, cos_t, sin_t, self.HD,
                               getattr(self, "rope_dim", self.HD), T)
            if r is not None:
                return r
        return t * cos_t + self._rot(t) * sin_t

    def _rope1(self, t):
        """Rotary embedding at the single decode position, against the persistent cos/sin
        row the capture path keeps. The choice of fused or expression is `_rope_qk`'s."""
        return self._rope_qk(t, self.cos_b, self.sin_b, 1)

    def _rope_np(self, pos, T=1):
        """rope cos/sin of shape (T, HD).

        `pos` is a scalar start position, or -- when the config gives `rope_sections` -- an
        (axes, T) array of one position per axis per token. Multi-axis rope is what lets a
        model place an image in two dimensions instead of flattening it into the text
        sequence: each axis owns a contiguous slice of the rotary dims, and a text token
        simply has the same position on every axis.

        The sections come from the config and nothing else, so a two-axis or four-axis model
        works by arithmetic rather than by being recognised. They cover `rope_dim`, NOT
        `HD` -- with `partial_rotary_factor` below one those differ, and sizing them to HD
        silently rotates the wrong dims.

        With `partial_rotary_factor < 1` only the first `rope_dim` dims are rotated: the tail
        gets cos=1, sin=0, which is the identity under `x*cos + rot(x)*sin`, so the forward
        paths need no special case."""
        rd = getattr(self, "rope_dim", self.HD)
        inv = 1.0 / (self.theta ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
        sec = getattr(self, "rope_sections", None)
        if sec:
            n = len(sec)
            p = np.asarray(pos, np.float64)
            if p.ndim == 0:                        # text-only run: every axis moves together
                p = np.repeat(np.arange(float(p), float(p) + T)[None, :], n, 0)
            ang = p[:, :, None] * inv[None, None, :]          # (axes, T, rd/2)
            emb = np.concatenate([ang, ang], -1)              # (axes, T, rd)
            ec, es = np.cos(emb), np.sin(emb)
            # `emb` is the half-frequencies twice over, so the widths repeat with it
            widths = list(sec) * 2
            cc, ss, st = [], [], 0
            for i, w in enumerate(widths):
                cc.append(ec[i % n, :, st:st + w]); ss.append(es[i % n, :, st:st + w]); st += w
            cos = np.concatenate(cc, -1).astype(np.float32)
            sin = np.concatenate(ss, -1).astype(np.float32)
        else:
            ang = np.arange(pos, pos + T, dtype=np.float64)[:, None] * inv[None, :]
            emb = np.concatenate([ang, ang], -1)              # (T, rd)
            cos = np.cos(emb).astype(np.float32); sin = np.sin(emb).astype(np.float32)
        if rd < self.HD:                                      # pad the un-rotated tail: identity
            pad = self.HD - rd
            cos = np.concatenate([cos, np.ones((cos.shape[0], pad), np.float32)], -1)
            sin = np.concatenate([sin, np.zeros((sin.shape[0], pad), np.float32)], -1)
        return cos, sin

    def _logits(self, hlast):
        return self._logits_tensor(hlast).numpy()[0]

    def _logits_tensor(self, hlast):
        parts = [blk(hlast) for blk in self.head]
        # A one-block head is already the requested vocabulary tensor.  Sending it through
        # cat copied the entire vocabulary once per token (about 600KB on Qwen3-0.6B) for
        # an operation whose input and output are identical.
        return parts[0] if len(parts) == 1 else wt.cat(parts, axis=-1)

    @staticmethod
    def _token_from_device(value):
        value = value.get() if hasattr(value, "get") else value
        return int(np.asarray(value).reshape(-1)[0])

    def _prepare_device_greedy(self, logits):
        """Build/read device argmax and correctness-gate it against full logits once."""
        if self.__dict__.get("_device_argmax_ok") is False:
            return None, None
        try:
            token_buf = wt.vocab_argmax(logits.data)
            token = self._token_from_device(token_buf)
            if "_device_argmax_ok" not in self.__dict__:
                want = int(logits.numpy()[0].argmax())
                self._device_argmax_ok = token == want
                if not self._device_argmax_ok:
                    return None, None
            return token_buf, token
        except Exception:
            self._device_argmax_ok = False
            return None, None

    def _device_greedy_ok(self):
        """Whether argmax may stay on-device without changing generation semantics."""
        sp = getattr(self, "_sampling", None) or {}
        if sp.get("do_sample") or sp.get("constraint") is not None:
            return False
        if float(sp.get("repetition_penalty", 1.0) or 1.0) != 1.0:
            return False
        if float(sp.get("presence_penalty", 0.0) or 0.0):
            return False
        if float(sp.get("frequency_penalty", 0.0) or 0.0):
            return False
        mn = int(sp.get("min_new_tokens", 0) or 0)
        eos = getattr(getattr(self, "tok", None), "eos_ids", ())
        if mn and len(self._seen) - self._gen_start < mn and eos:
            return False
        if getattr(self, "_greedy_execution", "device") != "full":
            return True
        # "full" from a composition tuned while the model's default call SAMPLES (pick mode
        # "full") never compared the two ways of picking a greedy token -- only full logits
        # were a candidate. The greedy batch tuning did compare them, for exactly these
        # calls: a chunk size above zero is device selection measured faster than the
        # host-fed path (0.6B: 6.19 against 6.91 ms a token). Without this a greedy call on
        # such a model read 600 KB of logits back every token although that had lost.
        plan = getattr(self, "decode_plan", None) or {}
        return (plan.get("pick_mode") == "full"
                and int(getattr(self, "_greedy_chunk_size", 0) or 0) > 0)

    def _accept_token(self, tok, con=None, sp=None):
        """Record a selected token and apply the constraint's state transition."""
        sp = sp if sp is not None else (getattr(self, "_sampling", None) or {})
        self._seen.append(int(tok))
        if con is not None:
            self._con_text += self._con_dec.push([int(tok)])
            from . import constrain
            v = self.__dict__.pop("_con_after", None)
            if v == constrain.THEN_END:
                self._con_end = True
            if v == constrain.THEN_FREE or self.__dict__.pop("_con_release", False):
                sp["constraint"] = None
                self._sampling["constraint"] = None
        return int(tok)

    def _head_argmax(self, hlast):
        logits = self._logits_tensor(hlast)
        if self._device_greedy_ok():
            _token_buf, token = self._prepare_device_greedy(logits)
            if _token_buf is not None:
                return self._accept_token(token)
        return CausalLM._pick_tensor(self, logits)

    def _pick_tensor(self, logits):
        """Keep full-vocabulary data in JS for a supported unconstrained GPU readback.

        Python owns only tensor/buffer IDs, generation options and the selected
        scalar. The JS worker reads the GPU result, applies penalties and samples
        in-place in shared memory; the vector never enters a NumPy array.
        """
        sp = getattr(self, "_sampling", None) or {}
        data = getattr(logits, "data", None)
        buffer = getattr(data, "buffer", None)
        shape = getattr(data, "shape", ())
        module = type(buffer).__module__ if buffer is not None else ""
        direct = (sp.get("constraint") is None and buffer is not None
                  and len(shape) == 2 and int(shape[0]) == 1
                  and int(getattr(data, "offset", 0)) == 0
                  and np.dtype(getattr(data, "dtype", np.float32)) == np.dtype(np.float32)
                  and int(getattr(buffer, "size", -1)) >= int(shape[1]))
        if direct and module == "wgpy_backends.webgl.webgl_buffer":
            texture = buffer.texture_shape
            direct = (texture.type == 5126 and texture.elements_per_pixel == 1)
        if not direct or module not in ("wgpy_backends.webgl.webgl_buffer",
                                       "wgpy_backends.webgpu.webgpu_buffer"):
            return self._pick(logits.numpy()[0])

        sampling = bool(sp.get("do_sample"))
        rng = sp.get("rng")
        options = {
            "doSample": sampling,
            "temperature": float(sp.get("temperature", 1.0) or 1.0),
            "topP": float(sp.get("top_p", 1.0) or 1.0),
            "topK": int(sp.get("top_k", 0) or 0),
            "minP": float(sp.get("min_p", 0.0) or 0.0),
            "random": float(rng.random() if rng is not None else np.random.random())
                      if sampling else 0.0,
            "repetitionPenalty": float(sp.get("repetition_penalty", 1.0) or 1.0),
            "presencePenalty": float(sp.get("presence_penalty", 0.0) or 0.0),
            "frequencyPenalty": float(sp.get("frequency_penalty", 0.0) or 0.0),
            "blockEos": bool(int(sp.get("min_new_tokens", 0) or 0)
                             > len(self._seen) - self._gen_start),
            "eosIds": list(self.tok.eos_ids) if self.tok.eos_ids else [],
        }
        # Candidate for the unconstrained full-softmax case only.  The source
        # format, model name and task are irrelevant; a different top-k/p,
        # history penalty or EOS mask keeps the established JS sampler.  `auto`
        # selects the per-device/vocabulary result only after it was measured
        # and placed in the kernel profile; an unmeasured device stays on JS.
        sample_key = ("vocab_sample_full", int(shape[1]), "webgpu")
        sample_route = getattr(self, "_sample_execution", "auto")
        if sample_route == "auto":
            sample_route = wt._TUNED.get(sample_key, "js")
        if sample_route == "gpu" and module == "wgpy_backends.webgpu.webgpu_buffer":
            # Python schedules a scalar route only. The JS worker validates the
            # sampling mode, and the main-thread JS/GPU own metadata, dispatch,
            # reduction and four-byte readback. Unsupported modes use JS sampling.
            options["execution"] = "gpu"
        if not getattr(self, "_js_sampler_initialized", False):
            options["seen"] = list(self._seen)
        if module == "wgpy_backends.webgl.webgl_buffer":
            from wgpy_backends.webgl.platform import get_platform
            token = get_platform().sampleLogits(buffer.buffer_id, int(shape[1]), options)
        else:
            from wgpy_backends.webgpu.platform import get_platform
            token = get_platform().sampleLogits(
                buffer.buffer_id, buffer.texture_shape.byte_length,
                int(shape[1]), options)
        self._js_sampler_initialized = True
        return self._accept_token(token)

    def _pick(self, logits):
        """Turn logits into a token id using the current sampling parameters.

        Greedy when `temperature <= 0` (or no sampling params were given), otherwise
        temperature + top-k/top-p/min-p sampling, with an optional repetition penalty and
        an optional output constraint. Parameters come from `generate(...)`, from the
        model's own generation_config.json, or from `load(...)`."""
        sp = getattr(self, "_sampling", None) or {}
        self._js_sampler_initialized = False
        lg = np.asarray(logits, np.float32)
        rp = float(sp.get("repetition_penalty", 1.0) or 1.0)
        if rp != 1.0 and self._seen:
            idx = np.unique(np.asarray(self._seen, np.int64))
            idx = idx[(idx >= 0) & (idx < lg.size)]
            v = lg[idx]
            lg = lg.copy()
            lg[idx] = np.where(v > 0, v / rp, v * rp)   # the usual asymmetric form
        # OpenAI-style additive penalties, which are a different knob from the multiplicative
        # repetition_penalty above and compose with it: presence is a flat charge for having
        # appeared at all, frequency scales with the count.
        pp = float(sp.get("presence_penalty", 0.0) or 0.0)
        fp = float(sp.get("frequency_penalty", 0.0) or 0.0)
        if (pp or fp) and self._seen:
            ids = np.asarray(self._seen, np.int64)
            ids = ids[(ids >= 0) & (ids < lg.size)]
            if ids.size:
                idx, cnt = np.unique(ids, return_counts=True)
                lg = lg.copy()
                lg[idx] -= pp + fp * cnt.astype(np.float32)
        mn = int(sp.get("min_new_tokens", 0) or 0)
        if mn and len(self._seen) - self._gen_start < mn and self.tok.eos_ids:
            eos = np.asarray(self.tok.eos_ids, np.int64)
            eos = eos[(eos >= 0) & (eos < lg.size)]
            lg = lg.copy()
            lg[eos] = -np.inf                           # not allowed to stop yet
        con = sp.get("constraint")
        # A constraint with nothing to say here is still asked -- about ONE token instead of
        # the whole vocabulary.
        #
        # Both constrained paths below must rank 150k logits and decode ~64 candidates
        # before they can ask anything, and that is most of what a token costs: measured on
        # a 27B, a reply with tool names constrained ran at 0.8 tok/s against 5.2 without.
        # In a reply of prose every one of those questions goes to a constraint that permits
        # everything on offer.
        #
        # Skipping it outright is NOT sound, and the soundness check said so: `allows` judges
        # `text + piece`, so one piece carrying `<tool_call>{"name": "r` turns a dormant text
        # into a name that has to be checked -- 189 refusals in 3024 checks over texts this
        # would have skipped. So the PIECE is what gets asked about. Pick the token
        # unconstrained, decode that one piece, ask once. Allowed is the same answer the full
        # path would reach, because that path walks the ranking and takes the first
        # acceptable token, which this is. Refused, and the full path runs having lost only
        # one decode.
        #
        # `getattr`, not a direct call: `build` returns an object that merely HAS `allows` as
        # it is, without wrapping it, so a caller's own constraint is not required to inherit
        # anything -- and must not be required to grow a method to keep working. No `dormant`
        # means "ask me about everything", which is what it got before this existed.
        _dormant = getattr(con, "dormant", None)
        _allows = getattr(con, "allows", None)
        tok = None
        if (con is not None and callable(_dormant) and callable(_allows)
                and not callable(getattr(con, "decide", None))
                and _dormant(self._con_text)):
            t1 = self._sample(lg, sp) if sp.get("do_sample") else int(lg.argmax())
            if _allows(self._con_text, self._piece1(t1)):
                tok = t1
        if tok is not None:
            pass
        elif con is not None and callable(getattr(con, "decide", None)):
            tok = self._pick_decided(lg, con, sp)
        elif con is not None:
            tok = self._pick_constrained(lg, con, sp)
        else:
            tok = self._sample(lg, sp) if sp.get("do_sample") else int(lg.argmax())
        return self._accept_token(tok, con, sp)

    def _sample(self, lg, sp):
        from . import lm_engine
        mp = float(sp.get("min_p", 0.0) or 0.0)
        if mp > 0:                              # keep only what is within min_p of the top
            m = lg.max()
            keep = lg >= m + math.log(max(mp, 1e-9))
            lg = np.where(keep, lg, -np.inf)
        return int(lm_engine.sample_nucleus(
            lg / max(float(sp.get("temperature", 1.0) or 1.0), 1e-5),
            top_p=float(sp.get("top_p", 1.0) or 1.0),
            top_k=int(sp.get("top_k", 0) or 10 ** 9), rng=sp.get("rng")))

    def _piece1(self, tid):
        """The text of one token, memoised.

        The candidate window is re-decoded on every step -- 64 tokenizer calls per token --
        and the same handful of ids come back over and over, since a constrained reply keeps
        choosing from the same small set. Measured before this: a constraint cost 6.5ms a
        token on a step that is otherwise 12.5ms, and the callback itself was a lambda. It is
        the decoding that costs, and a token's text never changes.
        """
        memo = self.__dict__.get("_piece1_memo")
        if memo is None:
            memo = self.__dict__["_piece1_memo"] = {}
        got = memo.get(tid)
        if got is None:
            got = memo[tid] = self.tok.decode([tid])
        return got

    def _pieces(self):
        """id -> text for the whole vocabulary, built once. Only the widened constraint search
        needs it, and only for models that are actually asked to satisfy a constraint."""
        if getattr(self, "_piece_tab", None) is None:
            dec = self.tok.decode
            self._piece_tab = [dec([i]) for i in range(len(self.tok.dec))]
        return self._piece_tab

    def _pick_decided(self, lg, con, sp):
        """One question per token: here is everything on offer, best first -- what is
        permitted, and what happens next?

        The other form (`_pick_constrained`) asks about one candidate at a time and is
        bounded at the window size while the model's own top choices are acceptable; when
        they are not it widens, and a constraint that permits only something rare gets asked
        about the whole vocabulary -- measured at 153,408 questions for a single token. This
        form is asked once and answers with the permitted SET, so a small known set (a tool
        name, an enum value, a grammar's next terminals) costs one call and no widening.

        A permitted string that matches nothing on offer is looked up in the vocabulary and
        used anyway. That is how a caller forces a token the model ranked nowhere, which is
        the case a predicate cannot express: it can only reject what it is shown.
        """
        from . import constrain
        n = int(lg.size)
        k0 = max(1, int(sp.get("constraint_candidates", 64) or 64))
        for k in (min(k0, n), min(k0 * 16, n), n):
            if k < n:
                idx = np.argpartition(-lg, k - 1)[:k]
                order = idx[np.argsort(-lg[idx])]
            else:
                order = np.argsort(-lg)
            ids = [int(t) for t in order]
            cands = [self._piece1(t) for t in ids]
            v, opts = con.decide(self._con_text, cands)
            v = constrain.verdict(v)
            if v.then == constrain.THEN_END and not v.take:
                # About the REPLY, not about a token: complete as it stands.
                self._con_end = True
                return int(self.tok.eot)
            if not v.allow:
                if v.then == constrain.THEN_FREE:
                    # None of these, and nothing further to ask -- so the model's own ranking
                    # decides from here.
                    self._con_release = True
                    return int(lg.argmax())
                continue                      # nothing here; widen and ask again
            keep = [t for t, p in zip(ids, cands) if constrain.piece_ok(p, opts)]
            if not keep and opts:
                # Ranked nowhere, so nothing here ordered them: put the model's own
                # preference back on the forced set before picking from it.
                keep = self._ids_for(opts)
                keep.sort(key=lambda t: -lg[t])
            if not keep:
                continue
            if not sp.get("do_sample"):
                got = keep[0]
            else:
                sub = np.full_like(lg, -np.inf)
                arr = np.asarray(keep, np.int64)
                sub[arr] = lg[arr]
                got = int(self._sample(sub, sp))
            if v.then != constrain.THEN_ASK:
                self._con_after = v.then
            return got
        return int(lg.argmax())

    def _ids_for(self, options):
        """Token ids that spell a step towards one of `options`, for a permitted string the
        model ranked nowhere.

        Costs one pass over the vocabulary, the first time a caller actually forces
        something -- 150k single-token decodes here -- so it is built on demand and kept.
        """
        from . import constrain
        index = self.__dict__.get("_piece_index")
        if index is None:
            index = self.__dict__["_piece_index"] = list(enumerate(self._pieces()))
        return [i for i, p in index if p and constrain.piece_ok(p, options)]

    def _pick_constrained(self, lg, con, sp):
        """Sample from just the candidates the constraint accepts.

        The likely handful is checked first: scoring a 250k-token vocabulary through a
        constraint costs more than the model step that produced the logits, and usually buys
        nothing, since what matters is which tokens are allowed rather than the exact
        ordering of ones the model was never going to pick.

        But the window WIDENS instead of giving up. The opening tokens of a constrained reply
        are often ones the model ranks nowhere -- asked in prose for JSON it wants to answer
        in prose, and `{` may sit far outside the top 64. Falling back to an unconstrained
        pick there does not just lose one token: that token makes every later prefix invalid,
        so no candidate is ever acceptable again and the constraint is silently dropped for
        the whole reply. Widening to the full vocabulary is the difference between a
        constraint that holds and one that only appears to. The plain pick remains the last
        resort, for a constraint nothing at all can satisfy -- that must not hang."""
        n = int(lg.size)
        k0 = max(1, int(sp.get("constraint_candidates", 64) or 64))
        allowed, order = [], None
        for k in (min(k0, n), min(k0 * 16, n), n):
            if k < n:
                idx = np.argpartition(-lg, k - 1)[:k]
                order = idx[np.argsort(-lg[idx])]
                pieces = None
            else:
                order = np.argsort(-lg)
                pieces = self._pieces()
            from . import constrain
            allowed, after, released = [], {}, False
            for t in order:
                ti = int(t)
                if released:
                    # Told to stop asking: the rest of this step's candidates are as good as
                    # the model ranked them.
                    allowed.append(ti)
                    continue
                v = constrain.verdict(
                    con.allows(self._con_text,
                               pieces[ti] if pieces is not None and ti < len(pieces)
                               else self._piece1(ti)))
                if v.then == constrain.THEN_END and not v.take:
                    # A statement about the REPLY, not about this token: complete as it
                    # stands. The remaining candidates are not worth scoring.
                    self._con_end = True
                    return int(self.tok.eot)
                if not v.allow:
                    if v.then == constrain.THEN_FREE:
                        # Not this one, and nothing further to ask about -- so the release
                        # takes effect from here, including the rest of this scan.
                        released = True
                        self._con_release = True
                    continue
                allowed.append(ti)
                if v.then != constrain.THEN_ASK:
                    after[ti] = v.then
            if allowed:
                break
        if not allowed:
            return int(lg.argmax())
        if not sp.get("do_sample"):
            got = allowed[0]
        else:
            sub = np.full_like(lg, -np.inf)
            sub[np.asarray(allowed, np.int64)] = lg[np.asarray(allowed, np.int64)]
            got = int(self._sample(sub, sp))
        self._con_after = after.get(got)
        return got

    # The KV cache is the only memory that grows with context: fp32, K and V, NKV × HD per
    # token per full-attention layer (recurrent layers keep a fixed-size state instead). A
    # context nobody asked for must not eat memory nobody budgeted, so an undeclared size
    # is trimmed to this; an explicit `lmax` always wins.
    _KV_BUDGET = 2 << 30

    # What "nobody said" means for temperature. A model that ships a generation_config.json
    # states its own -- Qwen3 asks for 0.6, and that file always wins over this. This is only
    # for the case where NOTHING states one, which is every model loaded from a bare .gguf:
    # the file carries the weights and the chat template but no sampling settings, so before
    # this the sampler fell through to zero and decoded greedily. Chat models are trained and
    # evaluated sampled, and greedy is where the degenerate repeats live.
    _DEFAULT_TEMPERATURE = 0.6

    # Rows the KV cache starts with, and the step it grows by. The floor is small enough that
    # loading a model costs almost nothing in cache, and doubling means a long conversation
    # reallocates a handful of times rather than once per turn.
    _KV_FLOOR = 512

    # Rows kept ahead of the reply, rather than room for the longest reply it could give.
    # `max_new` unset means "until the model stops", which `_plan_length` turns into the whole
    # remaining context -- reserving that up front is exactly the eager allocation this
    # replaced. The decode loop grows the cache when it actually reaches the end.
    _KV_HEADROOM = 256

    # Does a routed layer renormalise its top-k weights to sum to 1?
    #
    # Both conventions exist and they compute different numbers, so a model run under the
    # wrong one answers fluently and wrongly -- the failure that is hardest to notice. A
    # GGUF may say (`{arch}.expert_weights_norm`) and then the file wins; most files do not
    # say, including every one tested here, so where it is silent this records what the
    # architecture's own implementation does. Two measured cases, same code, same prompt:
    # OLMoE renormalised answers "The answer 1 "The Sky's the Earth the (Sydney-)#athy-o's"";
    # not renormalised it answers "The sky appears blue because the Earth's atmosphere
    # scatters sunlight in all directions".
    #
    # An architecture that is not listed keeps the commoner convention and SAYS so, because
    # a silent guess is the thing this exists to prevent.
    _EXPERT_WEIGHTS_NORM = {
        "qwen3moe": True,          # config norm_topk_prob=true
        "qwen2moe": False,         # config default norm_topk_prob=false
        "olmoe": False,            # OlmoeConfig norm_topk_prob=false
        "llama": True,             # Mixtral converts to this arch; it divides by the sum
        "deepseek2": True,
    }

    # Decode cost used to scale with the context because a step scanned the whole KV buffer
    # and re-uploaded a full-length mask every token. Both are gone -- the fused attention
    # reads the live length from the step control block, and the mask is only built for the
    # general path -- so a step now costs the same at 512 as at 4096 (measured: 35.6 vs
    # 35.0 ms). Context is therefore a memory question again, and only memory caps it.

    def _auto_lmax(self, requested, declared):
        """The context size to load with: an explicit request, else what the model declares,
        capped by the KV memory budget."""
        n_full = sum(1 for t in getattr(self, "layer_types", []) if t == "full_attention")
        n_full = n_full or self.L
        per_tok = 2 * n_full * self.NKV * self.HD * 4
        cap = max(512, self._KV_BUDGET // per_tok)
        if requested:
            got = max(512, min(int(requested), cap))
            if got < int(requested):
                print("webtorch: context %d -> %d tokens (KV cache budget %.1f GB)"
                      % (int(requested), got, self._KV_BUDGET / 2 ** 30))
            return got
        want = int(declared) if declared else 2048
        got = max(512, min(want, cap))
        if got < want:
            print("webtorch: context %d -> %d tokens (KV cache budget %.1f GB)"
                  % (want, got, self._KV_BUDGET / 2 ** 30))
        return got

    def _plan_length(self, ids, max_new, max_length, truncate):
        """Reconcile the requested lengths with what this model can actually hold.

        `lmax` -- the KV cache the model was loaded with -- is a hard ceiling on prompt plus
        generation, and running past it silently corrupts the cache. A prompt that does not
        fit is either truncated from the front (a chat history wants its most recent turns)
        or refused, and `max_new` is clipped to whatever room is left.

        No `max_new` at all means "as much as the model allows": its own generation_config
        when it ships one, otherwise the rest of the context — the model still stops itself
        at its end token; the number only bounds it."""
        lmax = int(self.lmax)
        d = getattr(self, "gen_defaults", {}) or {}
        if max_new is not None:
            mx = int(max_new)
        elif d.get("max_new_tokens"):
            mx = int(d["max_new_tokens"])
        else:
            mx = max(1, lmax - len(ids))
        if max_length is None:
            max_length = d.get("max_length")
        if max_length:
            mx = min(mx, int(max_length) - len(ids))
        ids = list(ids)
        if len(ids) >= lmax:
            if not truncate:
                raise ValueError(
                    "prompt is %d tokens but this model was loaded with a %d-token context; "
                    "load it with a larger lmax, shorten the prompt, or pass truncate=True"
                    % (len(ids), lmax))
            keep = max(1, lmax - max(1, min(mx, lmax // 4)))
            ids = ids[-keep:]
        return ids, max(1, min(mx, lmax - len(ids)))

    def _stop_now(self):
        """True when this generation should end early.

        Two reasons, checked between tokens: an output constraint says the text is complete,
        or someone asked the SDK to stop (`webtorch.cancel()`). The second is a plain read of
        the flag, not the raising checkpoint the IO path uses -- a stopped generation returns
        the tokens it has, it does not throw them away."""
        from .webio import cancel_requested
        if cancel_requested():
            return True
        if self.__dict__.get("_con_end"):
            return True
        con = (getattr(self, "_sampling", None) or {}).get("constraint")
        return bool(con is not None and con.finished(self._con_text))

    # ---- tool calling, as API ----------------------------------------------------------
    #
    # A caller decides WHICH tools to offer and implements them; that is its business. How a
    # model is told they exist, how what it writes back is read, and how a result is returned
    # to it are all facts about the model, so they live here and are read from its own
    # template. Nothing above this line needs to know any of it.

    def split_reasoning(self, text):
        """Separate a reply's reasoning span from its answer:
        {"reasoning", "answer", "open"}. `open` means the span was never closed -- the reply
        is still inside it.

        Which tags mark a reasoning span is a fact about models, not about any one caller, so
        every form the SDK knows is recognised here rather than one being spelled out at each
        call site. A live stream does not need this (`stream(channels=True)` labels the
        pieces as they arrive); this is for text that was stored as one string, where the
        answer has to be told from the reasoning again afterwards.

        A dangling closer with no opener is read as "all of it was reasoning": that is what a
        reply saved with only part of the template text looks like, and the alternative is
        showing the reasoning as the answer.
        """
        txt = text if isinstance(text, str) else ""
        for o, c in _THINK_TAGS:
            if o not in txt and c in txt:
                txt = o + txt
            if not txt.startswith(o):
                continue
            k = txt.find(c)
            if k >= 0:
                return {"reasoning": txt[len(o):k].strip(),
                        "answer": txt[k + len(c):].lstrip(), "open": False}
            return {"reasoning": txt[len(o):].strip(), "answer": "", "open": True}
        return {"reasoning": None, "answer": txt, "open": False}

    def tools_supported(self):
        """Whether this model can be given tools at all.

        The one question a caller has. WHICH definition shape its template reads, how it
        writes a call, how a result gets back to it -- all of that is settled inside
        `generate(tools=...)` and the methods below, and none of it is a caller's business:
        a decision made from it would be a decision made about a chat template, which is
        exactly what this library exists to not make anyone do.
        """
        return self.tok.tools_shape() is not None

    def parse_tool_calls(self, text, tools=None):
        """Every tool call in `text`, in order, each with the span it occupies."""
        from . import toolcall
        return toolcall.parse(text, tools, self.tok.tool_call_format(tools))

    def tool_calls(self, text, tools=None):
        """Just the calls, without spans."""
        from . import toolcall
        return toolcall.calls(text, tools, self.tok.tool_call_format(tools))

    def strip_tool_calls(self, text, tools=None):
        """`text` with the call markup removed -- what a reader should be shown."""
        from . import toolcall
        return toolcall.strip(text, tools, self.tok.tool_call_format(tools))

    def render_tool_call(self, name, args, tools=None):
        """One call written the way this model writes them."""
        from . import toolcall
        return toolcall.render(name, args, self.tok.tool_call_format(tools))

    def tool_round_messages(self, assistant_text, calls, results):
        """The turns to append after a round of tool calls -- what the model said, then what
        each call returned -- shaped the way THIS model's template reads them.

        A caller supplies the reply text, the calls it ran and their results; whether the
        calls travel as structured `tool_calls` or as the model's own text, and which id
        field ties a result to its call, are read from the template here.
        """
        from . import toolcall
        f = self.tok.tool_result_format()
        return toolcall.round_messages(assistant_text, calls, results,
                                       id_shape=f["call_id_shape"],
                                       keeps_name=f["keeps_name"],
                                       id_field=f["id_field"], via=f["via"])

    def suggest_tool(self, name, tools, args=None):
        """Which registered tools a name that matched none could plausibly have meant, best
        first. Evidence for a caller's own decision -- nothing is run on it."""
        from . import toolcall
        return toolcall.suggest(name, tools, args)

    def tool_result_message(self, call, content):
        """The message that carries one tool's result back, shaped for this model."""
        from . import toolcall
        f = self.tok.tool_result_format()
        return toolcall.result_message(call, content, keeps_name=f["keeps_name"],
                                       id_field=f["id_field"], via=f["via"])

    def _tool_name_constraint(self, tools, constraint):
        """Materialise `constraint="tool_names"` into a constraint over this call's tools.

        ASKED FOR, never assumed. Holding a model to the names it was given changes what it
        is allowed to say, and whether that trade is worth making is the caller's decision,
        not this library's: the guard also makes a model that would have called the wrong
        tool sometimes call none at all (measured 2 of 8 -> 4 of 8), and which of those is
        the better failure depends on what the caller does next. So it is offered and not
        installed -- pass `constraint="tool_names"`, alone or beside another constraint.

        It is the repair that sampling cannot be. A model that has talked itself into
        `run_2.py` is CERTAIN of it by the time it writes the call -- p=0.9998 on the wrong
        token, measured -- so nothing that reweights the distribution recovers it; only
        removing the token from the candidate set does.

        Both halves are derived, never written down: the names come from this call's own
        `tools`, the delimiter from the model's own template (`tool_call_format`). A model
        whose call format cannot be read is left unconstrained rather than guessed at.
        """
        want = constraint
        rest = None
        if isinstance(constraint, (list, tuple)):
            want = "tool_names" if "tool_names" in constraint else None
            rest = [c for c in constraint if c != "tool_names"] or None
        if want is not True and want != "tool_names":
            return constraint
        constraint = rest
        names = []
        for t in (tools or []):
            if not isinstance(t, dict):
                continue
            f = t.get("function")
            n = (f.get("name") if isinstance(f, dict) else None) or t.get("name")
            if n:
                names.append(str(n))
        if not names:
            return constraint
        fmt = self.tok.tool_call_format(tools) or {}
        if not fmt.get("open"):
            return constraint
        from . import constrain
        tc = constrain.ToolNameConstraint(names, fmt.get("open"), fmt.get("payload"))
        if constraint is None:
            return tc
        return constrain.AllOf([constrain.build(constraint), tc])

    def _set_sampling(self, temperature=None, top_p=None, top_k=None, seed=None, do_sample=None,
                      repetition_penalty=None, min_p=None, constraint=None,
                      min_new_tokens=None, prompt_ids=None, presence_penalty=None,
                      frequency_penalty=None, stop=None, **_ignored):
        """Install generation parameters. Defaults from the model's generation_config.json (or
        from load()) are kept; anything passed to generate() overrides them for that call."""
        from . import constrain
        base = dict(getattr(self, "gen_defaults", {}) or {})
        for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k),
                     ("seed", seed), ("do_sample", do_sample), ("min_p", min_p),
                     ("repetition_penalty", repetition_penalty),
                     ("presence_penalty", presence_penalty),
                     ("frequency_penalty", frequency_penalty),
                     ("min_new_tokens", min_new_tokens)):
            if v is not None:
                base[k] = v
        # An ABSENT temperature is not a temperature of zero. It used to be read as one, and
        # the difference is the whole reply: a model loaded from a bare .gguf has no
        # generation_config.json to state what it wants, so nothing set a temperature, so
        # every such model decoded greedily -- no top_p, no top_k, no repetition penalty.
        # Greedy has no way out of a loop it walks into, and small models walk into them:
        # asked for a product of two large numbers, a 0.6B answered with eight hundred
        # consecutive "6"s. Adding two characters to the prompt was enough to flip it between
        # a sane answer and that, which is what a missing sampler looks like from outside.
        if "temperature" not in base:
            base["temperature"] = self._DEFAULT_TEMPERATURE
        # An EXPLICIT zero is still greedy, and says so louder than any default: a model whose
        # generation_config ships `do_sample: true` would otherwise keep sampling through it,
        # and the reply would not reproduce even though the caller asked for the deterministic
        # one. Every mainstream API reads a zero temperature this way.
        if float(base.get("temperature") or 0) <= 0:
            base["do_sample"] = False
        elif base.get("do_sample") is None:
            base["do_sample"] = True
        if base.get("seed") is not None:
            base["rng"] = np.random.default_rng(int(base["seed"]))
        stops = stop if stop is not None else base.get("stop")
        if isinstance(stops, str):
            stops = [stops]
        base["stop"] = stops
        con = constrain.build(constraint if constraint is not None else base.get("constraint"))
        if stops:
            sc = constrain.StopConstraint(stops)
            con = sc if con is None else constrain.AllOf([con, sc])
        if con is not None:
            con.reset()
        base["constraint"] = con
        self._sampling = base
        self._js_sampler_initialized = False
        # Repetition penalty counts the prompt too, as it does elsewhere; the constraint
        # sees only what this generation produced.
        self._seen = list(prompt_ids or [])
        self._gen_start = len(self._seen)
        self._con_text = ""
        # A verdict belongs to the generation that produced it. Left set, `"stop"` or
        # `"last"` from one reply would end the NEXT one before its first token.
        self.__dict__.pop("_con_end", None)
        self.__dict__.pop("_con_after", None)
        self.__dict__.pop("_con_release", None)
        # Constraint and stop matching read the text produced so far, so it has to be decoded
        # the same careful way the stream is -- a half-written character would otherwise put
        # a replacement char in the middle of the string a stop sequence is matched against.
        self._con_dec = self.tok.stream_decoder()
        return base

    def _init_state(self):
        L, NKV, HD, H = self.L, self.NKV, self.HD, self.H
        NH = self.NH
        # `lmax` is the CONTEXT this model may run to -- the contract `_plan_length` holds a
        # prompt to. `kv_cap` is how many rows are actually on the device right now, and it
        # starts at the floor and grows to meet what a generation asks for. They used to be
        # the same number, which meant the whole context was paid for the moment the model
        # loaded: a 30B at a 10922-token context took 2GB of KV before a word was typed, and
        # a 0.6B took the same 2GB -- five times the model itself -- for a chat that uses a
        # few hundred tokens. On a machine already holding 13GB of weights that is the
        # difference between fitting and swapping.
        LMAX = self.kv_cap = min(int(self.lmax), self._KV_FLOOR)
        # Settle the shape constants now, while a load is already taking seconds, rather than
        # inside the first reply. Left to happen lazily it lands in the middle of the first
        # generation and shows there: measured, the first reply ran at 9.6 tok/s against the
        # 109 every reply after it. A cost paid where someone is already waiting is not the
        # same cost as one paid where they are reading.
        # A first load must not spend minutes searching a combinatorial decode space.
        # The same full tuner remains callable explicitly with no deadline for offline
        # benchmarking; the interactive path has a wall-clock budget across all stages.
        self._warm_deadline = time.perf_counter() + 8.0
        self._warm_shapes()
        _load_stage("warm-state")
        # recurrent states for linear-attention layers (fixed size, independent of length)
        self.lin_state = [lay["linear"].new_state() if lay.get("linear") else None
                          for lay in getattr(self, "layers", [])]
        # Only full-attention layers hold a context-length K/V cache; a recurrent layer
        # carries its fixed-size state instead, so allocating a cache for it would be pure
        # waste — on a stack that is mostly linear that is most of the KV memory.
        self._kv_i = []; n = 0
        for i in range(L):
            if self._is_linear_layer(i):
                self._kv_i.append(-1)
            else:
                self._kv_i.append(n); n += 1
        if self._gpu:                                  # capture path: fixed scatter cache + persistent inputs
            # `_empty`, not `_zeros`: the latter is host-backed in this WgPy build
            # (np.zeros then a staging upload), so every row costs RAM on the host as well
            # as on the device -- 56 of these at a grown context is a WASM heap that goes
            # from 613MB to 1.5GB and, being an emscripten heap, never gives it back. No row
            # is ever read before it is written (see `_kv_reserve` and the note in
            # `generate`), so there is nothing for the zero-fill to protect.
            # Half as many elements when the cache holds packed halves -- see `kv_f16`.
            # It is still an fp32 tensor to the host; only the shaders read it as pairs.
            KVW = wt.kv_cache_hd(HD)
            self.Kc = [wt.Tensor(wt._empty((NKV, LMAX, KVW))) for _ in range(n)]
            self.Vc = [wt.Tensor(wt._empty((NKV, LMAX, KVW))) for _ in range(n)]
            self.h_in = wt.Tensor(np.zeros((1, H), np.float32))
            self.cos_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.sin_b = wt.Tensor(np.zeros((1, HD), np.float32))
            self.mask_b = wt.Tensor(np.zeros((1, 1, LMAX), np.float32))
            self.ctl = xp.asarray(np.array([0, 1, NKV, HD, LMAX], np.int32))
            # Does the fused decode attention cover this model's shapes? If it does, it reads
            # the length from `ctl` and never looks at the mask -- and the mask is the single
            # most expensive thing in a decode step, because keeping it current means
            # allocating and uploading LMAX floats EVERY token. That upload, not the
            # attention itself, is what made a large context slow down short conversations.
            self._fused_attn = False
            if wt._GQA_FUSED and n:
                probe = wt.gqa_decode(wt.Tensor(np.zeros((NH, 1, HD), np.float32)),
                                      self.Kc[0], self.Vc[0], self.mask_b, 1.0, ctl=self.ctl)
                self._fused_attn = probe is not None
            _load_stage("warm-oracle")
            self._tune_decode_composition()
            _load_stage("warm-step")
            wt._gpu_release_idle_pool()
            self._warm_decode_step()
            _load_stage("warm-greedy")
            if time.perf_counter() + 3.0 < self._warm_deadline:
                self._tune_greedy_chunks()
            else:
                self._greedy_chunk_size = 0
                self._greedy_chunk_capture_ready = False
                self.decode_plan["greedy_chunk"] = 0
                self.decode_plan["greedy_chunk_budget_skipped"] = True
        elif wt._webgl_ready():
            # WebGL has no command capture, but it still owns an independent complete-API
            # choice (most visibly full-vocabulary readback versus its fragment argmax).
            # Do not inherit the WebGPU plan: this backend has different primitives and a
            # different cost balance.
            wt._gpu_release_idle_pool()
            self._tune_webgl_decode_composition()

    def _stored_linears(self):
        """All encoded Linear leaves reachable from this model, without naming a format."""
        seen = set(); found = []

        def walk(value):
            oid = id(value)
            if oid in seen:
                return
            seen.add(oid)
            if isinstance(value, wt.GGMLLinear):
                found.append(value); return
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            elif isinstance(value, wt.Module):
                for item in vars(value).values():
                    walk(item)

        walk(self.layers); walk(self.head)
        return found

    def _decode_pick_mode(self):
        """The selection path the next call will actually execute.

        Load-time tuning runs before ``_set_sampling``.  In that case derive the same
        default that ``_set_sampling`` will install, rather than benchmarking a device
        argmax for a model whose normal API call samples the full vocabulary.
        """
        if getattr(self, "_sampling", None) is not None:
            return "device" if CausalLM._device_greedy_ok(self) else "full"
        defaults = getattr(self, "gen_defaults", None) or {}
        temperature = float(defaults.get("temperature", CausalLM._DEFAULT_TEMPERATURE) or 0)
        sample = defaults.get("do_sample")
        sample = False if temperature <= 0 else (True if sample is None else bool(sample))
        if (sample or defaults.get("constraint") is not None
                or float(defaults.get("repetition_penalty", 1.0) or 1.0) != 1.0
                or float(defaults.get("presence_penalty", 0.0) or 0.0)
                or float(defaults.get("frequency_penalty", 0.0) or 0.0)
                or int(defaults.get("min_new_tokens", 0) or 0)):
            return "full"
        return "device"

    def _decode_sampler_route(self, pick_mode=None):
        """The token-selection implementation actually reached by this decoder.

        Full-vocabulary JS readback and GPU sampling have different queue,
        storage and synchronization costs.  A complete-decoder profile measured
        with one cannot justify reusing the same upper-level plan with the other.
        Unsupported sampling options always fall back to JS in the worker, even
        when the requested execution override says GPU.
        """
        mode = pick_mode or CausalLM._decode_pick_mode(self)
        if mode != "full" or not getattr(self, "_gpu", False):
            return "none"
        sp = getattr(self, "_sampling", None) or getattr(self, "gen_defaults", None) or {}
        temperature = float(sp.get("temperature", CausalLM._DEFAULT_TEMPERATURE) or 0)
        sampling = sp.get("do_sample")
        sampling = temperature > 0 if sampling is None else bool(sampling)
        eligible = (sampling and temperature > 0 and sp.get("constraint") is None
                    and float(sp.get("top_p", 1) or 1) >= 1
                    and int(sp.get("top_k", 0) or 0) <= 0
                    and float(sp.get("min_p", 0) or 0) <= 0
                    and float(sp.get("repetition_penalty", 1) or 1) == 1
                    and not float(sp.get("presence_penalty", 0) or 0)
                    and not float(sp.get("frequency_penalty", 0) or 0)
                    and not int(sp.get("min_new_tokens", 0) or 0))
        if not eligible:
            return "js"
        route = getattr(self, "_sample_execution", "auto")
        if route == "auto":
            route = wt._TUNED.get(("vocab_sample_full", int(getattr(self, "VOCAB", 0)),
                                   "webgpu"), "js")
        return "gpu" if route == "gpu" else "js"

    def _decode_composition_key(self, linears, qshapes, pick_mode=None,
                                backend="webgpu"):
        """Name a measured whole-decoder plan by source content identity and topology.

        The browser keeps profiles under its backend/device key; this additional key
        prevents one model's upper-layer choice from leaking to another model with a
        different graph. It is derived from the source identifier and tensor *shapes*,
        never from a recognised model-name or task category.
        """
        source = getattr(self, "_gguf", None) or getattr(self, "base", None)
        if not source:
            return None
        import hashlib
        topology = {
            "source": str(source),
            "pick_mode": pick_mode or CausalLM._decode_pick_mode(self),
            "geometry": [int(getattr(self, name, 0) or 0) for name in
                         ("L", "H", "NH", "NKV", "HD", "VOCAB", "n_experts",
                          "top_k", "bits", "gs")],
            "layers": list(getattr(self, "layer_types", ()) or ()),
            "weights": sorted((str(m.type_name), int(m.Kt), int(m.Nt))
                              for m in linears),
            "q4_shapes": sorted((int(k), int(n)) for k, n in qshapes),
        }
        if backend == "webgpu":
            topology["sampler_route"] = CausalLM._decode_sampler_route(
                self, pick_mode or CausalLM._decode_pick_mode(self))
        digest = hashlib.sha256(json.dumps(topology, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()[:24]
        # WebGPU v2 measured the actual sampling path but did not distinguish JS
        # from GPU sampling. A profile from one is not evidence for the other's
        # upper-level composition. WebGL still uses its unchanged v1 topology.
        version = "decode_plan_v3" if backend == "webgpu" else "decode_plan_v1"
        return (version, backend, digest)

    def _activate_decode_profile_for_call(self):
        """Switch complete-API plans when a call changes the token-selection path.

        A greedy profile is not evidence for a full-logit sampling call, or vice versa.
        Reuse only a measured matching plan. If one is absent, the existing correct
        arithmetic stays in place and its missing performance validation is explicit.
        """
        if (not getattr(self, "_gpu", False)
                or not getattr(self, "decode_plan", None)
                or not self._capturable()):
            return
        mode = CausalLM._decode_pick_mode(self)
        if mode != "device" and getattr(self, "_greedy_chunk_capture_ready", False):
            CausalLM._release_idle_greedy_capture(
                self, wt._adam_kernel["platform"], mode)
        sampler_route = CausalLM._decode_sampler_route(self, mode)
        if (self.decode_plan.get("pick_mode") == mode
                and self.decode_plan.get("sampler_route") == sampler_route):
            # A preceding call may have requested an unprofiled mode.  The
            # measured plan is active again now; do not leave its diagnostic
            # status claiming that the missing mode is still in use.
            self.decode_plan["active_pick_mode"] = mode
            self.decode_plan["active_sampler_route"] = sampler_route
            self.decode_plan.pop("mode_profile_missing", None)
            return
        linears = self._stored_linears()
        qshapes = sorted({(m.Kt, m.Nt) for m in linears
                          if m.type_name == "Q4_K" and m.execution == "auto"})
        key = CausalLM._decode_composition_key(self, linears, qshapes, mode)
        saved = wt._TUNED.get(key) if key is not None else None
        plan = saved.get("plan") if isinstance(saved, dict) else None
        if (isinstance(plan, (tuple, list)) and len(plan) in (8, 9)
                and isinstance(plan[1], (tuple, list))
                and len(plan[1]) == len(qshapes)):
            CausalLM._tune_decode_composition(self, interactive=True)
        else:
            self.decode_plan["active_pick_mode"] = mode
            self.decode_plan["active_sampler_route"] = sampler_route
            self.decode_plan["mode_profile_missing"] = True

    def _tune_decode_composition(self, *, interactive=None):
        """Choose the fastest correct complete decode composition for this model instance.

        Lower APIs retain their own ``auto`` routes.  This layer is allowed to override
        them because candidates that win alone can lose when adjacent commands share GPU
        occupancy or enter a captured graph.  Timing therefore replays the entire captured
        trunk+head and the winning plan is exposed as ``decode_plan``.
        """
        pick_mode = CausalLM._decode_pick_mode(self)
        sampler_route = CausalLM._decode_sampler_route(self, pick_mode)
        _missing = object()
        _pick_fields = ("_sampling", "_seen", "_gen_start", "_con_text", "_con_dec",
                        "_js_sampler_initialized", "_con_end", "_con_after",
                        "_con_release")
        _saved_pick = {name: self.__dict__.get(name, _missing) for name in _pick_fields}

        def reset_probe_sampling():
            """Give each plan the same real sampling semantics and random stream."""
            import copy
            old = _saved_pick["_sampling"]
            settings = (old if old is not _missing and old is not None
                        else (getattr(self, "gen_defaults", None) or {}))
            names = ("temperature", "top_p", "top_k", "do_sample", "min_p",
                     "repetition_penalty", "presence_penalty", "frequency_penalty",
                     "min_new_tokens", "constraint", "stop")
            options = {name: (copy.deepcopy(settings[name]) if name in ("constraint", "stop")
                              else settings[name]) for name in names if name in settings}
            CausalLM._set_sampling(self, seed=1729, prompt_ids=[], **options)

        incumbent = getattr(self, "decode_plan", None)
        self.capture_ready = False
        self._decode_graph_signature = None
        self._decode_graph_logits = None
        self._decode_graph_token = None
        if interactive is None:
            interactive = hasattr(self, "_warm_deadline")
        deadline = (getattr(self, "_warm_deadline", float("inf"))
                    if interactive else float("inf"))
        self._add_rms_execution = "auto"
        self._greedy_execution = "device"
        self.decode_plan = {"add_rmsnorm": "auto", "q4_decode": "auto",
                            "qkv": "auto", "gate_up": "auto",
                            "qk_norm_rope": "auto",
                            "kv_write": "auto",
                            "head_shape": "auto",
                            "projection_shapes": {"o": "auto", "down": "auto"},
                            "greedy_pick": "device", "pick_mode": pick_mode,
                            "sampler_route": sampler_route}
        if not self._gpu or not self._capturable():
            return self.decode_plan
        class _WarmLimit(Exception):
            pass
        try:
            import itertools as _it
            import statistics as _stats
            plat = wt._adam_kernel["platform"]
            stored_linears = self._stored_linears()
            q4 = [m for m in stored_linears
                  if m.type_name == "Q4_K" and m.execution == "auto"]
            qshapes = sorted({(m.Kt, m.Nt) for m in q4})
            qindex = {shape: i for i, shape in enumerate(qshapes)}
            profile_key = CausalLM._decode_composition_key(
                self, stored_linears, qshapes, pick_mode)
            # Exhaust every per-shape combination while it is tractable.  More shapes use
            # the lower layer's own auto vector plus the two uniform overrides; this keeps
            # load time bounded without collapsing all shapes into one global vote.
            if not qshapes:
                qroutes = [()]
            elif len(qshapes) <= 6:
                qroutes = [("auto",) * len(qshapes)]
                qroutes += list(_it.product(("stored", "dp4a"), repeat=len(qshapes)))
            else:
                qroutes = [("auto",) * len(qshapes), ("stored",) * len(qshapes),
                           ("dp4a",) * len(qshapes)]
            qroutes = list(dict.fromkeys(qroutes))
            pick_routes = (("device", "full") if pick_mode == "device"
                           else ("full",))
            plans = [(a, q, p) for q in qroutes for a in ("fused", "composed")
                     for p in pick_routes]

            # A plan that won for greedy selection may still be the best plan when a
            # caller samples. Keep it as a candidate, but require a comparison using
            # this call's actual full-logit or device-pick path before reusing it.
            prior_plan = None
            if isinstance(incumbent, dict):
                routes = incumbent.get("q4_decode")
                if isinstance(routes, dict):
                    routes = tuple(routes.get("%dx%d" % shape, "auto")
                                   for shape in qshapes)
                elif routes == "auto":
                    routes = ("auto",) * len(qshapes)
                elif isinstance(routes, (tuple, list)):
                    routes = tuple(routes)
                if isinstance(routes, tuple) and len(routes) == len(qshapes):
                    prior_plan = (incumbent.get("add_rmsnorm"), routes,
                                  incumbent.get("greedy_pick"), incumbent.get("qkv"),
                                  incumbent.get("gate_up"), incumbent.get("qk_norm_rope"),
                                  incumbent.get("kv_write"), incumbent.get("head_shape"))
                    shapes = incumbent.get("projection_shapes")
                    if isinstance(shapes, dict):
                        prior_plan += ((shapes.get("o", "auto"),
                                        shapes.get("down", "auto")),)
                    if (prior_plan[0] not in ("fused", "composed")
                            or any(x not in ("auto", "stored", "dp4a") for x in routes)
                            or prior_plan[2] not in ("device", "full")):
                        prior_plan = None

            def select(plan, original_width=False):
                self._add_rms_execution = plan[0]
                self._greedy_execution = plan[2]
                self._qkv_execution = plan[3] if len(plan) > 3 else "auto"
                self._gate_up_execution = plan[4] if len(plan) > 4 else "auto"
                self._qk_norm_rope_execution = plan[5] if len(plan) > 5 else "auto"
                self._kv_write_execution = plan[6] if len(plan) > 6 else "auto"
                self._head_shape_execution = plan[7] if len(plan) > 7 else "auto"
                for block in self.head:
                    if isinstance(block, wt.GGMLLinear):
                        block.decode_shape = self._head_shape_execution
                # Physical leaf shapes are independently usable through GGMLLinear, but
                # the containing decode API may prefer a different combination.  Reset
                # both groups for every candidate so a previous probe cannot leak into
                # the next capture or into the original-width correctness oracle.
                projection_shapes = plan[8] if len(plan) > 8 else ("auto", "auto")
                for lay in self.layers:
                    for key, route in zip(("o", "down"), projection_shapes):
                        block = lay.get(key)
                        if isinstance(block, wt.GGMLLinear):
                            block.decode_shape = route
                # `decode_execution` is an override for this containing API only.  Reset
                # every encoded leaf first so an exact-oracle run cannot leak its forced
                # stored route into the candidate that follows it.  The oracle forces ALL
                # formats to their original-width path, not only the Q4_K leaves which have
                # a DP4A phase-two candidate today.
                for lin in stored_linears:
                    lin.decode_execution = "stored" if original_width else None
                if not original_width:
                    for lin in q4:
                        lin.decode_execution = plan[1][qindex[(lin.Kt, lin.Nt)]]

            def captured_sample(plan, steps):
                """Time the real upper API route, including its active sampler."""
                reset_probe_sampling()
                select(plan); self._reset_linear_state(); self._set_inputs(0, 0)
                plat.beginCapture("decode")
                logits = self._decode_fwd()
                token = (wt.vocab_argmax(logits.data)
                         if plan[2] == "device" and pick_mode == "device" else None)
                (token.get() if token is not None else logits.numpy())
                plat.endCapture()
                nxt = (self._accept_token(self._token_from_device(token))
                       if token is not None else CausalLM._pick_tensor(self, logits))
                t0 = time.perf_counter()
                for pos in range(1, steps + 1):
                    self._set_inputs(nxt, pos); plat.replay("decode")
                    nxt = (self._accept_token(self._token_from_device(token))
                           if token is not None else CausalLM._pick_tensor(self, logits))
                return (time.perf_counter() - t0) / steps

            # Correctness is judged against the exact original-width composition, never
            # against whichever candidate happens to be enumerated first.  In particular,
            # a lower-level ``auto`` is allowed to select DP4A in phase two, so it is not a
            # suitable numerical oracle for the other upper-layer combinations.
            reference_plan = ("composed", ("stored",) * len(qshapes), "full",
                              "separate", "separate", "composed", "separate", None)
            if interactive and profile_key is not None:
                cached = wt._TUNED.get(profile_key)
                if isinstance(cached, dict):
                    saved = cached.get("plan")
                    if (isinstance(saved, (list, tuple)) and len(saved) in (8, 9)
                            and isinstance(saved[1], (list, tuple))
                            and len(saved[1]) == len(qshapes)):
                        chosen = (saved[0], tuple(saved[1]), *saved[2:])
                        select(chosen)
                        projection_shapes = (chosen[8] if len(chosen) > 8
                                             else ("auto", "auto"))
                        self.decode_plan = {
                            "add_rmsnorm": chosen[0],
                            "q4_decode": {"%dx%d" % shape: chosen[1][i]
                                          for i, shape in enumerate(qshapes)},
                            "greedy_pick": chosen[2], "qkv": chosen[3],
                            "gate_up": chosen[4], "qk_norm_rope": chosen[5],
                            "kv_write": chosen[6], "head_shape": chosen[7],
                            "projection_shapes": dict(zip(("o", "down"),
                                                           projection_shapes)),
                            "median_ms": cached.get("median_ms"),
                            "pick_mode": pick_mode,
                            "sampler_route": sampler_route,
                            "profile_reused": True,
                        }
                        return self.decode_plan
            # Semantic correctness is sequential.  A candidate that agrees only on the
            # first token can diverge on the very next state, so it is not eligible for
            # timing.  Four steps cover the state transition, KV write/read and reuse of a
            # generated token while keeping load-time tuning bounded on 30B-class models.
            semantic_steps = 4
            def semantic_trace(plan, original_width=False):
                """Read sequential logits and select with the real fixed-seed sampler.

                Four eager forwards on a large MoE paid Python/JS dispatch and shader
                setup four times (20.2 s on the local 30B). A replay changes the input
                token and position but retains the exact same arithmetic and KV writes.
                Keep the whole-logit check; only remove repeated host dispatch overhead.
                """
                reset_probe_sampling()
                select(plan, original_width=original_width)
                self._reset_linear_state()
                trace = []
                token = 0
                self._set_inputs(token, 0)
                try:
                    if original_width:
                        _load_stage("warm-record")
                    plat.beginCapture("decode_semantic")
                    try:
                        logits = self._decode_fwd()
                        picked_gpu = (wt.vocab_argmax(logits.data)
                                      if plan[2] == "device" and pick_mode == "device"
                                      else None)
                        logits.numpy()
                    finally:
                        plat.endCapture()
                    if original_width:
                        _load_stage("warm-replay")
                    for pos in range(semantic_steps):
                        if pos:
                            self._set_inputs(token, pos)
                            plat.replay("decode_semantic")
                        got = np.asarray(logits.numpy()[0], np.float32).copy()
                        picked = (self._accept_token(self._token_from_device(picked_gpu))
                                  if picked_gpu is not None
                                  else CausalLM._pick_tensor(self, logits))
                        trace.append((got, picked))
                        token = picked
                    return trace
                finally:
                    # This temporary graph must not pin its scratch buffers for the
                    # lifetime of the loaded model. Same-name recapture releases them.
                    plat.beginCapture("decode_semantic")
                    plat.endCapture()

            # A full-model record is indivisible and can exceed the entire interactive
            # budget (16.6 s on the local 30B), even when 3 s remain at entry.  The
            # loader cannot measure its cost in advance. Interactive load therefore
            # selects the original-width exact composition without starting an oracle;
            # explicit offline tuning remains available with interactive=False.
            if interactive:
                raise _WarmLimit()
            reference_trace = semantic_trace(reference_plan, original_width=True)
            if any(not np.all(np.isfinite(logits)) for logits, _ in reference_trace):
                raise RuntimeError("exact stored decode produced non-finite logits")
            _load_stage("warm-search")

            last_good = reference_plan
            tested_candidates = 0

            def check_budget():
                if time.perf_counter() >= deadline:
                    raise _WarmLimit()

            semantic_cache = {}

            def correct(plan):
                """Full logits and token sequence against the original-width oracle."""
                nonlocal tested_candidates
                if plan in semantic_cache:
                    return semantic_cache[plan]
                try:
                    check_budget()
                    approximate = any(route in ("auto", "dp4a") for route in plan[1])
                    limit = 0.03 if approximate else 2e-5
                    for (reference, reference_pick), (got, picked) in zip(
                            reference_trace, semantic_trace(plan)):
                        if not np.all(np.isfinite(got)):
                            semantic_cache[plan] = False
                            return False
                        scale = max(1e-6, float(np.abs(reference).max()))
                        if (float(np.abs(got - reference).max()) / scale > limit
                                or picked != reference_pick):
                            semantic_cache[plan] = False
                            return False
                    semantic_cache[plan] = True
                    tested_candidates += 1
                    return True
                except _WarmLimit:
                    raise
                except Exception:
                    semantic_cache[plan] = False
                    return False

            def tournament(candidates):
                """Screen all combinations, then require paired evidence for replacement."""
                nonlocal last_good
                valid = []
                for plan in candidates:
                    check_budget()
                    if correct(plan):
                        valid.append(plan)
                    # The interactive pass compares an orthogonal slice of the same
                    # addressable routes.  An explicit offline call without a deadline
                    # still examines the entire candidate set.
                    if interactive and len(valid) >= 4:
                        break
                if not valid:
                    raise RuntimeError("no correct full-decode composition survived validation")
                samples = {}
                for plan in valid:
                    check_budget()
                    samples[plan] = []
                baseline = valid[0]
                # Correctness-screening is not performance-screening. The old offline
                # search timed only the first four valid plans, silently excluding
                # later physical combinations from the claimed upper-layer optimum.
                finalists = valid if not interactive else valid[:4]
                if baseline not in finalists:
                    finalists.append(baseline)
                # Five paired rounds are the minimum for the exact sign test to accept a
                # repeatable positive result (5/5 wins -> p=1/32). No fixed gain cutoff.
                for r in range(5):
                    order = finalists if not (r & 1) else list(reversed(finalists))
                    for plan in order:
                        check_budget()
                        samples[plan].append(captured_sample(plan, 2))
                chosen = baseline
                for plan in sorted(finalists, key=lambda p: _stats.median(samples[p])):
                    if plan != chosen and wt._paired_faster(samples, plan, chosen):
                        chosen = plan
                last_good = chosen
                return chosen, samples

            # First settle the lower auto routes plus residual/pick/Q4 interaction.  Then
            # measure the layer-level parallel projection overrides at that whole-decode
            # point, and finally re-run every first-stage candidate under the winning
            # projection pair.  This is the upward pass the API needs: local winners remain
            # callable, while a peer combination can replace them only at this layer.
            _load_stage("tuning", done=0, total=8)
            initial, _ = tournament(plans)
            _load_stage("tuning", done=1, total=8)
            # The fused projection layer has its own locally measured workgroup shape, but
            # the complete decoder may prefer another one once QKV, MLP and head commands
            # share a graph.  Keep those physical routes addressable so this upper layer can
            # override the local `auto` rather than being trapped by it.
            group_modes = ("auto", "fused", "separate", "fused:default",
                           "fused:balanced", "fused:compact", "fused:narrow")
            def parallel_capable(parts):
                parts = tuple(parts)
                return bool(parts and all(isinstance(x, wt.GGMLLinear) for x in parts)
                            and len({x.type_name for x in parts}) == 1
                            and wt._GGML_TYPES[parts[0].type_name][4] is None)

            attention_layers = [lay for i, lay in enumerate(self.layers)
                                if not self._is_linear_layer(i)]
            qkv_modes = (group_modes if any(parallel_capable(
                (lay.get("q"), lay.get("k"), lay.get("v")))
                for lay in attention_layers) else ("auto",))
            # Routed MoE never calls the dense gate/up pair.  Enumerating seven values for
            # a variable that cannot reach the graph measured the identical 30B plan seven
            # times and turned load-time tuning into a 20+ minute apparent hang.
            dense_layers = [lay for lay in self.layers if not lay.get("experts")]
            gate_modes = (group_modes if any(parallel_capable(
                (lay.get("gate"), lay.get("up"))) for lay in dense_layers)
                else ("auto",))
            qk_modes = (("auto", "composed", "fused")
                        if attention_layers and all(lay.get("qn") is not None
                                                    and lay.get("kn") is not None
                                                    for lay in attention_layers)
                        else ("composed",))
            kv_modes = (("auto", "separate", "fused")
                        if attention_layers else ("separate",))
            group_plans = [initial + (q, g, r, "auto")
                           for q in qkv_modes for g in gate_modes for r in qk_modes]
            grouped, _ = tournament(group_plans)
            _load_stage("tuning", done=2, total=8)
            # K/V has a locally addressable auto choice, then participates in a full upward
            # pass.  Re-run the peer projection/QK combinations under its winner: a fused
            # scatter that wins alone is not assumed to win beside a different fused QKV,
            # and a projection choice made beside separate writes is likewise not frozen.
            kv_grouped, _ = tournament([
                grouped[:6] + (route,) for route in kv_modes])
            _load_stage("tuning", done=3, total=8)
            regrouped, _ = tournament([
                initial + (q, g, r, kv_grouped[6])
                for q in qkv_modes for g in gate_modes for r in qk_modes])
            composed, _ = tournament([
                regrouped[:6] + (route,) for route in kv_modes])
            _load_stage("tuning", done=4, total=8)
            head_modes = (("auto", None, "balanced", "compact", "narrow", "shortk")
                          if self.head and all(isinstance(x, wt.GGMLLinear)
                                               for x in self.head) else ("auto",))
            headed, _ = tournament([composed + (route,) for route in head_modes])
            _load_stage("tuning", done=5, total=8)
            # The vocabulary projection is adjacent to every candidate above in the
            # captured graph.  Re-run the peer layer choices under its winner, then give
            # KV and the head one final replacement opportunity before the outer Q4/
            # residual/pick pass.  This prevents a head-local winner from being treated as
            # automatically optimal in the containing decoder.
            regrouped_head, _ = tournament([
                initial + (q, g, r, composed[6], headed[7])
                for q in qkv_modes for g in gate_modes for r in qk_modes])
            rekv, _ = tournament([
                regrouped_head[:6] + (route, regrouped_head[7])
                for route in kv_modes])
            _load_stage("tuning", done=6, total=8)
            final_group, _ = tournament([
                rekv[:7] + (route,) for route in head_modes])
            _load_stage("tuning", done=7, total=8)
            refined = [plan + (final_group[3], final_group[4], final_group[5],
                               final_group[6], final_group[7])
                       for plan in plans]
            # Keep the previously proven fully fused composition in the *whole-API*
            # tournament.  A sequence of one-axis tournaments can prune it before all
            # of its peers are present, even though the combination wins end to end.
            # Its physical QKV shapes are separate candidates: the locally fastest
            # fused workgroup need not be the fastest after capture and sampling.
            # These are capability candidates, never model-name defaults; every one
            # still faces the sequential semantic oracle and paired device timings.
            restored = []
            if ("fused" in qkv_modes and self.head
                    and all(isinstance(block, wt.GGMLLinear) for block in self.head)):
                restored = [
                    ("fused", ("stored",) * len(qshapes), "full", qroute,
                     "separate", "fused", "fused", "compact")
                    for qroute in ("fused", "fused:balanced", "fused:compact")]
            chosen, samples = tournament(list(dict.fromkeys(
                ([prior_plan] if prior_plan is not None else []) + restored + refined)))
            # A lower matvec may select its own fastest shape, while the attention-output
            # and MLP-down peers can interact through graph scheduling and cache pressure.
            # Compare the complete decode API with both groups independently and together,
            # then revisit the adjacent residual/QKV composition under the winning pair.
            # No model identity or fixed percentage threshold enters this choice.
            o_modes = (("auto", "balanced", "compact", "narrow")
                       if any(isinstance(lay.get("o"), wt.GGMLLinear)
                              for lay in self.layers) else ("auto",))
            down_modes = (("auto", "balanced", "compact", "narrow")
                          if any(isinstance(lay.get("down"), wt.GGMLLinear)
                                 for lay in self.layers) else ("auto",))
            shape_pairs = list(_it.product(o_modes, down_modes))
            base = chosen[:8]
            shaped, _ = tournament([base + (pair,) for pair in shape_pairs])
            recomposed, _ = tournament([
                (add, *shaped[1:3], qkv, *shaped[4:8], shaped[8])
                for add in ("composed", "fused") for qkv in qkv_modes])
            chosen, samples = tournament([
                recomposed[:8] + (pair,) for pair in shape_pairs])
            _load_stage("tuning", done=8, total=8)
            select(chosen)
            qchosen = ({"%dx%d" % shape: chosen[1][i]
                        for i, shape in enumerate(qshapes)} if qshapes else {})
            self.decode_plan = {"add_rmsnorm": chosen[0], "q4_decode": qchosen,
                                "qkv": chosen[3],
                                "gate_up": chosen[4],
                                "qk_norm_rope": chosen[5],
                                "kv_write": chosen[6],
                                "head_shape": chosen[7],
                                "projection_shapes": dict(zip(("o", "down"), chosen[8])),
                                "greedy_pick": chosen[2],
                                "pick_mode": pick_mode,
                                "sampler_route": sampler_route,
                                "median_ms": round(_stats.median(samples[chosen]) * 1000, 4)
                                if chosen in samples and samples[chosen] else None}
            if profile_key is not None:
                wt._TUNED[profile_key] = {
                    "plan": [chosen[0], list(chosen[1]), *chosen[2:]],
                    "median_ms": self.decode_plan["median_ms"],
                }
            return self.decode_plan
        except _WarmLimit:
            # A candidate is never selected merely because the budget ran out: use the
            # last fully checked and measured composition, or the exact original-width
            # oracle if the search never completed a tournament.
            chosen = locals().get("last_good", locals().get("reference_plan"))
            if chosen is not None:
                select(chosen, original_width=chosen == reference_plan)
                self.decode_plan = {
                    "add_rmsnorm": chosen[0], "q4_decode": chosen[1],
                    "greedy_pick": chosen[2],
                    "qkv": chosen[3] if len(chosen) > 3 else "auto",
                    "gate_up": chosen[4] if len(chosen) > 4 else "auto",
                    "qk_norm_rope": chosen[5] if len(chosen) > 5 else "auto",
                    "kv_write": chosen[6] if len(chosen) > 6 else "auto",
                    "head_shape": chosen[7] if len(chosen) > 7 else "auto",
                    "projection_shapes": dict(zip(("o", "down"),
                                                   chosen[8] if len(chosen) > 8
                                                   else ("auto", "auto"))),
                    "pick_mode": pick_mode,
                    "sampler_route": sampler_route,
                    "budget_limited": True,
                    "tested_candidates": locals().get("tested_candidates", 0),
                }
            return self.decode_plan
        except Exception as exc:
            self._add_rms_execution = "auto"
            self._greedy_execution = "device"
            self._qkv_execution = "auto"
            self._gate_up_execution = "auto"
            self._qk_norm_rope_execution = "auto"
            self._kv_write_execution = "auto"
            self._head_shape_execution = "auto"
            for block in getattr(self, "head", ()):
                if isinstance(block, wt.GGMLLinear):
                    block.decode_shape = "auto"
            for lay in getattr(self, "layers", ()):
                for key in ("o", "down"):
                    block = lay.get(key)
                    if isinstance(block, wt.GGMLLinear):
                        block.decode_shape = "auto"
            for lin in locals().get("stored_linears", ()):
                lin.decode_execution = None
            self.decode_plan["error"] = "%s: %s" % (type(exc).__name__, exc)
            raise RuntimeError("decode composition tuning failed: %s" % exc) from exc
        finally:
            # Semantic traces and candidate timing write synthetic token rows into the
            # live KV buffers.  A later chat must not treat the pre-tuning `_kv_ids` as
            # evidence that those rows still hold its prefix.
            try:
                CausalLM._kv_drop(self)
                self._reset_linear_state()
            finally:
                for name, value in _saved_pick.items():
                    if value is _missing:
                        self.__dict__.pop(name, None)
                    else:
                        self.__dict__[name] = value

    def _tune_webgl_decode_composition(self, *, interactive=True):
        """Choose WebGL's complete decode API route from WebGL measurements.

        Exact packed scalar/vec4 choices are made independently inside each weight
        operation.  At this top layer the remaining real alternative is whether greedy
        selection stays in a fragment shader and returns four bytes, or reads the complete
        vocabulary back.  Both execute the same eager growing-cache decoder and are gated
        on a sequential token trace selected by the full-logit reference.
        """
        pick_mode = CausalLM._decode_pick_mode(self)
        self._add_rms_execution = "auto"
        self._qkv_execution = "auto"
        self._gate_up_execution = "auto"
        self._greedy_execution = "full"
        self.decode_plan = {"backend": "webgl", "add_rmsnorm": "composed",
                            "packed_exact": "device-measured",
                            "qkv": "auto", "gate_up": "auto",
                            "qk_norm_rope": "composed",
                            "kv_write": "composed",
                            "head_shape": "device-measured",
                            "greedy_pick": "full", "pick_mode": pick_mode}
        if self._gpu or not wt._webgl_ready():
            return self.decode_plan
        # A complete WebGL tournament runs hundreds of eager 30B forwards. Keep it
        # available for explicit offline calibration, but never put that search on
        # every interactive load. A matching device/model profile is the only
        # evidence that lets the fast load claim an optimal measured route; when it
        # is absent, expose the exact baseline as pending rather than silently
        # claiming that the first candidate won.
        linears = CausalLM._stored_linears(self)
        profile_key = CausalLM._decode_composition_key(
            self, linears, (), pick_mode, backend="webgl")
        cached = wt._TUNED.get(profile_key) if profile_key is not None else None
        saved = cached.get("plan") if isinstance(cached, dict) else None
        if (interactive and isinstance(saved, (tuple, list)) and len(saved) == 2
                and saved[0] in ("auto", "separate", "fused")
                and saved[1] in (("full", "device") if pick_mode == "device"
                                 else ("full",))):
            self._gate_up_execution, self._greedy_execution = saved
            self.decode_plan.update({"gate_up": saved[0], "greedy_pick": saved[1],
                                     "median_ms": cached.get("median_ms"),
                                     "profile_reused": True})
            return self.decode_plan
        if interactive:
            self.decode_plan["profile_pending"] = True
            return self.decode_plan
        import statistics as _stats
        saved_seen = list(getattr(self, "_seen", ()))
        saved_sampling = self.__dict__.get("_sampling", None)
        saved_gen_start = self.__dict__.get("_gen_start", None)
        samples = {}

        def run(plan):
            gate_route, pick_route = plan
            self._gate_up_execution = gate_route
            self._greedy_execution = pick_route
            self._seen = []
            self._gen_start = 0
            if pick_mode == "device":
                self._sampling = {"do_sample": False, "temperature": 0.0}
            else:
                settings = dict(saved_sampling or getattr(self, "gen_defaults", None) or {})
                temperature = float(settings.get("temperature", CausalLM._DEFAULT_TEMPERATURE)
                                    or 0)
                settings["temperature"] = temperature
                settings["do_sample"] = (False if temperature <= 0 else
                                         bool(settings.get("do_sample", True)))
                settings["constraint"] = None
                settings["rng"] = np.random.default_rng(1729)
                self._sampling = settings
            self._reset_linear_state()
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            t0 = time.perf_counter()
            tokens = []
            token = 0
            for pos in range(4):
                token = int(self._kv_forward([token], pos, cache))
                tokens.append(token)
            return (time.perf_counter() - t0) / len(tokens), tuple(tokens)

        try:
            # Compile and establish the semantic oracle before measuring either route.
            reference_plan = ("separate", "full")
            _, reference = run(reference_plan)
            candidates = []
            for gate_route in ("auto", "separate", "fused"):
                for pick_route in (("full", "device") if pick_mode == "device"
                                   else ("full",)):
                    plan = (gate_route, pick_route)
                    _, picked = run(plan)
                    if (picked == reference
                            and (pick_route != "device"
                                 or self.__dict__.get("_device_argmax_ok", True))):
                        candidates.append(plan)
                        samples[plan] = []
            if reference_plan not in candidates:
                raise RuntimeError("the exact WebGL reference route did not reproduce")
            for r in range(9):
                order = candidates if not (r & 1) else list(reversed(candidates))
                for plan in order:
                    elapsed, picked = run(plan)
                    if picked != reference:
                        raise RuntimeError("%s WebGL route selected %r, expected %r" %
                                           (plan, picked, reference))
                    samples[plan].append(elapsed)
            chosen = wt._measured_choice(samples, candidates, default=reference_plan)
            self._gate_up_execution, self._greedy_execution = chosen
            self.decode_plan["gate_up"] = chosen[0]
            self.decode_plan["greedy_pick"] = chosen[1]
            self.decode_plan["median_ms"] = round(_stats.median(samples[chosen]) * 1000, 4)
            if profile_key is not None:
                wt._TUNED[profile_key] = {
                    "plan": list(chosen), "median_ms": self.decode_plan["median_ms"]}
            return self.decode_plan
        except Exception as exc:
            self._greedy_execution = "full"
            self._gate_up_execution = "auto"
            self.decode_plan["error"] = "%s: %s" % (type(exc).__name__, exc)
            return self.decode_plan
        finally:
            self._reset_linear_state()
            self._seen = saved_seen
            if saved_sampling is None:
                self.__dict__.pop("_sampling", None)
            else:
                self._sampling = saved_sampling
            if saved_gen_start is None:
                self.__dict__.pop("_gen_start", None)
            else:
                self._gen_start = saved_gen_start

    def _can_chunk_greedy(self):
        """Whether this instance can feed a captured greedy step entirely on-device.

        This is capability/shape driven.  A tied, single-block Q6_K table lets the output
        head's original packed buffer serve as the embedding table too; split heads and
        other stored formats keep the ordinary one-token API until they have their own
        exact row decoder.  No model name or model category participates.
        """
        return bool(
            self._gpu and self._capturable()
            and getattr(self, "_tied_embed_head", False)
            and len(getattr(self, "head", ())) == 1
            and isinstance(self.head[0], wt.GGMLLinear)
            and self.head[0].type_name == "Q6_K"
            and int(self.head[0].Kt) == int(self.H)
            and int(self.head[0].Nt) == int(self.embed.shape[0])
            and not any(self._is_linear_layer(i) for i in range(self.L)))

    def _ensure_chunk_buffers(self, count=4):
        count = max(1, int(count))
        if getattr(self, "_chunk_capacity", 0) >= count:
            return
        self._chunk_capacity = count
        self._chunk_tokens = xp.empty((count + 1,), np.int32)
        self._chunk_cos = wt._empty((count, self.HD))
        self._chunk_sin = wt._empty((count, self.HD))

    def _embedding_row_candidates(self):
        candidates = ["transposed"]
        if (getattr(self, "_embed_packed_rows", None) is not None
                and int(getattr(self, "_embed_row_words", 0)) > 0):
            candidates.append("compact")
        return tuple(candidates)

    def _embedding_row_key(self):
        head = self.head[0]
        return ("embedding_row", head.type_name, int(head.Kt), int(head.Nt),
                int(getattr(self, "_embed_row_words", 0)))

    def _decode_chunk_input(self, step=0, increment=False, execution=None):
        """Decode one exact embedding row through an explicitly addressable layout."""
        head = self.head[0]
        route = self._embedding_row_execution if execution is None else execution
        if route == "auto":
            route = wt._TUNED.get(self._embedding_row_key(), "transposed")
        if route not in self._embedding_row_candidates():
            raise ValueError("embedding row execution is unavailable: %s" % route)
        compact = route == "compact"
        packed = self._embed_packed_rows if compact else head.packed
        stride = self._embed_row_words if compact else head.Nt
        return wt.q6k_decode_input(
            self._chunk_tokens, packed, head.Kt, stride,
            self._chunk_cos, self._chunk_sin,
            self.h_in, self.cos_b, self.sin_b, self.ctl,
            step=step, increment=increment, layout=route)

    def _tune_embedding_row(self):
        """Choose a bit-exact row layout locally, before whole-decoder composition."""
        key = self._embedding_row_key()
        if key in wt._TUNED:
            return wt._TUNED[key]
        candidates = self._embedding_row_candidates()
        self._ensure_chunk_buffers(1)
        probe = min(37, int(self.embed.shape[0]) - 1)
        self._set_chunk_inputs(probe, 0, 1)
        reference = None; valid = []
        for route in candidates:
            try:
                self._decode_chunk_input(execution=route)
                got = np.asarray(self.h_in.numpy()[0], np.float32).copy()
                if reference is None:
                    reference = got
                if np.array_equal(got, reference):
                    valid.append(route)
            except Exception:
                continue
        if not valid:
            raise RuntimeError("no exact Q6_K embedding-row route")
        samples = {route: [] for route in valid}

        def bench(route):
            t0 = time.perf_counter()
            for _ in range(16):
                self._decode_chunk_input(execution=route)
            self.h_in.numpy()
            return (time.perf_counter() - t0) / 16.0

        for route in valid:
            bench(route)
        for r in range(9):
            order = valid if not (r & 1) else list(reversed(valid))
            for route in order:
                samples[route].append(bench(route))
        chosen = wt._measured_choice(samples, valid, default=valid[0])
        wt._TUNED[key] = chosen
        return chosen

    def _set_chunk_inputs(self, token, pos, count):
        """One host crossing for a complete greedy chunk, not one crossing per token."""
        self._ensure_chunk_buffers(count)
        seed = np.zeros((self._chunk_capacity + 1,), np.int32)
        seed[0] = int(token)
        c, s = self._rope_np(int(pos), int(count))
        cbuf = np.zeros((self._chunk_capacity, self.HD), np.float32)
        sbuf = np.zeros_like(cbuf)
        cbuf[:count] = np.asarray(c, np.float32).reshape(count, self.HD)
        sbuf[:count] = np.asarray(s, np.float32).reshape(count, self.HD)
        self._chunk_tokens.buffer.set_data(seed)
        self._chunk_cos.buffer.set_data(cbuf.reshape(-1))
        self._chunk_sin.buffer.set_data(sbuf.reshape(-1))
        self.ctl.buffer.set_data(np.asarray(
            [int(pos), 1, self.NKV, self.HD, self.kv_cap], np.int32))

    def _capture_greedy_chunk(self, count, name="decode_chunk", row_execution=None):
        """Record and execute `count` chained exact-input greedy steps."""
        plat = wt._adam_kernel["platform"]
        head = self.head[0]
        plat.beginCapture(name)
        try:
            for i in range(int(count)):
                self._decode_chunk_input(
                    step=i, increment=(i > 0), execution=row_execution)
                logits = self._decode_fwd()
                wt.vocab_argmax(logits.data, self._chunk_tokens, i + 1)
            values = np.asarray(self._chunk_tokens.get(), np.int32).copy()
        finally:
            plat.endCapture()
        if name == "decode_chunk":
            self._greedy_chunk_capture_ready = True
        return values

    def _replay_greedy_chunk(self, count, name="decode_chunk"):
        wt._adam_kernel["platform"].replay(name)
        return np.asarray(self._chunk_tokens.get(), np.int32)[:int(count) + 1].copy()

    def _release_idle_greedy_capture(self, plat, active_pick_mode):
        """Unpin a calibrated greedy graph until a greedy caller needs it."""
        if active_pick_mode != "device" and self._greedy_chunk_capture_ready:
            plat.beginCapture("decode_chunk")
            plat.endCapture()
            self._greedy_chunk_capture_ready = False

    def _tune_greedy_chunks(self):
        """Select the fastest correct complete greedy API batch on this WebGPU device.

        Lower-layer `auto` choices remain untouched.  Counts 1/2/4 compete against the
        ordinary host-fed one-token path using complete captured decode steps, including
        uploads and the synchronising token readback.  The winner is therefore an API-level
        composition, not a dispatch microbenchmark.
        """
        self._greedy_chunk_size = 0
        self._greedy_chunk_capture_ready = False
        if not self._can_chunk_greedy():
            if getattr(self, "decode_plan", None) is not None:
                self.decode_plan["greedy_chunk"] = 0
            return 0
        import statistics as _stats
        plat = wt._adam_kernel["platform"]
        active_pick_mode = CausalLM._decode_pick_mode(self)
        saved_sampling = self.__dict__.get("_sampling", None)
        saved_seen = list(getattr(self, "_seen", ()))
        name = "decode_chunk_tune"
        # The race below records ten decode graphs and replays them for rounds -- 5.1 of a
        # 0.6B's 9.2 s load with every other measurement already remembered, because this
        # one was not. Its verdict is per device, model topology and kernel build, the same
        # as the decode composition's, so it is kept under that digest and reused.
        memo_key = None
        try:
            linears = self._stored_linears()
            qshapes = sorted({(m.Kt, m.Nt) for m in linears
                              if m.type_name == "Q4_K" and m.execution == "auto"})
            plan_key = CausalLM._decode_composition_key(self, linears, qshapes, "device")
            if plan_key is not None:
                memo_key = ("greedy_chunk_v1",) + tuple(plan_key[1:])
        except Exception:
            memo_key = None
        recorded = []
        try:
            self._set_sampling(temperature=0, do_sample=False, prompt_ids=[])
            self._ensure_chunk_buffers(4)
            memo = wt._TUNED.get(memo_key) if memo_key is not None else None
            if (isinstance(memo, dict) and memo.get("count") in (0, 1, 2, 4)
                    and (memo["count"] == 0
                         or memo.get("row") in self._embedding_row_candidates())):
                chosen = (int(memo["count"]), str(memo.get("row") or "host"))
                self._greedy_chunk_size = chosen[0]
                self._embedding_row_execution = (chosen[1] if chosen[0] else "auto")
                self.decode_plan["greedy_chunk"] = chosen[0]
                self.decode_plan["embedding_row"] = self._embedding_row_execution
                self.decode_plan["greedy_chunk_ms"] = memo.get("median_ms")
                self.decode_plan["greedy_chunk_reused"] = True
                if chosen[0]:
                    self._set_chunk_inputs(0, 0, chosen[0])
                    self._capture_greedy_chunk(chosen[0], row_execution=chosen[1])
                return self._greedy_chunk_size
            local_row = self._tune_embedding_row()
            self.decode_plan["embedding_row_auto"] = local_row
            # Both physical layouts have already matched one another bit-for-bit.  Match the
            # winning reconstruction against the independent host GGUF decoder as well.
            probe = min(37, int(self.embed.shape[0]) - 1)
            self._set_chunk_inputs(probe, 0, 1)
            self._decode_chunk_input(execution=local_row)
            got = np.asarray(self.h_in.numpy()[0], np.float32)
            ref = np.asarray(self.embed[probe], np.float32)
            if not np.array_equal(got, ref):
                self.decode_plan["greedy_chunk_error"] = "Q6_K row reconstruction differs"
                self.decode_plan["greedy_chunk"] = 0
                return 0

            # Each candidate is recorded ONCE, under its own name, and timed by replaying it:
            # the graphs are position-independent (inputs, rope rows and `ctl` are buffers
            # written before each replay). Recording a candidate again before every timed
            # sample -- 45 recordings in all -- is what made this race overrun its load
            # budget on every load of a 0.6B, and give up with no verdict to keep.
            def cname(which, row_route):
                return name if not which else "%s_%d_%s" % (name, which, row_route)

            def record(which, row_route="transposed", token=0, pos=0):
                if which:
                    self._set_chunk_inputs(token, pos, which)
                    return self._capture_greedy_chunk(
                        which, cname(which, row_route), row_execution=row_route)
                self._set_inputs(token, pos)
                self._chunk_tokens.buffer.set_data(np.zeros((5,), np.int32))
                plat.beginCapture(name)
                try:
                    logits = self._decode_fwd()
                    wt.vocab_argmax(logits.data, self._chunk_tokens, 1)
                    values = np.asarray(self._chunk_tokens.get(), np.int32).copy()
                finally:
                    plat.endCapture()
                return values

            # Token correctness is sequential: every candidate must reproduce the ordinary
            # host-fed route, not merely choose the same first token.
            ref_tokens = []
            values = record(0, token=0, pos=0)
            token = int(values[1]); ref_tokens.append(token)
            for pos in range(1, 4):
                self._set_inputs(token, pos)
                plat.replay(name)
                token = int(np.asarray(self._chunk_tokens.get()).reshape(-1)[1])
                ref_tokens.append(token)
            valid = [(0, "host")]
            for row_route in self._embedding_row_candidates():
                for count in (1, 2, 4):
                    values = record(count, row_route, 0, 0)
                    if values[1:count + 1].tolist() == ref_tokens[:count]:
                        valid.append((count, row_route))
            recorded = [cname(*c) for c in valid if c[0]]

            samples = {c: [] for c in valid}

            def bench(candidate):
                which, row_route = candidate
                graph = cname(which, row_route)
                token = 0; pos = 64
                steps = 4 if hasattr(self, "_warm_deadline") else 16
                t0 = time.perf_counter()
                if which == 0:
                    for _ in range(steps):
                        self._set_inputs(token, pos)
                        plat.replay(graph)
                        token = int(np.asarray(self._chunk_tokens.get()).reshape(-1)[1])
                        pos += 1
                else:
                    for _ in range(steps // which):
                        self._set_chunk_inputs(token, pos, which)
                        plat.replay(graph)
                        values = np.asarray(self._chunk_tokens.get(), np.int32)
                        token = int(values[which]); pos += which
                return (time.perf_counter() - t0) / steps

            for r in range(5 if hasattr(self, "_warm_deadline") else 15):
                order = valid if not (r & 1) else list(reversed(valid))
                for candidate in order:
                    if time.perf_counter() >= getattr(self, "_warm_deadline", float("inf")):
                        self.decode_plan["greedy_chunk_budget_skipped"] = True
                        self.decode_plan["greedy_chunk"] = 0
                        return 0
                    samples[candidate].append(bench(candidate))
            # This is the highest generation API, so latency is primary and memory is the
            # tie-breaker.  Walking candidates in allocation order can get stuck on a noisy
            # intermediate route and never reach the actual fastest route.  Rank by measured
            # latency, then still require paired evidence against the low-memory baseline;
            # there is no percentage threshold, and an inconclusive comparison stays on the
            # baseline.  Lower-layer `_measured_choice` deliberately keeps its memory-first
            # policy because those APIs may be called outside generation.
            baseline = (0, "host")
            ranked = sorted(valid, key=lambda c: _stats.median(samples[c]))
            chosen = baseline
            for candidate in ranked:
                if candidate != chosen and wt._paired_faster(samples, candidate, chosen):
                    chosen = candidate
            self._greedy_chunk_size = int(chosen[0])
            self._embedding_row_execution = (chosen[1] if chosen[0] else "auto")
            self.decode_plan["greedy_chunk"] = int(chosen[0])
            self.decode_plan["embedding_row"] = self._embedding_row_execution
            self.decode_plan["greedy_chunk_ms"] = round(
                _stats.median(samples[chosen]) * 1000, 4)
            self.decode_plan["greedy_chunk_candidates_ms"] = {
                "%s/%s" % (candidate[0], candidate[1]): round(
                    _stats.median(samples[candidate]) * 1000, 4)
                for candidate in valid}
            if memo_key is not None:
                wt._TUNED[memo_key] = {"count": int(chosen[0]), "row": str(chosen[1]),
                                       "median_ms": self.decode_plan["greedy_chunk_ms"]}
            # This graph is independent of prompt position: inputs, rope rows and `ctl`
            # are mutable buffers populated immediately before replay.  Keep one verified
            # recording for the model lifetime instead of charging every API call for graph
            # construction.  KV growth invalidates it below because that moves buffers.
            if chosen[0]:
                self._set_chunk_inputs(0, 0, chosen[0])
                self._capture_greedy_chunk(
                    chosen[0], row_execution=chosen[1])
            return self._greedy_chunk_size
        except Exception as exc:
            self._greedy_chunk_size = 0
            self.decode_plan["greedy_chunk"] = 0
            self.decode_plan["greedy_chunk_error"] = "%s: %s" % (type(exc).__name__, exc)
            return 0
        finally:
            # Overwrite every tuning graph so none of them pins model buffers.
            for graph in [name] + list(recorded):
                try:
                    plat.beginCapture(graph); plat.endCapture()
                except Exception:
                    pass
            self._seen = saved_seen
            if saved_sampling is None:
                self.__dict__.pop("_sampling", None)
            else:
                self._sampling = saved_sampling
            # The loader may calibrate the greedy route even when the model's
            # normal call samples. The selected chunk size remains available,
            # but its captured graph cannot be called by that sampling route.
            # Replacing only this named graph releases its pins without touching
            # the active decode graph. A later greedy call records it lazily.
            CausalLM._release_idle_greedy_capture(self, plat, active_pick_mode)

    def _warm_decode_step(self):
        """Run one decode step here, where nothing is recording.

        A kernel is registered the first time a matmul of its shape runs, and registering it
        RUNS it: the numerical self-check that guards against a shader which compiled to
        nothing dispatches the kernel it is checking. If the first such shape is reached
        inside `beginCapture`, those dispatches are recorded into the decode graph -- and a
        recorded graph is replayed in full for every token of the reply.

        Measured on a dense 27B, the same graph from the same source, recorded twice in one
        reply -- once at the start and once when the cache ran out of rows:

            first recording   6990 buffers   3477 dispatches
            re-recording      3524 buffers   1749 dispatches

        1749 is what this model needs; the repository's own record for it after the fused
        SwiGLU landed is 1759. The first recording carries 1728 more, which is 27 x 64 --
        one layer's worth of commands, times its layer count -- and replays them for every
        token until the second recording replaces it. That token is where the per-token cost
        falls three-fold and stays down.

        What exactly those 1728 are has not been pinned down. What is certain is that they
        are recorded by the FIRST `_decode_fwd` and not by a later one, so running one before
        anything is recording is what keeps them out of the graph. `_warm_shapes` cannot do
        it: it runs before the state a decode step needs exists, and it feeds the weights one
        at a time rather than running the step, so nothing between them -- the recurrent
        scan, the fused attention at one position, the elementwise chain at one row -- is
        reached at the shape a decode step uses.

        The recurrent layers advance their state, so it is put back: that state belongs to a
        conversation, and this is not one.
        """
        try:
            tok = int(self.eos_ids[0]) if getattr(self, "eos_ids", None) else 0
            # Twice, and the two are compared. Once is what the fix needs; the second run is
            # what says the fix is still working -- if some later change gives a first call
            # commands a second one does not have, the load says so by name instead of a reply
            # going three times slower for reasons no one can see.
            wt._count_dispatch_names(True)
            a = self._dispatch_names()
            _t0 = time.perf_counter()
            self._set_inputs(tok, 0)
            _t1 = time.perf_counter()
            first_logits = self._decode_fwd()
            _t2 = time.perf_counter()
            first_logits.numpy()
            _t3 = time.perf_counter()
            del first_logits
            b = self._dispatch_names()
            self._set_inputs(tok, 0)
            _t4 = time.perf_counter()
            second_logits = self._decode_fwd()
            _t5 = time.perf_counter()
            second_logits.numpy()
            _t6 = time.perf_counter()
            del second_logits
            self._warm_step_profile = {
                "first_input_s": _t1 - _t0,
                "first_dispatch_s": _t2 - _t1,
                "first_sync_s": _t3 - _t2,
                "second_input_s": _t4 - _t3,
                "second_dispatch_s": _t5 - _t4,
                "second_sync_s": _t6 - _t5,
            }
            c = self._dispatch_names()
            first = {k: b.get(k, 0) - a.get(k, 0) for k in b}
            again = {k: c.get(k, 0) - b.get(k, 0) for k in c}
            extra = {k: first.get(k, 0) - again.get(k, 0) for k in first
                     if first.get(k, 0) != again.get(k, 0)}
            import js
            if extra:
                js.console.log(
                    "warm: the first decode step dispatched %d commands a second does not: %s"
                    % (sum(extra.values()),
                       ", ".join("%s x%d" % (k, v) for k, v in
                                 sorted(extra.items(), key=lambda kv: -kv[1])[:12])))
            # And what one settled step is made of, by kernel. A step's cost is its commands,
            # and "591 against 567" says only that four and twenty went somewhere; this says
            # which kernels they are. Printed once, at load, from the second run -- the one
            # with nothing left to register.
            js.console.log("decode step: %d dispatches | %s"
                           % (sum(again.values()),
                              "  ".join("%s x%d" % (k, v) for k, v in
                                        sorted(again.items(), key=lambda kv: -kv[1])
                                        if v)))
        except Exception:
            pass                      # best effort: a model that cannot do this still loads
        finally:
            wt._count_dispatch_names(False)
            self._reset_linear_state()

    @staticmethod
    def _dispatch_names():
        """Dispatches issued so far, per kernel name. Empty off the WebGPU path."""
        try:
            from wgpy_backends.webgpu.platform import WebGPUPlatform
            return dict(WebGPUPlatform.by_name)
        except Exception:
            return {}

    # ---- prefill (fresh, fills KV 0..P-1) ----
    def _capturable(self):
        """Can a decode step be replayed as a fixed command sequence?

        Two things make it not. A recurrent layer whose step runs on the host reads and
        writes state outside the graph. And a sparse-MoE layer picks its experts on the
        host -- the router's output comes back, numpy chooses the top-k, and that choice is
        what decides which expert kernels get dispatched. A capture freezes those dispatches,
        so every later token would be routed to whatever the FIRST one happened to select:
        the reply starts correctly and then repeats one token forever, which is exactly what
        it looked like. Routing on the device would lift this."""
        rec = [j for j in range(len(self.layers)) if self._is_linear_layer(j)]
        if rec and not all(self.layers[j]["linear"]._gpu_step_ok() for j in rec):
            return False
        # A MoE layer is capturable exactly when its experts are stacked: the shader then
        # reads the chosen expert from a buffer the host rewrites each step, so the command
        # list is the same one every token. Without stacking, the routing decision picks
        # which kernels get dispatched, and a capture would freeze the first token's choice.
        if any(lay.get("moe") and not lay["moe"].get("stacked") for lay in self.layers):
            return False
        return bool(self._gpu)

    def _mlp(self, lay, x):
        """Dense SwiGLU, or generic sparse-MoE (router top-k + optional shared expert) when the
        layer carries `moe` — same layer dict shape as `lm_engine.build_lm`, so the proven
        generic MoE path is reused instead of a second implementation."""
        if lay.get("moe"):
            from . import lm_engine
            return lm_engine.moe_mlp(self, lay, x)
        activated = wt.parallel_swiglu((lay["gate"], lay["up"]), x,
                                       execution=self._gate_up_execution)
        return lay["down"](activated)

    def _qkv(self, lay, x, T, apply_norm=True):
        """q,k,v projections -> (heads, T, HD). Applies per-head QK-norm (RMSNorm over head_dim,
        before rope) when `lay` carries `qn`/`kn` weights.

        Some decoders fuse an output GATE into the query projection: it then emits
        2 * n_heads * head_dim, laid out per head as [q | gate]. That is detected from the
        projection's own width — not from a model name — and the gate is stashed on the layer
        for `_attn_out` to apply."""
        qraw, kraw, vraw = wt.parallel_linear((lay["q"], lay["k"], lay["v"]), x,
                                              execution=self._qkv_execution)
        wide = int(np.prod(qraw.shape)) // max(T, 1) >= 2 * self.NH * self.HD
        if wide:                                        # fused [q | gate] per head
            qg = qraw.reshape(T, self.NH, 2 * self.HD)
            q = wt._slice_last(qg, 0, self.HD)
            lay["_gate"] = wt._slice_last(qg, self.HD, 2 * self.HD).reshape(T, self.NH * self.HD)
        else:
            q = qraw.reshape(T, self.NH, self.HD)
            lay["_gate"] = None
        k = kraw.reshape(T, self.NKV, self.HD)
        # QK-norm comes in two widths and the WEIGHT says which. One head_dim of weight is
        # the per-head form (Qwen3): each head is normalised over its own dims. A weight as
        # wide as the whole projection is the other form (OLMoE): the normalisation runs over
        # every head at once, which is a different denominator, not just a differently shaped
        # scale. Reading it off the weight is what keeps this generic -- applying the per-head
        # form to a full-width weight raises nothing, it silently divides by the wrong number,
        # and the model answers with fluent nonsense.
        if apply_norm:
            q = self._qk_norm(q, lay.get("qn"), T, self.NH)
            k = self._qk_norm(k, lay.get("kn"), T, self.NKV)
        v = vraw.reshape(T, self.NKV, self.HD).permute(1, 0, 2)
        return q.permute(1, 0, 2), k.permute(1, 0, 2), v

    def _qk_norm_rope(self, lay, q, k):
        """Decode-time Q/K norm+rope layer with an independently usable ``auto`` route.

        WebGPU can combine both tensors into one dispatch.  WebGL and unsupported norm
        shapes compose their existing primitives at this nearest equivalent layer.  A
        containing full-decoder tuner may request either route explicitly.
        """
        route = getattr(self, "_qk_norm_rope_execution", "auto")

        def composed():
            return (self._rope1(self._qk_norm(q, lay.get("qn"), 1, self.NH)),
                    self._rope1(self._qk_norm(k, lay.get("kn"), 1, self.NKV)))

        def fused():
            return wt.qk_norm_rope_decode(
                q, k, lay.get("qn"), lay.get("kn"), self.cos_b, self.sin_b,
                self.HD, getattr(self, "rope_dim", self.HD), self.eps)

        key = ("qk_norm_rope", self.NH, self.NKV, self.HD,
               getattr(self, "rope_dim", self.HD))
        correct_key = key + ("correct",)
        if route == "composed" or (route == "auto" and wt._TUNED.get(key) == "composed"):
            return composed()
        probe = fused()
        if probe is None:
            return composed()
        if correct_key not in wt._TUNED:
            rq, rk = composed(); fq, fk = probe
            qa, ka = np.asarray(rq.numpy()), np.asarray(rk.numpy())
            qb, kb = np.asarray(fq.numpy()), np.asarray(fk.numpy())
            scale = max(1e-6, float(np.abs(qa).max()), float(np.abs(ka).max()))
            wt._TUNED[correct_key] = bool(
                np.all(np.isfinite(qb)) and np.all(np.isfinite(kb))
                and max(float(np.abs(qa - qb).max()),
                        float(np.abs(ka - kb).max())) / scale < 2e-5)
        valid_fused = bool(wt._TUNED[correct_key])
        if route == "composed":
            return composed()
        if route == "fused":
            if not valid_fused:
                raise RuntimeError("fused Q/K norm+rope failed numerical validation")
            return probe
        if route != "auto":
            raise ValueError("Q/K norm+rope execution must be auto, composed, or fused")
        if key not in wt._TUNED:
            candidates = ["composed"] + (["fused"] if valid_fused else [])
            samples = {name: [] for name in candidates}

            def bench(name):
                out = None; t0 = time.perf_counter()
                for _ in range(8):
                    out = fused() if name == "fused" else composed()
                out[-1].numpy()
                return (time.perf_counter() - t0) / 8.0

            for name in candidates:
                bench(name)
            for r in range(9):
                order = candidates if not (r & 1) else list(reversed(candidates))
                for name in order:
                    samples[name].append(bench(name))
            wt._TUNED[key] = wt._measured_choice(samples, candidates,
                                                  default="composed")
        return probe if wt._TUNED[key] == "fused" else composed()

    def _attn_out(self, lay, o, T, rows=False):
        """Attention output -> hidden. Applies the sigmoid output gate when the query projection
        carried one (see `_qkv`), then the output projection. `rows`: `o` is already
        (T, heads * head_dim), as `causal_attention_cache` writes it."""
        # At one position the permute rearranges nothing: `o` is (nh, 1, hd) and laid out
        # exactly as (1, nh*hd) already is, because an axis of length 1 cannot interleave
        # anything. Its STRIDES say otherwise, so `reshape` cannot see that and falls back to
        # copying the whole tensor -- an identity copy, `out0 = in0`, once per layer per
        # token. On a 0.6B that was 28 of a decode step's 591 dispatches. Removed because it
        # computes nothing, NOT on a theory about command counts -- that theory was tested a
        # few hundred lines below, fusing a MoE layer's gate and up into one dispatch, and
        # the step did not move.
        #
        # The elements and their order are identical either way; this only declines to ask
        # for a layout that is already the one it has.
        y = (o if rows else o.reshape(T, self.NH * self.HD) if T == 1
             else o.permute(1, 0, 2).reshape(T, self.NH * self.HD))
        g = lay.get("_gate")
        if g is not None:
            y = y * g.sigmoid()
        return lay["o"](y)

    def _embed_ids(self, ids, embeds=None):
        """Input embeddings for a prompt. `embeds` (T,H) overrides the token lookup — the hook
        multimodal models use to splice image/audio embeddings into the sequence."""
        if embeds is not None:
            return wt.Tensor(np.asarray(embeds, np.float32))
        return wt.Tensor(np.asarray(self.embed[np.asarray(ids, np.int64)], np.float32))

    def _reset_linear_state(self):
        """Clear every linear-attention layer's recurrent state (start of a generation)."""
        for st in getattr(self, "lin_state", []) or []:
            if st is not None:
                st.reset()

    def _is_linear_layer(self, i):
        """True when layer `i` is a linear-attention (recurrent state) layer rather than softmax
        attention. Driven by the config's `layer_types` / `full_attention_interval`."""
        lt = getattr(self, "layer_types", None)
        return bool(lt) and i < len(lt) and lt[i] != "full_attention"

    def _linear_mixer(self, i, lay, x, T):
        """Run a linear-attention layer's recurrence -> (T, H) Tensor. The per-layer state lives
        in `self.lin_state[i]`, so prefill and incremental decode share one implementation."""
        st = self.lin_state[i]
        # Hand the Tensor over as-is: the layer's decode path stays on the device, and
        # pulling it to the host here would undo that.
        y = lay["linear"].forward(x, st)
        return y if isinstance(y, wt.Tensor) else wt.Tensor(np.asarray(y, np.float32))

    def _kv_prefix(self, ids, embeds=None, rope_pos=None):
        """How many leading tokens of `ids` the KV cache already holds, and can keep.

        A chat re-sends its whole history every turn, so turn N's prompt is turn N-1's prompt
        plus what has been said since. Recomputing the shared part costs what the first turn
        cost -- and prefill is the expensive half: on a 30B MoE a 500-token prompt is ~90s of
        prefill against 30 tokens/s of decode, which is why a reply that triggers a tool and
        comes back for a second round appears to stall for a minute between the two. Those
        keys and values are already in the cache and unchanged; this finds how many.

        Reuse needs the cached rows to still mean what they meant when they were written:

        - Spliced embeddings (`embeds`) and explicit rotary positions (`rope_pos`) both break
          the token-to-position correspondence this compares on, so they start over.
        - A recurrent (linear-attention) layer carries a state rather than a per-position
          cache, and that state has already run PAST the shared prefix -- there is no row to
          rewind to. Models with any such layer start over.

        One token is always left to run: prefill produces the next-token logits, so a
        completely cached prompt still has to push its last token through."""
        if embeds is not None or rope_pos is not None:
            return 0
        if any(self._is_linear_layer(i) for i in range(len(self.layers))):
            return 0
        have = self.__dict__.get("_kv_ids")
        if not have:
            return 0
        n, lim = 0, min(len(have), len(ids) - 1)
        while n < lim and have[n] == ids[n]:
            n += 1
        return n

    def _kv_growing(self, ids, embeds=None, rope_pos=None):
        """The growing cache to continue from, and how many of `ids` it already holds.

        The counterpart of `_kv_prefix` for the path that cannot capture -- WebGL, a
        host-side mixer, an unstacked MoE. That path built a whole new cache for every
        generation and prefilled the prompt from zero, so a chat re-paid for its entire
        history on every turn exactly as the capture path used to. The cache lives on the
        model instead, and a new turn keeps the rows its prompt still agrees with.

        Returns `(cache, keep)`. A cache that cannot be continued is replaced, not repaired.
        """
        keep = self._kv_prefix(ids, embeds, rope_pos)
        cache = self.__dict__.get("_gcache")
        if cache is not None and keep:
            have = cache.length()
            if have is not None and have >= keep:
                cache.truncate(keep)              # drop the last reply, keep the shared head
                return cache, keep
        cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
        self._gcache = cache
        return cache, 0

    @staticmethod
    def _shape_owner_of(layers, type_name, nt, kt):
        """One layer that has this (format, N, K), or None. The warm pass keeps shapes, not
        layers, so getting back to something callable takes a second look."""
        for lay in layers or []:
            for v in lay.values():
                if (getattr(v, "packed", None) is not None and hasattr(v, "Kt")
                        and v.type_name == type_name
                        and int(v.Nt) == int(nt) and int(v.Kt) == int(kt)):
                    return v
        return None

    def _warm_shapes(self):
        """Run each distinct quantised matmul shape once, so the tuner settles during the
        load instead of during the first reply. Best-effort: a model whose weights are not
        of this kind, or a backend that cannot run it, simply skips."""
        try:
            # Drain first. Everything below is measurement -- the tuner runs each candidate
            # thread shape and times it -- and a timing is only as good as the queue it runs
            # against. This runs immediately after the whole model has gone to the GPU, so
            # the first sync of every one of those ~480 rounds waits for the tail of that
            # upload rather than for the kernel it is supposed to be timing.
            #
            # Measured on a 27B: this phase reported 48.8s on a real load, against 0.07s
            # re-run later on the same settled machine. Seventy times, for the same work.
            # One readback here costs one wait instead of hundreds.
            wt.Tensor(np.zeros((1, 1), np.float32)).numpy()
            _t0 = time.perf_counter()
            # Settle the independent one-row MoE reduction before any decoder graph is
            # captured. Its auto tuner checks numerical equivalence and paired timing;
            # doing that for the first time *inside* capture would replay all calibration
            # dispatches on every generated token. This is keyed by operator shape, not
            # model identity, and runs once per distinct (top-k, hidden width).
            reductions = set()
            for lay in getattr(self, "layers", ()):
                moe = lay.get("moe")
                if not moe or not moe.get("stacked"):
                    continue
                rk = (int(moe["top_k"]), int(self.H))
                if rk in reductions:
                    continue
                reductions.add(rk)
                rng = np.random.default_rng(rk[0] * 65537 + rk[1])
                trial = wt.Tensor(rng.normal(0, 1, (rk[0], rk[1])).astype(np.float32))
                weights = wt.Tensor(np.linspace(0.2, 1.0, rk[0], dtype=np.float32))
                wt.moe_weighted_sum(trial, weights, rk[0]).numpy()
                del trial, weights
            if reductions:
                wt.gpu_reap()
            seen = set()
            held = []
            owners = {}
            walked = set()
            def walk(value):
                oid = id(value)
                if oid in walked:
                    return
                walked.add(oid)
                if isinstance(value, (wt.GGMLLinear, wt.GGMLMoELinear)):
                    moe = isinstance(value, wt.GGMLMoELinear)
                    key = (value.type_name, int(value.Nt), int(value.Kt), moe)
                    owners.setdefault(key, value)
                    return
                if isinstance(value, dict):
                    for item in value.values():
                        walk(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        walk(item)
                elif isinstance(value, wt.Module) or (
                        # A layer this package defines without deriving Module -- the
                        # linear-attention block is one -- still holds Linears. Missing it
                        # left 17 of a hybrid 27B's projection shapes out of both this warm
                        # pass and the route ladder, so its first prompt raced them (4.5 s).
                        not isinstance(value, wt.Tensor) and hasattr(value, "__dict__")
                        and type(value).__module__.startswith("webtorch.")):
                    for item in vars(value).values():
                        walk(item)
            walk(getattr(self, "layers", []))
            # What the layers hold is what a prompt's length reaches; the output head is
            # read for the last position only, so it is built here but not laddered below.
            layer_keys = set(owners)
            walk(getattr(self, "head", []))
            eidx = wt._empty_i32((1,)) if any(k[3] for k in owners) else None
            if eidx is not None:
                eidx.buffer.set_data(np.zeros((1,), np.int32))
            for key, v in owners.items():
                seen.add(key)
                x = wt.Tensor(np.zeros((1, int(v.Kt)), np.float32))
                # Stacked MoE experts live under lay["moe"]["stacked"], not directly
                # under a layer. Missing them made the four semantic forwards pay for
                # their stored-kernel registration on every large-model load.
                held.append(v(x, eidx) if key[3] else v(x))
                # No per-shape readback: queue order and the final read settle them all.
            if held:
                held[-1].numpy()
            held.clear()
            # And once more with a BATCH, because the two are different kernels and only one
            # of them was being built here. A shader compiles on its first dispatch, not when
            # it is registered, so a variant nothing dispatched during the load is one the
            # reader waits for -- and the row count above is 1, which reaches the decode GEMV
            # and nothing else. The batched kernel, and the unpacking path that a wide enough
            # matmul prefers over it, were both built by the first PROMPT instead: this phase
            # reported nothing while the first reply spent seconds before its first token.
            #
            # Again with three rows, because the prefill path branches on the row count and
            # one row reaches only the decode GEMV. Three is the first count that leaves it
            # for the batched kernel, which the reader's first prompt was compiling instead.
            #
            # This pass forces the stored path: it only builds kernels. Which route each
            # row count should take is measured right after it, over a ladder of row
            # counts (see the calibration below), not on the reader's first prompt.
            #
            # One shape per distinct (format, N, K), not one per layer, so this is a pass over
            # a few dozen tensors rather than over the model.
            for m in (3,):
                try:
                    for tname, nt, kt, moe in list(seen):
                        if moe:
                            continue  # the decode expert path has its own shape above
                        lay = owners.get((tname, nt, kt, False))
                        if lay is None:
                            continue
                        old_execution = getattr(lay, "execution", None)
                        if old_execution is not None:
                            lay.execution = "stored"
                        try:
                            held.append(lay(wt.Tensor(np.zeros((m, int(kt)), np.float32))))
                        finally:
                            if old_execution is not None:
                                lay.execution = old_execution
                    if held:
                        held[-1].numpy()
                except Exception:
                    pass
                held.clear()
            # Route races belong to the load, not to the first prompt. The first reply of a
            # 0.6B spent 343 ms of its prefill racing stored against unpacked for the
            # 256-row bucket of seven weight shapes. Each distinct (format, N, K) a layer
            # holds is now raced over a 16..512 row ladder here, bisected where the winner
            # changes; a prefill of any length then takes the nearest measured bucket and
            # races nothing. Unpacking a weight to race it costs here what it cost on that
            # first prompt -- the same transient, at the same moment of a session's life.
            # Not the output head: a prefill reads its last row only, and laddering a
            # 151936-wide head to 512 rows cost a 0.6B's load six seconds for a shape no
            # prompt produces.
            ladder = sorted(k for k in layer_keys if not k[3])
            _load_stage("tuning", done=0, total=len(ladder))
            rng = np.random.default_rng(0)
            for i, (tname, nt, kt, moe) in enumerate(ladder):
                lay = owners.get((tname, nt, kt, False))
                if lay is None or getattr(lay, "execution", None) != "auto":
                    continue

                def probe(m, lay=lay, kt=kt):
                    x = wt.Tensor(rng.standard_normal((m, int(kt))).astype(np.float32))
                    wt._sync_small(lay(x))
                wt.calibrate_rows(probe, 512, lo=16, step=4, defer=True)
                _load_stage("tuning", done=i + 1, total=len(ladder))
            # A prompt's routed experts: per slot GEMV against grouped small GEMMs, by how
            # many slots there are. One ladder per distinct stacked projection; the probe
            # routes random slots over all of the layer's experts.
            routed = {}
            for lay in self.layers:
                st = (lay.get("moe") or {}).get("stacked") if isinstance(lay, dict) else None
                if st:
                    for name, slot_rows in (("gate_up", False), ("down", True)):
                        lin = st.get(name)
                        if callable(getattr(lin, "forward_routed", None)):
                            routed.setdefault((lin.type_name, lin.Kt, lin.Nt), (lin, slot_rows))
            for (tname, kt, nt), (lin, slot_rows) in sorted(routed.items()):
                kk = max(1, int(self.top_k or 2))

                def rprobe(m, lin=lin, slot_rows=slot_rows, kk=kk):
                    S = max(kk, (int(m) // kk) * kk)
                    rows = S if slot_rows else S // kk
                    x = wt.Tensor(rng.standard_normal((rows, int(lin.Kt))).astype(np.float32))
                    e = wt._empty_i32((S,))
                    e.buffer.set_data(rng.integers(0, int(lin.n_experts), S).astype(np.int32))
                    wt._sync_small(lin.forward_routed(x, e, kk, slot_rows))
                wt.calibrate_rows(rprobe, 4096, lo=16, step=4, defer=True)
            # Prefill attention against the cache: how far to split the keys over workgroups
            # depends on how much context a few new rows read, so the ladder walks the
            # context with a short new segment. Reads one layer's cache; writes nothing.
            # Its own small cache, not the model's: the model's is allocated by the first
            # prefill, after this -- probing it here skipped the ladder, and the first long
            # prompt then raced the splits inside its answer.
            kv0 = next((j for j in range(len(self.layers))
                        if not self._is_linear_layer(j)), None)
            if self._gpu and wt.kv_f16() and kv0 is not None and self.HD % 2 == 0:
                top = 4096
                words = int(self.NKV) * top * (int(self.HD) // 2)
                K0 = wt.Tensor(np.zeros((words,), np.float32))
                V0 = wt.Tensor(np.zeros((words,), np.float32))
                sc0 = 1.0 / math.sqrt(self.HD)

                def aprobe(m, K0=K0, V0=V0, sc0=sc0, top=top):
                    t = min(32, int(m))
                    q = wt.Tensor(rng.standard_normal((self.NH, t, self.HD)).astype(np.float32))
                    o = wt.causal_attention_cache(q, K0, V0, int(m) - t, self.NH, self.NKV,
                                                  self.HD, top, sc0)
                    if o is not None:
                        wt._sync_small(o)
                wt.calibrate_rows(aprobe, top, lo=64, step=4, defer=True)
            _warm_s = time.perf_counter() - _t0
            wt.flash_tune(self.NH, self.NKV, self.HD)
            # What this phase actually spent, broken down, so the next slow load is a table
            # rather than an argument. Timing only; it changes nothing.
            try:
                c = wt.tune_cost()
                # ``shapes`` and ``variants`` are different units: a shape evaluates
                # several variants, while one compiled variant can serve several shapes.
                # Reporting them as done/total produced impossible progress such as 9/6.
                # The dispatched shape set is the actual finite work list for this pass.
                _load_stage("warming", done=len(seen), total=len(seen))
                import js
                # The kernel build stamp goes on this line because this line is what gets
                # quoted back. Three readings today could not be attributed to a version --
                # "is this the new code?" is not answerable from a tok/s -- and the stamp
                # that answers it already existed, unprinted.
                # `shapes`/`variants` count what the TUNER did, and a reused profile makes
                # the tuner do nothing -- which reads as "warming did nothing" and hides the
                # part that still has to happen. Dispatching each shape is what compiles its
                # shader, profile or no profile, so what that pass covered is reported on its
                # own: how many distinct shapes it ran, and how long it took. A load that
                # reports 0 there is a load whose first reply compiles the kernels.
                js.console.log(
                    "warm: %d dispatched in %.1fs | tuner %d shapes, %d variants,"
                    " %.1fs (build %.1fs, check %.1fs) | kernels %s"
                    % (len(seen), _warm_s, c.get("shapes", 0), c.get("variants", 0),
                       c.get("tune_s", 0.0), c.get("build_s", 0.0), c.get("check_s", 0.0),
                       wt._kernel_build()[:8]))
            except Exception:
                pass
        except Exception:
            pass

    def _kv_reserve(self, need):
        """Make sure the KV cache holds at least `need` rows, growing it if it does not.

        The cache is sized to the conversation rather than to the context the model COULD
        run to. Loading a model no longer costs the whole context up front -- which on a 30B
        was 2GB before a word was typed, and the same 2GB on a 0.6B whose weights are 0.4GB
        -- and a chat that stays short never pays for one that does not.

        Growth copies the rows already written, so the prefix a conversation has built up
        survives it: reallocating is exactly when a conversation is getting long, and
        throwing the cache away there would re-prefill the whole history.
        """
        need = int(need)
        cap = int(getattr(self, "kv_cap", 0))
        if need <= cap or not getattr(self, "_gpu", False):
            return
        # Double until it fits, so a long conversation reallocates a handful of times rather
        # than once a turn, and never past what the model may actually run to.
        new = max(cap * 2, self._KV_FLOOR)
        while new < need:
            new *= 2
        new = min(new, int(self.lmax))
        if new <= cap:
            return
        NKV, HD = self.NKV, self.HD
        # Carry over the WHOLE old cache, not the part `_kv_ids` names. That list is written
        # when a reply ENDS, so while one is being written it still names the previous turn --
        # and this runs mid-reply, every time `pos` reaches the cap. Trusting it here threw
        # away every row the reply had just written; on the first reply of a session there is
        # no list at all, so it kept nothing and the model went on attending over zeros. That
        # is a reply that starts perfectly and turns to noise a few hundred tokens in.
        # Rows past what is live are never read -- attention scans `pos + 1` of them -- so
        # copying all `cap` of them is simply correct, and costs one pass over a buffer that
        # is at most half the size of the one being allocated.
        KVW = int(self.Kc[0].data.shape[-1]) if self.Kc else wt.kv_cache_hd(HD)
        for box in (self.Kc, self.Vc):
            for i, old in enumerate(box):
                grown = wt.Tensor(wt._empty((NKV, new, KVW)))
                grown.data[:, :cap, :] = old.data[:, :cap, :]
                box[i] = grown
        self.mask_b = wt.Tensor(np.zeros((1, 1, new), np.float32))
        self.ctl.buffer.set_data(np.array([0, 1, NKV, HD, new], np.int32))
        self.kv_cap = new
        # Every kernel was captured against the old buffers and the old stride.
        self.capture_ready = False
        self._greedy_chunk_capture_ready = False
        self._decode_graph_signature = None
        self._decode_graph_logits = None
        self._decode_graph_token = None

    def _kv_commit(self, held, embeds=None, rope_pos=None):
        """Record what the cache rows actually hold. MUST run on every exit path.

        `held` is maintained so that it always names exactly the rows written so far, so
        committing it is right whether the reply ended, was stopped, raised, or -- for the
        streaming pair, which are generators -- was simply abandoned by its consumer when
        the page stopped pulling. Committing only on the happy path is what broke: the rows
        for the abandoned run stay on the device while `_kv_ids` still names the PREVIOUS
        turn, and `_kv_prefix` then reuses rows that hold a different conversation. That
        reads as a model that has lost its mind -- repeating, drifting into text from an
        earlier turn -- with a forward pass that is provably still exact.

        Spliced embeddings and explicit rotary positions have no token-to-row
        correspondence to record, so they drop instead; see `_kv_prefix`.
        """
        if embeds is None and rope_pos is None:
            self._kv_ids = held
        else:
            self._kv_drop()

    def _kv_drop(self):
        """Forget what the cache holds. Anything that makes the cached rows unusable --
        releasing the model, resizing the cache, a run that wrote rows it did not track --
        calls this, and the next generation prefills from zero."""
        self.__dict__.pop("_kv_ids", None)
        self.__dict__.pop("_gcache", None)      # the growing cache is named by those ids too

    def _moe_prefill_api_key(self, rows):
        """Whole-prefill route identity, separate from a single MoE layer's choice.

        The host keeps this build's profile under its physical-device key. The
        source identity and actual layer geometry keep different model graphs
        apart without recognising any model name or task category.
        """
        if rows <= 1:
            return None
        source = getattr(self, "_gguf", None) or getattr(self, "base", None)
        if not source:
            return None
        digest = getattr(self, "_moe_prefill_api_digest", None)
        if digest is None:
            layout = []
            for layer in getattr(self, "layers", ()):
                moe = layer.get("moe") if isinstance(layer, dict) else None
                stacked = moe.get("stacked") if moe else None
                if stacked:
                    layout.append((int(moe.get("n_experts") or stacked["gate"].n_experts),
                                   int(moe["top_k"]),
                                   str(getattr(stacked["gate_up"], "type_name", "?")),
                                   str(getattr(stacked["down"], "type_name", "?"))))
                else:
                    layout.append(None)
            if not any(entry is not None for entry in layout):
                return None
            import hashlib
            topology = {"source": str(source), "geometry": [
                int(getattr(self, name, 0) or 0) for name in
                ("L", "H", "NH", "NKV", "HD", "VOCAB")],
                "layers": layout}
            digest = hashlib.sha256(json.dumps(topology, sort_keys=True,
                                                separators=(",", ":")).encode()
                                    ).hexdigest()[:24]
            self._moe_prefill_api_digest = digest
        backend = ("webgpu" if wt._adam_backend_ready() else
                   "webgl" if wt._webgl_ready() else "cpu")
        bucket = 1 << (int(rows).bit_length() - 1)
        base = (backend, digest, bucket)
        phase = ("warm" if base in getattr(self, "_moe_prefill_api_seen", ())
                 else "cold")
        return ("moe_prefill_api_v3", *base, phase)

    def _moe_prefill_api_choice(self, rows):
        key = CausalLM._moe_prefill_api_key(self, rows)
        route = wt._TUNED.get(key) if key is not None else None
        if route in ("host", "device"):
            return route
        if key is not None and key[1] in ("webgpu", "webgl"):
            # No interactive multi-layer tournament: that made load/first
            # response take tens to hundreds of seconds. Use the no-readback
            # exact device route provisionally, and leave this bucket pending
            # for an explicit paired full-API profile of the new JS host route.
            self._moe_prefill_api_pending = getattr(
                self, "_moe_prefill_api_pending", set())
            self._moe_prefill_api_pending.add(key)
            return "device"
        return "auto"

    def _moe_prefill_api_completed(self, rows):
        key = CausalLM._moe_prefill_api_key(self, rows)
        if key is not None:
            self._moe_prefill_api_seen = getattr(self, "_moe_prefill_api_seen", set())
            self._moe_prefill_api_seen.add(key[1:4])

    def profile_moe_prefill_composition(self, ids=(1, 2, 3, 4), rounds=5):
        """Explicit offline whole-model MoE route A/B; not part of interactive load.

        This exercises the same prefill API as generation, checks the final
        token and hidden state, then compares paired *warm* calls. A cold
        winner cannot be inferred from repeated warm runs; it remains pending
        until separately measured. Synthetic calls invalidate the live KV
        prefix, so the next generation rebuilds that prefix from its text.
        """
        ids = tuple(int(token) for token in ids)
        if len(ids) <= 1:
            raise ValueError("whole-prefill profiling needs at least two rows")
        if int(rounds) <= 0:
            raise ValueError("rounds must be positive")
        key = CausalLM._moe_prefill_api_key(self, len(ids))
        if key is None or key[1] not in ("webgpu", "webgl"):
            raise ValueError("no GPU stacked-MoE prefill route to profile")
        import statistics as _stats
        old_route = self.__dict__.get("_moe_prefill_execution")
        had_route = "_moe_prefill_execution" in self.__dict__

        def run(route):
            self._moe_prefill_execution = route
            self._reset_linear_state()
            t0 = time.perf_counter()
            try:
                if self._capturable():
                    picked = self._prefill(ids)
                    hidden = self._last_prefill_hidden
                else:
                    cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
                    picked = self._kv_forward(ids, 0, cache)
                    hidden = self._last_hidden
                vector = np.asarray(hidden.numpy(), np.float32).copy()
                elapsed = time.perf_counter() - t0
                return elapsed, int(picked), vector
            finally:
                CausalLM._kv_drop(self)

        try:
            _, reference_token, reference = run("host")
            _, candidate_token, candidate = run("device")
            if not np.all(np.isfinite(reference)):
                raise RuntimeError("host whole-prefill result is not finite")
            if candidate.shape != reference.shape:
                raise RuntimeError("device whole-prefill route changed hidden shape")
            scale = max(1e-6, float(np.abs(reference).max()))
            error = float(np.abs(candidate - reference).max()) / scale
            if (candidate_token != reference_token
                    or not np.all(np.isfinite(candidate)) or error >= 1e-4):
                raise RuntimeError("device whole-prefill route failed the host numerical "
                                   "gate (relative error %.3g)" % error)
            # The two correctness passes have warmed both routes. They are not
            # timing pairs, and must not be misreported as cold-path evidence.
            key = CausalLM._moe_prefill_api_key(self, len(ids))
            if key is None or key[-1] != "warm":
                raise RuntimeError("whole-prefill route did not enter its warm phase")
            samples = {"host": [], "device": []}
            for index in range(int(rounds)):
                order = (("host", "device") if index % 2 == 0
                         else ("device", "host"))
                for route in order:
                    elapsed, picked, vector = run(route)
                    if picked != reference_token or not np.all(np.isfinite(vector)):
                        raise RuntimeError("whole-prefill route changed its result while "
                                           "being measured")
                    samples[route].append(elapsed)
            choice = wt._measured_choice(samples, ("host", "device"), default="host")
            wt._TUNED[key] = choice
            return {"key": key, "choice": choice,
                    "median_seconds": {route: _stats.median(times)
                                       for route, times in samples.items()},
                    "samples_seconds": samples, "relative_error": error,
                    "evidence": wt._paired_evidence(samples, "device", "host"),
                    "phase": "warm", "cold_profile_pending": True}
        finally:
            if had_route:
                self._moe_prefill_execution = old_route
            else:
                self.__dict__.pop("_moe_prefill_execution", None)
            CausalLM._kv_drop(self)
            self._reset_linear_state()
            import gc
            gc.collect()
            wt._gpu_release_idle_pool()

    def _prefill(self, ids, embeds=None, start=0):
        """Run `ids` through the model, writing their keys and values into the cache at
        `start`, and return the argmax of the last position's logits.

        `start` > 0 means the cache already holds the `start` tokens before these -- see
        `_kv_prefix`. The new tokens then carry rotary positions `start..start+T-1` and
        attend back over everything from 0, so the result is identical to prefilling the
        whole sequence; only the work already done is skipped."""
        # A cold GPU working set and a settled one can prefer opposite MoE routes.
        # Record which shape classes completed an entire prefill. An incomplete run must
        # not turn its cold profile into a warm decision on the next attempt.
        self._moe_prefill_seen = getattr(self, "_moe_prefill_seen", set())
        self._moe_prefill_pending = set()
        H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        LMAX = self.kv_cap                    # the rows on the device, not the context
        # The prompt is NOT padded. Row alignment is the fp32 matmul's requirement and it
        # is met inside the matmul (see `_MATMUL_ROW_ALIGN` in `ggml_matmul`), where the
        # extra rows are arithmetic and nothing else. Padding here instead makes every LAYER
        # see the extra positions, and a recurrent layer advances its state once per
        # position -- so a stack that is mostly recurrent either lost the fast path or
        # decoded from a state that had run past the end of the prompt. Neither is a trade
        # worth making for rows the matmul can add itself.
        T_real = len(ids)
        T = len(ids)
        end = start + T
        c, s = self._rope_np(start, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        # Attend over the positions this prompt actually fills, not the whole cache. The
        # cache is sized for the context, so scanning all of it costs LMAX/T times more for
        # nothing -- a 28-token prompt in a 16k context is 585x the work, and it showed:
        # attention was 15% of prefill. Slicing the live rows out costs NKV*T*HD floats
        # (~114KB here), which is nothing next to what it saves.
        #
        # The mask is (T, end), not (T, T): a continued prefill's queries see every cached
        # position before them as well as their own, so the causal diagonal sits at `start`.
        m = np.triu(np.full((T, end), -1e9, np.float32), start + 1)
        mask = wt.Tensor(m.reshape(1, T, end))
        h = self._embed_ids(ids, embeds)
        sc = 1.0 / math.sqrt(HD)
        x = self._rms(h, self.layers[0]["in_ln"]) if self.layers else None
        fin = self._rms(h, self.final_norm) if not self.layers else None
        for i, lay in enumerate(self.layers):
            if self._is_linear_layer(i):                       # recurrent (fixed-state) layer
                mix = self._linear_mixer(i, lay, x, T)
            else:                                              # softmax attention layer
                q, k, v = self._qkv(lay, x, T)
                q = self._rope_qk(q, cos_t, sin_t, T); k = self._rope_qk(k, cos_t, sin_t, T)
                K, V = self.Kc[self._kv_i[i]], self.Vc[self._kv_i[i]]
                K.data = wt.kv_write(K.data, wt._contig(k).data, start, T, NKV, HD, LMAX)
                V.data = wt.kv_write(V.data, wt._contig(v).data, start, T, NKV, HD, LMAX)
                # Where the backend has it, attention reads the packed cache where it lies
                # and writes the out-projection's rows: no widened copy of the span, no
                # transpose back. None keeps the path below.
                rows = wt.causal_attention_cache(q, K, V, start, NH, NKV, HD, LMAX, sc)
                if rows is not None:
                    mix = self._attn_out(lay, rows, T, rows=True)
                # Prefill widens the span it attends over back to fp32. It was copying
                # that span anyway, so the three prefill attention kernels never need to
                # know how the cache is stored.
                elif wt.kv_f16() and HD % 2 == 0:
                    Kp = wt.Tensor(wt.kv_unpack(K.data, end, NKV, HD, LMAX))
                    Vp = wt.Tensor(wt.kv_unpack(V.data, end, NKV, HD, LMAX))
                else:
                    Kp = wt.Tensor(wt._contig(K.data[:, :end, :]))
                    Vp = wt.Tensor(wt._contig(V.data[:, :end, :]))
                if rows is None:
                    # `causal_start` instead of the mask tensor: the shape is what the mask
                    # says, so the kernel derives it and nothing seq-squared is built or sent.
                    o = wt.gqa_attention(q, Kp, Vp, mask, scale=sc, causal_start=start)
                    mix = self._attn_out(lay, o, T)
            h, post = self._add_rms(h, mix, lay["post_ln"])
            mlp = self._mlp(lay, post)
            if i + 1 < len(self.layers):
                h, x = self._add_rms(h, mlp, self.layers[i + 1]["in_ln"])
            else:
                h, fin = self._add_rms(h, mlp, self.final_norm)
            # Every eighth layer, hand back what the layers before it finished with. A
            # prompt's intermediates are ~100MB a layer here, and holding all of them to the
            # end of the prefill is several gigabytes on a machine that has none to spare --
            # measured: the ledger climbed 13.5GB to 19.6GB across one prefill without this
            # and stays at 14.0-14.3GB with it. It also makes the prefill faster rather than
            # slower (100.2s to 87.2s), because the memory it stops using was being paged.
            if (i & 7) == 7:
                wt.gpu_reap(budgeted=True)
        # Kept, not just passed on: the load-time smoke test needs the logits this produced,
        # and the tensor is built here either way.
        # The LAST REAL row, which is not the last row when the prompt was padded above.
        self._last_prefill_hidden = wt.Tensor(wt._contig(
            fin.data[T_real - 1:T_real]))
        picked = self._head_argmax(self._last_prefill_hidden)
        self._moe_prefill_seen.update(self._moe_prefill_pending)
        CausalLM._moe_prefill_api_completed(self, T_real)
        return picked

    def _prefill_or_replay_one(self, ids, embeds=None, start=0):
        """Use an already-validated decode graph for one new cached-prefix row.

        A continued chat often has exactly one uncached template token.  Running that
        row through the general prefill builds and submits a second, shape-specific
        command list over the whole model.  The captured decode graph writes the same
        KV row and produces its logits with the current token and position; reusing it
        also avoids reintroducing a cold GPU working set just before the first token.
        The graph key checks the physical plan and buffers, and excludes recurrent
        state.  Embedding overrides and an absent cached prefix keep the general path.
        """
        if (start > 0 and len(ids) == 1 and embeds is None
                and getattr(self, "_gpu", False)
                and getattr(self, "capture_ready", False)):
            split = wt.gqa_tune(self.NH, self.NKV, self.HD, start + 1)
            key = CausalLM._decode_graph_key(self, split)
            if (key is not None and key == getattr(self, "_decode_graph_signature", None)
                    and getattr(self, "_decode_graph_logits", None) is not None):
                self._set_inputs(int(ids[0]), start)
                wt._adam_kernel["platform"].replay("decode")
                self._last_prefill_route = "decode_replay"
                token_buf = getattr(self, "_decode_graph_token", None)
                if token_buf is not None:
                    return self._accept_token(self._token_from_device(token_buf))
                return CausalLM._pick_tensor(self, self._decode_graph_logits)
        self._last_prefill_route = "general"
        return self._prefill(ids, embeds=embeds, start=start)

    def _set_inputs(self, token, pos):
        NKV, HD, LMAX = self.NKV, self.HD, self.kv_cap
        self.h_in.data.buffer.set_data(np.asarray(self.embed[token], np.float32))
        c, s = self._rope_np(pos)
        self.cos_b.data.buffer.set_data(c.reshape(-1)); self.sin_b.data.buffer.set_data(s.reshape(-1))
        if not getattr(self, "_fused_attn", False):    # only the general path reads the mask
            m = np.zeros((1, 1, LMAX), np.float32); m[0, 0, pos + 1:] = -1e9
            self.mask_b.data.buffer.set_data(m)
        self.ctl.buffer.set_data(np.array([pos, 1, NKV, HD, LMAX], np.int32))

    def _decode_graph_key(self, split):
        """Only reuse a graph when its buffers and physical execution plan are stable.

        A recurrent layer may replace its device state during prompt prefill, so its graph
        is deliberately re-recorded.  This is a capability check, not a model-name rule.
        WebGL has no command capture and keeps its equivalent eager API path.
        """
        if not getattr(self, "_gpu", False) or any(
                isinstance(lay, dict) and lay.get("linear") is not None
                for lay in getattr(self, "layers", ())):
            return None
        return (int(self.kv_cap), repr(split), bool(self._device_greedy_ok()),
                bool(getattr(self, "_fused_attn", False)),
                *(getattr(self, name, None) for name in (
                    "_add_rms_execution", "_qkv_execution", "_gate_up_execution",
                    "_qk_norm_rope_execution", "_kv_write_execution",
                    "_head_shape_execution", "_greedy_execution",
                    "_moe_reduce_execution")),
                int(getattr(wt, "_WEIGHT_TUNE_CALLS", 0)))

    def _capture_or_replay_decode(self, plat, split):
        """Execute the first step once, retaining its output for the next compatible turn.

        Recording executes a real step.  Replaying an existing graph also executes exactly
        one real step; neither may be followed by an extra replay at that same position.
        The output Tensor stays referenced while the graph is pinned so a later turn can
        sample its new logits without recording 1,000+ identical dispatches again.
        """
        key = CausalLM._decode_graph_key(self, split)
        if (key is not None and getattr(self, "capture_ready", False)
                and getattr(self, "_decode_graph_signature", None) == key
                and getattr(self, "_decode_graph_logits", None) is not None):
            plat.replay("decode")
            token = getattr(self, "_decode_graph_token", None)
            # Replay submission is asynchronous. Synchronise before returning so the
            # first-token timer includes the real GPU wait just as recording does.
            if token is None:
                self._decode_graph_logits.numpy()
                picked = None
            else:
                picked = self._token_from_device(token)
            return self._decode_graph_logits, token, picked

        plat.beginCapture("decode")
        try:
            logits = self._decode_fwd()
            token = picked = None
            if self._device_greedy_ok():
                token, picked = self._prepare_device_greedy(logits)
            if token is None:
                logits.numpy()
        finally:
            plat.endCapture()
        self.capture_ready = True
        key = CausalLM._decode_graph_key(self, split)
        self._decode_graph_signature = key
        self._decode_graph_logits = logits if key is not None else None
        self._decode_graph_token = token if key is not None else None
        return logits, token, picked

    def _decode_fwd(self, profile=None):
        H, NH, NKV, HD, LMAX = self.H, self.NH, self.NKV, self.HD, self.kv_cap
        sc = 1.0 / math.sqrt(HD); h = self.h_in

        # Profiling is completely absent from production.  A list requests the simple
        # sync-after-stage trace; a dict requests one captured graph per stage so the
        # profiler can replay exactly the command lists production uses.
        _last_mark = [time.perf_counter()]
        _active = [None]
        def start(stage, layer=-1):
            if profile is None:
                return
            _active[0] = (stage, int(layer))
            if isinstance(profile, dict):
                n = len(profile["segments"])
                name = "decode_profile_%03d" % n
                profile["active_name"] = name
                wt._adam_kernel["platform"].beginCapture(name)

        def mark(value):
            if profile is None:
                return
            probe = value[-1] if isinstance(value, (tuple, list)) else value
            stage, layer = _active[0]
            if isinstance(profile, dict):
                wt._adam_kernel["platform"].endCapture()
                probe.numpy()                 # finish the recording before the next segment
                profile["segments"].append({"layer": layer, "stage": stage,
                                             "name": profile.pop("active_name"),
                                             "probe": probe})
                return
            probe.numpy()
            now = time.perf_counter()
            profile.append({"layer": int(layer), "stage": stage,
                            "raw_ms": (now - _last_mark[0]) * 1000.0})
            _last_mark[0] = now

        start("input_norm")
        x = self._rms(h, self.layers[0]["in_ln"]) if self.layers else None
        fin = self._rms(h, self.final_norm) if not self.layers else None
        mark(x if self.layers else fin)
        for i, lay in enumerate(self.layers):
            if self._is_linear_layer(i):
                # A recurrent mixer is capturable once its step runs on the device: the
                # commands are the same every token, and the state it reads and writes in
                # place is a buffer like any other.
                start("mixer", i); mix = self._linear_mixer(i, lay, x, 1)
                mark(mix)
            else:
                start("qkv", i); q, k, v = self._qkv(lay, x, 1, apply_norm=False)
                mark((q, k, v))
                start("rope", i)
                q, k = self._qk_norm_rope(lay, q, k)
                mark((q, k))
                K, V = self.Kc[self._kv_i[i]], self.Vc[self._kv_i[i]]
                start("kv_write", i)
                K.data, V.data = wt.kv_write_pair(
                    K.data, V.data, wt._contig(k).data, wt._contig(v).data,
                    0, 1, NKV, HD, LMAX, ctl=self.ctl,
                    execution=self._kv_write_execution)
                mark(v)
                # One dispatch for the single decode position; falls back for anything the
                # fused kernel does not cover.
                # `ctl` carries the position, so the fused kernel scans only what the
                # conversation has actually filled -- decode speed follows the conversation,
                # not the context the model was loaded with.
                start("attention_out", i)
                o = (wt.gqa_decode(q, K, V, self.mask_b, sc, ctl=self.ctl)
                     if wt._GQA_FUSED else None)
                if o is None:
                    o = wt.gqa_attention(q, K, V, self.mask_b, scale=sc)
                mix = self._attn_out(lay, o, 1)
                mark(mix)
            start("post_add_norm", i)
            h, post = self._add_rms(h, mix, lay["post_ln"])
            mark(post)
            start("mlp", i)
            mlp = self._mlp(lay, post)
            mark(mlp)
            if i + 1 < len(self.layers):
                start("next_add_norm", i)
                h, x = self._add_rms(h, mlp, self.layers[i + 1]["in_ln"])
                mark(x)
            else:
                start("final_add_norm", i)
                h, fin = self._add_rms(h, mlp, self.final_norm)
                mark(fin)
        start("head")
        logits = self._logits_tensor(fin)
        mark(logits)
        return logits

    def profile_decode_layers(self, rounds=3, token=0, pos=0):
        """Measure one-token decode by layer/stage plus the unsplit whole-step latency.

        Each stage is captured separately, then those graphs are replayed in dependency
        order.  The measured empty-replay/readback cost is subtracted as ``net_ms``.  The
        complete decode is also captured as one graph; ``full_step_ms`` is therefore the
        same path production uses and remains the number candidate selection must optimise.
        """
        if not self._gpu:
            if wt._webgl_ready():
                return self._profile_webgl_decode_layers(rounds, token, pos)
            raise RuntimeError("layer decode profiling requires a GPU backend")
        # Profiling writes synthetic token rows into the model's actual captured KV
        # buffers. Invalidate the previous chat's prefix *before* the first write so
        # even a failed/interrupted profile cannot make a later reply reuse stale rows.
        self._kv_drop()
        import statistics as _stats
        rounds = max(1, int(rounds)); token = int(token); pos = int(pos)
        self._reset_linear_state(); self._set_inputs(token, pos)
        self._decode_fwd().numpy()                         # compile/warm outside the sample

        plat = wt._adam_kernel["platform"]
        cap = {"segments": []}
        self._reset_linear_state(); self._set_inputs(token, pos)
        self._decode_fwd(profile=cap)

        # Same host/worker/readback fixed cost as a segment, but with no GPU dispatch.
        # Replaying each graph several times before one readback amortises the map/copy cost
        # that otherwise dwarfs a 20-80 microsecond layer fragment.
        segment_repeat = 16
        plat.beginCapture("decode_profile_empty"); plat.endCapture()
        sync_ms = []
        for _ in range(max(7, rounds)):
            t0 = time.perf_counter()
            for _r in range(segment_repeat):
                plat.replay("decode_profile_empty")
            self.h_in.numpy()
            sync_ms.append((time.perf_counter() - t0) * 1000.0 / segment_repeat)
        sync = _stats.median(sync_ms)

        grouped = {(s["layer"], s["stage"]): [] for s in cap["segments"]}
        for _ in range(rounds):
            self._reset_linear_state(); self._set_inputs(token, pos)
            for segment in cap["segments"]:
                t0 = time.perf_counter()
                for _r in range(segment_repeat):
                    plat.replay(segment["name"])
                segment["probe"].numpy()
                grouped[(segment["layer"], segment["stage"])].append(
                    (time.perf_counter() - t0) * 1000.0 / segment_repeat)

        # Whole captured graph, including the active API token-selection route.
        self._reset_linear_state(); self._set_inputs(token, pos)
        plat.beginCapture("decode_profile_full")
        logits = self._decode_fwd()
        token_t = wt.vocab_argmax(logits.data) if self._device_greedy_ok() else None
        (token_t.get() if token_t is not None else logits.numpy())
        plat.endCapture()
        whole = []
        for _ in range(max(7, rounds)):
            self._set_inputs(token, pos)
            t0 = time.perf_counter(); plat.replay("decode_profile_full")
            (token_t.get() if token_t is not None else logits.numpy())
            whole.append((time.perf_counter() - t0) * 1000.0)
        self._reset_linear_state()

        stages = []
        for (layer, stage), vals in grouped.items():
            raw = _stats.median(vals)
            stages.append({"layer": layer, "stage": stage,
                           "raw_ms": round(raw, 4),
                           "net_ms": round(max(0.0, raw - sync), 4)})
        layer_totals = []
        for layer in range(len(self.layers)):
            vals = [s["net_ms"] for s in stages if s["layer"] == layer]
            layer_totals.append({"layer": layer, "net_ms": round(sum(vals), 4)})
        result = {"rounds": rounds, "segment_repeat": segment_repeat,
                "segment_overhead_ms": round(sync, 4),
                "full_step_ms": round(_stats.median(whole), 4),
                "profiled_raw_ms": round(sum(s["raw_ms"] for s in stages), 4),
                "profiled_net_ms": round(sum(s["net_ms"] for s in stages), 4),
                "stages": stages, "layers": layer_totals,
                "decode_plan": dict(getattr(self, "decode_plan", {}))}
        # Profiling creates hundreds of short-lived captures.  Drop their pins; the next
        # generation records its normal single graph again.
        plat.resetCaptures(); self.capture_ready = False
        self._greedy_chunk_capture_ready = False
        self._decode_graph_signature = None
        self._decode_graph_logits = None
        self._decode_graph_token = None
        return result

    def _profile_webgl_decode_layers(self, rounds=3, token=0, pos=0):
        """WebGL counterpart of :meth:`profile_decode_layers`.

        WebGL cannot replay command captures, so its stages are measured on the eager
        growing-cache path with a synchronization after each boundary.  A no-work readback
        is measured separately and subtracted from every stage.  The unsplit whole-step
        number is timed without instrumentation and is the production number; stage values
        diagnose proportions rather than pretending WebGPU capture timings apply here.
        """
        import statistics as _stats
        rounds = max(1, int(rounds)); token = int(token); pos = max(0, int(pos))
        saved_seen = list(getattr(self, "_seen", ()))
        saved_sampling = self.__dict__.get("_sampling", None)
        self._sampling = {"do_sample": False, "temperature": 0.0}

        def fresh_cache():
            self._reset_linear_state()
            cache = wt.KVCache(self.L, self.NKV, self.HD, self.lmax)
            if pos:
                self._seen = []
                self._kv_forward([token] * pos, 0, cache)
            return cache

        try:
            # Compile every production shader before either the trace or whole-step sample.
            self._seen = []; self._kv_forward([token], pos, fresh_cache())
            probe = wt.Tensor(np.zeros((1,), np.float32)); probe.numpy()
            overhead = []
            for _ in range(max(7, rounds)):
                t0 = time.perf_counter(); probe.numpy()
                overhead.append((time.perf_counter() - t0) * 1000.0)
            sync = _stats.median(overhead)

            grouped = {}
            for _ in range(rounds):
                trace = []; self._seen = []
                self._kv_forward([token], pos, fresh_cache(), profile=trace)
                for row in trace:
                    grouped.setdefault((row["layer"], row["stage"]), []).append(
                        row["raw_ms"])

            whole = []
            for _ in range(max(7, rounds)):
                self._seen = []; cache = fresh_cache()
                t0 = time.perf_counter(); self._kv_forward([token], pos, cache)
                whole.append((time.perf_counter() - t0) * 1000.0)

            stages = []
            for (layer, stage), vals in grouped.items():
                raw = _stats.median(vals)
                stages.append({"layer": layer, "stage": stage,
                               "raw_ms": round(raw, 4),
                               "net_ms": round(max(0.0, raw - sync), 4)})
            layer_totals = []
            for layer in range(len(self.layers)):
                vals = [s["net_ms"] for s in stages if s["layer"] == layer]
                layer_totals.append({"layer": layer, "net_ms": round(sum(vals), 4)})
            return {"backend": "webgl", "rounds": rounds,
                    "segment_overhead_ms": round(sync, 4),
                    "full_step_ms": round(_stats.median(whole), 4),
                    "profiled_raw_ms": round(sum(s["raw_ms"] for s in stages), 4),
                    "profiled_net_ms": round(sum(s["net_ms"] for s in stages), 4),
                    "stages": stages, "layers": layer_totals,
                    "decode_plan": dict(getattr(self, "decode_plan", {}))}
        finally:
            self._reset_linear_state(); self._seen = saved_seen
            if saved_sampling is None:
                self.__dict__.pop("_sampling", None)
            else:
                self._sampling = saved_sampling

    # ---- WebGL path: growing KVCache, fresh forward (no in-place capture) ----
    def _kv_forward(self, ids, pos, cache, embeds=None, rope_pos=None, profile=None):
        """`pos` is the cache position -- how many tokens are already stored -- and is what
        the attention mask is built from. `rope_pos`, when given, is the ROTARY position:
        with multi-axis rope the two are not the same number. An image of 4x4 patches is 16
        tokens but advances the position by 4, because its patches sit beside each other in
        two dimensions rather than one after another. Conflating them puts the text after an
        image at the wrong distance from it."""
        T = len(ids); H, NH, NKV, HD = self.H, self.NH, self.NKV, self.HD
        if T > 1:
            self._moe_prefill_seen = getattr(self, "_moe_prefill_seen", set())
            self._moe_prefill_pending = set()  # WebGL/non-capturable peer of _prefill
        _last_mark = [time.perf_counter()]
        _active = [None]

        def start(stage, layer=-1):
            if profile is not None:
                _active[0] = (stage, int(layer))

        def mark(value):
            if profile is None:
                return
            probe = value[-1] if isinstance(value, (tuple, list)) else value
            if hasattr(probe, "numpy"):
                probe.numpy()
            elif hasattr(probe, "get"):
                probe.get()
            now = time.perf_counter(); stage, layer = _active[0]
            profile.append({"layer": int(layer), "stage": stage,
                            "raw_ms": (now - _last_mark[0]) * 1000.0})
            _last_mark[0] = now

        start("embed_norm")
        c, s = self._rope_np(pos if rope_pos is None else rope_pos, T)
        cos_t, sin_t = wt.Tensor(c), wt.Tensor(s)
        h = self._embed_ids(ids, embeds)
        sc = 1.0 / math.sqrt(HD)
        x = self._rms(h, self.layers[0]["in_ln"]) if self.layers else None
        fin = self._rms(h, self.final_norm) if not self.layers else None
        mark(x if self.layers else fin)
        for i, lay in enumerate(self.layers):
            if self._is_linear_layer(i):                       # recurrent (fixed-state) layer
                start("mixer", i)
                mix = self._linear_mixer(i, lay, x, T)
                mark(mix)
            else:
                start("qkv", i)
                q, k, v = self._qkv(lay, x, T)
                mark((q, k, v))
                start("rope", i)
                q = self._rope_qk(q, cos_t, sin_t, T); k = self._rope_qk(k, cos_t, sin_t, T)
                mark((q, k))
                start("attention_out", i)
                o = cache.attn(i, q, k, v, pos, scale=sc)
                mix = self._attn_out(lay, o, T)
                mark(mix)
            start("post_add_norm", i)
            h, post = self._add_rms(h, mix, lay["post_ln"])
            mark(post)
            start("mlp", i)
            mlp = self._mlp(lay, post)
            mark(mlp)
            if i + 1 < len(self.layers):
                start("next_add_norm", i)
                h, x = self._add_rms(h, mlp, self.layers[i + 1]["in_ln"])
                mark(x)
            else:
                start("final_add_norm", i)
                h, fin = self._add_rms(h, mlp, self.final_norm)
                mark(fin)
        # Keep the last position's hidden state: a multi-token-prediction head drafts from it,
        # and recomputing the trunk to get it back would defeat the point of drafting.
        self._last_hidden = wt.Tensor(wt._contig(fin.data[-1:]))
        start("head")
        picked = self._head_argmax(self._last_hidden)
        mark(picked)
        if T > 1:
            self._moe_prefill_seen.update(self._moe_prefill_pending)
            CausalLM._moe_prefill_api_completed(self, T)
        return picked

    def stream(self, prompt=None, max_new=None, system="You are a helpful assistant.",
               messages=None, tools=None, ids=None, temperature=None, top_p=None,
               top_k=None, seed=None, do_sample=None, repetition_penalty=None, min_p=None,
               constraint=None, max_length=None, min_new_tokens=None, truncate=True,
               enable_thinking=None, presence_penalty=None, frequency_penalty=None,
               stop=None, channels=False, require_known_tools=False,
               **chat_kw):
        """Streaming decode: yield each new token's text as it is produced (render live,
        bounded memory). WebGPU replays a captured step per token; WebGL grows a cache.
        Takes the same parameters as `generate`.

        `channels=True` yields `{"channel": "thinking"|"content", "text": ...}` instead of
        bare text. A reader cannot work this out for itself: models whose template opens the
        reasoning span emit only the CLOSING tag, so pairing tags in the output alone shows
        reasoning as if it were the answer until that tag lands. Tags are also split across
        token boundaries at will, and are held back here rather than leaked into the wrong
        channel. Models without reasoning simply yield everything as "content".

        While the stream runs, `self.stream_n` is how many tokens it has produced -- which is
        NOT how many pieces it has yielded, so anything showing live progress should read it
        rather than count. When the stream ends, `self.last_stream` holds the reply's timings;
        `truncated` there says the length limit, not the model, ended the reply."""
        box = {}
        it = self._stream_raw(prompt, max_new, system, messages, tools, ids, temperature,
                              top_p, top_k, seed, do_sample, repetition_penalty, min_p,
                              constraint, max_length, min_new_tokens, truncate,
                              enable_thinking, presence_penalty, frequency_penalty, stop,
                              box, require_known_tools, **chat_kw)
        return self._stream_channels(it, box) if channels else it

    def _stream_channels(self, it, box):
        """Label a raw text stream by channel. `box` is filled by the generator once it has
        the prompt, which is the only thing that says whether reasoning is already open."""
        ch = None
        for piece in it:
            if ch is None:
                ch = _Channels(box.get("close"))
            for c, t in ch.feed(piece):
                yield {"channel": c, "text": t}
        for c, t in (ch.flush() if ch is not None else ()):
            yield {"channel": c, "text": t}

    def _stream_raw(self, prompt=None, max_new=None, system="You are a helpful assistant.",
                    messages=None, tools=None, ids=None, temperature=None, top_p=None,
                    top_k=None, seed=None, do_sample=None, repetition_penalty=None,
                    min_p=None, constraint=None, max_length=None, min_new_tokens=None,
                    truncate=True, enable_thinking=None, presence_penalty=None,
                    frequency_penalty=None, stop=None, box=None,
                    require_known_tools=False, **chat_kw):
        """The decode loop itself: yields plain text per token."""
        self._check_live()
        eot = self.tok.eot
        self._reset_linear_state()
        if enable_thinking is None:
            enable_thinking = bool((getattr(self, "gen_defaults", {}) or {})
                                   .get("enable_thinking", False))
        if ids is None:
            ids = self.tok.encode_chat(prompt, system, messages=messages, tools=tools,
                                       enable_thinking=enable_thinking, **chat_kw)
        ids, max_new = self._plan_length(ids, max_new, max_length, truncate)
        P = len(ids)
        if box is not None:
            # Does the rendered prompt END inside a reasoning span? That is what decides the
            # starting channel, and only the template knows -- so ask the template's output.
            tail = self.tok.decode(ids[-16:]).rstrip() if ids else ""
            for o, c in _THINK_TAGS:
                if tail.endswith(o):
                    box["close"] = c
                    break
        self._set_sampling(temperature, top_p, top_k, seed, do_sample, repetition_penalty,
                           min_p, self._tool_name_constraint(
                               tools, True if require_known_tools else constraint),
                           min_new_tokens, prompt_ids=ids,
                           presence_penalty=presence_penalty,
                           frequency_penalty=frequency_penalty, stop=stop)
        CausalLM._activate_decode_profile_for_call(self)
        t0 = time.perf_counter()
        # How many tokens this stream has produced so far, readable WHILE it runs. A reader
        # counting the pieces it receives is not counting tokens: a token that completes no
        # character yields nothing, and neither does one whose text is being held back to see
        # which channel it belongs to. Anything reporting live progress needs this instead.
        self.stream_n = 0

        # Where a token's time went, split at the one boundary that decides what to do about
        # it: the GPU step, and everything the host does between steps. A reply that is slow
        # because the device is busy and one that is slow because the sampler is are the same
        # number from outside, and they have nothing in common as problems.
        span = {"gpu": 0.0, "pick": 0.0, "path": "?", "each": [], "recut": [], "pins": []}

        def _stats(n, truncated, steps, t_first):
            each = span["each"]
            out = {"n": n, "truncated": bool(truncated),
                   "ttft_s": round(getattr(self, "_stream_ttft", 0.0), 3),
                   "tok_s": _decode_rate(t_first, steps),
                   "path": span["path"],
                   # What the reply ran against. Every cost that scales with the conversation
                   # scales with this, and a reader comparing two replies cannot tell a slower
                   # model from a longer conversation without it.
                   "context": int(P),
                   "gpu_ms": round(span["gpu"] * 1000 / max(1, steps), 2),
                   "pick_ms": round(span["pick"] * 1000 / max(1, steps), 2)}
            if span.get("prefill_s") is not None:
                out["prefill_s"] = round(float(span["prefill_s"]), 3)
            if span.get("post_prefill_s") is not None:
                out["post_prefill_s"] = round(float(span["post_prefill_s"]), 3)
            if span.get("prefill_route") is not None:
                out["prefill_route"] = span["prefill_route"]
            if getattr(self, "decode_plan", None):
                out["decode_plan"] = dict(self.decode_plan)
            # The mean hides the one shape that changes what to do about a slow reply: an
            # opening that is slower than the rest. A mean says "the model is slow"; a head
            # far above the tail says "it was slow until something warmed up", and those are
            # different problems. A spread (p90 over median) cannot tell them apart -- a slow
            # first tenth sits just under p90 and reads as perfectly uniform.
            if len(each) >= 20:
                # Ten means, one per tenth of the reply. Two numbers say whether the cost
                # moved; ten say WHERE -- a ramp that eases off, a cliff at one point, and a
                # steady climb are three different causes and the first and last token cannot
                # tell them apart.
                b = len(each) / 10.0
                curve = [each[int(i * b):int((i + 1) * b)] for i in range(10)]
                out["gpu_ms_curve"] = [round(sum(c) * 1000 / len(c), 2) for c in curve if c]
                out["gpu_ms_head"] = out["gpu_ms_curve"][0]
                out["gpu_ms_tail"] = out["gpu_ms_curve"][-1]
            # Where the decode graph was re-recorded, in tokens. It happens when the cache
            # runs out of rows mid-reply, and it is the only thing in the loop that REPLACES
            # what every later step runs. If the per-token cost changes shape at one of these
            # points, that is the cause; if it changes anywhere else, this rules the graph out.
            if span["recut"]:
                out["recaptured_at"] = list(span["recut"])
            # What each recording of the decode graph ended up holding: buffers pinned, bytes
            # pinned, bytes left in the reuse pool. Same kernels and same dispatch shapes on
            # both sides of a re-record, so if the cost changes there, this is where the
            # difference is.
            # [buffers, pinned bytes, pool bytes, dispatches after, dispatches before]
            if span["pins"]:
                out["pins"] = [list(p) for p in span["pins"]]
            if span.get("prefilled") is not None:
                out["prefilled"] = int(span["prefilled"])
                out["prefill_d"] = int(span.get("prefill_d") or 0)
                out["prefill_tune_s"] = round(float(span.get("prefill_tune_s") or 0.0), 3)
                out["prefill_tune_calls"] = int(span.get("prefill_tune_calls") or 0)
            self.last_stream = out
            return out

        # Same capture condition as `generate` -- see `_capturable`.
        if not self._capturable():
            # Continue from what this cache already holds -- see `_kv_growing`.
            cache, keep = self._kv_growing(ids)
            nxt = self._kv_forward(ids[keep:], keep, cache)
            self._stream_ttft = time.perf_counter() - t0
            held = list(ids)
            pos = P; n = 0; steps = 0
            span["path"] = "grow"               # the path that cannot capture -- see below
            dec = self.tok.stream_decoder()
            t_first = time.perf_counter()
            try:
                while n < max_new and nxt != eot:
                    piece = dec.push([nxt]); n += 1; steps += 1; self.stream_n = n
                    if piece:
                        yield piece
                    if self._stop_now() or n >= max_new:
                        break                   # final token needs no extra forward
                    held.append(nxt)            # its row is written by the call below
                    _tg = time.perf_counter()
                    nxt = self._kv_forward([nxt], pos, cache); pos += 1
                    _step = time.perf_counter() - _tg
                    span["gpu"] += _step; span["each"].append(_step)
            finally:
                self._kv_commit(held)
            tail = dec.flush()
            if tail:
                yield tail
            _stats(n, n >= max_new and nxt != eot, steps, t_first)
            return
        # Continue from whatever of this prompt the cache already holds, and do not clear it
        # first -- the same two points as in `generate`.
        self._kv_reserve(P + min(max_new, self._KV_HEADROOM))
        keep = self._kv_prefix(ids)
        # What the prompt actually cost, as two numbers rather than one. `ttft_s` alone
        # cannot separate "every row is expensive" from "something happened once": a turn
        # that re-uses its prefix computes a handful of rows and a first turn computes all of
        # them, and the same seconds mean opposite things in the two cases.
        _pd0 = _pin_stats()[3]
        _ts0 = wt._WEIGHT_TUNE_SECONDS
        _tc0 = wt._WEIGHT_TUNE_CALLS
        g0 = CausalLM._prefill_or_replay_one(self, ids[keep:], start=keep)
        span["prefill_s"] = self._stream_ttft = time.perf_counter() - t0
        span["prefill_route"] = self._last_prefill_route
        span["prefilled"] = int(P - keep)
        span["prefill_d"] = int(_pin_stats()[3] - _pd0)
        span["prefill_tune_s"] = wt._WEIGHT_TUNE_SECONDS - _ts0
        span["prefill_tune_calls"] = wt._WEIGHT_TUNE_CALLS - _tc0
        held = list(ids)
        plat = wt._adam_kernel["platform"]
        chunk = (int(getattr(self, "_greedy_chunk_size", 0) or 0)
                 if self._device_greedy_ok() else 0)
        if chunk:
            # The highest API's measured greedy route: one host submission/readback covers
            # several autoregressive steps.  Every internal step still reads the original
            # packed Q6_K row; only the synchronisation boundary moved.
            span["path"] = "replay-chunk%d" % chunk
            dec = self.tok.stream_decoder()
            current = int(g0); pos = P; n = 0; steps = 0
            captured = bool(getattr(self, "_greedy_chunk_capture_ready", False))
            t_first = time.perf_counter()
            try:
                if current != eot and n < max_new:
                    piece = dec.push([current]); n += 1; self.stream_n = n
                    if piece:
                        yield piece
                while current != eot and n < max_new and not self._stop_now():
                    if pos + chunk > self.kv_cap:
                        span["recut"].append(n)
                        self._kv_reserve(pos + chunk)
                        captured = False
                    _tg = time.perf_counter()
                    self._set_chunk_inputs(current, pos, chunk)
                    values = (self._replay_greedy_chunk(chunk) if captured
                              else self._capture_greedy_chunk(chunk))
                    captured = True; self.capture_ready = True
                    elapsed = time.perf_counter() - _tg
                    used = 0
                    for token in values[1:chunk + 1]:
                        held.append(current)       # this input's row was written by the step
                        # Match the ordinary replay path exactly: a selected token always
                        # crosses the generation-state transition, including EOS.  Plain
                        # greedy decoding does not consult `_seen` while choosing this
                        # chunk, but later API calls and diagnostics still observe it.
                        current = self._accept_token(int(token))
                        pos += 1; steps += 1; used += 1
                        if current == eot:
                            break
                        piece = dec.push([current]); n += 1; self.stream_n = n
                        if piece:
                            yield piece
                        if n >= max_new or self._stop_now():
                            break
                    if used:
                        span["gpu"] += elapsed
                        span["each"].extend([elapsed / used] * used)
                    if current == eot or n >= max_new or self._stop_now():
                        break
            finally:
                self._kv_commit(held)
            tail = dec.flush()
            if tail:
                yield tail
            # `_decode_rate` removes the already-available first token.  `steps` counts only
            # the subsequent device steps here (unlike the scalar loop, whose counter also
            # includes that first token), so include it explicitly before asking for rate.
            _stats(n, n >= max_new and current != eot, steps + 1, t_first)
            return
        self._set_inputs(g0, P)
        decode_split = wt.gqa_tune(self.NH, self.NKV, self.HD, P + 1)
        wt._set_split(decode_split)
        _d0 = _pin_stats()[3]
        logits_t, token_t, token0 = CausalLM._capture_or_replay_decode(
            self, plat, decode_split)
        # Recording OR reusing the graph executed the step once. Consume those logits
        # instead of replaying the same position, which advances recurrent state twice.
        captured_step_pending = True
        # The first token waits for this decode execution, whether recorded or replayed.
        # Reporting only prefill hid that wait from the UI's first-token figure.
        self._stream_ttft = time.perf_counter() - t0
        span["post_prefill_s"] = self._stream_ttft - span["prefill_s"]
        span["pins"].append(_pin_stats() + (_d0,))
        nxt = g0; pos = P; n = 0; steps = 0
        span["path"] = "replay"
        dec = self.tok.stream_decoder()
        t_first = time.perf_counter()
        try:
            while n < max_new and nxt != eot:
                piece = dec.push([nxt]); n += 1; steps += 1; self.stream_n = n
                if piece:
                    yield piece
                if self._stop_now() or n >= max_new:
                    # Capture already wrote the first token's KV row. Later final
                    # tokens have not been forwarded yet, just as in generate().
                    if captured_step_pending:
                        held.append(nxt)
                        captured_step_pending = False
                    break
                _step = 0.0
                if captured_step_pending:
                    captured_step_pending = False
                elif pos >= self.kv_cap:            # out of rows -- see `generate`
                    span["recut"].append(n)
                    self._kv_reserve(pos + 1)
                    self._set_inputs(nxt, pos)
                    _d0 = _pin_stats()[3]
                    logits_t, token_t, token0 = CausalLM._capture_or_replay_decode(
                        self, plat, decode_split)
                    span["pins"].append(_pin_stats() + (_d0,))
                else:
                    _tg = time.perf_counter()
                    self._set_inputs(nxt, pos); plat.replay("decode")
                    _step += time.perf_counter() - _tg
                held.append(nxt)                # capture or replay wrote row `pos`
                _tp = time.perf_counter()
                if token_t is not None:
                    _token = self._token_from_device(token_t)
                _step += time.perf_counter() - _tp    # the readback waits for the step
                span["gpu"] += _step; span["each"].append(_step)
                # Device argmax is part of the replay and its four-byte readback belongs to
                # GPU time.  Token bookkeeping remains CPU pick time, just like sampling.
                _tp = time.perf_counter()
                nxt = (CausalLM._pick_tensor(self, logits_t) if token_t is None
                       else self._accept_token(_token)); pos += 1
                span["pick"] += time.perf_counter() - _tp
        finally:
            self._kv_commit(held)
        tail = dec.flush()
        if tail:
            yield tail
        _stats(n, n >= max_new and nxt != eot, steps, t_first)

    def release(self):
        """Free this model's weights (layers, embeddings, head, KV cache). The object must not
        be used afterwards; load it again to use it. See `webtorch.release`.

        Several entry points can share one loaded model (`load()`, `pipeline()`,
        `from_pretrained` dedup on the weights); this releases ONE handle, and the weights
        drop only when the last handle does."""
        from . import _sdk
        self.capture_ready = False
        self._decode_graph_signature = None
        self._decode_graph_logits = None
        self._decode_graph_token = None
        self._kv_drop()                 # the rows it named are about to stop existing
        _sdk._impl_release(self)
        return self

    def _abort_build(self):
        """A load that dies part-way — stopped, or a bad/unsupported file — drops the weights
        it had already uploaded, so a failed load never strands half a model in GPU memory.
        The object was never published to the impl cache, so this is a plain drop."""
        try:
            self.release()
        except Exception:
            pass

    def _check_live(self):
        if self.__dict__.get("_released"):
            raise RuntimeError("this model has been released; load it again to use it")

    def generate(self, prompt=None, max_new=None, system="You are a helpful assistant.",
                 messages=None, tools=None, ids=None, embeds=None, rope_pos=None,
                 temperature=None, top_p=None, top_k=None, seed=None, do_sample=None,
                 repetition_penalty=None, min_p=None, constraint=None,
                 max_length=None, min_new_tokens=None, truncate=True,
                 stream=False, enable_thinking=None, presence_penalty=None,
                 frequency_penalty=None, stop=None, channels=False,
                 require_known_tools=False, **chat_kw):
        """Chat prompt -> decode -> GenResult. WebGPU replays a captured decode step per
        token (~20x); WebGL uses a correct growing-cache forward.

        `stream=True` yields each token's text as it is produced instead of returning one
        GenResult at the end (same parameters either way; `stream(...)` is a shorthand for
        it). Render live, keep the reply visible while it is being written.

        `enable_thinking` defaults to False unless the model or `load(...)` set otherwise: a
        model whose chat template knows the option
        answers directly instead of writing out its reasoning first; pass True to get the
        reasoning too. A template that does not know the option ignores it.

        `messages` is a full conversation instead of the `prompt`/`system` shorthand.

        `stop="..."` or `stop=[...]` ends the reply at any of those strings, on top of the
        model's own end-of-turn token. `presence_penalty`/`frequency_penalty` are the
        additive OpenAI-style knobs; `repetition_penalty` is the multiplicative HF one, and
        they compose. Every sampling parameter here can also be set once at load time (as
        `load(..., temperature=...)`) or come from the model's own generation_config.json;
        what is passed to this call wins for this call.

        Any other keyword goes to the model's chat template unchanged -- `tools=[...]`,
        `reasoning_effort="low"`, whatever that model documents. This is why those work
        without the SDK knowing a thing about them: the template ships with the model and
        is the only place that knows what its options are called. A model whose template
        ignores a keyword simply ignores it.

        `ids`/`embeds` override the prompt encoding: pass prebuilt token ids and/or (T,H)
        input embeddings to decode from a sequence assembled elsewhere. That is the generic
        hook used for multimodality (image/audio embeddings spliced into the token
        embeddings) -- see `webtorch.MultimodalLM` -- so no model-specific decode path is
        needed."""
        self._check_live()
        if stream:
            if embeds is not None:
                raise ValueError("stream=True does not take embeds= — prefill embeddings "
                                 "need the batch path")
            return self.stream(prompt, max_new=max_new, system=system, messages=messages,
                               tools=tools, ids=ids, temperature=temperature, top_p=top_p,
                               top_k=top_k, seed=seed, do_sample=do_sample,
                               repetition_penalty=repetition_penalty, min_p=min_p,
                               constraint=constraint, max_length=max_length,
                               min_new_tokens=min_new_tokens, truncate=truncate,
                               enable_thinking=enable_thinking,
                               presence_penalty=presence_penalty,
                               frequency_penalty=frequency_penalty, stop=stop,
                               channels=channels,
                               require_known_tools=require_known_tools,
                               **chat_kw)
        eot = self.tok.eot
        if enable_thinking is None:
            enable_thinking = bool((getattr(self, "gen_defaults", {}) or {})
                                   .get("enable_thinking", False))
        if ids is None:
            ids = self.tok.encode_chat(prompt, system, messages=messages, tools=tools,
                                       enable_thinking=enable_thinking, **chat_kw)
        ids, max_new = self._plan_length(ids, max_new, max_length, truncate)
        P = len(ids)
        self._set_sampling(temperature, top_p, top_k, seed, do_sample, repetition_penalty,
                           min_p, self._tool_name_constraint(
                               tools, True if require_known_tools else constraint),
                           min_new_tokens, prompt_ids=ids,
                           presence_penalty=presence_penalty,
                           frequency_penalty=frequency_penalty, stop=stop)
        CausalLM._activate_decode_profile_for_call(self)
        self._reset_linear_state()                     # fresh recurrent state per generation

        # A decode step can only be captured if it is the same sequence of GPU commands every
        # token. That rules out a recurrent mixer whose state advances on the HOST -- but not
        # one whose step runs on the device, where the state is just a buffer being read and
        # written in place. So hybrids are captured when every recurrent layer can do that,
        # and fall back to the growing-cache forward when one cannot.
        if not self._capturable():                     # WebGL, host-side mixer, or sparse MoE
            cache, keep = self._kv_growing(ids, embeds, rope_pos)
            t0 = time.perf_counter()
            g0 = self._kv_forward(ids[keep:], keep, cache, embeds=embeds, rope_pos=rope_pos)
            ttft = time.perf_counter() - t0
            held = list(ids)
            gen = [g0]; nxt = g0; pos = P; steps = 0
            # Where rope carries on from. Without media the two counters agree and this is
            # just P; with it, the prompt's rotary positions ran ahead of, or behind, its
            # token count, and the reply has to continue from where THEY ended.
            rnext = P if rope_pos is None else int(np.asarray(rope_pos).max()) + 1
            t_first = time.perf_counter()      # the first token exists as of here
            try:
                while len(gen) < max_new and not self._stop_now():
                    held.append(nxt)                # its row is written by the call below
                    nxt = self._kv_forward([nxt], pos, cache, rope_pos=rnext)
                    pos += 1; rnext += 1; steps += 1
                    if nxt == eot:
                        break
                    gen.append(nxt)
            finally:
                self._kv_commit(held, embeds, rope_pos)
            return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                             round(ttft, 3), _decode_rate(t_first, steps + 1))

        # WebGPU capture path.
        #
        # The cache is NOT cleared first. Prefill writes every row it goes on to read, and a
        # decode step attends over `pos` positions only, so no run ever reads a row it did
        # not write -- clearing was defensive, and on a 30B (2GB of KV) it cost 738ms of
        # every single generation. It is also what made continuing from a cached prefix
        # impossible, which is the far bigger saving: see `_kv_prefix`.
        self._kv_reserve(P + min(max_new, self._KV_HEADROOM))
        keep = self._kv_prefix(ids, embeds, rope_pos)
        t0 = time.perf_counter(); g0 = CausalLM._prefill_or_replay_one(
            self, ids[keep:], embeds=embeds, start=keep)
        ttft = time.perf_counter() - t0
        held = list(ids)                               # rows 0..P-1 hold these
        plat = wt._adam_kernel["platform"]
        chunk = (int(getattr(self, "_greedy_chunk_size", 0) or 0)
                 if self._device_greedy_ok() else 0)
        if chunk:
            # Complete-API phase-two route selected by `_tune_greedy_chunks`.  It is only
            # entered for unconstrained greedy decoding; sampling/penalties/constraints keep
            # the ordinary one-token path because they need the host between every step.
            t_first = time.perf_counter()
            gen = [g0]; current = int(g0); pos = P; steps = 0
            captured = bool(getattr(self, "_greedy_chunk_capture_ready", False))
            try:
                while current != eot and len(gen) < max_new:
                    if pos + chunk > self.kv_cap:
                        self._kv_reserve(pos + chunk)
                        captured = False
                    self._set_chunk_inputs(current, pos, chunk)
                    values = (self._replay_greedy_chunk(chunk) if captured
                              else self._capture_greedy_chunk(chunk))
                    captured = True; self.capture_ready = True
                    for token in values[1:chunk + 1]:
                        held.append(current)
                        current = self._accept_token(int(token))
                        pos += 1; steps += 1
                        if current == eot:
                            break
                        gen.append(current)
                        if len(gen) >= max_new:
                            break
            finally:
                self._kv_commit(held, embeds, rope_pos)
            return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                             round(ttft, 3), _decode_rate(t_first, steps))
        self._set_inputs(g0, P)
        decode_split = wt.gqa_tune(self.NH, self.NKV, self.HD, P + 1)
        wt._set_split(decode_split)
        logits_t, token_t, token0 = CausalLM._capture_or_replay_decode(
            self, plat, decode_split)
        # The capture or existing graph ran the step for real; consume its logits now.
        # Replaying the same position would advance a recurrent state twice.
        t_first = time.perf_counter()          # the first token exists as of here
        gen = [g0]; steps = 1
        held.append(g0)                        # the captured step wrote row P
        nxt = (CausalLM._pick_tensor(self, logits_t) if token_t is None
               else self._accept_token(token0)); pos = P + 1
        # Only what was actually written is claimed: the reply usually ends on a token whose
        # own keys and values were never needed, and claiming it would corrupt the next turn.
        try:
            while nxt != eot and len(gen) < max_new:
                gen.append(nxt)
                if len(gen) >= max_new or self._stop_now():
                    break
                if pos >= self.kv_cap:
                    # Out of rows. Growing moves the buffers the capture was recorded
                    # against, so this step is captured afresh instead of replayed --
                    # beginCapture runs it for real, so the token still comes out of this
                    # iteration.
                    self._kv_reserve(pos + 1)
                    self._set_inputs(nxt, pos)
                    logits_t, token_t, token0 = CausalLM._capture_or_replay_decode(
                        self, plat, decode_split)
                else:
                    self._set_inputs(nxt, pos)
                    plat.replay("decode")
                held.append(nxt)               # row `pos` holds it as of this replay
                nxt = (CausalLM._pick_tensor(self, logits_t) if token_t is None
                       else self._accept_token(self._token_from_device(token_t)))
                pos += 1; steps += 1
        finally:
            self._kv_commit(held, embeds, rope_pos)
        return GenResult(self.tok.decode([g for g in gen if g != eot]), gen,
                         round(ttft, 3), _decode_rate(t_first, steps))
