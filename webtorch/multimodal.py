"""Generic multimodal (vision/audio-language) support.

Every VLM/ALM follows the same three steps, whatever its encoder:

    1. ENCODE   a non-text input (image / audio / video) into a sequence of embeddings that
                live in the decoder's hidden space,
    2. SPLICE   those embeddings into the token-embedding sequence at placeholder-token
                positions the tokenizer produced,
    3. DECODE   with the ordinary causal-LM machinery.

Only step 1 (and the placeholder id) differs per model. This module makes that the *only*
thing a new modality has to supply:

    class MyEncoder:                       # the ENCODER protocol
        token_id = 151655                  # placeholder token spliced over (or .placeholder)
        def encode(self, media, **kw): ...  # -> (n, H) ndarray of embeddings
        def n_tokens(self, media, **kw): ...# optional: count before encoding
        def positions(self, media, **kw):   # optional: (axes, n) rotary layout for its own
            ...                             # tokens, relative to 0 -- see `rope_positions`

    webtorch.register_encoder("my-vision", loader)      # loader(**kw) -> encoder (async)

`MultimodalLM` then pairs ANY CausalLM (the whole generic decoder family — Llama/Qwen2/Qwen3,
dense or MoE, int4/int8/fp16) with ANY registered encoder, so multimodality is not tied to one
model family. Multi-axis rope is part of the shared path rather than a reason to subclass: the
decoder reads its `rope_sections` from its own config, and the encoder — which is the only
thing that knows how its media is laid out in space — supplies the positions for the tokens
it produced. Neither of them needs to know which model the other is.
"""
import numpy as np

from . import _core as wt


# ----------------------------- encoder registry -----------------------------
_ENCODERS = {}


def register_encoder(name, loader, default=False):
    """Register a media encoder so `MultimodalLM` / `pipeline` can build it. `loader(**kw)` is
    async and returns an object implementing the ENCODER protocol (see module docstring).
    Third-party encoders plug in without modifying the SDK."""
    _ENCODERS[name] = loader
    if default or "__default__" not in _ENCODERS:
        _ENCODERS["__default__"] = name


def list_encoders():
    return sorted(k for k in _ENCODERS if k != "__default__")


async def load_encoder(name="auto", **kw):
    key = _ENCODERS.get("__default__") if name in ("auto", None) else name
    if key not in _ENCODERS:
        raise ValueError("no encoder %r (registered: %s) — add one with register_encoder()"
                         % (name, list_encoders()))
    return await _ENCODERS[key](**kw)


# ----------------------------- generic splice -------------------------------
def splice_embeddings(token_embeds, ids, media_embeds, placeholder_id):
    """Replace the rows of `token_embeds` whose token id == `placeholder_id` with
    `media_embeds`, in order. This is the one step every VLM/ALM shares.

    token_embeds : (T, H) float32   — embeddings of the prompt's tokens
    ids          : (T,) int         — the prompt's token ids
    media_embeds : (n, H) float32   — encoder output
    -> (T, H) with the media embeddings spliced in.
    """
    ids = np.asarray(ids)
    out = np.array(token_embeds, np.float32, copy=True)
    slots = np.where(ids == placeholder_id)[0]
    n = len(media_embeds)
    if len(slots) != n:
        raise ValueError("prompt has %d placeholder token(s) but the encoder produced %d "
                         "embedding(s) — the prompt must contain exactly one placeholder per "
                         "media embedding" % (len(slots), n))
    if n:
        out[slots] = np.asarray(media_embeds, np.float32)
    return out


class MultimodalLM:
    """Pair ANY CausalLM with ANY registered media encoder.

        lm  = await webtorch.AutoModelForCausalLM.from_pretrained(path)   # any decoder
        enc = await webtorch.load_encoder("my-vision", ...)               # any encoder
        mm  = webtorch.MultimodalLM(lm, enc)
        print(mm.generate("Describe this.", media=img))

    `.generate(prompt, media=None, ...)` falls back to plain text generation when `media` is
    None, so a multimodal model is a strict superset of a text one. The decoder is untouched:
    this only builds the input embeddings, which is exactly the part that is model-specific."""

    def __init__(self, lm, encoder=None, placeholder_id=None):
        self.lm = lm
        self.encoder = encoder
        self.placeholder_id = (placeholder_id if placeholder_id is not None
                               else getattr(encoder, "token_id", getattr(encoder, "placeholder", None)))

    # expose the decoder's surface so a MultimodalLM is drop-in for a CausalLM
    def __getattr__(self, name):
        return getattr(self.__dict__["lm"], name)

    def embed_prompt(self, ids, media=None, **kw):
        """token ids (+ optional media) -> (T, H) input embeddings, media spliced in."""
        emb = self.lm.embed[np.asarray(ids, np.int64)].astype(np.float32)
        if media is None:
            return emb
        if self.encoder is None:
            raise RuntimeError("no encoder installed: MultimodalLM(lm, encoder=…)")
        if self.placeholder_id is None:
            raise RuntimeError("encoder does not declare a placeholder token id "
                               "(set `.token_id` on it, or pass placeholder_id=…)")
        media_embeds = np.asarray(self.encoder.encode(media, **kw), np.float32)
        return splice_embeddings(emb, ids, media_embeds, self.placeholder_id)

    def rope_positions(self, ids, media=None, **kw):
        """(axes, T) rotary positions for `ids`, or None when the decoder does not use them.

        Text advances every axis together, one per token -- which is ordinary rope written
        as several copies. A media block does not: the encoder returns the layout for its own
        tokens, relative to zero, and it is placed at the running position and advances it by
        its own extent rather than by its token count. That is the whole point of multi-axis
        rope: a grid of patches sits beside itself in two dimensions instead of stretching
        the sequence.

        The encoder supplies the layout because it is the only thing that knows it. An
        encoder without `positions` still works: its tokens then advance one at a time, which
        is what a decoder without `rope_sections` would have done anyway.
        """
        sec = getattr(self.lm, "rope_sections", None)
        if not sec:
            return None
        n_ax = len(sec)
        ids = np.asarray(ids)
        lay = None
        if media is not None and self.encoder is not None:
            f = getattr(self.encoder, "positions", None)
            if f is not None:
                lay = np.asarray(f(media, **kw), np.int64)
                if lay.ndim != 2 or lay.shape[0] != n_ax:
                    raise ValueError("encoder.positions must return (%d, n); got %r"
                                     % (n_ax, getattr(lay, "shape", None)))
        out, cur, i, ph = [], 0, 0, self.placeholder_id
        while i < len(ids):
            if lay is not None and ph is not None and ids[i] == ph:
                out.append(lay + cur)
                i += lay.shape[1]
                cur += int(lay.max()) + 1
            else:
                out.append(np.full((n_ax, 1), cur, np.int64)); i += 1; cur += 1
        return np.concatenate(out, 1) if out else None

    def build_prompt(self, prompt, media=None, system="You are a helpful assistant.", **kw):
        """(ids, embeds) for a prompt that may contain media. The encoder decides how many
        placeholder tokens the media needs (`n_tokens`, else the encoded length), and they are
        inserted before the text — the standard layout for VLMs."""
        n = 0
        if media is not None:
            if self.encoder is None:
                raise RuntimeError("no encoder installed: MultimodalLM(lm, encoder=…)")
            n_fn = getattr(self.encoder, "n_tokens", None)
            n = int(n_fn(media, **kw)) if n_fn else len(self.encoder.encode(media, **kw))
        ids = self.lm.tok.encode_chat(prompt, system)
        if n:
            ph = [self.placeholder_id] * n
            # place the media block right before the user's text: after the final
            # <|im_start|>assistant marker's preceding user content is model-specific, so the
            # simple, model-agnostic choice is to prefix the whole prompt body.
            ids = ids[:1] + ph + ids[1:]
        return ids, (self.embed_prompt(ids, media, **kw) if media is not None else None)

    def generate(self, prompt, media=None, max_new=64, system="You are a helpful assistant.", **kw):
        """Generate text, optionally conditioned on `media`. With `media=None` this is exactly
        the decoder's own text generation; with media, the encoder's embeddings are spliced
        into the input embeddings and the SAME decode path runs (no model-specific branch)."""
        if media is None:
            return self.lm.generate(prompt, max_new=max_new, system=system)
        ids, embeds = self.build_prompt(prompt, media, system=system, **kw)
        # Multi-axis rope is handled here rather than by delegating to a method on the
        # decoder: the decoder knows the sections, the encoder knows the layout, and this is
        # the only place that has both.
        rope_pos = self.rope_positions(ids, media, **kw)
        return self.lm.generate(prompt, max_new=max_new, system=system, ids=ids,
                                embeds=embeds, rope_pos=rope_pos)
