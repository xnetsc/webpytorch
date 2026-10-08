"""Byte-level BPE cuts text the way the model was trained on: the pre-tokenizer's classes are
Unicode letters and numbers, not ASCII, and a merge applies to every occurrence of its pair."""
from webtorch.llm import BPETokenizer, _bytes_to_unicode


def _tok(words, merges):
    b2u = _bytes_to_unicode()
    vocab = {ch: i for i, ch in enumerate(sorted(set(b2u.values())))}
    for w in words:
        vocab.setdefault(w, len(vocab))
    return BPETokenizer(vocab, merges)


def test_a_merge_applies_to_every_occurrence_left_to_right():
    tok = _tok(["ĠĠ", "ĠĠĠ"], ["Ġ Ġ", "ĠĠ Ġ"])
    assert tok._bpe("ĠĠĠ") == ["ĠĠĠ"]
    assert tok._bpe("ĠĠĠĠ") == ["ĠĠ", "ĠĠ"]
    assert tok.encode("   ") == [tok.enc["ĠĠĠ"]]


def test_the_pre_tokenizer_knows_letters_beyond_ascii():
    tok = _tok([], [])
    cut = tok.pat.findall
    # Punctuation joins the letters after it; a run of spaces leaves one for the next word.
    assert cut("[kind] choice  many") == ["[kind", "]", " choice", " ", " many"]
    assert cut("中文混排、emoji 🙂") == ["中文混排", "、emoji", " 🙂"]
    assert cut("café naïve") == ["café", " naïve"]
    assert cut("x 12,345") == ["x", " ", "1", "2", ",", "3", "4", "5"]
    assert cut("):\n    return") == ["):\n", "   ", " return"]
    assert cut("I DON'T") == ["I", " DON", "'T"]
