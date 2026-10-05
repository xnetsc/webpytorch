import asyncio
import struct

import pytest

from webtorch import ggufload as G


def _s(text):
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, vtype, payload):
    return _s(key) + struct.pack("<I", vtype) + payload


def _array(etype, items):
    if etype == G.STRING:
        body = b"".join(_s(x) for x in items)
    elif etype == G.ARRAY:
        # items are (inner_type, inner_items)
        body = b"".join(_array(t, xs) for t, xs in items)
    else:
        fmt = {G.U32: "<I", G.I32: "<i", G.F32V: "<f"}[etype]
        body = b"".join(struct.pack(fmt, x) for x in items)
    return struct.pack("<IQ", etype, len(items)) + body


TOKENS = ["<pad>", "<eos>", "héllo", "▁world"] * 50
TYPES = [3, 3, 1, 1] * 50
MERGES = [(G.STRING, ["a", "b"]), (G.STRING, ["▁", "w"]), (G.STRING, ["he", "llo"])] * 40


def _gguf():
    kvs = [
        _kv("general.architecture", G.STRING, _s("anything")),
        _kv("general.alignment", G.U32, struct.pack("<I", 32)),
        _kv("anything.tokenizer_json", G.STRING, _s('{"model": {"vocab": {}}}' * 200)),
        _kv("tokenizer.ggml.tokens", G.ARRAY, _array(G.STRING, TOKENS)),
        _kv("tokenizer.ggml.token_type", G.ARRAY, _array(G.I32, TYPES)),
        _kv("tokenizer.ggml.merges", G.ARRAY, _array(G.ARRAY, MERGES)),
        _kv("anything.after_the_arrays", G.U32, struct.pack("<I", 7)),
    ]
    infos = []
    for name, dims, ttype, off in (("a.weight", [4, 2], 1, 0), ("b.weight", [8], 0, 64)):
        infos.append(_s(name) + struct.pack("<I", len(dims))
                     + b"".join(struct.pack("<Q", d) for d in dims)
                     + struct.pack("<IQ", ttype, off))
    head = struct.pack("<IIQQ", G.GGUF_MAGIC, 3, len(infos), len(kvs))
    return head + b"".join(kvs) + b"".join(infos)


def test_metadata_arrays_are_not_decoded_until_read():
    buf = _gguf()
    version, meta, infos, data_start = G.parse_header(buf)
    toks = meta["tokenizer.ggml.tokens"]
    assert isinstance(toks, G.LazyArray)
    assert len(toks) == len(TOKENS)
    assert "decoded" not in repr(toks)          # walked over, not decoded
    assert meta["anything.after_the_arrays"] == 7  # the walk got past them correctly
    assert [i["name"] for i in infos] == ["a.weight", "b.weight"]
    assert data_start % 32 == 0 and data_start >= len(buf)
    # Read, they are exactly what an eager decode gave.
    assert list(toks) == TOKENS and toks[2] == "héllo"
    assert "decoded" in repr(toks)
    assert list(meta["tokenizer.ggml.token_type"]) == TYPES
    assert meta["tokenizer.ggml.merges"] == [list(x) for _, x in MERGES]
    # And they behave as the lists callers used to get.
    assert meta["tokenizer.ggml.token_type"] and toks == TOKENS
    assert [t for i, t in enumerate(toks) if TYPES[i] == 3][:2] == ["<pad>", "<eos>"]


def test_a_short_buffer_says_so():
    buf = _gguf()
    with pytest.raises(EOFError):
        G.parse_header(buf[:len(buf) // 2])


def test_read_header_fetches_each_byte_once_and_resumes_where_it_stopped():
    """The old ladder re-read and re-walked the header from the top on every short read --
    four passes over a 60-MB header. Each read must now fetch only bytes not yet held."""
    buf = _gguf()
    reads = []

    async def read(start, end):
        reads.append((start, end))
        return buf[start:end + 1]

    # A first read far smaller than the header forces several resumptions, including
    # stops in the middle of a string array and of a nested array.
    got = asyncio.run(G.read_header(read, first=64, limit=1 << 20))
    assert got[0] == 3 and len(got[2]) == 2
    assert reads[0][0] == 0
    for (_, prev_end), (start, _) in zip(reads, reads[1:]):
        assert start == prev_end + 1, reads           # contiguous, never overlapping
    assert list(got[1]["tokenizer.ggml.tokens"]) == TOKENS
    assert got[1]["tokenizer.ggml.merges"] == [list(x) for _, x in MERGES]
    whole = G.parse_header(buf)
    assert (got[0], got[2], got[3]) == (whole[0], whole[2], whole[3])


def test_read_header_limit_and_truncated_file():
    buf = _gguf()

    async def read(start, end):
        return buf[start:end + 1]

    with pytest.raises(ValueError, match="header limit"):
        asyncio.run(G.read_header(read, first=64, limit=128))

    async def short(start, end):
        return buf[:len(buf) // 2][start:end + 1]

    with pytest.raises(EOFError):
        asyncio.run(G.read_header(short, first=64, limit=1 << 20))
