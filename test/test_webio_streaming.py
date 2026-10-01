import asyncio
import tempfile

import webtorch.webio as webio
from webtorch.webio import _read_streaming


class _Headers:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values.get(name.lower())


class _Chunk:
    def __init__(self, data=None):
        self.done = data is None
        self.value = None if data is None else _Value(data)


class _Value:
    def __init__(self, data):
        self.data = data

    def to_py(self):
        return self

    def tobytes(self):
        return self.data


class _Reader:
    def __init__(self, parts):
        self.parts = iter(parts)

    async def read(self):
        return _Chunk(next(self.parts, None))

    def releaseLock(self):
        pass


class _Response:
    def __init__(self, parts, headers, status=200):
        self.status = status
        self.js_response = type("JSResponse", (), {
            "body": type("Body", (), {"getReader": lambda _self: _Reader(parts)})(),
            "headers": _Headers(headers),
        })()


def test_streaming_does_not_compare_decoded_gzip_bytes_to_wire_content_length():
    response = _Response([b"decoded body"], {
        "content-length": "4",
    })
    assert asyncio.run(_read_streaming(response, "https://example.test/model.json")) == b"decoded body"


def test_streaming_still_rejects_a_short_identity_range():
    response = _Response([b"abc"], {"content-range": "bytes 0-3/20"}, status=206)
    try:
        asyncio.run(_read_streaming(response, "https://example.test/model.bin"))
    except Exception as error:
        assert "truncated response" in str(error)
    else:
        raise AssertionError("short identity response was accepted")


def test_http_get_uses_an_open_ended_range_for_offset_only_reads():
    seen = []
    original = webio._fetch_once

    async def fake_fetch(url, byte_range, headers):
        seen.append((url, byte_range, headers))
        return b"tail"

    webio._fetch_once = fake_fetch
    try:
        assert asyncio.run(webio.http_get("https://example.test/model.bin", 7, None)) == b"tail"
    finally:
        webio._fetch_once = original
    assert seen == [("https://example.test/model.bin", "bytes=7-", None)]


def test_whole_file_read_does_not_start_a_duplicate_prefetch():
    fetches = []
    sizes = []

    async def scenario():
        async def fetch(key, offset, length):
            fetches.append((key, offset, length))
            return b"whole file"

        async def size(key):
            sizes.append(key)
            return 10

        read = webio.prefetch_whole_file(fetch, size=size)
        assert await read("model.json", 0, None) == b"whole file"
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert fetches == [("model.json", 0, None)]
    assert sizes == []


def test_whole_file_read_fetches_past_a_cached_prefix():
    body = b'{"complete": true}'
    fetches = []
    original_get, original_size = webio.http_get, webio.http_size

    async def fake_get(url, offset=0, length=None, headers=None):
        fetches.append((url, offset, length))
        return body[offset:] if length is None else body[offset:offset + length]

    async def fake_size(_url, _headers=None):
        return len(body)

    async def scenario(cache_dir):
        read = webio.hub_read(
            lambda _repo, path: "https://example.test/" + path,
            cache_dir=cache_dir,
            prefetch=False,
        )
        assert await read("org/repo/config.json", 0, 4) == body[:4]
        assert await read("org/repo/config.json", 0, None) == body

    webio.http_get, webio.http_size = fake_get, fake_size
    try:
        with tempfile.TemporaryDirectory() as cache_dir:
            asyncio.run(scenario(cache_dir))
    finally:
        webio.http_get, webio.http_size = original_get, original_size

    assert fetches == [
        ("https://example.test/config.json", 0, 4),
        ("https://example.test/config.json", 4, None),
    ]


def test_read_progress_counts_unique_ranges_not_retries_or_overlap():
    seen = []
    webio.set_read_progress(seen.append)
    try:
        webio._report("model.gguf", 100, 200, 0)
        webio._report("model.gguf", 100, 200, 0)     # exact retry
        webio._report("model.gguf", 100, 200, 50)    # 50 new bytes
        webio._report("model.gguf", 100, 200, 150)   # clamped at EOF
    finally:
        webio.set_read_progress(None)

    assert [event["done"] for event in seen] == [100, 100, 150, 200]
    assert all(event["done"] <= event["total"] for event in seen)


def test_read_progress_merges_disjoint_ranges_when_the_gap_arrives():
    seen = []
    webio.set_read_progress(seen.append)
    try:
        webio._report("model.gguf", 25, 100, 0)
        webio._report("model.gguf", 25, 100, 75)
        webio._report("model.gguf", 50, 100, 25)
    finally:
        webio.set_read_progress(None)

    assert [event["done"] for event in seen] == [25, 50, 100]


def test_read_progress_reclamps_early_ranges_when_eof_becomes_known():
    seen = []
    webio.set_read_progress(seen.append)
    try:
        webio._report("model.gguf", 150, None, 0)
        webio._report("model.gguf", 50, 100, 0)
    finally:
        webio.set_read_progress(None)

    assert [event["done"] for event in seen] == [150, 100]


def test_modelscope_reader_covers_both_origins_because_they_are_not_copies():
    """Measured, three reads each: `mccoysc/xDecision` is on .ai and 404 on .cn, while
    `convaiinnovations/laya-multilingual` is the other way round. A reader pinned to one
    origin silently cannot see part of the hub, and which origin has a repo is not something
    a caller can be expected to know."""
    import inspect

    assert webio.MODELSCOPE_ORIGINS == ("https://modelscope.cn", "https://modelscope.ai")

    built = []
    real = webio._hub_reader
    webio._hub_reader = lambda to_url, *a, **kw: built.append((to_url, kw)) or "reader"
    try:
        assert webio.modelscope_read() == "reader"
        to_url, kw = built.pop()
        urls = [f("org/repo", "weights.bin") for f in to_url]
        assert urls == [
            "https://modelscope.cn/models/org/repo/resolve/master/weights.bin",
            "https://modelscope.ai/models/org/repo/resolve/master/weights.bin",
        ], urls
        # One digest per origin, each asking the origin it belongs to -- a hash from the
        # wrong host would wave through bytes it never saw.
        assert len(kw["digest"]) == 2
        for fn, origin in zip(kw["digest"], webio.MODELSCOPE_ORIGINS):
            assert inspect.getclosurevars(fn).nonlocals["ep"] == origin

        # Pinned to one origin when the caller names one, as before.
        assert webio.modelscope_read(endpoint="https://modelscope.ai") == "reader"
        to_url, kw = built.pop()
        assert [f("org/repo", "w") for f in to_url] == [
            "https://modelscope.ai/models/org/repo/resolve/master/w"]
        assert kw.get("digest") is None
    finally:
        webio._hub_reader = real


def test_one_origins_dead_api_does_not_silence_the_other():
    """The latch that stops re-asking an API that will not answer is per origin. Shared, the
    first origin's CORS failure would also stop the second from ever being asked."""
    webio._API_DEAD.clear()
    webio._API_DEAD["modelscope@https://modelscope.cn"] = True
    assert not webio._API_DEAD.get("modelscope@https://modelscope.ai")
    webio._API_DEAD.clear()
