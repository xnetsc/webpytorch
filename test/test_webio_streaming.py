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
