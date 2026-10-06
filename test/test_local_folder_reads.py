"""A folder the person picked is read from disk and nowhere else: a loader probing it for an
optional file it does not contain must be told "not found", not sent to the network."""
import asyncio

import pytest

from webtorch import webio


@pytest.fixture
def clean_local(monkeypatch):
    monkeypatch.setattr(webio, "_local_files", {})
    monkeypatch.setattr(webio, "_local_roots", {})
    yield


def test_roots_follow_the_files_registered_under_them(clean_local):
    webio.use_model_file(object(), "pick/config.json")
    webio.use_model_file(object(), "pick/tokenizer.json")
    webio.use_model_file(object(), "single.gguf")
    assert webio._local_roots == {"pick": 2}
    assert webio._local_miss("pick/encoder/config.json")
    assert not webio._local_miss("pick/config.json")
    assert not webio._local_miss("other/config.json")
    assert not webio._local_miss("single.gguf")
    assert not webio._local_miss("https://host/pick/x.json")
    webio.forget_model_file("pick/config.json")
    webio.forget_model_file("pick/tokenizer.json")
    assert webio._local_roots == {}
    assert not webio._local_miss("pick/encoder/config.json")


def test_a_missing_file_in_a_picked_folder_never_reaches_the_network(clean_local, monkeypatch,
                                                                      tmp_path):
    fetched = []

    async def http_get(url, offset=0, length=None, headers=None):
        fetched.append(("GET", url))
        return b"remote"

    async def http_size(url, headers=None):
        fetched.append(("HEAD", url))
        return 6

    async def read_local(handle, offset, length):
        return b"{}"

    async def no_cache(*a, **k):
        return None

    monkeypatch.setattr(webio, "_in_browser", lambda: True)
    monkeypatch.setattr(webio, "http_get", http_get)
    monkeypatch.setattr(webio, "http_size", http_size)
    monkeypatch.setattr(webio, "_read_local_file", read_local)
    monkeypatch.setattr(webio, "read_cache", no_cache)
    saved = webio.get_io_read(), webio.get_io_write()
    try:
        webio.use_default_io(cache=True, cache_dir=str(tmp_path), prefetch=False,
                             persist=False)
        read = webio.get_io_read()
        webio.use_model_file(object(), "pick/config.json")
        assert asyncio.run(read("pick/config.json")) == b"{}"
        for probe in ("pick/encoder/config.json", "pick/decision_config.json",
                      "pick/tokenizer/tokenizer.json"):
            with pytest.raises(FileNotFoundError):
                asyncio.run(read(probe))
        assert fetched == []
    finally:
        webio.set_io_read(saved[0])
        webio.set_io_write(saved[1])
