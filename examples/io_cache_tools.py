"""Generic IO tools: `make_cached_reader` + the persistent-cache management API.

Runs anywhere (host CPython or in Pyodide) — it uses no GPU and no models, just the IO layer.
On the host: `python examples/io_cache_tools.py`. In Pyodide: `await main()`.

Shows:
  * make_cached_reader over a CUSTOM async transport (cache + read-ahead + adaptive concurrency)
  * HttpError to signal a rate-limit so the adaptive limiter backs off and retries
  * cache management: cache_write / cache_list / cache_hosts / cache_size / cache_read /
    cache_delete / cache_clear — entries separated by host/domain
For a real hub, the transport is just `webtorch.http_get` / `http_size` (see hf_read); the same
management functions apply to `webtorch.default_cache_dir()`.
"""
import asyncio, tempfile
import webtorch


async def main():
    cache_dir = tempfile.mkdtemp()

    # ---- 1) make_cached_reader over a custom transport -------------------------------------
    # A toy "object store" holding one 2 MB blob; the transport rate-limits the first 2 hits.
    blob = bytes((i * 13) & 0xFF for i in range(2_000_000))
    hits = {"rl": 0}

    async def fetch(key, offset, length):                 # key == the mapped cache/fetch key
        if hits["rl"] < 2 and offset == 0:                # pretend the server throttles at first
            hits["rl"] += 1
            raise webtorch.HttpError(429, "too many requests")   # -> limiter backs off & retries
        return blob if length is None else blob[offset:offset + length]

    async def size(key):
        return len(blob)

    reader = webtorch.make_cached_reader(
        fetch, size=size, key=lambda name: "s3://mybucket/" + name,   # host-like unique cache key
        cache_dir=cache_dir, max_parallel=4, prefetch=True, chunk_mb=1)

    head = await reader("bigfile.bin", 0, 16)             # first read survives the 429s (retried)
    print("make_cached_reader: read", len(head), "bytes; rate-limit hits handled =", hits["rl"])
    for _ in range(200):                                  # let the background prefetch fill the file
        await asyncio.sleep(0.01)
        if any(e["complete"] for e in await webtorch.cache_list(cache_dir=cache_dir)):
            break
    print("  (background prefetch cached the whole file: served from disk on later reads)")

    # ---- 2) cache management ---------------------------------------------------------------
    # Pre-seed two more entries under DIFFERENT hosts to show domain separation.
    await webtorch.cache_write("https://huggingface.co/Qwen/Demo/resolve/main/config.json",
                               b'{"hello":"hf"}', cache_dir=cache_dir)
    await webtorch.cache_write("https://modelscope.cn/api/v1/models/Qwen/Demo/repo?Revision=master&FilePath=config.json",
                               b'{"hello":"ms"}', cache_dir=cache_dir)

    print("\ncache_list:")
    for e in await webtorch.cache_list(cache_dir=cache_dir):
        print("  %-8s %8d  %s" % (e["host"], e["size"], e["key"]))

    print("cache_hosts (per-domain):", await webtorch.cache_hosts(cache_dir=cache_dir))
    print("cache_size total bytes  :", await webtorch.cache_size(cache_dir=cache_dir))

    hf_cfg = await webtorch.cache_read(
        "https://huggingface.co/Qwen/Demo/resolve/main/config.json", cache_dir=cache_dir)
    print("cache_read HF config    :", hf_cfg)

    await webtorch.cache_delete(
        "https://modelscope.cn/api/v1/models/Qwen/Demo/repo?Revision=master&FilePath=config.json",
        cache_dir=cache_dir)
    print("after cache_delete(ms)  :", [e["host"] for e in await webtorch.cache_list(cache_dir=cache_dir)])

    removed = await webtorch.cache_clear(cache_dir=cache_dir, host="huggingface.co")
    print("cache_clear(host=hf)    : removed %d, remaining %d entries"
          % (removed, len(await webtorch.cache_list(cache_dir=cache_dir))))


if __name__ == "__main__":
    asyncio.run(main())
