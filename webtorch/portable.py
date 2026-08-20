"""Move cached models in and out of the browser.

A model that took an hour to download should be movable without downloading it again --
onto a USB stick, to another machine, into a colleague's browser. That is what this does,
and it works on the cache rather than on any particular model format, so it applies to
anything the readers can fetch.

Everything here streams. The entries are multi-gigabyte, the browser holds them as chunk
records, and no step is allowed to assemble one in memory: export walks chunks and hands
them to a `write` callback, import fills chunks from a `read` callback. The caller owns the
actual IO, exactly as `set_io_read`/`set_io_write` do elsewhere in the SDK.

Shape of the output follows the model:

  * one file (a `.gguf`, say) is written out as itself -- no wrapper, so the exported file
    is the model and any other tool can read it;
  * several files (config + tokenizer + shards) are written as one ZIP, stored uncompressed
    because quantized weights do not compress and the copy would cost time for nothing.

`import_model` accepts either, telling them apart by content rather than by file name.
"""

import binascii
import json
import struct

from . import webio

__all__ = ["export_model", "import_model", "model_groups"]

_ZIP64_LIMIT = 0xFFFFFFFF


def _label(directory, sample_key):
    """A short human-readable name for a group.

    Hub URLs put the repo before a revision marker ("/resolve/<rev>", "/blob/<rev>"), so
    the last path segment is usually a branch name rather than anything meaningful. This
    walks back past such markers to the segments that actually name the repo, and falls
    back to the last segment when the layout is something else.
    """
    parts = [p for p in directory.split("/") if p]
    for i, p in enumerate(parts):
        if p in ("resolve", "blob", "raw") and i >= 1:
            parts = parts[max(0, i - 2):i]
            break
    else:
        parts = parts[-2:]
    name = "/".join(p for p in parts if p not in ("models", "api", "v1")) or directory
    return "%s (%s)" % (name, sample_key.rsplit("/", 1)[-1])


async def model_groups(cache_dir=None):
    """Cached entries grouped into models: `[{"name", "label", "keys", "size", "files"}]`.

    Grouping is by the directory part of the key, which is what separates one repo's files
    from another's. A single-file entry forms its own group.
    """
    items = await webio.list_cache(cache_dir)
    groups = {}
    for e in items:
        key = e["key"]
        name = key.rsplit("/", 1)[0] if "/" in key else key
        g = groups.setdefault(name, {"name": name, "label": _label(name, key),
                                     "keys": [], "size": 0, "total": 0, "files": 0,
                                     "complete": True, "partial": []})
        g["keys"].append(key)
        g["size"] += e["size"]
        # `total` is the full length where the host revealed it; fall back to what is held
        # so a partial group never claims to be smaller than it is.
        g["total"] += e.get("total") or e["size"]
        g["files"] += 1
        if not e["complete"]:
            g["complete"] = False
            g["partial"].append(key)
    return sorted(groups.values(), key=lambda g: g["name"])


async def _stream_entry(key, write, cache_dir=None, on_bytes=None):
    """Feed one cached entry to `write`, a chunk at a time. -> (bytes, crc32)."""
    store = webio._make_store(cache_dir or webio._default_hub_cache())
    await store.open()
    meta = await store.meta(key)
    have = await store.have(key)
    if not have:
        raise KeyError("nothing cached for %r" % key)
    total = 0
    crc = 0
    for i in range(max(have) + 1):
        b = await store.get(key, i)
        if b is None:
            raise ValueError("entry %r is incomplete: chunk %d is missing" % (key, i))
        await write(b)
        crc = binascii.crc32(b, crc)
        total += len(b)
        if on_bytes is not None:
            on_bytes(total)
        del b
    if meta["size"] is not None and total != meta["size"]:
        raise ValueError("entry %r is incomplete: %d of %d bytes" % (key, total, meta["size"]))
    return total, crc & 0xFFFFFFFF


def _zip_name(key):
    """Name inside the archive: the file's own name.

    A group is one directory's worth of files, so basenames are already unique within it,
    and keeping only the basename means `import_model(..., key=<that directory>)` restores
    exactly the keys the export came from.
    """
    return key.rsplit("/", 1)[-1]


async def export_model(keys, write, cache_dir=None, on_progress=None):
    """Write cached entries out through `write` (async, called with `bytes`).

    `keys` is one key or several, from `list_cache` / `model_groups`. One key is written as
    the file itself; several become a stored ZIP. Returns the number of bytes written.
    """
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    if not keys:
        raise ValueError("nothing to export")

    # Refuse before writing a byte. A half-downloaded entry would otherwise produce a file
    # that looks like a model and is not one, and the failure would surface later, in
    # whatever tried to read it.
    store = webio._make_store(cache_dir or webio._default_hub_cache())
    await store.open()
    incomplete = []
    for k in keys:
        m = await store.meta(k)
        if not m["complete"]:
            have = await store.have(k)
            held = sum(1 for _ in have)
            want = ("?" if m["size"] is None
                    else (m["size"] + m["chunk"] - 1) // m["chunk"])
            incomplete.append("%s (%s of %s chunks)" % (k, held, want))
    if incomplete:
        raise ValueError("cannot export a model that is not fully downloaded: "
                         + "; ".join(incomplete))

    written = [0]

    async def out(b):
        await write(b)
        written[0] += len(b)
        if on_progress is not None:
            on_progress(written[0])

    if len(keys) == 1:                       # the model IS the file; do not wrap it
        await _stream_entry(keys[0], out, cache_dir)
        return written[0]

    # Stored ZIP. Sizes and CRCs are not known before the data is read, so each entry is
    # written with a data descriptor (general-purpose bit 3) and the real values follow it.
    central = []
    for key in keys:
        name = _zip_name(key).encode("utf-8")
        offset = written[0]
        await out(struct.pack("<IHHHHHIIIHH", 0x04034B50, 45, 0x08, 0, 0, 0,
                              0, 0, 0, len(name), 0))
        await out(name)
        size, crc = await _stream_entry(key, out, cache_dir)
        if size >= _ZIP64_LIMIT:             # 64-bit data descriptor
            await out(struct.pack("<IIQQ", 0x08074B50, crc, size, size))
        else:
            await out(struct.pack("<IIII", 0x08074B50, crc, size, size))
        central.append((name, crc, size, offset))

    cd_start = written[0]
    for name, crc, size, offset in central:
        z64 = size >= _ZIP64_LIMIT or offset >= _ZIP64_LIMIT
        extra = b""
        if z64:
            extra = struct.pack("<HHQQQ", 0x0001, 24, size, size, offset)
        await out(struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 45, 45, 0x08, 0, 0, 0,
                              crc,
                              _ZIP64_LIMIT if z64 else size,
                              _ZIP64_LIMIT if z64 else size,
                              len(name), len(extra), 0, 0, 0, 0,
                              _ZIP64_LIMIT if z64 else offset))
        await out(name)
        if extra:
            await out(extra)
    cd_size = written[0] - cd_start
    n = len(central)
    if cd_start >= _ZIP64_LIMIT or cd_size >= _ZIP64_LIMIT or n >= 0xFFFF:
        z64_eocd = written[0]
        await out(struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, 45, 45, 0, 0, n, n,
                              cd_size, cd_start))
        await out(struct.pack("<IIQI", 0x07064B50, 0, z64_eocd, 1))
        await out(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 0xFFFF, 0xFFFF,
                              _ZIP64_LIMIT, _ZIP64_LIMIT, 0))
    else:
        await out(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, n, n, cd_size, cd_start, 0))
    return written[0]


async def _fill_entry(key, read, size, cache_dir=None, chunk_mb=16, on_progress=None):
    """Fill one cache entry from `read(offset, length) -> bytes`, a span at a time."""
    chunk = chunk_mb << 20
    await webio.delete_cache(key, cache_dir)       # replace, never merge with older bytes
    done = 0
    while done < size:
        n = min(chunk, size - done)
        b = await read(done, n)
        if not b:
            raise ValueError("import ended early: %d of %d bytes for %r" % (done, size, key))
        # Through write_cache, which is what records the bytes as present; putting chunks
        # into the store directly leaves them invisible to every reader.
        await webio.write_cache(key, bytes(b), cache_dir, offset=done, total=size,
                                chunk_mb=chunk_mb)
        done += len(b)
        if on_progress is not None:
            on_progress(done)
        del b
    return done


async def import_model(read, size, key=None, cache_dir=None, on_progress=None):
    """Put an exported model back into the cache, so loading it needs no network.

    `read(offset, length) -> bytes` supplies the exported bytes -- a plain file or a ZIP
    written by `export_model`; the two are told apart by the archive signature, not by any
    file name. A plain file needs `key` (the cache key to store it under, e.g.
    "modelscope.cn/models/org/repo/resolve/master/model.gguf"); a ZIP carries its own
    names, and `key` is then the prefix they are stored under.

    Returns the list of keys written.
    """
    head = bytes(await read(0, 4))
    if head[:4] not in (b"PK\x03\x04", b"PK\x05\x06"):
        if not key:
            raise ValueError("a plain file needs `key`, the cache key to store it under")
        await _fill_entry(key, read, size, cache_dir, on_progress=on_progress)
        return [key]

    # ZIP: walk local headers. Entries written by export_model are stored (never deflated)
    # and carry their sizes in a data descriptor, so the central directory is read first --
    # it is the only place the sizes are known before the data.
    names = await _zip_central(read, size)
    out = []
    for name, csize, offset in names:
        nlen, elen = struct.unpack("<HH", bytes(await read(offset + 26, 4)))
        start = offset + 30 + nlen + elen
        full = ("%s/%s" % (key.rstrip("/"), name)) if key else name

        async def sub(o, n, _s=start):
            return await read(_s + o, n)

        await _fill_entry(full, sub, csize, cache_dir, on_progress=on_progress)
        out.append(full)
    return out


async def _zip_central(read, size):
    """-> [(name, size, local_header_offset)] from a ZIP's central directory."""
    tail_n = min(size, 66000)
    tail = bytes(await read(size - tail_n, tail_n))
    p = tail.rfind(b"PK\x05\x06")
    if p < 0:
        raise ValueError("not a ZIP: no end-of-central-directory record")
    n, cd_size, cd_off = struct.unpack("<HII", tail[p + 10:p + 20])
    z = tail.rfind(b"PK\x06\x06")
    if z >= 0:                               # ZIP64: the real values live here
        n, _n2, cd_size, cd_off = struct.unpack("<QQQQ", tail[z + 24:z + 56])
    cd = bytes(await read(cd_off, cd_size))
    out = []
    q = 0
    while q + 46 <= len(cd) and cd[q:q + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", cd[q + 20:q + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[q + 28:q + 34])
        off = struct.unpack("<I", cd[q + 42:q + 46])[0]
        name = cd[q + 46:q + 46 + nlen].decode("utf-8", "replace")
        extra = cd[q + 46 + nlen:q + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, off):       # pull the 64-bit values out of the extra field
            e = 0
            while e + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[e:e + 4])
                if hid == 0x0001:
                    vals = struct.unpack("<%dQ" % (hsz // 8), extra[e + 4:e + 4 + (hsz // 8) * 8])
                    it = iter(vals)
                    if usize == 0xFFFFFFFF:
                        usize = next(it, usize)
                    if csize == 0xFFFFFFFF:
                        csize = next(it, csize)
                    if off == 0xFFFFFFFF:
                        off = next(it, off)
                    break
                e += 4 + hsz
        out.append((name, csize, off))
        q += 46 + nlen + elen + clen
    if not out:
        raise ValueError("ZIP has no entries")
    return out
