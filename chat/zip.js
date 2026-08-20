/* Minimal ZIP writer/reader for chat export/import.
   Produces a real .zip (deflate via the browser's CompressionStream, store as fallback),
   so the file opens in any unzip tool and re-imports here. */
(function (g) {
  const enc = new TextEncoder(), dec = new TextDecoder();
  let TBL = null;
  function crcTable() {
    if (TBL) return TBL;
    TBL = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      TBL[n] = c >>> 0;
    }
    return TBL;
  }
  function crc32(buf) {
    const t = crcTable(); let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) c = t[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  async function deflateRaw(bytes) {
    if (typeof CompressionStream === 'undefined') return null;
    try {
      const cs = new CompressionStream('deflate-raw');
      const ab = await new Response(new Blob([bytes]).stream().pipeThrough(cs)).arrayBuffer();
      return new Uint8Array(ab);
    } catch (e) { return null; }
  }
  async function inflateRaw(bytes) {
    const ds = new DecompressionStream('deflate-raw');
    const ab = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
    return new Uint8Array(ab);
  }
  function w32(a, o, v) { a[o] = v & 255; a[o+1] = (v>>>8)&255; a[o+2] = (v>>>16)&255; a[o+3] = (v>>>24)&255; }
  function w16(a, o, v) { a[o] = v & 255; a[o+1] = (v>>>8)&255; }

  /** files: [{name, data:Uint8Array|string}] -> Blob (a valid .zip) */
  async function zip(files) {
    const parts = [], central = []; let off = 0;
    for (const f of files) {
      const nameB = enc.encode(f.name);
      const raw = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
      const crc = crc32(raw);
      let body = await deflateRaw(raw), method = 8;
      if (!body) { body = raw; method = 0; }                 // store if no CompressionStream
      const lh = new Uint8Array(30 + nameB.length);
      w32(lh, 0, 0x04034b50); w16(lh, 4, 20); w16(lh, 6, 0); w16(lh, 8, method);
      w16(lh, 10, 0); w16(lh, 12, 0);                        // time/date (unset)
      w32(lh, 14, crc); w32(lh, 18, body.length); w32(lh, 22, raw.length);
      w16(lh, 26, nameB.length); w16(lh, 28, 0);
      lh.set(nameB, 30);
      parts.push(lh, body);
      const ch = new Uint8Array(46 + nameB.length);
      w32(ch, 0, 0x02014b50); w16(ch, 4, 20); w16(ch, 6, 20); w16(ch, 8, 0);
      w16(ch, 10, method); w16(ch, 12, 0); w16(ch, 14, 0);
      w32(ch, 16, crc); w32(ch, 20, body.length); w32(ch, 24, raw.length);
      w16(ch, 28, nameB.length); w16(ch, 30, 0); w16(ch, 32, 0); w16(ch, 34, 0);
      w16(ch, 36, 0); w32(ch, 38, 0); w32(ch, 42, off);
      ch.set(nameB, 46);
      central.push(ch);
      off += lh.length + body.length;
    }
    const cdSize = central.reduce((s, c) => s + c.length, 0);
    const end = new Uint8Array(22);
    w32(end, 0, 0x06054b50); w16(end, 4, 0); w16(end, 6, 0);
    w16(end, 8, files.length); w16(end, 10, files.length);
    w32(end, 12, cdSize); w32(end, 16, off); w16(end, 20, 0);
    return new Blob([...parts, ...central, end], { type: 'application/zip' });
  }

  /** ArrayBuffer of a .zip -> {name: string} (text entries) */
  async function unzip(buf) {
    const a = new Uint8Array(buf), out = {};
    const r32 = o => (a[o] | (a[o+1]<<8) | (a[o+2]<<16) | (a[o+3]<<24)) >>> 0;
    const r16 = o => a[o] | (a[o+1]<<8);
    let i = 0;
    while (i < a.length - 4) {
      if (r32(i) !== 0x04034b50) { i++; continue; }
      const method = r16(i + 8), csize = r32(i + 18), usize = r32(i + 22);
      const nlen = r16(i + 26), elen = r16(i + 28);
      const name = dec.decode(a.subarray(i + 30, i + 30 + nlen));
      const start = i + 30 + nlen + elen;
      let body = a.subarray(start, start + csize);
      if (method === 8) body = await inflateRaw(body);
      out[name] = dec.decode(body);
      i = start + csize;
      if (!csize && usize) break;      // defensive: avoid spinning on a malformed entry
    }
    return out;
  }
  g.WTZip = { zip, unzip };
})(window);
