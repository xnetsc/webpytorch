"""GGUF (llama.cpp) format loader — independent module, NOT part of webtorch.

Parses the GGUF container (header + metadata KV + tensor info) and dequantizes
ggml block-quant tensors to fp32, byte-compatible with llama.cpp. Dequant is
vectorized numpy. Hook the returned fp32 arrays into webtorch like any weights.

`parse_header(buf)` -> (version, metadata, tensor_infos, data_start).
`dequant(ggml_type, raw_bytes, n_elements)` -> np.float32 (row-major, ggml order).
`GGML_NAMES` maps type ids to names; `dims` in tensor_infos are ggml order
(innermost first) — reverse for a numpy/torch (out,in) shape.
"""
import struct
import numpy as np

GGUF_MAGIC = 0x46554747

# metadata value type enum
U8, I8, U16, I16, U32, I32, F32V, BOOLV, STRING, ARRAY, U64, I64, F64 = range(13)

# ggml tensor types (subset we handle + names)
GGML_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K"}


class _R:
    def __init__(self, b):
        self.b = memoryview(b); self.p = 0

    def raw(self, n):
        if self.p + n > len(self.b):
            raise EOFError("GGUF header exceeds buffer (need >= %d bytes)" % (self.p + n))
        v = self.b[self.p:self.p + n]; self.p += n; return v

    def u32(self): return int.from_bytes(self.raw(4), "little")
    def i32(self): return int.from_bytes(self.raw(4), "little", signed=True)
    def u64(self): return int.from_bytes(self.raw(8), "little")
    def i64(self): return int.from_bytes(self.raw(8), "little", signed=True)
    def string(self):
        n = self.u64(); return bytes(self.raw(n)).decode("utf-8", "replace")

    def value(self, t):
        if t == U8:  return self.raw(1)[0]
        if t == I8:  return int.from_bytes(self.raw(1), "little", signed=True)
        if t == U16: return int.from_bytes(self.raw(2), "little")
        if t == I16: return int.from_bytes(self.raw(2), "little", signed=True)
        if t == U32: return self.u32()
        if t == I32: return self.i32()
        if t == F32V: return struct.unpack("<f", self.raw(4))[0]
        if t == BOOLV: return self.raw(1)[0] != 0
        if t == STRING: return self.string()
        if t == U64: return self.u64()
        if t == I64: return self.i64()
        if t == F64: return struct.unpack("<d", self.raw(8))[0]
        if t == ARRAY:
            at = self.u32(); n = self.u64()
            return [self.value(at) for _ in range(n)]
        raise ValueError("bad metadata type %d" % t)


def parse_header(buf):
    r = _R(buf)
    if r.u32() != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        k = r.string(); vt = r.u32(); meta[k] = r.value(vt)
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        infos.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = meta.get("general.alignment", 32)
    data_start = (r.p + align - 1) // align * align
    return version, meta, infos, data_start


# --------------------------- dequantizers ----------------------------------
def _f16(u8_pairs):  # (blk,2) uint8 -> (blk,1) float32
    return np.ascontiguousarray(u8_pairs).view(np.float16).astype(np.float32)


def _q4k_scales(s):  # s (blk,12) uint8 -> sc,m each (blk,8) float32  (get_scale_min_k4)
    blk = s.shape[0]
    sc = np.empty((blk, 8), np.float32); m = np.empty((blk, 8), np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            m[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0x0F) | ((s[:, j - 4] >> 6) << 4)
            m[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, m


def dequant(ttype, raw, n):
    name = GGML_NAMES.get(ttype, str(ttype))
    u8 = np.frombuffer(raw, np.uint8)
    if name == "F32":
        return np.frombuffer(raw, np.float32)[:n].astype(np.float32)
    if name == "F16":
        return np.frombuffer(raw, np.float16)[:n].astype(np.float32)
    if name == "Q8_0":                       # 32 vals / 34 bytes: f16 d + int8[32]
        blk = n // 32
        b = u8[:blk * 34].reshape(blk, 34)
        d = _f16(b[:, :2])
        qs = b[:, 2:].view(np.int8).astype(np.float32)
        return (d * qs).reshape(-1)[:n]
    if name == "Q4_0":                       # 32 vals / 18 bytes: f16 d + 16 nibble-pairs
        blk = n // 32
        b = u8[:blk * 18].reshape(blk, 18)
        d = _f16(b[:, :2]); qs = b[:, 2:]
        out = np.empty((blk, 32), np.float32)
        out[:, :16] = d * ((qs & 0x0F).astype(np.float32) - 8)
        out[:, 16:] = d * ((qs >> 4).astype(np.float32) - 8)
        return out.reshape(-1)[:n]
    if name == "Q5_0":                       # 32 vals / 22 bytes: f16 d, u32 qh(5th bits), 16 pairs
        blk = n // 32
        b = u8[:blk * 22].reshape(blk, 22)
        d = _f16(b[:, :2])
        qh = np.ascontiguousarray(b[:, 2:6]).view(np.uint32).reshape(blk, 1)
        qs = b[:, 6:22]
        j = np.arange(16, dtype=np.uint32)
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        out = np.empty((blk, 32), np.float32)
        out[:, :16] = d * (((qs & 0x0F) | xh0).astype(np.int32) - 16)
        out[:, 16:] = d * (((qs >> 4) | xh1).astype(np.int32) - 16)
        return out.reshape(-1)[:n]
    if name == "Q5_1":                       # 32 vals / 24 bytes: f16 d, f16 m, u32 qh, 16 pairs
        blk = n // 32
        b = u8[:blk * 24].reshape(blk, 24)
        d = _f16(b[:, :2]); mn = _f16(b[:, 2:4])
        qh = np.ascontiguousarray(b[:, 4:8]).view(np.uint32).reshape(blk, 1)
        qs = b[:, 8:24]
        j = np.arange(16, dtype=np.uint32)
        xh0 = ((qh >> j) << 4) & 0x10
        xh1 = (qh >> (j + 12)) & 0x10
        out = np.empty((blk, 32), np.float32)
        out[:, :16] = d * ((qs & 0x0F) | xh0).astype(np.float32) + mn
        out[:, 16:] = d * ((qs >> 4) | xh1).astype(np.float32) + mn
        return out.reshape(-1)[:n]
    if name == "Q4_1":                       # 32 vals / 20 bytes: f16 d, f16 m, 16 pairs
        blk = n // 32
        b = u8[:blk * 20].reshape(blk, 20)
        d = _f16(b[:, :2]); mn = _f16(b[:, 2:4]); qs = b[:, 4:]
        out = np.empty((blk, 32), np.float32)
        out[:, :16] = d * (qs & 0x0F).astype(np.float32) + mn
        out[:, 16:] = d * (qs >> 4).astype(np.float32) + mn
        return out.reshape(-1)[:n]
    if name == "Q4_K":                       # 256 vals / 144 bytes
        blk = n // 256
        b = u8[:blk * 144].reshape(blk, 144)
        d = _f16(b[:, 0:2]); dmin = _f16(b[:, 2:4])
        sc, m = _q4k_scales(b[:, 4:16])
        qs = b[:, 16:144]
        out = np.empty((blk, 256), np.float32)
        for g in range(4):                   # 4 groups of 32 bytes -> 8 sub-blocks
            q = qs[:, g * 32:(g + 1) * 32]
            i0 = 2 * g
            d1 = d * sc[:, i0:i0 + 1]; m1 = dmin * m[:, i0:i0 + 1]
            d2 = d * sc[:, i0 + 1:i0 + 2]; m2 = dmin * m[:, i0 + 1:i0 + 2]
            out[:, i0 * 32:(i0 + 1) * 32] = d1 * (q & 0x0F).astype(np.float32) - m1
            out[:, (i0 + 1) * 32:(i0 + 2) * 32] = d2 * (q >> 4).astype(np.float32) - m2
        return out.reshape(-1)[:n]
    if name == "Q6_K":                       # 256 vals / 210 bytes
        blk = n // 256
        b = u8[:blk * 210].reshape(blk, 210)
        ql = b[:, 0:128]; qh = b[:, 128:192]
        sc = b[:, 192:208].view(np.int8).astype(np.float32)
        d = _f16(b[:, 208:210])
        out = np.empty((blk, 256), np.float32)
        for half in range(2):
            qlh = ql[:, half * 64:(half + 1) * 64]
            qhh = qh[:, half * 32:(half + 1) * 32]
            sch = sc[:, half * 8:(half + 1) * 8]
            base = half * 128
            for l in range(32):
                ii = l // 16
                q1 = ((qlh[:, l] & 0x0F) | (((qhh[:, l] >> 0) & 3) << 4)).astype(np.int16) - 32
                q2 = ((qlh[:, l + 32] & 0x0F) | (((qhh[:, l] >> 2) & 3) << 4)).astype(np.int16) - 32
                q3 = ((qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)).astype(np.int16) - 32
                q4 = ((qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)).astype(np.int16) - 32
                out[:, base + l] = d[:, 0] * sch[:, ii + 0] * q1
                out[:, base + l + 32] = d[:, 0] * sch[:, ii + 2] * q2
                out[:, base + l + 64] = d[:, 0] * sch[:, ii + 4] * q3
                out[:, base + l + 96] = d[:, 0] * sch[:, ii + 6] * q4
        return out.reshape(-1)[:n]
    raise NotImplementedError("dequant not implemented for %s" % name)


# block byte size per type (for slicing a tensor's raw bytes)
_BLOCK = {"F32": (1, 4), "F16": (1, 2), "Q8_0": (32, 34), "Q4_0": (32, 18),
          "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
          "Q4_K": (256, 144), "Q6_K": (256, 210)}


# Quantization types this loader can dequantize (derived from _BLOCK, which mirrors the
# `dequant` implementations). Anything else — notably the IQ i-quants (IQ1/IQ2/IQ3/IQ4_XS/
# IQ4_NL), which use importance-matrix codebooks — is rejected up-front with a clear error.
SUPPORTED_NAMES = frozenset(_BLOCK)
SUPPORTED_TYPES = frozenset(t for t, n in GGML_NAMES.items() if n in _BLOCK)


def is_supported(ttype):
    return GGML_NAMES.get(ttype) in _BLOCK


def tensor_nbytes(ttype, n):
    name = GGML_NAMES.get(ttype, str(ttype))
    bs, by = _BLOCK[name]
    return (n // bs) * by if bs > 1 else n * by
