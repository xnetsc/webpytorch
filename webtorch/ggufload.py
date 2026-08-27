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
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
              # i-quants (importance-matrix codebook quantizations)
              16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
              21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16",
              # ternary, microscaling-float and sub-2-bit types
              34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
GGML_IDS = {v: k for k, v in GGML_NAMES.items()}

# IQ4 non-linear codebook: 4-bit indices select from these 16 levels (ggml kvalues_iq4nl).
_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                   1, 13, 25, 38, 53, 69, 89, 113], np.float32)


# E2M1 values, doubled -- ggml's kvalues_fp4, shared by MXFP4 and NVFP4.
_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)



# Kept as float32 scalars so `np.where` returns float32. A bare 0.0 is a Python float,
# which makes the result float64 and doubles every dequantised block behind it.
_F0 = np.float32(0.0)
_F4 = np.float32(4.0)
_F16 = np.float32(16.0)

def _e8m0_half(e):
    """MXFP4's shared exponent: 0.5 * 2^(e-127), with denormal patterns below 2."""
    e = np.asarray(e, np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (np.maximum(e, 1) - 1) << 23)
    return bits.astype(np.uint32).view(np.float32)


def _ue4m3(x):
    """NVFP4's per-sub-block scale: unsigned 4-bit exponent (bias 7), 3-bit mantissa, halved."""
    x = np.asarray(x, np.uint32)
    exp = (x >> 3) & 0xF
    man = (x & 7).astype(np.float32)
    raw = np.where(exp == 0, man * 2.0 ** -9,
                   (1.0 + man / 8.0) * 2.0 ** (exp.astype(np.float32) - 7))
    return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5).astype(np.float32)


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
        # All four groups at once: the low nibbles are the even sub-blocks and the high
        # nibbles the odd ones, so the whole thing is two strided writes. Same reason as
        # Q6_K -- a numpy call costs the same whatever its size, so the loop was the cost.
        q = qs.reshape(blk, 4, 32)
        ds = (d * sc).reshape(blk, 8, 1)
        ms = (dmin * m).reshape(blk, 8, 1)
        out = np.empty((blk, 8, 32), np.float32)
        out[:, 0::2, :] = (q & 0x0F).astype(np.float32) * ds[:, 0::2] - ms[:, 0::2]
        out[:, 1::2, :] = (q >> 4).astype(np.float32) * ds[:, 1::2] - ms[:, 1::2]
        return out.reshape(-1)[:n]
    if name == "Q6_K":                       # 256 vals / 210 bytes
        blk = n // 256
        b = u8[:blk * 210].reshape(blk, 210)
        ql = b[:, 0:128]; qh = b[:, 128:192]
        sc = b[:, 192:208].view(np.int8).astype(np.float32)
        d = _f16(b[:, 208:210])
        out = np.empty((blk, 256), np.float32)
        # Vectorised over the 32 positions rather than looping them. Under WASM a numpy call
        # costs ~10us of fixed overhead whatever its size, so the per-element loop this
        # replaces made ~768 calls per invocation -- about 7ms, and this is the table a
        # decode step reads its token embedding from, once per token. Same arithmetic, same
        # bits out (checked against the loop on random blocks).
        d0 = d[:, 0:1]
        ii = np.arange(32) // 16
        for half in range(2):
            qlh = ql[:, half * 64:(half + 1) * 64]
            qhh = qh[:, half * 32:(half + 1) * 32]
            sch = sc[:, half * 8:(half + 1) * 8]
            base = half * 128
            lo = qlh[:, :32]; hi = qlh[:, 32:]
            q1 = ((lo & 0x0F) | ((qhh & 3) << 4)).astype(np.int16) - 32
            q2 = ((hi & 0x0F) | (((qhh >> 2) & 3) << 4)).astype(np.int16) - 32
            q3 = ((lo >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.int16) - 32
            q4 = ((hi >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.int16) - 32
            out[:, base:base + 32] = d0 * sch[:, ii + 0] * q1
            out[:, base + 32:base + 64] = d0 * sch[:, ii + 2] * q2
            out[:, base + 64:base + 96] = d0 * sch[:, ii + 4] * q3
            out[:, base + 96:base + 128] = d0 * sch[:, ii + 6] * q4
        return out.reshape(-1)[:n]
    if name == "Q5_K":                       # 256 vals / 176 bytes
        # f16 d | f16 dmin | scales[12] (6-bit, get_scale_min_k4) | qh[32] (5th bit) | qs[128]
        blk = n // 256
        b = u8[:blk * 176].reshape(blk, 176)
        d = _f16(b[:, 0:2]); dmin = _f16(b[:, 2:4])
        sc, m = _q4k_scales(b[:, 4:16])
        qh = b[:, 16:48]; qs = b[:, 48:176]
        out = np.empty((blk, 256), np.float32)
        for g in range(4):                   # 4 groups of 64 values -> 2 sub-blocks each
            ql = qs[:, g * 32:(g + 1) * 32]
            i0 = 2 * g
            u1 = np.uint8(1 << i0); u2 = np.uint8(1 << (i0 + 1))
            # np.float32, not 16.0: two Python floats make `np.where` return a float64
            # array, and float32 + float64 promotes the whole dequantised block to eight
            # bytes a value on its way to being stored as four.
            lo = (ql & 0x0F).astype(np.float32) + np.where(qh & u1, _F16, _F0)
            hi = (ql >> 4).astype(np.float32) + np.where(qh & u2, _F16, _F0)
            out[:, i0 * 32:(i0 + 1) * 32] = d * sc[:, i0:i0+1] * lo - dmin * m[:, i0:i0+1]
            out[:, (i0+1) * 32:(i0+2) * 32] = d * sc[:, i0+1:i0+2] * hi - dmin * m[:, i0+1:i0+2]
        return out.reshape(-1)[:n]
    if name == "Q3_K":                       # 256 vals / 110 bytes
        # hmask[32] | qs[64] (2-bit) | scales[12] (6-bit signed-ish, -32) | f16 d
        blk = n // 256
        b = u8[:blk * 110].reshape(blk, 110)
        hm = b[:, 0:32]; qs = b[:, 32:96]; sraw = b[:, 96:108]; d_all = _f16(b[:, 108:110])
        a = sraw.view(np.uint32).reshape(blk, 3).astype(np.uint32)   # aux[0..2]
        k1, k2 = np.uint32(0x03030303), np.uint32(0x0F0F0F0F)
        tmp = a[:, 2].copy()
        aux = np.empty((blk, 4), np.uint32)
        aux[:, 2] = ((a[:, 0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
        aux[:, 3] = ((a[:, 1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
        aux[:, 0] = (a[:, 0] & k2) | (((tmp >> 0) & k1) << 4)
        aux[:, 1] = (a[:, 1] & k2) | (((tmp >> 2) & k1) << 4)
        sc = aux.view(np.int8).reshape(blk, 16).astype(np.float32) - 32.0
        out = np.empty((blk, 256), np.float32); o = 0; is_ = 0
        for nblk in range(2):                # two halves of 128 values
            q = qs[:, nblk * 32:(nblk + 1) * 32]
            for j in range(4):
                shift = 2 * j
                mbit = np.uint8(1 << (nblk * 4 + j))
                for half in range(2):        # 16 low + 16 high values
                    ql = q[:, half * 16:(half + 1) * 16]
                    hmm = hm[:, half * 16:(half + 1) * 16]
                    v = ((ql >> shift) & 3).astype(np.float32) - np.where(hmm & mbit, _F0, _F4)
                    out[:, o:o + 16] = (d_all * sc[:, is_:is_ + 1]) * v
                    o += 16; is_ += 1
        return out.reshape(-1)[:n]
    if name == "Q2_K":                       # 256 vals / 84 bytes
        # scales[16] (4-bit scale + 4-bit min) | qs[64] (2-bit) | f16 d | f16 dmin
        blk = n // 256
        b = u8[:blk * 84].reshape(blk, 84)
        sc8 = b[:, 0:16]; qs = b[:, 16:80]
        d = _f16(b[:, 80:82]); dmin = _f16(b[:, 82:84])
        out = np.empty((blk, 256), np.float32); o = 0; is_ = 0
        for nblk in range(2):
            q = qs[:, nblk * 32:(nblk + 1) * 32]
            for j in range(4):
                shift = 2 * j
                for half in range(2):
                    ql = q[:, half * 16:(half + 1) * 16]
                    sc = sc8[:, is_:is_ + 1]
                    dl = d * (sc & 0x0F).astype(np.float32)
                    ml = dmin * (sc >> 4).astype(np.float32)
                    out[:, o:o + 16] = dl * ((ql >> shift) & 3).astype(np.float32) - ml
                    o += 16; is_ += 1
        return out.reshape(-1)[:n]
    if name == "IQ4_NL":                     # 32 vals / 18 bytes: f16 d + 16 packed nibbles
        blk = n // 32
        b = u8[:blk * 18].reshape(blk, 18)
        d = _f16(b[:, 0:2])                  # (blk,1)
        qs = b[:, 2:18]                      # (blk,16) two 4-bit indices per byte
        lo = _IQ4NL[(qs & 0x0F)]             # first 16 values
        hi = _IQ4NL[(qs >> 4)]               # next 16 values
        return (d * np.concatenate([lo, hi], axis=1)).reshape(-1)[:n]
    if name == "IQ4_XS":                     # 256 vals / 136 bytes
        # block_iq4_xs: f16 d | u16 scales_h | u8 scales_l[4] | u8 qs[128]
        # Each of the 8 sub-blocks of 32 has a 6-bit scale: low nibble from scales_l,
        # high 2 bits from scales_h; the sub-block scale is d * (ls - 32).
        blk = n // 256
        b = u8[:blk * 136].reshape(blk, 136)
        d = _f16(b[:, 0:2])                                          # (blk,1)
        sh = (b[:, 2].astype(np.uint32) | (b[:, 3].astype(np.uint32) << 8))   # (blk,) scales_h
        sl = b[:, 4:8]                                               # (blk,4) scales_l
        qs = b[:, 8:136].reshape(blk, 8, 16)                         # 8 sub-blocks x 16 bytes
        ib = np.arange(8)
        low = (sl[:, ib // 2] >> (4 * (ib % 2))) & 0x0F              # (blk,8) low nibble
        high = (sh[:, None] >> (2 * ib)) & 0x03                      # (blk,8) high 2 bits
        ls = (low.astype(np.int32) | (high.astype(np.int32) << 4)) - 32
        dl = d * ls.astype(np.float32)                               # (blk,8) sub-block scales
        lo = _IQ4NL[(qs & 0x0F)]                                     # (blk,8,16)
        hi = _IQ4NL[(qs >> 4)]                                       # (blk,8,16)
        vals = np.concatenate([lo, hi], axis=2)                      # (blk,8,32)
        return (vals * dl[:, :, None]).reshape(-1)[:n]
    if name == "BF16":                       # top 16 bits of an fp32
        # Bounded slices: widening BF16 costs several times its own size in live temporaries,
        # and a head block is large enough for that to matter under 32-bit WASM.
        src = np.frombuffer(raw, np.uint16)[:n]
        out = np.empty(src.size, np.float32)
        for i in range(0, src.size, 1 << 22):
            j = min(src.size, i + (1 << 22))
            out[i:j] = (src[i:j].astype(np.uint32) << 16).view(np.float32)
        return out
    if name == "TQ1_0":                      # 256 vals / 54 bytes: qs[48] | qh[4] | f16 d
        # Ternary, five values to a byte: each is recovered by multiplying the byte by a
        # power of three (mod 256) and taking the top of byte*3.
        blk = n // 256
        b = u8[:blk * 54].reshape(blk, 54)
        d = _f16(b[:, 52:54])
        out = np.empty((blk, 256), np.float32); o = 0
        for j, cnt in ((0, 32), (32, 16)):
            seg = b[:, j:j + cnt].astype(np.uint16)
            for p in range(5):
                q = (seg * (3 ** p)) & 0xFF
                out[:, o:o + cnt] = ((q * 3) >> 8).astype(np.float32) - 1.0
                o += cnt
        qh = b[:, 48:52].astype(np.uint16)
        for p in range(4):
            q = (qh * (3 ** p)) & 0xFF
            out[:, o:o + 4] = ((q * 3) >> 8).astype(np.float32) - 1.0
            o += 4
        return (out * d).reshape(-1)[:n]
    if name == "TQ2_0":                      # 256 vals / 66 bytes: qs[64] | f16 d
        blk = n // 256
        b = u8[:blk * 66].reshape(blk, 66)
        d = _f16(b[:, 64:66])
        out = np.empty((blk, 256), np.float32); o = 0
        for j in (0, 32):
            seg = b[:, j:j + 32]
            for l in range(4):               # one bit-plane at a time, 32 values each
                out[:, o:o + 32] = ((seg >> (2 * l)) & 3).astype(np.float32) - 1.0
                o += 32
        return (out * d).reshape(-1)[:n]
    if name == "MXFP4":                      # 32 vals / 17 bytes: E8M0 e | qs[16]
        blk = n // 32
        b = u8[:blk * 17].reshape(blk, 17)
        d = _e8m0_half(b[:, 0])[:, None]
        qs = b[:, 1:17]
        out = np.empty((blk, 32), np.float32)
        out[:, :16] = _FP4[qs & 0x0F]
        out[:, 16:] = _FP4[qs >> 4]
        return (out * d).reshape(-1)[:n]
    if name == "NVFP4":                      # 64 vals / 36 bytes: UE4M3 d[4] | qs[32]
        blk = n // 64
        b = u8[:blk * 36].reshape(blk, 36)
        ds = _ue4m3(b[:, 0:4])               # one scale per 16-value sub-block
        out = np.empty((blk, 4, 16), np.float32)
        for s in range(4):
            q = b[:, 4 + s * 8:4 + (s + 1) * 8]
            out[:, s, :8] = _FP4[q & 0x0F]
            out[:, s, 8:] = _FP4[q >> 4]
        return (out * ds[:, :, None]).reshape(-1)[:n]
    if name == "Q1_0":                       # 128 vals / 18 bytes: f16 d | one bit each
        blk = n // 128
        b = u8[:blk * 18].reshape(blk, 18)
        d = _f16(b[:, 0:2])
        bits = np.unpackbits(b[:, 2:18], axis=1, bitorder="little")
        return np.where(bits != 0, d, -d).reshape(-1)[:n]
    if name == "Q2_0":                       # 64 vals / 18 bytes: f16 d | 2 bits each
        blk = n // 64
        b = u8[:blk * 18].reshape(blk, 18)
        d = _f16(b[:, 0:2])
        qs = b[:, 2:18]
        out = np.empty((blk, 16, 4), np.float32)
        for l in range(4):
            out[:, :, l] = ((qs >> (2 * l)) & 3).astype(np.float32)
        return ((out.reshape(blk, 64) - 1.0) * d).reshape(-1)[:n]
    if name in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M"):
        return _dequant_iq(name, u8, n)
    raise NotImplementedError("dequant not implemented for %s" % name)


def _dequant_iq(name, u8, n):
    """i-quants: the packed bytes are INDICES into ggml's codebook grids (see webtorch.iqtables,
    extracted from ggml-common.h) plus a sign pattern and a per-sub-block scale. Vectorized
    transcriptions of ggml's dequantize_row_iq* reference implementations."""
    from . import iqtables as T
    blk = n // 256
    nb32 = 8                                     # QK_K/32 sub-blocks of 32 values
    if name == "IQ2_XXS":                        # d | qs[32] uint16  -> 2+64 = 66 bytes
        b = u8[:blk * 66].reshape(blk, 66)
        d = _f16(b[:, 0:2])
        a32 = np.ascontiguousarray(b[:, 2:66]).view(np.uint32).reshape(blk, nb32, 2)
        a8 = a32.view(np.uint8).reshape(blk, nb32, 8)          # aux8 = first 4 bytes
        db = d[:, :, None] * (0.5 + (a32[:, :, 1] >> 28).astype(np.float32))[:, :, None] * 0.25
        idx = a8[:, :, 0:4].astype(np.int32)                   # (blk,nb32,4) grid index
        sidx = ((a32[:, :, 1:2] >> (7 * np.arange(4))) & 127).astype(np.int32)
        vals = T.IQ2XXS_GRID_U8[idx].astype(np.float32) * T.SIGNS[sidx]   # (blk,nb32,4,8)
        return (vals.reshape(blk, nb32, 32) * db).reshape(-1)[:n]
    if name == "IQ2_XS":                         # d | qs[32] uint16 | scales[8] -> 2+64+8 = 74
        b = u8[:blk * 74].reshape(blk, 74)
        d = _f16(b[:, 0:2])
        qs = np.ascontiguousarray(b[:, 2:66]).view(np.uint16).reshape(blk, nb32, 4).astype(np.int32)
        sc = b[:, 66:74]
        db = np.stack([(0.5 + (sc & 0xF).astype(np.float32)) * 0.25,
                       (0.5 + (sc >> 4).astype(np.float32)) * 0.25], -1)   # (blk,nb32,2)
        db = d[:, :, None] * np.repeat(db, 2, axis=2)                      # l//2 -> (blk,nb32,4)
        vals = T.IQ2XS_GRID_U8[qs & 511].astype(np.float32) * T.SIGNS[qs >> 9]
        return (vals * db[:, :, :, None]).reshape(-1)[:n]
    if name == "IQ2_S":                          # d | qs[64] | signs[32] | qh[8] | scales[8] = 82
        b = u8[:blk * 82].reshape(blk, 82)
        d = _f16(b[:, 0:2])
        # ggml: `signs = qs + QK_K/8` — the sign bytes are the upper half of the qs field
        qs = b[:, 2:34].reshape(blk, nb32, 4).astype(np.int32)
        sg = b[:, 34:66].reshape(blk, nb32, 4).astype(np.int32)
        qh = b[:, 66:74].astype(np.int32)
        sc = b[:, 74:82]
        db = np.stack([(0.5 + (sc & 0xF).astype(np.float32)) * 0.25,
                       (0.5 + (sc >> 4).astype(np.float32)) * 0.25], -1)
        db = d[:, :, None] * np.repeat(db, 2, axis=2)
        hi = ((qh[:, :, None] << (8 - 2 * np.arange(4))) & 0x300)
        # IQ2_S uses the sign byte directly as a mask (not a ksigns index)
        vals = T.IQ2S_GRID_U8[qs | hi].astype(np.float32) * T.SIGNS_BYTE[sg]
        return (vals * db[:, :, :, None]).reshape(-1)[:n]
    if name == "IQ3_XXS":                        # d | qs[96] (64 idx + 32 scales/signs) = 98
        b = u8[:blk * 98].reshape(blk, 98)
        d = _f16(b[:, 0:2])
        qs = b[:, 2:66].reshape(blk, nb32, 8).astype(np.int32)              # 8 grid idx / sub-block
        a32 = np.ascontiguousarray(b[:, 66:98]).view(np.uint32).reshape(blk, nb32)
        db = d * (0.5 + (a32 >> 28).astype(np.float32)) * 0.5               # (blk,nb32)
        sidx = ((a32[:, :, None] >> (7 * np.arange(4))) & 127).astype(np.int32)
        g = T.IQ3XXS_GRID_U8[qs].astype(np.float32).reshape(blk, nb32, 4, 8)  # pairs -> 8 values
        vals = g * T.SIGNS[sidx]
        return (vals.reshape(blk, nb32, 32) * db[:, :, None]).reshape(-1)[:n]
    if name == "IQ1_S":                          # d | qs[32] | qh[8] uint16 = 2+32+16 = 50
        b = u8[:blk * 50].reshape(blk, 50)
        d = _f16(b[:, 0:2])                                    # (blk,1)
        qs = b[:, 2:34].reshape(blk, nb32, 4).astype(np.uint16)
        qh = np.ascontiguousarray(b[:, 34:50]).view(np.uint16).reshape(blk, nb32)
        # per sub-block scale and the shared +/-0.125 offset, both carried in qh's high bits
        dl = d * (2 * ((qh >> 12) & 7).astype(np.float32) + 1.0)               # (blk,nb32)
        delta = np.where((qh & 0x8000) != 0, -T.IQ1S_DELTA, T.IQ1S_DELTA).astype(np.float32)
        # grid index: 8 low bits from qs, 3 high bits from qh, a different triple per l
        shift = (3 * np.arange(4, dtype=np.uint16))[None, None, :]
        idx = qs | (((qh[:, :, None] >> shift) & 7) << 8)                      # (blk,nb32,4)
        g = T.IQ1S_GRID_I8[idx].astype(np.float32)                             # (blk,nb32,4,8)
        out = dl[:, :, None, None] * (g + delta[:, :, None, None])
        return out.reshape(-1)[:n]

    if name == "IQ1_M":                          # qs[32] | qh[16] | scales[8] = 56, no separate d
        b = u8[:blk * 56].reshape(blk, 56)
        qs = b[:, 0:32].reshape(blk, nb32, 4).astype(np.uint16)
        qh = b[:, 32:48].reshape(blk, nb32, 2).astype(np.uint16)
        sc = np.ascontiguousarray(b[:, 48:56]).view(np.uint16).reshape(blk, 4)
        # the block scale is spread over the four scale words' spare nibbles
        su = ((sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00f0)
              | ((sc[:, 2] >> 4) & 0x0f00) | (sc[:, 3] & 0xf000)).astype(np.uint16)
        d = _f16(su.view(np.uint8).reshape(blk, 2))                            # (blk,1)
        ib = np.arange(nb32)
        # two half-sub-block scales per 32 values, packed 3 bits each
        s_lo = (sc[:, ib // 2] >> (6 * (ib % 2) + 0)) & 7
        s_hi = (sc[:, ib // 2] >> (6 * (ib % 2) + 3)) & 7
        dl1 = d * (2 * s_lo.astype(np.float32) + 1.0)
        dl2 = d * (2 * s_hi.astype(np.float32) + 1.0)
        idx = np.empty((blk, nb32, 4), np.uint16)
        idx[:, :, 0] = qs[:, :, 0] | ((qh[:, :, 0] << 8) & 0x700)
        idx[:, :, 1] = qs[:, :, 1] | ((qh[:, :, 0] << 4) & 0x700)
        idx[:, :, 2] = qs[:, :, 2] | ((qh[:, :, 1] << 8) & 0x700)
        idx[:, :, 3] = qs[:, :, 3] | ((qh[:, :, 1] << 4) & 0x700)
        dq = T.IQ1S_DELTA
        delta = np.stack([np.where((qh[:, :, 0] & 0x08) != 0, -dq, dq),
                          np.where((qh[:, :, 0] & 0x80) != 0, -dq, dq),
                          np.where((qh[:, :, 1] & 0x08) != 0, -dq, dq),
                          np.where((qh[:, :, 1] & 0x80) != 0, -dq, dq)],
                         axis=-1).astype(np.float32)                           # (blk,nb32,4)
        g = T.IQ1S_GRID_I8[idx].astype(np.float32)                             # (blk,nb32,4,8)
        # the first two groups of 8 take the low scale, the last two the high one
        dl = np.stack([dl1, dl1, dl2, dl2], axis=-1)                           # (blk,nb32,4)
        out = dl[:, :, :, None] * (g + delta[:, :, :, None])
        return out.reshape(-1)[:n]

    if name == "IQ3_S":                          # d | qs[64] | qh[8] | signs[32] | scales[4] = 110
        b = u8[:blk * 110].reshape(blk, 110)
        d = _f16(b[:, 0:2])
        qs = b[:, 2:66].reshape(blk, nb32, 8).astype(np.int32)
        qh = b[:, 66:74].astype(np.int32)
        sg = b[:, 74:106].reshape(blk, nb32, 4).astype(np.int32)
        sc = b[:, 106:110]
        dbp = np.stack([(1 + 2 * (sc & 0xF)).astype(np.float32),
                        (1 + 2 * (sc >> 4)).astype(np.float32)], -1).reshape(blk, nb32)
        db = d * dbp                                                        # (blk,nb32)
        # high bit per grid index: even slots from bit (8-2l), odd from (7-2l)
        sh_e = (8 - 2 * np.arange(4)); sh_o = (7 - 2 * np.arange(4))
        hi = np.empty((b.shape[0], nb32, 8), np.int32)
        hi[:, :, 0::2] = (qh[:, :, None] << sh_e) & 256
        hi[:, :, 1::2] = (qh[:, :, None] << sh_o) & 256
        g = T.IQ3S_GRID_U8[qs | hi].astype(np.float32).reshape(blk, nb32, 4, 8)
        # IQ3_S uses the sign byte directly as a mask (not a ksigns index)
        vals = g * T.SIGNS_BYTE[sg]
        return (vals.reshape(blk, nb32, 32) * db[:, :, None]).reshape(-1)[:n]
    raise NotImplementedError(name)


# block byte size per type (for slicing a tensor's raw bytes)
_BLOCK = {"F32": (1, 4), "F16": (1, 2), "Q8_0": (32, 34), "Q4_0": (32, 18),
          "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
          "Q4_K": (256, 144), "Q6_K": (256, 210),
          "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
          "Q5_K": (256, 176), "Q3_K": (256, 110), "Q2_K": (256, 84),
          "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ2_S": (256, 82),
          "IQ3_XXS": (256, 98), "IQ3_S": (256, 110),
          "IQ1_S": (256, 50), "IQ1_M": (256, 56),
          "BF16": (1, 2), "TQ1_0": (256, 54), "TQ2_0": (256, 66),
          "MXFP4": (32, 17), "NVFP4": (64, 36), "Q1_0": (128, 18), "Q2_0": (64, 18)}


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
