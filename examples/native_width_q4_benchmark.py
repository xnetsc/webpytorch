"""Interleaved phase-one benchmark: exact scalar vs exact vector quant decoders.

Both sides consume the same original GGML bytes and FP32 activations.  This benchmark is
allowed to decide only between equivalent same-width implementations; it says nothing about
the later cross-width/materialized routing phase.
"""
import json
import statistics
import time

import numpy as np
from js import pythonIO

from webtorch import core as wt


K = 4096
N = 3072
M_VALUES = (1, 2, 32, 128)
ROUNDS = 7
REPEATS = 4

SCALAR = {
    "Q4_0": """
    let o = base + b * 18u; let d = F16(o); let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 2u + j);
      ACC(kb + j, d * (f32(q & 15u) - 8.0));
      ACC(kb + 16u + j, d * (f32(q >> 4u) - 8.0));
    }""",
    "Q4_1": """
    let o = base + b * 20u; let d = F16(o); let mn = F16(o + 2u); let kb = b * 32u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let q = B(o + 4u + j);
      ACC(kb + j, d * f32(q & 15u) + mn);
      ACC(kb + 16u + j, d * f32(q >> 4u) + mn);
    }""",
    "Q4_K": """
    let o = base + b * 144u; let d = F16(o); let dmin = F16(o + 2u);
    let so = o + 4u; let qo = o + 16u; let kb = b * 256u;
    for (var g: u32 = 0u; g < 4u; g = g + 1u) {
      let i0 = 2u * g; let s1 = k4sc(so, i0); let s2 = k4sc(so, i0 + 1u);
      let d1 = d * s1.x; let m1 = dmin * s1.y;
      let d2 = d * s2.x; let m2 = dmin * s2.y;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let q = B(qo + g * 32u + l);
        ACC(kb + i0 * 32u + l, d1 * f32(q & 15u) - m1);
        ACC(kb + (i0 + 1u) * 32u + l, d2 * f32(q >> 4u) - m2);
      }
    }""",
    "Q5_0": wt._Q5_0_DEC,
    "Q5_1": wt._Q5_1_DEC,
    "Q6_K": wt._Q6K_DEC,
    "Q1_0": wt._Q1_0_DEC,
    "Q2_0": wt._Q2_0_DEC,
    "TQ2_0": wt._TQ2_0_DEC,
    "IQ4_NL": wt._IQ4NL_DEC,
    "IQ2_XXS": wt._IQ2XXS_DEC,
    "TQ1_0": wt._TQ1_0_DEC,
    "MXFP4": wt._MXFP4_DEC,
    "NVFP4": wt._NVFP4_DEC,
    "IQ1_S": wt._IQ1S_DEC,
    "IQ1_M": wt._IQ1M_DEC,
    "Q8_0": """
    let o = base + b * 34u; let d = F16(o); let kb = b * 32u;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      ACC(kb + j, d * I8(o + 2u + j));
    }""",
}

VECTOR = {
    "Q4_0": wt._Q4_0_VEC_DEC,
    "Q4_1": wt._Q4_1_VEC_DEC,
    "Q4_K": wt._Q4K_DEC,
    "Q5_0": wt._Q5_0_VEC_DEC,
    "Q5_1": wt._Q5_1_VEC_DEC,
    "Q6_K": wt._Q6K_VEC_DEC,
    "Q1_0": wt._Q1_0_VEC_DEC,
    "Q2_0": wt._Q2_0_VEC_DEC,
    "TQ2_0": wt._TQ2_0_VEC_DEC,
    "IQ4_NL": wt._IQ4NL_VEC_DEC,
    "IQ2_XXS": wt._IQ2XXS_VEC_DEC,
    "TQ1_0": wt._TQ1_0_VEC_DEC,
    "MXFP4": wt._MXFP4_VEC_DEC,
    "NVFP4": wt._NVFP4_VEC_DEC,
    "IQ1_S": wt._IQ1S_VEC_DEC,
    "IQ1_M": wt._IQ1M_VEC_DEC,
    "Q8_0": wt._Q8_0_DEC,
}

VECTOR_HELPERS = {
    "Q4_0": wt._Q4V_FN,
    "Q4_1": wt._Q4V_FN,
    "Q4_K": wt._Q4V_FN,
    "Q5_0": wt._Q5V_FN,
    "Q5_1": wt._Q5V_FN,
    "Q6_K": wt._Q6V_FN,
    "Q1_0": "",
    "Q2_0": "",
    "TQ2_0": "",
    "IQ4_NL": "",
    "IQ2_XXS": "",
    "TQ1_0": wt._TQ1V_FN,
    "MXFP4": "",
    "NVFP4": "",
    "IQ1_S": "",
    "IQ1_M": "",
    "Q8_0": "",
}


def make_raw(name, rng):
    vals, block_bytes = wt._GGML_TYPES[name][2:4]
    out = bytearray()
    for _ in range(N * (K // vals)):
        if name == "Q4_0":
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "Q4_1":
            out.extend(np.asarray([0.02, -0.15], np.float16).tobytes())
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "Q4_K":
            out.extend(np.asarray([0.02, 0.01], np.float16).tobytes())
            out.extend(rng.integers(0, 256, 12, dtype=np.uint8).tobytes())
            out.extend(rng.integers(0, 256, 128, dtype=np.uint8).tobytes())
        elif name == "Q5_0":
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(0, 256, 4, dtype=np.uint8).tobytes())
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "Q5_1":
            out.extend(np.asarray([0.02, -0.15], np.float16).tobytes())
            out.extend(rng.integers(0, 256, 4, dtype=np.uint8).tobytes())
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "Q6_K":
            out.extend(rng.integers(0, 256, 128, dtype=np.uint8).tobytes())
            out.extend(rng.integers(0, 256, 64, dtype=np.uint8).tobytes())
            out.extend(rng.integers(-16, 17, 16, dtype=np.int8).tobytes())
            out.extend(np.float16(0.02).tobytes())
        elif name == "Q8_0":
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(-128, 128, 32, dtype=np.int8).tobytes())
        elif name in ("Q1_0", "Q2_0", "IQ4_NL"):
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "TQ2_0":
            out.extend(rng.integers(0, 256, 64, dtype=np.uint8).tobytes())
            out.extend(np.float16(0.02).tobytes())
        elif name == "TQ1_0":
            out.extend(rng.integers(0, 256, 52, dtype=np.uint8).tobytes())
            out.extend(np.float16(0.02).tobytes())
        elif name == "MXFP4":
            out.append(127)
            out.extend(rng.integers(0, 256, 16, dtype=np.uint8).tobytes())
        elif name == "NVFP4":
            out.extend(np.full(4, 64, np.uint8).tobytes())
            out.extend(rng.integers(0, 256, 32, dtype=np.uint8).tobytes())
        elif name == "IQ1_S":
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(0, 256, 48, dtype=np.uint8).tobytes())
        elif name == "IQ1_M":
            out.extend(rng.integers(0, 256, 48, dtype=np.uint8).tobytes())
            h = int(np.asarray([0.02], np.float16).view(np.uint16)[0])
            s = rng.integers(0, 4096, 4, dtype=np.uint16)
            s[0] |= np.uint16((h & 0x000F) << 12)
            s[1] |= np.uint16((h & 0x00F0) << 8)
            s[2] |= np.uint16((h & 0x0F00) << 4)
            s[3] |= np.uint16(h & 0xF000)
            out.extend(s.tobytes())
        else:
            out.extend(np.float16(0.02).tobytes())
            out.extend(rng.integers(0, 256, 64, dtype=np.uint8).tobytes())
    assert len(out) == N * (K // vals) * block_bytes
    return bytes(out)


def add_variant(type_name, mode, small, mrow):
    key = (type_name, mode, small, False,
           (wt._GGML_KSG, mrow) if mode == 0 else 0)
    if key not in wt._ggml_k["added"]:
        wt._ggml_add(type_name, mode, small, False, mrow)
        wt._ggml_k["added"].add(key)


def run_variant(x, packed, type_name):
    m = int(x.shape[0]); vals = wt._GGML_TYPES[type_name][2]
    if wt._webgl_ready() and not wt._adam_backend_ready():
        return wt._ggml_run_gl(x, packed, type_name, K, N)
    mode = m if m <= 2 else 0
    small = wt._shape_kind(N, K, vals) if mode == 1 else None
    mrow = wt._ggml_mrow(vals, m) if mode == 0 else None
    add_variant(type_name, mode, small, mrow)
    return wt._ggml_run(x, packed, type_name, K, N, small=small)


def timed(fn):
    out = None
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = fn()
    out.get()
    return (time.perf_counter() - t0) * 1000.0 / REPEATS


def compare(vector, scalar):
    vg = np.asarray(vector().get()); sg = np.asarray(scalar().get())
    scale = max(1e-6, float(np.abs(sg).max()))
    rel = float(np.abs(vg - sg).max()) / scale
    if not np.all(np.isfinite(vg)) or rel >= 1e-5:
        raise RuntimeError("same-width vector/scalar mismatch: relative error %g" % rel)
    samples = {"vector": [], "scalar": []}
    for r in range(ROUNDS):
        order = (("vector", vector), ("scalar", scalar))
        if r & 1:
            order = tuple(reversed(order))
        for label, fn in order:
            samples[label].append(timed(fn))
    vm = statistics.median(samples["vector"])
    sm = statistics.median(samples["scalar"])
    return {"vector_ms": round(vm, 4), "scalar_ms": round(sm, 4),
            "vector_speedup": round(sm / vm, 3), "relative_error": rel}


def main():
    global K, N, M_VALUES, ROUNDS, REPEATS
    if not (wt._adam_backend_ready() or wt._webgl_ready()):
        raise RuntimeError("WebGPU/WebGL platform is unavailable")
    if wt._webgl_ready() and not wt._adam_backend_ready():
        # Fragment execution repeats the full K reduction for every output pixel.  Cover
        # decode, verify and two batch buckets without turning a backend audit into a
        # multi-hour shader stress test.
        K, N = 1024, 512
        M_VALUES, ROUNDS, REPEATS = (1, 2, 16, 64), 3, 2
    rng = np.random.default_rng(884)
    result = {"phase": 1,
              "backend": "webgpu" if wt._adam_backend_ready() else "webgl",
              "same_width_only": True, "formats": []}
    for name in ("TQ1_0", "MXFP4", "NVFP4", "IQ1_S", "IQ1_M", "Q8_0",
                 "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q5_1", "Q6_K",
                 "Q1_0", "Q2_0", "TQ2_0", "IQ4_NL", "IQ2_XXS"):
        scalar_alias = "BENCH_SCALAR_" + name
        vector_alias = "BENCH_VECTOR_" + name
        dec, helpers, vals, block_bytes, grid = wt._GGML_TYPES[name]
        scalar_helpers = helpers
        for candidate_helper in (wt._Q4V_FN, wt._Q5V_FN, wt._Q6V_FN, wt._TQ1V_FN):
            scalar_helpers = scalar_helpers.replace(candidate_helper, "")
        vector_helpers = scalar_helpers + VECTOR_HELPERS[name]
        wt._GGML_TYPES[scalar_alias] = (SCALAR[name], scalar_helpers,
                                        vals, block_bytes, grid)
        wt._GGML_TYPES[vector_alias] = (VECTOR[name], vector_helpers,
                                        vals, block_bytes, grid)
        layer = wt.GGMLLinear(make_raw(name, rng), name, K, N, execution="stored")
        row = {"format": name, "shapes": []}
        for m in M_VALUES:
            x = wt.xp.asarray(rng.standard_normal((m, K)).astype(np.float32))
            try:
                metric = compare(lambda x=x: run_variant(x, layer.packed, vector_alias),
                                 lambda x=x: run_variant(x, layer.packed, scalar_alias))
            except Exception as exc:
                row["error"] = "%s: %s" % (type(exc).__name__, exc)
                print("NATIVE_WIDTH %s REJECT %s" % (name, row["error"]))
                break
            metric["M"] = m
            row["shapes"].append(metric)
        result["formats"].append(row)
        del wt._GGML_TYPES[scalar_alias]
        del wt._GGML_TYPES[vector_alias]
        if "error" not in row:
            print("NATIVE_WIDTH %s %s" % (name, ", ".join(
                "M%d:%.2fx" % (x["M"], x["vector_speedup"]) for x in row["shapes"])))
    result["ok"] = not any("error" in row for row in result["formats"])
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
