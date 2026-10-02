"""Phase-one GPTQ int4/int8 benchmark: scalar unpack vs exact packed-word vec4 dots.

Both candidates consume the same original qweight/qzeros/scales buffers and FP32
activations. No operand is requantized or materialized at another width.
"""
import json
import statistics
import time

import numpy as np
from js import pythonIO

from webtorch import core as wt


K, N, GS = 4096, 3072, 128
M_VALUES = (1, 2, 32, 128)
ROUNDS, REPEATS = 7, 4


def add_kernel(bits, gemv, vector):
    suffix = "%d_%s_%s" % (bits, "gemv" if gemv else "gemm",
                            "vector" if vector else "scalar")
    name = "bench_gptq_" + suffix
    if wt._adam_backend_ready():
        plat = wt._adam_kernel["platform"]
        src = (wt._gptq_gemv_src(bits, GS, vector=vector) if gemv else
               wt._gptq_src(wt._GPTQ_WGSL, bits, GS, vector=vector))
        plat.addKernel(name, {"source": src,
            "bindingTypes": ["read-only-storage"] * 4 + ["storage", "read-only-storage"]})
    else:
        wt._copy_kernel["plat"].addKernel(name, {
            "source": wt._gptq_src(wt._GL_GPTQ, bits, GS, vector=vector)})
    return name


def run(name, x, qweight, qzeros, scales, bits):
    m = int(x.shape[0]); gemv = m == 1
    out = wt._empty((m, N))
    if wt._webgl_ready() and not wt._adam_backend_ready():
        wt._copy_kernel["plat"].runKernel({"name": name,
            "inputs": [{"name": "tex_x", "id": x.buffer.buffer_id},
                       {"name": "tex_s", "id": scales.buffer.buffer_id},
                       {"name": "tex_qw", "id": qweight.buffer.buffer_id},
                       {"name": "tex_qz", "id": qzeros.buffer.buffer_id}],
            "output": out.buffer.buffer_id,
            "uniforms": [{"name": "_ka_tex_output_texture_w",
                           "value": out.buffer.texture_shape.width, "type": "int"},
                         {"name": "M", "value": m, "type": "int"},
                         {"name": "N", "value": N, "type": "int"},
                         {"name": "K", "value": K, "type": "int"},
                         {"name": "gs", "value": GS, "type": "int"}]})
        return out
    meta = wt._adam_kernel["make_meta"]((m, N, K, GS), "u4,u4,u4,u4")
    wt._adam_kernel["platform"].runKernel({"name": name,
        "tensors": [x.buffer.buffer_id, qweight.buffer.buffer_id, qzeros.buffer.buffer_id,
                    scales.buffer.buffer_id, out.buffer.buffer_id, meta.buffer_id],
        "workGroups": {"x": (N + 63) // 64, "y": 1 if gemv else (m + 3) // 4, "z": 1}})
    return out


def timed(fn):
    out = None; t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = fn()
    out.get()
    return (time.perf_counter() - t0) * 1000.0 / REPEATS


def compare(vector, scalar):
    vg = np.asarray(vector().get()); sg = np.asarray(scalar().get())
    rel = float(np.abs(vg - sg).max()) / max(1e-6, float(np.abs(sg).max()))
    if not np.all(np.isfinite(vg)) or rel >= 1e-5:
        raise RuntimeError("GPTQ exact vector/scalar mismatch: relative error %g" % rel)
    samples = {"vector": [], "scalar": []}
    for r in range(ROUNDS):
        order = (("vector", vector), ("scalar", scalar))
        if r & 1:
            order = tuple(reversed(order))
        for label, fn in order:
            samples[label].append(timed(fn))
    vm = statistics.median(samples["vector"]); sm = statistics.median(samples["scalar"])
    return {"vector_ms": round(vm, 4), "scalar_ms": round(sm, 4),
            "vector_speedup": round(sm / vm, 3), "relative_error": rel}


def main():
    global K, N, M_VALUES, ROUNDS, REPEATS
    if not (wt._adam_backend_ready() or wt._webgl_ready()):
        raise RuntimeError("WebGPU/WebGL platform is unavailable")
    if wt._webgl_ready() and not wt._adam_backend_ready():
        K, N = 1024, 512
        M_VALUES, ROUNDS, REPEATS = (1, 2, 16, 64), 3, 2
    rng = np.random.default_rng(732)
    result = {"phase": 1,
              "backend": "webgpu" if wt._adam_backend_ready() else "webgl",
              "family": "GPTQ",
              "same_width_only": True, "formats": []}
    for bits in (4, 8):
        W = (rng.standard_normal((K, N), dtype=np.float32) * np.float32(0.02))
        qw, qz, sc, _, _ = wt._gptq_quantize(W, GS, bits)
        del W
        qw = wt.xp.asarray(qw); qz = wt.xp.asarray(qz); sc = wt.xp.asarray(sc)
        names = {(gemv, vector): add_kernel(bits, gemv, vector)
                 for gemv in (False, True) for vector in (False, True)}
        row = {"format": "GPTQ_INT%d" % bits, "shapes": []}
        for m in M_VALUES:
            x = wt.xp.asarray(rng.standard_normal((m, K), dtype=np.float32))
            gemv = m == 1
            metric = compare(
                lambda x=x, gemv=gemv: run(names[(gemv, True)], x, qw, qz, sc, bits),
                lambda x=x, gemv=gemv: run(names[(gemv, False)], x, qw, qz, sc, bits))
            metric["M"] = m; row["shapes"].append(metric)
        result["formats"].append(row)
        print("GPTQ_NATIVE %s %s" % (row["format"], ", ".join(
            "M%d:%.2fx" % (x["M"], x["vector_speedup"]) for x in row["shapes"])))
    result["ok"] = True
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
