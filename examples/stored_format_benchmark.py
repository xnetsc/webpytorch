"""Phase-two routing benchmark for exact stored weights versus explicit alternatives.

Phase one is now gated separately by the full native-width format matrix and the GGML/GPTQ
same-width A/B suites. This benchmark therefore has routing authority: correctness comes
first, then the fastest verified implementation wins per family/format/shape/device.
"""
import json
import statistics
import time

import cupy as cp
import numpy as np
from js import pythonIO

from webtorch import core as wt


M_VALUES = (1, 2, 8, 32, 128)
ROUNDS = 5
# Two repetitions keep the largest materialized candidates from retaining several expanded
# copies in the pool while five interleaved rounds still provide a stable median.
REPEATS = 2
K = 1024
N = 512


def correctness_gate():
    cases = ((1, False), (2, False), (0, False), (1, True), (0, True))
    for type_name in sorted(wt._GGML_TYPES):
        for mode, moe in cases:
            wt._ggml_selfcheck(type_name, mode, moe=moe)


def timed(run):
    out = None
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = run()
    out.numpy()                         # one synchronization after a batch of dispatches
    return (time.perf_counter() - t0) * 1000.0 / REPEATS


def compare(stored, alternative):
    # Warm both paths, then alternate their order so clock/thermal drift cannot privilege
    # the path that happened to run first.
    stored().numpy(); alternative().numpy()
    samples = {"stored": [], "alternative": []}
    for r in range(ROUNDS):
        order = (("stored", stored), ("alternative", alternative))
        if r & 1:
            order = tuple(reversed(order))
        for name, fn in order:
            samples[name].append(timed(fn))
    sm = statistics.median(samples["stored"])
    am = statistics.median(samples["alternative"])
    ratio = am / sm
    verdict = "stored_faster" if ratio > 1.05 else ("stored_slower" if ratio < 0.95 else "neutral")
    return {"stored_ms": round(sm, 4), "alternative_ms": round(am, 4),
            "stored_speedup": round(ratio, 3), "verdict": verdict}


def ggml_results():
    rows = []
    rng = np.random.default_rng(17)
    for type_name in sorted(wt._GGML_TYPES):
        vals, block_bytes = wt._GGML_TYPES[type_name][2:4]
        raw = bytes((K // vals) * N * block_bytes)
        layer = wt.GGMLLinear(raw, type_name, K, N, execution="stored")
        materialized_ok = wt.ggml_dequant_ok(type_name)
        row = {"format": type_name, "alternative": "materialized_f32",
               "materialized_verified": bool(materialized_ok), "shapes": []}
        for m in M_VALUES:
            x = wt.Tensor(rng.standard_normal((m, K)).astype(np.float32))
            if materialized_ok:
                layer.execution = "stored"

                def stored(layer=layer, x=x):
                    layer.execution = "stored"
                    return layer(x)

                def alternative(layer=layer, x=x):
                    layer.execution = "materialized"
                    return layer(x)

                measured = compare(stored, alternative)
                measured["M"] = m
                row["shapes"].append(measured)
            else:
                row["shapes"].append({"M": m, "verdict": "stored_required"})
        if materialized_ok:
            probe = wt.Tensor(rng.standard_normal((128, K)).astype(np.float32))
            layer.execution = "auto"
            layer(probe).numpy()
            row["auto_M128"] = wt._TUNED.get(
                ("weight_exec", "ggml", type_name, K, N, 128), "stored")
        rows.append(row)
        print("STORED_BENCH %s %s" % (type_name,
              ", ".join("M%d:%s" % (x["M"], x["verdict"]) for x in row["shapes"])))
    return rows


def gptq_results():
    rows = []
    rng = np.random.default_rng(23)
    for bits in (4, 8):
        dense = wt.Linear(K, N)
        encoded = wt.QuantizedLinear.from_linear(dense, group_size=128, bits=bits)
        row = {"format": "GPTQ_INT%d" % bits, "alternative": "materialized_f32", "shapes": []}
        for m in M_VALUES:
            x = wt.Tensor(rng.standard_normal((m, K)).astype(np.float32))
            # Correctness before speed for this non-GGUF storage path as well.
            encoded.execution = "stored"
            got = encoded(x).numpy(); ref = dense(x).numpy()
            rel = float(np.abs(got - ref).max()) / (float(np.abs(ref).max()) + 1e-9)

            def stored(encoded=encoded, x=x):
                encoded.execution = "stored"
                return encoded(x)

            def alternative(encoded=encoded, x=x):
                encoded.execution = "materialized"
                return encoded(x)

            measured = compare(stored, alternative)
            measured.update({"M": m, "relative_error": round(rel, 6)})
            row["shapes"].append(measured)
        probe = wt.Tensor(rng.standard_normal((128, K)).astype(np.float32))
        encoded.execution = "stored"
        want = encoded(probe).numpy()
        encoded.execution = "auto"
        got = encoded(probe).numpy()
        row["auto_M128"] = wt._TUNED.get(
            ("weight_exec", "gptq", row["format"], K, N, 128), "stored")
        row["auto_relative_error"] = round(
            float(np.abs(got - want).max()) / (float(np.abs(want).max()) + 1e-9), 7)
        rows.append(row)
        print("STORED_BENCH %s %s" % (row["format"],
              ", ".join("M%d:%s" % (x["M"], x["verdict"]) for x in row["shapes"])))
    return rows


def webgl_routes():
    """WebGL keeps packed execution because it has no storage-buffer materializer.

    This is an explicit backend route, not an exception fallback.  Exercise GPTQ against
    its source dense layer so phase two still has a numerical gate on the backend where the
    alternative cannot exist.
    """
    ggml = [{"format": name, "selected": "stored",
             "alternative": "unavailable_on_webgl",
             "reason": "WebGL fragment backend has no compute storage-buffer materializer"}
            for name in sorted(wt._GGML_TYPES)]
    rng = np.random.default_rng(2302)
    gptq = []
    for bits in (4, 8):
        dense = wt.Linear(K, N)
        encoded = wt.QuantizedLinear.from_linear(dense, group_size=128, bits=bits)
        shapes = []
        for m in M_VALUES:
            x = wt.Tensor(rng.standard_normal((m, K)).astype(np.float32))
            got = encoded(x).numpy(); ref = dense(x).numpy()
            rel = float(np.abs(got - ref).max()) / (float(np.abs(ref).max()) + 1e-9)
            if not np.all(np.isfinite(got)) or rel >= 0.15:
                raise RuntimeError("WebGL GPTQ_INT%d accuracy gate failed at M%d: %g"
                                   % (bits, m, rel))
            shapes.append({"M": m, "relative_error_vs_source": round(rel, 6)})
        gptq.append({"format": "GPTQ_INT%d" % bits, "selected": "stored",
                     "alternative": "unavailable_on_webgl", "shapes": shapes})
    return ggml, gptq


def main():
    if not (wt._adam_backend_ready() or wt._webgl_ready()):
        raise RuntimeError("WebGPU/WebGL platform is unavailable: %s" % wt.backend_reason())
    started = time.perf_counter()
    print("CORRECTNESS_GATE starting")
    correctness_gate()
    print("CORRECTNESS_GATE passed")
    if wt._webgl_ready() and not wt._adam_backend_ready():
        ggml, non_gguf = webgl_routes()
        result = {"backend": "webgl", "correctness_gate": True,
                  "routing_authority": True,
                  "status": "phase_two_production_routing",
                  "ggml": ggml, "non_gguf": non_gguf,
                  "seconds": round(time.perf_counter() - started, 3), "ok": True}
        print("RESULT " + json.dumps(result)); pythonIO.result = json.dumps(result)
        return
    result = {"backend": cp.get_backend_name(), "correctness_gate": True,
              "routing_authority": True,
              "status": "phase_two_production_routing",
              "shape": {"K": K, "N": N, "M": list(M_VALUES)},
              "rounds": ROUNDS, "repeats_per_round": REPEATS,
              "ggml": ggml_results(), "non_gguf": gptq_results()}
    result["seconds"] = round(time.perf_counter() - started, 3)
    result["ok"] = True
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
