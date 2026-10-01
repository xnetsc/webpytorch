"""Provisional diagnostic for encoded weights against explicit alternatives.

This is deliberately not allowed to drive runtime routing yet.  Matching the independent
decoder proves correctness, but it does not prove that the original-width kernel has received
all applicable hardware optimisations.  The comparison becomes routing evidence only after
that native-path completion gate is satisfied for the format under test.
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
REPEATS = 4
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


def main():
    if not wt._adam_backend_ready():
        raise RuntimeError("WebGPU compute platform is unavailable: %s" % wt.backend_reason())
    started = time.perf_counter()
    print("CORRECTNESS_GATE starting")
    correctness_gate()
    print("CORRECTNESS_GATE passed")
    result = {"backend": cp.get_backend_name(), "correctness_gate": True,
              "routing_authority": False,
              "status": "provisional_until_native_width_paths_are_optimized",
              "shape": {"K": K, "N": N, "M": list(M_VALUES)},
              "rounds": ROUNDS, "repeats_per_round": REPEATS,
              "ggml": ggml_results(), "non_gguf": gptq_results()}
    result["seconds"] = round(time.perf_counter() - started, 3)
    result["ok"] = True
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
