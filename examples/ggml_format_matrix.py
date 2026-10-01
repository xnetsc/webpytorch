"""Browser correctness gate for every accepted GGML storage format.

This deliberately runs only the stored-block kernels.  Performance comparisons are a
separate phase and must not begin unless every case here passes.
"""
import json
import time

import cupy as cp
import numpy as np
from js import pythonIO

from webtorch import core as wt


def main():
    if not wt._adam_backend_ready():
        raise RuntimeError("WebGPU compute platform is unavailable: %s" % wt.backend_reason())
    result = {
        "backend": cp.get_backend_name(),
        "execution": "stored",
        "formats": [],
        "failures": [],
    }
    cases = (
        ("gemv", 1, False),
        ("gemv2", 2, False),
        ("gemm", 0, False),
        ("moe-gemv", 1, True),
        ("moe-gemm", 0, True),
    )
    started = time.perf_counter()
    for type_name in sorted(wt._GGML_TYPES):
        row = {"format": type_name, "cases": [], "ok": True}
        for label, mode, moe in cases:
            t0 = time.perf_counter()
            try:
                wt._ggml_selfcheck(type_name, mode, moe=moe)
                case = {"case": label, "ok": True,
                        "seconds": round(time.perf_counter() - t0, 3)}
            except Exception as exc:
                case = {"case": label, "ok": False,
                        "seconds": round(time.perf_counter() - t0, 3),
                        "error": "%s: %s" % (type(exc).__name__, exc)}
                row["ok"] = False
                result["failures"].append({"format": type_name, **case})
            row["cases"].append(case)
        result["formats"].append(row)
        print("GGML_MATRIX %s %s" % (type_name, "PASS" if row["ok"] else "FAIL"))

    # A pipeline can be accepted and still invalidate the command buffer only when it is
    # dispatched.  This small end-to-end check caught exactly that in the decision feature
    # reduction: all 140 weight cases passed while real model logits became uniform zeros.
    logits = np.asarray([[5.5190744, -1.7310995, -2.535373]], np.float32)
    shifted = logits.reshape(-1) - logits.max()
    probs = np.exp(shifted); probs /= probs.sum()
    ordered = np.sort(probs)[::-1]
    expected = np.asarray([
        ordered[0], ordered[0] - ordered[1],
        -(probs * np.log(np.maximum(probs, 1e-9))).sum() / np.log(3.0),
        3.0 / 255.0,
    ], np.float32)
    got = np.asarray(wt.decision_features(wt.Tensor(logits), 3).numpy()).reshape(-1)
    feature_ok = bool(np.all(np.isfinite(got))
                      and np.allclose(got, expected, rtol=2e-5, atol=2e-6))
    result["decision_features"] = {
        "ok": feature_ok,
        "expected": expected.tolist(),
        "actual": got.tolist(),
    }
    if not feature_ok:
        result["failures"].append({"format": "decision_features",
                                    "case": "gpu-reduction", "ok": False})
    print("DECISION_FEATURES %s" % ("PASS" if feature_ok else "FAIL"))

    result["format_count"] = len(result["formats"])
    result["case_count"] = sum(len(row["cases"]) for row in result["formats"])
    result["seconds"] = round(time.perf_counter() - started, 3)
    result["ok"] = not result["failures"]
    print("RESULT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
