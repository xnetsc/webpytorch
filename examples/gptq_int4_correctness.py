"""Correctness of the rewritten tiled int4/int8 GPTQ dequant-matmul kernel.
Compare against a numpy dequant reference across shapes, M=1 and M>1,
and against the original QuantizedLinear.from_linear path."""
import json
import numpy as np
from js import pythonIO
import cupy as cp
from webtorch import core as wt

backend = cp.get_backend_name()


def np_dequant(qweight, qzeros, scales, K, N, gs, bits):
    """Unpack GPTQ -> dense fp32 (K,N), independent of the GPU kernel."""
    per = 32 // bits
    mask = (1 << bits) - 1
    W = np.zeros((K, N), np.float32)
    for k in range(K):
        q = (qweight[k // per] >> ((k % per) * bits)) & mask
        g = k // gs
        z = (qzeros[g, np.arange(N) // per] >> ((np.arange(N) % per) * bits)) & mask
        W[k] = scales[g] * (q.astype(np.float32) - z.astype(np.float32))
    return W


def main():
    r = {"backend": backend, "cases": []}
    rng = np.random.default_rng(0)
    worst = 0.0
    for (K, N, gs, bits) in [(128, 64, 32, 4), (256, 128, 128, 4), (3584, 512, 128, 4),
                             (512, 1536, 128, 4), (128, 24, 32, 4), (256, 64, 32, 8)]:
        W = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        qw, qz, sc, _, _ = wt._gptq_quantize(W, gs, bits)
        b = np.zeros((N,), np.float32)
        ql = wt.QuantizedLinear(qw, qz, sc, b, K, N, K, N, gs, bits)
        Wd = np_dequant(qw, qz, sc, K, N, gs, bits)     # what the kernel SHOULD compute with
        for M in (1, 3, 4, 5, 32):
            x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
            got = ql(wt.Tensor(x)).numpy()
            ref = x @ Wd
            err = float(np.abs(got - ref).max()) / (float(np.abs(ref).max()) + 1e-9)
            worst = max(worst, err)
            if err > 1e-4:
                r["cases"].append({"K": K, "N": N, "gs": gs, "bits": bits, "M": M,
                                   "rel_err": round(err, 6), "FAIL": True})
    r["worst_rel_err_vs_numpy_dequant"] = worst
    r["n_failures"] = len(r["cases"])

    # end-to-end: a real quantized Linear still matches its fp32 source closely
    lin = wt.Linear(256, 128)
    x = wt.Tensor((rng.standard_normal((8, 256)) * 0.5).astype(np.float32))
    y_fp32 = lin(x).numpy()
    qlin = wt.QuantizedLinear.from_linear(lin, group_size=128, bits=4)
    y_q = qlin(x).numpy()
    r["quantized_vs_fp32_rel_err"] = round(float(np.abs(y_q - y_fp32).max()) /
                                           float(np.abs(y_fp32).max()), 4)
    r["ok"] = bool(worst < 1e-4 and r["quantized_vs_fp32_rel_err"] < 0.15)
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


main()
