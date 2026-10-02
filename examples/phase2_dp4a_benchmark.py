"""Phase two: exact stored GPTQ versus activation-int8 DP4A, including pack cost."""
import json, statistics, time
import numpy as np
from js import pythonIO
from webtorch import core as wt

K, N, GS = 4096, 3072, 128
M_VALUES = (1, 2, 32, 128)
ROUNDS, REPEATS = 7, 4


def timed(fn):
    out = None; t0 = time.perf_counter()
    for _ in range(REPEATS): out = fn()
    out.get()
    return (time.perf_counter() - t0) * 1000 / REPEATS


def main():
    if not (wt._adam_backend_ready() or wt._webgl_ready()):
        raise RuntimeError("WebGPU/WebGL platform is unavailable")
    if wt._webgl_ready() and not wt._adam_backend_ready():
        # GLSL ES 3.00 fragment shaders have no packed integer dot instruction or compute
        # storage pass for activation packing.  The corresponding WebGL implementation is
        # therefore the already-verified exact packed GPTQ path, selected explicitly.
        result = {"phase": 2, "backend": "webgl",
                  "candidate": "activation_int8_dp4a", "candidate_available": False,
                  "selected": "stored",
                  "reason": "WebGL has no packed INT8 dot/compute packing primitive",
                  "ok": True}
        print("RESULT "+json.dumps(result)); pythonIO.result=json.dumps(result)
        return
    rng = np.random.default_rng(926)
    result = {"phase": 2, "backend": "webgpu",
              "candidate": "activation_int8_dp4a", "candidate_available": True,
              "formats": []}
    for bits in (4, 8):
        W = rng.standard_normal((K, N), dtype=np.float32) * np.float32(0.02)
        qw, qz, sc, _, _ = wt._gptq_quantize(W, GS, bits); del W
        qw=wt.xp.asarray(qw); qz=wt.xp.asarray(qz); sc=wt.xp.asarray(sc)
        row={"format":"GPTQ_INT%d"%bits,"shapes":[]}
        for m in M_VALUES:
            x=wt.xp.asarray(rng.standard_normal((m,K),dtype=np.float32))
            exact=lambda x=x: wt._gptq_matmul(x,qw,qz,sc,K,N,GS,bits)
            dp4a=lambda x=x: wt._gptq_dp4a_matmul(x,qw,qz,sc,K,N,GS,bits)
            a=np.asarray(exact().get()); b=np.asarray(dp4a().get())
            rel=float(np.abs(a-b).max())/max(1e-6,float(np.abs(a).max()))
            if not np.all(np.isfinite(b)) or rel>=0.03:
                raise RuntimeError("DP4A accuracy gate failed for int%d M%d: %g"%(bits,m,rel))
            samples={"exact":[],"dp4a":[]}
            for r in range(ROUNDS):
                order=(("exact",exact),("dp4a",dp4a))
                if r&1: order=tuple(reversed(order))
                for label,fn in order: samples[label].append(timed(fn))
            em=statistics.median(samples["exact"]); dm=statistics.median(samples["dp4a"])
            speed=em/dm
            row["shapes"].append({"M":m,"exact_ms":round(em,4),"dp4a_ms":round(dm,4),
                "dp4a_speedup":round(speed,3),"relative_error":rel,
                "selected":"dp4a" if speed>1.05 else "stored"})
        result["formats"].append(row)
        print("PHASE2 %s %s"%(row["format"],", ".join("M%d:%s %.2fx"%(s["M"],s["selected"],s["dp4a_speedup"]) for s in row["shapes"])))
    result["ok"]=True
    print("RESULT "+json.dumps(result)); pythonIO.result=json.dumps(result)


main()
