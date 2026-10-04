"""Phase two: exact stored GPTQ versus activation-int8 DP4A, including pack cost."""
import json, statistics, time
import numpy as np
from js import pythonIO
from webtorch import core as wt

K, N, GS = 4096, 3072, 128
M_VALUES = (1, 2, 32, 128)
ROUNDS, REPEATS = 9, 4


def timed(fn):
    out = None; t0 = time.perf_counter()
    for _ in range(REPEATS): out = fn()
    out.get()
    return (time.perf_counter() - t0) * 1000 / REPEATS


def autogptq_zeros(qzeros, bits):
    """Encode generated zero points with AutoGPTQ's stored (zero - 1) convention."""
    src = np.asarray(qzeros, np.uint32)
    out = np.zeros_like(src)
    mask = (1 << bits) - 1
    for j in range(32 // bits):
        v = (src >> np.uint32(j * bits)) & np.uint32(mask)
        out |= ((v - np.uint32(1)) & np.uint32(mask)) << np.uint32(j * bits)
    return out.view(np.int32)


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
        # AutoGPTQ stores zero-1.  Validate that convention separately before timing so
        # the production DP4A candidate is not proven only for locally generated tensors.
        qz_auto=wt.xp.asarray(autogptq_zeros(np.asarray(qz.get()),bits))
        zx=wt.xp.asarray(rng.standard_normal((2,K),dtype=np.float32))
        za=np.asarray(wt._gptq_matmul(zx,qw,qz_auto,sc,K,N,GS,bits,zoff=1.0).get())
        zb=np.asarray(wt._gptq_dp4a_matmul(zx,qw,qz_auto,sc,K,N,GS,bits,zoff=1.0).get())
        zrel=float(np.abs(za-zb).max())/max(1e-6,float(np.abs(za).max()))
        if not np.all(np.isfinite(zb)) or zrel>=0.03:
            raise RuntimeError("DP4A AutoGPTQ zero-offset gate failed for int%d: %g"%(bits,zrel))
        row={"format":"GPTQ_INT%d"%bits,"autogptq_zero_offset_relative_error":zrel,
             "shapes":[]}
        for m in M_VALUES:
            x=wt.xp.asarray(rng.standard_normal((m,K),dtype=np.float32))
            exact=lambda x=x: wt._gptq_matmul(x,qw,qz,sc,K,N,GS,bits)
            dp4a=lambda x=x: wt._gptq_dp4a_matmul(x,qw,qz,sc,K,N,GS,bits)
            a=np.asarray(exact().get()); b=np.asarray(dp4a().get())
            rel=float(np.abs(a-b).max())/max(1e-6,float(np.abs(a).max()))
            if not np.all(np.isfinite(b)) or rel>=0.03:
                raise RuntimeError("DP4A accuracy gate failed for int%d M%d: %g"%(bits,m,rel))
            samples={"stored":[],"dp4a":[]}
            for r in range(ROUNDS):
                order=(("stored",exact),("dp4a",dp4a))
                if r&1: order=tuple(reversed(order))
                for label,fn in order: samples[label].append(timed(fn))
            em=statistics.median(samples["stored"]); dm=statistics.median(samples["dp4a"])
            speed=em/dm
            selected=wt._measured_choice(samples,("stored","dp4a"),default="stored")
            evidence=wt._paired_evidence(samples,"dp4a","stored")
            row["shapes"].append({"M":m,"exact_ms":round(em,4),"dp4a_ms":round(dm,4),
                "dp4a_speedup":round(speed,3),"relative_error":rel,
                "selected":selected,
                "stable_positive":bool(selected == "dp4a"),
                "paired_wins":evidence["wins"],"paired_rounds":evidence["pairs"],
                "one_sided_p":round(evidence["p"],6)})
        result["formats"].append(row)
        print("PHASE2 %s %s"%(row["format"],", ".join("M%d:%s %.2fx"%(s["M"],s["selected"],s["dp4a_speedup"]) for s in row["shapes"])))
    result["ok"]=True
    print("RESULT "+json.dumps(result)); pythonIO.result=json.dumps(result)


main()
