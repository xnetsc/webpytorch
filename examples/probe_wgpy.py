"""Probe WgPy cupy shim for ops the vocoder GPU port needs: sin, strided
slice-assign, matmul. Prints PROBE_RESULT."""
import json
import numpy as np
from js import pythonIO
from webtorch import core as wt
xp = wt.xp
res = {"GPU": bool(getattr(wt, "GPU", False))}

def probe(name, fn):
    try:
        res[name] = fn()
    except Exception as e:
        res[name] = "ERR: " + str(e)[:120]

a = xp.asarray(np.linspace(0, 3, 8).astype(np.float32))
probe("sin", lambda: float(xp.asnumpy(xp.sin(a)).sum()))
probe("cos", lambda: float(xp.asnumpy(xp.cos(a)).sum()))
def strided_assign():
    z = xp.asarray(np.zeros((2, 12), np.float32))
    y = xp.asarray(np.ones((2, 4), np.float32))
    z[:, 0:12:3] = y            # strided slice-assign (convT scatter)
    return float(xp.asnumpy(z).sum())
probe("strided_assign", strided_assign)
def slice_add():
    z = xp.asarray(np.zeros((2, 12), np.float32))
    y = xp.asarray(np.ones((2, 4), np.float32))
    z[:, 0:12:3] += y
    return float(xp.asnumpy(z).sum())
probe("strided_iadd", slice_add)
probe("matmul2d", lambda: float(xp.asnumpy(xp.asarray(np.ones((4,5),np.float32)) @ xp.asarray(np.ones((5,3),np.float32))).sum()))
probe("power2", lambda: float(xp.asnumpy(a * a).sum()))
probe("clip", lambda: float(xp.asnumpy(xp.clip(a, 0.0, 1.0)).sum()) if hasattr(xp,'clip') else "no clip")
probe("where", lambda: float(xp.asnumpy(xp.where(a > 1.0, a, a * 0.1)).sum()) if hasattr(xp,'where') else "no where")

def wt_sin():
    t = wt.Tensor(np.linspace(0, 3, 8).astype(np.float32))
    got = wt.sin(t).numpy(); exp = np.sin(np.linspace(0, 3, 8).astype(np.float32))
    return float(np.abs(got - exp).max())
probe("wt_sin_err", wt_sin)

print("PROBE_RESULT " + json.dumps(res))
pythonIO.result = json.dumps(res)
