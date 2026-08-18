"""MoE series in the browser — load a Qwen2-MoE (int4) via the generic loader and
run it on WebGPU, verifying the next-token prediction matches the transformers
reference recorded offline. Proves the generic MoE path (router top-k + shared
expert + int4 QuantizedLinear experts) runs end-to-end in the browser.
"""
import io, json, time
import numpy as np
from js import pythonIO
from pyodide.http import pyfetch
from webtorch import core as wt
from webtorch import lm_engine


async def main():
    r = await pyfetch("/models/moe_tiny.npz")
    Z = {k: v for k, v in np.load(io.BytesIO(bytes(await r.bytes()))).items()}
    cfg = json.loads(bytes(Z["config"]).decode())

    def get(name): return Z[name]
    def tensor(a): return wt.Tensor(np.asarray(a, np.float32))
    def linear(wname, bias=True):
        base = wname[:-7]
        K, N, Kp, Np = [int(v) for v in Z[base + ".dims"]]
        b = Z[base + ".bias"] if (bias and base + ".bias" in Z) else np.zeros((N,), np.float32)
        return wt.QuantizedLinear(Z[base + ".qweight"], Z[base + ".qzeros"], Z[base + ".scales"],
                                  b, K, N, Kp, Np, cfg["gs"], cfg["bits"])

    t0 = time.perf_counter()
    lm = lm_engine.build_lm(cfg, get, linear, tensor)          # generic dense/MoE loader
    head = linear("lm_head.weight", bias=False)
    lm.head = lambda h: head(h)
    emb = Z["model.embed_tokens.weight"].astype(np.float32)

    ids = [int(x) for x in Z["ref_ids"]]
    last, _ = lm.prefill(wt.Tensor(emb[np.array(ids)]))
    logits = lm.head(last).numpy().reshape(-1)
    argmax = int(logits.argmax())
    ref_i4 = int(Z["ref_argmax_int4"][0])       # offline int4 (what the browser int4 kernel must match)
    ref_fp = int(Z["ref_argmax"][0])            # fp32 transformers reference (fp32 path matched to 9.7e-8)
    logit_err = float(np.abs(logits - Z["ref_logits_int4"]).max())
    dt = time.perf_counter() - t0

    res = {"arch": "Qwen2-MoE", "layers": cfg["L"], "experts": cfg["num_experts"],
           "top_k": cfg["num_experts_per_tok"], "shared_expert": cfg["shared_expert"], "bits": cfg["bits"],
           "browser_argmax": argmax, "offline_int4_argmax": ref_i4, "transformers_fp32_argmax": ref_fp,
           "logit_max_err_vs_offline_int4": logit_err, "match_int4": bool(argmax == ref_i4),
           "seconds": round(dt, 2)}
    print("MOE_RESULT " + json.dumps(res))
    pythonIO.result = json.dumps(res)


await main()
