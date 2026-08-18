"""Regression check: unmodified nanoGPT model.py still runs (exec/build/forward/
train/generate) after the torchshim rewrite. Real third-party torch file, fetched
raw from GitHub, not edited."""
import json, traceback
import numpy as np
from js import pythonIO
from pyodide.http import pyfetch
import cupy as cp
from webtorch import torchshim
torchshim.install()

URL = "https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py"


async def main():
    r = {"backend": cp.get_backend_name()}
    import torch
    try:
        src = await (await pyfetch(URL)).string()
        ns = {"__name__": "model"}
        exec(compile(src, "model.py", "exec"), ns)          # verbatim
        cfg = ns["GPTConfig"](block_size=16, vocab_size=32, n_layer=2, n_head=2,
                              n_embd=16, dropout=0.0, bias=False)
        model = ns["GPT"](cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        np.random.seed(0)
        X = torch.tensor(np.random.randint(0, 32, (4, 8)))
        Y = torch.tensor(np.random.randint(0, 32, (4, 8)))
        curve = []
        for step in range(80):
            logits, loss = model(X, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 20 == 0 or step == 79:
                curve.append(round(float(loss.item()), 3))
        gen = model.generate(torch.zeros(1, 4), max_new_tokens=5)
        r.update({"ran_unmodified": True, "n_params": model.get_num_params(),
                  "loss_curve": curve, "generate_shape": list(gen.shape),
                  "weight_tying_held": bool(model.transformer.wte.weight is model.lm_head.weight),
                  "trained": bool(curve[-1] < curve[0] * 0.5)})
        r["ok"] = r["trained"] and r["weight_tying_held"]
    except Exception as e:
        r["err"] = "%s: %s" % (type(e).__name__, e)
        r["tb"] = traceback.format_exc()[-500:]
        r["ok"] = False
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


await main()
