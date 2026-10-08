"""Decode speed with a LoRA adapter on every projection, per adapter route.

Loads the 0.6B GGUF once, then attaches adapters made here in memory -- rank 16 on q/k/v/o/
gate/up/down of every layer, nothing read from or written to disk: a zero one (B = 0, so the
text must stay exactly what the model writes without it) and a random one. Each runs with the
adapter route forced in turn ("composed": two matmuls and an add; "fused:<n>": one dispatch,
n outputs per workgroup) and then raced ("auto", what a load picks on this device).

Reported per setting: median decode tok/s of three greedy replies, whether the text equals
the reference (no adapter for the zero one; the composed route for the random one), and the
dispatches of one decode step by kernel.
"""
import json

import numpy as np
from js import pythonIO

from webtorch import _core as wt
from webtorch import llm, use_default_io


GGUF = "/models/Qwen3-0.6B-Q4_K_M.gguf"
PROMPT = "Give me three tips for staying focused while working. Answer briefly."
NEW = 64
RANK = 16


def adapter(model, zero, seed=0):
    rng = np.random.default_rng(seed)
    mods = {}
    for i, lay in enumerate(model.layers):
        for path, slot in llm.CausalLM._ADAPTER_SLOTS.items():
            lin = lay.get(slot)
            if lin is None or not hasattr(lin, "Kt"):
                continue
            K, N = int(lin.Kt), int(lin.Nt)
            A = (rng.standard_normal((RANK, K)) / np.sqrt(K)).astype(np.float32)
            B = (np.zeros((N, RANK), np.float32) if zero
                 else (rng.standard_normal((N, RANK)) * 0.02).astype(np.float32))
            mods["model.layers.%d.%s" % (i, path)] = (A, B, 2.0)
    return mods


def force(route, mods):
    """The adapter route every decode row takes from the next attach on; "auto" races."""
    for k in [k for k in wt._TUNED if k[:2] == ("weight_exec", "lora")]:
        del wt._TUNED[k]
    if route == "auto":
        return
    for A, B, _ in mods.values():
        r4 = -(-A.shape[0] // 4) * 4
        wt._TUNED[("weight_exec", "lora", "r%d" % r4, A.shape[1], B.shape[0], 1)] = route


def step_dispatches(model):
    """One decode step's dispatches by kernel, outside any recording."""
    wt._count_dispatch_names(True)
    try:
        before = model._dispatch_names()
        model._set_inputs(0, 0)
        model._decode_fwd().numpy()
        after = model._dispatch_names()
    finally:
        wt._count_dispatch_names(False)
        model._reset_linear_state()
    got = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
    return sum(got.values()), dict(sorted(got.items(), key=lambda kv: -kv[1]))


def replies(model):
    model.generate(PROMPT, max_new=8, do_sample=False, enable_thinking=False)   # records
    outs = [model.generate(PROMPT, max_new=NEW, do_sample=False, enable_thinking=False)
            for _ in range(3)]
    return outs[0].text, sorted(round(o.decode_tok_s, 1) for o in outs)[1]


async def main():
    use_default_io(cache=False)
    model = await llm.CausalLM.from_gguf(GGUF, weights="native")
    text0, tps = replies(model)
    n, by = step_dispatches(model)
    result = {"gguf": GGUF, "rank": RANK,
              "none": {"tok_s": tps, "step_dispatches": n, "by_kernel": by}}
    print("none", tps, "tok/s", n, "dispatches")
    for name, zero in (("zero", True), ("random", False)):
        mods = adapter(model, zero)
        reference = text0 if zero else None
        for route in wt._LORA_ROUTES + ("auto",):
            force(route, mods)
            model.attach_adapter(mods)          # recordings dropped, decode step warmed again
            text, tps = replies(model)
            if reference is None:
                reference = text                # the composed route comes first
            n, by = step_dispatches(model)
            chosen = sorted({v for k, v in wt._TUNED.items()
                             if k[:2] == ("weight_exec", "lora")})
            result["%s/%s" % (name, route)] = {"tok_s": tps, "same_text": text == reference,
                                               "step_dispatches": n, "routes": chosen,
                                               "by_kernel": by}
            print(name, route, tps, "tok/s", n, "dispatches", "same" if text == reference
                  else "DIFFERENT", chosen)
            model.detach_adapters()
    pythonIO.result = json.dumps(result)


await main()
