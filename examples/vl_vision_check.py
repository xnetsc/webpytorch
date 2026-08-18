"""Correctness check for the webtorch Qwen2.5-VL vision tower (vl.VLVisionTower).

Runs an inline numpy reference (the exact HF algorithm, validated offline to
~1e-4 vs transformers) and the webtorch fp32 port on the SAME tiny random
weights, and compares. Isolates the port's op-level correctness from int4
quantization and model downloads. Prints VISION_CHECK json.
"""
import json
import numpy as np
from js import pythonIO
from webtorch import core as wt
from webtorch import vl

CFG = dict(hidden_size=64, depth=3, num_heads=4, spatial_merge_size=2,
           patch_size=14, temporal_patch_size=2, window_size=112,
           fullatt_block_indexes=[1], out_hidden_size=128, intermediate=48, in_channels=3)
GRID = np.array([[1, 8, 12]], np.int64)          # 96 patches -> 24 merged; window ws=4
FULL = set(CFG["fullatt_block_indexes"])
H, NH = CFG["hidden_size"], CFG["num_heads"]; HD = H // NH; MU = 4


def make_weights(seed=0):
    r = np.random.RandomState(seed); W = {}
    def g(shape, s=0.1): return (r.randn(*shape) * s).astype(np.float32)
    W["visual.patch_embed.proj.weight"] = g((H, 3, 2, 14, 14))
    for i in range(CFG["depth"]):
        p = "visual.blocks.%d." % i
        W[p + "norm1.weight"] = (r.rand(H) + 0.5).astype(np.float32)
        W[p + "norm2.weight"] = (r.rand(H) + 0.5).astype(np.float32)
        W[p + "attn.qkv.weight"] = g((3 * H, H)); W[p + "attn.qkv.bias"] = g((3 * H,))
        W[p + "attn.proj.weight"] = g((H, H)); W[p + "attn.proj.bias"] = g((H,))
        I = CFG["intermediate"]
        W[p + "mlp.gate_proj.weight"] = g((I, H)); W[p + "mlp.gate_proj.bias"] = g((I,))
        W[p + "mlp.up_proj.weight"] = g((I, H)); W[p + "mlp.up_proj.bias"] = g((I,))
        W[p + "mlp.down_proj.weight"] = g((H, I)); W[p + "mlp.down_proj.bias"] = g((H,))
    W["visual.merger.ln_q.weight"] = (r.rand(H) + 0.5).astype(np.float32)
    W["visual.merger.mlp.0.weight"] = g((H * MU, H * MU)); W["visual.merger.mlp.0.bias"] = g((H * MU,))
    W["visual.merger.mlp.2.weight"] = g((CFG["out_hidden_size"], H * MU))
    W["visual.merger.mlp.2.bias"] = g((CFG["out_hidden_size"],))
    return W


def np_reference(W, pv, grid):
    """Inline numpy port of the validated Qwen2.5-VL vision forward."""
    tw = vl.VLVisionTower(CFG)                    # reuse the index/rope helpers
    def rms(x, w, eps=1e-6):
        v = (x * x).mean(-1, keepdims=True); return (x / np.sqrt(v + eps)) * w
    def silu(x): return x / (1.0 + np.exp(-x))
    def gelu(x):  # tanh approx (matches webtorch.gelu)
        c = 0.7978845608028654; return x * (np.tanh((x + 0.044715 * x**3) * c) + 1.0) * 0.5
    rpe = tw._rot_pos_emb(grid); widx, cuw = tw._window_index(grid)
    cu = np.concatenate([[0], np.cumsum(np.repeat(grid[:, 1] * grid[:, 2], grid[:, 0].tolist()))]).astype(np.int64)
    seq = pv.shape[0]
    h = pv @ W["visual.patch_embed.proj.weight"].reshape(H, -1).T
    h = h.reshape(seq // MU, MU, -1)[widx].reshape(seq, -1)
    rp = rpe.reshape(seq // MU, MU, -1)[widx].reshape(seq, -1)
    emb = np.concatenate([rp, rp], -1); cos = np.cos(emb); sin = np.sin(emb)
    def rot(x):
        return np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)
    def segmask(cc):
        seg = np.zeros(seq, np.int32)
        for i in range(len(cc) - 1): seg[cc[i]:cc[i + 1]] = i
        return np.where(seg[:, None] == seg[None, :], 0.0, -1e9).astype(np.float32)
    mf, mw = segmask(cu), segmask(cuw)
    for i in range(CFG["depth"]):
        p = "visual.blocks.%d." % i; x = rms(h, W[p + "norm1.weight"])
        qkv = x @ W[p + "attn.qkv.weight"].T + W[p + "attn.qkv.bias"]
        q = qkv[:, :H].reshape(seq, NH, HD).transpose(1, 0, 2)
        k = qkv[:, H:2 * H].reshape(seq, NH, HD).transpose(1, 0, 2)
        v = qkv[:, 2 * H:].reshape(seq, NH, HD).transpose(1, 0, 2)
        q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
        sc = (q @ k.transpose(0, 2, 1)) * (HD ** -0.5) + (mf if i in FULL else mw)
        sc -= sc.max(-1, keepdims=True); a = np.exp(sc); a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(seq, H)
        h = h + (o @ W[p + "attn.proj.weight"].T + W[p + "attn.proj.bias"])
        x = rms(h, W[p + "norm2.weight"])
        gt = silu(x @ W[p + "mlp.gate_proj.weight"].T + W[p + "mlp.gate_proj.bias"])
        up = x @ W[p + "mlp.up_proj.weight"].T + W[p + "mlp.up_proj.bias"]
        h = h + ((gt * up) @ W[p + "mlp.down_proj.weight"].T + W[p + "mlp.down_proj.bias"])
    hm = rms(h, W["visual.merger.ln_q.weight"]).reshape(seq // MU, H * MU)
    hm = gelu(hm @ W["visual.merger.mlp.0.weight"].T + W["visual.merger.mlp.0.bias"])
    hm = hm @ W["visual.merger.mlp.2.weight"].T + W["visual.merger.mlp.2.bias"]
    return hm[np.argsort(widx)]


def main():
    W = make_weights()
    r = np.random.RandomState(1)
    pv = (r.randn(GRID[0, 0] * GRID[0, 1] * GRID[0, 2], 3 * 2 * 14 * 14) * 0.1).astype(np.float32)
    ref = np_reference(W, pv, GRID)
    tower = vl.VLVisionTower.from_weights(CFG, W, bits=None)     # fp32
    out = tower.forward(pv, GRID).numpy()
    err = float(np.abs(out - ref).max()); rel = err / (np.abs(ref).max() + 1e-9)
    res = {"ref_shape": list(ref.shape), "out_shape": list(out.shape),
           "max_abs_err": err, "rel_err": rel, "ok": bool(rel < 1e-3)}
    print("VISION_CHECK " + json.dumps(res))
    pythonIO.result = json.dumps(res)


main()
