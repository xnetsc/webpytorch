"""END-TO-END GGUF: load a REAL trained llama model straight from a llama.cpp
GGUF (stories15M-q8_0, 26MB) into webtorch, run GPU inference, and generate text.

Two checks:
  1. webtorch GPU forward == independent numpy reference (proves the GPU path).
  2. the generated text is coherent English (proves the GGUF weight layout +
     interleaved RoPE are right -- a wrong layout yields gibberish).

Note: GGUF llama uses ggml NORM rope = INTERLEAVED pairs (x[2i], x[2i+1]);
HF llama uses rotate_half with permuted q/k. Must use the interleaved form here.
"""
import json, math
import numpy as np
from js import pythonIO
from pyodide.http import pyfetch
import cupy as cp
from webtorch import core as wt
import ggufload as G

backend = cp.get_backend_name()
URL = "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf"


def T_(t):                       # transposed VIEW (no copy); webtorch matmul reads strides
    return wt.Tensor(wt._swap_last2(t.data))


def rope_tab(T, hd, base, offset=0):
    inv = 1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float64) / hd))   # (hd/2,)
    ang = (np.arange(offset, offset + T, dtype=np.float64)[:, None]) * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)   # (T, hd/2)


# ----------------------------- numpy reference ------------------------------
def llama_numpy(C, W, ids):
    H, nh, nkv, L, eps = C["H"], C["nh"], C["nkv"], C["L"], C["eps"]
    hd = H // nh; rep = nh // nkv; T = len(ids)
    cos, sin = rope_tab(T, hd, C["base"])

    def rms(x, w):
        return (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)) * w

    def rope(x):                                   # x (heads,T,hd) interleaved pairs
        x = x.reshape(x.shape[0], T, hd // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        return np.stack([o0, o1], -1).reshape(-1, T, hd)

    mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
    h = W["token_embd.weight"][ids]
    for i in range(L):
        p = "blk.%d." % i
        x = rms(h, W[p + "attn_norm.weight"])
        q = (x @ W[p + "attn_q.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)
        k = (x @ W[p + "attn_k.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        v = (x @ W[p + "attn_v.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        q, k = rope(q), rope(k)
        if rep > 1:
            k = np.repeat(k, rep, 0); v = np.repeat(v, rep, 0)
        a = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(hd) + mask
        a = a - a.max(-1, keepdims=True); a = np.exp(a); a /= a.sum(-1, keepdims=True)
        o = np.matmul(a, v).transpose(1, 0, 2).reshape(T, H)
        h = h + o @ W[p + "attn_output.weight"].T
        x = rms(h, W[p + "ffn_norm.weight"])
        g = x @ W[p + "ffn_gate.weight"].T
        u = x @ W[p + "ffn_up.weight"].T
        h = h + (g * (1 / (1 + np.exp(-g))) * u) @ W[p + "ffn_down.weight"].T
    h = rms(h, W["output_norm.weight"])
    return h @ W["output.weight"].T


# ----------------------------- webtorch GPU ---------------------------------
def llama_webtorch(C, W, Wt, ids):
    H, nh, nkv, L, eps = C["H"], C["nh"], C["nkv"], C["L"], C["eps"]
    hd = H // nh; rep = nh // nkv; T = len(ids)
    cos, sin = rope_tab(T, hd, C["base"])
    cos_t = wt.Tensor(cos.reshape(1, T, hd // 2, 1))
    sin_t = wt.Tensor(sin.reshape(1, T, hd // 2, 1))
    mask_t = wt.Tensor(np.triu(np.full((T, T), -1e9, np.float32), 1))
    xp = wt.xp
    rep_idx = xp.asarray(np.repeat(np.arange(nkv), rep).astype(np.int32)) if rep > 1 else None

    def rms(x, w):
        return (x / ((x * x).mean(axis=-1, keepdims=True) + eps).sqrt()) * w

    def rope(x, heads):                            # (heads,T,hd) interleaved
        x = x.reshape(heads, T, hd // 2, 2)
        x0 = wt._slice_last(x, 0, 1)               # (heads,T,hd/2,1)
        x1 = wt._slice_last(x, 1, 2)
        o0 = x0 * cos_t - x1 * sin_t
        o1 = x0 * sin_t + x1 * cos_t
        return wt.cat([o0, o1], axis=-1).reshape(heads, T, hd)

    h = wt.Tensor(W["token_embd.weight"][ids])
    for i in range(L):
        p = "blk.%d." % i
        x = rms(h, Wt[p + "attn_norm.weight"])
        q = x.matmul(T_(Wt[p + "attn_q.weight"])).reshape(T, nh, hd).permute(1, 0, 2)
        k = x.matmul(T_(Wt[p + "attn_k.weight"])).reshape(T, nkv, hd).permute(1, 0, 2)
        v = x.matmul(T_(Wt[p + "attn_v.weight"])).reshape(T, nkv, hd).permute(1, 0, 2)
        q = rope(q, nh); k = rope(k, nkv)
        if rep > 1:
            k = wt.Tensor(k.data[rep_idx]); v = wt.Tensor(v.data[rep_idx])
        a = wt.bmm(q, k.permute(0, 2, 1)) * (1.0 / math.sqrt(hd)) + mask_t
        a = wt.softmax(a)
        o = wt.bmm(a, v).permute(1, 0, 2).reshape(T, H)
        h = h + o.matmul(T_(Wt[p + "attn_output.weight"]))
        x = rms(h, Wt[p + "ffn_norm.weight"])
        g = x.matmul(T_(Wt[p + "ffn_gate.weight"]))
        u = x.matmul(T_(Wt[p + "ffn_up.weight"]))
        h = h + (wt.silu(g) * u).matmul(T_(Wt[p + "ffn_down.weight"]))
    h = rms(h, Wt["output_norm.weight"])
    return h.matmul(T_(Wt["output.weight"]))


def detok(tokens, ids):
    out = []
    for i in ids:
        p = tokens[i]
        if p.startswith("<0x") and p.endswith(">"):
            out.append(chr(int(p[3:-1], 16)))
        elif p in ("<s>", "</s>", "<unk>"):
            continue
        else:
            out.append(p.replace("▁", " "))
    return "".join(out)


async def main():
    r = {"backend": backend}
    resp = await pyfetch(URL)
    buf = bytes(await resp.bytes())
    ver, meta, infos, ds = G.parse_header(buf)

    C = {"H": meta["llama.embedding_length"], "L": meta["llama.block_count"],
         "nh": meta["llama.attention.head_count"],
         "nkv": meta.get("llama.attention.head_count_kv", meta["llama.attention.head_count"]),
         "eps": meta.get("llama.attention.layer_norm_rms_epsilon", 1e-5),
         "base": meta.get("llama.rope.freq_base", 10000.0)}
    r["arch"] = meta["general.architecture"]; r["config"] = C

    W = {}
    for t in infos:
        n = int(np.prod(t["dims"]))
        nb = G.tensor_nbytes(t["type"], n)
        off = ds + t["offset"]
        W[t["name"]] = G.dequant(t["type"], buf[off:off + nb], n).reshape(tuple(reversed(t["dims"])))
    r["n_tensors"] = len(W)
    r["quant_types"] = sorted({G.GGML_NAMES.get(t["type"]) for t in infos})

    Wt = {k: wt.Tensor(v.copy()) for k, v in W.items() if k != "token_embd.weight"}

    bos = int(meta.get("tokenizer.ggml.bos_token_id", 1))
    ids = [bos]
    # correctness: GPU vs numpy on a short prompt
    probe = [bos, 1128, 590, 263]
    ref = llama_numpy(C, W, probe)
    got = llama_webtorch(C, W, Wt, probe).numpy()
    r["max_abs_err"] = float(np.abs(ref - got).max())
    r["argmax_match"] = bool((ref.argmax(-1) == got.argmax(-1)).all())

    # greedy generation on the GPU path -> real English if the layout is right
    for _ in range(60):
        lg = llama_webtorch(C, W, Wt, ids).numpy()
        ids.append(int(lg[-1].argmax()))
        if ids[-1] == int(meta.get("tokenizer.ggml.eos_token_id", 2)):
            break
    toks = meta["tokenizer.ggml.tokens"]
    r["generated_ids"] = ids[:12]
    r["text"] = detok(toks, ids)
    r["ok"] = bool(r["max_abs_err"] < 1e-3 and r["argmax_match"] and len(r["text"]) > 20)
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


await main()
