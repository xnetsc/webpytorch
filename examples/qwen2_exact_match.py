"""Load a REAL Qwen2 checkpoint from HuggingFace and run inference on webtorch's
GPU backend, proving the logits EXACTLY match an independent numpy reference
(built from the HF Qwen2 spec). Same code path loads Qwen2.5-0.5B — only sandbox
memory stops the 0.5B here; correctness is what we verify.

Model: peft-internal-testing/tiny-dummy-qwen2 (real qwen2 arch, F32, tied embed,
rope_theta=1e6, QKV bias, GQA 4q/2kv) — architecturally identical to Qwen2.5-0.5B.
"""
import json, math
import numpy as np
from js import pythonIO
from pyodide.http import pyfetch
import cupy as cp
from webtorch import core as wt

backend = cp.get_backend_name()
REPO = "peft-internal-testing/tiny-dummy-qwen2"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{fn}"


# ------------------------------ HF I/O -------------------------------------
async def fetch_bytes(fn):
    r = await pyfetch(RESOLVE.format(repo=REPO, fn=fn))
    return await r.bytes()


def parse_safetensors(buf):
    b = bytes(buf)
    n = int.from_bytes(b[:8], "little")
    hdr = json.loads(b[8:8 + n].decode("utf-8"))
    hdr.pop("__metadata__", None)
    base = 8 + n
    W = {}
    for name, info in hdr.items():
        a, z = info["data_offsets"]
        raw = b[base + a: base + z]
        dt = info["dtype"]
        if dt == "F32":
            arr = np.frombuffer(raw, np.float32)
        elif dt == "BF16":
            u16 = np.frombuffer(raw, np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, np.float16).astype(np.float32)
        else:
            raise ValueError("dtype " + dt)
        W[name] = arr.reshape(info["shape"]).astype(np.float32)
    return W


# --------------------------- RoPE tables -----------------------------------
def rope_cos_sin(T, hd, theta):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))   # (hd/2,)
    t = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]            # (T, hd/2)
    emb = np.concatenate([t, t], axis=-1)                                # (T, hd)  HF layout
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


# ------------------- independent NUMPY reference ---------------------------
def qwen2_numpy(cfg, W, ids, trace=None):
    H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, theta = H // nh, cfg["rms_norm_eps"], cfg["rope_theta"]
    rep = nh // nkv
    T = len(ids)
    cos, sin = rope_cos_sin(T, hd, theta)

    def rms(x, w):
        v = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(v + eps)) * w

    def rot(x):  # (heads,T,hd) rotate_half
        x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
        return np.concatenate([-x2, x1], axis=-1)

    mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
    h = W["model.embed_tokens.weight"][ids]                               # (T,H)
    if trace is not None: trace["embed"] = h.copy()
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        x = rms(h, W[p + "input_layernorm.weight"])
        q = x @ W[p + "self_attn.q_proj.weight"].T + W[p + "self_attn.q_proj.bias"]
        k = x @ W[p + "self_attn.k_proj.weight"].T + W[p + "self_attn.k_proj.bias"]
        v = x @ W[p + "self_attn.v_proj.weight"].T + W[p + "self_attn.v_proj.bias"]
        q = q.reshape(T, nh, hd).transpose(1, 0, 2)                       # (nh,T,hd)
        k = k.reshape(T, nkv, hd).transpose(1, 0, 2)
        v = v.reshape(T, nkv, hd).transpose(1, 0, 2)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        if trace is not None:
            trace[f"l{i}.q"] = q.copy(); trace[f"l{i}.k"] = k.copy()
        k = np.repeat(k, rep, axis=0); v = np.repeat(v, rep, axis=0)      # GQA
        att = np.matmul(q, k.transpose(0, 2, 1)) / math.sqrt(hd) + mask
        att = att - att.max(-1, keepdims=True)
        att = np.exp(att); att /= att.sum(-1, keepdims=True)
        o = np.matmul(att, v).transpose(1, 0, 2).reshape(T, H)
        if trace is not None: trace[f"l{i}.attn"] = o.copy()
        o = o @ W[p + "self_attn.o_proj.weight"].T
        h = h + o
        if trace is not None: trace[f"l{i}.post_attn"] = h.copy()
        x = rms(h, W[p + "post_attention_layernorm.weight"])
        g = x @ W[p + "mlp.gate_proj.weight"].T
        u = x @ W[p + "mlp.up_proj.weight"].T
        h = h + (g * (1 / (1 + np.exp(-g))) * u) @ W[p + "mlp.down_proj.weight"].T
        if trace is not None: trace[f"l{i}.post_mlp"] = h.copy()
    h = rms(h, W["model.norm.weight"])
    if trace is not None: trace["final_hidden"] = h.copy()
    return h @ W["model.embed_tokens.weight"].T                           # tied lm_head


# --------------------- webtorch GPU forward --------------------------------
def qwen2_webtorch(cfg, Wt, ids, cos_t, sin_t, mask_t, trace=None):
    T = len(ids)
    H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh; rep = nh // nkv; eps = cfg["rms_norm_eps"]
    xp = wt.xp
    rep_idx = xp.asarray(np.repeat(np.arange(nkv), rep).astype(np.int32))  # GQA expand index

    def rms(x, w):
        v = (x * x).mean(axis=-1, keepdims=True)
        return (x / (v + eps).sqrt()) * w

    def rot(x):
        x1 = wt._slice_last(x, 0, hd // 2)
        x2 = wt._slice_last(x, hd // 2, hd)
        return wt.cat([x2 * (-1.0), x1], axis=-1)

    def rope(x):
        return x * cos_t + rot(x) * sin_t

    h = Wt["model.embed_tokens.weight"].numpy()[ids]
    h = wt.Tensor(h)                                                       # (T,H)
    if trace is not None: trace["embed"] = h.numpy()
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        x = rms(h, Wt[p + "input_layernorm.weight"])
        q = x.matmul(Wt[p + "self_attn.q_proj.weight_T"]) + Wt[p + "self_attn.q_proj.bias"]
        k = x.matmul(Wt[p + "self_attn.k_proj.weight_T"]) + Wt[p + "self_attn.k_proj.bias"]
        v = x.matmul(Wt[p + "self_attn.v_proj.weight_T"]) + Wt[p + "self_attn.v_proj.bias"]
        q = q.reshape(T, nh, hd).permute(1, 0, 2)
        k = k.reshape(T, nkv, hd).permute(1, 0, 2)
        v = v.reshape(T, nkv, hd).permute(1, 0, 2)
        q = rope(q); k = rope(k)
        if trace is not None:
            trace[f"l{i}.q"] = q.numpy(); trace[f"l{i}.k"] = k.numpy()
        o = wt.gqa_attention(q, k, v, mask_t, scale=1.0 / math.sqrt(hd))   # no KV expansion
        if trace is not None:
            trace[f"l{i}.attn"] = o.permute(1, 0, 2).reshape(T, H).numpy()
        o = o.permute(1, 0, 2).reshape(T, H).matmul(Wt[p + "self_attn.o_proj.weight_T"])
        h = h + o
        if trace is not None: trace[f"l{i}.post_attn"] = h.numpy()
        x = rms(h, Wt[p + "post_attention_layernorm.weight"])
        g = x.matmul(Wt[p + "mlp.gate_proj.weight_T"])
        u = x.matmul(Wt[p + "mlp.up_proj.weight_T"])
        h = h + (wt.silu(g) * u).matmul(Wt[p + "mlp.down_proj.weight_T"])
        if trace is not None: trace[f"l{i}.post_mlp"] = h.numpy()
    h = rms(h, Wt["model.norm.weight"])
    if trace is not None: trace["final_hidden"] = h.numpy()
    return h.matmul(Wt["model.embed_tokens.weight_T"])


def to_webtorch(cfg, W):
    Wt = {}
    for name, arr in W.items():
        Wt[name] = wt.Tensor(arr.copy())
        if name.endswith(".weight") and arr.ndim == 2:                    # pre-transpose (out,in)->(in,out)
            Wt[name + "_T"] = wt.Tensor(np.ascontiguousarray(arr.T))
    return Wt


async def main():
    r = {"backend": backend, "repo": REPO}
    cfg = json.loads(await (await pyfetch(RESOLVE.format(repo=REPO, fn="config.json"))).string())
    W = parse_safetensors(await fetch_bytes("model.safetensors"))
    r["config"] = {k: cfg[k] for k in ("hidden_size", "num_hidden_layers",
                   "num_attention_heads", "num_key_value_heads", "vocab_size", "rope_theta")}
    Wt = to_webtorch(cfg, W)

    # Isolate the very wide LM head from attention/rope/cache.  WebGL texture tiling can be
    # correct for the small projections and still wrap a vocabulary-sized output; reporting
    # both widths turns that backend bug into an immediately local result instead of a
    # mysteriously different generated token eight layers later.
    probe = np.random.default_rng(2026).standard_normal((6, cfg["hidden_size"])).astype(np.float32)
    href = probe @ W["model.embed_tokens.weight"].T
    hgot = wt.Tensor(probe).matmul(Wt["model.embed_tokens.weight_T"]).numpy()
    narrow_w = np.ascontiguousarray(W["model.embed_tokens.weight"][:512].T)
    narrow_ref = probe @ narrow_w
    narrow_got = wt.Tensor(probe).matmul(wt.Tensor(narrow_w)).numpy()
    r["head_max_abs_err"] = float(np.abs(href - hgot).max())
    r["head_argmax_match"] = bool((href.argmax(-1) == hgot.argmax(-1)).all())
    r["head_narrow_max_abs_err"] = float(np.abs(narrow_ref - narrow_got).max())

    ids = [3, 14, 159, 26, 53, 58]                                        # fixed prompt (random-weight model)
    H, nh = cfg["hidden_size"], cfg["num_attention_heads"]; hd = H // nh
    cos, sin = rope_cos_sin(len(ids), hd, cfg["rope_theta"])
    cos_t, sin_t = wt.Tensor(cos), wt.Tensor(sin)
    mask_t = wt.Tensor(np.triu(np.full((len(ids), len(ids)), -1e9, np.float32), 1))

    np_trace, wt_trace = {}, {}
    ref = qwen2_numpy(cfg, W, ids, np_trace)
    got = qwen2_webtorch(cfg, Wt, ids, cos_t, sin_t, mask_t, wt_trace).numpy()
    r["stage_max_abs_err"] = {
        name: float(np.abs(np_trace[name] - wt_trace[name]).max()) for name in np_trace
    }
    r["logits_shape"] = list(got.shape)
    r["max_abs_err"] = float(np.abs(ref - got).max())
    r["argmax_match"] = bool((ref.argmax(-1) == got.argmax(-1)).all())

    # greedy generation (webtorch) vs numpy — token sequences must be identical
    def gen(fwd_last):
        seq = list(ids)
        for _ in range(8):
            seq.append(int(fwd_last(seq)))
        return seq[len(ids):]

    def wt_last(seq):
        c, s = rope_cos_sin(len(seq), hd, cfg["rope_theta"])
        m = wt.Tensor(np.triu(np.full((len(seq), len(seq)), -1e9, np.float32), 1))
        lg = qwen2_webtorch(cfg, Wt, seq, wt.Tensor(c), wt.Tensor(s), m).numpy()
        return lg[-1].argmax()

    def np_last(seq):
        return qwen2_numpy(cfg, W, seq)[-1].argmax()

    gwt, gnp = gen(wt_last), gen(np_last)
    r["gen_webtorch"] = gwt
    r["gen_numpy"] = gnp
    r["gen_match"] = bool(gwt == gnp)
    r["ok"] = bool(r["head_max_abs_err"] < 1e-3 and r["head_argmax_match"]
                   and r["head_narrow_max_abs_err"] < 1e-3
                   and r["max_abs_err"] < 1e-3 and r["argmax_match"] and r["gen_match"])
    print("RESULT " + json.dumps(r))
    pythonIO.result = json.dumps(r)


await main()
