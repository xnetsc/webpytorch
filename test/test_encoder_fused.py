"""Fused encoder kernels: attention off the packed projection, LayerNorm with the residual
add, half-arithmetic dense matmul, and the feature detection that gates it.

Host tests check what holds without a GPU -- the fallbacks are the expressions they replace,
the encoder's restructured residual stream is the same arithmetic, the generated kernels
have the shape the dispatch assumes. The `*_in_the_browser` tests need WebGPU and skip
elsewhere; run them in the page's SDK worker.
"""
import re

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import decision
from webtorch.encoder import EncoderConfig, TextEncoder


def _tiny_encoder(layers=4, hidden=64, heads=2, ffn=48, vocab=40, window=4, seed=0):
    cfg = EncoderConfig({"hidden_size": hidden, "num_hidden_layers": layers,
                         "num_attention_heads": heads, "intermediate_size": ffn,
                         "vocab_size": vocab, "global_attn_every_n_layers": 3,
                         "local_attention": 2 * window, "norm_eps": 1e-5,
                         "hidden_activation": "gelu"})
    rng = np.random.default_rng(seed)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.2).astype(np.float32)
    p = "encoder."
    w = {p + "embeddings.tok_embeddings.weight": r(vocab, hidden),
         p + "embeddings.norm.weight": 1 + r(hidden), p + "final_norm.weight": 1 + r(hidden)}
    for i in range(layers):
        q = p + "layers.%d." % i
        if i:
            w[q + "attn_norm.weight"] = 1 + r(hidden)
        w[q + "attn.Wqkv.weight"] = r(3 * hidden, hidden)
        w[q + "attn.Wo.weight"] = r(hidden, hidden)
        w[q + "mlp_norm.weight"] = 1 + r(hidden)
        w[q + "mlp.Wi.weight"] = r(2 * ffn, hidden)
        w[q + "mlp.Wo.weight"] = r(hidden, ffn)
    return TextEncoder(cfg, w)


def _old_layers(enc, x, masks, B):
    """The stack as it was written before every add was fused with the norm after it."""
    x = enc._norm(x, enc.p + "embeddings.norm")
    for i in range(enc.cfg.layers):
        kind = enc.cfg.layer_types[i]
        an = "%slayers.%d.attn_norm" % (enc.p, i)
        xa = enc._norm(x, an) if enc._has(an + ".weight") else x
        x = x + enc._attn(xa, i, kind, masks[kind], B)
        x = x + enc._mlp(enc._norm(x, "%slayers.%d.mlp_norm" % (enc.p, i)), i)
    return enc._norm(x, enc.p + "final_norm")


def test_fused_residual_norms_are_the_same_arithmetic_on_the_host():
    if wt._adam_backend_ready():
        pytest.skip("host fallback test")
    enc = _tiny_encoder()
    ids = np.array([1, 5, 9, 3, 7, 2, 8, 4, 6, 1, 3], dtype=np.int64)
    T = len(ids)
    masks = {k: enc._mask(T, None, k) for k in set(enc.cfg.layer_types)}
    want = np.asarray(_old_layers(enc, enc._embed(ids), masks, 1).numpy())
    got = np.asarray(enc._layers(enc._embed(ids), masks, 1).numpy())
    assert np.array_equal(got, want)


def test_add_layernorm_falls_back_to_the_two_operations():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 12)).astype(np.float32)
    y = rng.standard_normal((5, 12)).astype(np.float32)
    g = rng.standard_normal(12).astype(np.float32)
    b = rng.standard_normal(12).astype(np.float32)
    s, ln = wt.add_layernorm(wt.Tensor(x), wt.Tensor(y), wt.Tensor(g), wt.Tensor(b))
    v = x + y
    mu = v.mean(-1, keepdims=True)
    ref = (v - mu) / np.sqrt(((v - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
    assert np.allclose(np.asarray(s.numpy()), v)
    assert np.allclose(np.asarray(ln.numpy()), ref, atol=1e-5)


def test_no_gpu_means_no_features_and_no_fused_attention():
    if wt._adam_backend_ready():
        pytest.skip("host test")
    assert wt.gpu_features() == {}
    qkv = wt.Tensor(np.zeros((6, 3 * 2 * 32), np.float32))
    assert wt.fused_attention(qkv, 2, 32, 6, 0.1) is None
    assert wt.rope_qk(qkv, qkv, qkv, 2, 32, 6) is None


def test_attention_kernel_shape_matches_its_dispatch():
    for HD, tile in ((64, "4x4"), (64, "2x8"), (128, "4x4"), (32, "2x8")):
        RI, CJ = wt._ATTN_TILES[tile]
        src = wt._attn_src(HD, RI, CJ)
        assert "@workgroup_size(8, 8, 1)" in src
        # whole-vec4 probability writes only: a component store to a shared vec4 races
        assert "pt[(ty + 0u) * %du + tx * %du + 0u] = vec4<f32>(" % (2 * CJ + 1, CJ // 4) in src
        assert re.search(r"pt\[[^\]]*\]\[", src) is None
        # one output vec4 per tx and (row, e): HD/32 of them per row
        assert src.count("out[(b * T + qr) * DO") == RI * (HD // 32)


def test_half_matmul_kernel_flushes_in_f32_and_needs_the_feature():
    src = wt._mm_half_src()
    assert src.startswith("enable f16;")
    assert "S0_0 = S0_0 + vec4<f32>(h0_0)" in src
    assert "% 32u) == 0u" in src                       # default: every 32 products
    mid = wt._mm_half_src(FLUSH=16)                    # shorter than a stage: inside it
    assert "if (((kk + 1u) % 4u) == 0u)" in mid


def test_half_matmul_is_offered_only_where_the_device_has_half_arithmetic(monkeypatch):
    calls = []
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_contig", lambda a: a)
    monkeypatch.setattr(wt, "_weight_execution",
                        lambda *a, **k: calls.append(k.get("candidates")) or "f16")
    monkeypatch.setattr(wt, "_mm_half", lambda xd, wd, M, K, N: np.zeros((M, N), np.float32))

    class X(object):
        shape = (3, 64)
    monkeypatch.setattr(wt, "gpu_features", lambda: {"f16": True})
    out = wt.matmul_f16w(X(), np.zeros(4, np.float32), 64, 64)
    assert calls == [("f32", "f16")] and tuple(out.shape) == (3, 64)
    calls.clear()
    monkeypatch.setattr(wt, "gpu_features", lambda: {"f16": False})
    try:                                               # goes on to the f32 kernel: no race
        wt.matmul_f16w(X(), np.zeros(4, np.float32), 64, 64)
    except Exception:
        pass
    assert calls == []


def test_decision_calibration_ladders_dense_half_only_with_the_feature(monkeypatch):
    ladders = []
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "calibrate_rows", lambda probe, top, **k: ladders.append(probe))

    class Enc(object):
        _ten = {("w", "f16"): wt.Tensor(np.zeros((4, 4), np.float32))}
        shape_of = {"w": (64, 32)}
        cfg = type("C", (), {"layer_types": []})()

    class Impl(object):
        enc = Enc()
        _ten = {}

    class IO(object):
        @staticmethod
        def load_stage(*a):
            pass
    monkeypatch.setattr(wt, "gpu_features", lambda: {"f16": False})
    decision._calibrate_routes(Impl(), IO())
    assert ladders == []
    monkeypatch.setattr(wt, "gpu_features", lambda: {"f16": True})
    decision._calibrate_routes(Impl(), IO())
    assert len(ladders) == 1


# ---- in the browser -------------------------------------------------------------------

def _reference_attention(qkv, H, HD, T, B, scale, mask, cos=None, sin=None):
    x = qkv.reshape(B, T, 3, H, HD).astype(np.float64)
    q, k, v = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    if cos is not None:
        half = HD // 2

        def rot(t):
            r = np.concatenate([-t[..., half:], t[..., :half]], -1)
            return t * cos[None, :, None, :] + r * sin[None, :, None, :]
        q, k = rot(q), rot(k)
    s = np.einsum("bqhd,bkhd->bhqk", q, k) * scale
    m = np.asarray(mask, np.float64)
    if m.ndim == 2:
        s = s + m
    else:
        s = s + m.reshape(B, -1, T, T)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", p, v).reshape(B * T, H * HD)


def test_fused_attention_matches_the_reference_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(3)
    for (B, T, H, HD, window, planes, rope) in ((1, 37, 2, 64, 0, "one", True),
                                                (3, 50, 4, 64, 8, "seq", True),
                                                (2, 41, 2, 128, 0, "head", False),
                                                (1, 70, 3, 32, 16, "one", True)):
        qkv = rng.standard_normal((B * T, 3 * H * HD)).astype(np.float32)
        idx = np.arange(T)
        base = np.zeros((T, T), np.float32)
        if window:
            base[np.abs(idx[:, None] - idx[None, :]) > window] = -1e9
        if planes == "one":
            mask = base
        else:
            per = 1 if planes == "seq" else H
            mask = np.repeat(base[None], B * per, 0)
            for b in range(B):
                mask[b * per:(b + 1) * per, :, T - 3 - b:] = -1e9      # padding per sequence
        cos = sin = None
        if rope:
            c, s_ = wt.rope_tables(T, HD, 10000.0)
            cos, sin = np.asarray(c, np.float32), np.asarray(s_, np.float32)
        for tile in wt._ATTN_TILES:
            args = dict(mask=wt.Tensor(mask), window=window, B=B,
                        cos=wt.Tensor(cos) if rope else None, sin=wt.Tensor(sin) if rope else None)
            wt._TUNED[("weight_exec", "attention", "f32", HD, 1 if window else 0,
                       1 << (T - 1).bit_length())] = tile
            got = np.asarray(wt.fused_attention(wt.Tensor(qkv), H, HD, T, 0.125, **args).numpy())
            want = _reference_attention(qkv, H, HD, T, B, 0.125, mask, cos, sin)
            assert np.abs(got - want).max() < 1e-4 * max(1.0, np.abs(want).max()), (B, T, HD, tile)


def test_row_layernorm_matches_numpy_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(4)
    for rows, W in ((519, 768), (3, 772), (1, 1024), (7, 4096), (2, 300), (2, 30)):
        x = (rng.standard_normal((rows, W)) * 3 + 1).astype(np.float32)
        y = rng.standard_normal((rows, W)).astype(np.float32)
        g = rng.standard_normal(W).astype(np.float32)
        b = rng.standard_normal(W).astype(np.float32)

        def ref(v):
            mu = v.mean(-1, keepdims=True)
            return (v - mu) / np.sqrt(((v - mu) ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
        got = np.asarray(wt.layernorm(wt.Tensor(x), wt.Tensor(g), wt.Tensor(b)).numpy())
        assert np.abs(got - ref(x)).max() < 1e-4, (rows, W)
        s, ln = wt.add_layernorm(wt.Tensor(x), wt.Tensor(y), wt.Tensor(g), wt.Tensor(b))
        assert np.array_equal(np.asarray(s.numpy()), x + y)
        assert np.abs(np.asarray(ln.numpy()) - ref(x + y)).max() < 1e-4, (rows, W)


def test_half_arithmetic_matmul_stays_within_its_bound_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    if not wt.gpu_features().get("f16"):
        pytest.skip("this device has no shader-f16")
    rng = np.random.default_rng(5)
    for M, K, N in ((1, 64, 64), (37, 96, 128), (519, 768, 2304), (70, 1152, 768)):
        w = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
        pk = wt.pack_half_weight(w)                    # (N_out, K_in), as checkpoints store it
        x = rng.standard_normal((M, K)).astype(np.float32)
        ref = x.astype(np.float64) @ w.astype(np.float64).T
        scale = np.abs(ref).max()
        f32 = np.asarray(wt.matmul_f16w(wt.Tensor(x), pk, K, N, execution="f32").numpy())
        f16 = np.asarray(wt.matmul_f16w(wt.Tensor(x), pk, K, N, execution="f16").numpy())
        assert np.abs(f32 - ref).max() / scale < 1e-5, (M, K, N)
        assert np.abs(f16 - ref).max() / scale < 5e-3, (M, K, N)
