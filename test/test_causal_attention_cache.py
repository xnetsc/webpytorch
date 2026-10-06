"""Prefill attention read straight from the packed-half KV cache (`causal_attention_cache`).

Host tests cover what holds without a GPU; the browser test checks the kernel against a
float64 reference on the cache's own half values, across head dims, GQA ratios, prompt
lengths, cached prefixes and every key split the race can choose.
"""
import re

import numpy as np
import pytest

from webtorch import _core as wt


def test_no_webgpu_means_no_cache_attention():
    if wt._adam_backend_ready():
        pytest.skip("host test")
    q = wt.Tensor(np.zeros((2, 3, 64), np.float32))
    assert wt.causal_attention_cache(q, q, q, 0, 2, 1, 64, 16, 0.125) is None


def test_split_count_follows_the_target_and_the_context():
    # Enough query-block x head workgroups already: no split, whatever the target.
    assert wt._cattn_split(96, T=512, end=512, NH=16) == 1
    # A short new segment after a long context splits the keys up to the target...
    assert wt._cattn_split(192, T=28, end=2028, NH=16) == 12
    # ...but never into slices shorter than 64 keys.
    assert wt._cattn_split(384, T=8, end=100, NH=4) == 2
    assert wt._cattn_split(1, T=8, end=4000, NH=4) == 1


def test_kernel_writes_whole_probability_vectors_and_masks_causally():
    for HD in (64, 128):
        src = wt._cattn_src(HD)
        assert re.search(r"pt\[[^\]]*\]\[", src) is None      # no component stores to shared
        assert "key < ke && key <= qp0" in src                  # causal and range mask
        assert src.count("out4[(qr * cm.NH + h)") == 4 * (HD // 32)


def _reference(qf, kh, vh, start, NH, NKV, scale):
    T = qf.shape[1]
    end = start + T
    out = np.empty((T, NH, qf.shape[2]))
    for h in range(NH):
        g = h // (NH // NKV)
        s = qf[h].astype(np.float64) @ kh[g, :end].astype(np.float64).T * scale
        s[np.arange(end)[None, :] > (start + np.arange(T))[:, None]] = -np.inf
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[:, h] = p @ vh[g, :end].astype(np.float64)
    return out.reshape(T, -1)


def test_cache_attention_matches_the_reference_in_the_browser():
    if not (wt._adam_backend_ready() and wt.kv_f16()):
        pytest.skip("requires the WebGPU browser backend with a half KV cache")
    rng = np.random.default_rng(11)
    for (NH, NKV, HD, LMAX, T, start) in ((16, 8, 128, 1024, 28, 180), (8, 2, 64, 512, 70, 0),
                                          (4, 4, 128, 4096, 9, 3000), (6, 3, 64, 256, 1, 37)):
        kh = rng.standard_normal((NKV, LMAX, HD)).astype(np.float16)
        vh = rng.standard_normal((NKV, LMAX, HD)).astype(np.float16)
        kc = wt.Tensor(np.ascontiguousarray(kh).view(np.uint32).view(np.float32).reshape(-1))
        vc = wt.Tensor(np.ascontiguousarray(vh).view(np.uint32).view(np.float32).reshape(-1))
        qf = rng.standard_normal((NH, T, HD)).astype(np.float32)
        scale = 1.0 / np.sqrt(HD)
        want = _reference(qf, kh, vh, start, NH, NKV, scale)
        for target in wt._CATTN_TARGETS:
            got = np.asarray(wt._cattn_run(target, wt._contig(wt.Tensor(qf).data), kc.data,
                                           vc.data, T, start, NH, NKV, HD, LMAX,
                                           scale).get()).reshape(T, -1)
            assert np.abs(got - want).max() < 1e-4, (NH, NKV, HD, T, start, target)


def test_rows_rope_and_cache_writes_match_the_reference_in_the_browser():
    """`rope_qk_rows_kv` from the projections' (T, heads, HD) rows: q rotated heads-first, k
    rotated and v written as halves into the packed caches at rows start.."""
    if not (wt._adam_backend_ready() and wt.kv_f16()):
        pytest.skip("requires the WebGPU browser backend with a half KV cache")
    rng = np.random.default_rng(12)
    for (T, NH, NKV, HD, LMAX, start) in ((7, 4, 2, 64, 64, 10), (33, 16, 8, 128, 128, 0)):
        q = rng.standard_normal((T, NH, HD)).astype(np.float32)
        k = rng.standard_normal((T, NKV, HD)).astype(np.float32)
        v = rng.standard_normal((T, NKV, HD)).astype(np.float32)
        c, s = wt.rope_tables(T, HD, 10000.0, offset=start)
        c, s = np.asarray(c, np.float32), np.asarray(s, np.float32)
        kc = wt.Tensor(np.zeros((NKV * LMAX * HD // 2,), np.float32))
        vc = wt.Tensor(np.zeros((NKV * LMAX * HD // 2,), np.float32))
        kc.data = kc.data.reshape(NKV, LMAX, HD // 2)
        vc.data = vc.data.reshape(NKV, LMAX, HD // 2)
        got = wt.rope_qk_rows_kv(wt.Tensor(q), wt.Tensor(k), wt.Tensor(v), wt.Tensor(c),
                                 wt.Tensor(s), kc, vc, start, NH, NKV, HD, LMAX)
        half = HD // 2

        def rot(x):
            r = np.concatenate([-x[..., half:], x[..., :half]], -1)
            return x * c[:, None, :] + r * s[:, None, :]
        assert np.abs(np.asarray(got.numpy()) - rot(q).transpose(1, 0, 2)).max() < 1e-5
        kh = np.asarray(kc.data.get(), np.float32).view(np.uint32).view(np.float16)
        vh = np.asarray(vc.data.get(), np.float32).view(np.uint32).view(np.float16)
        kh = kh.reshape(NKV, LMAX, HD).astype(np.float32)[:, start:start + T]
        vh = vh.reshape(NKV, LMAX, HD).astype(np.float32)[:, start:start + T]
        # The rotation is f32 on both sides but not bit-identical (contraction), so a value
        # sitting on a half's rounding boundary can land one half-ulp either way.
        kref = rot(k).transpose(1, 0, 2)
        assert np.all(np.abs(kh - kref) <= 2.0 ** -10 * np.maximum(np.abs(kref), 2.0 ** -14))
        assert np.abs(vh - v.astype(np.float16).astype(np.float32).transpose(1, 0, 2)).max() == 0
