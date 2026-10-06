"""Routed experts for a whole prompt grouped by expert (`moe_group`, `_ggml_tiled_moe_src`,
`GGMLMoELinear.forward_routed`): one weight read per routed expert instead of per slot."""
import re

import numpy as np
import pytest

from webtorch import _core as wt


def test_grouped_kernels_generate_for_every_tiled_format():
    for fmt in wt._TILED_FORMATS:
        for half in (False, True):
            src, gbind = wt._ggml_tiled_moe_src(fmt, half)
            base = wt._ggml_tiled_half_src(fmt) if half else wt._GGML_TILED[fmt]
            grid = "binding(4) var<storage,read> gr:" in base.replace("> gr ", "> gr:")
            assert gbind == (5 if grid else 4)
            assert "@binding(%d) var<storage,read> grp" % gbind in src
            assert "packed[EOFF + w * ND4 + c4]" in src
            assert "array_c[cx + o0 * ND4]" in src            # rows written to their slot
            assert not re.search(r"\b(HASB_|HALFFMA)", src)


def test_grouping_is_refused_past_its_workgroup():
    if wt._adam_backend_ready():
        pytest.skip("host test")
    assert wt.moe_group(None, 16, 300) is None


def test_routed_names_survive_a_saved_profile():
    assert {"slots", "grouped", "grouped_half"} <= wt._route_names()
    assert "moe_grouped_activation_f16" in wt._PHASE2_CROSS_WIDTH


def _stack(fmt, E, K, N, rng):
    halves = {"Q4_K": (0, 2), "Q3_K": (108,), "Q6_K": (208,), "IQ4_XS": (0,)}
    vals, blk = wt._GGML_TYPES[fmt][2], wt._GGML_TYPES[fmt][3]
    nb = K // vals
    chunks = []
    for _ in range(E):
        raw = rng.integers(0, 256, (N, nb, blk), dtype=np.uint8)
        for o in halves[fmt]:
            h = (rng.random((N, nb)) * 0.004 + 0.0005).astype(np.float16)
            raw[:, :, o:o + 2] = h.view(np.uint8).reshape(N, nb, 2)
        chunks.append(raw.tobytes())
    return wt.GGMLMoELinear(chunks, fmt, K, N)


def test_grouped_routes_match_the_slot_route_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(5)
    for fmt in ("Q4_K", "Q3_K", "Q6_K", "IQ4_XS"):
        E, K, N, T, k = 16, 512, 192, 37, 4
        st = _stack(fmt, E, K, N, rng)
        S = T * k
        x = rng.standard_normal((T, K)).astype(np.float32)
        ei = rng.integers(0, E, S).astype(np.int32)
        ei[:5] = 3                                    # an expert with more than one tile's rows
        eidx = wt._empty_i32((S,))
        eidx.buffer.set_data(ei)
        ref = np.asarray(st.forward(wt.Tensor(np.repeat(x, k, axis=0)), eidx).data.get())
        ref = ref.reshape(S, N)
        scale = float(np.abs(ref).max())
        group = wt.moe_group(eidx, S, E)
        halves = (False, True) if wt.gpu_features().get("f16") else (False,)
        for half in halves:
            for slot_rows, xs in ((False, x), (True, np.repeat(x, k, axis=0))):
                got = np.asarray(wt._moe_grouped_run(st, wt.Tensor(xs).data, group, k, slot_rows,
                                                     S, half).get()).reshape(S, N)
                bound = 1e-2 if half else 1e-5
                assert float(np.abs(got - ref).max()) / scale < bound, (fmt, half, slot_rows)
