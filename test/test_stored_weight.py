import inspect

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import ggufload


def test_original_width_decode_shape_candidates_cover_distinct_workgroup_splits():
    assert wt._cfg_for("balanced", 256) == (32, 8)
    assert wt._cfg_for("compact", 256) == (32, 4)
    assert wt._selfcheck_shape("balanced", 256)[1] >= 3
    assert wt._selfcheck_shape("compact", 256)[1] >= 3
    source = inspect.getsource(wt._ggml_shape_for)
    assert '("narrow", "balanced", "compact", "shortk", None)' in source


def test_every_decodable_gguf_type_has_a_native_compute_kernel():
    """A storage type may not be accepted and then silently converted for lack of a kernel."""
    assert set(ggufload.SUPPORTED_NAMES) == set(wt._GGML_TYPES)


def test_webgl_row_aligned_half_weight_matmul_matches_flat_weight():
    if not wt._webgl_ready():
        pytest.skip("requires the WebGL browser backend")
    import wgpy as cp

    rng = np.random.default_rng(44)
    lhs = rng.integers(-4, 5, size=(5, 71)).astype(np.float32) * 0.25
    weight = rng.integers(-4, 5, size=(71, 17)).astype(np.float32) * 0.25
    row_weight = wt.webgl_half_matrix(weight).data
    assert row_weight.buffer.texture_shape.width == 17
    assert row_weight.buffer.texture_shape.height == 71
    actual = cp.asnumpy(cp.asarray(lhs) @ row_weight)
    baseline = cp.asnumpy(cp.asarray(lhs) @ cp.asarray(weight))
    np.testing.assert_array_equal(actual, baseline)


def test_q8_native_path_vectorizes_the_original_block_without_requantizing_activations():
    src = wt._Q8_0_DEC
    assert "let p = B4(o + 2u + j * 4u)" in src
    assert "ACC4(kb + j * 4u, d * q)" in src
    assert "round(" not in src
    assert "dot4I8Packed" not in src


def test_q4_native_vector_candidates_never_create_an_alternate_weight_buffer():
    assert "fn Q4LO(p: u32)" in wt._Q4V_FN
    assert "fn Q4HI(p: u32)" in wt._Q4V_FN
    for src in (wt._Q4_0_VEC_DEC, wt._Q4_1_VEC_DEC, wt._Q4K_DEC):
        assert "B4(" in src
        assert "ACC4(" in src
        assert "round(" not in src


def test_measured_positive_q4_vector_paths_are_enabled_without_cross_width_routing():
    for name in ("Q4_0", "Q4_1", "Q4_K"):
        assert "ACC4(" in wt._GGML_TYPES[name][0]
        assert "round(" not in wt._GGML_TYPES[name][0]


def test_q5_candidates_preserve_the_original_high_bit_plane_and_activation_width():
    for src in (wt._Q5_0_VEC_DEC, wt._Q5_1_VEC_DEC):
        assert "U32(" in src
        assert "B4(" in src
        assert "ACC4(" in src
        assert "round(" not in src


def test_q5_same_width_selection_keeps_unstable_decode_scalar_and_vectorizes_other_modes():
    assert "ACC4(" in wt._GGML_TYPES["Q5_0"][0]
    assert "fn Q5LO" not in wt._ggml_src("Q5_1", 1)
    assert "fn Q5LO" in wt._ggml_src("Q5_1", 2)
    assert "fn Q5LO" in wt._ggml_src("Q5_1", 0, mrow=4)


def test_q6_candidate_preserves_low_and_high_planes_at_original_width():
    assert "fn Q6V" in wt._Q6V_FN
    assert "B4(" in wt._Q6K_VEC_DEC
    assert "ACC4(" in wt._Q6K_VEC_DEC
    assert "round(" not in wt._Q6K_VEC_DEC


def test_measured_positive_q6_vector_path_is_enabled_for_every_shape():
    assert wt._GGML_TYPES["Q6_K"][0] is wt._Q6K_VEC_DEC
    for mode in (1, 2, 0):
        assert "ACC4(" in wt._ggml_src("Q6_K", mode, mrow=4 if mode == 0 else None)


def test_remaining_same_width_candidates_do_not_create_alternate_weight_buffers():
    for src in (wt._Q1_0_VEC_DEC, wt._Q2_0_VEC_DEC, wt._TQ2_0_VEC_DEC,
                wt._IQ4NL_VEC_DEC, wt._IQ2XXS_VEC_DEC, wt._TQ1_0_VEC_DEC,
                wt._MXFP4_VEC_DEC, wt._NVFP4_VEC_DEC, wt._IQ1S_VEC_DEC,
                wt._IQ1M_VEC_DEC):
        assert "ACC4(" in src
        assert "round(" not in src


def test_same_width_routing_is_by_storage_format_and_operator_mode():
    assert "ACC4(" in wt._GGML_TYPES["Q1_0"][0]
    assert "ACC4(" in wt._GGML_TYPES["IQ2_XXS"][0]
    for name in ("Q2_0", "TQ2_0", "IQ4_NL"):
        assert "ACC4(" not in wt._ggml_src(name, 1).replace("fn ACC4", "")
        assert "ACC4(" in wt._ggml_src(name, 2)
        assert "ACC4(" in wt._ggml_src(name, 0, mrow=4)


def test_iq1_vector_candidates_read_the_original_signed_grid_bytes():
    assert "fn GI8V" in wt._GRID_FN
    for src in (wt._IQ1S_VEC_DEC, wt._IQ1M_VEC_DEC):
        assert "GI8V(" in src
        assert "G4V(" not in src


def test_measured_positive_low_bit_paths_are_enabled_for_every_shape():
    for name in ("TQ1_0", "MXFP4", "NVFP4", "IQ1_S", "IQ1_M"):
        assert "ACC4(" in wt._GGML_TYPES[name][0]
        assert "round(" not in wt._GGML_TYPES[name][0]


def test_stored_weight_materializes_only_when_an_operator_requests_an_array():
    values = np.arange(-16, 16, dtype=np.int8)
    raw = np.asarray([0.5], dtype=np.float16).tobytes() + values.tobytes()
    weight = wt.GGMLWeight(raw, "Q8_0", (1, 32),
                           type_id=ggufload.GGML_IDS["Q8_0"])

    assert wt.weight_shape(weight) == (1, 32)
    np.testing.assert_array_equal(
        wt.materialize_weight(weight),
        (values.astype(np.float16) * np.float16(0.5)).reshape(1, 32),
    )


def test_non_gguf_quantized_module_uses_the_same_stored_linear_interface():
    module = wt.QuantizedLinear.from_autogptq(
        np.zeros((4, 8), np.int32), np.zeros((1, 1), np.int32),
        np.ones((1, 8), np.float32), np.zeros((8,), np.float32),
        gs=32, bits=8,
    )

    assert wt.stored_linear(module) is module
    assert module.storage_format == "GPTQ_INT8"
    assert module.execution == "auto"


def test_gptq_int4_int8_use_exact_packed_weight_fp32_dot_paths():
    for bits in (4, 8):
        gemm = wt._gptq_src(wt._GPTQ_WGSL, bits, 32)
        gemv = wt._gptq_gemv_src(bits, 32)
        for src in (gemm, gemv):
            assert "GPTQACC" not in src
            assert "dot(vec4<f32>" in src
            assert "round(" not in src
            assert "dot4I8Packed" not in src
        scalar = wt._gptq_src(wt._GPTQ_WGSL, bits, 32, vector=False)
        assert "for (var j: u32" in scalar
    assert wt._gptq_exact_vector(4) is True
    assert wt._gptq_exact_vector(8) is False


def test_webgl_gptq_has_the_same_exact_packed_vec4_candidate():
    vector = wt._gptq_src(wt._GL_GPTQ, 4, 32, vector=True)
    scalar = wt._gptq_src(wt._GL_GPTQ, 4, 32, vector=False)
    for src in (vector, scalar):
        assert "GPTQGLACC" not in src
        assert "round(" not in src
        assert "dot4I8Packed" not in src
    assert "part += dot(vec4(" in vector
    assert "for(int j=0;j<PER;j++)" not in vector
    assert "for(int j=0;j<8;j++)" in scalar


def test_phase_two_cross_width_policy_is_explicit_for_both_browser_backends():
    route = wt._PHASE2_CROSS_WIDTH["gptq_activation_int8_dp4a"]
    assert route == {
        "webgpu": "measured_per_format_shape_device",
        "webgl": "primitive_unavailable_keep_stored",
    }
    assert "requires packed_4x8_integer_dot_product" in wt._GPTQ_DP4A_WGSL
    assert "dot4I8Packed" in wt._gptq_dp4a_src(4)
    assert "dot4I8Packed" in wt._gptq_dp4a_src(8)
    assert "dot4I8Packed" not in wt._GL_GPTQ
    forward = inspect.getsource(wt.QuantizedLinear.forward)
    assert '("stored", "dp4a", "materialized")' in forward
    assert "_gptq_dp4a_matmul" in forward


def test_every_ggml_decoder_can_generate_a_webgl_shader():
    for name in wt._GGML_TYPES:
        src = wt._ggml_src_gl(name, False, False)
        assert "#version 300 es" in src
        assert "void main()" in src
        assert "DECODE" not in src
        assert "ACC4(" not in src.replace("void ACC4(", "") or "dot(vec4" in src


def test_webgl_same_width_routing_is_independent_and_shape_bucketed():
    # Q6_K's WebGL vector path loses for decode/small batch but wins for a large batch;
    # WebGPU uses its own independently measured routing.
    assert wt._GGML_GL_MODE_DECODERS["Q6_K"][1][0] is wt._Q6K_DEC
    assert wt._GGML_GL_MODE_DECODERS["Q6_K"][3][0] is wt._Q6K_DEC
    assert 0 not in wt._GGML_GL_MODE_DECODERS["Q6_K"]
    assert wt._ggml_src_gl("Q6_K", False, False, mode=1) != \
           wt._ggml_src_gl("Q6_K", False, False, mode=0)
    assert wt._ggml_name_gl("Q6_K", False, False, mode=1) != \
           wt._ggml_name_gl("Q6_K", False, False, mode=0)


def test_gptq_same_width_routing_is_backend_specific(monkeypatch):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    assert wt._gptq_exact_vector(4, 1) is True
    assert wt._gptq_exact_vector(4, 64) is False
    assert wt._gptq_exact_vector(8, 1) is True
    assert wt._gptq_exact_vector(8, 16) is False
    assert wt._gptq_exact_vector(8, 64) is True


def test_low_level_stored_baseline_and_production_auto_policy_are_explicit():
    assert inspect.signature(wt.ggml_matmul).parameters["execution"].default == "stored"

    class EncodedWeight:
        def as_linear(self, bias=None, execution="stored"):
            return {"bias": bias, "execution": execution}

    assert wt.stored_linear(EncodedWeight(), bias="b") == {
        "bias": "b", "execution": "auto",
    }
    assert wt.stored_linear(EncodedWeight(), execution="auto") == {
        "bias": None, "execution": "auto",
    }


def test_execution_policy_is_shape_based_cached_and_profiled():
    key = ("weight_exec", "ggml", "Q8_0", 1024, 512, 128)
    before = wt._TUNED.get(key)
    wt._TUNED[key] = "materialized"
    try:
        assert wt._weight_execution("ggml", "Q8_0", 1024, 512, 100,
                                    lambda _which: None) == "materialized"
        profile = wt.kernel_profile()
        assert profile["tuned"]["weight_exec|ggml|Q8_0|1024|512|128"] == "materialized"
    finally:
        if before is None:
            wt._TUNED.pop(key, None)
        else:
            wt._TUNED[key] = before

    # Decode and two-row verification are measured too: a local win must not be discarded
    # merely because a broader batch bucket did not win.
    calls = []

    class Ready:
        def get(self):
            return None

    assert wt._weight_execution("gptq", "GPTQ_INT4_TEST", 1024, 512, 2,
                                lambda which: calls.append(which) or Ready()) in {
                                    "stored", "materialized"
                                }
    assert calls


def test_execution_candidate_failure_cannot_cache_a_silent_downgrade():
    key = ("weight_exec", "failure_probe", "f32", 19, 23, 8)
    wt._TUNED.pop(key, None)

    def run(which):
        if which == "new":
            raise RuntimeError("shader compile failed")
        return np.asarray([1.0], np.float32)

    with pytest.raises(RuntimeError, match="execution candidate 'new' failed"):
        wt._weight_execution("failure_probe", "f32", 19, 23, 5, run,
                             candidates=("base", "new"), rounds=5, repeat=1)
    assert key not in wt._TUNED


def test_phase_two_dp4a_choice_round_trips_through_device_profile():
    key = ("weight_exec", "gptq", "GPTQ_INT4", 4096, 3072, 2)
    before = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = "dp4a"
        profile = wt.kernel_profile()
        assert profile["tuned"]["weight_exec|gptq|GPTQ_INT4|4096|3072|2"] == "dp4a"
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key] == "dp4a"
    finally:
        if before is None:
            wt._TUNED.pop(key, None)
        else:
            wt._TUNED[key] = before


def test_execution_choice_has_no_percentage_cutoff_and_rejects_noise():
    # A repeatable 1% win is a win; no arbitrary 5% policy may erase it.
    stable = {
        "stored": [1.0] * 9,
        "candidate": [0.99] * 9,
    }
    assert wt._paired_faster(stable, "candidate", "stored")
    assert wt._measured_choice(stable, ("stored", "candidate")) == "candidate"

    # A larger median that alternates direction is not stable evidence.  The lower-memory
    # first candidate remains selected until the local measurements prove a benefit.
    noisy = {
        "stored": [1.0] * 9,
        "candidate": [0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 0.8],
    }
    assert not wt._paired_faster(noisy, "candidate", "stored")
    assert wt._measured_choice(noisy, ("stored", "candidate")) == "stored"


def test_autogptq_zero_offset_is_part_of_the_materialized_shader():
    source = wt._gptq_src(wt._DQF_WGSL, 4, zoff=1.0)
    assert "ZOFFf" not in source
    assert "f32(zv) + 1.0" in source


def test_tiled_route_reads_the_stored_blocks_and_rounds_nothing_below_f32():
    """The tiled candidate is a same-width route: it binds the stored buffer, keeps the
    integers exact (int8 fits a half), and applies the block scale to an f32 partial sum."""
    assert set(wt._GGML_TILED) <= set(wt._GGML_TYPES)
    src = wt._GGML_TILED["Q8_0"]
    assert "var<storage,read> packed: array<vec4<u32>>" in src
    assert "round(" not in src and "dot4I8Packed" not in src
    assert "h2(sx(ga, 24u), sx(gb, 24u))" in src       # raw int8, no scale applied
    assert "s00 = p00 * d0 + s00" in src                # scale on the partial sum
    assert wt._ggml_tiled_ok("Q8_0", 768, 2304)
    assert not wt._ggml_tiled_ok("Q8_0", 768, 2302)     # output is written as vec4
    assert not wt._ggml_tiled_ok("Q8_0", 770, 2304)     # whole blocks only


def test_tiled_split_cuts_k_only_for_few_workgroups():
    assert wt._ggml_tiled_split(8, 768) == 4         # 12 workgroups
    assert wt._ggml_tiled_split(64, 1152) == 4       # 24
    assert wt._ggml_tiled_split(128, 768) == 1       # 48: measured slower when cut
    assert wt._ggml_tiled_split(64, 2304) == 1       # 72



def _fresh_route_state(monkeypatch):
    monkeypatch.setattr(wt, "_TUNED", {})
    monkeypatch.setattr(wt, "_CALIBRATED", set())
    monkeypatch.setattr(wt, "_NEAREST", {})
    monkeypatch.setattr(wt, "_CALIB_TOUCHED", [])


def test_calibration_bisects_only_where_the_winner_changes(monkeypatch):
    _fresh_route_state(monkeypatch)

    def probe(m):
        key = ("weight_exec", "fam", "fmt", 1, 2, m)
        wt._CALIB_TOUCHED.append(key)
        wt._TUNED.setdefault(key, "a" if m < 100 else "b")

    assert wt.calibrate_rows(probe, 512, lo=16, step=8) == [16, 64, 512]
    assert ("weight_exec", "fam", "fmt", 1, 2) in wt._CALIBRATED

    _fresh_route_state(monkeypatch)

    def same(m):
        key = ("weight_exec", "fam", "fmt", 1, 2, m)
        wt._CALIB_TOUCHED.append(key)
        wt._TUNED.setdefault(key, "a")

    assert wt.calibrate_rows(same, 512, lo=16, step=8) == [16, 512]


def test_an_answer_borrows_the_nearest_calibrated_bucket_and_never_races(monkeypatch):
    _fresh_route_state(monkeypatch)
    prefix = ("weight_exec", "ggml", "Q8_0", 768, 2304)
    wt._TUNED[prefix + (16,)] = "stored"
    wt._TUNED[prefix + (512,)] = "tiled"
    raced = []

    def run(which):
        raced.append(which)
        raise AssertionError("an answer must not race")

    # Not calibrated: the old behaviour (race now) is kept, which the run above refuses.
    with pytest.raises(RuntimeError):
        wt._weight_execution("ggml", "Q8_0", 768, 2304, 519, run, candidates=("stored", "tiled"))
    raced.clear()
    wt._CALIBRATED.add(prefix)
    assert wt._weight_execution("ggml", "Q8_0", 768, 2304, 519, run,
                                candidates=("stored", "tiled")) == "tiled"     # 1024 -> 512
    assert wt._weight_execution("ggml", "Q8_0", 768, 2304, 40, run,
                                candidates=("stored", "tiled")) == "stored"    # 64 -> 16
    assert raced == []
    assert prefix + (1024,) not in wt._TUNED          # a borrowed choice is not a measurement
    # While calibrating, the same call measures instead of borrowing.
    with wt._calibrating():
        with pytest.raises(RuntimeError):
            wt._weight_execution("ggml", "Q8_0", 768, 2304, 519, run,
                                 candidates=("stored", "tiled"))


def test_calibrated_prefixes_round_trip_through_the_device_profile(monkeypatch):
    _fresh_route_state(monkeypatch)
    prefix = ("weight_exec", "ggml", "Q8_0", 768, 2304)
    wt._TUNED[prefix + (512,)] = "tiled"
    wt._CALIBRATED.add(prefix)
    profile = wt.kernel_profile()
    assert profile["calibrated"] == ["weight_exec|ggml|Q8_0|768|2304"]
    _fresh_route_state(monkeypatch)
    assert wt.use_kernel_profile(profile) >= 2
    assert prefix in wt._CALIBRATED and wt._TUNED[prefix + (512,)] == "tiled"


def test_the_sdk_keeps_its_measurements_unless_a_host_says_no():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    host = (root / "webtorch/js/webtorch-host.js").read_text()
    main = (root / "webtorch/js/webtorch-main.js").read_text()
    assert "remember = !(a && a.rememberTuning === false);" in host
    assert "rememberTuning: opts.rememberTuning !== false" in main
    assert "rememberTuning" not in (root / "chat/app.js").read_text()
