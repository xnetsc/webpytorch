"""Measured layer-composition routes survive a same-build, same-device reload."""

import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import llm


def test_recorded_device_profile_is_build_bound_and_importable(monkeypatch):
    """Preserve the measured route as data, never a model-name runtime branch."""
    path = (Path(__file__).resolve().parents[1] / "profiles"
            / "webgpu_apple_metal3_2026-10-04.json")
    profile = json.loads(path.read_text())
    assert profile["build"] == wt._kernel_build()
    assert profile["provenance"]["page_tok_s_range"][1] < 140
    key = profile["provenance"]["decode_profile_key"]
    assert profile["tuned"][key]["plan"] == [
        "fused", ["stored"] * 5, "full", "fused", "separate", "fused",
        "fused", "compact", ["auto", "auto"]]
    monkeypatch.setattr(wt, "_TUNED", {})
    monkeypatch.setattr(wt, "_GQA_TUNED", {})
    monkeypatch.setattr(wt, "_CHECKED", {})
    monkeypatch.setattr(wt, "_DEQ_OK", {})
    assert wt.use_kernel_profile(profile) >= 20
    assert wt._TUNED[tuple(key.split("|"))]["plan"] == profile["tuned"][key]["plan"]
    assert wt._TUNED[("vocab_sample_full", 151936, "webgpu")] == "gpu"


def test_whole_prefill_profile_is_independent_of_the_layer_route(monkeypatch):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    stacked = {"gate": SimpleNamespace(n_experts=128),
               "gate_up": SimpleNamespace(type_name="Q3_K"),
               "down": SimpleNamespace(type_name="Q4_K")}
    model = SimpleNamespace(base="local/content-hash.gguf", L=2, H=2048,
                            NH=32, NKV=4, HD=64, VOCAB=151936,
                            layers=[{"moe": {"stacked": stacked, "n_experts": 128,
                                              "top_k": 8}}, {}])
    cold = llm.CausalLM._moe_prefill_api_key(model, 4)
    assert cold[:2] == ("moe_prefill_api_v3", "webgl")
    assert cold[-2:] == (4, "cold")
    assert llm.CausalLM._moe_prefill_api_key(model, 96)[-2:] == (64, "cold")
    lower = ("moe_prefill_route_js_v3", "webgl", 128, 8, 4, "Q3_K", "Q4_K", "cold")
    before = {key: wt._TUNED.get(key) for key in (cold, lower)}
    try:
        wt._TUNED[cold] = "device"
        wt._TUNED[lower] = "host"
        assert llm.CausalLM._moe_prefill_api_choice(model, 4) == "device"
        profile = wt.kernel_profile()
        wt._TUNED.pop(cold)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[cold] == "device"
        assert wt._TUNED[lower] == "host"
        llm.CausalLM._moe_prefill_api_completed(model, 4)
        warm = llm.CausalLM._moe_prefill_api_key(model, 4)
        assert warm[:-1] == cold[:-1] and warm[-1] == "warm"
        assert llm.CausalLM._moe_prefill_api_choice(model, 4) == "device"
        assert warm in model._moe_prefill_api_pending
        profile["tuned"]["|".join(map(str, cold))] = "wrong"
        wt._TUNED.pop(cold)
        wt.use_kernel_profile(profile)
        assert cold not in wt._TUNED
    finally:
        for key, old in before.items():
            wt._TUNED.pop(key, None)
            if old is not None:
                wt._TUNED[key] = old


@pytest.mark.parametrize("bad_device", [False, True])
def test_whole_prefill_calibration_checks_result_before_saving(monkeypatch,
                                                               bad_device):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "KVCache", lambda *args: object())
    monkeypatch.setattr(wt, "_measured_choice", lambda *args, **kw: "device")
    stacked = {"gate": SimpleNamespace(n_experts=8),
               "gate_up": SimpleNamespace(type_name="Q3_K"),
               "down": SimpleNamespace(type_name="Q3_K")}
    model = SimpleNamespace(base="local/quantized.gguf", L=1, H=16, NH=2,
                            NKV=1, HD=8, VOCAB=128, lmax=128,
                            layers=[{"moe": {"stacked": stacked,
                                              "top_k": 2}}],
                            _capturable=lambda: False,
                            _reset_linear_state=lambda: None)

    def forward(ids, pos, cache):
        value = (2.0 if bad_device and model._moe_prefill_execution == "device"
                 else 1.0)
        model._last_hidden = SimpleNamespace(
            numpy=lambda: np.full((1, 16), value, np.float32))
        llm.CausalLM._moe_prefill_api_completed(model, len(ids))
        return 5

    model._kv_forward = forward
    cold = llm.CausalLM._moe_prefill_api_key(model, 4)
    warm = (*cold[:-1], "warm")
    previous = wt._TUNED.get(warm)
    wt._TUNED.pop(warm, None)
    try:
        if bad_device:
            with pytest.raises(RuntimeError, match="numerical gate"):
                llm.CausalLM.profile_moe_prefill_composition(model, rounds=2)
            assert warm not in wt._TUNED
        else:
            result = llm.CausalLM.profile_moe_prefill_composition(model, rounds=2)
            assert result["key"] == warm
            assert result["choice"] == wt._TUNED[warm] == "device"
            assert result["phase"] == "warm"
            assert result["cold_profile_pending"] is True
            assert len(result["samples_seconds"]["host"]) == 2
    finally:
        wt._TUNED.pop(warm, None)
        if previous is not None:
            wt._TUNED[warm] = previous


def test_layer_route_profile_round_trip_and_rejects_unknown_routes():
    routes = {
        ("qk_norm_rope", 32, 4, 128, 128): "fused",
        ("kv_write_pair", 4, 128, 512, True): "separate",
        ("embedding_row", "Q6_K", 4096, 152064, 840): "compact",
    }
    before = {key: wt._TUNED.get(key) for key in routes}
    try:
        wt._TUNED.update(routes)
        profile = wt.kernel_profile()
        for key in routes:
            wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= len(routes)
        assert {key: wt._TUNED[key] for key in routes} == routes

        for key in routes:
            profile["tuned"]["|".join(map(str, key))] = "unmeasured"
            wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert all(key not in wt._TUNED for key in routes)
    finally:
        for key, value in before.items():
            wt._TUNED.pop(key, None)
            if value is not None:
                wt._TUNED[key] = value


def test_full_vocabulary_sampler_profile_round_trip_is_backend_scoped():
    key = ("vocab_sample_full", 151936, "webgpu")
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = "gpu"
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key] == "gpu"
        profile["tuned"]["|".join(map(str, key))] = "unmeasured"
        wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert key not in wt._TUNED
        profile["tuned"]["vocab_sample_full|151936|webgl"] = "gpu"
        wt.use_kernel_profile(profile)
        assert ("vocab_sample_full", 151936, "webgl") not in wt._TUNED
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_attention_profile_round_trip_for_flash_and_gqa():
    flash_key = ("flash_tile", 32, 4, 128)
    gqa_key = (32, 4, 128, 64)
    before_flash = wt._TUNED.get(flash_key)
    before_gqa = wt._GQA_TUNED.get(gqa_key)
    try:
        wt._TUNED[flash_key] = (16, 8)
        wt._GQA_TUNED[gqa_key] = 8
        profile = wt.kernel_profile()
        wt._TUNED.pop(flash_key)
        wt._GQA_TUNED.pop(gqa_key)
        assert wt.use_kernel_profile(profile) >= 2
        assert wt._TUNED[flash_key] == (16, 8)
        assert wt._GQA_TUNED[gqa_key] == 8

        profile["tuned"]["|".join(map(str, flash_key))] = [256, 256]
        profile["gqa_tuned"]["|".join(map(str, gqa_key))] = 3
        wt._TUNED.pop(flash_key)
        wt._GQA_TUNED.pop(gqa_key)
        wt.use_kernel_profile(profile)
        assert flash_key not in wt._TUNED
        assert gqa_key not in wt._GQA_TUNED
    finally:
        wt._TUNED.pop(flash_key, None)
        wt._GQA_TUNED.pop(gqa_key, None)
        if before_flash is not None:
            wt._TUNED[flash_key] = before_flash
        if before_gqa is not None:
            wt._GQA_TUNED[gqa_key] = before_gqa


def test_complete_decode_plan_round_trip_rejects_invalid_or_changed_source():
    model = SimpleNamespace(base="local/model@12345678.gguf", L=48, H=2048,
                            layers=[])
    key = llm.CausalLM._decode_composition_key(model, (), ())
    assert key[0] == "decode_plan_v3"
    assert llm.CausalLM._decode_composition_key(
        model, (), (), backend="webgl")[0] == "decode_plan_v1"
    plan = ["fused", [], "full", "auto", "auto", "auto", "auto", "auto"]
    saved = {"plan": plan, "median_ms": 25.67}
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = saved
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key] == saved

        profile["tuned"]["|".join(key)]["plan"][0] = "invalid"
        wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert key not in wt._TUNED

        model.base = "local/different@87654321.gguf"
        assert llm.CausalLM._decode_composition_key(model, (), ()) != key
        model.base = "local/model@12345678.gguf"
        model.H += 1
        assert llm.CausalLM._decode_composition_key(model, (), ()) != key
        model.H -= 1
        assert llm.CausalLM._decode_composition_key(model, (), (), "device") != key
        legacy = ("decode_plan_v1", "webgpu", key[2])
        profile["tuned"]["|".join(legacy)] = saved
        legacy_sampler_agnostic = ("decode_plan_v2", "webgpu", key[2])
        profile["tuned"]["|".join(legacy_sampler_agnostic)] = saved
        wt.use_kernel_profile(profile)
        assert legacy not in wt._TUNED
        assert legacy_sampler_agnostic not in wt._TUNED
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_complete_decode_projection_shape_profile_round_trip_and_validation():
    model = SimpleNamespace(base="local/weights@12345678.gguf", layers=[])
    key = llm.CausalLM._decode_composition_key(model, (), ())
    old = wt._TUNED.get(key)
    plan = ["fused", [], "full", "fused:compact", "separate",
            "composed", "separate", None, ["balanced", "balanced"]]
    try:
        wt._TUNED[key] = {"plan": plan, "median_ms": 7.4}
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key]["plan"] == plan
        profile["tuned"]["|".join(key)]["plan"][8][1] = "shortk"
        wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert key not in wt._TUNED
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_webgpu_complete_plan_key_tracks_effective_sampler_not_requested_route():
    model = SimpleNamespace(base="local/weights.gguf", L=28, H=1024,
                            VOCAB=151936, layers=[], _gpu=True,
                            _sampling={"temperature": 0.6, "do_sample": True},
                            _sample_execution="js")
    js_key = llm.CausalLM._decode_composition_key(model, (), ())
    webgl_key = llm.CausalLM._decode_composition_key(
        model, (), (), backend="webgl")
    model._sample_execution = "gpu"
    gpu_key = llm.CausalLM._decode_composition_key(model, (), ())
    assert js_key != gpu_key
    assert js_key[0] == gpu_key[0] == "decode_plan_v3"
    assert llm.CausalLM._decode_composition_key(
        model, (), (), backend="webgl") == webgl_key
    # Top-p cannot use the current GPU sampler; an override falls back to JS.
    model._sampling["top_p"] = 0.9
    assert llm.CausalLM._decode_composition_key(model, (), ()) != gpu_key
    model._sample_execution = "js"
    assert llm.CausalLM._decode_composition_key(model, (), ()) == js_key


def test_webgpu_upper_tuner_measures_the_actual_sampling_path():
    source = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert "reset_probe_sampling()" in source
    assert source.count("CausalLM._pick_tensor(self, logits)") >= 2
    assert "_saved_pick.items()" in source


def test_historical_fused_bundle_is_screened_at_the_complete_api(monkeypatch):
    """A winning peer combination must not disappear between axis tournaments."""
    class Platform:
        def beginCapture(self, name):
            pass

        def endCapture(self):
            pass

        def replay(self, name):
            pass

    def block():
        x = wt.GGMLLinear.__new__(wt.GGMLLinear)
        x.type_name = "Q6_K"
        x.Kt = x.Nt = 1024
        x.execution = "auto"
        x.decode_shape = "auto"
        x.decode_execution = None
        return x

    observed = set()
    logits = SimpleNamespace(numpy=lambda: np.array([[0.0, 1.0]], np.float32))
    layer = {key: block() for key in ("q", "k", "v", "o", "down")}
    layer.update(qn=object(), kn=object())
    model = SimpleNamespace(
        base="local/weights.gguf", _gpu=True, layers=[layer], head=[block()],
        _capturable=lambda: True, _stored_linears=lambda: (),
        _is_linear_layer=lambda i: False,
        _reset_linear_state=lambda: None, _set_inputs=lambda *args: None,
        gen_defaults={"temperature": 0.6, "do_sample": True},
    )

    def forward():
        observed.add((model._add_rms_execution, model._qkv_execution,
                      model._qk_norm_rope_execution, model._kv_write_execution,
                      model.head[0].decode_shape))
        return logits

    model._decode_fwd = forward
    monkeypatch.setattr(wt, "_adam_kernel", {"platform": Platform()})
    monkeypatch.setattr(llm, "_load_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm.CausalLM, "_decode_composition_key",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(llm.CausalLM, "_kv_drop", lambda self: None)
    monkeypatch.setattr(llm.CausalLM, "_set_sampling",
                        lambda self, **kwargs: setattr(self, "_seen", []))
    monkeypatch.setattr(llm.CausalLM, "_pick_tensor",
                        lambda self, x: llm.CausalLM._accept_token(self, 1))
    llm.CausalLM._tune_decode_composition(model, interactive=False)
    assert ("fused", "fused", "fused", "fused", "compact") in observed


def test_unused_greedy_chunk_capture_is_unpinned_without_losing_its_choice():
    calls = []
    plat = SimpleNamespace(beginCapture=lambda name: calls.append(("begin", name)),
                           endCapture=lambda: calls.append(("end", None)))
    model = SimpleNamespace(_greedy_chunk_capture_ready=True,
                            _greedy_chunk_size=2)
    llm.CausalLM._release_idle_greedy_capture(model, plat, "full")
    assert calls == [("begin", "decode_chunk"), ("end", None)]
    assert not model._greedy_chunk_capture_ready
    assert model._greedy_chunk_size == 2
    calls.clear()
    llm.CausalLM._release_idle_greedy_capture(model, plat, "full")
    assert calls == []
    model._greedy_chunk_capture_ready = True
    llm.CausalLM._release_idle_greedy_capture(model, plat, "device")
    assert calls == [] and model._greedy_chunk_capture_ready


def test_webgpu_upper_tuner_rejects_same_logits_with_different_sampled_token(monkeypatch):
    """A numerically equal candidate is not correct if real sampling diverges."""
    class Platform:
        timed_routes = []

        def beginCapture(self, name):
            if name == "decode":
                self.timed_routes.append(model._add_rms_execution)

        def endCapture(self):
            pass

        def replay(self, name):
            pass

    logits = SimpleNamespace(numpy=lambda: np.array([[0.0, 1.0]], np.float32))
    model = SimpleNamespace(
        base="local/sampling.gguf", _gpu=True, layers=[], head=[],
        _sampling={"do_sample": True}, _seen=[9], _gen_start=0,
        _capturable=lambda: True, _stored_linears=lambda: (),
        _reset_linear_state=lambda: None, _set_inputs=lambda *args: None,
        _decode_fwd=lambda: logits,
    )
    platform = Platform()
    monkeypatch.setattr(wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm, "_load_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm.CausalLM, "_decode_composition_key",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(llm.CausalLM, "_kv_drop", lambda self: None)

    def set_sampling(self, **kwargs):
        self._sampling = {"do_sample": True}
        self._seen = []
        self._gen_start = 0

    def sample(self, _logits):
        token = 2 if self._add_rms_execution == "fused" else 1
        return llm.CausalLM._accept_token(self, token)

    monkeypatch.setattr(llm.CausalLM, "_set_sampling", set_sampling)
    monkeypatch.setattr(llm.CausalLM, "_pick_tensor", sample)
    before_sampling, before_seen = model._sampling, model._seen
    chosen = llm.CausalLM._tune_decode_composition(model, interactive=False)
    assert chosen["add_rmsnorm"] == "composed"
    assert platform.timed_routes and set(platform.timed_routes) == {"composed"}
    assert model._sampling is before_sampling and model._seen is before_seen


def test_decode_profile_mode_matches_the_real_default_and_call_sampling():
    model = SimpleNamespace(base="local/model.gguf", layers=[])
    assert llm.CausalLM._decode_pick_mode(model) == "full"
    default_key = llm.CausalLM._decode_composition_key(model, (), ())

    model.gen_defaults = {"temperature": 0.0, "do_sample": True}
    assert llm.CausalLM._decode_pick_mode(model) == "device"
    assert llm.CausalLM._decode_composition_key(model, (), ()) != default_key

    model._sampling = {"temperature": 0.6, "do_sample": True}
    assert llm.CausalLM._decode_pick_mode(model) == "full"
    assert llm.CausalLM._decode_composition_key(model, (), ()) == default_key

    model._sampling = {"temperature": 0.0, "do_sample": False}
    assert llm.CausalLM._decode_pick_mode(model) == "device"


def test_call_does_not_reuse_a_plan_measured_for_another_pick_mode(monkeypatch):
    model = SimpleNamespace(
        base="local/model.gguf", layers=[], _gpu=True,
        _capturable=lambda: True, _stored_linears=lambda: (),
        _sampling={"temperature": 0.0, "do_sample": False},
        decode_plan={"pick_mode": "full", "add_rmsnorm": "fused"},
    )
    called = []
    monkeypatch.setattr(llm.CausalLM, "_tune_decode_composition",
                        lambda self, **kw: called.append(kw))
    llm.CausalLM._activate_decode_profile_for_call(model)
    assert called == []
    assert model.decode_plan["mode_profile_missing"] is True
    assert model.decode_plan["active_pick_mode"] == "device"

    key = llm.CausalLM._decode_composition_key(model, (), (), "device")
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = {"plan": ["fused", [], "device", "auto", "auto",
                                   "auto", "auto", "auto"]}
        llm.CausalLM._activate_decode_profile_for_call(model)
        assert called == [{"interactive": True}]
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_return_to_measured_sampling_plan_clears_stale_missing_status():
    model = SimpleNamespace(
        base="local/model.gguf", layers=[], _gpu=True,
        _capturable=lambda: True,
        _sampling={"temperature": 0.6, "do_sample": True},
        decode_plan={"pick_mode": "full", "sampler_route": "js",
                     "active_pick_mode": "device", "active_sampler_route": "none",
                     "mode_profile_missing": True},
    )
    llm.CausalLM._activate_decode_profile_for_call(model)
    assert model.decode_plan["active_pick_mode"] == "full"
    assert model.decode_plan["active_sampler_route"] == "js"
    assert "mode_profile_missing" not in model.decode_plan


def test_webgl_complete_api_tunes_the_default_sampling_path(monkeypatch):
    observed = []
    model = SimpleNamespace(
        base="local/webgl.gguf", _gpu=False, layers=[], head=[],
        L=1, NKV=1, HD=4, lmax=8,
        _reset_linear_state=lambda: None,
    )

    def forward(ids, pos, cache):
        observed.append((model._sampling["do_sample"], model._greedy_execution))
        return pos + 1

    model._kv_forward = forward
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "KVCache", lambda *args: object())
    plan = llm.CausalLM._tune_webgl_decode_composition(model, interactive=False)
    assert plan["pick_mode"] == "full"
    assert observed and all(sample and route == "full"
                            for sample, route in observed)
    assert not hasattr(model, "_sampling")


def test_webgl_interactive_load_reuses_profile_or_exposes_pending(monkeypatch):
    model = SimpleNamespace(
        base="local/webgl@12345678.gguf", _gpu=False, layers=[], head=[],
        L=1, NKV=1, HD=4, lmax=8,
        _kv_forward=lambda *args: (_ for _ in ()).throw(
            AssertionError("interactive load must not run the offline tournament")),
    )
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    key = llm.CausalLM._decode_composition_key(
        model, (), (), backend="webgl")
    old = wt._TUNED.pop(key, None)
    try:
        pending = llm.CausalLM._tune_webgl_decode_composition(model)
        assert pending["profile_pending"] is True
        assert pending["greedy_pick"] == "full"
        wt._TUNED[key] = {"plan": ["fused", "full"], "median_ms": 200.0}
        chosen = llm.CausalLM._tune_webgl_decode_composition(model)
        assert chosen["profile_reused"] is True
        assert chosen["gate_up"] == "fused"
        assert model._gate_up_execution == "fused"
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_webgl_complete_plan_profile_round_trip():
    model = SimpleNamespace(base="local/webgl@12345678.gguf", layers=[])
    key = llm.CausalLM._decode_composition_key(
        model, (), (), backend="webgl")
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = {"plan": ["fused", "full"], "median_ms": 200.0}
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key]["plan"] == ["fused", "full"]
        profile["tuned"]["|".join(key)]["plan"] = ["invalid", "full"]
        wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert key not in wt._TUNED
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_interactive_load_reuses_only_a_matching_complete_plan():
    model = SimpleNamespace(
        base="local/model@12345678.gguf", _gpu=True,
        _capturable=lambda: True, _stored_linears=lambda: (),
        _reset_linear_state=lambda: None,
        _warm_deadline=time.perf_counter() + 5,
        layers=[], head=[], _kv_ids=[1, 2],
    )
    key = llm.CausalLM._decode_composition_key(model, (), ())
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = {
            "plan": ["fused", [], "full", "auto", "auto", "auto", "auto", "auto"],
            "median_ms": 25.67,
        }
        chosen = llm.CausalLM._tune_decode_composition(model)
        assert chosen["profile_reused"] is True
        assert chosen["add_rmsnorm"] == "fused"
        assert chosen["median_ms"] == 25.67
        assert model._add_rms_execution == "fused"
        assert not hasattr(model, "_kv_ids")
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_interactive_complete_plan_restores_group_shapes_without_changing_leaf_auto():
    output = wt.GGMLLinear.__new__(wt.GGMLLinear)
    down = wt.GGMLLinear.__new__(wt.GGMLLinear)
    for block in (output, down):
        block.type_name = "Q6_K"
        block.Kt = block.Nt = 1024
        block.execution = "auto"
        block.decode_execution = None
        block.decode_shape = "auto"
    model = SimpleNamespace(
        base="local/weights@12345678.gguf", _gpu=True,
        _capturable=lambda: True, _stored_linears=lambda: (output, down),
        _reset_linear_state=lambda: None,
        _warm_deadline=time.perf_counter() + 5,
        layers=[{"o": output, "down": down}], head=[], _kv_ids=[1, 2],
    )
    key = llm.CausalLM._decode_composition_key(model, (output, down), ())
    old = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = {
            "plan": ["fused", [], "full", "auto", "auto", "auto", "auto",
                     "auto", ["balanced", "compact"]],
            "median_ms": 7.4,
        }
        selected = llm.CausalLM._tune_decode_composition(model)
        assert selected["profile_reused"] is True
        assert selected["projection_shapes"] == {"o": "balanced", "down": "compact"}
        assert (output.decode_shape, down.decode_shape) == ("balanced", "compact")
        assert (output.execution, down.execution) == ("auto", "auto")
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old


def test_persisted_full_fusion_plan_restores_every_route_after_reload():
    """A saved complete-API route must not degrade to individual leaf auto votes."""
    def block(type_name):
        x = wt.GGMLLinear.__new__(wt.GGMLLinear)
        x.type_name = type_name
        x.Kt = x.Nt = 1024
        x.execution = "auto"
        x.decode_execution = None
        x.decode_shape = "auto"
        return x

    q4 = block("Q4_K")
    head = block("Q6_K")
    model = SimpleNamespace(
        base="local/content-hash.gguf", _gpu=True, layers=[{"o": q4}],
        head=[head], _stored_linears=lambda: (q4, head),
        _capturable=lambda: True, _reset_linear_state=lambda: None,
        _sampling={"temperature": 0.6, "do_sample": True},
        _sample_execution="gpu", _warm_deadline=time.perf_counter() + 5,
    )
    key = llm.CausalLM._decode_composition_key(
        model, (q4, head), ((1024, 1024),))
    old = wt._TUNED.get(key)
    plan = ["fused", ["stored"], "full", "fused", "separate", "fused",
            "fused", "compact", ["auto", "auto"]]
    try:
        wt._TUNED[key] = {"plan": plan, "median_ms": None}
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        chosen = llm.CausalLM._tune_decode_composition(model, interactive=True)
        assert chosen["profile_reused"] is True
        assert chosen["sampler_route"] == "gpu"
        assert (chosen["add_rmsnorm"], chosen["qkv"], chosen["gate_up"],
                chosen["qk_norm_rope"], chosen["kv_write"],
                chosen["head_shape"]) == (
                    "fused", "fused", "separate", "fused", "fused", "compact")
        assert (model._add_rms_execution, model._qkv_execution,
                model._qk_norm_rope_execution, model._kv_write_execution) == (
                    "fused", "fused", "fused", "fused")
        assert q4.decode_execution == "stored"
        assert head.decode_shape == "compact"
    finally:
        wt._TUNED.pop(key, None)
        if old is not None:
            wt._TUNED[key] = old
