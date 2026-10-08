"""Optional kernels invalidate their own measurements, not unrelated model routes."""
from types import SimpleNamespace

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch.llm import CausalLM


@pytest.fixture
def profile_state(monkeypatch):
    for name in ("_KBUILD", "_TUNED", "_GQA_TUNED", "_CHECKED", "_DEQ_OK",
                 "_ROUTE_ALT", "_NEAREST", "_REMEASURED_AT"):
        monkeypatch.setattr(wt, name, {})
    monkeypatch.setattr(wt, "_CALIBRATED", set())
    monkeypatch.setattr(wt, "_ROUTE_IDS", {"in_use": 0, "unused": None})
    monkeypatch.setattr(wt, "_REMEASURE_SEQ", [0])


def test_adding_optional_shader_preserves_pre_extension_stamp(profile_state, monkeypatch):
    base = wt._kernel_build()
    full = wt._kernel_digest(True)
    assert full != base
    # With the optional template absent this is the original all-shader algorithm.
    monkeypatch.setattr(wt, "_LORA_WGSL", "")
    wt._KBUILD.clear()
    assert wt._kernel_digest(True) == base
    assert wt._kernel_build() == base


def test_extension_edit_keeps_shared_routes_but_changes_own_keys(profile_state, monkeypatch):
    base = wt._kernel_build()
    extension = wt._lora_build()
    plain = ("weight_exec", "ggml", "Q8_0", 64, 32, 1)
    adapted = ("weight_exec", "lora", "r16:" + extension, 64, 32, 1)
    wt._TUNED.update({plain: "stored", adapted: "fused:256"})
    saved = wt.kernel_profile()
    monkeypatch.setattr(wt, "_LORA_WGSL", wt._LORA_WGSL + "\n// changed extension\n")
    wt._KBUILD.clear()
    wt._TUNED.clear()
    assert wt._kernel_build() == base
    assert wt._lora_build() != extension
    assert wt.use_kernel_profile(saved) >= 1
    assert wt._TUNED[plain] == "stored"
    assert ("weight_exec", "lora", "r16:" + wt._lora_build(), 64, 32, 1) not in wt._TUNED


def test_only_exact_legacy_build_migrates(profile_state, monkeypatch):
    key = ("weight_exec", "ggml", "Q8_0", 64, 32, 1)
    saved = {"build": wt._kernel_digest(True), "tuned": {"|".join(map(str, key)): "stored"}}
    assert wt.use_kernel_profile(saved) == 1
    assert wt._TUNED[key] == "stored"
    shared = wt.kernel_profile()
    monkeypatch.setattr(wt, "_GEGLU_WGSL", wt._GEGLU_WGSL + "\n// changed shared kernel\n")
    wt._KBUILD.clear()
    wt._TUNED.clear()
    assert wt.use_kernel_profile(saved) == 0
    assert wt.use_kernel_profile(shared) == 0
    assert wt.use_kernel_profile({"build": "not-this-build", "tuned": saved["tuned"]}) == 0
    assert not wt._TUNED


@pytest.mark.parametrize("backend", ["webgpu", "webgl"])
def test_upper_adapter_plan_depends_on_extension_but_plain_plan_does_not(
        profile_state, monkeypatch, backend):
    model = SimpleNamespace(base="local/content", _gpu=backend == "webgpu")
    plain = SimpleNamespace(type_name="Q8_0", Kt=64, Nt=32, lora=None)
    adapted = SimpleNamespace(type_name="Q8_0", Kt=64, Nt=32,
                              lora=SimpleNamespace(rank=16))
    def key(layer):
        return CausalLM._decode_composition_key(model, [layer], [], "device", backend)
    p, a = key(plain), key(adapted)
    monkeypatch.setattr(wt, "_LORA_WGSL", wt._LORA_WGSL + "\n// changed adapter\n")
    wt._KBUILD.clear()
    assert key(plain) == p
    assert key(adapted) != a


def test_adapter_leaf_race_uses_extension_stamp(profile_state, monkeypatch):
    lo = wt.LoRA(np.ones((2, 4), np.float32), np.ones((3, 2), np.float32), 1)
    calls = []
    def race(kind, dtype, *args, **kw):
        calls.append((kind, dtype))
        return "composed"
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_weight_execution", race)
    monkeypatch.setattr(wt.LoRA, "_run", lambda self, which, x, y: y)
    x, y = np.zeros((1, 4), np.float32), np.zeros((1, 3), np.float32)
    lo.add(x, y)
    assert calls[-1] == ("lora", "r4:" + wt._lora_build())
    monkeypatch.setattr(wt, "_LORA_WGSL", wt._LORA_WGSL + "\n// changed adapter\n")
    wt._KBUILD.clear()
    lo.add(x, y)
    assert calls[0] != calls[1]


@pytest.mark.parametrize("backend", ["webgpu", "webgl"])
def test_non_gguf_adapters_also_invalidate_upper_plan(profile_state, monkeypatch, backend):
    model = object.__new__(CausalLM)
    model.base = "local/non-gguf-content"
    model._gpu = backend == "webgpu"
    model.layers = [{"q": SimpleNamespace(lora=SimpleNamespace(k=64, n=32, rank=16))}]
    # No GGML leaves: this is precisely why checking only `linears` is insufficient.
    before = model._decode_composition_key([], [], "device", backend)
    monkeypatch.setattr(wt, "_LORA_WGSL", wt._LORA_WGSL + "\n// changed adapter\n")
    wt._KBUILD.clear()
    assert model._decode_composition_key([], [], "device", backend) != before


def test_existing_base_profile_is_not_discarded_for_optional_shader(profile_state, monkeypatch):
    # Runs unchanged against the pre-fix implementation: it rejected this at import.
    wt._TUNED[("weight_exec", "ggml", "Q8_0", 64, 32, 1)] = "stored"
    saved = wt.kernel_profile()
    monkeypatch.setattr(wt, "_LORA_WGSL", wt._LORA_WGSL + "\n// optional-only edit\n")
    wt._KBUILD.clear()
    wt._TUNED.clear()
    assert wt.use_kernel_profile(saved) > 0
