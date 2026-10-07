"""No kernel runs on a device that cannot run it, and nothing assumes a capability.

Every WGSL feature, language feature and limit a kernel needs is read from its source and
checked against what the device reported -- or WebGPU's guaranteed minimum where it reported
nothing -- when the kernel is registered (`KernelUnsupported`). A race leaves such a
candidate out; a choice made without a race (the thread shape, the flash tile, the greedy
chunk path, the packed-dot route) asks first.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from types import SimpleNamespace

from webtorch import _core as wt

ROOT = Path(__file__).resolve().parents[1]


def _sources():
    """Real kernels from the generators, with the bindings they are registered with."""
    out = {
        "ggml_q4k_narrow": (wt._ggml_src("Q4_K", 1, wt._cfg_for("narrow", 256)),
                            wt._ggml_binds("Q4_K", False)),
        "ggml_q4k_default": (wt._ggml_src("Q4_K", 1, wt._cfg_for(None, 256)),
                             wt._ggml_binds("Q4_K", False)),
        "mm_half": (wt._mm_half_src(), ["read-only-storage"] * 2 + ["storage", "read-only-storage"]),
        "attn": (wt._attn_src(64, 4, 4), ["read-only-storage"] * 3 + ["storage", "read-only-storage"]),
        "attn_packed": (wt._attn_src(64, 4, 4, packed=True),
                        ["read-only-storage"] * 3 + ["storage", "read-only-storage"]),
        "q4k_dp4a": (wt._Q4K_DP4A_WGSL, ["read-only-storage"] * 2 + ["storage", "read-only-storage"]),
        "q6k_decode_input": (wt._Q6K_DECODE_INPUT_WGSL,
                             ["read-only-storage"] * 4 + ["storage"] * 4 + ["read-only-storage"]),
    }
    return {k: {"source": s, "bindingTypes": b} for k, (s, b) in out.items()}


def _check(info):
    """Run the platform's check in a child process (it imports Pyodide's `js`)."""
    script = r'''
import json, sys, types
sys.modules["js"] = types.SimpleNamespace(gpu=None)
from wgpy_backends.webgpu.platform import unsupported_reason
ks, info = json.loads(sys.stdin.read())
print(json.dumps({k: unsupported_reason(v["source"], v["bindingTypes"], info) for k, v in ks.items()}))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "webgpu")
    r = subprocess.run([sys.executable, "-c", script], input=json.dumps([_sources(), info]),
                       cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


def test_a_device_with_only_the_guarantees_is_offered_only_what_fits_it():
    got = _check({})
    assert got["ggml_q4k_default"] is None and got["attn"] is None and got["attn_packed"] is None
    assert "shader-f16" in got["mm_half"]
    assert "packed_4x8_integer_dot_product" in got["q4k_dp4a"]
    assert "workgroup memory" in got["ggml_q4k_narrow"]              # 17408 > 16384
    assert "9 storage buffers" in got["q6k_decode_input"]


def test_a_device_reporting_more_runs_all_of_them():
    got = _check({"f16": True, "subgroups": True, "wgsl": ["packed_4x8_integer_dot_product"],
                  "maxStorageBuffers": 10, "maxWorkgroupStorage": 32768,
                  "maxInvocations": 1024})
    assert all(v is None for v in got.values()), got


def test_unreported_limits_are_the_guaranteed_minimum_not_unlimited(monkeypatch):
    monkeypatch.setattr(wt, "gpu_features", lambda: {})
    assert wt._device_limit("maxStorageBuffers") == 8
    assert wt._device_limit("maxWorkgroupStorage") == 16384
    assert not wt._wgsl_feature("packed_4x8_integer_dot_product")
    assert not wt._flash_fits(16, 16, 128)                           # 25.2 KB
    monkeypatch.setattr(wt, "gpu_features", lambda: {"maxWorkgroupStorage": 32768})
    assert wt._flash_fits(16, 16, 128)


def test_a_race_leaves_out_what_the_device_cannot_run_and_still_fails_on_a_bug(monkeypatch):
    class KernelUnsupported(RuntimeError):
        pass
    monkeypatch.setattr(wt, "_TUNED", {})
    monkeypatch.setattr(wt, "_sync_small", lambda a: None)

    def run(which):
        if which == "fancy":
            raise KernelUnsupported("fancy needs shader-f16, which this device does not have")
        return which
    assert wt._weight_execution("unit", "f32", 8, 8, 4, run,
                                candidates=("plain", "fancy")) == "plain"

    def broken(which):
        if which == "fancy":
            raise ValueError("wrong answer")
        return which
    monkeypatch.setattr(wt, "_TUNED", {})
    with pytest.raises(RuntimeError, match="execution candidate 'fancy' failed"):
        wt._weight_execution("unit", "f32", 8, 16, 4, broken, candidates=("plain", "fancy"))


def test_an_approximation_that_misses_its_bound_is_left_out_and_an_exact_one_raises(
        monkeypatch):
    monkeypatch.setattr(wt, "_TUNED", {})
    monkeypatch.setattr(wt, "_GATE_EXCLUDED", [])
    monkeypatch.setattr(wt, "_sync_small", lambda a: None)
    run = lambda which: which
    # The half route's bound is what admits it: missing it, it is not in the race.
    assert wt._weight_execution("unit", "f32", 8, 8, 4, run, candidates=("plain", "half"),
                                check=lambda w: w != "half", approximate=("half",)) == "plain"
    assert wt._GATE_EXCLUDED == [(("weight_exec", "unit", "f32", 8, 8, 4), "half")]
    # A route that computes the same values and does not match them is a bug.
    with pytest.raises(RuntimeError, match="execution candidate 'tiled' failed"):
        wt._weight_execution("unit", "f32", 8, 16, 4, run, candidates=("plain", "tiled"),
                             check=lambda w: w != "tiled", approximate=("half",))


def test_an_approximation_is_admitted_per_weight_on_inputs_with_outliers(monkeypatch):
    calls = []

    def matmul(xf, packed, type_name, K, N, execution="stored", **kw):
        calls.append(execution)
        out = np.ones((xf.shape[0], N), np.float32)
        if execution == "tiled_half":
            out[0, 0] = 1.02                     # 2% of the output scale: past a 1% bound
        return SimpleNamespace(get=lambda: out)
    monkeypatch.setattr(wt, "ggml_matmul", matmul)
    monkeypatch.setattr(wt, "Tensor", lambda x: SimpleNamespace(data=x))
    monkeypatch.setattr(wt, "_APPROX_HOLDS", {})
    assert not wt._approx_holds("tiled_half", "Q4_K", 64, 8, object(), 1e-2)
    assert wt._APPROX_HOLDS[("tiled_half", "Q4_K", 64, 8)] == pytest.approx(0.02, rel=1e-3)
    n = len(calls)
    assert not wt._approx_holds("tiled_half", "Q4_K", 64, 8, object(), 1e-2)
    assert len(calls) == n                      # once per weight


def test_choices_made_without_a_race_ask_the_device_first(monkeypatch):
    import inspect
    from webtorch.llm import CausalLM
    assert 'wt._device_limit("maxStorageBuffers") >= 9' in inspect.getsource(
        CausalLM._can_chunk_greedy)
    src = inspect.getsource(wt.ggml_matmul)
    assert '_wgsl_feature("packed_4x8_integer_dot_product")' in src
    # The thread shape falls back to the default where the narrow one does not fit.
    monkeypatch.setattr(wt, "_ggml_variant_ok", lambda *a, **k: False)
    assert wt._auto_kind("Q4_K", 256, 4096) is None
    monkeypatch.setattr(wt, "_ggml_variant_ok", lambda *a, **k: True)
    assert wt._auto_kind("Q4_K", 256, 4096) == "narrow"
