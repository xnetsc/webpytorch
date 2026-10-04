"""Repeated MoE activation rows stay value-identical on both backend paths."""

from types import SimpleNamespace

import numpy as np
import pytest

from webtorch import _core as wt


def test_repeat_rows_cpu_contract(monkeypatch):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(wt.repeat_rows(x, 3), np.repeat(x, 3, axis=0))
    assert wt.repeat_rows(x, 1) is x
    with pytest.raises(ValueError, match="2-D"):
        wt.repeat_rows(x[0], 2)
    with pytest.raises(ValueError, match="positive integer"):
        wt.repeat_rows(x, 0)
    with pytest.raises(TypeError, match="GPU-backed"):
        wt.repeat_rows(x, 2, execution="device")


def test_repeat_rows_webgl_dispatch_matches_original_values(monkeypatch):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "_contig", lambda data: data)
    monkeypatch.setattr(wt, "_empty", lambda shape: np.empty(shape, np.float32))
    calls = []

    def run(name, source, inputs, out, uniforms):
        calls.append((name, source, uniforms))
        out[:] = np.repeat(inputs[0][1], 2, axis=0)
        return out

    monkeypatch.setattr(wt, "_gl_run", run)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(wt.repeat_rows(x, 2, execution="device"),
                                  np.repeat(x, 2, axis=0))
    assert calls[0][0] == "repeat_rows_gl"
    assert dict(calls[0][2]) == {"u_H": 4, "u_k": 2, "u_n": 24}


def test_repeat_rows_webgpu_dispatch_uses_one_output_write(monkeypatch):
    calls = []
    platform = SimpleNamespace(
        addKernel=lambda name, spec: calls.append(("add", name, spec)),
        runKernel=lambda spec: calls.append(("run", spec)))
    meta = SimpleNamespace(buffer_id=3)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_contig", lambda data: data)
    monkeypatch.setattr(wt, "_adam_kernel", {
        "platform": platform,
        "make_meta": lambda values, fmt: calls.append(("meta", values, fmt)) or meta,
    })
    monkeypatch.setattr(wt, "_repeat_rows_k", {"gpu": False})
    x = SimpleNamespace(shape=(3, 4), dtype=np.dtype("float32"),
                        buffer=SimpleNamespace(buffer_id=1))
    out = SimpleNamespace(shape=(6, 4), buffer=SimpleNamespace(buffer_id=2))
    monkeypatch.setattr(wt, "_empty", lambda shape: out)

    assert wt.repeat_rows(x, 2, execution="device") is out
    assert calls[0][0:2] == ("add", "repeat_rows")
    assert "(row / rm.k)" in calls[0][2]["source"]
    assert calls[1] == ("meta", (4, 2, 24, 0), "u4,u4,u4,u4")
    assert calls[2][1]["tensors"] == [1, 2, 3]


def test_repeat_rows_shape_profile_is_device_local_and_round_trips(monkeypatch):
    monkeypatch.setattr(wt, "_kernel_build", lambda: "repeat-test")
    monkeypatch.setattr(wt, "_TUNED", {})
    profile = {"build": "repeat-test", "tuned": {
        "weight_exec|repeat_rows|webgl|4|2|4": "device",
    }}
    assert wt.use_kernel_profile(profile) == 1
    assert wt._TUNED[("weight_exec", "repeat_rows", "webgl", 4, 2, 4)] == "device"
