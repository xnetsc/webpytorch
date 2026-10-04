"""An idle-pool trim must have the same safe contract on both GPU backends."""

import sys
import types

from webtorch import _core as wt


def test_idle_pool_trim_routes_to_each_backend_without_releasing_live_buffers(monkeypatch):
    calls = []
    gpu_module = types.ModuleType("wgpy_backends.webgpu.webgpu_buffer")
    gl_module = types.ModuleType("wgpy_backends.webgl.webgl_buffer")
    gpu_module.release_pooled_buffers = lambda: calls.append("webgpu_pool")
    gl_module.release_pooled_buffers = lambda: calls.append("webgl_pool")
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)
    monkeypatch.setitem(sys.modules, gl_module.__name__, gl_module)

    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    wt._gpu_release_idle_pool()
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    wt._gpu_release_idle_pool()
    assert calls == ["webgpu_pool", "webgl_pool"]


def test_transfer_window_release_routes_to_each_backend(monkeypatch):
    calls = []
    gpu_module = types.ModuleType("wgpy_backends.webgpu.webgpu_buffer")
    gl_module = types.ModuleType("wgpy_backends.webgl.webgl_buffer")
    gpu_module.release_comm_buffer = lambda: calls.append("webgpu_comm")
    gl_module.release_comm_buffer = lambda: calls.append("webgl_comm")
    monkeypatch.setitem(sys.modules, gpu_module.__name__, gpu_module)
    monkeypatch.setitem(sys.modules, gl_module.__name__, gl_module)

    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    wt._release_transfer_memory()
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    wt._release_transfer_memory()
    assert calls == ["webgpu_comm", "webgl_comm"]
