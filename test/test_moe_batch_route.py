"""The MoE routing primitive keeps its API identical across GPU backends."""

from types import SimpleNamespace
import time

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import lm_engine
from webtorch import llm


class _Array:
    def __init__(self, shape, bid):
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        self.buffer = SimpleNamespace(buffer_id=bid)


def test_webgpu_router_dispatches_one_workgroup_per_input_row(monkeypatch):
    calls = []
    platform = SimpleNamespace(
        addKernel=lambda name, spec: calls.append(("add", name, spec)),
        runKernel=lambda desc: calls.append(("run", desc)),
    )
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    monkeypatch.setattr(wt, "_adam_kernel", {
        "platform": platform,
        "make_meta": lambda values, layout: (
            calls.append(("meta", values, layout)) or SimpleNamespace(buffer_id=4)),
    })
    monkeypatch.setitem(wt._moe_r, "added", False)
    wt.moe_route(_Array((98, 128), 1), _Array((98 * 8,), 2),
                 _Array((98 * 8,), 3), 128, 8)
    assert calls[0][0:2] == ("add", "moe_route")
    assert calls[1] == ("meta", (128, 8, 1, 98), "u4,u4,u4,u4")
    assert calls[2][1]["workGroups"] == {"x": 98, "y": 1, "z": 1}
    assert calls[2][1]["tensors"] == [1, 2, 3, 4]
    assert "lg[base + e]" in wt._MOE_ROUTE_WGSL
    assert "eidx[out_base + s]" in wt._MOE_ROUTE_WGSL
    assert "picked_den" in wt._MOE_ROUTE_WGSL


def test_webgl_router_addresses_every_row_and_slot(monkeypatch):
    calls = []
    monkeypatch.setattr(wt, "_gl_run", lambda *args: calls.append(args))
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    wt.moe_route(_Array((3, 16), 1), _Array((6,), 2), _Array((6,), 3),
                 16, 2, norm=False)
    assert calls[0][0] == "moe_idx_gl"
    assert ("u_T", 3) in calls[0][4]
    assert ("u_T", 3) in calls[1][4]
    assert ("u_norm", 0) in calls[1][4]
    assert "int base = row * u_ne" in wt._MOE_IDX_GL
    assert "If(row * u_k + j)" in wt._MOE_W_GL
    assert "den_sel" in wt._MOE_W_GL
    # `flat` is a GLSL interpolation qualifier, not a legal local variable name. This
    # compiled to no kernel and silently returned zero-valued expert indices on WebGL.
    assert "int flat =" not in wt._MOE_IDX_GL
    assert "int flat =" not in wt._MOE_W_GL


@pytest.mark.parametrize("logits,indices,weights", [
    (_Array((2, 8), 1), _Array((7,), 2), _Array((8,), 3)),
    (_Array((2, 8), 1), _Array((8,), 2), _Array((7,), 3)),
    (_Array((2, 7), 1), _Array((8,), 2), _Array((8,), 3)),
])
def test_router_rejects_incompatible_batch_buffers(logits, indices, weights):
    with pytest.raises(ValueError):
        wt.moe_route(logits, indices, weights, 8, 4)


def test_batched_moe_layer_keeps_router_scores_on_device(monkeypatch):
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, np.float32)
            self.shape = self.data.shape

        def reshape(self, *shape):
            return Tensor(self.data.reshape(*shape))

        def sum(self, axis):
            return Tensor(self.data.sum(axis=axis))

        def __mul__(self, other):
            return Tensor(self.data * other.data)

    class RouterScores(Tensor):
        def numpy(self):
            raise AssertionError("batched routing crossed back to the CPU")

    def route(scores, indices, weights, ne, k, norm):
        assert scores.shape == (3, 4)
        assert indices.shape == weights.shape == (6,)
        indices[:] = np.tile([1, 3], 3)
        weights[:] = np.tile([0.6, 0.4], 3)

    monkeypatch.setattr(wt, "Tensor", Tensor)
    monkeypatch.setattr(wt, "_contig", lambda data: data)
    monkeypatch.setattr(wt, "_empty_i32", lambda shape: np.zeros(shape, np.int32))
    monkeypatch.setattr(wt, "_empty", lambda shape: np.zeros(shape, np.float32))
    monkeypatch.setattr(wt, "moe_route", route)
    repeated = []

    def repeat_rows(data, k, execution="auto"):
        repeated.append((k, execution))
        return np.repeat(data, k, axis=0)

    monkeypatch.setattr(wt, "repeat_rows", repeat_rows)
    monkeypatch.setattr(wt, "swiglu", lambda gate, up=None: gate)
    projection = SimpleNamespace(forward=lambda x, eidx: x)
    moe = {
        "gate": lambda x: RouterScores(np.ones((3, 4), np.float32)),
        "top_k": 2,
        "stacked": {"gate": SimpleNamespace(n_experts=4),
                    "gate_up": projection, "down": projection},
    }
    result = lm_engine.moe_mlp(SimpleNamespace(
        H=4, _moe_prefill_execution="auto",
        _moe_prefill_api_choice=lambda rows: "device"),
                               {"moe": moe},
                               Tensor(np.ones((3, 4), np.float32)))
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result.data, 1.0)
    assert repeated == [(2, "device")]


def test_one_row_moe_route_uses_fully_overwritten_device_buffers(monkeypatch):
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data, np.float32)
            self.shape = self.data.shape

    allocations = []
    monkeypatch.setattr(wt, "Tensor", Tensor)
    monkeypatch.setattr(wt, "_empty_i32", lambda shape: np.full(shape, -1, np.int32))

    def empty(shape):
        allocations.append(shape)
        return np.full(shape, np.nan, np.float32)

    def route(logits, indices, weights, ne, k, norm):
        assert indices.tolist() == [-1, -1]
        assert np.isnan(weights).all()
        indices[:] = [0, 1]
        weights[:] = [0.75, 0.25]

    monkeypatch.setattr(wt, "_empty", empty)
    monkeypatch.setattr(wt, "moe_route", route)
    monkeypatch.setattr(wt, "moe_weighted_sum", lambda values, weights, k,
                        execution="auto": Tensor(
                            (values.data * weights.data[:, None]).sum(0, keepdims=True)))
    monkeypatch.setattr(lm_engine, "_swiglu", lambda value: value)
    gate_up = SimpleNamespace(forward=lambda x, indices: Tensor(
        np.repeat(x.data, len(indices), axis=0)))
    down = SimpleNamespace(forward=lambda x, indices: x)
    moe = {"gate": lambda x: Tensor(np.array([[0.75, 0.25]], np.float32)),
           "top_k": 2, "stacked": {"gate": SimpleNamespace(n_experts=2),
                                    "gate_up": gate_up, "down": down}}
    x = Tensor([[2.0, 4.0]])
    result = lm_engine.moe_mlp(SimpleNamespace(), {"moe": moe}, x)
    np.testing.assert_allclose(result.data, x.data)
    assert allocations == [(2,)]
    assert moe["_gpu_route"]["ew"].data.tolist() == [0.75, 0.25]


def test_moe_prefill_route_profile_round_trip_and_rejects_invalid_policy():
    key = ("moe_prefill_route", "webgpu", 128, 8, 64, "Q3_K", "Q4_K", "cold")
    before = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = "device"
        profile = wt.kernel_profile()
        assert profile["tuned"]["|".join(map(str, key))] == "device"
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key] == "device"
        profile["tuned"]["|".join(map(str, key))] = "unmeasured"
        wt._TUNED.pop(key)
        wt.use_kernel_profile(profile)
        assert key not in wt._TUNED
    finally:
        if before is None:
            wt._TUNED.pop(key, None)
        else:
            wt._TUNED[key] = before


def test_fused_moe_reduction_dispatches_one_output_per_token_channel(monkeypatch):
    calls = []

    class Tensor:
        def __init__(self, data):
            self.data = data

    platform = SimpleNamespace(
        addKernel=lambda name, spec: calls.append(("add", name, spec)),
        runKernel=lambda desc: calls.append(("run", desc)),
    )
    monkeypatch.setattr(wt, "Tensor", Tensor)
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "_empty", lambda shape: _Array(shape, 3))
    monkeypatch.setattr(wt, "_adam_kernel", {
        "platform": platform,
        "make_meta": lambda values, layout: (
            calls.append(("meta", values, layout)) or SimpleNamespace(buffer_id=4)),
    })
    monkeypatch.setitem(wt._moe_reduce_added, "webgpu", False)
    out = wt.moe_weighted_sum(Tensor(_Array((16, 32), 1)),
                              Tensor(_Array((16,), 2)), 8, execution="fused")
    assert out.data.shape == (2, 32)
    assert calls[0][0:2] == ("add", "moe_weighted_sum")
    assert calls[1] == ("meta", (2, 8, 32, 0), "u4,u4,u4,u4")
    assert calls[2][1]["workGroups"] == {"x": 1, "y": 1, "z": 1}
    assert calls[2][1]["tensors"] == [1, 2, 3, 4]
    assert "Yf(p * u_h + col) * Wf(p)" in wt._MOE_REDUCE_GL


@pytest.mark.parametrize("shape,weights,k", [
    ((7, 32), 7, 8), ((8, 32), 7, 8), ((8, 32), 8, 0),
])
def test_moe_reduction_rejects_invalid_shapes(shape, weights, k):
    with pytest.raises(ValueError):
        wt.moe_weighted_sum(_Array(shape, 1), _Array((weights,), 2), k)


def test_moe_reduction_choice_round_trips_through_device_profile():
    key = ("moe_reduce", "webgl", 8, 2048, 128)
    before = wt._TUNED.get(key)
    try:
        wt._TUNED[key] = "fused"
        profile = wt.kernel_profile()
        wt._TUNED.pop(key)
        assert wt.use_kernel_profile(profile) >= 1
        assert wt._TUNED[key] == "fused"
    finally:
        if before is None:
            wt._TUNED.pop(key, None)
        else:
            wt._TUNED[key] = before


@pytest.mark.parametrize("seconds_left", [-1, 60])
def test_interactive_warm_budget_never_starts_whole_model_oracle(seconds_left):
    model = SimpleNamespace(
        _gpu=True, _capturable=lambda: True, _stored_linears=lambda: (),
        _reset_linear_state=lambda: None,
        _warm_deadline=time.perf_counter() + seconds_left,
        _kv_ids=[1, 2, 3], _gcache=object(),
        layers=[], head=[],
    )
    plan = llm.CausalLM._tune_decode_composition(model)
    assert plan["budget_limited"] is True
    assert plan["tested_candidates"] == 0
    assert plan["greedy_pick"] == "full"
    assert not hasattr(model, "_kv_ids")
    assert not hasattr(model, "_gcache")
