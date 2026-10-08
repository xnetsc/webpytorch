"""LoRA adapters at inference: read from PEFT's own files, attached by the module path they were
trained on, applied by every call of the projection they sit on, and refused where this engine
would apply only part of what the adapter does."""
import asyncio
import json
import struct

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import _sdk, adapters, webio
from webtorch.llm import CausalLM


@pytest.fixture
def disk_io(monkeypatch):
    async def read(name, offset=0, length=None):
        with open(name, "rb") as f:
            f.seek(offset)
            return f.read() if length is None else f.read(length)
    monkeypatch.setattr(webio, "_IO", read)
    monkeypatch.setattr(webio, "_local_files", {})
    monkeypatch.setattr(webio, "_local_roots", {})


def _safetensors(path, tensors):
    header, blobs, at = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [at, at + arr.nbytes]}
        blobs.append(arr.tobytes()); at += arr.nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw))); f.write(raw)
        for b in blobs:
            f.write(b)


PEFT_CONFIG = {"peft_type": "LORA", "peft_version": "0.21.0", "r": 2, "lora_alpha": 4,
               "lora_dropout": 0.05, "bias": "none", "use_dora": False, "use_rslora": False,
               "fan_in_fan_out": False, "lora_bias": False, "modules_to_save": None,
               "inference_mode": True, "init_lora_weights": True, "qalora_group_size": 16,
               "megatron_core": "megatron.core", "use_qalora": False, "arrow_config": None,
               "alora_invocation_tokens": None, "target_modules": ["q_proj"],
               "auto_mapping": {"base_model_class": "X"}, "rank_pattern": {},
               "alpha_pattern": {}, "loftq_config": {}}


def test_scale_follows_rank_alpha_rslora_and_per_module_patterns():
    assert adapters.lora_scale({"r": 16, "lora_alpha": 32}, "model.layers.0.mlp.up_proj") == 2.0
    assert adapters.lora_scale({"r": 16, "lora_alpha": 32, "use_rslora": True}, "x") == 8.0
    cfg = {"r": 8, "lora_alpha": 8, "rank_pattern": {"q_proj": 4}, "alpha_pattern": {"q_proj": 16}}
    assert adapters.lora_scale(cfg, "model.layers.1.self_attn.q_proj") == 4.0
    assert adapters.lora_scale(cfg, "model.layers.1.self_attn.k_proj") == 1.0


def test_config_options_that_change_the_forward_pass_are_refused():
    adapters.check_config(PEFT_CONFIG)                       # a plain LoRA, as PEFT writes it
    for key, value in (("use_dora", True), ("alora_invocation_tokens", [1, 2]),
                       ("modules_to_save", ["lm_head"]), ("bias", "all"),
                       ("use_qalora", True), ("some_future_option", {"on": 1})):
        with pytest.raises(NotImplementedError):
            adapters.check_config(dict(PEFT_CONFIG, **{key: value}))
    with pytest.raises(NotImplementedError):
        adapters.check_config(dict(PEFT_CONFIG, peft_type="IA3"))


def test_read_lora_pairs_a_and_b_by_module_path(disk_io, tmp_path):
    A = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    B = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    (tmp_path / "adapter_config.json").write_text(json.dumps(PEFT_CONFIG))
    _safetensors(tmp_path / "adapter_model.safetensors", {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B})
    got = asyncio.run(adapters.read_lora(str(tmp_path)))
    assert list(got) == ["model.layers.0.self_attn.q_proj"]
    a, b, scale = got["model.layers.0.self_attn.q_proj"]
    np.testing.assert_array_equal(a, A)
    np.testing.assert_array_equal(b, B)
    assert scale == 2.0

    _safetensors(tmp_path / "adapter_model.safetensors", {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A})
    with pytest.raises(ValueError, match="only lora_A"):
        asyncio.run(adapters.read_lora(str(tmp_path)))


def test_an_adapter_is_added_to_every_call_and_absent_without_one():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((5, 3)).astype(np.float32)        # (out, in)
    A = rng.standard_normal((2, 3)).astype(np.float32)
    B = rng.standard_normal((5, 2)).astype(np.float32)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    lin = wt.UnquantizedLinear(W)
    base = lin(wt.Tensor(x)).numpy()
    np.testing.assert_allclose(base, x @ W.T, rtol=1e-5, atol=1e-5)
    lin.lora = wt.LoRA(A, B, 0.5)
    want = x @ W.T + 0.5 * (x @ A.T) @ B.T
    np.testing.assert_allclose(lin(wt.Tensor(x)).numpy(), want, rtol=1e-5, atol=1e-5)
    # The fused paths read stored weights directly, so an adapted projection takes its own.
    other = wt.UnquantizedLinear(W)
    a, b = wt.parallel_linear((lin, other), wt.Tensor(x))
    np.testing.assert_allclose(a.numpy(), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.numpy(), base, rtol=1e-5, atol=1e-5)
    lin.lora = None
    np.testing.assert_allclose(lin(wt.Tensor(x)).numpy(), base, rtol=1e-6, atol=1e-6)


class _LinearAttn(object):
    def __init__(self, hk, hv, dk, dv, H):
        self.hk, self.hv, self.dk, self.dv = hk, hv, dk, dv
        self.w = {"qkv": wt.UnquantizedLinear(np.zeros((2 * hk * dk + hv * dv, H), np.float32)),
                  "g": wt.UnquantizedLinear(np.zeros((hv * dv, H), np.float32)),
                  "beta": wt.UnquantizedLinear(np.zeros((hv, H), np.float32)),
                  "o": wt.UnquantizedLinear(np.zeros((H, hv * dv), np.float32))}


def _model(layers):
    m = CausalLM.__new__(CausalLM)
    m.layers = layers
    return m


def test_adapters_land_on_the_projection_their_module_path_names():
    H = 6
    q = wt.UnquantizedLinear(np.zeros((8, H), np.float32))
    m = _model([{"q": q, "linear": None}])
    A = np.ones((2, H), np.float32); B = np.ones((8, 2), np.float32)
    placed = m.attach_adapter({"model.layers.0.self_attn.q_proj": (A, B, 1.0)})
    assert placed == ["model.layers.0.self_attn.q_proj"] and q.lora is not None
    assert CausalLM._adapted(m) == [q.lora]          # what the load's warm-up races
    with pytest.raises(NotImplementedError):
        m.attach_adapter({"model.layers.0.self_attn.rotary": (A, B, 1.0)})
    with pytest.raises(ValueError):
        m.attach_adapter({"model.layers.0.self_attn.q_proj": (A, np.ones((7, 2)), 1.0)})
    with pytest.raises(ValueError):
        m.attach_adapter({"model.layers.3.self_attn.q_proj": (A, B, 1.0)})
    m.detach_adapters()
    assert q.lora is None and CausalLM._adapted(m) == []


def test_value_heads_move_to_the_tiled_order_the_layer_runs_in():
    # A Hugging Face weight groups value heads by key head (head j belongs to key head
    # j // r); the layer runs them tiled (value head v reads key head v % hk). Whatever an
    # adapter adds to HF value head j must land where that head runs.
    hk, hv, dk, dv, H = 2, 4, 3, 2, 5
    r = hv // hk
    la = _LinearAttn(hk, hv, dk, dv, H)
    m = _model([{"linear": la}])
    rank = 1
    qk = 2 * hk * dk
    # B rows tagged with the HF head they belong to (q/k rows tagged -1).
    Bqkv = np.concatenate([np.full((qk, rank), -1.0),
                           np.repeat(np.arange(hv, dtype=np.float32), dv)[:, None]])
    Bg = np.repeat(np.arange(hv, dtype=np.float32), dv)[:, None]
    Bb = np.arange(hv, dtype=np.float32)[:, None]
    Ao = np.repeat(np.arange(hv, dtype=np.float32), dv)[None, :]
    one = np.ones((rank, H), np.float32)
    m.attach_adapter({"model.layers.0.linear_attn.in_proj_qkv": (one, Bqkv, 1.0),
                      "model.layers.0.linear_attn.in_proj_z": (one, Bg, 1.0),
                      "model.layers.0.linear_attn.in_proj_b": (one, Bb, 1.0),
                      "model.layers.0.linear_attn.out_proj": (Ao, np.ones((H, rank)), 1.0)})

    def hf_head_of_tiled(v):
        g, i = v % hk, v // hk            # tiled slot v: key head g, i-th of its group
        return g * r + i

    want = np.array([hf_head_of_tiled(v) for v in range(hv)], np.float32)
    v_rows = la.w["qkv"].lora.Bt.numpy().T[qk:, 0]
    np.testing.assert_array_equal(v_rows.reshape(hv, dv)[:, 0], want)
    np.testing.assert_array_equal(la.w["qkv"].lora.Bt.numpy().T[:qk, 0], -1.0)
    np.testing.assert_array_equal(la.w["g"].lora.Bt.numpy().T[:, 0].reshape(hv, dv)[:, 0], want)
    np.testing.assert_array_equal(la.w["beta"].lora.Bt.numpy().T[:, 0], want)
    np.testing.assert_array_equal(la.w["o"].lora.At.numpy()[:, 0].reshape(hv, dv)[:, 0], want)


def test_an_adapter_beside_the_model_is_found_and_one_in_a_subfolder_is_not(tmp_path):
    (tmp_path / "m.gguf").write_bytes(b"")
    assert adapters.adapter_for(str(tmp_path / "m.gguf"), None, "gguf") is None
    sub = tmp_path / "lora"; sub.mkdir()
    (sub / adapters.ADAPTER_CONFIG).write_text("{}")
    (sub / adapters.ADAPTER_WEIGHTS).write_bytes(b"")
    assert adapters.adapter_for(str(tmp_path / "m.gguf"), None, "gguf") is None
    assert adapters.adapter_for(str(tmp_path / "m.gguf"), str(sub) + "/", "gguf") == str(sub)
    (tmp_path / adapters.ADAPTER_CONFIG).write_text("{}")
    (tmp_path / adapters.ADAPTER_WEIGHTS).write_bytes(b"")
    assert adapters.adapter_for(str(tmp_path / "m.gguf"), None, "gguf") == str(tmp_path)
    assert adapters.adapter_for(str(tmp_path / "m.gguf"), False, "gguf") is None
    assert adapters.adapter_for(str(tmp_path), None, None) == str(tmp_path)
    # A name with no folder, and a served one, are not listed: nothing is probed for.
    assert adapters.adapter_for("m.gguf", None, "gguf") is None
    assert adapters.adapter_for("https://host/org/repo/m.gguf", None, "gguf") is None


def test_a_picked_folder_lists_its_own_files_only(monkeypatch):
    monkeypatch.setattr(webio, "_local_files", {})
    monkeypatch.setattr(webio, "_local_roots", {})
    webio.use_model_file(object(), "pick/m.gguf")
    webio.use_model_file(object(), "pick/lora/adapter_config.json")
    webio.use_model_file(object(), "picked/other.json")
    assert webio.files_under("pick") == ["lora/adapter_config.json", "m.gguf"]
    assert webio.files_under("pick/lora") == ["adapter_config.json"]
    assert webio.files_under("pick/none") == []
    assert webio.files_under("https://host/pick") is None


def test_the_same_weights_with_another_adapter_are_another_model():
    a = _sdk._impl_cache_key("m.gguf", "auto", None, None, "native", None)
    b = _sdk._impl_cache_key("m.gguf", "auto", None, None, "native", "lora")
    assert a != b
    assert "adapter" in _sdk._LLM_OPTS


# ---- the adapter in one dispatch ---------------------------------------------------------------

def test_a_rank_padded_to_four_adds_exactly_nothing():
    rng = np.random.default_rng(1)
    for r in (1, 3, 4, 5):
        A = rng.standard_normal((r, 7)).astype(np.float32)
        B = rng.standard_normal((9, r)).astype(np.float32)
        lo = wt.LoRA(A, B, 0.25)
        assert lo.rank == r and lo.r4 == -(-r // 4) * 4
        assert not lo.At.numpy()[:, r:].any() and not lo.Bt.numpy()[r:].any()
        x = rng.standard_normal((3, 7)).astype(np.float32)
        y = rng.standard_normal((3, 9)).astype(np.float32)
        got = np.asarray(lo.add(wt.xp.asarray(x), wt.xp.asarray(y)), np.float32)
        np.testing.assert_allclose(got, y + 0.25 * (x @ A.T) @ B.T, rtol=1e-5, atol=1e-5)


def test_without_webgpu_the_adapter_is_composed_and_a_fused_request_says_so():
    if wt._adam_backend_ready():
        pytest.skip("host behaviour")
    lo = wt.LoRA(np.ones((2, 4), np.float32), np.ones((3, 2), np.float32), 1.0)
    x, y = wt.xp.asarray(np.ones((1, 4), np.float32)), wt.xp.asarray(np.zeros((1, 3), np.float32))
    np.testing.assert_allclose(np.asarray(lo.add(x, y)), 8.0)
    np.testing.assert_allclose(np.asarray(lo.add(x, y, execution="composed")), 8.0)
    with pytest.raises(RuntimeError):
        lo.add(x, y, execution="fused:256")
    with pytest.raises(ValueError):
        lo.add(x, y, execution="fused")


def test_fused_kernel_source_fills_every_constant_and_fits_workgroup_memory():
    for qp in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for opt in (1, 4, 16):
            src = wt._lora_src(qp, opt)
            for token in ("QPu", "LANESu", "S1u", "OPTu", "WGu", "(WG)"):
                assert token not in src, (qp, opt, token)
            # Two vec4 arrays: the lane partials (one per thread) and t (one per rank group).
            assert "array<vec4<f32>, %du>" % wt._LORA_WG in src
            assert "array<vec4<f32>, %du>" % qp in src
            assert 16 * (wt._LORA_WG + qp) <= 16384
            lanes = wt._LORA_WG // qp
            assert "const LANES: u32 = %du;" % lanes in src
            assert "const S1: u32 = %du;" % min(lanes, 16) in src
    assert set(wt._LORA_ROUTES) == {"composed", "fused:256", "fused:1024", "fused:4096"}
    assert wt._LORA_ROUTES[0] == "composed"          # an unproven race keeps it


def test_shared_projections_add_each_adapter_to_its_own_output():
    rng = np.random.default_rng(2)
    H, F = 6, 5
    Wg, Wu = (rng.standard_normal((F, H)).astype(np.float32) for _ in range(2))
    gate, up = wt.UnquantizedLinear(Wg), wt.UnquantizedLinear(Wu)
    A = rng.standard_normal((3, H)).astype(np.float32)
    B = rng.standard_normal((F, 3)).astype(np.float32)
    up.lora = wt.LoRA(A, B, 2.0)
    x = rng.standard_normal((1, H)).astype(np.float32)
    g = x @ Wg.T
    u = x @ Wu.T + 2.0 * (x @ A.T) @ B.T
    a, b = wt.parallel_linear((gate, up), wt.Tensor(x))
    np.testing.assert_allclose(a.numpy(), g, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.numpy(), u, rtol=1e-5, atol=1e-5)
    # The projection without its adapter is what a shared dispatch computes.
    np.testing.assert_allclose(wt._bare(up, wt.Tensor(x)).numpy(), x @ Wu.T, rtol=1e-5, atol=1e-5)
    act = wt.parallel_swiglu((gate, up), wt.Tensor(x)).numpy()
    np.testing.assert_allclose(act, g / (1.0 + np.exp(-g)) * u, rtol=1e-4, atol=1e-5)


def _q8_0(W):
    """`W` (out, in) as GGUF Q8_0 bytes, and the values those bytes hold."""
    N, K = W.shape
    blocks = W.reshape(N, K // 32, 32).astype(np.float32)
    d = (np.abs(blocks).max(-1) / 127.0).astype(np.float16)
    q = np.round(blocks / np.maximum(d.astype(np.float32), 1e-8)[..., None])
    q = q.clip(-127, 127).astype(np.int8)
    rec = np.zeros((N, K // 32), dtype=[("d", "<f2"), ("q", "i1", (32,))])
    rec["d"], rec["q"] = d, q
    return rec.tobytes(), (d.astype(np.float32)[..., None] * q).reshape(N, K)


def test_the_fused_adapter_matches_numpy_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(3)
    for (M, K, N, r) in ((1, 1024, 2048, 16), (1, 3072, 1024, 64), (2, 37, 300, 3),
                         (3, 513, 777, 1), (1, 96, 48, 128), (1, 2048, 1024, 13),
                         (1, 64, 5000, 1024)):
        A = (rng.standard_normal((r, K)) / np.sqrt(K)).astype(np.float32)
        B = rng.standard_normal((N, r)).astype(np.float32)
        lo = wt.LoRA(A, B, 0.75)
        x = rng.standard_normal((M, K)).astype(np.float32)
        y = rng.standard_normal((M, N)).astype(np.float32)
        want = y + 0.75 * (x @ A.T) @ B.T
        for route in wt._LORA_ROUTES:
            got = np.asarray(lo.add(wt.xp.asarray(x), wt.xp.asarray(y), execution=route).get())
            assert np.abs(got - want).max() <= 1e-5 * max(1.0, np.abs(want).max()), (M, K, N, r,
                                                                                        route)


def test_adapted_projections_keep_their_shared_dispatch_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(4)
    K, Ns = 256, (512, 256, 256)
    lins, refs = [], []
    for N in Ns:
        raw, W = _q8_0(rng.standard_normal((N, K)).astype(np.float32) * 0.1)
        lins.append(wt.GGMLLinear(raw, "Q8_0", K, N, execution="stored"))
        refs.append(W)
    loras = [(rng.standard_normal((8, K)).astype(np.float32) * 0.1,
              rng.standard_normal((N, 8)).astype(np.float32)) for N in Ns]
    lins[0].lora = wt.LoRA(*loras[0], 2.0)
    lins[2].lora = wt.LoRA(*loras[2], 2.0)
    x = rng.standard_normal((1, K)).astype(np.float32)
    want = [x @ W.T for W in refs]
    want[0] = want[0] + 2.0 * (x @ loras[0][0].T) @ loras[0][1].T
    want[2] = want[2] + 2.0 * (x @ loras[2][0].T) @ loras[2][1].T
    for execution in ("separate", "fused"):
        wt._count_dispatch_names(True)
        from wgpy_backends.webgpu.platform import WebGPUPlatform
        before = dict(WebGPUPlatform.by_name)
        got = wt.parallel_linear(lins, wt.Tensor(x), execution=execution)
        names = {k for k, v in WebGPUPlatform.by_name.items() if v != before.get(k, 0)}
        wt._count_dispatch_names(False)
        for g, w in zip(got, want):
            np.testing.assert_allclose(g.numpy(), w, rtol=2e-4, atol=2e-4 * np.abs(w).max())
        shared = any(n.startswith("ggml_parallel3") for n in names)
        assert shared == (execution == "fused"), names
