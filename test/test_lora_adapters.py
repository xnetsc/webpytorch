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
    with pytest.raises(NotImplementedError):
        m.attach_adapter({"model.layers.0.self_attn.rotary": (A, B, 1.0)})
    with pytest.raises(ValueError):
        m.attach_adapter({"model.layers.0.self_attn.q_proj": (A, np.ones((7, 2)), 1.0)})
    with pytest.raises(ValueError):
        m.attach_adapter({"model.layers.3.self_attn.q_proj": (A, B, 1.0)})
    m.detach_adapters()
    assert q.lora is None


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
