import inspect
from pathlib import Path

from webtorch import parity_contracts
from webtorch import _core as wt
from webtorch import llm


def test_every_public_backend_capability_has_a_complete_path():
    required = {
        "tensor_autograd", "dense_linear", "ggml_stored_linear", "gptq_stored_linear",
        "moe_linear", "attention", "kv_cache", "generation", "gdn_linear_attention",
        "conv_training", "decision_features", "onnx_model", "model_io_cache_progress",
    }
    contracts = parity_contracts()
    assert set(contracts) == required
    for contract in contracts.values():
        assert contract["common_level"] in {"atomic", "decoder", "operator", "attention",
                                             "layer", "internal", "api"}
        assert contract["webgpu"] and contract["webgl"]
        assert "missing" not in contract["webgpu"].lower()
        assert "missing" not in contract["webgl"].lower()


def test_all_accepted_ggml_formats_generate_both_backend_programs():
    for name in wt._GGML_TYPES:
        for mode in (1, 2, 0):
            assert "@compute" in wt._ggml_src(name, mode, mrow=(1 if mode == 0 else None))
        # WebGL independently routes one row, two rows, small batches and large batches.
        # Generate every production variant so an optimisation can never exist on paper
        # while its peer backend fails at shader translation.
        for mode in (1, 2, 3, 0):
            assert "#version 300 es" in wt._ggml_src_gl(name, False, False, mode=mode)


def test_gptq_int4_int8_generate_both_backend_programs():
    for bits in (4, 8):
        assert "@compute" in wt._gptq_src(wt._GPTQ_WGSL, bits, 32)
        assert "#version 300 es" in wt._gptq_src(wt._GL_GPTQ, bits, 32)


def test_model_smoke_uses_each_backends_complete_cache_path():
    src = inspect.getsource(llm.CausalLM._smoke)
    assert "if self._capturable()" in src
    assert "self._prefill(ids)" in src
    assert "wt.KVCache(" in src
    assert "self._kv_forward(ids, 0, cache)" in src


def test_webgl_native_bmm_indexes_every_batch_not_only_the_first():
    source = (Path(__file__).parents[1] / "webgl" / "wgpy_backends" / "webgl"
              / "webgl_array_func.py").read_text()
    bmm = source.split("    def batched_matmul(", 1)[1].split("    def matmul(", 1)[0]
    assert "int b = flat_idx / (M * N);" in bmm
    assert "get_lhs(b, oi, k) * get_rhs(b, k, oj)" in bmm
    assert "if (oi >= M)" not in bmm
