import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from webtorch import parity_contracts
from webtorch import _core as wt
from webtorch import llm
from webtorch import onnxrt


def test_every_public_backend_capability_has_a_complete_path():
    required = {
        "tensor_autograd", "dense_linear", "ggml_stored_linear", "ggml_phase2_linear",
        "gptq_stored_linear",
        "moe_linear", "attention", "kv_cache", "generation", "gdn_linear_attention",
        "qk_norm_rope", "transformer_layer_boundary", "parallel_projection",
        "parallel_mlp_activation",
        "conv_training",
        "decision_features", "onnx_model", "model_io_cache_progress",
    }
    contracts = parity_contracts()
    assert set(contracts) == required
    for contract in contracts.values():
        assert contract["common_level"] in {"atomic", "decoder", "operator", "attention",
                                             "layer", "internal", "api"}
        assert contract["webgpu"] and contract["webgl"]
        assert "missing" not in contract["webgpu"].lower()
        assert "missing" not in contract["webgl"].lower()


def test_quantized_onnx_contract_is_truthful_and_format_driven():
    contract = parity_contracts()["onnx_model"]
    assert contract["common_level"] == "api"
    assert "shared graph interpreter" in contract["webgpu"]
    assert "shared graph interpreter" in contract["webgl"]
    for name in ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
                 "MatMulInteger", "QLinearMatMul", "ConvInteger", "QLinearConv"):
        assert name in onnxrt._OPS


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


def test_webgl_growing_cache_shader_is_registered_once_before_weight_upload(monkeypatch):
    calls = []
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "_gl_kernels", set())
    monkeypatch.setitem(wt._copy_kernel, "plat",
                        SimpleNamespace(addKernel=lambda *args: calls.append(args)))
    wt._webgl_prepare_growing_cache()
    wt._webgl_prepare_growing_cache()
    assert len(calls) == 1
    assert calls[0][0] == "cat2_gl"
    assert "#version 300 es" in calls[0][1]["source"]
    assert "wt._webgl_prepare_growing_cache()" in inspect.getsource(llm.CausalLM._from_gguf)


def test_q4k_cross_width_candidate_keeps_a_webgl_operator_equivalent():
    # WebGPU may use its packed INT8 dot instruction after a numerical/performance gate.
    # WebGL lacks that compute primitive, but exposes the same Linear result through its
    # original packed Q4_K fragment path at the nearest common (operator) layer.
    assert "packed_4x8_integer_dot_product" in wt._Q4K_DP4A_WGSL
    assert "dot4I8Packed" in wt._Q4K_DP4A_WGSL
    assert "#version 300 es" in wt._ggml_src_gl("Q4_K", False, False, mode=1)
    contract = parity_contracts()["ggml_phase2_linear"]
    assert contract["common_level"] == "operator"


def test_q6k_cross_width_candidate_uses_original_weight_and_has_webgl_equivalent():
    assert "packed_4x8_integer_dot_product" in wt._Q6K_DP4A_WGSL
    assert "dot4I8Packed" in wt._Q6K_DP4A_WGSL
    assert "b*210u" in wt._Q6K_DP4A_WGSL
    assert "o+208u" in wt._Q6K_DP4A_WGSL
    assert "type_name in (\"Q4_K\", \"Q6_K\")" in inspect.getsource(wt.ggml_matmul)
    assert "#version 300 es" in wt._ggml_src_gl("Q6_K", False, False, mode=1)


def test_q6k_device_decode_input_reads_original_210_byte_blocks():
    src = wt._Q6K_DECODE_INPUT_WGSL
    assert "o=b*210u" in src
    assert "o+208u" in src
    assert "tokens[im.slot]" in src
    # Position from the device counter, rotary rows from a per-position table.
    assert "let p=ctl[0]+i32(im.inc); ctl[0]=p; wpos=p;" in src
    assert "cos_table[p*im.HD+j]" in src
    assert "vocab_argmax(x, out=None, offset=0)" in inspect.getsource(wt.vocab_argmax)


def test_residual_norm_composition_is_aligned_at_the_layer_boundary():
    import numpy as np
    a = wt.Tensor(np.asarray([[1.0, -2.0, 3.0, 4.0]], np.float32))
    b = wt.Tensor(np.asarray([[0.5, 1.0, -1.0, 2.0]], np.float32))
    w = wt.Tensor(np.asarray([1.0, 0.5, 2.0, -1.0], np.float32))
    summed, normed = wt.add_rmsnorm(a, b, w, 1e-6)
    want_sum = np.asarray(a.data) + np.asarray(b.data)
    want_norm = want_sum / np.sqrt(np.mean(want_sum * want_sum, axis=-1, keepdims=True) + 1e-6)
    want_norm *= np.asarray(w.data)
    assert np.array_equal(np.asarray(summed.data), want_sum)
    assert np.allclose(np.asarray(normed.data), want_norm, rtol=1e-6, atol=1e-6)
    contract = parity_contracts()["transformer_layer_boundary"]
    assert contract["common_level"] == "layer"
    for fn in (llm.CausalLM._prefill, llm.CausalLM._decode_fwd, llm.CausalLM._kv_forward):
        assert "self._add_rms" in inspect.getsource(fn)


def test_parallel_projection_is_one_layer_contract_on_both_backends():
    # The WebGPU source keeps each packed weight in its own binding and emits each result
    # directly, while the common function also accepts ordinary modules (the WebGL/CPU
    # equivalent when there is no multi-output physical pass).
    src = wt._ggml_parallel_src("Q4_K", 3)
    assert "var<storage,read> w0" in src
    assert "var<storage,read> w2" in src
    assert "var<storage,read_write> out2" in src
    assert "row - (0u + gm.estride + gm.eslot)" in src
    assert "STORE(nn, tot);" in src
    contract = parity_contracts()["parallel_projection"]
    assert contract["common_level"] == "layer"
    assert "same layer operation" in contract["webgl"]


def test_parallel_projection_shapes_are_exact_addressable_and_upper_overridable():
    assert "@workgroup_size(32, 8u)" in wt._ggml_parallel_src(
        "Q4_K", 3, "balanced")
    assert "@workgroup_size(32, 4u)" in wt._ggml_parallel_src(
        "Q4_K", 3, "compact")
    source = inspect.getsource(wt.parallel_linear)
    for route in ("fused:default", "fused:balanced", "fused:compact", "fused:narrow"):
        assert route in source
    upper = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert '"fused:balanced"' in upper
    assert '"fused:compact"' in upper


def test_stored_weight_physical_shape_is_addressable_and_head_overridable():
    linear = inspect.getsource(wt.GGMLLinear)
    assert 'self.decode_shape = "auto"' in linear
    assert "shape_execution=" in linear
    matmul = inspect.getsource(wt.ggml_matmul)
    assert 'shape_execution="auto"' in matmul
    assert 'if shape_execution == "auto" else shape_execution' in matmul
    upper = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert "head_modes" in upper
    assert '"head_shape": chosen[7]' in upper


def test_upper_tuner_prunes_routes_that_cannot_reach_the_model_graph():
    upper = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert "parallel_capable" in upper
    assert "dense_layers" in upper and "gate_modes" in upper
    assert "attention_layers" in upper
    assert "qkv_modes" in upper and "qk_modes" in upper and "kv_modes" in upper


def test_complete_api_tuners_require_sequential_semantic_equivalence():
    gpu = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert "semantic_steps = 4" in gpu
    assert "semantic_trace(reference_plan, original_width=True)" in gpu
    assert 'plat.replay("decode_semantic")' in gpu
    assert "for lin in stored_linears" in gpu
    assert "reference_trace" in gpu and "semantic_cache" in gpu
    assert "picked != reference_pick" in gpu

    gl = inspect.getsource(llm.CausalLM._tune_webgl_decode_composition)
    assert "for pos in range(4)" in gl
    assert "tuple(tokens)" in gl
    assert "picked != reference" in gl


def test_qk_norm_rope_has_one_webgpu_dispatch_and_webgl_layer_equivalent():
    src = wt._QK_NORM_ROPE_WGSL
    assert "@compute @workgroup_size(256)" in src
    assert "red[t]=sum" in src
    assert "outv=nv*cb[j]" in src
    route = inspect.getsource(llm.CausalLM._qk_norm_rope)
    decode = inspect.getsource(llm.CausalLM._decode_fwd)
    assert "wt.qk_norm_rope_decode" in route
    assert "self._qk_norm" in route and "self._rope1" in route
    assert "self._qk_norm_rope" in decode
    assert 'qk_modes = (("auto", "composed", "fused")' in inspect.getsource(
        llm.CausalLM._tune_decode_composition)
    contract = parity_contracts()["qk_norm_rope"]
    assert contract["common_level"] == "layer"


def test_kv_pair_write_is_exactly_one_webgpu_dispatch_and_one_common_interface():
    for src in (wt._KVWRITE_PAIR_WGSL, wt._KVWRITE_PAIR_F16_WGSL):
        assert "if (which == 0u)" in src
        assert "else" in src
        assert "@compute @workgroup_size(64)" in src
    route = inspect.getsource(wt.kv_write_pair)
    assert 'execution="auto"' in route
    assert '_kv_pair_auto' in route
    assert '_kv_write_pair_fused' in route
    assert 'kv_write(kcache' in route and 'kv_write(vcache' in route
    decode = inspect.getsource(llm.CausalLM._decode_fwd)
    assert "wt.kv_write_pair" in decode
    upper = inspect.getsource(llm.CausalLM._tune_decode_composition)
    assert '("auto", "separate", "fused")' in upper
    contract = parity_contracts()["kv_cache"]
    assert contract["common_level"] == "attention"


def test_parallel_mlp_activation_is_measured_at_the_layer_on_both_backends():
    contract = parity_contracts()["parallel_mlp_activation"]
    assert contract["common_level"] == "layer"
    assert "measured" in contract["webgpu"]
    assert "measured" in contract["webgl"]
    source = inspect.getsource(wt.parallel_swiglu)
    assert 'execution="auto"' in source
    assert 'execution not in ("auto", "separate", "fused") + fused_modes' in source
    assert 'backend_name = "webgpu" if _adam_backend_ready() else "webgl"' in source
    assert "_measured_choice" in source
    assert "parallel_swiglu" in inspect.getsource(llm.CausalLM._mlp)


def test_model_smoke_uses_each_backends_complete_cache_path():
    src = inspect.getsource(llm.CausalLM._smoke)
    assert "capturable = self._capturable()" in src
    assert "if capturable:" in src
    assert "self._prefill(ids)" in src
    assert "wt.KVCache(" in src
    assert "self._kv_forward(ids, 0, cache)" in src


def test_load_smoke_proves_one_webgl_decode_without_batched_prefill_tuning(
        monkeypatch):
    monkeypatch.setattr(wt, "KVCache", lambda *args: object())
    seen = []
    model = SimpleNamespace(tok=SimpleNamespace(vocab_size=100),
                            L=1, NKV=1, HD=4, lmax=8,
                            _capturable=lambda: False,
                            _logits=lambda hidden: [0.0, 1.0],
                            _kv_drop=lambda: None)

    def forward(ids, pos, cache):
        seen.append(tuple(ids))
        model._last_hidden = SimpleNamespace(numpy=lambda: np.ones((1, 4)))
        return 1

    model._kv_forward = forward
    assert llm.CausalLM._smoke(model) is True
    assert seen == [(1,)]

    model._capturable = lambda: True
    model._prefill = lambda ids: seen.append(tuple(ids))
    model._last_prefill_hidden = SimpleNamespace(numpy=lambda: np.ones((1, 4)))
    assert llm.CausalLM._smoke(model) is True
    assert seen[-1] == (1, 2, 3, 4)


def test_webgl_native_bmm_indexes_every_batch_not_only_the_first():
    source = (Path(__file__).parents[1] / "webgl" / "wgpy_backends" / "webgl"
              / "webgl_array_func.py").read_text()
    bmm = source.split("    def batched_matmul(", 1)[1].split("    def matmul(", 1)[0]
    assert "int b = flat_idx / (M * N);" in bmm
    assert "get_lhs(b, oi, k) * get_rhs(b, k, oj)" in bmm
    assert "if (oi >= M)" not in bmm


def test_browser_host_forwards_explicit_greedy_sampling_choice():
    """The JS API must not silently discard do_sample=False from a caller."""
    host = (Path(__file__).parents[1] / "webtorch/js/webtorch-host.js").read_text()
    forwarded = host[host.index('_PASS = ('):host.index('for _k in _PASS')]
    assert '"do_sample"' in forwarded


def test_warming_progress_uses_one_unit_for_done_and_total():
    source = inspect.getsource(llm.CausalLM._warm_shapes)
    assert '_load_stage("warming", done=len(seen), total=len(seen))' in source
    assert 'total=int(c.get("variants")' not in source
