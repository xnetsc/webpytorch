import asyncio
import json

import numpy as np

import webtorch.decision as decision
from webtorch.ggufload import GGML_IDS


def test_decision_gguf_detection_uses_tensor_structure_not_architecture_name():
    names = {
        "encoder.embeddings.tok_embeddings.weight": object(),
        "scorer.3.weight": object(),
    }
    assert decision.looks_like_decision(names)
    assert not decision.looks_like_decision({"token_embd.weight": object()})


def test_decision_gguf_metadata_uses_the_declared_namespace():
    meta = {
        "general.architecture": "previously-unseen-decision-family",
        "previously-unseen-decision-family.encoder_config": json.dumps({"hidden_size": 7}),
    }
    assert decision._gguf_meta_json(meta, "encoder_config") == {"hidden_size": 7}
    assert decision._gguf_meta_json(meta, "agent_config") is None


def test_decision_gguf_decodes_f16_and_q8_without_model_specific_names():
    f16 = np.asarray([1.25, -2.5], dtype=np.float16)
    q8_values = np.arange(-16, 16, dtype=np.int8)
    q8 = np.asarray([0.5], dtype=np.float16).tobytes() + q8_values.tobytes()
    payload = f16.tobytes() + q8
    original = decision._rng

    async def fake_rng(_path, start, end):
        return payload[start:end + 1]

    decision._rng = fake_rng
    try:
        got_f16 = asyncio.run(decision._gguf_tensor("model.gguf", 0, {
            "name": "encoder.small.weight", "dims": [2],
            "type": GGML_IDS["F16"], "offset": 0,
        }))
        got_q8 = asyncio.run(decision._gguf_tensor("model.gguf", 0, {
            "name": "scorer.weight", "dims": [32],
            "type": GGML_IDS["Q8_0"], "offset": len(f16.tobytes()),
        }))
    finally:
        decision._rng = original

    np.testing.assert_array_equal(got_f16, f16)
    np.testing.assert_array_equal(got_q8, q8_values.astype(np.float16) * np.float16(0.5))


def test_decision_gguf_preserves_supported_quantized_linear_blocks():
    q8_values = np.arange(-16, 16, dtype=np.int8)
    block = np.asarray([0.5], dtype=np.float16).tobytes() + q8_values.tobytes()
    payload = block + block
    original_rng = decision._rng
    original_supported = decision.wt.ggml_native_supported

    async def fake_rng(_path, start, end):
        return payload[start:end + 1]

    decision._rng = fake_rng
    decision.wt.ggml_native_supported = lambda kind: kind == "Q8_0"
    try:
        got = asyncio.run(decision._gguf_weight("model.gguf", 0, {
            "name": "encoder.layers.0.attn.Wqkv.weight", "dims": [32, 2],
            "type": GGML_IDS["Q8_0"], "offset": 0,
        }))
    finally:
        decision._rng = original_rng
        decision.wt.ggml_native_supported = original_supported

    assert isinstance(got, decision.wt.GGMLWeight)
    assert got.shape == (2, 32)
    assert got.type_name == "Q8_0"
    np.testing.assert_array_equal(
        got.materialize(),
        np.tile(q8_values.astype(np.float16) * np.float16(0.5), (2, 1)),
    )


def test_decision_gguf_dense_matrix_uses_the_same_values_as_other_containers():
    f16 = np.asarray([[1.25, -2.5], [3.0, 0.125]], dtype=np.float16)
    f32 = np.asarray([[1.25, -2.5], [3.0, 0.125]], dtype=np.float32)
    payload = f16.tobytes() + f32.tobytes()
    original_rng = decision._rng
    original_supported = decision.wt.ggml_native_supported

    async def fake_rng(_path, start, end):
        return payload[start:end + 1]

    decision._rng = fake_rng
    decision.wt.ggml_native_supported = lambda _kind: True
    try:
        for kind, value, offset in (("F16", f16, 0), ("F32", f32, f16.nbytes)):
            got = asyncio.run(decision._gguf_weight("model.gguf", 0, {
                "name": "encoder.layers.0.attn.Wqkv.weight", "dims": [2, 2],
                "type": GGML_IDS[kind], "offset": offset,
            }))
            assert isinstance(got, np.ndarray)
            assert got.dtype == value.dtype
            np.testing.assert_array_equal(got, value)
    finally:
        decision._rng = original_rng
        decision.wt.ggml_native_supported = original_supported


def test_decision_gguf_does_not_preserve_a_non_linear_quantized_tensor():
    q8_values = np.arange(-16, 16, dtype=np.int8)
    payload = np.asarray([0.5], dtype=np.float16).tobytes() + q8_values.tobytes()
    original_rng = decision._rng
    original_supported = decision.wt.ggml_native_supported

    async def fake_rng(_path, start, end):
        return payload[start:end + 1]

    decision._rng = fake_rng
    decision.wt.ggml_native_supported = lambda _kind: True
    try:
        got = asyncio.run(decision._gguf_weight("model.gguf", 0, {
            "name": "temperature", "dims": [32],
            "type": GGML_IDS["Q8_0"], "offset": 0,
        }))
    finally:
        decision._rng = original_rng
        decision.wt.ggml_native_supported = original_supported

    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, q8_values.astype(np.float16) * np.float16(0.5))


def test_container_is_read_off_the_name_and_nothing_else_is():
    from webtorch.webio import container_of

    assert container_of("org/repo/weights-Q8_0.gguf") == "gguf"
    assert container_of("org/repo/model.safetensors") == "safetensors"
    assert container_of("a/b/graph.ONNX") == "onnx"
    # A repo or a served directory is not a container, and neither is some other file.
    assert container_of("org/repo") == ""
    assert container_of("org/repo/") == ""
    assert container_of("org/weights.bin") == ""
    # A name that merely SOUNDS like a decision model is not one; only the tensors say.
    assert decision.looks_like_decision(["decision.scorer.weight"]) is False
    assert decision.looks_like_decision(["encoder.x.weight", "scorer.0.weight"]) is True


def test_there_is_one_decision_loader_and_the_container_is_internal():
    """A second, format-named entry point would make every caller carry a fact about file
    layout in order to ask a question about models."""
    assert not hasattr(decision, "load_decision_gguf")
    assert callable(decision.load_decision)

    seen = {}

    async def fake_gguf(src, **kw):
        seen["gguf"] = src
        return "a decision model"

    async def fake_dir(src, **kw):
        seen["dir"] = src
        return "a decision model"

    before = decision._from_gguf, decision._from_directory
    decision._from_gguf, decision._from_directory = fake_gguf, fake_dir
    try:
        assert asyncio.run(decision.load_decision("org/repo/x.gguf")) == "a decision model"
        assert seen == {"gguf": "org/repo/x.gguf"}
        seen.clear()
        assert asyncio.run(decision.load_decision("org/repo")) == "a decision model"
        assert seen == {"dir": "org/repo"}
        seen.clear()
        # A caller that already knows the container is believed rather than second-guessed.
        assert asyncio.run(decision.load_decision("org/repo/odd-name", container="gguf"))
        assert seen == {"gguf": "org/repo/odd-name"}
        seen.clear()
        # A lone weights file in a container that cannot carry a config is not one.
        assert asyncio.run(decision.load_decision("org/repo/m.safetensors")) is None
        assert seen == {}
    finally:
        decision._from_gguf, decision._from_directory = before
