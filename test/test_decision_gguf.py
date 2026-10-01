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
