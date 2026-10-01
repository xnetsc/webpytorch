import inspect

import numpy as np

from webtorch import _core as wt
from webtorch import ggufload


def test_every_decodable_gguf_type_has_a_native_compute_kernel():
    """A storage type may not be accepted and then silently converted for lack of a kernel."""
    assert set(ggufload.SUPPORTED_NAMES) == set(wt._GGML_TYPES)


def test_q8_native_path_vectorizes_the_original_block_without_requantizing_activations():
    src = wt._Q8_0_DEC
    assert "let p = B4(o + 2u + j * 4u)" in src
    assert "ACC4(kb + j * 4u, d * q)" in src
    assert "round(" not in src
    assert "dot4I8Packed" not in src


def test_stored_weight_materializes_only_when_an_operator_requests_an_array():
    values = np.arange(-16, 16, dtype=np.int8)
    raw = np.asarray([0.5], dtype=np.float16).tobytes() + values.tobytes()
    weight = wt.GGMLWeight(raw, "Q8_0", (1, 32),
                           type_id=ggufload.GGML_IDS["Q8_0"])

    assert wt.weight_shape(weight) == (1, 32)
    np.testing.assert_array_equal(
        wt.materialize_weight(weight),
        (values.astype(np.float16) * np.float16(0.5)).reshape(1, 32),
    )


def test_non_gguf_quantized_module_uses_the_same_stored_linear_interface():
    module = wt.QuantizedLinear.from_autogptq(
        np.zeros((4, 8), np.int32), np.zeros((1, 1), np.int32),
        np.ones((1, 8), np.float32), np.zeros((8,), np.float32),
        gs=32, bits=8,
    )

    assert wt.stored_linear(module) is module
    assert module.storage_format == "GPTQ_INT8"
    assert module.execution == "stored"


def test_stored_execution_is_the_default_and_policy_is_forwarded_generically():
    assert inspect.signature(wt.ggml_matmul).parameters["execution"].default == "stored"

    class EncodedWeight:
        def as_linear(self, bias=None, execution="stored"):
            return {"bias": bias, "execution": execution}

    assert wt.stored_linear(EncodedWeight(), bias="b") == {
        "bias": "b", "execution": "stored",
    }
    assert wt.stored_linear(EncodedWeight(), execution="auto") == {
        "bias": None, "execution": "auto",
    }


def test_execution_policy_is_shape_based_cached_and_profiled():
    key = ("weight_exec", "ggml", "Q8_0", 1024, 512, 128)
    before = wt._TUNED.get(key)
    wt._TUNED[key] = "materialized"
    try:
        assert wt._weight_execution("ggml", "Q8_0", 1024, 512, 100,
                                    lambda _which: None) == "materialized"
        profile = wt.kernel_profile()
        assert profile["tuned"]["weight_exec|ggml|Q8_0|1024|512|128"] == "materialized"
    finally:
        if before is None:
            wt._TUNED.pop(key, None)
        else:
            wt._TUNED[key] = before

    # Decode and two-row verification stay in the original representation; every measured
    # format won there, so neither should pay a runtime tune.
    assert wt._weight_execution("gptq", "GPTQ_INT4", 1024, 512, 2,
                                lambda _which: None) == "stored"


def test_autogptq_zero_offset_is_part_of_the_materialized_shader():
    source = wt._gptq_src(wt._DQF_WGSL, 4, zoff=1.0)
    assert "ZOFFf" not in source
    assert "f32(zv) + 1.0" in source
