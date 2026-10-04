import numpy as np

from webtorch import onnxrt


def call(name, *args, **kwargs):
    return onnxrt._OPS[name](*args, **kwargs)


def test_quantize_and_dequantize_linear_scalar_and_per_axis():
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], np.float32)
    q = call("QuantizeLinear", x, np.float32(0.5), np.int8(0))[0]
    np.testing.assert_array_equal(q, [[-2, 0, 2], [4, -4, 1]])
    np.testing.assert_allclose(
        call("DequantizeLinear", q, np.float32(0.5), np.int8(0))[0],
        q.astype(np.float32) * 0.5,
    )

    scales = np.array([0.5, 1.0, 0.25], np.float32)
    zeros = np.array([1, 2, 3], np.uint8)
    qp = call("QuantizeLinear", x, scales, zeros, axis=1)[0]
    np.testing.assert_array_equal(qp, [[0, 2, 7], [5, 0, 5]])
    np.testing.assert_allclose(
        call("DequantizeLinear", qp, scales, zeros, axis=1)[0],
        (qp.astype(np.int32) - zeros) * scales,
    )


def test_matmul_integer_keeps_integer_operands_and_int32_accumulator():
    a = np.array([[4, 5, 6], [7, 8, 9]], np.uint8)
    b = np.array([[1, 2], [3, 4], [5, 6]], np.int8)
    got = call("MatMulInteger", a, b, np.uint8(4), np.int8(1))[0]
    want = (a.astype(np.int32) - 4) @ (b.astype(np.int32) - 1)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, want)


def test_qlinear_matmul_requantizes_from_int32_reference():
    a = np.array([[12, 14, 10]], np.uint8)
    b = np.array([[3, -1], [5, 2], [0, 4]], np.int8)
    got = call(
        "QLinearMatMul",
        a, np.float32(0.25), np.uint8(10),
        b, np.float32(0.5), np.int8(1),
        np.float32(0.125), np.int8(0),
    )[0]
    acc = (a.astype(np.int32) - 10) @ (b.astype(np.int32) - 1)
    want = np.clip(np.rint(acc * 0.25 * 0.5 / 0.125), -128, 127).astype(np.int8)
    assert got.dtype == np.int8
    np.testing.assert_array_equal(got, want)


def test_dynamic_quantize_linear_covers_zero_and_returns_declared_types():
    x = np.array([-1.0, 0.0, 2.0], np.float32)
    q, scale, zero = call("DynamicQuantizeLinear", x)
    assert q.dtype == np.uint8
    assert scale.dtype == np.float32
    assert zero.dtype == np.uint8
    restored = (q.astype(np.int32) - int(zero)) * float(scale)
    np.testing.assert_allclose(restored, x, atol=float(scale))


def test_conv_integer_and_qlinear_conv_keep_int8_weights_and_int32_accumulation():
    x = np.array([[[[4, 5, 6], [7, 8, 9], [10, 11, 12]]]], np.uint8)
    w = np.array([[[[1, 2], [3, 4]]], [[[4, 3], [2, 1]]]], np.int8)
    acc = call("ConvInteger", x, w, np.uint8(4), np.array([1, 2], np.int8))[0]
    assert acc.dtype == np.int32
    centered_x = x.astype(np.int32) - 4
    centered_w = w.astype(np.int32) - np.array([1, 2], np.int32).reshape(2, 1, 1, 1)
    want = onnxrt._conv_nd(centered_x, centered_w, None, None, None, None, 1,
                           out_dtype=np.int32)
    np.testing.assert_array_equal(acc, want)

    bias = np.array([2, -3], np.int32)
    got = call(
        "QLinearConv",
        x, np.float32(0.5), np.uint8(4),
        w, np.array([0.25, 0.5], np.float32), np.array([1, 2], np.int8),
        np.float32(0.25), np.uint8(10), bias,
    )[0]
    scaled = (want + bias.reshape(1, 2, 1, 1)).astype(np.float32)
    scaled *= 0.5 * np.array([0.25, 0.5], np.float32).reshape(1, 2, 1, 1)
    expected = np.clip(np.rint(scaled / 0.25) + 10, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(got, expected)
