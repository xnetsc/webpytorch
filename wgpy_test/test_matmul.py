import numpy as np
import pytest
import wgpy as cp


def allclose(expected, actual, rtol=1e-2, atol=1e-2):
    np.testing.assert_allclose(expected, actual, rtol=rtol, atol=atol)


def test_matmul():
    n1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    n2 = np.array([[-0.5, 1.5], [2.0, 0.0]], dtype=np.float32)
    t1 = cp.asarray(n1)
    t2 = cp.asarray(n2)
    t3 = t1 @ t2
    n3 = cp.asnumpy(t3)
    allclose(n1 @ n2, n3)


def test_webgl_row_aligned_half_rhs_matches_general_matmul():
    if "webgl" not in type(cp.asarray(np.zeros((1, 1), np.float32))).__module__:
        pytest.skip("requires WebGL")
    from wgpy_backends.webgl.ndarray import ndarray as GLArray
    from wgpy_backends.webgl.texture import WebGL2RenderingContext as GL
    from wgpy_backends.webgl.texture import WebGLArrayTextureShape
    from wgpy_backends.webgl.webgl_buffer import WebGLBuffer

    rng = np.random.default_rng(44)
    lhs = rng.integers(-4, 5, size=(5, 71)).astype(np.float32) * 0.25
    rhs = rng.integers(-4, 5, size=(71, 17)).astype(np.float32) * 0.25
    texture = WebGLArrayTextureShape(71, 17, internal_format=GL.R16F,
                                     format=GL.RED, type=GL.HALF_FLOAT)
    buffer = WebGLBuffer(rhs.size, np.dtype(np.float32), texture)
    buffer.set_data(rhs)
    row_rhs = GLArray(rhs.shape, np.float32, buffer=buffer)
    assert row_rhs.buffer.texture_shape.width == 17
    assert row_rhs.buffer.texture_shape.height == 71
    np.testing.assert_array_equal(cp.asnumpy(cp.asarray(lhs) @ row_rhs),
                                  cp.asnumpy(cp.asarray(lhs) @ cp.asarray(rhs)))


def test_matmul_multi_shape():
    np.random.seed(1)
    for m in [1, 8, 17, 65]:
        for n in [1, 8, 17, 65]:
            for k in [1, 8, 17, 65]:
                n1 = np.random.randint(-4, 5, size=(m, k)).astype(np.float32)
                n2 = np.random.randint(-4, 5, size=(k, n)).astype(np.float32)
                n3actual = cp.asnumpy(cp.asarray(n1) @ cp.asarray(n2))
                n3expect = n1 @ n2
                allclose(n3actual, n3expect)

                out = cp.empty(n3expect.shape, dtype=np.float32)
                cp.matmul(cp.asarray(n1), cp.asarray(n2), out=out)
                allclose(cp.asnumpy(out), n3expect)


def test_batched_matmul_tiled_attention_shapes():
    """Batch tiles keep each sequence separate, including tail rows and K lanes."""
    rng = np.random.default_rng(17)
    for batch, rows, cols, width in ((2, 33, 49, 48), (3, 160, 160, 64)):
        lhs = rng.normal(size=(batch, rows, width)).astype(np.float32)
        rhs = rng.normal(size=(batch, width, cols)).astype(np.float32)
        actual = cp.asnumpy(cp.asarray(lhs) @ cp.asarray(rhs))
        np.testing.assert_allclose(actual, lhs @ rhs, rtol=3e-3, atol=3e-3)
        # Attention supplies the right operand as a transposed strided view.
        original = np.ascontiguousarray(rhs.transpose(0, 2, 1))
        actual_view = cp.asnumpy(cp.asarray(lhs) @ cp.asarray(original).transpose(0, 2, 1))
        np.testing.assert_allclose(actual_view, lhs @ rhs, rtol=3e-3, atol=3e-3)
