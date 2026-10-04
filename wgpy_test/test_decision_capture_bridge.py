"""Browser-only bridge test: Pyodide references, JS staging, real GPU upload/readback.

No model or network weight transfer is involved.  The two rows deliberately contain
different tokens so a stale captured-input buffer is observable.
"""
import numpy as np
import wgpy as cp
from js import gl, gpu


def active_backend():
    return gpu if gpu.isAvailable() else gl


def test_js_stages_two_fresh_rows_from_raw_bytes():
    ids = np.asarray([[1, 2], [2, 1]], dtype=np.int64)
    valid = np.ones((2, 2), dtype=np.int64)
    table = np.asarray([[0], [3], [7]], dtype=np.float16)
    x = cp.asarray(np.zeros((4,), dtype=np.float32))
    mask = cp.asarray(np.zeros((8,), dtype=np.float32))
    active_backend().stageDecisionCapture(x.buffer.buffer_id,
                             {"full_attention": mask.buffer.buffer_id},
                             ids.view(np.uint8), valid.view(np.uint8),
                             table.view(np.uint8), "f16", 2, 2, 2, 1, 3, 0, 1, 0)
    np.testing.assert_array_equal(cp.asnumpy(x), [3, 7, 7, 3])
    np.testing.assert_array_equal(cp.asnumpy(mask), np.zeros((8,), np.float32))


def test_js_rewrites_same_buffers_for_new_question_rows():
    ids = np.asarray([[2, 2], [1, 1]], dtype=np.int64)
    valid = np.asarray([[1, 0], [1, 1]], dtype=np.int64)
    table = np.asarray([[0], [3], [7]], dtype=np.float16)
    x = cp.asarray(np.zeros((4,), dtype=np.float32))
    mask = cp.asarray(np.zeros((8,), dtype=np.float32))
    active_backend().stageDecisionCapture(x.buffer.buffer_id,
                             {"full_attention": mask.buffer.buffer_id},
                             ids.view(np.uint8), valid.view(np.uint8),
                             table.view(np.uint8), "f16", 2, 2, 2, 1, 3, 0, 1, 0)
    np.testing.assert_array_equal(cp.asnumpy(x), [7, 7, 3, 3])
    np.testing.assert_array_equal(cp.asnumpy(mask), [0, -1e9, 0, -1e9, 0, 0, 0, 0])


def test_js_key_mask_upload_and_query_axis_broadcast():
    lengths = np.asarray([2, 4], dtype=np.int32)
    mask = cp.asarray(np.zeros((4, 1, 4), dtype=np.float32))
    active_backend().stageDecisionKeyMask(
        mask.buffer.buffer_id, lengths.view(np.uint8), 2, 2, 4)
    expected = np.zeros((4, 1, 4), dtype=np.float32)
    expected[:2, :, 2:] = -1e9
    np.testing.assert_array_equal(cp.asnumpy(mask), expected)
    scores = cp.asarray(np.zeros((4, 3, 4), dtype=np.float32))
    np.testing.assert_array_equal(cp.asnumpy(scores + mask),
                                  np.broadcast_to(expected, (4, 3, 4)))
