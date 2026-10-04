"""Browser gate for standard quantized ONNX operators on either selected backend.

ONNX is a graph/container contract, not GGUF or GPTQ storage.  These operators therefore
keep their ONNX INT8/UINT8 tensors and INT32 accumulators instead of translating the model
into another quantization format.  The graph interpreter is the shared API-level fallback
when neither graphics backend has a lower, faster equivalent for the graph as a whole.
"""
import json

import numpy as np
from js import pythonIO

import webtorch
from webtorch import onnxrt


def main():
    a = np.array([[12, 14, 10]], np.uint8)
    b = np.array([[3, -1], [5, 2], [0, 4]], np.int8)
    acc = onnxrt._OPS["MatMulInteger"](a, b, np.uint8(10), np.int8(1))[0]
    want_acc = (a.astype(np.int32) - 10) @ (b.astype(np.int32) - 1)

    q = onnxrt._OPS["QLinearMatMul"](
        a, np.float32(0.25), np.uint8(10),
        b, np.float32(0.5), np.int8(1),
        np.float32(0.125), np.int8(0),
    )[0]
    want_q = np.clip(np.rint(want_acc * 0.25 * 0.5 / 0.125), -128, 127).astype(np.int8)

    x = np.array([[[[4, 5, 6], [7, 8, 9], [10, 11, 12]]]], np.uint8)
    w = np.array([[[[1, 2], [3, 4]]]], np.int8)
    conv = onnxrt._OPS["ConvInteger"](x, w, np.uint8(4), np.int8(1))[0]
    want_conv = onnxrt._conv_nd(
        x.astype(np.int32) - 4, w.astype(np.int32) - 1,
        None, None, None, None, 1, out_dtype=np.int32)

    result = {
        "backend": webtorch.backend(),
        "common_level": "onnx-api",
        "same_width_inputs": True,
        "int32_accumulator": str(acc.dtype) == "int32" and str(conv.dtype) == "int32",
        "matmul_integer": bool(np.array_equal(acc, want_acc)),
        "qlinear_matmul": bool(np.array_equal(q, want_q)),
        "conv_integer": bool(np.array_equal(conv, want_conv)),
    }
    result["ok"] = all(result[k] for k in (
        "same_width_inputs", "int32_accumulator", "matmul_integer",
        "qlinear_matmul", "conv_integer"))
    print("ONNX_QUANT " + json.dumps(result))
    pythonIO.result = json.dumps(result)


main()
