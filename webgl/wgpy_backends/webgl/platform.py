# platform call interface
import numpy as np
from js import gl  # Pyodide-dependent


class WebGLPlatform:
    def __init__(self) -> None:
        self._latest_comm_buf = None

    def getDeviceInfo(self) -> dict:
        return gl.getDeviceInfo().to_py()

    def createBuffer(self, buffer_id: int, texture_shape_json: str):
        return gl.createBuffer(buffer_id, texture_shape_json)

    def disposeBuffer(self, buffer_id: int):
        return gl.disposeBuffer(buffer_id)

    def setCommBuf(self, buffer: np.ndarray):
        self._latest_comm_buf = buffer
        return gl.setCommBuf(buffer)

    def releaseCommBuf(self):
        self._latest_comm_buf = None
        return gl.releaseCommBuf()

    def setData(self, buffer_id: int, js_ctor_type: str, size: int):
        status = gl.setData(buffer_id, js_ctor_type, size)
        if status == -1:
            raise RuntimeError("WebGL upload failed; release and reload the model")
        if not status:
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            status = gl.setData(buffer_id, js_ctor_type, size)
            if status == -1:
                raise RuntimeError("WebGL upload failed; release and reload the model")
            if not status:
                raise ValueError("setData failed twice")

    def setDataFromArray(self, buffer_id: int, array: np.ndarray,
                         js_ctor_type: str, byte_length: int):
        """Upload an exact, contiguous NumPy array without a Python staging copy."""
        status = gl.setDataFromArray(buffer_id, array, js_ctor_type, byte_length)
        if status < 0:
            raise RuntimeError("WebGL direct upload failed; release and reload the model")

    def getData(self, buffer_id: int, js_ctor_type: str, size: int):
        status = gl.getData(buffer_id, js_ctor_type, size)
        if status == -1:
            raise RuntimeError("WebGL readback failed; release and reload the model")
        if not status:
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            status = gl.getData(buffer_id, js_ctor_type, size)
            if status == -1:
                raise RuntimeError("WebGL readback failed; release and reload the model")
            if not status:
                raise ValueError("getData failed twice")

    def sampleLogits(self, buffer_id: int, size: int, options: dict) -> int:
        """Select a token in JS from a GPU readback; Python handles only IDs/options."""
        return int(gl.sampleLogits(buffer_id, size, options))

    def routeHost(self, logits_id: int, index_id: int, weight_id: int,
                  rows: int, experts: int, k: int, renormalize: bool):
        """Route a GPU logits buffer entirely in JS; Python passes buffer handles."""
        return gl.routeHost(logits_id, index_id, weight_id,
                            rows, experts, k, renormalize)

    def addKernel(self, name, descriptor):
        return gl.addKernel(name, descriptor)

    def runKernel(self, descriptor):
        return gl.runKernel(descriptor)

    def beginCapture(self, name):
        from wgpy_backends.webgl.webgl_buffer import begin_capture_pin
        result = gl.beginCapture(name)
        begin_capture_pin(name)
        return result

    def endCapture(self):
        from wgpy_backends.webgl.webgl_buffer import end_capture_pin
        end_capture_pin()
        return gl.endCapture()

    def resetCaptures(self):
        """Drop every recorded capture graph and its pins, JS side included.
        Sent at model release, BEFORE the buffered disposeBuffer messages, so
        those are no longer refused by the JS-side pin set."""
        from wgpy_backends.webgl.webgl_buffer import reset_capture_pins
        reset_capture_pins()
        return gl.resetCaptures()

    def releaseCapture(self, name):
        from wgpy_backends.webgl.webgl_buffer import release_capture_pin
        result = gl.releaseCapture(name)
        release_capture_pin(name)
        return result

    def replay(self, name):
        return gl.replay(name)

    def clearBuffer(self, buffer_id):
        """Zero a buffer where it lives, in command order. No host data crosses."""
        return gl.clearBuffer(buffer_id)


_instance = None


def get_platform() -> WebGLPlatform:
    global _instance
    if _instance is None:
        _instance = WebGLPlatform()
    return _instance
