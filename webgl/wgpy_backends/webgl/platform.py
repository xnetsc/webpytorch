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

    def setData(self, buffer_id: int, js_ctor_type: str, size: int):
        if not gl.setData(buffer_id, js_ctor_type, size):
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            if not gl.setData(buffer_id, js_ctor_type, size):
                raise ValueError("setData failed twice")

    def getData(self, buffer_id: int, js_ctor_type: str, size: int):
        if not gl.getData(buffer_id, js_ctor_type, size):
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            if not gl.getData(buffer_id, js_ctor_type, size):
                raise ValueError("getData failed twice")

    def addKernel(self, name, descriptor):
        return gl.addKernel(name, descriptor)

    def runKernel(self, descriptor):
        return gl.runKernel(descriptor)

    def beginCapture(self, name):
        from wgpy_backends.webgl.webgl_buffer import begin_capture_pin
        begin_capture_pin()
        return gl.beginCapture(name)

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

    def replay(self, name):
        return gl.replay(name)


_instance = None


def get_platform() -> WebGLPlatform:
    global _instance
    if _instance is None:
        _instance = WebGLPlatform()
    return _instance
