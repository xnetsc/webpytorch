"""Capture buffers follow the aliasing rules verified for each backend."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("backend", ["webgpu", "webgl"])
def test_capture_buffer_reuse_is_backend_safe(backend):
    # The backend imports Pyodide's js module. Isolate its globals in a child process.
    script = r'''
import sys, types
events = []
class Device:
    def disposeBuffer(self, bid):
        events.append(bid)
sys.modules["js"] = types.SimpleNamespace(gpu=Device(), gl=Device())
from wgpy_backends.BACKEND import BUFFER as wb
SHAPE
wb.get_platform = lambda: Device()

wb.begin_capture_pin("first")
wb._maybe_pin(11, 64)
wb._pool_put(shape, 11)  # no Python owner, but the graph already references it
if "BACKEND" == "webgpu":
    assert wb._pool_get(shape) == 11  # validated WebGPU recording reuses dead scratch
    assert 11 not in wb._orphaned
    wb._pool_put(shape, 11)
else:
    # A browser same-input capture produced different answers with this alias,
    # but not without it. The exact WebGL hazard remains to be isolated.
    assert wb._pool_get(shape) is None
wb.end_capture_pin()
assert wb._pool_get(shape) is None  # it must never leak into the ordinary pool
assert 11 in wb._orphaned

wb.begin_capture_pin("second")
wb._maybe_pin(11, 64)  # shared with the first recorded graph
wb._pool_put(shape, 11)
assert wb._pool_get(shape) is None  # neither graph may alias the other's live data
wb.end_capture_pin()
wb.release_capture_pin("first")
assert events == []
wb.release_capture_pin("second")
assert events == [11]
assert 11 not in wb._orphaned
'''
    script = script.replace("BACKEND", backend).replace("BUFFER", backend + "_buffer")
    if backend == "webgpu":
        shape = ('from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape\n'
                 'shape = WebGPUArrayTextureShape(64, "f32", "f32")')
    else:
        shape = ('from wgpy_backends.webgl.texture import WebGLArrayTextureShape, '
                 'WebGL2RenderingContext as GL\n'
                 'shape = WebGLArrayTextureShape(4, 4, internal_format=GL.R32F, '
                 'format=GL.RED, type=GL.FLOAT)')
    script = script.replace("SHAPE", shape)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / backend)
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
