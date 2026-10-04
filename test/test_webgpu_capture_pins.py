"""Regression for profile-time capture reset while the model remains loaded."""

import os
from pathlib import Path
import subprocess
import sys


def test_capture_reset_unpins_live_buffers_and_destroys_only_orphans():
    # WgPy normally imports the browser's ``js`` module.  A child interpreter keeps
    # this stub out of the rest of the host-side test suite.
    script = r'''
import sys, types

events = []
class GPU:
    def resetCaptures(self):
        events.append("reset")
    def disposeBuffer(self, bid):
        events.append(("dispose", bid))

sys.modules["js"] = types.SimpleNamespace(gpu=GPU())
from wgpy_backends.webgpu import webgpu_buffer as wb
from wgpy_backends.webgpu.platform import WebGPUPlatform
from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape

plat = WebGPUPlatform()
wb.get_platform = lambda: plat
orphan_shape = WebGPUArrayTextureShape(64, "f32", "f32")
live_shape = WebGPUArrayTextureShape(128, "f32", "f32")
plat._gpu_note(11, 64)
plat._gpu_note(12, 128)
wb.performance_metrics["webgpu.buffer.buffer_count"] = 2
wb.performance_metrics["webgpu.buffer.buffer_size"] = 192

wb.begin_capture_pin("profile_a")
wb._maybe_pin(11, 64)
wb._maybe_pin(12, 128)
wb.end_capture_pin()
wb._pool_put(orphan_shape, 11)  # owner died, graph still owns it
assert wb.capture_pin_stats()[:2] == (2, 192)

plat.resetCaptures()
assert events == ["reset", ("dispose", 11)]
assert wb.capture_pin_stats()[:2] == (0, 0)
assert not wb._orphaned and not wb._pins
assert plat.gpuBytes()[0] == 128
assert wb.performance_metrics["webgpu.buffer.buffer_count"] == 1
assert wb.performance_metrics["webgpu.buffer.buffer_size"] == 128

# The still-live Tensor releases through its normal finalizer, not resetCaptures.
wb._pool_put(live_shape, 12)
assert 12 in wb._pool[live_shape]
assert ("dispose", 12) not in events
'''
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "webgpu")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_one_capture_keeps_shared_pins_and_frees_only_its_orphans():
    script = r'''
import sys, types

events = []
class GPU:
    def releaseCapture(self, name):
        events.append(("release", name))
    def disposeBuffer(self, bid):
        events.append(("dispose", bid))

sys.modules["js"] = types.SimpleNamespace(gpu=GPU())
from wgpy_backends.webgpu import webgpu_buffer as wb
from wgpy_backends.webgpu.platform import WebGPUPlatform
from wgpy_backends.webgpu.texture import WebGPUArrayTextureShape

plat = WebGPUPlatform()
wb.get_platform = lambda: plat
shape = WebGPUArrayTextureShape(64, "f32", "f32")
for bid in (11, 12, 13):
    plat._gpu_note(bid, 64)
wb.performance_metrics["webgpu.buffer.buffer_count"] = 3
wb.performance_metrics["webgpu.buffer.buffer_size"] = 192

wb.begin_capture_pin("cold")
wb._maybe_pin(11, 64)
wb._maybe_pin(12, 64)
wb.end_capture_pin()
wb.begin_capture_pin("hot")
wb._maybe_pin(12, 64)
wb._maybe_pin(13, 64)
wb.end_capture_pin()
wb._pool_put(shape, 11)
wb._pool_put(shape, 12)

plat.releaseCapture("cold")
assert events == [("release", "cold"), ("dispose", 11)]
assert 12 in wb._pinned_ids and 12 in wb._orphaned
assert 13 in wb._pinned_ids
plat.releaseCapture("hot")
assert events == [("release", "cold"), ("dispose", 11),
                  ("release", "hot"), ("dispose", 12)]
assert not wb._pinned_ids and not wb._orphaned
assert plat.gpuBytes()[0] == 64  # live 13 is still owned by Python
'''
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "webgpu")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
