"""Exercise the identical, browser-independent upload-choice policy in each backend."""
import ast
from collections import namedtuple
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _policy(backend):
    source = ROOT / backend / "wgpy_backends" / backend / f"{backend}_buffer.py"
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "_upload_auto")
    clock = [0.0]
    env = {"_upload_profiles": {}, "perf_counter": lambda: clock[0],
           "texture_type_to_element_itemsize": {7: 4}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), env)
    return env, clock


@pytest.mark.parametrize("backend", ["webgl", "webgpu"])
@pytest.mark.parametrize("winner", ["staged", "direct"])
def test_repeated_shape_chooses_faster_actual_upload_without_fixed_margin(backend, winner):
    env, clock = _policy(backend)
    Shape = namedtuple("Shape", "element_count type")
    key = Shape(16, 7) if backend == "webgl" else (64, "f32", "<f4")
    calls = []

    def staged():
        calls.append("staged")
        clock[0] += 0.001 if winner == "staged" else 0.002

    def direct():
        calls.append("direct")
        clock[0] += 0.001 if winner == "direct" else 0.002

    auto = env["_upload_auto"]
    auto(key, staged, direct)
    assert calls == ["staged"], "a one-off shape must not be uploaded twice"
    for _ in range(13):
        auto(key, staged, direct)
    assert len(calls) == 14, "calibration must never duplicate an upload"
    assert env["_upload_profiles"][key]["choice"] == winner
    before = len(calls)
    auto(key, staged, direct)
    assert calls[before:] == [winner], "settled auto must use only the winner"
