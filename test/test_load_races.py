"""What a load's route races cost, and what they are not allowed to give up for it.

A race's cost is the GPU work it times. These pin the rules that cut that work -- a
candidate out once it has lost every round by complete separation, a one-run warm-up, a
small self-check shape for an explicitly raced variant, a young-generation collect on the
allocation path -- and that none of them can drop a measured win.
"""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from webtorch import _core as wt
from webtorch import linear_attn as la


def _sampler(table):
    """`sample(c)` returning table[c][round] -- a scripted timing per candidate per round."""
    seen = {c: 0 for c in table}
    calls = []

    def sample(c):
        calls.append(c)
        v = table[c][seen[c]]
        seen[c] += 1
        return v
    return sample, calls


def test_a_candidate_that_lost_every_round_by_separation_stops_being_timed():
    sample, calls = _sampler({"a": [1.0] * 5, "b": [2.0, 2.1, 2.2, 2.0, 2.1],
                              "c": [1.02, 0.99, 1.01, 1.0, 1.0]})
    times = wt._race(("a", "b", "c"), sample, 5)
    assert set(times) == {"a", "c"}                     # c overlaps a: it stays in
    assert calls.count("b") == 3 and calls.count("a") == 5 and calls.count("c") == 5


def test_no_candidate_is_dropped_before_three_rounds():
    sample, calls = _sampler({"a": [1.0] * 5, "b": [9.0] * 5})
    times = wt._race(("a", "b"), sample, 2)
    assert set(times) == {"a", "b"} and len(calls) == 4


def test_measuring_stops_once_one_candidate_is_left():
    sample, calls = _sampler({"a": [1.0] * 9, "b": [3.0] * 9, "c": [4.0] * 9})
    times = wt._race(("a", "b", "c"), sample, 9)
    assert list(times) == ["a"] and len(calls) == 9    # three rounds of three, then done


def test_a_winner_proven_against_every_survivor_ends_the_race_early():
    table = {"a": [1.0, 1.1, 1.0, 1.1, 1.0, 1.0, 1.0],
             "b": [1.05, 1.0, 1.2, 1.15, 1.1, 1.1, 1.1]}
    sample, calls = _sampler(table)
    times = wt._race(("a", "b"), sample, 7, early_from=5)
    assert len(times["a"]) == 7                         # b won round two: no proof by five
    sample, calls = _sampler({"a": [1.0] * 9, "b": [1.01, 0.99] + [1.1] * 7})
    times = wt._race(("a", "b"), sample, 9, early_from=5)
    # b never separates (it won round two) but a has won the paired rounds since;
    # with p <= 0.05 needing five wins out of the pairs, the race stops once that holds.
    assert len(times["a"]) < 9 and set(times) == {"a", "b"}


def test_tune_warms_each_candidate_with_its_warm_not_its_bench(monkeypatch):
    import time
    monkeypatch.setattr(wt, "_TUNED", {})
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    benches, warms = [], []
    cur = {}

    def bench():
        benches.append(cur["v"])
        clock[0] += 1.0 if cur["v"] == "x" else 2.0
    choice = wt.tune(("t", 1), ("y", "x"), lambda v: cur.update(v=v), bench,
                     warm=lambda: warms.append(cur["v"]))
    assert warms == ["y", "x"]
    # Three rounds separate y (2.0 every time) from x (1.0): six timed runs, not ten.
    assert len(benches) == 6 and choice == "x"


def test_route_race_warms_each_candidate_with_one_run(monkeypatch):
    monkeypatch.setattr(wt, "_TUNED", {})
    monkeypatch.setattr(wt, "_sync_small", lambda a: None)
    runs = []
    wt._weight_execution("unit", "f32", 8, 8, 4, lambda w: runs.append(w) or w,
                         candidates=("stored", "other"), rounds=5, repeat=4)
    # One warm-up each, then batches of four for at most five rounds.
    assert runs[:2] == ["stored", "other"]
    assert len(runs) <= 2 + 5 * 2 * 4


def test_an_explicitly_raced_short_k_variant_is_checked_at_the_small_shape():
    vals = 256
    assert wt._selfcheck_shape("shortk", vals) is None          # unreachable by routing
    shape = wt._selfcheck_shape("shortk", vals, explicit=True)
    assert shape == (wt._SMALL_N + 64, 3)
    src = __import__("inspect").getsource(wt._ggml_shape_for)
    assert "_selfcheck_shape(kind, vals, explicit=True)" in src


def test_reset_clears_the_state_in_place_and_nothing_crosses_from_the_host(monkeypatch):
    log = []
    monkeypatch.setattr(wt, "device_zeros", lambda n: log.append(("zeros", n)) or ["z", n])
    monkeypatch.setattr(wt, "device_clear", lambda b: log.append(("clear", b[1])) or b)
    monkeypatch.setattr(wt, "xp", SimpleNamespace(
        asarray=lambda a: log.append(("upload", a.size)) or ["u", a.size]))
    st = la.LinearAttentionState(2, 4, 4, 4, 24)
    assert st._S is None and st._conv is None                 # no host copy is made up front
    g = st.gpu()
    assert log == [("zeros", 32), ("zeros", 72)]
    log.clear()
    st.reset()
    assert log == [("clear", 32), ("clear", 72)]
    assert st.gpu() is g and log == [("clear", 32), ("clear", 72)]   # same buffers, no upload


def test_the_host_path_reads_zeros_without_a_readback_after_a_reset(monkeypatch):
    monkeypatch.setattr(wt, "device_zeros", lambda n: ["z", n])
    monkeypatch.setattr(wt, "device_clear", lambda b: b)
    monkeypatch.setattr(wt, "xp", SimpleNamespace(asarray=lambda a: ["u", a.size]))
    reads = []
    monkeypatch.setattr(wt, "cp", SimpleNamespace(
        asnumpy=lambda a: reads.append(a) or np.ones(a[1], np.float32)), raising=False)
    st = la.LinearAttentionState(2, 4, 4, 4, 24)
    st.gpu(); st.reset()
    assert st.host().S.sum() == 0 and reads == []
    st.gpu()                                                  # the device writes it now
    st.host()
    assert len(reads) == 2 and st.S.sum() == 32


def test_a_host_written_state_is_uploaded(monkeypatch):
    log = []
    monkeypatch.setattr(wt, "device_zeros", lambda n: log.append("zeros") or ["z", n])
    monkeypatch.setattr(wt, "xp", SimpleNamespace(
        asarray=lambda a: log.append(float(a.sum())) or ["u", a.size]))
    st = la.LinearAttentionState(2, 4, 4, 4, 24)
    st.host().S[:] = 1.0
    st.gpu()
    assert log == [32.0, 0.0]                                 # S as written, conv as zeros


@pytest.mark.parametrize("held, full", [(1100, False), (1300, True)])
def test_the_allocation_path_collects_young_first(held, full):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys, types, gc
sys.modules["js"] = types.SimpleNamespace(gpu=types.SimpleNamespace(), gl=None)
from wgpy_backends.webgpu import webgpu_buffer as wb
calls = []
gc.collect = lambda generation=2: calls.append(generation)
wb.get_platform = lambda: types.SimpleNamespace(gpuBytes=lambda: (HELD, 0, 0))
wb._live_floor = 1000; wb._reap_budget = 200; wb._bytes_since_reap = 500
wb._maybe_reap()
assert calls[0] == 1, calls
assert (len(calls) == 2) == FULL, calls
assert wb._bytes_since_reap == 0
'''.replace("HELD", str(held)).replace("FULL", str(full))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "webgpu")
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_device_clear_and_zeros_on_the_device_in_the_browser():
    """Both backends: a clear zeroes a buffer that held data, in command order, and a
    zeros array is zeros whatever buffer the pool hands back."""
    if not (wt._adam_backend_ready() or wt._webgl_ready()):
        pytest.skip("requires a browser GPU backend")
    a = wt.xp.asarray(np.arange(1, 1001, dtype=np.float32))
    b = wt.xp.asarray(np.full(1000, 7.0, np.float32))
    wt.device_clear(a)
    assert np.all(np.asarray(a.get()) == 0) and np.all(np.asarray(b.get()) == 7)
    for n in (1000, 3 * 128 * 128):
        assert np.all(np.asarray(wt.device_zeros(n).get()) == 0)


def test_self_check_blocks_are_drawn_once_per_format_in_the_browser():
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    wt._SELFCHECK_BLOCKS.pop(("Q4_K", 3), None)
    wt._selfcheck_one("Q4_K", 1, "narrow", False, 2, 3)
    first = wt._SELFCHECK_BLOCKS[("Q4_K", 3)]
    wt._selfcheck_one("Q4_K", 1, "balanced", False, wt._SMALL_N + 64, 3)
    assert wt._SELFCHECK_BLOCKS[("Q4_K", 3)] is first          # reused, not redrawn
