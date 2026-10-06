"""Recordings made at a capacity (`elastic`, `_dyn`), race samples that time only the work
(`_RaceClock`, `_settle`), and racing the loaded model's routes again (`remeasure`)."""
import pytest

from webtorch import _core as wt
from webtorch import webio


# ---- rules a recording replays with --------------------------------------------------------

def test_a_launcher_rule_must_give_the_count_it_computed_at_the_capacity():
    with wt.elastic(512, 64):
        assert wt._elastic_rows(512) and not wt._elastic_rows(511)
        desc = {"name": "mm", "workGroups": {"x": 36, "y": 16, "z": 1}}
        wt._dyn(desc, y=("rows", 1, 32))
        assert desc["dyn"] == {"y": ["rows", 1, 32]}
        bad = {"name": "mm", "workGroups": {"x": 36, "y": 15, "z": 1}}
        with pytest.raises(RuntimeError, match="gives 16 at the capacity"):
            wt._dyn(bad, y=("rows", 1, 32))
        # Past the dispatch limit the platform folds the grid: replayed as recorded.
        big = {"name": "ew", "workGroups": {"x": 512 * 9000 // 64, "y": 1, "z": 1}}
        wt._dyn(big, x=("rows", 9000, 64))
        assert "dyn" not in big
    assert not wt._elastic_rows(512)


def test_elastic_recordings_do_not_nest():
    with wt.elastic(128, 16):
        with pytest.raises(RuntimeError, match="do not nest"):
            with wt.elastic(256, 32):
                pass


def test_a_settled_clock_costs_one_round_and_a_climbing_one_is_waited_out():
    times = iter([3.0, 2.0, 1.5, 1.5, 1.6])
    rounds = []

    def sample(c):
        rounds.append(c)
        return next(times)
    per = wt._settle(("a",), sample, {"a": 4.0})
    assert per == {"a": 1.5} and len(rounds) == 4      # 3.0, 2.0, 1.5 faster; 1.5 is not
    rounds.clear()
    assert wt._settle(("a",), sample, {"a": 1.0}, busy=True) == {"a": 1.0} and not rounds


# ---- racing again --------------------------------------------------------------------------

@pytest.fixture
def raced(monkeypatch):
    """Two routes raced through `tune` on a fake clock: "fast" wins both."""
    import time
    clock = [1000.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    for name in ("_TUNED", "_USED", "_TUNE_AGAIN", "_AGAIN_FN", "_PROBE_FOR"):
        monkeypatch.setattr(wt, name, {})
    monkeypatch.setattr(wt, "_ROUTE_HOOKS", [])
    monkeypatch.setattr(wt, "_REMEASURED", [])
    monkeypatch.setattr(wt, "_RACE_AGAIN", set())
    monkeypatch.setattr(wt, "_LAST_SAMPLE_END", [0.0])
    monkeypatch.setattr(wt, "_REMEASURE_SEQ", [0])
    monkeypatch.setattr(wt, "_REMEASURED_AT", {})
    monkeypatch.setattr(wt, "_DEFERRED", [])
    monkeypatch.setattr(webio, "cancel_requested", lambda: False)
    cost = {"slow": 2.0, "fast": 1.0}
    cur = {}

    def bench():
        clock[0] += cost[cur["v"]]
    for key in (("route", 1), ("route", 2)):
        assert wt.tune(key, ("slow", "fast"), lambda v: cur.update(v=v), bench,
                       warm=lambda: None) == "fast"
    return {"clock": clock, "cost": cost, "cur": cur}


def test_remeasure_races_again_every_used_route_and_reports_what_changed(raced):
    raced["cost"].update(slow=0.5)                   # the device changed: slow is fast now
    hooked = []
    wt.on_routes_changed(lambda keys: hooked.append(set(keys)) or [{"rebuild": "x"}])
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "complete" and report["changed"] == 2
    assert [(m["key"], m["before"], m["after"]) for m in report["measured"]] == [
        ("route|1", "fast", "slow"), ("route|2", "fast", "slow")]
    assert wt._TUNED[("route", 1)] == wt._TUNED[("route", 2)] == "slow"
    assert hooked == [{("route", 1), ("route", 2)}] and report["rebuilds"] == [{"rebuild": "x"}]
    assert not report["unmeasurable"] and report["discarded"] is None


def test_a_stop_keeps_what_finished_and_drops_the_race_it_interrupted(raced, monkeypatch):
    raced["cost"].update(slow=0.5)
    after = {"n": 0}

    def cancel_requested():
        # The first race finishes; the stop arrives once the second has begun (the first
        # look after the first race is the one between the two, which is not inside a race).
        if wt._TUNED.get(("route", 1)) != "slow":
            return False
        after["n"] += 1
        return after["n"] >= 2
    monkeypatch.setattr(webio, "cancel_requested", cancel_requested)
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "stopped"
    assert [m["key"] for m in report["measured"]] == ["route|1"]
    assert wt._TUNED[("route", 1)] == "slow"                    # finished: kept
    assert wt._TUNED[("route", 2)] == "fast"                    # interrupted: unchanged
    assert report["discarded"]["key"] == "route|2" and not report["not_reached"]


def test_running_out_of_time_is_the_same_stop_and_the_rest_goes_on_when_idle(raced):
    raced["cost"].update(slow=0.5)
    report = wt.remeasure(budget_s=1e-9)
    # The time runs out inside the first race: dropped, like a stop's.
    assert report["status"] == "out_of_time" and not report["measured"]
    assert report["discarded"]["key"] == "route|1"
    assert [r["key"] for r in report["continuing"]] == ["route|2"] and not report["not_reached"]
    assert wt._TUNED[("route", 1)] == wt._TUNED[("route", 2)] == "fast"
    # Between calls, the race the budget did not reach runs (`calibrate_deferred`).
    while wt._DEFERRED:
        wt.calibrate_deferred(0)
    assert wt._TUNED[("route", 2)] == "slow" and wt._TUNED[("route", 1)] == "fast"


def test_each_remeasure_starts_with_the_routes_raced_longest_ago(raced):
    first = wt.remeasure(budget_s=1e-9)            # route 1 dropped, route 2 continues idle
    while wt._DEFERRED:
        wt.calibrate_deferred(0)                   # route 2 raced: it is the newest now
    report = wt.remeasure(budget_s=1e9)
    assert first["discarded"]["key"] == "route|1"
    assert [m["key"] for m in report["measured"]] == ["route|1", "route|2"]
    # And what a remeasure reached is kept with the profile, for the next session.
    saved = wt.kernel_profile()["remeasured"]
    assert saved["seq"] == 2 and saved["at"] == {"route|1": 2, "route|2": 2}


def test_a_newer_remeasure_drops_what_an_older_one_left_for_idle(raced):
    wt.remeasure(budget_s=1e-9)
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "complete"
    calls = len(wt._REMEASURED)
    while wt._DEFERRED:
        wt.calibrate_deferred(0)
    assert len(wt._REMEASURED) == calls             # the stale idle race did not run


def test_a_route_nothing_can_race_again_is_reported_not_skipped(raced, monkeypatch):
    monkeypatch.setattr(wt, "_CANNOT_AGAIN", {})
    wt._USED[("weight_exec", "mystery", "f32", 8, 8, 16)] = True
    wt.cannot_remeasure(("plan", 1), "searched only offline")
    report = wt.remeasure(budget_s=1e9)
    assert [(u["key"], u["why"]) for u in report["unmeasurable"]] == [
        ("weight_exec|mystery|f32|8|8|16", "no probe that runs it was kept"),
        ("plan|1", "searched only offline")]


def test_what_a_model_used_goes_with_it():
    wt._USED[("x",)] = True
    wt._TUNE_AGAIN[("x",)] = {}
    wt.on_routes_changed(lambda keys: [])
    wt.calibration_drop()
    assert not wt._USED and not wt._TUNE_AGAIN and not wt._ROUTE_HOOKS


def test_the_sdk_refuses_to_remeasure_with_nothing_loaded(monkeypatch):
    import webtorch
    from webtorch import _sdk
    monkeypatch.setattr(_sdk, "_LOADED", {})
    monkeypatch.setattr(_sdk, "_IMPL_CACHE", {})
    with pytest.raises(RuntimeError, match="needs a loaded model"):
        webtorch.remeasure()


def test_a_route_is_raced_again_with_every_argument_it_was_first_raced_with(raced):
    calls = []
    cur = {}

    def bench(n):
        calls.append(n)
        raced["clock"][0] += {"a": 1.0, "b": 2.0}[cur["v"]] * n
    assert wt.tune(("sized", 1), ("b", "a"), lambda v: cur.update(v=v), bench,
                   warm=lambda: None, sized=True) == "a"
    del calls[:]
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "complete" and calls          # bench(n) again, not bench()
