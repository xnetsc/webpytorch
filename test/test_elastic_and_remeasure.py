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
    monkeypatch.setattr(wt, "_RACE_SECONDS", {})
    for name in ("_ROUTE_ALT", "_SPEED", "_APPLY_FN"):
        monkeypatch.setattr(wt, name, {})
    monkeypatch.setattr(wt, "_ROUTE_IDS", {"in_use": 0, "unused": None})
    monkeypatch.setattr(wt, "_SAMPLE_AFTER", [0.0])
    monkeypatch.setattr(wt, "_SLOWER_TOLD", set())
    monkeypatch.setattr(wt, "_SLOWER_HOOKS", [])
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
    wt._RACE_SECONDS.clear()           # no duration known: the first race is started
    report = wt.remeasure(budget_s=wt._REMEASURE_RESERVE_S + 1e-9)
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
    wt._RACE_SECONDS.clear()
    first = wt.remeasure(budget_s=wt._REMEASURE_RESERVE_S + 1e-9)  # route 1 dropped, 2 idle
    while wt._DEFERRED:
        wt.calibrate_deferred(0)                   # route 2 raced: it is the newest now
    report = wt.remeasure(budget_s=1e9)
    assert first["discarded"]["key"] == "route|1"
    assert [m["key"] for m in report["measured"]] == ["route|1", "route|2"]
    # And what a remeasure reached is kept with the profile, for the next session.
    saved = wt.kernel_profile()["remeasured"]
    assert saved["seq"] == 2 and saved["at"] == {"route|1": 2, "route|2": 2}


def test_a_race_is_started_only_when_its_last_duration_fits_and_the_budget_holds(raced):
    raced["cost"].update(slow=0.5)
    wt._RACE_SECONDS.update({("route", 1): 4.0, ("route", 2): 50.0})
    report = wt.remeasure(budget_s=20.0)
    assert report["status"] == "out_of_time" and report["discarded"] is None
    assert [m["key"] for m in report["measured"]] == ["route|1"]
    assert [r["key"] for r in report["continuing"]] == ["route|2"]
    assert report["elapsed_ms"] <= report["budget_ms"]
    # What it took this time is what the next remeasure judges it by.
    assert wt._RACE_SECONDS[("route", 1)] == report["measured"][0]["ms"] / 1000


def test_a_piece_that_does_not_fit_in_what_is_left_is_not_started(monkeypatch):
    import time
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(webio, "cancel_requested", lambda: False)
    monkeypatch.setattr(wt, "_REMEASURING", [1])
    monkeypatch.setattr(wt, "_REMEASURE_UNTIL", [10.0])
    monkeypatch.setattr(wt, "_OutOfTime", wt._out_of_time_class())
    wt._remeasure_checkpoint(9.0)                  # 9 s + the reserve fit in 10
    clock[0] = 1.0
    with pytest.raises(wt._OutOfTime):
        wt._remeasure_checkpoint(9.0)              # 9 s + the reserve do not fit in 9


def test_a_composite_search_cut_short_reports_the_remeasure_as_cut_short(raced, monkeypatch):
    monkeypatch.setattr(wt, "_REMEASURE_DETAIL", {})
    raced["cost"].update(slow=0.5)
    plan = ("plan", 1)

    def search():
        # Kept one change, then the budget ended it between two pieces of its work.
        wt._TUNED[plan] = "b"
        wt._note_remeasured(plan, "a", "b", "host", 1.0,
                            detail={"tried": 2, "adopted": 1, "stopped": "out_of_time"})
    wt._TUNED[plan] = "a"
    wt.register_remeasure(plan, search)
    wt._REMEASURED_AT.update({"route|1": 5, "route|2": 5})     # the plan is the oldest
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "out_of_time"
    assert [m["key"] for m in report["measured"]] == ["plan|1"]
    assert report["measured"][0]["detail"]["stopped"] == "out_of_time"
    assert wt._TUNED[plan] == "b"                               # what it won stays
    # Out of time: the operator races it did not reach go on when idle, as after any other.
    assert [r["key"] for r in report["continuing"]] == ["route|1", "route|2"]


def test_a_newer_remeasure_drops_what_an_older_one_left_for_idle(raced):
    wt.remeasure(budget_s=wt._REMEASURE_RESERVE_S + 1e-9)
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


def test_raced_again_the_choice_in_use_stays_unless_something_is_proven_faster(raced):
    # A tie now: the static default ("slow", listed first) would have been picked before.
    raced["cost"].update(slow=1.0, fast=1.0)
    report = wt.remeasure(budget_s=1e9)
    assert report["status"] == "complete" and report["changed"] == 0
    assert wt._TUNED[("route", 1)] == wt._TUNED[("route", 2)] == "fast"


def test_an_anchor_beaten_outright_gives_way_to_the_leader():
    times = {"a": [1.0, 1.1, 1.0, 1.05, 1.0], "b": [0.5, 0.52, 0.5, 0.51, 0.5]}
    # The incumbent "c" was dropped by complete separation: the leader takes its place.
    assert wt._anchored_choice(times, ("a", "b"), "c", "a") == "b"
    # In the race and not beaten with evidence: it stays.
    tie = {"a": [1.0, 1.0, 1.0, 1.0, 1.0], "b": [1.0, 1.0, 1.0, 1.0, 1.0]}
    assert wt._anchored_choice(tie, ("a", "b"), "b", "a") == "b"
    # First raced (no incumbent): an inconclusive race keeps the default.
    assert wt._anchored_choice(tie, ("a", "b"), None, "a") == "a"


# ---- two sets of routes: the one in use and the one not in use -----------------------------

def test_a_remeasure_that_changes_routes_keeps_the_set_that_was_in_use(raced):
    raced["cost"].update(slow=0.5)
    hooked = []
    wt.on_routes_changed(lambda keys: hooked.append(set(keys)) or [])
    wt.remeasure(budget_s=1e9)
    sets = wt.route_sets()
    assert sets["in_use"]["id"] == 1 and sets["unused"]["id"] == 0
    assert sets["differ"] == ["route|1", "route|2"]
    out = wt.switch_routes()
    assert out == {"in_use": 0, "unused": 1, "changed": 2, "rebuilds": []}
    assert wt._TUNED[("route", 1)] == wt._TUNED[("route", 2)] == "fast"
    assert hooked[-1] == {("route", 1), ("route", 2)}       # their recordings are rebuilt
    wt.switch_routes()                                       # and back: both are kept
    assert wt._TUNED[("route", 1)] == wt._TUNED[("route", 2)] == "slow"
    assert wt.route_sets()["in_use"]["id"] == 1


def test_the_set_not_in_use_is_the_one_pushed_out(raced):
    raced["cost"].update(slow=0.5)
    wt.remeasure(budget_s=1e9)                   # set 1 (slow) in use, set 0 (fast) not
    wt.switch_routes()                            # set 0 in use, set 1 not
    raced["cost"].update(fast=0.25, slow=0.75)
    wt._TUNED[("route", 1)] = "slow"              # what the next remeasure has to beat
    report = wt.remeasure(budget_s=1e9)
    assert [m["after"] for m in report["measured"]] == ["fast", "fast"]
    # What it made is in use; the set that was in use is the one not in use; set 1, not
    # in use when it began, is gone.
    sets = wt.route_sets()
    assert sets["in_use"]["id"] == 2 and sets["unused"]["id"] == 0
    assert wt._ROUTE_ALT == {("route", 1): "slow"}
    wt.switch_routes()
    assert wt._TUNED[("route", 1)] == "slow" and wt._TUNED[("route", 2)] == "fast"


def test_one_set_raises_rather_than_pretending_to_switch(raced):
    assert wt.route_sets()["unused"] is None
    with pytest.raises(RuntimeError, match="one set of routes"):
        wt.switch_routes()


def test_slower_in_use_is_told_once_and_nothing_is_switched(raced):
    raced["cost"].update(slow=0.5)
    for _ in range(3):
        wt.note_speed("decode:replay", 0.10)      # set 0, in use before the remeasure
    report = wt.remeasure(budget_s=1e9)
    told = []
    wt.on_routes_slower(told.append)
    clock = raced["clock"]
    # Right after a remeasure the device is still hot: nothing counts until it has rested.
    wt.note_speed("decode:replay", 0.30)
    assert wt.route_sets()["in_use"]["speeds"] == {}
    clock[0] += report["elapsed_ms"] / 1000 + 1
    for s_ in (0.12, 0.13, 0.11):
        wt.note_speed("decode:replay", s_)
    assert len(told) == 1 and told[0]["kind"] == "decode:replay"
    assert told[0]["in_use"] == 1 and told[0]["unused"] == 0
    assert told[0]["unused_s"] == [0.1, 0.1, 0.1]
    wt.note_speed("decode:replay", 0.14)
    assert len(told) == 1                                    # once per set and kind
    assert wt.route_sets()["in_use"]["id"] == 1              # the caller decides


def test_not_slower_in_each_of_three_is_not_told(raced):
    raced["cost"].update(slow=0.5)
    for _ in range(3):
        wt.note_speed("decide:256", 0.040)
    report = wt.remeasure(budget_s=1e9)
    raced["clock"][0] += report["elapsed_ms"] / 1000 + 1
    told = []
    wt.on_routes_slower(told.append)
    for s_ in (0.050, 0.039, 0.060):              # one is not slower than every one before
        wt.note_speed("decide:256", s_)
    assert not told


def test_a_composite_choice_is_put_in_force_when_its_set_is(raced):
    plan = ("plan", 1)
    applied = []
    wt._TUNED[plan] = {"plan": "a"}

    def search():
        wt._TUNED[plan] = {"plan": "b"}
        wt._note_replaced(plan, {"plan": "a"})
        wt._note_remeasured(plan, "a", "b", "host", 1.0)
    wt.register_remeasure(plan, search)
    wt.register_route_apply(plan, lambda: applied.append(wt._TUNED.get(plan)))
    wt.remeasure(budget_s=1e9)
    wt.switch_routes()
    assert applied == [{"plan": "a"}]
    wt.switch_routes()
    assert applied[-1] == {"plan": "b"}


def test_both_sets_are_kept_with_the_profile(raced):
    raced["cost"].update(slow=0.5)
    wt.remeasure(budget_s=1e9)
    wt.switch_routes()
    saved = wt.kernel_profile()
    assert saved["route_ids"] == {"in_use": 0, "unused": 1}
    assert saved["alternate"] == {"route|1": "slow", "route|2": "slow"}
