import math
import warnings

import numpy as np

from webtorch.decision import (
    DecisionConfig,
    DecisionModel,
    calibration_error,
    fit_temperature,
    safe_temperature,
)


def test_temperature_guard_handles_extreme_and_invalid_values():
    assert safe_temperature(0.1006) == 0.5
    assert safe_temperature(99) == 5.0
    assert safe_temperature("bad") == 1.0
    assert safe_temperature(float("nan")) == 1.0

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        cfg = DecisionConfig({
            "temperature": [1.9, 1.2, 2.0],
            "temperature_by_options": {"choice:11+": 0.1006},
        })
    assert cfg.temp_for("choice", 20) == 0.5
    assert cfg.temperature_by_options_raw["choice:11+"] == 0.1006
    assert cfg.calibration()["status"] == "guarded"
    assert cfg.calibration()["domain_calibrated"] is False
    assert len(cfg.calibration()["adjustments"]) == 1
    assert "uncalibrated" in str(seen[0].message)


def test_ece_means_ninety_percent_is_right_nine_times_out_of_ten():
    probabilities = [np.array([0.9, 0.1])] * 10
    targets = [0] * 9 + [1]
    assert calibration_error(probabilities, targets) < 1e-12


def test_temperature_fit_repairs_overconfidence_without_changing_winners():
    # Eight right and four wrong predictions, all with the same much-too-sharp margin.
    rows = [np.array([4.0, 0.0])] * 8 + [np.array([0.0, 4.0])] * 4
    targets = [0] * 8 + [0] * 4
    winners = [int(x.argmax()) for x in rows]

    report = fit_temperature(rows, targets)

    assert 1.0 < report["temperature"] <= 5.0
    assert report["nll_after"] < report["nll_before"]
    assert report["ece_after"] < report["ece_before"]
    assert winners == [int((x / report["temperature"]).argmax()) for x in rows]
    assert math.isfinite(report["temperature"])


def test_temperature_fit_accepts_mixed_width_rows_in_one_bucket():
    rows = [np.array([3.0, 1.0, 0.0]), np.array([2.0, 0.0, 1.0, -1.0]),
            np.array([2.5, 0.0, 1.0, -1.0, -2.0])]
    report = fit_temperature(rows, [0, 2, 0])
    assert report["samples"] == 3
    assert 0.5 <= report["temperature"] <= 5.0


def test_model_calibration_updates_only_the_matching_generic_bucket():
    class FakeDecision:
        calibrate = DecisionModel.calibrate
        _target_index = staticmethod(DecisionModel._target_index)

        def __init__(self):
            self.cfg = DecisionConfig({"temperature": [1.0, 1.0, 1.0]})

        def _prepare_questions(self, state, questions):
            q = questions["route"]
            return [("route", q, "choice", [1, 2], [0, 1], ["billing", "technical"])], 4

        def _raw_questions(self, built):
            qid, q, qtype, _ids, markers, labels = built[0]
            # Deliberately over-confident and sometimes wrong, so the fitted T must soften.
            logits = np.array([4.0, 0.0]) if q["instructions"] != "wrong" else np.array([0.0, 4.0])
            return [(qid, q, qtype, markers, labels, logits, 1.0)]

    question = lambda text: {"route": {"type": "choice", "instructions": text,
                                        "criteria": {"billing": None, "technical": None}}}
    examples = [
        {"state": "x", "questions": question("right"), "answers": {"route": "billing"}},
        {"state": "x", "questions": question("right"), "answers": {"route": "billing"}},
        {"state": "x", "questions": question("wrong"), "answers": {"route": "billing"}},
    ]
    model = FakeDecision()
    report = model.calibrate(examples, min_samples=3)
    assert report["groups"]["choice:2"]["temperature"] > 1.0
    assert model.cfg.temp_for("choice", 2) == report["groups"]["choice:2"]["temperature"]
    assert model.cfg.calibration()["status"] == "held-out"
    assert model.cfg.calibration()["groups"] == ["choice:2"]
