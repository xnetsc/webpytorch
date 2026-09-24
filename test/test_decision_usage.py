import numpy as np

from webtorch.decision import DecisionModel


def test_decide_reuses_identical_encoder_inputs_and_reports_actual_work():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, ids):
            self.calls.append(tuple(ids))
            return np.asarray(ids)

    class Config:
        qtypes = ["choice", "score", "noul"]

        @staticmethod
        def temp_for(_qtype, _count):
            return 1.0

    class Model:
        decide = DecisionModel.decide
        _raw_questions = DecisionModel._raw_questions

        def __init__(self):
            self.enc = Encoder()
            self.cfg = Config()

        @staticmethod
        def batch_pays(_longest, _count):
            return False

        @staticmethod
        def _score(_hidden, _markers, _qtype):
            return np.asarray([0.0, 1.0]), 0.75

        @staticmethod
        def _prepare_questions(_state, questions):
            ids = [7, 8, 9]
            built = [(qid, q, "noul", ids, [0, 1], ["false", "true"])
                     for qid, q in questions.items()]
            return built, sum(len(row[3]) for row in built)

    questions = {
        "first": {"type": "noul"},
        "same_input": {"type": "noul"},
    }
    model = Model()
    result = model.decide("state", questions)

    assert model.enc.calls == [(7, 8, 9)]
    assert result["answers"]["first"] == result["answers"]["same_input"]
    assert result["usage"] == {
        "input_tokens": 6,
        "output_tokens": 0,
        "questions": 2,
        "sequence_tokens": {"first": 3, "same_input": 3},
        "encoder_tokens": 3,
        "encoder_passes": 1,
        "batched": False,
    }
