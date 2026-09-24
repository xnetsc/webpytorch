import numpy as np

from webtorch._core import Tensor
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


def test_batched_decisions_keep_encoder_rows_on_device_until_scoring():
    class Encoder:
        def __init__(self):
            self.device_calls = []

        def _encode_many_device(self, seqs):
            self.device_calls.append(seqs)
            padded = max(map(len, seqs))
            rows = np.zeros((len(seqs), padded, 1), dtype=np.float32)
            for index, seq in enumerate(seqs):
                rows[index, :len(seq), 0] = seq
            return Tensor(rows.reshape(-1, 1)), list(map(len, seqs)), padded

        def encode_many(self, _seqs):
            raise AssertionError("device batch must not be read back through encode_many")

    class Config:
        qtypes = ["noul"]

    class Model:
        _raw_questions = DecisionModel._raw_questions

        def __init__(self):
            self.enc = Encoder()
            self.cfg = Config()
            self.scored = []

        @staticmethod
        def batch_pays(_longest, _count):
            return True

        def _score(self, hidden, _markers, _qtype):
            self.scored.append(hidden.numpy().reshape(-1).tolist())
            return np.asarray([0.0, 1.0]), 0.75

    q = {"type": "noul"}
    built = [
        ("short", q, "noul", [1, 2], [0, 1], ["false", "true"]),
        ("long", q, "noul", [3, 4, 5], [0, 1], ["false", "true"]),
    ]
    execution = {}
    model = Model()
    model._raw_questions(built, execution)

    assert model.enc.device_calls == [[[1, 2], [3, 4, 5]]]
    assert model.scored == [[1.0, 2.0], [3.0, 4.0, 5.0]]
    assert execution == {"encoder_tokens": 6, "encoder_passes": 1, "batched": True}
