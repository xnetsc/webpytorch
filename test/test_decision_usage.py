import numpy as np
from types import SimpleNamespace

from webtorch._core import Tensor
from webtorch.decision import DecisionModel, _named_first
from webtorch import _core as wt
from webtorch.encoder import TextEncoder


def test_named_layout_fallback_keeps_the_model_root():
    assert _named_first(None, ("tokenizer/", "")) == ["tokenizer/", ""]


class _FakePlatform(object):
    """A backend that records and replays, without a device behind it."""

    def __init__(self):
        self.replayed = []
        self.captured = []

    def replay(self, name):
        self.replayed.append(name)

    def beginCapture(self, name):
        self.captured.append(name)

    def endCapture(self):
        pass


def _bucket_of(length):
    """The slot a length lands in -- the same rounding `_replayed` does."""
    b = TextEncoder._BUCKET
    return int(((length + b - 1) // b) * b)


def _with_platform(plat):
    """`_replayed` reaches the backend through the module-level kernel table; swap it for the
    duration of a test and put back whatever was there."""
    before = wt._adam_kernel.get("platform")
    wt._adam_kernel["platform"] = plat
    return before


class _FakeEncoder(object):
    """Enough of a TextEncoder for `_replayed` to run with no backend under it."""

    _BUCKET = TextEncoder._BUCKET
    _CAP_MAX = TextEncoder._CAP_MAX
    _replayed = TextEncoder._replayed

    def __init__(self, hidden=4):
        self.cfg = SimpleNamespace(max_positions=1024)
        self.hidden = hidden
        self._cap = {}
        self._cap_seen = set()
        self.made = []
        self.written = []

    @staticmethod
    def _capture_ok():
        return True

    def _cap_make(self, length):
        self.made.append(length)
        rows = np.arange(length * self.hidden, dtype=np.float32).reshape(length, self.hidden)
        slot = {"T": length, "recorded": True, "out": SimpleNamespace(numpy=lambda: rows),
                "name": "fake%d" % length, "x": None, "masks": {}}
        self._cap[length] = slot
        return slot

    def _cap_write(self, slot, ids, T, valid):
        self.written.append((slot["T"], T))

    def _layers(self, x, masks, B):
        raise AssertionError("a recorded slot must replay, not run the stack again")


def test_encoder_capture_rounds_the_length_up_to_a_bucket():
    enc = _FakeEncoder()
    before = _with_platform(_FakePlatform())
    # First sight of a bucket records nothing: it is the repeat that pays for the capture.
    assert enc._replayed(np.arange(37), 37, 1, None) is None
    assert enc._cap_seen == {_bucket_of(37)}
    assert enc.made == []
    # A DIFFERENT length in the same bucket is that repeat -- which is the whole point of
    # bucketing, since two decision sequences are never the same length twice.
    assert _bucket_of(51) == _bucket_of(37), "pick two lengths that share a bucket"
    try:
        enc._replayed(np.arange(51), 51, 1, None)
    finally:
        wt._adam_kernel["platform"] = before
    assert enc.made == [_bucket_of(37)]


def test_encoder_capture_stays_bounded():
    enc = _FakeEncoder()
    enc._cap = {n: object() for n in range(enc._CAP_MAX)}
    enc._cap_seen.add(_bucket_of(53))
    assert enc._replayed(np.arange(53), 53, 1, None) is None
    assert enc.made == []


def test_encoder_replay_answers_for_the_ids_it_was_given():
    """A padded pass is bit-identical on the real positions, but it has MORE rows, and the
    decision head's attention has no mask -- so handing the padding on moves its logits."""
    enc = _FakeEncoder(hidden=3)
    T = 37
    Tb = _bucket_of(T)
    enc._cap_seen.add(Tb)                    # pretend the bucket has been seen once
    plat = _FakePlatform()
    before = _with_platform(plat)
    try:
        got = enc._replayed(np.arange(T), T, 1, None)
    finally:
        wt._adam_kernel["platform"] = before
    assert plat.replayed == ["fake%d" % Tb]
    assert enc.made == [Tb]
    assert enc.written == [(Tb, T)]
    rows = np.asarray(got.numpy()) if hasattr(got, "numpy") else np.asarray(got.data)
    assert rows.shape == (T, 3)
    whole = np.arange(Tb * 3, dtype=np.float32).reshape(Tb, 3)
    assert np.array_equal(rows, whole[:T])


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


def test_releasing_a_model_hands_memory_back_even_when_no_known_name_matched():
    """`_HEAVY` is a list of attribute names, and a decision model's weights hang off `enc`,
    which is not one of them. Taking "nothing recognised" for "nothing to give back" left
    459 MB and 903 buffers on the device after `release()` had returned."""
    from webtorch import _core, _sdk

    class Weighty(object):
        """No `_HEAVY` name anywhere in here, and it knows that itself."""

        def __init__(self):
            self.enc = object()
            self.dropped = False

        def release(self):
            self.enc = None
            self.dropped = True

    calls = []
    before = _core._gpu_release_memory
    _core._gpu_release_memory = lambda: calls.append(1)
    try:
        obj = Weighty()
        assert _sdk._free(obj) is True
        assert obj.dropped and obj.enc is None
        assert calls == [1], "the device was never told, so the pins were never let go"
        # And the plain path, where a name does match, still only tells it once.
        calls.clear()
        plain = type("Plain", (), {})()
        plain.layers = [1, 2, 3]
        assert _sdk._drop_heavy(plain) is True
        assert calls == [1]
    finally:
        _core._gpu_release_memory = before


def test_a_decision_model_drops_its_encoder_and_its_tensors():
    from webtorch.decision import DecisionModel

    enc = _FakeEncoder()
    enc.released = False

    def release():
        enc.released = True
    enc.release = release

    m = DecisionModel.__new__(DecisionModel)
    m.__dict__.update(enc=enc, _ten={"a": object()}, _src={"b": object()})
    m.release()
    assert enc.released
    assert m.__dict__["enc"] is None and m.__dict__["_ten"] == {}
    assert m.__dict__["_released"] is True
