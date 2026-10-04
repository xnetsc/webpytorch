"""A streamed decode must consume the graph-recording execution exactly once."""

import numpy as np

from webtorch import llm


class _Logits:
    def __init__(self, model):
        self.model = model

    def numpy(self):
        self.model.reads = getattr(self.model, "reads", 0) + 1
        values = np.zeros((1, 32), np.float32)
        values[0, 7 + self.model.executions] = 1.0
        return values


class _Platform:
    def __init__(self, model):
        self.model = model
        self.replays = 0
        self.recordings = 0

    def beginCapture(self, _name):
        self.recordings += 1

    def endCapture(self):
        pass

    def replay(self, _name):
        self.replays += 1
        self.model.executions += 1


class _Decoder:
    def push(self, values):
        return str(values[0]) + " "

    def flush(self):
        return ""


class _Tokenizer:
    eot = -1

    def stream_decoder(self):
        return _Decoder()

    def decode(self, values):
        return " ".join(str(value) for value in values)


class _Model:
    tok = _Tokenizer()
    NH = NKV = HD = 1
    kv_cap = 32
    _KV_HEADROOM = 4
    decode_plan = {}

    def __init__(self):
        self.executions = 0
        self.committed = None
        self.capture = True

    def _check_live(self):
        pass

    def _reset_linear_state(self):
        pass

    def _plan_length(self, ids, max_new, *_):
        return ids, max_new

    def _set_sampling(self, *_args, **_kwargs):
        pass

    def _tool_name_constraint(self, _tools, constraint):
        return constraint

    def _capturable(self):
        return self.capture

    def _kv_growing(self, _ids):
        return object(), 0

    def _kv_forward(self, _ids, _position, _cache):
        self.executions += 1
        return 6 + self.executions

    def _kv_reserve(self, _length):
        pass

    def _kv_prefix(self, _ids, *_args):
        return 0

    def _prefill(self, _ids, embeds=None, start=0):
        return 7

    def _device_greedy_ok(self):
        return False

    def _set_inputs(self, _token, _position):
        pass

    def _decode_fwd(self):
        self.executions += 1
        return _Logits(self)

    def _pick(self, logits):
        return int(np.argmax(logits))

    def _stop_now(self):
        return False

    def _kv_commit(self, held, *_args):
        self.committed = list(held)


def test_stream_consumes_initial_recording_without_replay(monkeypatch):
    model = _Model()
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))

    pieces = list(llm.CausalLM._stream_raw(model, ids=[1], max_new=3))

    assert pieces == ["7 ", "8 ", "9 "]
    assert model.executions == 2  # one capture + one distinct-position replay
    assert platform.replays == 1
    assert model.committed == [1, 7, 8]  # the final token needs no KV forward


def test_one_token_stream_does_not_replay_or_select_an_unused_token(monkeypatch):
    model = _Model()
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))

    assert list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1)) == ["7 "]
    assert model.executions == 1
    assert platform.replays == 0
    assert model.committed == [1, 7]


def test_full_attention_reuses_a_matching_decode_graph_across_streams(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))

    first = list(llm.CausalLM._stream_raw(model, ids=[1], max_new=3))
    second = list(llm.CausalLM._stream_raw(model, ids=[1], max_new=3))

    assert first == ["7 ", "8 ", "9 "]
    assert second == ["7 ", "10 ", "11 "]
    assert model.executions == 4
    assert platform.recordings == 1
    assert platform.replays == 3  # one in the first reply, two in the second


def test_decode_graph_rebuilds_when_split_or_buffers_change(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)
    split = [1]
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: split[0])
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))

    list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1))
    split[0] = 2
    list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1))
    model.kv_cap *= 2
    list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1))
    assert platform.recordings == 3


def test_recurrent_state_does_not_reuse_a_prior_turns_graph(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = [{"linear": object()}]
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))

    list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1))
    list(llm.CausalLM._stream_raw(model, ids=[1], max_new=1))
    assert platform.recordings == 2


def test_nonstreaming_api_reuses_the_same_compatible_graph(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)

    first = llm.CausalLM.generate(model, ids=[1], max_new=3)
    second = llm.CausalLM.generate(model, ids=[1], max_new=3)

    assert first.tokens == [7, 8, 9]
    assert second.tokens == [7, 10, 11]
    assert platform.recordings == 1
    assert platform.replays == 3


def test_reused_first_step_synchronizes_before_reporting_first_token():
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)

    llm.CausalLM._capture_or_replay_decode(model, platform, 1)
    reads_after_capture = model.reads
    llm.CausalLM._capture_or_replay_decode(model, platform, 1)

    assert platform.recordings == 1
    assert platform.replays == 1
    assert model.reads == reads_after_capture + 1


def test_one_cached_prefix_row_reuses_compatible_graph(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)

    llm.CausalLM._capture_or_replay_decode(model, platform, 1)
    model._prefill = lambda *_args, **_kw: (_ for _ in ()).throw(
        AssertionError("compatible graph should replace general prefill"))
    result = llm.CausalLM._prefill_or_replay_one(model, [99], start=4)

    assert result == 9  # the replayed graph's logits, not stale capture logits
    assert platform.recordings == 1
    assert platform.replays == 1
    assert model.reads == 2  # synchronised before returning the sampled token
    assert model._last_prefill_route == "decode_replay"


def test_one_row_replay_falls_back_at_each_capability_boundary(monkeypatch):
    model = _Model()
    model._gpu = True
    model.layers = []
    platform = _Platform(model)
    monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    llm.CausalLM._capture_or_replay_decode(model, platform, 1)
    model._prefill = lambda *_args, **_kw: 7

    assert llm.CausalLM._prefill_or_replay_one(model, [99], start=0) == 7
    assert llm.CausalLM._prefill_or_replay_one(model, [99, 100], start=4) == 7
    assert llm.CausalLM._prefill_or_replay_one(
        model, [99], embeds=np.zeros((1, 1), np.float32), start=4) == 7
    model.kv_cap *= 2
    assert llm.CausalLM._prefill_or_replay_one(model, [99], start=4) == 7
    model.kv_cap //= 2
    model.layers = [{"linear": object()}]
    assert llm.CausalLM._prefill_or_replay_one(model, [99], start=4) == 7
    model.layers = []
    model._gpu = False  # WebGL and CPU retain the equivalent general prefill.
    assert llm.CausalLM._prefill_or_replay_one(model, [99], start=4) == 7
    assert platform.replays == 0
    assert model._last_prefill_route == "general"


def test_stream_and_nonstream_use_one_row_replay_at_their_public_api(monkeypatch):
    monkeypatch.setattr(llm.wt, "_set_split", lambda _value: None)
    monkeypatch.setattr(llm.wt, "gqa_tune", lambda *_args: 1)
    monkeypatch.setattr(llm, "_pin_stats", lambda: (0, 0, 0, 0, 0))
    for stream in (True, False):
        model = _Model()
        model._gpu = True
        model.layers = []
        platform = _Platform(model)
        monkeypatch.setattr(llm.wt, "_adam_kernel", {"platform": platform})
        first = (list(llm.CausalLM._stream_raw(model, ids=[1, 2], max_new=1))
                 if stream else llm.CausalLM.generate(model, ids=[1, 2], max_new=1).tokens)
        model._kv_prefix = lambda _ids, *_args: 1
        second = (list(llm.CausalLM._stream_raw(model, ids=[1, 2], max_new=1))
                  if stream else llm.CausalLM.generate(model, ids=[1, 2], max_new=1).tokens)
        assert first == (["7 "] if stream else [7])
        assert second == (["9 "] if stream else [9])
        assert platform.recordings == 1
        assert platform.replays == 2  # prefill row, then first decode step
        assert model._last_prefill_route == "decode_replay"
        if stream:
            assert model.last_stream["prefill_route"] == "decode_replay"


def test_non_capture_stream_stops_before_unneeded_forward():
    model = _Model()
    model.capture = False

    assert list(llm.CausalLM._stream_raw(model, ids=[1], max_new=3)) == [
        "7 ", "8 ", "9 "]
    assert model.executions == 3  # prefill plus the two needed next tokens
    assert model.committed == [1, 7, 8]
