from types import SimpleNamespace

import numpy as np

from webtorch.llm import CausalLM


def _model(**sampling):
    model = object.__new__(CausalLM)
    model._sampling = sampling
    model._seen = []
    model._gen_start = 0
    model.tok = SimpleNamespace(eos_ids=[2])
    return model


def test_device_greedy_only_covers_semantically_plain_argmax():
    assert _model()._device_greedy_ok()
    assert not _model(do_sample=True)._device_greedy_ok()
    assert not _model(repetition_penalty=1.1)._device_greedy_ok()
    assert not _model(presence_penalty=0.1)._device_greedy_ok()
    assert not _model(frequency_penalty=0.1)._device_greedy_ok()
    assert not _model(constraint=object())._device_greedy_ok()


def test_minimum_length_keeps_the_full_logit_eos_mask_until_satisfied():
    model = _model(min_new_tokens=2)
    assert not model._device_greedy_ok()
    model._seen[:] = [7, 8]
    assert model._device_greedy_ok()


def test_device_token_acceptance_updates_generation_state():
    model = _model()
    assert model._accept_token(17) == 17
    assert model._seen == [17]


def test_chunk_routes_use_the_same_token_state_transition_as_scalar_replay():
    source = __import__("inspect").getsource(CausalLM)
    assert source.count("current = self._accept_token(int(token))") == 2
    assert "replay-chunk%d" in source
    assert source.count("_decode_rate(t_first, steps + 1)") == 1
    assert source.count("self._greedy_chunks(current, pos, chunk, max_new - ") == 2
    assert "_release_idle_greedy_capture" in source


def test_chunk_capture_is_reused_until_kv_buffers_move():
    capture = __import__("inspect").getsource(CausalLM._capture_greedy_chunk)
    reserve = __import__("inspect").getsource(CausalLM._kv_reserve)
    tune = __import__("inspect").getsource(CausalLM._tune_greedy_chunks)
    assert 'name="decode_chunk"' in capture
    assert "self._greedy_chunk_capture_ready = True" in capture
    assert "self._greedy_chunk_capture_ready = False" in reserve
    assert "chosen[0], row_execution=chosen[1]" in tune


def test_embedding_row_layout_is_exact_gated_locally_and_upper_overridable():
    import inspect
    from webtorch import _core as wt
    assert "nrow*im.N+wo" in wt._Q6K_DECODE_ROW_INPUT_WGSL
    local = inspect.getsource(CausalLM._tune_embedding_row)
    assert "np.array_equal" in local
    assert "_measured_choice" in local
    capture = inspect.signature(CausalLM._capture_greedy_chunk)
    assert "row_execution" in capture.parameters
    upper = inspect.getsource(CausalLM._tune_greedy_chunks)
    assert "for row_route in self._embedding_row_candidates()" in upper
    assert "ranked = sorted(valid" in upper
    assert "_paired_faster(samples, candidate, chosen)" in upper
    assert "greedy_chunk_candidates_ms" in upper


def test_chunk_capability_is_format_and_shape_driven_not_model_named():
    source = __import__("inspect").getsource(CausalLM._can_chunk_greedy)
    assert 'type_name == "Q6_K"' in source
    assert "_tied_embed_head" in source
    assert "model_name" not in source
    assert "qwen" not in source.lower()


class _Device:
    """Stands in for the GPU under `_greedy_chunks`: chunk k (from 0) yields tokens
    100k+1 .. 100k+chunk, a staged read copies the token buffer as it is at that point of
    the queue, and the log says what was queued, staged, collected and when."""

    def __init__(self, chunk):
        self.chunk, self.made, self.slots, self.log = chunk, 0, {}, []

    def produce(self):
        k = self.made; self.made += 1
        return np.array([0] + [100 * k + i + 1 for i in range(self.chunk)], np.int32)

    def replayStaged(self, name, buffer_id, nbytes, stage, collect):
        assert buffer_id == 11 and nbytes == (self.chunk + 1) * 4
        if name:
            self.last = self.produce()
        if stage >= 0:
            assert stage not in self.slots, "a slot was staged again before it was collected"
            self.slots[stage] = self.last.tobytes()
        self.log.append((bool(name), stage, collect))
        if collect < 0:
            return None
        data = self.slots.pop(collect)
        return SimpleNamespace(to_bytes=lambda: data)

    def in_flight(self):
        return len(self.slots)


def _driver(monkeypatch, chunk=4, kv_cap=64, ready=True):
    from webtorch import _core as wt
    dev = _Device(chunk)
    monkeypatch.setitem(wt._adam_kernel, "platform", dev)
    model = object.__new__(CausalLM)
    model.kv_cap = kv_cap
    model._greedy_chunk_capture_ready = ready
    model._chunk_tokens = SimpleNamespace(buffer=SimpleNamespace(buffer_id=11))
    calls = []
    model._set_chunk_inputs = lambda token, pos, count: calls.append(("inputs", token, pos))

    def capture(count, row_execution=None):
        calls.append(("capture", dev.in_flight()))
        model._greedy_chunk_capture_ready = True
        return dev.produce()

    def reserve(need):
        calls.append(("reserve", need, dev.in_flight()))
        model.kv_cap = 2 * need
        model._greedy_chunk_capture_ready = False

    model._capture_greedy_chunk = capture
    model._kv_reserve = reserve
    return model, dev, calls


def test_pipelined_chunks_queue_the_next_before_reading_and_stop_at_the_budget(monkeypatch):
    model, dev, calls = _driver(monkeypatch)
    got = [t.tolist() for t in model._greedy_chunks(7, 20, 4, 10)]
    assert got == [[1, 2, 3, 4], [101, 102, 103, 104], [201, 202, 203, 204]]
    # Queue chunk 0; then each round queues the next and collects the oldest; the round
    # whose queued tokens already cover the budget only collects.
    assert dev.log == [(True, 0, -1), (True, 1, 0), (True, 2, 1), (False, -1, 2)]
    assert calls == [("inputs", 7, 20)]              # the host's token, once
    assert dev.made == 3 and dev.in_flight() == 0


def test_a_caller_that_stops_leaves_at_most_one_chunk_running(monkeypatch):
    model, dev, calls = _driver(monkeypatch)
    for tokens in model._greedy_chunks(7, 20, 4, 1000):
        break
    assert tokens.tolist() == [1, 2, 3, 4]
    assert dev.made == 2 and dev.in_flight() == 1


def test_the_cache_grows_with_nothing_in_flight_and_the_chain_restarts_from_the_last_token(
        monkeypatch):
    model, dev, calls = _driver(monkeypatch, kv_cap=10)
    got = [t.tolist() for t in model._greedy_chunks(7, 0, 4, 16)]
    assert [g[-1] for g in got] == [4, 104, 204, 304]
    # Rows 0..7 fit; the third chunk would not, so it is not queued until the first two
    # are read, the cache has grown, and the graph is recorded against the new buffers.
    assert calls == [("inputs", 7, 0), ("reserve", 12, 0), ("inputs", 104, 8), ("capture", 0)]
    assert dev.in_flight() == 0


def test_a_graph_not_yet_recorded_is_recorded_once_then_replayed(monkeypatch):
    model, dev, calls = _driver(monkeypatch, chunk=2, ready=False)
    got = [t.tolist() for t in model._greedy_chunks(7, 0, 2, 6)]
    assert got == [[1, 2], [101, 102], [201, 202]]
    assert calls == [("inputs", 7, 0), ("capture", 0)]
    assert dev.log == [(True, 0, -1), (True, 1, 0), (False, -1, 1)]


def test_decode_input_takes_its_position_from_the_device_counter_in_the_browser():
    """`q6k_decode_input`: the row of the token in `slot`, decoded from the original Q6_K
    bytes in either layout, and the rotary rows of the position the counter advances to --
    nothing per step from the host, which is what lets one chunk follow another unread."""
    import pytest
    from webtorch import _core as wt
    from webtorch import ggufload
    if not wt._adam_backend_ready():
        pytest.skip("requires the WebGPU browser backend")
    rng = np.random.default_rng(5)
    K, N, HD, rows = 1024, 40, 8, 6
    raw = rng.integers(0, 256, (N, K // 256, 210), dtype=np.uint8)
    d = (rng.random((N, K // 256)) * 0.04 + 0.002).astype(np.float16)
    raw[:, :, 208:210] = d.view(np.uint8).reshape(N, K // 256, 2)
    raw = raw.tobytes()
    ref = ggufload.dequant(ggufload.GGML_IDS["Q6_K"], raw, N * K).reshape(N, K)
    lin = wt.GGMLLinear(raw, "Q6_K", K, N, execution="stored")
    compact = wt.xp.asarray(np.frombuffer(raw, np.int32).copy())
    cos = rng.standard_normal((rows, HD)).astype(np.float32)
    sin = rng.standard_normal((rows, HD)).astype(np.float32)
    ct, st = wt.xp.asarray(cos), wt.xp.asarray(sin)
    tokens = wt.xp.asarray(np.array([0, 17, 3], np.int32))
    h = wt.Tensor(np.zeros((1, K), np.float32))
    cb = wt.Tensor(np.zeros((1, HD), np.float32)); sb = wt.Tensor(np.zeros((1, HD), np.float32))
    ctl = wt.xp.asarray(np.array([-1, 1, 1, HD, 64], np.int32))

    def step(slot, layout, increment=True):
        packed, stride = (lin.packed, lin.Nt) if layout == "transposed" else (compact, K // 256 * 210 // 4)
        wt.q6k_decode_input(tokens, packed, lin.Kt, stride, ct, st, h, cb, sb, ctl,
                            slot=slot, increment=increment, layout=layout)
        return (np.asarray(h.numpy()).reshape(-1), np.asarray(cb.numpy()).reshape(-1),
                np.asarray(sb.numpy()).reshape(-1), int(np.asarray(ctl.get())[0]))

    got, c, s, pos = step(1, "transposed")
    assert np.array_equal(got, ref[17]) and pos == 0
    assert np.array_equal(c, cos[0]) and np.array_equal(s, sin[0])
    got, c, s, pos = step(2, "compact")
    assert np.array_equal(got, ref[3]) and pos == 1 and np.array_equal(c, cos[1])
    got, c, s, pos = step(2, "compact", increment=False)
    assert pos == 1 and np.array_equal(c, cos[1])
    ctl.buffer.set_data(np.array([50, 1, 1, HD, 64], np.int32))
    got, c, s, pos = step(1, "transposed")
    assert pos == 51 and np.array_equal(c, cos[rows - 1]) and np.array_equal(s, sin[rows - 1])
