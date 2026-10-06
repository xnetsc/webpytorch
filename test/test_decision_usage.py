import numpy as np
import pytest
from types import SimpleNamespace

from webtorch._core import Tensor
from webtorch.decision import DecisionConfig, DecisionModel, _named_first
from webtorch import _core as wt
from webtorch.encoder import TextEncoder


def test_webgl_half_weight_texture_preserves_logical_rows_when_possible():
    assert wt._webgl_half_texture_extent(768, 2304, 8192) == (2304, 768)
    assert wt._webgl_half_texture_extent(8192, 1, 8192) == (1, 8192)
    # An axis too wide for the device retains the general flat layout.
    assert wt._webgl_half_texture_extent(16, 9000, 8192) == (8192, 18)


def test_batched_matmul_never_silently_becomes_per_head_python_loop():
    lhs = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    rhs = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    np.testing.assert_array_equal(wt._bmm_raw(lhs, rhs), lhs @ rhs)

    class BrokenBatch:
        def __matmul__(self, _other):
            raise RuntimeError('native batched kernel failed')

        def __getitem__(self, _index):
            raise AssertionError('must not slice into serial per-head matmuls')

    with pytest.raises(RuntimeError, match='native batched kernel failed'):
        wt._bmm_raw(BrokenBatch(), rhs)


def test_decision_warmup_failure_is_reported_instead_of_skipped():
    from webtorch.decision import _warm_decision

    class BrokenModel:
        def decide(self, *_args):
            raise RuntimeError("warm forward failed")

    stages = []
    with pytest.raises(RuntimeError, match="warm forward failed"):
        _warm_decision(BrokenModel(), SimpleNamespace(qtypes=["choice"]),
                       SimpleNamespace(load_stage=stages.append))
    assert stages == ["warm"]


def test_named_layout_fallback_keeps_the_model_root():
    assert _named_first(None, ("tokenizer/", "")) == ["tokenizer/", ""]


def test_shared_state_is_tokenized_once_for_multiple_question_rows():
    class Tokenizer:
        dec = {}

        def __init__(self):
            self.calls = []

        def encode(self, value):
            self.calls.append(value)
            return [len(word) + 10 for word in value.split()]

    class Model:
        _prepare_questions = DecisionModel._prepare_questions
        build_sequence = DecisionModel.build_sequence

        def __init__(self):
            self.tok = Tokenizer()
            self.cfg = DecisionConfig({"max_len": 100, "head_max_len": 40,
                                       "question_types": ["choice"]})
            self.mask_id, self.cls_id, self.sep_id = 9, 1, 2

    model = Model()
    questions = {
        "billing": {"type": "choice", "instructions": "which team", "criteria": ["yes", "no"]},
        "urgent": {"type": "choice", "instructions": "how urgent", "criteria": ["today", "later"]},
    }
    built, _ = model._prepare_questions("one shared state", questions)
    assert len(built) == 2
    assert model.tok.calls.count("one shared state") == 1
    assert built[0][3][-4:] == built[1][3][-4:]


def test_decision_layout_comes_from_checkpoint_metadata():
    declared = DecisionConfig({"question_layout": "per_question"})
    assert (declared.question_layout, declared.question_layout_source) == (
        "per_question", "checkpoint_metadata")
    joint = DecisionConfig({"question_layout": "joint"})
    assert (joint.question_layout, joint.question_layout_source) == (
        "joint", "checkpoint_metadata")
    assert DecisionConfig({}).question_layout is None
    assert DecisionConfig({"head_max_len": 256, "max_prefixes": 6}).question_layout is None


def test_legacy_question_layout_requires_checkpoint_schema_not_a_model_name():
    config = {"max_len": 1024, "head_max_len": 256,
              "head_layers": 2, "max_prefixes": 6}
    tensors = {name: None for name in (
        "type_emb.weight", "scorer.0.weight", "scorer.3.weight",
        "act_head.0.weight", "act_head.2.weight",
        "head.layers.0.self_attn.in_proj_weight")}
    inferred = DecisionConfig(dict(config, model_name="arbitrary"), weight_names=tensors)
    assert (inferred.question_layout, inferred.question_layout_source) == (
        "per_question", "checkpoint_structure")
    assert DecisionConfig(dict(config, model_name="laya"),
                          weight_names={}).question_layout is None
    tensors.pop("scorer.3.weight")
    assert DecisionConfig(config, weight_names=tensors).question_layout is None


def test_ambiguous_or_joint_layout_cannot_silently_run_independent_rows():
    import pytest

    for layout in (None, "joint"):
        cfg = DecisionConfig({} if layout is None else {"question_layout": layout})
        with pytest.raises((ValueError, NotImplementedError), match="question_layout"):
            DecisionModel(None, cfg, {}, None, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="unsupported decision question_layout"):
        DecisionConfig({"question_layout": "guess"})


class _FakePlatform(object):
    """A backend that records and replays, without a device behind it."""

    def __init__(self):
        self.replayed = []
        self.captured = []
        self.released = []

    def replay(self, name):
        self.replayed.append(name)

    def beginCapture(self, name):
        self.captured.append(name)

    def endCapture(self):
        pass

    def releaseCapture(self, name):
        self.released.append(name)


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

    @staticmethod
    def _packed_ok():
        return False                 # the padded pass, which these tests are about

    @staticmethod
    def _capture_platform():
        return wt._adam_kernel["platform"]

    def _cap_make(self, length):
        self.made.append(length)
        rows = np.arange(length * self.hidden, dtype=np.float32).reshape(length, self.hidden)
        slot = {"T": length, "recorded": True, "out": Tensor(rows),
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
    try:
        assert enc._replayed(np.arange(37), 37, 1, None) is None
        assert enc._cap_seen == {_bucket_of(37)}
        assert enc.made == []
        # Different lengths must share this bucket even if its width changes.
        assert _bucket_of(39) == _bucket_of(37)
        enc._replayed(np.arange(39), 39, 1, None)
    finally:
        wt._adam_kernel["platform"] = before
    assert enc.made == [_bucket_of(37)]


def test_encoder_capture_stays_bounded():
    enc = _FakeEncoder()
    enc._cap = {n: {"name": "old%d" % n} for n in range(enc._CAP_MAX)}
    enc._cap_seen.add(_bucket_of(53))
    plat = _FakePlatform()
    before = _with_platform(plat)
    try:
        enc._replayed(np.arange(53), 53, 1, None)
    finally:
        wt._adam_kernel["platform"] = before
    assert len(enc._cap) == enc._CAP_MAX
    assert 0 not in enc._cap and _bucket_of(53) in enc._cap
    assert plat.released == ["old0"]
    assert enc.made == [_bucket_of(53)]


def test_encoder_capture_eviction_keeps_recently_used_shape():
    enc = _FakeEncoder()
    hot = _bucket_of(53)
    enc._cap = {n: {"name": "old%d" % n, "T": n, "recorded": True,
                    "out": Tensor(np.zeros((n, enc.hidden), np.float32))}
                for n in (32, hot, 96, 128)}
    enc._cap_seen.add(160)
    plat = _FakePlatform()
    before = _with_platform(plat)
    try:
        enc._replayed(np.arange(53), 53, 1, None)
        enc._replayed(np.arange(150), 150, 1, None)
    finally:
        wt._adam_kernel["platform"] = before
    assert plat.released == ["old32"]
    assert hot in enc._cap and 160 in enc._cap


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


def test_batched_encoder_capture_trims_each_row_without_mixing_questions():
    class BatchEncoder(_FakeEncoder):
        def _cap_make(self, length, B=1):
            rows = np.arange(B * length * self.hidden, dtype=np.float32)
            slot = {"T": length, "recorded": True,
                    "out": Tensor(rows.reshape(B * length, self.hidden)),
                    "name": "batch%d_%d" % (B, length), "x": None, "masks": {}}
            self._cap[(B, length)] = slot
            return slot

        def _cap_write(self, slot, ids, T, valid, B=1):
            self.written.append((B, T, np.asarray(ids).reshape(B, T).tolist(),
                                 np.asarray(valid).reshape(B, T).tolist()))

    enc = BatchEncoder(hidden=3)
    B, T, Tb = 2, 37, _bucket_of(37)
    enc.cfg.hidden = enc.hidden
    enc.cfg.heads = 2
    enc.cfg.layer_types = ["full_attention"]
    enc._cap_seen.add((B, Tb))
    plat = _FakePlatform()
    before = _with_platform(plat)
    ids = np.arange(B * T, dtype=np.int64)
    valid = np.ones((B, T), dtype=np.int64)
    valid[0, -1] = 0
    try:
        out = enc._replayed(ids, T, B, valid)
    finally:
        wt._adam_kernel["platform"] = before
    assert plat.replayed == ["batch2_%d" % Tb]
    assert enc.written == [(B, T, ids.reshape(B, T).tolist(), valid.tolist())]
    assert out.shape == (B * T, 3)
    expected = np.arange(B * Tb * 3, dtype=np.float32).reshape(B, Tb, 3)[:, :T]
    np.testing.assert_array_equal(out.numpy(), expected.reshape(B * T, 3))


def test_batched_capture_rewrites_new_questions_and_each_attention_mask():
    class Buffer:
        def __init__(self):
            self.writes = []

        def set_data(self, value):
            self.writes.append(np.asarray(value).copy())

    class Target:
        def __init__(self):
            self.data = SimpleNamespace(buffer=Buffer())

    enc = TextEncoder.__new__(TextEncoder)
    enc.cfg = SimpleNamespace(pad_id=0, hidden=2, heads=2, window=None)
    enc._embed_rows = lambda ids: np.repeat(np.asarray(ids)[:, None], 2, axis=1)
    enc._mask_array = TextEncoder._mask_array.__get__(enc)
    slot = {"T": 4, "x": Target(), "masks": {"full_attention": Target()}}
    enc._cap_write(slot, [1, 2, 3, 4, 5, 6], 3, [[1, 1, 0], [1, 1, 1]], B=2)
    enc._cap_write(slot, [7, 8, 9, 10, 11, 12], 3, [[1, 1, 1], [1, 0, 1]], B=2)
    first, second = slot["x"].data.buffer.writes
    np.testing.assert_array_equal(first.reshape(2, 4, 2)[:, :, 0],
                                  [[1, 2, 3, 0], [4, 5, 6, 0]])
    np.testing.assert_array_equal(second.reshape(2, 4, 2)[:, :, 0],
                                  [[7, 8, 9, 0], [10, 11, 12, 0]])
    first_mask, second_mask = [m.reshape(4, 4, 4)
                               for m in slot["masks"]["full_attention"].data.buffer.writes]
    np.testing.assert_array_equal(first_mask[0, 0], [0, 0, -1e9, -1e9])
    np.testing.assert_array_equal(first_mask[2, 0], [0, 0, 0, -1e9])
    np.testing.assert_array_equal(second_mask[0, 0], [0, 0, 0, -1e9])
    np.testing.assert_array_equal(second_mask[2, 0], [0, -1e9, 0, -1e9])


def test_float16_capture_passes_zero_copy_byte_view_to_js(monkeypatch):
    import sys

    seen = {}

    def stage(x_id, mask_ids, ids, valid, table, kind, *shape):
        seen.update(x_id=x_id, mask_ids=mask_ids, ids=ids, valid=valid,
                    table=table, kind=kind, shape=shape)

    monkeypatch.setitem(sys.modules, "js", SimpleNamespace(
        gpu=SimpleNamespace(stageDecisionCapture=stage)))
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    enc = TextEncoder.__new__(TextEncoder)
    enc.cfg = SimpleNamespace(pad_id=0, hidden=2, vocab=3, heads=1, window=None)
    source = np.asarray([[0, 0], [3, 4], [7, 8]], dtype=np.float16)
    enc._emb_host = source
    target = lambda ident: SimpleNamespace(data=SimpleNamespace(
        buffer=SimpleNamespace(buffer_id=ident)))
    slot = {"T": 2, "x": target(7), "masks": {"full_attention": target(9)}}
    ids = np.asarray([1, 2, 2, 1], dtype=np.int64)
    valid = np.ones(4, dtype=np.int64)
    enc._cap_write(slot, ids, 2, valid, B=2)
    assert seen["kind"] == "f16"
    assert seen["table"].dtype == np.uint8
    assert np.shares_memory(seen["table"], source)
    assert seen["table"].nbytes == source.nbytes
    assert seen["ids"].dtype == np.uint8
    assert seen["valid"].dtype == np.uint8
    assert np.shares_memory(seen["ids"], ids)
    assert np.shares_memory(seen["valid"], valid)
    assert seen["mask_ids"] == {"full_attention": 9}


def test_webgpu_capture_stages_one_mask_plane_per_question_not_per_head(monkeypatch):
    import sys

    calls = []
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setitem(sys.modules, "js", SimpleNamespace(gpu=SimpleNamespace(
        stageDecisionCapture=lambda *args: calls.append(args))))
    enc = TextEncoder.__new__(TextEncoder)
    enc.cfg = SimpleNamespace(pad_id=0, hidden=2, vocab=3, heads=4,
                              window=1, layer_types=["sliding_attention"])
    enc._emb_host = np.zeros((3, 2), dtype=np.float16)
    enc._cap = {}
    slot = enc._cap_make(8, B=2)
    assert slot["mask_heads"] == 1
    assert slot["masks"]["sliding_attention"].shape == (2, 8, 8)
    target = lambda ident: SimpleNamespace(data=SimpleNamespace(
        buffer=SimpleNamespace(buffer_id=ident)))
    slot["x"] = target(7)
    slot["masks"] = {"sliding_attention": target(9)}
    enc._cap_write(slot, np.ones(16, np.int64), 8,
                   np.ones((2, 8), np.int64), B=2)
    assert calls and calls[0][-2:] == (1, 1)


def test_selected_webgpu_qk_layout_fails_instead_of_running_generic_math(monkeypatch):
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setattr(wt, "banded_qk_scores", lambda *_args: None)
    enc = TextEncoder.__new__(TextEncoder)
    enc.cfg = SimpleNamespace(window=2)
    q = Tensor(np.zeros((2, 3, 4), np.float32))
    mask = Tensor(np.zeros((3, 3), np.float32))
    with pytest.raises(RuntimeError, match="WebGPU fused QK"):
        enc._out(q, q, q, mask, "", 3, 2, 4, kind="sliding_attention")


def test_capture_inputs_use_host_values_without_device_roundtrip():
    enc = TextEncoder.__new__(TextEncoder)
    enc.cfg = SimpleNamespace(window=1, heads=2)
    enc.p = "encoder."
    enc._EMB_DEVICE_MAX = 1
    name = "encoder.embeddings.tok_embeddings.weight"
    source = np.arange(40, dtype=np.float16).reshape(10, 4)
    enc.shape_of = {name: source.shape}
    enc._src = {name: source}
    enc._emb_host = None
    gathered = enc._embed_rows(np.asarray([3, 1, 3]))
    np.testing.assert_array_equal(gathered, source[[3, 1, 3]].astype(np.float32))
    assert name not in enc._src
    np.testing.assert_array_equal(enc._mask_array(3, [1, 1, 0], "full_attention"),
                                  enc._mask(3, [1, 1, 0], "full_attention").numpy())
    np.testing.assert_array_equal(enc._mask_array(3, [1, 1, 0], "sliding_attention"),
                                  enc._mask(3, [1, 1, 0], "sliding_attention").numpy())


def test_decision_row_selection_keeps_duplicate_rows_and_order_on_cpu():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    selected = DecisionModel._select_rows(Tensor(values), [5, 1, 5, 0])
    np.testing.assert_array_equal(selected.numpy(), values[[5, 1, 5, 0]])
    np.testing.assert_array_equal(
        wt.gather_rows(Tensor(values), [5, 1, 5, 0]).numpy(),
        values[[5, 1, 5, 0]])


def test_webgpu_row_gather_does_not_use_reserved_wgsl_identifier():
    # A binding named `meta` made createComputePipeline fail without a useful
    # exception at the Python call site. The destination then appeared to be
    # valid data (often all zero), silently changing decision answers.
    import re

    shader = wt._GATHER_ROWS_WGSL
    assert re.search(r"@binding\(3\)\s+var<storage,read>\s+gather_meta\b", shader)
    assert not re.search(r"\b(?:var|let|const|struct)\s+meta\b", shader)
    assert not re.search(r"\bvar<[^>]+>\s+meta\b", shader)


def test_cpu_fused_qkv_take_matches_slices_transpose_and_rope():
    rng = np.random.default_rng(9)
    B, T, H, HD = 2, 5, 3, 8
    values = rng.normal(size=(B * T, 3 * H * HD)).astype(np.float32)
    cos, sin = wt.rope_tables(T, HD)
    for which in range(3):
        old = values.reshape(B, T, 3, H, HD)[:, :, which]
        old = old.transpose(0, 2, 1, 3).reshape(B * H, T, HD)
        if which < 2:
            left, right = old[..., :HD // 2], old[..., HD // 2:]
            old = old * cos + np.concatenate([-right, left], axis=-1) * sin
            got = wt.qkv_take(Tensor(values), which, H, HD, T, Tensor(cos), Tensor(sin), B)
        else:
            got = wt.qkv_take(Tensor(values), which, H, HD, T, B=B)
        np.testing.assert_allclose(got.numpy(), old, rtol=2e-6, atol=2e-6)


def test_cpu_fused_geglu_matches_composed_gelu():
    rng = np.random.default_rng(10)
    gate = rng.normal(size=(7, 32)).astype(np.float32) * 8
    up = rng.normal(size=(7, 32)).astype(np.float32)
    values = np.concatenate([gate, up], axis=-1)
    expected = (wt.gelu(Tensor(gate)) * Tensor(up)).numpy()
    got = wt.geglu_split(Tensor(values), 32)
    np.testing.assert_allclose(got.numpy(), expected, rtol=2e-6, atol=2e-6)


def test_decision_head_fused_qkv_preserves_full_and_selected_answers():
    rng = np.random.default_rng(13)
    T, H, HD, D, F = 7, 2, 4, 8, 11
    p = "head.layers.0."
    shapes = {"self_attn.in_proj": (D, 3 * D),
              "self_attn.out_proj": (D, D),
              "linear1": (D, F), "linear2": (F, D)}
    weights = {p + name: Tensor(rng.normal(size=shape).astype(np.float32) * .1)
               for name, shape in shapes.items()}

    class Head:
        heads = H
        head_dim = HD
        _head_layer = DecisionModel._head_layer
        _last_head_selected = DecisionModel._last_head_selected
        _select_rows = staticmethod(DecisionModel._select_rows)

        @staticmethod
        def _ln(x, _name):
            return x

        @staticmethod
        def _lin(x, name):
            return x.matmul(weights[name])

    x = Tensor(rng.normal(size=(T, D)).astype(np.float32))
    head = Head()
    qkv = x.matmul(weights[p + "self_attn.in_proj"])
    q = wt._slice_last(qkv, 0, D).reshape(T, H, HD).permute(1, 0, 2)
    k = wt._slice_last(qkv, D, 2 * D).reshape(T, H, HD).permute(1, 0, 2)
    v = wt._slice_last(qkv, 2 * D, 3 * D).reshape(T, H, HD).permute(1, 0, 2)
    o = wt.bmm(wt.softmax(wt.bmm(q, wt.transpose_last2(k)) * (HD ** -.5)), v)
    o = o.permute(1, 0, 2).reshape(T, D)
    expected = x + head._lin(o, p + "self_attn.out_proj")
    expected = expected + head._lin(wt.ReLU()(head._lin(expected, p + "linear1")),
                                     p + "linear2")
    np.testing.assert_allclose(head._head_layer(x, 0).numpy(), expected.numpy(),
                               rtol=2e-6, atol=2e-6)
    rows = [0, 3, 6]
    for queries_only in (False, True):
        selected = head._last_head_selected(x, 0, rows, queries_only=queries_only)
        np.testing.assert_allclose(selected.numpy(), expected.numpy()[rows],
                                   rtol=2e-6, atol=2e-6)
    # Independent rows in one forward must not attend to each other's positions or to
    # padding, including after the two head layers' residual connections.
    first = x.numpy()[:4]
    second = x.numpy()
    padded = np.concatenate([first, np.zeros((T - len(first), D), np.float32)], axis=0)
    both = Tensor(np.concatenate([padded, second], axis=0))
    attn_mask = np.zeros((2 * H, T, T), np.float32)
    attn_mask[:H, :, len(first):] = -1e9
    parallel = head._head_layer(both, 0, B=2, mask=Tensor(attn_mask)).numpy()
    broadcast_mask = np.zeros((2 * H, 1, T), np.float32)
    broadcast_mask[:H, :, len(first):] = -1e9
    broadcast = head._head_layer(both, 0, B=2, mask=Tensor(broadcast_mask)).numpy()
    np.testing.assert_allclose(broadcast, parallel, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(parallel[:len(first)],
                               head._head_layer(Tensor(first), 0).numpy(),
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(parallel[T:], head._head_layer(x, 0).numpy(),
                               rtol=2e-6, atol=2e-6)


def test_batched_selected_queries_equal_full_bidirectional_head():
    rng = np.random.default_rng(130)
    B, T, H, HD, D, F = 2, 7, 2, 4, 8, 11
    p = "head.layers.0."
    shapes = {"self_attn.in_proj": (D, 3 * D),
              "self_attn.out_proj": (D, D),
              "linear1": (D, F), "linear2": (F, D)}
    weights = {p + name: Tensor(rng.normal(size=shape).astype(np.float32) * .1)
               for name, shape in shapes.items()}

    class Head:
        heads = H
        head_dim = HD
        _head_layer = DecisionModel._head_layer
        _last_head_selected_many = DecisionModel._last_head_selected_many
        _select_rows = staticmethod(DecisionModel._select_rows)

        @staticmethod
        def _ln(x, _name):
            return x

        @staticmethod
        def _lin(x, name):
            return x.matmul(weights[name])

    first = rng.normal(size=(4, D)).astype(np.float32)
    second = rng.normal(size=(T, D)).astype(np.float32)
    padded = np.concatenate([first, np.zeros((T - 4, D), np.float32)], axis=0)
    both = Tensor(np.concatenate([padded, second], axis=0))
    mask = np.zeros((B * H, T, T), np.float32)
    mask[:H, :, 4:] = -1e9
    head = Head()
    full = head._head_layer(both, 0, B=B, mask=Tensor(mask)).numpy()
    markers = [[1, 3], [2, 4, 6]]
    broadcast_mask = np.zeros((B * H, 1, T), np.float32)
    broadcast_mask[:H, :, 4:] = -1e9
    selected, rows_per_batch = head._last_head_selected_many(
        both, 0, [4, T], T, markers, Tensor(broadcast_mask))
    selected = selected.numpy()
    assert rows_per_batch == 4
    np.testing.assert_allclose(selected[:3], full[[0, 1, 3]], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(selected[4:], full[[T, T + 2, T + 4, T + 6]],
                               rtol=2e-6, atol=2e-6)

    class Batch(Head):
        n_head_layers = 1
        _score_many = DecisionModel._score_many
        _decision_key_mask = DecisionModel._decision_key_mask

        @staticmethod
        def _t(_name):
            return Tensor(np.zeros((1, D), np.float32))

        @staticmethod
        def _packed_score_many(states, padded, markers, selected_rows=0):
            data = states.numpy()
            width = max(map(len, markers))
            scores = np.zeros((len(markers), width), np.float32)
            for b, ms in enumerate(markers):
                base = b * (selected_rows or padded)
                rows = [j + 1 for j in range(len(ms))] if selected_rows else ms
                scores[b, :len(ms)] = data[[base + r for r in rows], 0]
            actions = np.tile([2., 0.], len(markers)).astype(np.float32)
            return Tensor(np.concatenate([scores.ravel(), actions])), width

    class Full(Batch):
        _last_head_selected_many = None

    from_selected = Batch()._score_many(both, [4, T], T, markers, [0, 0])
    from_full = Full()._score_many(both, [4, T], T, markers, [0, 0])
    for (selected_logits, selected_act), (full_logits, full_act) in zip(
            from_selected, from_full):
        np.testing.assert_allclose(selected_logits, full_logits, rtol=2e-6, atol=2e-6)
        assert abs(selected_act - full_act) < 2e-6


def test_parallel_decision_head_reads_each_questions_own_markers():
    class Model:
        _score_many = DecisionModel._score_many
        _decision_key_mask = DecisionModel._decision_key_mask
        _select_rows = staticmethod(DecisionModel._select_rows)
        _last_head_selected_many = None
        heads = 2
        n_head_layers = 1

        @staticmethod
        def _t(_name):
            return Tensor(np.zeros((2, 4), dtype=np.float32))

        @staticmethod
        def _head_layer(h, _i, B=1, mask=None):
            assert B == 2 and mask.shape == (4, 1, 5)
            np.testing.assert_array_equal(mask.numpy()[:2, :, 3:], -1e9)
            return h

        @staticmethod
        def _packed_score_many(h, padded, markers, _selected_rows=0):
            values = h.numpy()
            logits = np.stack([values[[b * padded + int(r) for r in ms], 0]
                               for b, ms in enumerate(markers)])
            actions = np.tile([2., 0.], len(markers)).astype(np.float32)
            return Tensor(np.concatenate([logits.ravel(), actions])), logits.shape[1]

    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    model = Model()
    model._head_profile = True
    outputs = model._score_many(Tensor(values), [3, 5], 5, [[1, 2], [2, 4]], [0, 1])
    np.testing.assert_array_equal(outputs[0][0], values[[1, 2], 0])
    np.testing.assert_array_equal(outputs[1][0], values[[7, 9], 0])
    assert outputs[0][1] == outputs[1][1]
    assert set(model._head_timing) == {
        "prepare_ms", "layers_queue_ms", "scores_queue_ms",
        "gpu_wait_readback_ms", "result_ms", "layer_ms", "selected_ms", "full_ms"}
    assert all(t >= 0 for t in model._head_timing.values() if isinstance(t, (int, float)))
    assert all(t >= 0 for t in model._head_timing["layer_ms"])


def test_batch_route_has_no_token_cutoff_or_user_facing_scalar_exploration():
    class Model:
        batch_pays = staticmethod(DecisionModel.batch_pays)
        _batch_route = DecisionModel._batch_route
        _batch_observed = DecisionModel._batch_observed

    model = Model()
    assert model.batch_pays(500, 3)
    assert not model.batch_pays(40, 1)
    for duration in (105, 103, 72, 69):
        route, key = model._batch_route(486, 3, [3, 2, 4])
        assert route == "batch"
        model._batch_observed(key, route, duration)
    # Separate distinct-question calibration can still provide the comparator.
    for duration in (130, 128, 127):
        model._batch_observed(key, "scalar", duration)
    assert model._batch_route(486, 3, [3, 2, 4])[0] == "batch"
    assert model._batch_route(513, 3, [3, 2, 4])[0] == "batch", "new shape starts batched"
    # A one-millisecond measured scalar win has no arbitrary 5% gate.
    model._batch_profiles[key] = {"batch": [140, 139, 126],
                                  "scalar": [145, 144, 125]}
    assert model._batch_route(486, 3, [3, 2, 4])[0] == "scalar"


@pytest.mark.parametrize('count', [3, 6])
def test_long_distinct_questions_use_one_encoder_and_head_pass(count):
    lengths = [162 + i for i in range(count)]
    padded = max(lengths)

    class Encoder:
        def _encode_many_device(self, seqs):
            assert len(seqs) == count
            return Tensor(np.zeros((count * padded, 2), np.float32)), lengths, padded

    class Model:
        _raw_questions = DecisionModel._raw_questions
        batch_pays = staticmethod(DecisionModel.batch_pays)

        def __init__(self):
            self.enc = Encoder()
            self.cfg = SimpleNamespace(qtypes=["choice"])
            self.head_calls = 0

        def _score_many(self, hidden, lengths, padded, markers, types):
            self.head_calls += 1
            assert hidden.shape == (count * max(lengths), 2)
            assert lengths == [162 + i for i in range(count)]
            assert padded == max(lengths)
            assert markers == [(1, 2)] * count
            assert types == [0] * count
            return [(np.asarray([0.0, 1.0]), 0.8)] * count

        def _score(self, *_args):
            raise AssertionError("the scalar head must not run")

    question = {"type": "choice"}
    built = [(str(i), question, "choice", [i + 1] * (162 + i), [1, 2], ["a", "b"])
             for i in range(count)]
    usage = {"_profile": True}
    result = Model()
    assert len(result._raw_questions(built, usage)) == count
    assert result.head_calls == 1
    assert usage["encoder_passes"] == usage["head_passes"] == 1
    assert usage["batch_route"] == "batch"
    assert usage["batched"] is True


def test_decision_score_reads_numpy_output_on_cpu():
    class CpuScore:
        _score = DecisionModel._score
        n_head_layers = 0

        @staticmethod
        def _t(_name):
            return Tensor(np.zeros((1, 3), dtype=np.float32))

        @staticmethod
        def _packed_score(_hidden, _markers):
            return Tensor(np.asarray([2.0, -1.0, 0.0], dtype=np.float32))

    logits, act = CpuScore()._score(Tensor(np.zeros((4, 3), dtype=np.float32)), [1, 2], 0)
    np.testing.assert_array_equal(logits, [2.0, -1.0])
    assert act == 1.0


def test_execution_tuner_benchmarks_numpy_results_on_cpu():
    calls = []

    def run(which):
        calls.append(which)
        return np.asarray([1.0], dtype=np.float32)

    selected = wt._weight_execution("decision_cpu_readback_test", "rows", 3, 2, 4,
                                    run, candidates=("full", "selected"), rounds=5, repeat=1)
    assert selected in ("full", "selected")
    assert calls.count("full") >= 6 and calls.count("selected") >= 6


def test_decide_reuses_identical_encoder_inputs_and_reports_actual_work():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, ids):
            self.calls.append(tuple(ids))
            return np.asarray(ids)

    class Config:
        qtypes = ["choice", "score", "noul"]
        shapes = {"choice": "named", "score": "ordered", "noul": "fixed"}

        @classmethod
        def shape_of(cls, qtype):
            return cls.shapes.get(qtype, "named")

        @staticmethod
        def temp_for(_qtype, _count):
            return 1.0

    class Model:
        decide = DecisionModel.decide
        _raw_questions = DecisionModel._raw_questions

        def __init__(self):
            self.enc = Encoder()
            self.cfg = Config()
            self.score_calls = 0

        @staticmethod
        def batch_pays(_longest, _count):
            return False

        def _score(self, _hidden, _markers, _qtype):
            self.score_calls += 1
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
    assert model.score_calls == 1
    assert result["answers"]["first"] == result["answers"]["same_input"]
    assert result["usage"] == {
        "input_tokens": 6,
        "output_tokens": 0,
        "questions": 2,
        "sequence_tokens": {"first": 3, "same_input": 3},
        "encoder_tokens": 3,
        "encoder_passes": 1,
        "head_passes": 1,
        "batched": False,
    }
    profiled = model.decide("state", questions, profile=True)
    assert model.score_calls == 2
    assert profiled["answers"] == result["answers"]
    for stage in ("prepare_ms", "encoder_ms", "head_ms", "answer_ms"):
        assert profiled["usage"][stage] >= 0
    assert "_profile" not in profiled["usage"]


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

        @staticmethod
        def shape_of(_qtype):
            return "fixed"

    class Model:
        _raw_questions = DecisionModel._raw_questions

        def __init__(self):
            self.enc = Encoder()
            self.cfg = Config()
            self.scored = []

        @staticmethod
        def batch_pays(_longest, _count):
            return True

        def _score_many(self, hidden, lengths, padded, _markers, _types):
            rows = hidden.numpy().reshape(len(lengths), padded)
            self.scored.extend(rows[i, :length].tolist()
                               for i, length in enumerate(lengths))
            return [(np.asarray([0.0, 1.0]), 0.75) for _ in lengths]

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
    assert execution == {"encoder_tokens": 6, "encoder_passes": 1,
                         "head_passes": 1, "batched": True}


def test_batched_decisions_do_not_encode_or_score_duplicate_inputs():
    class Encoder:
        def __init__(self):
            self.calls = []

        def _encode_many_device(self, seqs):
            self.calls.append(seqs)
            padded = max(map(len, seqs))
            data = np.zeros((len(seqs) * padded, 1), dtype=np.float32)
            for index, seq in enumerate(seqs):
                data[index * padded:index * padded + len(seq), 0] = seq
            return Tensor(data), list(map(len, seqs)), padded

    class Model:
        _raw_questions = DecisionModel._raw_questions

        def __init__(self):
            self.enc = Encoder()
            self.cfg = SimpleNamespace(qtypes=["choice"])
            self.calls = []

        @staticmethod
        def batch_pays(_longest, count):
            return count >= 2

        def _score_many(self, hidden, lengths, padded, markers, _types):
            rows = hidden.numpy().reshape(len(lengths), padded)
            self.calls.extend((rows[i, :length].tolist(), list(markers[i]))
                              for i, length in enumerate(lengths))
            return [(np.asarray([0.0, 1.0]), 0.75) for _ in lengths]

    q = {"type": "choice"}
    built = [("a", q, "choice", [1, 2], [0, 1], ["a", "b"]),
             ("b", q, "choice", [1, 2], [0, 1], ["a", "b"]),
             ("c", q, "choice", [3, 4, 5], [0, 1], ["a", "b"])]
    model = Model()
    usage = {}
    answers = model._raw_questions(built, usage)
    assert [row[0] for row in answers] == ["a", "b", "c"]
    assert model.enc.calls == [[[1, 2], [3, 4, 5]]]
    assert model.calls == [([1.0, 2.0], [0, 1]), ([3.0, 4.0, 5.0], [0, 1])]
    assert usage == {"encoder_tokens": 6, "encoder_passes": 1,
                     "head_passes": 1, "batched": True}


def test_batched_decision_failure_is_not_hidden_by_scalar_execution():
    class Encoder:
        def _encode_many_device(self, _seqs):
            raise RuntimeError("batch device lost")

        def encode_many(self, _seqs):
            raise AssertionError("host batch downgrade")

        def encode(self, _seq):
            raise AssertionError("scalar downgrade")

    class Model:
        _raw_questions = DecisionModel._raw_questions
        enc = Encoder()
        cfg = SimpleNamespace(qtypes=["choice"])

        @staticmethod
        def batch_pays(_longest, count):
            return count >= 2

    q = {"type": "choice"}
    built = [("a", q, "choice", [1, 2], [0, 1], ["a", "b"]),
             ("b", q, "choice", [3, 4], [0, 1], ["a", "b"])]
    with pytest.raises(RuntimeError, match="batch device lost"):
        Model()._raw_questions(built)


def test_distinct_head_metadata_on_equal_encoder_ids_stays_batched():
    class Encoder:
        def __init__(self):
            self.calls = []

        def _encode_many_device(self, seqs):
            self.calls.append(seqs)
            return Tensor(np.zeros((2 * 3, 1), np.float32)), [3, 3], 3

    class Model:
        _raw_questions = DecisionModel._raw_questions
        cfg = SimpleNamespace(qtypes=["choice", "score"])

        def __init__(self):
            self.enc = Encoder()
            self.head_calls = []

        @staticmethod
        def batch_pays(_longest, count):
            return count >= 2

        def _score_many(self, _hidden, _lengths, _padded, markers, types):
            self.head_calls.append((markers, types))
            return [(np.asarray([float(i)]), 0.5) for i in types]

    built = [("a", {}, "choice", [1, 2, 3], [1], ["x"]),
             ("b", {}, "score", [1, 2, 3], [2], ["y"])]
    model = Model()
    usage = {}
    result = model._raw_questions(built, usage)
    assert [row[0] for row in result] == ["a", "b"]
    assert model.enc.calls == [[[1, 2, 3], [1, 2, 3]]]
    assert model.head_calls == [([(1,), (2,)], [0, 1])]
    assert usage["batched"] is True
    assert usage["encoder_passes"] == usage["head_passes"] == 1


def test_decision_action_features_match_the_original_host_formula():
    logits = np.asarray([1.25, -0.5, 0.75, 2.0], dtype=np.float32)
    p = np.exp(logits - logits.max()); p = p / p.sum()
    top2 = np.sort(p)[::-1][:2]
    expected = np.asarray([
        top2[0], top2[0] - top2[1],
        -(p * np.log(np.clip(p, 1e-9, 1.0))).sum() / np.log(len(logits)),
        len(logits) / 255.0,
    ], dtype=np.float32)

    got = wt.decision_features(Tensor(logits.reshape(1, -1)), len(logits)).numpy()[0]

    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_decision_action_features_keep_the_two_option_entropy_denominator():
    got = wt.decision_features(Tensor([[3.0]]), 1).numpy()[0]
    assert np.allclose(got, [1.0, 1.0, 0.0, 2.0 / 255.0])


def test_batched_decision_features_ignore_each_rows_padding():
    logits = Tensor(np.asarray([[2.0, -1.0, 100.0],
                                [0.5, 1.25, -0.75],
                                [3.0, -99.0, -99.0]], np.float32))
    actual = wt.decision_features_many(logits, [2, 3, 1]).numpy()
    expected = np.stack([
        wt.decision_features(Tensor(logits.numpy()[i, :k].reshape(1, -1)), k).numpy()[0]
        for i, k in enumerate((2, 3, 1))])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_batched_decision_scorer_and_action_match_separate_questions():
    rng = np.random.default_rng(212)
    D, T, B = 8, 6, 3
    weights = {
        "scorer.1": Tensor(rng.normal(size=(D, D)).astype(np.float32) * .1),
        "scorer.3": Tensor(rng.normal(size=(D, 1)).astype(np.float32) * .1),
        "act_head.0": Tensor(rng.normal(size=(D + 4, 5)).astype(np.float32) * .1),
        "act_head.2": Tensor(rng.normal(size=(5, 2)).astype(np.float32) * .1),
    }
    model = DecisionModel.__new__(DecisionModel)
    model._ln = lambda x, _name: x
    model._lin = lambda x, name: x.matmul(weights[name])
    model._select_rows = DecisionModel._select_rows
    markers = [[1, 3], [2, 4, 5], [1]]
    hidden = rng.normal(size=(B * T, D)).astype(np.float32)
    for selected in (False, True):
        if selected:
            width = max(map(len, markers)) + 1
            source = np.zeros((B * width, D), np.float32)
            for b, ms in enumerate(markers):
                source[b * width] = hidden[b * T]
                for j, pos in enumerate(ms):
                    source[b * width + j + 1] = hidden[b * T + pos]
        else:
            width = T
            source = hidden
        packed, option_width = model._packed_score_many(
            Tensor(source), T, markers, width if selected else 0)
        got = packed.numpy()
        for b, ms in enumerate(markers):
            own = source[b * width:b * width + len(ms) + 1] if selected else \
                source[[b * T] + [b * T + pos for pos in ms]]
            expected = model._packed_score(Tensor(own), range(1, len(ms) + 1)).numpy()
            np.testing.assert_allclose(got[b * option_width:b * option_width + len(ms)],
                                       expected[:len(ms)], rtol=1e-5, atol=1e-6)
            action_offset = B * option_width + b * 2
            np.testing.assert_allclose(got[action_offset:action_offset + 2],
                                       expected[len(ms):], rtol=1e-5, atol=1e-6)


def test_final_head_selected_rows_match_full_bidirectional_layer():
    rng = np.random.default_rng(19)
    model = DecisionModel.__new__(DecisionModel)
    model.heads = 2
    model.head_dim = 2
    weights = {
        "head.layers.0.self_attn.in_proj": rng.standard_normal((4, 12), dtype=np.float32) * 0.1,
        "head.layers.0.self_attn.out_proj": rng.standard_normal((4, 4), dtype=np.float32) * 0.1,
        "head.layers.0.linear1": rng.standard_normal((4, 7), dtype=np.float32) * 0.1,
        "head.layers.0.linear2": rng.standard_normal((7, 4), dtype=np.float32) * 0.1,
    }

    model._ln = lambda x, _name: x
    model._lin = lambda x, name: x.matmul(Tensor(weights[name]))
    hidden = Tensor(rng.standard_normal((6, 4), dtype=np.float32))
    rows = [0, 2, 5]
    full = model._head_layer(hidden, 0)
    expected = model._select_rows(full, rows).numpy()

    selected_full = model._last_head_selected(hidden, 0, rows, queries_only=False).numpy()
    selected_q = model._last_head_selected(hidden, 0, rows, queries_only=True).numpy()
    np.testing.assert_allclose(selected_full, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(selected_q, expected, rtol=2e-5, atol=2e-6)


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


def test_a_checkpoint_that_names_its_own_config_needs_no_entry_in_the_sdk():
    """The fallback list is for checkpoints that say nothing. One that declares its layout
    is read from the declaration, so its filename never has to be known here -- which is the
    difference between supporting a model and allow-listing one."""
    import asyncio
    from webtorch.decision import _DECISION_CONFIGS, _index, _named_first

    # A real published layout: the root config.json is an INDEX, not an architecture.
    declared = {
        "format_version": 1,
        "architecture": "SomeDecisionModel",
        "jellyfish_config_file": "jellyfish_config.json",
        "encoder_config_file": "encoder/config.json",
        "weights_file": "model.safetensors",
        "tokenizer_directory": "tokenizer",
    }

    async def read_json(_name):
        return declared

    said = asyncio.run(_index("org/repo", read_json))
    files, dirs = said["files"], said["dirs"]
    assert files["weights"] == "model.safetensors"
    assert dirs["tokenizer"] == "tokenizer/"

    named = [v for k, v in files.items() if k.endswith("config") and k != "encoder_config"]
    assert _named_first(named, _DECISION_CONFIGS)[0] == "jellyfish_config.json"
    # And no published model's filename is carried in the SDK itself.
    assert all("_config.json" == n[-len("_config.json"):] for n in _DECISION_CONFIGS)
    assert set(_DECISION_CONFIGS) == {"decision_config.json", "rl_agent_config.json"}


def test_question_types_come_from_the_checkpoint_and_shapes_drive_everything():
    """A checkpoint that names its own question types needs no entry in this engine.

    What used to happen: every type that was not literally "choice" or "score" was rendered
    as a two-outcome statement and answered under a key named after one model family's
    vocabulary. A model that called its types anything else got another model's answers.
    """
    from webtorch.decision import DecisionConfig, render_options

    # The checkpoints in circulation say how many types they have and nothing else.
    assert DecisionConfig({"temperature": [1.0, 1.0, 1.0]}).qtypes == ["choice", "score", "noul"]
    assert DecisionConfig({"temperature": [1.0, 1.0]}).qtypes == ["choice", "score"]
    # A fourth is named positionally and is asked the GENERAL way, not given the last
    # one's shape -- it is not a two-outcome question merely for being unrecognised.
    four = DecisionConfig({"temperature": [1.0] * 4})
    assert four.qtypes[3] == "type3" and four.shape_of("type3") == "named"

    # One that declares them is read, and nothing here had to change for it.
    cfg = DecisionConfig({"temperature": [1.0, 1.0, 1.0],
                          "question_types": [{"name": "route", "shape": "named"},
                                             {"name": "severity", "shape": "ordered"},
                                             {"name": "holds", "shape": "fixed"}]})
    assert cfg.qtypes == ["route", "severity", "holds"]
    assert [cfg.shape_of(t) for t in cfg.qtypes] == ["named", "ordered", "fixed"]
    # The index into the type embedding is the position it was declared at.
    assert cfg.qtypes.index("severity") == 1
    # A type the checkpoint never declared is asked the general way rather than guessed at.
    assert cfg.shape_of("not-a-type") == "named"

    # Rendering follows the shape, so "severity" is levels and "holds" is two outcomes --
    # neither of which could be known from the names.
    assert render_options(cfg.shape_of("severity"), ["low", "high"]) == (
        ["0", "1"], ["level 0: low", "level 1: high"])
    assert render_options(cfg.shape_of("holds"), None)[0] == ["false", "true"]
    assert render_options(cfg.shape_of("route"), ["a", "b"]) == (["a", "b"], ["a", "b"])


def test_a_truth_is_read_by_the_shape_of_the_question():
    from webtorch.decision import DecisionModel as D

    assert D._target_index("named", "b", ["a", "b", "c"]) == 1
    assert D._target_index("ordered", 2, ["0", "1", "2"]) == 2
    assert D._target_index("fixed", "yes", ["false", "true"]) == 1
    assert D._target_index("fixed", False, ["false", "true"]) == 0
    # A dict of truths may be keyed by the type's own name, whatever that is.
    assert D._target_index("named", {"route": "c"}, ["a", "b", "c"], "route") == 2


def test_the_type_count_comes_from_the_model_not_from_its_calibration_metadata():
    """Downstream callers ask for `noul` by name. A checkpoint whose temperature list is
    short or missing must not lose a type it actually has -- the type embedding says how
    many there are, and that is the model rather than metadata beside it."""
    from webtorch.decision import DecisionConfig, type_rows

    w = {"type_emb.weight": np.zeros((3, 8), np.float32)}
    assert type_rows(w) == 3
    assert type_rows({}) is None

    for cfg in ({"temperature": [1.0, 1.0]},        # short
                {"temperature": 1.0},               # a scalar
                {}):                                # absent
        c = DecisionConfig(cfg, type_count=type_rows(w))
        assert c.qtypes == ["choice", "score", "noul"], cfg
        assert c.shape_of("noul") == "fixed"


def test_the_answer_keys_downstream_reads_are_unchanged():
    """`laya-service` and the browser-use skill read `answer.choice`, `answer.score` and
    `answer.noul` off the wire. The shapes decide those keys now; for a checkpoint that
    declares no types -- which is every one published so far -- they must come out the
    same."""
    from webtorch.decision import DecisionConfig, render_options

    cfg = DecisionConfig({"temperature": [1.0, 1.0, 1.0]})
    keyed = {"named": "choice", "ordered": "score", "fixed": "noul"}
    assert {t: keyed[cfg.shape_of(t)] for t in cfg.qtypes} == {
        "choice": "choice", "score": "score", "noul": "noul"}

    # And the sequences built for them are byte-for-byte what they were.
    assert render_options(cfg.shape_of("choice"), ["a", "b"]) == (["a", "b"], ["a", "b"])
    assert render_options(cfg.shape_of("score"), ["lo", "hi"]) == (
        ["0", "1"], ["level 0: lo", "level 1: hi"])
    assert render_options(cfg.shape_of("noul"), None) == (
        ["false", "true"], ["false: no, the statement does not hold",
                            "true: yes, the statement holds"])



def test_a_batch_runs_end_to_end_only_when_every_question_is_a_prefix_of_real_tokens():
    class Packed(_FakeEncoder):
        _replayed_packed = TextEncoder._replayed_packed

        @staticmethod
        def _packed_ok():
            return True

    enc = Packed()
    plat = _FakePlatform()
    before = _with_platform(plat)
    try:
        holed = np.ones((2, 5), np.int64)
        holed[0, 2] = 0                          # a gap inside a question: not packable
        from webtorch.encoder import _NOT_PACKED
        assert enc._replayed_packed(np.zeros(10, np.int64), 5, 2, holed) is _NOT_PACKED
    finally:
        wt._adam_kernel["platform"] = before
