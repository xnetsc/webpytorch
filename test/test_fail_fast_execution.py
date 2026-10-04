"""Execution errors must not become a slower, apparently successful route."""

from types import SimpleNamespace

import numpy as np
import pytest

from webtorch import _core as wt


def test_selected_backend_probe_propagates_failure(monkeypatch):
    monkeypatch.setattr(wt, "GPU", True)
    monkeypatch.setitem(wt._adam_kernel, "platform", None)
    monkeypatch.setitem(wt._copy_kernel, "plat", None)

    def broken_name():
        raise RuntimeError("backend selection failed")

    monkeypatch.setattr(wt, "cp", SimpleNamespace(get_backend_name=broken_name),
                        raising=False)
    with pytest.raises(RuntimeError, match="backend selection failed"):
        wt._adam_backend_ready()
    with pytest.raises(RuntimeError, match="backend selection failed"):
        wt._webgl_ready()


def test_gqa_tuner_failure_does_not_cache_or_change_split(monkeypatch):
    key = (3, 1, 12, 4)
    monkeypatch.delitem(wt._GQA_TUNED, key, raising=False)
    before = wt._GQA_SPLIT, wt._GQA_SPLIT_ON

    def broken_decode(*_args, **_kw):
        raise RuntimeError("GQA kernel failed")

    monkeypatch.setattr(wt, "gqa_decode", broken_decode)
    with pytest.raises(RuntimeError, match="GQA kernel failed"):
        wt.gqa_tune(3, 1, 12, 4, candidates=(4,), rounds=1)
    assert key not in wt._GQA_TUNED
    assert (wt._GQA_SPLIT, wt._GQA_SPLIT_ON) == before


def test_kv_pair_tuner_failure_does_not_choose_separate(monkeypatch):
    key = ("kv_write_pair", 101, 64, 107, False)
    monkeypatch.delitem(wt._TUNED, key, raising=False)
    monkeypatch.setattr(wt, "xp", SimpleNamespace(asarray=lambda _value: (_ for _ in ()).throw(
        RuntimeError("KV upload failed"))))
    with pytest.raises(RuntimeError, match="KV pair auto candidate failed") as error:
        wt._kv_pair_auto(101, 64, 107, False)
    assert "KV upload failed" in str(error.value.__cause__)
    assert key not in wt._TUNED


def test_add_rmsnorm_failed_fused_candidate_is_not_cached_as_composed(monkeypatch):
    residual = wt.Tensor(np.ones((1, 1003), np.float32))
    update = wt.Tensor(np.ones((1, 1003), np.float32))
    weight = wt.Tensor(np.ones((1003,), np.float32))
    key = ("add_rmsnorm", 1, 1003)
    monkeypatch.delitem(wt._TUNED, key, raising=False)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: True)
    monkeypatch.setitem(wt._add_rms_k, "added", False)

    def broken_kernel(*_args, **_kw):
        raise RuntimeError("fused kernel failed")

    monkeypatch.setitem(wt._adam_kernel, "platform", SimpleNamespace(addKernel=broken_kernel))
    with pytest.raises(RuntimeError, match="add_rmsnorm candidate 'fused' failed") as error:
        wt.add_rmsnorm(residual, update, weight, 1e-6)
    assert "fused kernel failed" in str(error.value.__cause__)
    assert key not in wt._TUNED


def test_gguf_materialization_error_is_not_cached_as_stored_only(monkeypatch):
    monkeypatch.delitem(wt._DEQ_OK, "Q8_0", raising=False)

    def broken_matmul(*_args, **_kw):
        raise RuntimeError("packed reference failed")

    monkeypatch.setattr(wt, "ggml_matmul", broken_matmul)
    with pytest.raises(RuntimeError, match="GGUF materialized candidate failed") as error:
        wt.ggml_dequant_ok("Q8_0")
    assert "packed reference failed" in str(error.value.__cause__)
    assert "Q8_0" not in wt._DEQ_OK
