"""Failed warmup must surface, while leaving submission and recurrent state balanced."""
import types

import pytest

from webtorch import _core as wt
from webtorch.llm import CausalLM


@pytest.mark.parametrize('fail_at', ['dispatch', 'readback'])
def test_warm_failure_ends_submission_and_resets_state(monkeypatch, fail_at):
    events = []
    error = ValueError('broken kernel')
    plat = types.SimpleNamespace(
        beginSubmission=lambda: events.append('begin'),
        endSubmission=lambda: events.append('end'),
    )
    monkeypatch.setitem(wt._adam_kernel, 'platform', plat)
    monkeypatch.setattr(wt, '_count_dispatch_names', lambda enabled: events.append(enabled))
    model = CausalLM.__new__(CausalLM)
    model.eos_ids = [1]
    model._dispatch_names = lambda: {}
    model._set_inputs = lambda *args: None
    model._reset_linear_state = lambda: events.append('reset')

    def fail():
        raise error

    model._decode_fwd = fail if fail_at == 'dispatch' else lambda: types.SimpleNamespace(numpy=fail)
    with pytest.raises(RuntimeError, match='decoder warmup failed') as got:
        model._warm_decode_step()
    assert got.value.__cause__ is error
    assert events == [True, 'begin', 'end', False, 'reset']
