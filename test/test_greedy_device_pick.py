from types import SimpleNamespace

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
    assert source.count('_greedy_chunk_capture_ready", False') >= 2
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
