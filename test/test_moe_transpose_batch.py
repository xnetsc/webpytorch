"""Bounded expert uploads preserve both buffer lifetimes and backend contracts."""

import asyncio
from types import SimpleNamespace

from webtorch import _core as wt
from webtorch.llm import CausalLM


class _Array:
    def __init__(self, ident=1):
        self.buffer = SimpleNamespace(buffer_id=ident)


def test_pending_transpose_keeps_source_and_metadata_until_one_sync(monkeypatch):
    calls = []
    flag = _Array(4)
    source = _Array(1)
    destination = _Array(2)
    meta = SimpleNamespace(buffer_id=3)
    platform = SimpleNamespace(runKernel=lambda request: calls.append(request))
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    monkeypatch.setattr(wt, "_tr_k", {"added": True})
    monkeypatch.setattr(wt, "_adam_kernel", {
        "platform": platform, "make_meta": lambda *_: meta,
    })
    monkeypatch.setattr(wt, "_empty", lambda *_: flag)
    monkeypatch.setattr(wt, "cp", SimpleNamespace(
        asnumpy=lambda value: calls.append(("sync", value))), raising=False)

    pending = []
    assert wt.ggml_transpose(source, 1, 32, dst=destination, pending=pending) is destination
    assert pending == [(source, meta, flag)]
    assert len(calls) == 1 and calls[0]["name"] == "ggml_tr"
    wt._ggml_transpose_drain(pending)
    assert pending == []
    assert calls[1] == ("sync", flag)


def test_webgpu_expert_stack_drains_bounded_windows(monkeypatch):
    # Q8_0, K=32, N=1 is a single 34-byte GGML block per expert.
    chunks = [bytes([index % 256]) * 34 for index in range(70)]
    batches = []
    offsets = []
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    monkeypatch.setattr(wt, "_empty", lambda *_: _Array(2))
    monkeypatch.setattr(wt, "xp", SimpleNamespace(asarray=lambda _: _Array(1)))

    def transpose(src, n, rowb, dst, dstoff, pending):
        offsets.append(dstoff)
        pending.append((src, _Array(3), _Array(4)))
        return dst

    def drain(pending):
        if pending:
            batches.append(len(pending))
            pending.clear()

    monkeypatch.setattr(wt, "ggml_transpose", transpose)
    monkeypatch.setattr(wt, "_ggml_transpose_drain", drain)
    linear = wt.GGMLMoELinear(chunks, "Q8_0", 32, 1)
    assert batches == [32, 32, 6]
    assert offsets == [index * linear.estride for index in range(70)]
    assert linear.n_experts == 70


def test_webgl_expert_stack_keeps_one_pass_equivalent_interface(monkeypatch):
    calls = []
    chunks = [bytes([index]) * 34 for index in range(3)]
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "xp", SimpleNamespace(asarray=lambda _: _Array(1)))
    monkeypatch.setattr(wt, "_ggml_transpose_gl_stack",
                        lambda *args: calls.append(args) or _Array(2))
    linear = wt.GGMLMoELinear(chunks, "Q8_0", 32, 1)
    assert linear.n_experts == 3
    assert len(calls) == 1 and calls[0][3] == 3


def test_webgl_joined_experts_upload_identical_original_bytes(monkeypatch):
    # Gate and up remain views into their original read buffers until the one
    # WebGL staging array is filled; no concatenated whole-layer list is made.
    gate = bytes(range(68))
    up = bytes(range(100, 168))
    chunks = [memoryview(gate)[i * 34:(i + 1) * 34] for i in range(2)]
    also = [memoryview(up)[i * 34:(i + 1) * 34] for i in range(2)]
    uploads = []
    monkeypatch.setattr(wt, "_webgl_ready", lambda: True)
    monkeypatch.setattr(wt, "_adam_backend_ready", lambda: False)
    monkeypatch.setattr(wt, "xp", SimpleNamespace(
        asarray=lambda value: uploads.append(value.copy()) or _Array(1)))
    monkeypatch.setattr(wt, "_ggml_transpose_gl_stack", lambda *_: _Array(2))

    linear = wt.GGMLMoELinear(chunks, "Q8_0", 32, 2, also_chunks=also)
    assert linear.n_experts == 2
    assert uploads[0].view("u1").tobytes() == (
        gate[:34] + up[:34] + gate[34:] + up[34:])


def test_webgpu_joined_experts_upload_one_expert_at_a_time(monkeypatch):
    gate = [bytes([1]) * 34, bytes([2]) * 34]
    up = [bytes([3]) * 34, bytes([4]) * 34]
    uploads = []
    monkeypatch.setattr(wt, "_webgl_ready", lambda: False)
    monkeypatch.setattr(wt, "_empty", lambda *_: _Array(2))
    monkeypatch.setattr(wt, "xp", SimpleNamespace(
        asarray=lambda value: uploads.append(value.copy()) or _Array(1)))
    monkeypatch.setattr(wt, "ggml_transpose", lambda src, n, rowb, dst, dstoff, pending: dst)
    monkeypatch.setattr(wt, "_ggml_transpose_drain", lambda pending: pending.clear())

    linear = wt.GGMLMoELinear(gate, "Q8_0", 32, 2, also_chunks=up)
    assert linear.n_experts == 2
    assert [a.view("u1").tobytes() for a in uploads] == [gate[i] + up[i] for i in range(2)]


def test_gguf_loader_passes_views_not_whole_layer_copies(monkeypatch):
    model = CausalLM.__new__(CausalLM)
    model._ginfo = {
        "gate": {"dims": [32, 1, 2], "type": 8, "offset": 0},
        "up": {"dims": [32, 1, 2], "type": 8, "offset": 68},
    }
    model._gds = 0
    model._weights = "native"
    source = bytes(range(136))

    async def read(start, stop):
        return source[start:stop + 1]

    model._grng = read
    seen = []
    monkeypatch.setattr(wt, "ggml_native_supported", lambda _: True)

    def capture(chunks, type_name, K, N, also_chunks=None):
        seen.append((chunks, type_name, K, N, also_chunks))
        return object()

    monkeypatch.setattr(wt, "GGMLMoELinear", capture)
    result = asyncio.run(model._gexperts_stacked("gate", also="up"))
    assert result is not None
    chunks, type_name, K, N, also_chunks = seen[0]
    assert (type_name, K, N) == ("Q8_0", 32, 2)
    assert all(isinstance(part, memoryview) for part in chunks + also_chunks)
    assert [bytes(chunks[i]) + bytes(also_chunks[i]) for i in range(2)] == [
        source[0:34] + source[68:102], source[34:68] + source[102:136]]
