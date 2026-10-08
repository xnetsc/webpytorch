"""A decision head over a language model: recognised by what its folder holds, prompted in the
template it was trained with, answered over the slots its config declares."""
import asyncio
import json
import struct

import numpy as np
import pytest

from webtorch import _sdk, decision, webio
from webtorch.decision import SlotDecisionModel, _bare_v1, slot_layout


VERBAL = ["n", "y", "0", "1", "2", "A", "B", "C"]
CFG = {"hidden_size": 4,
       "slots": {"num_slots": 8, "ranges": {"noul": [0, 2], "score": [2, 5], "choice": [5, 8]},
                 "verbalizers": VERBAL, "template_version": "bare-v1"},
       "verbalizer_ids": [ord(v) for v in VERBAL],
       "adapter_subfolder": "adapter", "softcap": None}


class FakeTok(object):
    def encode(self, text):
        return [ord(c) for c in text]
    encode_special = encode


class FakeLM(object):
    H = 4
    lmax = 4096

    def __init__(self, h=None):
        self.h = np.asarray([0.5, -1.0, 2.0, 0.25] if h is None else h, np.float32)
        self.tok = FakeTok()
        self.prompts = []
        self.released = False

    def prefill_hidden(self, ids):
        self.prompts.append("".join(chr(i) for i in ids))
        return self.h

    def release(self):
        self.released = True


def _head():
    rng = np.random.default_rng(1)
    return (rng.standard_normal((8, 4)).astype(np.float32),
            rng.standard_normal(8).astype(np.float32))


def _model(temps=None):
    W, b = _head()
    lm = FakeLM()
    return SlotDecisionModel(lm, W, b, slot_layout(CFG), temperatures=temps), lm, W, b


def _softmax(z):
    p = np.exp(z - z.max())
    return p / p.sum()


def test_the_bare_v1_prompt_is_the_one_the_model_card_shows():
    got = _bare_v1("choice", "named",
                   "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
                   "Supplier response for this scenario.", ["A", "B", "C", "D"],
                   ["issue_warning", "renegotiate", "dual_source", "maintain"])
    assert got == ("[kind] choice\n"
                   "[state] SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.\n"
                   "[question] Supplier response for this scenario.\n"
                   "[options]\nA) issue_warning\nB) renegotiate\nC) dual_source\nD) maintain\n"
                   "[decision]:")
    assert _bare_v1("noul", "fixed", "s", "q", ["false", "true"], None).endswith(
        "[options]\nfalse\ntrue\n[decision]:")


def test_a_layout_is_read_from_the_slots_a_config_declares():
    lay = slot_layout(CFG)
    assert lay["num_slots"] == 8 and lay["ranges"]["choice"] == (5, 8)
    assert lay["template"] == "bare-v1" and lay["verbalizers"] == VERBAL
    assert slot_layout({"hidden_size": 4, "architectures": ["X"]}) is None
    assert slot_layout({"slots": {"num_slots": 2}}) is None
    with pytest.raises(ValueError):
        slot_layout({"slots": {"num_slots": 2, "ranges": {"a": [0, 3]}, "verbalizers": ["x", "y"]}})
    with pytest.raises(ValueError, match="verbalizer"):
        slot_layout({"slots": {"num_slots": 2, "ranges": {"a": [0, 2]}}})


def test_each_question_is_scored_over_the_first_slots_of_its_type():
    m, lm, W, b = _model(temps={"noul": 1.0, "choice": 2.0, "score": 0.8})
    out = m.decide("the state", {
        "c": {"type": "choice", "instructions": "which?", "criteria": ["left", "right"]},
        "n": {"type": "noul", "instructions": "is it?"},
        "s": {"type": "score", "instructions": "how much?"}})
    a = out["answers"]
    z = W @ lm.h + b
    pc = _softmax(z[5:7] / 2.0)
    assert a["c"]["probabilities"] == {"left": round(float(pc[0]), 4),
                                       "right": round(float(pc[1]), 4)}
    assert a["c"]["choice"] == ("left" if pc[0] > pc[1] else "right")
    pn = _softmax(z[0:2])
    assert a["n"]["noul"] == round(float(pn[1]), 4)
    assert list(a["n"]["probabilities"]) == ["n", "y"]
    ps = _softmax(z[2:5] / 0.8)
    assert a["s"]["score"] == round(float((np.arange(3) * ps).sum()), 4)
    assert "act_probability" not in a["c"]
    assert lm.prompts[0] == ("[kind] choice\n[state] the state\n[question] which?\n"
                             "[options]\nA) left\nB) right\n[decision]:")
    assert lm.prompts[1].endswith("[options]\nn\ny\n[decision]:")
    assert out["usage"]["questions"] == 3
    assert out["usage"]["input_tokens"] == sum(len(p) for p in lm.prompts)


def test_options_a_model_does_not_read_are_refused():
    m, _, _, _ = _model()
    with pytest.raises(ValueError, match="takes 2 to 3"):
        m.decide("s", {"c": {"type": "choice", "instructions": "q",
                             "criteria": ["a", "b", "c", "d"]}})
    with pytest.raises(ValueError, match="its own levels"):
        m.decide("s", {"s": {"type": "score", "instructions": "q",
                             "criteria": ["low", "mid", "high"]}})
    with pytest.raises(ValueError, match="its own levels"):
        m.decide("s", {"n": {"type": "noul", "instructions": "q",
                             "criteria": {"n": "no", "y": "yes"}}})
    # Naming the model's own levels is the same as naming none.
    m.decide("s", {"s": {"type": "score", "instructions": "q", "criteria": ["0", "1", "2"]}})
    with pytest.raises(ValueError, match="not a question type"):
        m.decide("s", {"x": {"type": "rank", "instructions": "q"}})


def test_unknown_templates_and_mismatched_heads_are_refused():
    W, b = _head()
    lay = slot_layout(CFG)
    with pytest.raises(NotImplementedError):
        SlotDecisionModel(FakeLM(), W, b, dict(lay, template="bare-v9"))
    with pytest.raises(ValueError):
        SlotDecisionModel(FakeLM(), W[:, :3], b, lay)


def test_temperatures_fit_on_the_same_raw_logits():
    m, lm, W, b = _model()
    examples = [{"state": "s%d" % i,
                 "questions": {"n": {"type": "noul", "instructions": "q"}},
                 "answers": {"n": i % 2}} for i in range(24)]
    report = m.calibrate(examples, by_options=False, min_samples=20)
    assert "noul" in report["groups"]
    assert m.surface()["calibration"]["domain_calibrated"]


def test_the_surface_says_which_types_take_options():
    m, _, _, _ = _model(temps={"noul": 1.0})
    t = m.surface()["takes"]["questions"]["types"]
    assert t["choice"]["needs"] == "options" and t["choice"]["max"] == 3
    assert t["score"]["needs"] is None and t["score"]["levels"] == ["0", "1", "2"]
    assert t["noul"]["shape"] == "fixed" and t["noul"]["needs"] is None
    assert m.surface()["calibration"]["status"] == "checkpoint"


# ---- recognition ------------------------------------------------------------------------
def _safetensors(path, tensors):
    header, blobs, at = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [at, at + arr.nbytes]}
        blobs.append(arr.tobytes()); at += arr.nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw))); f.write(raw)
        for blob in blobs:
            f.write(blob)


@pytest.fixture
def folder(monkeypatch, tmp_path):
    async def read(name, offset=0, length=None):
        with open(name, "rb") as f:
            f.seek(offset)
            return f.read() if length is None else f.read(length)
    monkeypatch.setattr(webio, "_IO", read)
    monkeypatch.setattr(webio, "_local_files", {})
    monkeypatch.setattr(webio, "_local_roots", {})
    loads = []

    async def from_pretrained(path, **kw):
        lm = FakeLM()
        loads.append((path, kw, lm))
        return lm
    monkeypatch.setattr(_sdk.AutoModelForCausalLM, "from_pretrained", staticmethod(from_pretrained))
    W, b = _head()
    (tmp_path / "any_name_config.json").write_text(json.dumps(CFG))
    (tmp_path / "temps.json").write_text(json.dumps({"per_kind": {"noul": 1.5, "choice": 0.9}}))
    _safetensors(tmp_path / "w.safetensors", {"proj.weight": W, "proj.bias": b})
    (tmp_path / "backbone.gguf").write_bytes(b"")
    ad = tmp_path / "adapter"; ad.mkdir()
    (ad / "adapter_config.json").write_text("{}")
    (ad / "adapter_model.safetensors").write_bytes(b"")
    return tmp_path, loads


def test_a_folder_with_slots_a_head_and_a_model_is_a_decision_model(folder):
    root, loads = folder
    m = asyncio.run(decision._slot_model(str(root / "backbone.gguf"), "gguf"))
    assert isinstance(m, SlotDecisionModel)
    path, kw, lm = loads[0]
    assert path == str(root / "backbone.gguf") and kw["adapter"] == str(root / "adapter")
    assert m.cfg.temp_for("noul", 2) == 1.5 and m.cfg.temp_for("score", 3) == 1.0
    assert m.surface()["adapter"] == str(root / "adapter")
    # The same folder named as a directory finds the one model in it.
    asyncio.run(decision._slot_model(str(root), None))
    assert loads[1][0] == str(root / "backbone.gguf")
    # adapter=False runs the head on the language model as it is.
    asyncio.run(decision._slot_model(str(root / "backbone.gguf"), "gguf", adapter=False))
    assert loads[2][1]["adapter"] is False


def test_a_language_model_folder_is_not_one_and_large_files_are_not_read(folder, monkeypatch):
    root, loads = folder
    (root / "any_name_config.json").write_text(json.dumps({"architectures": ["X"]}))
    (root / "tokenizer.json").write_text(json.dumps({"slots": CFG["slots"]}) + " " * (1 << 20))
    seen = []
    real = webio.read_json

    async def read_json(src, io=None):
        seen.append(src.rsplit("/", 1)[-1])
        return await real(src, io)
    monkeypatch.setattr(webio, "read_json", read_json)
    assert asyncio.run(decision._slot_model(str(root / "backbone.gguf"), "gguf")) is None
    assert "tokenizer.json" not in seen and not loads


def test_a_tokenizer_that_disagrees_with_the_head_is_refused(folder):
    root, loads = folder
    bad = dict(CFG, verbalizer_ids=[1] * 8)
    (root / "any_name_config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="token ids"):
        asyncio.run(decision._slot_model(str(root / "backbone.gguf"), "gguf"))
    assert loads[0][2].released


def test_two_unnamed_adapters_are_a_question_not_a_guess(folder):
    root, _ = folder
    cfg = dict(CFG); cfg.pop("adapter_subfolder")
    (root / "any_name_config.json").write_text(json.dumps(cfg))
    other = root / "other"; other.mkdir()
    (other / "adapter_config.json").write_text("{}")
    (other / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="2 adapters"):
        asyncio.run(decision._slot_model(str(root / "backbone.gguf"), "gguf"))
