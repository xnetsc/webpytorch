"""Decision models: a state and some typed questions in, probability-scaled answers out.

A decision model does not write. It reads the situation once and returns, for each question,
a probability over the answers the CALLER named -- which option, what level, how likely. There
is no token stream, no sampling and no stopping rule, so none of the decoder machinery applies;
what it needs is an encoder pass and a scorer, and that is all this module is.

The request shape is the one this class of model already answers in -- a state plus a map of
typed questions -- so an application that can talk to one can talk to another. The three
question types are the model's own vocabulary, read from its files, not a set this SDK
invented:

    choice   pick one of the options, with a probability for each
    score    place it on an ordered scale, with the expected level
    noul     how likely the statement is to hold, in [0, 1]

The SDK does not decide anything here. It does not pick a threshold, does not turn a
probability into a yes, and does not choose what to ask; a caller that wants a decision made
gets the numbers to make it with.
"""
import json
import math
import sys
import time
import warnings

import numpy as np

from . import _core as wt
from ._core import Tensor, bmm, gelu, layernorm, softmax, transpose_last2
from .encoder import EncoderConfig, TextEncoder, _f32


# A temperature is a divisor on logits. Zero is undefined; a tiny positive value turns an
# ordinary lead into a displayed certainty. This is not peculiar to one checkpoint: any
# classifier/decision model that publishes softmax values has this boundary. Keep the guard
# here, where that class of model is recognised, rather than in a model-name adapter.
TEMPERATURE_MIN = 0.5
TEMPERATURE_MAX = 5.0


def temperature_bucket(qtype, k):
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (qtype, size)


def safe_temperature(value, low=TEMPERATURE_MIN, high=TEMPERATURE_MAX):
    """A finite temperature inside a usable range; invalid values become the neutral 1.0.

    The bounds are arguments so a caller fitting a specialised family can state a different
    measured range instead of patching the implementation.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return min(float(high), max(float(low), value))


def _probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64).reshape(-1) / safe_temperature(temperature)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def calibration_error(probabilities, targets, bins=15):
    """Expected calibration error for classification distributions.

    A reported 0.9 is calibrated when predictions reported around 0.9 are right around 90%
    of the time. ECE measures the gap, weighted across equally wide confidence bins.
    """
    rows = [np.asarray(p, dtype=np.float64).reshape(-1) for p in probabilities]
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if not rows or len(rows) != len(y):
        raise ValueError("probabilities and targets must contain the same non-zero number of rows")
    conf = np.asarray([float(p.max()) for p in rows])
    correct = np.asarray([int(p.argmax()) == int(t) for p, t in zip(rows, y)], dtype=np.float64)
    out = 0.0
    for i in range(int(bins)):
        lo, hi = i / bins, (i + 1) / bins
        take = (conf >= lo) & ((conf <= hi) if i == bins - 1 else (conf < hi))
        if take.any():
            out += float(take.mean()) * abs(float(conf[take].mean()) - float(correct[take].mean()))
    return float(out)


def fit_temperature(logits, targets, low=TEMPERATURE_MIN, high=TEMPERATURE_MAX,
                    iterations=64):
    """Fit one post-hoc temperature by held-out negative log likelihood.

    `logits` may contain rows of different widths, which is needed when one option-count
    bucket contains three-, four- and five-way questions. The class choice is unchanged by
    a positive temperature; only the probability scale is repaired. The return value keeps
    the before/after evidence beside the fitted number so callers can reject a fit that did
    not improve their held-out data.
    """
    rows = [np.asarray(x, dtype=np.float64).reshape(-1) for x in logits]
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if not rows or len(rows) != len(y):
        raise ValueError("logits and targets must contain the same non-zero number of rows")
    for i, (row, target) in enumerate(zip(rows, y)):
        if row.size < 2 or not np.isfinite(row).all():
            raise ValueError("logits row %d must contain at least two finite values" % i)
        if target < 0 or target >= row.size:
            raise ValueError("target %d is outside logits row %d (width %d)"
                             % (target, i, row.size))
    low, high = float(low), float(high)
    if not (0 < low < high and math.isfinite(low) and math.isfinite(high)):
        raise ValueError("temperature bounds must be finite and satisfy 0 < low < high")

    def nll(t):
        loss = 0.0
        for row, target in zip(rows, y):
            z = row / t
            m = float(z.max())
            loss += (m + math.log(float(np.exp(z - m).sum()))) - float(z[target])
        return loss / len(rows)

    # Temperature is positive, so search log(T). Golden-section search needs no scipy and
    # gives a deterministic result in CPython and Pyodide.
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(math.exp(c)), nll(math.exp(d))
    for _ in range(int(iterations)):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a); fc = nll(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a); fd = nll(math.exp(d))
    fitted = safe_temperature(math.exp((a + b) * 0.5), low, high)
    before = [_probabilities(row, 1.0) for row in rows]
    after = [_probabilities(row, fitted) for row in rows]
    return {"temperature": fitted, "samples": len(rows),
            "nll_before": nll(1.0), "nll_after": nll(fitted),
            "ece_before": calibration_error(before, y),
            "ece_after": calibration_error(after, y)}


def serialize_state(state):
    """A state reaches the model as text. Anything that is not already text becomes JSON --
    not a rendering of this SDK's choosing, the one the model was trained against."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


# The SHAPES a question can have: what the caller has to supply, and what the answer means.
#
# A shape is the thing the engine can act on. The NAME of a question type is whatever a
# given checkpoint calls it, and nothing here may depend on that -- branching on the name
# means a model that calls its types anything else silently gets another model's answer
# shape, which is what used to happen: every type that was not "choice" or "score" was
# rendered as a two-outcome statement and answered under a key named after one particular
# model family's vocabulary.
#
#   named    the caller names the options; the answer is one of them
#   ordered  the caller names the levels in order; the answer is a place on that scale
#   fixed    the caller supplies no options; two outcomes, and the second is the answer
#
# `named` is the general one: any question type can be asked that way, so it is what an
# undeclared or unrecognised type gets. The other two are narrowings a checkpoint has to
# ask for, because only its training knows that its levels are ordered or its outcomes two.
SHAPES = ("named", "ordered", "fixed")
GENERAL_SHAPE = "named"

# Checkpoints in circulation say how many question types they have -- one temperature and
# one type embedding each -- without saying what they are. For those, and only those, the
# names and shapes below are the convention this engine supplies. A checkpoint that
# declares `question_types` is read instead, and none of this applies to it; that is the
# path a new model should take, and it needs no change here to be supported.
_CONVENTIONAL = (("choice", "named"), ("score", "ordered"), ("noul", "fixed"))


def type_rows(weights):
    """How many question types the WEIGHTS have: one row of the type embedding each.

    The authoritative count, which is why it is preferred over the config's temperature
    list. A temperature list usually agrees, but it is calibration metadata -- it can be
    short, absent, or a single scalar -- while the embedding IS the model: a checkpoint
    cannot answer a type it has no row for, and must be able to answer every type it has.
    """
    try:
        n = int(wt.weight_shape((weights or {}).get("type_emb.weight"))[0])
    except Exception:
        return None
    return n or None


def render_options(shape, criteria):
    """The answer texts, in label order, for a question of this SHAPE.

    The shapes are not interchangeable: `fixed` is always two outcomes in that order, which
    is what makes the second probability the answer.
    """
    if shape == "ordered":
        crit = list(criteria or [])
        return [str(i) for i in range(len(crit))], ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    if shape == "fixed":
        crit = criteria if isinstance(criteria, dict) else {}
        return ["false", "true"], [
            "false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]
    crit = {c: None for c in criteria} if isinstance(criteria, list) else dict(criteria or {})
    return list(crit.keys()), [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]


def _question_layout(config, weight_names=()):
    """Resolve the input contract from the checkpoint, never its repository/model name.

    Weight shapes alone cannot reveal whether questions interacted during training.  An
    explicit layout declaration is authoritative.  Older per-question-row checkpoints did
    not write one, so recognize their *complete* config/tensor schema; an unknown schema
    stays unknown rather than silently receiving independent-question semantics.
    """
    declared = config.get("question_layout")
    if declared is not None:
        if declared not in ("per_question", "joint"):
            raise ValueError("unsupported decision question_layout %r" % declared)
        return declared, "checkpoint_metadata"
    names = set(weight_names or ())
    legacy_config = {"max_len", "head_max_len", "head_layers", "max_prefixes"} <= set(config)
    legacy_head = {"type_emb.weight", "scorer.0.weight", "scorer.3.weight",
                   "act_head.0.weight", "act_head.2.weight"} <= names
    if legacy_config and legacy_head and any(n.startswith("head.layers.0.") for n in names):
        return "per_question", "checkpoint_structure"
    return None, "undetermined"


class DecisionConfig(object):
    """The decision half of the model's own config: how long a sequence may be, how the
    answer distribution is scaled, and which question types exist."""

    def __init__(self, cfg, qtypes=None, type_count=None, weight_names=()):
        c = dict(cfg or {})
        self.type_count = type_count
        self.raw = c
        self.max_len = int(c.get("max_len", 512))
        self.head_max_len = int(c.get("head_max_len", 192))
        self.head_layers = int(c.get("head_layers", 0))
        self.max_questions = int(c.get("max_prefixes", 0) or 0)
        self.option_tokens = int(c.get("option_tokens", 48))
        self.question_layout, self.question_layout_source = _question_layout(c, weight_names)
        # Post-hoc calibration, fitted by whoever trained the model. Two levels: a
        # temperature per question type, and a finer one per type AND option count, because
        # a two-way question and a twenty-way one do not need the same scaling.
        self.qtypes, self.shapes = self._read_types(c, qtypes, type_count)
        raw_temperature = c.get("temperature", [1.0] * len(self.qtypes))
        self.temperature_raw = (list(raw_temperature) if isinstance(raw_temperature, (list, tuple))
                                else [raw_temperature] * len(self.qtypes))
        raw_buckets = c.get("temperature_by_options", {})
        self.temperature_by_options_raw = (dict(raw_buckets)
                                           if isinstance(raw_buckets, dict) else {})
        self.temperature = [safe_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {
            k: safe_temperature(v) for k, v in self.temperature_by_options_raw.items()
        }
        self.temperature_adjustments = []
        entries = [("temperature[%d]" % i, raw, applied)
                   for i, (raw, applied) in enumerate(zip(self.temperature_raw, self.temperature))]
        entries += [(k, raw, self.temperature_by_options[k])
                    for k, raw in self.temperature_by_options_raw.items()]
        for name, raw, applied in entries:
            try:
                unchanged = float(raw) == applied
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                self.temperature_adjustments.append(
                    {"entry": name, "raw": repr(raw), "applied": applied})
        self.checkpoint_calibration = bool("temperature" in c or "temperature_by_options" in c)
        self.domain_calibrated = set()
        if self.temperature_adjustments:
            warnings.warn(
                "decision checkpoint temperatures were invalid or outside [%g, %g]; "
                "safe values were applied. Treat the affected probabilities as uncalibrated: %s"
                % (TEMPERATURE_MIN, TEMPERATURE_MAX,
                   ", ".join("%s=%s -> %g" % (x["entry"], x["raw"], x["applied"])
                             for x in self.temperature_adjustments)),
                RuntimeWarning, stacklevel=2)

    @staticmethod
    def _read_types(c, qtypes=None, type_count=None):
        """The question types this checkpoint has, in the order its type embedding is in.

        Declared, if it says so: `question_types` as a list -- order is meaningful, it is
        the index into the type embedding and the temperatures -- of names, or of
        `{"name", "shape"}`. That is how a model says what it answers, and a model that
        says it needs nothing added here to be supported.

        Otherwise the count is still knowable -- from the type embedding's rows, which is
        the model itself, and failing that from however many temperatures the file carries --
        and only the names and shapes are the convention above. Anything past the
        conventional ones is named positionally and takes the general shape rather than
        being quietly given the last one's.
        """
        declared = qtypes if qtypes is not None else c.get("question_types")
        names, shapes = [], {}
        if declared:
            for i, item in enumerate(declared):
                if isinstance(item, dict):
                    name = str(item.get("name") or item.get("type") or i)
                    shape = str(item.get("shape") or item.get("options") or GENERAL_SHAPE)
                else:
                    name = str(item)
                    shape = dict(_CONVENTIONAL).get(name, GENERAL_SHAPE)
                names.append(name)
                shapes[name] = shape if shape in SHAPES else GENERAL_SHAPE
            return names, shapes
        t = c.get("temperature")
        count = (type_count
                 or (len(t) if isinstance(t, (list, tuple)) and t else 0)
                 or len(_CONVENTIONAL))
        for i in range(count):
            name, shape = (_CONVENTIONAL[i] if i < len(_CONVENTIONAL)
                           else ("type%d" % i, GENERAL_SHAPE))
            names.append(name)
            shapes[name] = shape
        return names, shapes

    def shape_of(self, qtype):
        """What a question of this type takes and what its answer means. A type this
        checkpoint never declared is asked the general way rather than guessed at."""
        return self.shapes.get(qtype, GENERAL_SHAPE)

    def temp_for(self, qtype, k):
        idx = self.qtypes.index(qtype)
        default = self.temperature[idx] if idx < len(self.temperature) else 1.0
        return float(self.temperature_by_options.get(temperature_bucket(qtype, k), default))

    def calibration(self):
        if self.domain_calibrated:
            status = "held-out"
        elif self.temperature_adjustments:
            status = "guarded"
        elif self.checkpoint_calibration:
            status = "checkpoint"
        else:
            status = "uncalibrated"
        return {
            "method": "temperature-scaling",
            "status": status,
            "domain_calibrated": bool(self.domain_calibrated),
            "groups": sorted(self.domain_calibrated),
            "safe_range": [TEMPERATURE_MIN, TEMPERATURE_MAX],
            "adjustments": list(self.temperature_adjustments),
            "note": ("A displayed 0.90 is a model probability, not evidence of 90% accuracy "
                     "on this application's data. Fit on separate labelled held-out examples "
                     "before using probability thresholds."),
        }


def confidence(p):
    """How concentrated an answer is: 1 for a point mass, 0 for a uniform distribution.

    This deliberately says nothing about correctness or calibration. It reports the shape
    of this one distribution, not how often similarly shaped predictions prove correct.
    """
    k = len(p)
    if k < 2:
        return 1.0
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    return float(1.0 - ent / math.log(k))


def answer_confidence(p):
    """Probability mass on the reported answer: ``max(p)``.

    Temperature fitting and ECE operate on this quantity.  Keep it separate from
    ``confidence()``, which is distribution concentration and is not a calibrated
    probability of correctness.
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if not len(p):
        return 1.0
    return float(np.clip(p.max(), 0.0, 1.0))


class DecisionModel(wt.Module):
    """An encoder, a small transformer head, and a scorer that reads one position per option.

    Every option is written into the sequence behind a marker token, and the answer is the
    scorer's reading of those marker positions. That is why the options can be anything the
    caller names without retraining: they are input, not classes.
    """

    def __init__(self, enc_cfg, dec_cfg, weights, tokenizer, mask_id, cls_id, sep_id, pad_id):
        if dec_cfg.question_layout is None:
            raise ValueError("decision checkpoint does not declare question_layout and its "
                             "structure does not identify a supported question contract")
        if dec_cfg.question_layout != "per_question":
            raise NotImplementedError("decision checkpoint question_layout %r needs a matching "
                                      "sequence and head adapter" % dec_cfg.question_layout)
        self.cfg = dec_cfg
        self.enc = TextEncoder(enc_cfg, {k: v for k, v in weights.items()
                                         if k.startswith("encoder.")})
        head = {k: v for k, v in weights.items() if not k.startswith("encoder.")}
        # Same one-copy rule as the encoder: build each tensor once, drop the file's array.
        self.have = set(head)
        self._src = dict(head)
        self._ten = {}
        self.tok = tokenizer
        self.mask_id, self.cls_id, self.sep_id, self.pad_id = mask_id, cls_id, sep_id, pad_id
        self.hidden = enc_cfg.hidden
        self.heads = enc_cfg.heads
        self.head_dim = enc_cfg.head_dim
        self.eps = enc_cfg.eps
        # The head's layer count is whatever the checkpoint carries, not what a config claims.
        self.n_head_layers = 1 + max([-1] + [int(k.split(".")[2]) for k in self.have
                                             if k.startswith("head.layers.")])

    def release(self):
        """Drop the weights. See `TextEncoder.release` for why a model says this itself."""
        enc = self.__dict__.get("enc")
        if enc is not None and callable(getattr(enc, "release", None)):
            enc.release()
        self.__dict__.update(enc=None, _ten={}, _src={})
        self.__dict__["_released"] = True
        return self

    # ---- small helpers over the raw weights -----------------------------------------
    def _t(self, n, transposed=False):
        key = (n, transposed)
        got = self._ten.get(key)
        if got is None:
            a = _f32(self._src[n])
            got = Tensor(np.ascontiguousarray(a.T) if transposed else a)
            self._ten[key] = got
            if not (transposed and (n, False) in self._ten):
                self._src.pop(n, None)
        return got

    def _lin(self, x, n):
        """`y = x @ W^T + b`.

        Two spellings, because a projection that is a submodule saves as `name.weight` while
        one that is a plain parameter on its parent saves as `name_weight` -- which is how
        a stock multi-head attention stores its fused input projection. Same arithmetic; the
        checkpoint decides which name it is under.
        """
        wn = n + ".weight" if (n + ".weight") in self.have else n + "_weight"
        bn = n + ".bias" if (n + ".bias") in self.have else n + "_bias"
        key = (wn, "stored-linear")
        linear = self._ten.get(key)
        if linear is None:
            src = self._src.get(wn)
            native = wt.stored_linear(src)
            if native is not None:
                linear = native
                self._ten[key] = linear
                self._src.pop(wn, None)
        if linear is not None:
            y = linear(x)
        else:
            y = x.matmul(self._t(wn, transposed=True))
        return y + self._t(bn) if bn in self.have else y

    def _ln(self, x, n):
        b = (self._t(n + ".bias") if (n + ".bias") in self.have else self._zero())
        return layernorm(x, self._t(n + ".weight"), b, 1e-5)

    def _zero(self):
        if "__zero__" not in self._ten:
            self._ten["__zero__"] = Tensor(np.zeros((self.hidden,), np.float32))
        return self._ten["__zero__"]

    # ---- the sequence ----------------------------------------------------------------
    def build_sequence(self, state, qtype, instructions, criteria, *, state_ids=None):
        """`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`

        Options are placed ahead of state; if the checkpoint's own head budget overflows,
        each option is shortened evenly, matching the checkpoint's training/runtime
        contract. A larger head budget needs an explicit model-side calibration decision,
        not an implicit change to the scores users already interpret.
        """
        mask_str = self.tok.dec.get(self.mask_id, "")

        def clean(s):
            return str(s).replace(mask_str, " ") if mask_str else str(s)

        labels, opts = render_options(self.cfg.shape_of(qtype), criteria)
        head_ids = self.tok.encode("%s question: %s" % (qtype, clean(instructions)))
        opt_ids = [[self.mask_id] + self.tok.encode(" " + clean(o))[:self.cfg.option_tokens]
                   for o in opts]
        budget = self.cfg.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (self.cfg.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = self.cfg.head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[:max(8, budget)]
        ids = [self.cls_id] + head_ids + [self.sep_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(self.sep_id)
        room = max(0, self.cfg.max_len - len(ids) - 1)
        if state_ids is None:
            state_ids = self.tok.encode(clean(serialize_state(state)))
        ids = ids + state_ids[:room] + [self.sep_id]
        return ids[:self.cfg.max_len], [m for m in markers if m < self.cfg.max_len], labels

    # ---- the head --------------------------------------------------------------------
    def _head_layer(self, x, i, B=1, mask=None):
        """One pre-norm transformer block, in the layout the checkpoint was saved in.

        The activation is ReLU, not GELU: this head is a stock `TransformerEncoderLayer` and
        that is its default. It is the kind of detail that cannot be read off the weights --
        an activation has no parameters -- so it is read off the reference implementation.
        """
        p = "head.layers.%d." % i
        h, hd, T = self.heads, self.head_dim, x.shape[0] // B
        profile = getattr(self, "_head_profile", False)
        marks = []
        def mark(label):
            if profile:
                marks.append((label, time.perf_counter()))
        mark("start")
        a = self._ln(x, p + "norm1")
        mark("norm1")
        qkv = self._lin(a, p + "self_attn.in_proj")
        mark("qkv")
        d = h * hd
        q = None if qkv.requires_grad else wt.qkv_take(qkv, 0, h, hd, T, B=B)
        if q is not None:
            k = wt.qkv_take(qkv, 1, h, hd, T, B=B)
            v = wt.qkv_take(qkv, 2, h, hd, T, B=B)
        else:
            def heads_first(start, end):
                return (wt._slice_last(qkv, start, end).reshape(B, T, h, hd)
                        .permute(0, 2, 1, 3).reshape(B * h, T, hd))
            q, k, v = heads_first(0, d), heads_first(d, 2 * d), heads_first(2 * d, 3 * d)
        mark("qkv_layout")
        scores = bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))
        mark("qk")
        if mask is not None:
            scores = scores + mask
        mark("mask")
        probs = softmax(scores)
        mark("softmax")
        o = bmm(probs, v)
        mark("value")
        o = o.reshape(B, h, T, hd).permute(0, 2, 1, 3).reshape(B * T, d)
        x = x + self._lin(o, p + "self_attn.out_proj")
        mark("out_proj")
        f = self._ln(x, p + "norm2")
        f = self._lin(wt.ReLU()(self._lin(f, p + "linear1")), p + "linear2")
        mark("mlp")
        if profile:
            self._full_head_timing = {
                marks[j][0]: round((marks[j][1] - marks[j - 1][1]) * 1000, 3)
                for j in range(1, len(marks))
            }
        return x + f

    @staticmethod
    def _select_rows(x, rows):
        """Select arbitrary device rows through the common Tensor contract.

        On CPU, selecting rows directly avoids constructing a dense one-hot matrix and
        multiplying every row. Keep the measured device route for WebGPU and WebGL,
        and the differentiable matmul route for autograd.
        """
        rows = [int(r) for r in rows]
        if not x.requires_grad and not wt.GPU:
            return x[rows]
        direct = wt.gather_rows(x, rows)
        if direct is not None:
            return direct
        pick = np.zeros((len(rows), int(x.shape[0])), dtype=np.float32)
        pick[np.arange(len(rows)), rows] = 1.0
        return Tensor(pick).matmul(x)

    def _last_head_selected(self, x, i, rows, queries_only=False):
        """Compute only required output rows of the final transformer head layer.

        This is topology-based, not tied to a checkpoint or model name: when no later layer
        consumes the other rows, their output projection and MLP work is dead.  K/V still
        cover the whole bidirectional sequence.  ``queries_only`` additionally selects the
        required Q rows before attention; both variants preserve the original dtype.
        """
        p = "head.layers.%d." % i
        h, hd, T = self.heads, self.head_dim, x.shape[0]
        a = self._ln(x, p + "norm1")
        qkv = self._lin(a, p + "self_attn.in_proj")
        d = h * hd
        q0 = wt._slice_last(qkv, 0, d)
        k = None if qkv.requires_grad else wt.qkv_take(qkv, 1, h, hd, T)
        if k is not None:
            v = wt.qkv_take(qkv, 2, h, hd, T)
        else:
            k = wt._slice_last(qkv, d, 2 * d).reshape(T, h, hd).permute(1, 0, 2)
            v = wt._slice_last(qkv, 2 * d, 3 * d).reshape(T, h, hd).permute(1, 0, 2)
        if queries_only:
            q0 = self._select_rows(q0, rows)
            q = q0.reshape(len(rows), h, hd).permute(1, 0, 2)
            o = bmm(softmax(bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))), v)
            o = o.permute(1, 0, 2).reshape(len(rows), d)
        else:
            q = q0.reshape(T, h, hd).permute(1, 0, 2)
            o = bmm(softmax(bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))), v)
            o = o.permute(1, 0, 2).reshape(T, d)
            o = self._select_rows(o, rows)
        y = self._select_rows(x, rows) + self._lin(o, p + "self_attn.out_proj")
        f = self._ln(y, p + "norm2")
        f = self._lin(wt.ReLU()(self._lin(f, p + "linear1")), p + "linear2")
        return y + f

    def _last_head_selected_many(self, x, i, lengths, padded, markers, mask):
        """Final bidirectional head layer for only the rows consumed by each scorer.

        K/V still span every real token of each independent question.  Query and
        following projection/MLP rows are limited to CLS and option markers, padded
        only to a common *row* count so one batched matrix path serves every question.
        """
        B = len(lengths)
        R = max(len(ms) + 1 for ms in markers)
        h, hd, d = self.heads, self.head_dim, self.heads * self.head_dim
        p = "head.layers.%d." % i
        rows = []
        for b, ms in enumerate(markers):
            own = [0] + [int(m) for m in ms]
            if any(r < 0 or r >= lengths[b] for r in own):
                raise ValueError("decision option marker lies outside its question")
            rows.extend(b * padded + r for r in own + [0] * (R - len(own)))
        profile = getattr(self, "_head_profile", False)
        marks = []
        def mark(label):
            if profile:
                marks.append((label, time.perf_counter()))
        mark("rows")
        a = self._ln(x, p + "norm1")
        mark("norm1")
        qkv = self._lin(a, p + "self_attn.in_proj")
        mark("qkv")
        q0 = self._select_rows(wt._slice_last(qkv, 0, d), rows)
        mark("select_q")
        k = None if qkv.requires_grad else wt.qkv_take(qkv, 1, h, hd, padded, B=B)
        if k is not None:
            v = wt.qkv_take(qkv, 2, h, hd, padded, B=B)
        else:
            def heads_first(start, end):
                return (wt._slice_last(qkv, start, end).reshape(B, padded, h, hd)
                        .permute(0, 2, 1, 3).reshape(B * h, padded, hd))
            k, v = heads_first(d, 2 * d), heads_first(2 * d, 3 * d)
        mark("kv")
        q = q0.reshape(B, R, h, hd).permute(0, 2, 1, 3).reshape(B * h, R, hd)
        scores = bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5)) + mask
        o = bmm(softmax(scores), v)
        o = o.reshape(B, h, R, hd).permute(0, 2, 1, 3).reshape(B * R, d)
        mark("attention")
        y = self._select_rows(x, rows) + self._lin(o, p + "self_attn.out_proj")
        mark("out_proj")
        f = self._ln(y, p + "norm2")
        f = self._lin(wt.ReLU()(self._lin(f, p + "linear1")), p + "linear2")
        mark("mlp")
        if profile:
            self._selected_head_timing = {
                marks[i][0]: round((marks[i][1] - marks[i - 1][1]) * 1000, 3)
                for i in range(1, len(marks))
            }
        return y + f, R

    @staticmethod
    def batch_pays(longest, count):
        # Batching is *eligible* whenever rows are independent; a fixed token cutoff made
        # a three-question request of 162 tokens run three full encoder/head passes. Which
        # route wins is instead learned for this backend and physical shape below.
        return count >= 2

    def _batch_route(self, longest, count, option_counts):
        """Use the measured winner, defaulting to batch until both routes are measured.

        Product requests must not be used as an alternating scalar/batch experiment: that
        makes every other multi-question request linear again. Separate calibration may
        supply distinct-question measurements for both routes; absent that evidence, the
        one-forward batch path is the safe default. The profile is per model instance,
        hence scoped to its device, backend and weight representation.
        """
        backend = ("webgpu" if wt._adam_backend_ready() else
                   "webgl" if wt._webgl_ready() else "cpu")
        key = (backend, count, (longest + 31) // 32, max(option_counts or (0,)))
        plans = self.__dict__.setdefault("_batch_profiles", {})
        profile = plans.setdefault(key, {"batch": [], "scalar": []})
        if len(profile["batch"]) >= 3 and len(profile["scalar"]) >= 3:
            return ("batch" if np.median(profile["batch"][2:])
                    < np.median(profile["scalar"][2:]) else "scalar"), key
        return "batch", key

    def _batch_observed(self, key, route, elapsed_ms):
        profile = self._batch_profiles[key]
        profile[route].append(float(elapsed_ms))

    def _run_one(self, ids, markers, qtype_idx):
        return self._score(self.enc.encode(ids), markers, qtype_idx)

    def _packed_score(self, h, markers):
        """Scorer/action result as one device tensor and one eventual readback."""
        T = h.shape[0]
        m = self._select_rows(h, markers)
        pooled_t = self._select_rows(h, [0])
        s = self._ln(m, "scorer.0")
        s = self._lin(gelu(self._lin(s, "scorer.1")), "scorer.3")
        feats = wt.decision_features(s.reshape(1, -1), len(markers))
        a = wt.cat([pooled_t, feats], axis=1)
        a = self._lin(gelu(self._lin(a, "act_head.0")), "act_head.2")
        return wt.cat([s.reshape(-1), a.reshape(-1)], axis=0)

    def _packed_score_many(self, h, padded, markers, selected_rows=0):
        """Score every question's options and action in batched device operations.

        The checkpoint's batch axis has one sequence per question. Variable option
        counts are padded only in the scorer input; the feature reduction ignores those
        lanes. Neither scorer nor action head runs a Python loop over questions.
        """
        B = len(markers)
        width = max(map(len, markers))
        stride = selected_rows or padded
        pooled_rows = [b * stride for b in range(B)]
        option_rows = []
        for b, ms in enumerate(markers):
            own = ([b * stride + j + 1 for j in range(len(ms))] if selected_rows else
                   [b * stride + int(pos) for pos in ms])
            option_rows.extend(own + [own[0]] * (width - len(own)))
        pooled = self._select_rows(h, pooled_rows)
        selected = self._select_rows(h, option_rows)
        scores = self._ln(selected, "scorer.0")
        scores = self._lin(gelu(self._lin(scores, "scorer.1")), "scorer.3")
        logits = scores.reshape(B, width)
        features = wt.decision_features_many(logits, [len(ms) for ms in markers])
        action_input = wt.cat([pooled, features], axis=1)
        action = self._lin(gelu(self._lin(action_input, "act_head.0")), "act_head.2")
        return wt.cat([logits.reshape(-1), action.reshape(-1)], axis=0), width

    def _score(self, h, markers, qtype_idx):
        h = h + wt.embedding(self._t("type_emb.weight"),
                             np.full((h.shape[0],), qtype_idx, dtype=np.int64))
        for i in range(max(0, self.n_head_layers - 1)):
            h = self._head_layer(h, i)
        last = self.n_head_layers - 1
        rows = [0] + [int(m) for m in markers]

        def run(which):
            if which == "full" or last < 0:
                out = self._head_layer(h, last) if last >= 0 else h
                return self._packed_score(out, markers).data
            out = self._last_head_selected(h, last, rows, queries_only=(which == "selected_q"))
            # Selected rows are ordered [pooled, marker0, marker1, ...].
            return self._packed_score(out, range(1, len(rows))).data

        mode = "full"
        if last >= 0:
            reference = [None]

            def host(which):
                raw = run(which)
                return np.asarray(raw.get() if hasattr(raw, "get") else raw, np.float32)

            def correct(which):
                if which == "full":
                    return True
                if reference[0] is None:
                    reference[0] = host("full")
                got = host(which)
                if not np.all(np.isfinite(got)):
                    return False
                scale = max(1e-6, float(np.abs(reference[0]).max()))
                return float(np.abs(got - reference[0]).max()) / scale < 1e-3

            backend = ("webgpu" if wt._adam_backend_ready() else
                       "webgl" if wt._webgl_ready() else "cpu")
            token_bucket = ((int(h.shape[0]) + 31) // 32) * 32
            row_bucket = 1 << (len(rows) - 1).bit_length()
            mode = wt._weight_execution("decision_head_" + backend, "selected_rows",
                                        token_bucket, row_bucket, 1, run,
                                        candidates=("full", "selected_full", "selected_q"),
                                        check=correct, repeat=1)
        self._head_execution = mode
        raw = run(mode)
        packed = np.asarray(raw.get() if hasattr(raw, "get") else raw).reshape(-1)
        logits = packed[:len(markers)]
        act = packed[len(markers):]
        act = np.exp(act - act.max()); act = act / act.sum()
        return logits, float(act[0])

    def _decision_key_mask(self, lengths, padded):
        """Stage one broadcast key-validity row per head, without Python-side data work.

        Browser GPU paths fill the shared upload arena in JS; browser CPU fills a
        borrowed NumPy byte view in JS. Native non-Pyodide CPU tests retain NumPy
        because no JavaScript worker exists there.
        """
        B, heads = len(lengths), self.heads
        shape = (B * heads, 1, padded)
        length_bytes = np.asarray(lengths, dtype=np.int32).view(np.uint8)
        if wt._adam_backend_ready() or wt._webgl_ready():
            import js
            target = wt._empty(shape)
            backend = js.gpu if wt._adam_backend_ready() else js.gl
            backend.stageDecisionKeyMask(int(target.buffer.buffer_id), length_bytes,
                                         B, heads, padded)
            return Tensor(target)
        target = np.empty(shape, dtype=np.float32)
        if sys.platform == "emscripten":
            import js
            js.decision.fillKeyMask(target.view(np.uint8), length_bytes,
                                    B, heads, padded)
        else:
            target.fill(0)
            for b, length in enumerate(lengths):
                if length < padded:
                    target[b * heads:(b + 1) * heads, :, length:] = -1e9
        return Tensor(target)

    def _score_many(self, hidden, lengths, padded, markers, qtype_indices):
        """Run independent question rows through the whole transformer head in one batch.

        This is a layout capability of the per-question-row checkpoint contract, not a
        claim that every decision model's questions are independent.  Mask each row's
        padding before softmax; otherwise a shorter question changes when batched beside
        a longer one.  The scorer still reads that question's own option markers.
        """
        B = len(lengths)
        if B < 2 or len(markers) != B or len(qtype_indices) != B:
            raise ValueError("parallel decision head needs one row of metadata per question")
        profile = getattr(self, "_head_profile", False)
        started = time.perf_counter() if profile else 0
        types = np.repeat(np.asarray(qtype_indices, dtype=np.int64), padded)
        h = hidden + wt.embedding(self._t("type_emb.weight"), types)
        selected_final = self.n_head_layers > 0 and callable(
            getattr(self, "_last_head_selected_many", None))
        full_layers = self.n_head_layers - (1 if selected_final else 0)
        mask = self._decision_key_mask(lengths, padded) if self.n_head_layers else None
        prepared = time.perf_counter() if profile else 0
        layer_times = []
        for i in range(full_layers):
            layer_started = time.perf_counter() if profile else 0
            h = self._head_layer(h, i, B=B, mask=mask)
            if profile:
                layer_times.append(round((time.perf_counter() - layer_started) * 1000, 3))
        selected_rows = 0
        if selected_final:
            selected_started = time.perf_counter() if profile else 0
            h, selected_rows = self._last_head_selected_many(
                h, self.n_head_layers - 1, lengths, padded, markers, mask)
            if profile:
                layer_times.append(round((time.perf_counter() - selected_started) * 1000, 3))
        layers_queued = time.perf_counter() if profile else 0
        packed, option_width = self._packed_score_many(
            h, padded, markers, selected_rows)
        raw = packed.data
        scores_queued = time.perf_counter() if profile else 0
        combined = np.asarray(raw.get() if hasattr(raw, "get") else raw).reshape(-1)
        read_back = time.perf_counter() if profile else 0
        outputs = []
        action_start = B * option_width
        for row, ms in enumerate(markers):
            logits = combined[row * option_width:row * option_width + len(ms)]
            act = combined[action_start + row * 2:action_start + row * 2 + 2]
            act = np.exp(act - act.max()); act = act / act.sum()
            outputs.append((logits, float(act[0])))
        self._head_execution = "batched_selected_q" if selected_final else "batched_full"
        if profile:
            finished = time.perf_counter()
            # The readback is the first fence after encoder replay. Its wait includes
            # outstanding encoder AND head GPU work; it is not head-only GPU time.
            self._head_timing = {
                "prepare_ms": round((prepared - started) * 1000, 3),
                "layers_queue_ms": round((layers_queued - prepared) * 1000, 3),
                "scores_queue_ms": round((scores_queued - layers_queued) * 1000, 3),
                "gpu_wait_readback_ms": round((read_back - scores_queued) * 1000, 3),
                "result_ms": round((finished - read_back) * 1000, 3),
                "layer_ms": layer_times,
                "selected_ms": getattr(self, "_selected_head_timing", None),
                "full_ms": getattr(self, "_full_head_timing", None),
            }
        return outputs

    # ---- the API ---------------------------------------------------------------------
    def _prepare_questions(self, state, questions):
        total = 0
        # Every sequence is built first, because whether to run them together depends on how
        # long the longest one turned out to be.
        built = []
        state_ids = None
        if questions:
            mask_str = self.tok.dec.get(self.mask_id, "")
            state_text = str(serialize_state(state))
            if mask_str:
                state_text = state_text.replace(mask_str, " ")
            state_ids = self.tok.encode(state_text)
        for qid, q in (questions or {}).items():
            qtype = q["type"]
            if qtype not in self.cfg.qtypes:
                raise ValueError("%r is not a question type this model answers (%s)"
                                 % (qtype, ", ".join(self.cfg.qtypes)))
            ins = q.get("instructions")
            ins = ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)
            ids, markers, labels = self.build_sequence(state, qtype, ins, q.get("criteria"),
                                                       state_ids=state_ids)
            if len(markers) != len(labels):
                raise ValueError("question %r: its options need more than %d tokens, so %d of "
                                 "them have no place in the sequence to be scored at"
                                 % (qid, self.cfg.head_max_len, len(labels) - len(markers)))
            total += len(ids)
            built.append((qid, q, qtype, ids, markers, labels))
        return built, total

    def _raw_questions(self, built, execution=None):
        request_start = time.perf_counter()
        profile = execution is not None and execution.get("_profile", False)
        self._head_profile = profile
        self._head_timing = None
        encoder_ms = 0.0
        head_ms = 0.0
        device_batch = None
        # Repeated identical jobs share one result. Different question metadata can have
        # identical encoder ids but different head answers; give those jobs separate batch
        # rows rather than silently switching the whole request to scalar execution.
        jobs = {}
        for row in built:
            job = (tuple(row[3]), tuple(row[4]), row[2])
            jobs.setdefault(job, row)
        sequences = [row[3] for row in jobs.values()]
        route, route_key = "scalar", None
        if sequences and self.batch_pays(max(map(len, sequences)), len(sequences)):
            chooser = getattr(self, "_batch_route", None)
            if callable(chooser):
                route, route_key = chooser(max(map(len, sequences)), len(sequences),
                                           [len(row[4]) for row in jobs.values()])
            else:
                route = "batch"
        if route == "batch":
            stage_start = time.perf_counter()
            tune_start = wt._WEIGHT_TUNE_SECONDS if profile else 0.0
            tune_calls = wt._WEIGHT_TUNE_CALLS if profile else 0
            device_batch = self.enc._encode_many_device(sequences)
            encoder_ms += (time.perf_counter() - stage_start) * 1000
            if profile:
                execution["encoder_tune_ms"] = round(
                    (wt._WEIGHT_TUNE_SECONDS - tune_start) * 1000, 3)
                execution["encoder_tune_calls"] = wt._WEIGHT_TUNE_CALLS - tune_calls

        encoded = {}
        scored_cache = {}
        encoder_tokens = 0
        encoder_passes = 0
        head_passes = 0
        head_batched = False
        scored = []
        if device_batch is not None:
            if len(sequences) < 2:
                raise RuntimeError("batch route selected without multiple head jobs")
            batch_h, lengths, padded = device_batch
            stage_start = time.perf_counter()
            tune_start = wt._WEIGHT_TUNE_SECONDS if profile else 0.0
            tune_calls = wt._WEIGHT_TUNE_CALLS if profile else 0
            outputs = self._score_many(
                batch_h, lengths, padded,
                [job[1] for job in jobs],
                [self.cfg.qtypes.index(job[2]) for job in jobs])
            if len(outputs) != len(jobs):
                raise RuntimeError("batched decision head returned the wrong number of rows")
            head_passes = 1
            head_batched = True
            head_ms += (time.perf_counter() - stage_start) * 1000
            if profile:
                execution["head_tune_ms"] = round(
                    (wt._WEIGHT_TUNE_SECONDS - tune_start) * 1000, 3)
                execution["head_tune_calls"] = wt._WEIGHT_TUNE_CALLS - tune_calls
            scored_cache.update(zip(jobs, outputs))
        for qid, q, qtype, ids, markers, labels in built:
            key = tuple(ids)
            score_key = (key, tuple(markers), qtype)
            if score_key in scored_cache:
                logits, act = scored_cache[score_key]
                scored.append((qid, q, qtype, markers, labels, logits, act))
                continue
            if device_batch is not None:
                raise RuntimeError("batched decision head did not score a requested job")
            else:
                # Questions can collapse to the exact same encoder input (for example two
                # identical boolean statements under different caller ids). Encoding that
                # sequence twice cannot change the answer, so share it within this call.
                if key not in encoded:
                    stage_start = time.perf_counter()
                    encoded[key] = self.enc.encode(ids)
                    encoder_ms += (time.perf_counter() - stage_start) * 1000
                    encoder_tokens += len(ids)
                    encoder_passes += 1
                stage_start = time.perf_counter()
                logits, act = self._score(encoded[key], markers, self.cfg.qtypes.index(qtype))
                head_ms += (time.perf_counter() - stage_start) * 1000
            scored_cache[score_key] = (logits, act)
            head_passes += 1
            scored.append((qid, q, qtype, markers, labels, logits, act))
        if device_batch is not None:
            encoder_passes = 1 if sequences else 0
            encoder_tokens = len(sequences) * max(map(len, sequences)) if sequences else 0
        actual_route = "batch" if head_batched else "scalar"
        if route_key is not None and actual_route in ("batch", "scalar"):
            self._batch_observed(route_key, actual_route,
                                 (time.perf_counter() - request_start) * 1000)
        if execution is not None:
            execution.update({"encoder_tokens": encoder_tokens,
                              "encoder_passes": encoder_passes,
                              "head_passes": head_passes,
                              "batched": actual_route == "batch"})
            if profile:
                execution.update({"batch_route": actual_route,
                                  "encoder_ms": round(encoder_ms, 3),
                                  "head_ms": round(head_ms, 3)})
                capture = getattr(self.enc, "_last_capture_timing", None)
                if capture:
                    execution["encoder_capture"] = capture
                if head_batched and self._head_timing:
                    execution["head_timing"] = self._head_timing
            if hasattr(self, "_head_execution"):
                execution["head_execution"] = self._head_execution
        return scored

    def decide(self, state, questions, *, profile=False):
        """Answer every question about this state.

        `questions` is `{id: {"type", "instructions", "criteria"}}`, and the answers come back
        under the same ids. Each carries its whole distribution, not only the winner, because
        a caller that wants to act on "0.51 versus 0.49" has to be able to see it.
        """
        out = {}
        stage_start = time.perf_counter()
        built, total = self._prepare_questions(state, questions)
        prepare_ms = (time.perf_counter() - stage_start) * 1000
        execution = {"_profile": True} if profile else {}
        stage_start = time.perf_counter()
        for qid, q, qtype, markers, labels, logits, act in self._raw_questions(built, execution):
            z = logits / self.cfg.temp_for(qtype, len(markers))
            p = np.exp(z - z.max()); p = p / p.sum()
            shape = self.cfg.shape_of(qtype)
            legacy_confidence = answer_confidence(p) if shape == "fixed" else confidence(p)
            ans = {"type": qtype, "shape": shape,
                   "probabilities": {l: round(float(v), 4) for l, v in zip(labels, p)},
                   "confidence": round(legacy_confidence, 4),
                   "answer_confidence": round(answer_confidence(p), 4),
                   "act_probability": round(act, 4)}
            # Keyed by the shape, which is the thing that decides what the number MEANS. A
            # type this checkpoint never declared lands on the general shape and is answered
            # like any other named question, rather than being reported under a key borrowed
            # from one model family's two-outcome type.
            if shape == "ordered":
                ans["score"] = round(float((np.arange(len(p)) * p).sum()), 4)
                ans["legend"] = {str(i): c for i, c in enumerate(q.get("criteria") or [])}
            elif shape == "fixed":
                ans["noul"] = round(float(p[1]), 4)
            else:
                ans["choice"] = labels[int(p.argmax())]
            out[qid] = ans
        answer_ms = (time.perf_counter() - stage_start) * 1000 - (
            execution.get("encoder_ms", 0) + execution.get("head_ms", 0))
        usage = {"input_tokens": total, "output_tokens": 0,
                 "questions": len(built),
                 "sequence_tokens": {str(b[0]): len(b[3]) for b in built}}
        execution.pop("_profile", None)
        usage.update(execution)
        if profile:
            usage.update({"prepare_ms": round(prepare_ms, 3),
                          "answer_ms": round(max(0.0, answer_ms), 3)})
        return {"answers": out, "usage": usage}

    @staticmethod
    def _target_index(shape, truth, labels, qtype=None):
        """Which answer a labelled example says is right, as an index into `labels`.

        By shape, like everything else that depends on what an answer means: a `fixed`
        question's truth is a yes or a no however it was written down, and every other
        shape's is one of the labels, or its position among them.
        """
        if isinstance(truth, dict):
            truth = truth.get(qtype, truth.get(shape, truth.get("target", truth.get("label"))))
        if shape == "fixed":
            if isinstance(truth, str):
                v = truth.strip().lower()
                if v in ("true", "yes", "1"): return 1
                if v in ("false", "no", "0"): return 0
            if isinstance(truth, (bool, int, np.integer)) and int(truth) in (0, 1):
                return int(truth)
        else:
            if isinstance(truth, str) and truth in labels:
                return labels.index(truth)
            if isinstance(truth, (int, np.integer)) and not isinstance(truth, bool) \
                    and 0 <= int(truth) < len(labels):
                return int(truth)
        raise ValueError("label %r is not one of the answers for this %s question (%s)"
                         % (truth, qtype or shape, ", ".join(labels)))

    def calibrate(self, examples, by_options=True, min_samples=20):
        """Fit probability temperatures on separate labelled held-out examples.

        Each item is `{"state": ..., "questions": {...}, "answers": {id: truth}}`. A truth is
        read by the question's SHAPE: a named option (or its position) for `named`, a level
        index for `ordered`, a yes or a no for `fixed`.
        With `by_options=True` (the default), the same option-count buckets used at inference
        are fitted independently; otherwise one value is fitted per question type.

        The model's chosen class cannot change: positive temperature scaling only repairs
        the numeric probability scale. The fit always starts from raw logits and replaces the
        matching checkpoint temperature; it is never composed with, or fitted on, probabilities
        that have already been temperature-scaled. Fitting on training examples is not calibration.
        """
        try:
            min_samples = int(min_samples)
        except (TypeError, ValueError):
            raise ValueError("min_samples must be a positive integer")
        if min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        groups = {}
        for number, example in enumerate(examples or []):
            if not isinstance(example, dict):
                raise TypeError("calibration example %d must be a mapping" % number)
            questions = example.get("questions") or {}
            truths = example.get("answers", example.get("labels"))
            if not isinstance(truths, dict):
                raise ValueError("calibration example %d needs an answers mapping" % number)
            built, _ = self._prepare_questions(example.get("state", ""), questions)
            for qid, q, qtype, markers, labels, logits, _act in self._raw_questions(built):
                if qid not in truths:
                    raise ValueError("calibration example %d has no truth for question %r"
                                     % (number, qid))
                key = temperature_bucket(qtype, len(markers)) if by_options else qtype
                row = groups.setdefault(key, {"logits": [], "targets": []})
                row["logits"].append(np.asarray(logits, dtype=np.float64))
                row["targets"].append(
                    self._target_index(self.cfg.shape_of(qtype), truths[qid], labels, qtype))
        if not groups:
            raise ValueError("calibrate() needs at least one labelled question")

        report = {"method": "temperature-scaling", "group_by":
                  "question-type-and-option-count" if by_options else "question-type",
                  "groups": {}, "skipped": {}}
        fitted = {}
        for key, rows in groups.items():
            if len(rows["targets"]) < min_samples:
                report["skipped"][key] = {
                    "samples": len(rows["targets"]), "minimum": min_samples}
                continue
            result = fit_temperature(rows["logits"], rows["targets"])
            fitted[key] = result["temperature"]
            report["groups"][key] = result
        if not fitted:
            raise ValueError("no calibration group reached min_samples=%d; counts: %s"
                             % (min_samples, {k: len(v["targets"]) for k, v in groups.items()}))

        for key, value in fitted.items():
            if by_options:
                self.cfg.temperature_by_options[key] = value
            else:
                self.cfg.temperature[self.cfg.qtypes.index(key)] = value
            self.cfg.domain_calibrated.add(key)
        return report

    __call__ = decide

    # ---- what a caller (or a UI) can find out without being told ---------------------
    def surface(self):
        """What this model takes and returns, in the terms a caller works in.

        Here so that an application does not have to know which model it loaded to put a
        sensible interface in front of it: everything below is read from the model's own
        files, so a different decision model with different question types describes itself
        differently and the same interface still fits.
        """
        # Described by SHAPE, in the words someone deciding what to ask would use. The type
        # NAME is whatever the checkpoint calls it -- it is what the request has to say, and
        # for the checkpoints that never declared one it is a convention this engine supplied
        # ("noul" means nothing to anyone reading a form). What an interface actually needs is
        # what the type asks for and what it answers with, and that is the shape. Keyed that
        # way, a model with types nobody has seen describes itself correctly with no entry
        # here, which a table keyed by name could never do.
        described = {
            "named":   {"label": "Pick one", "min": 2, "needs": "options", "options": "named",
                        "help": "Name the things it may choose between. It answers with one "
                                "of them and says how likely each was.",
                        "asks_for": "the options to choose between",
                        "answer": "one of the options, with a probability for each"},
            "ordered": {"label": "Rate on a scale", "min": 2, "needs": "levels",
                        "options": "ordered",
                        "help": "Describe the levels in order, lowest first. It answers with "
                                "where on that scale this lands.",
                        "asks_for": "the levels of the scale, in order",
                        "answer": "where it lands on the scale"},
            "fixed":   {"label": "How likely is this true?", "min": 2, "needs": None,
                        "options": "fixed",
                        "help": "Write a statement. It answers with how likely that statement "
                                "is to hold, from 0 to 100%.",
                        "asks_for": "nothing -- just the statement",
                        "answer": "how likely the statement is to hold, in [0, 1]"},
        }
        types = {}
        for t in self.cfg.qtypes:
            shape = self.cfg.shape_of(t)
            types[t] = dict(described.get(shape) or described[GENERAL_SHAPE], shape=shape)
        return {
            "kind": "decision",
            "takes": {"state": {"kinds": ["text", "json"]},
                      "questions": {"types": types,
                                    "max": self.cfg.max_questions or None}},
            "returns": {"per_question": ["probabilities", "answer_confidence", "confidence",
                                           "act_probability"]},
            "question_layout": self.cfg.question_layout,
            "question_layout_source": self.cfg.question_layout_source,
            "calibration": self.cfg.calibration(),
            "limits": {"sequence_tokens": self.cfg.max_len,
                       "question_tokens": self.cfg.head_max_len,
                       "option_tokens": self.cfg.option_tokens},
        }


# ---- loading ---------------------------------------------------------------------------
# Config file names a decision model may carry. There is no way to list a served directory
# over HTTP, so the alternatives are probed. Which one a model uses says nothing about the
# model -- it is a naming habit -- so several are accepted rather than one being required.
#
# A list of naming habits grows, though: it was one name, then two, and the third checkpoint
# to come along called its file something else again. So `_index` is tried first. Some
# checkpoints ship a `config.json` that is not a config at all but a statement of where
# everything is, and a model that says where its own files are does not need to be guessed
# at. The probes stay for the ones that say nothing.
# Where a decision config sits when the checkpoint does not say. Only conventional names
# belong here: a checkpoint that uses its own name declares it in its index (`*_config_file`),
# which `_named_first` tries FIRST, so adding a published model's filename here buys nothing
# and starts a list that grows by one entry per model. One was here and did exactly nothing --
# the model it was added for declares the same name in its own `config.json`.
_DECISION_CONFIGS = ("decision_config.json", "rl_agent_config.json")
_ENCODER_DIRS = ("encoder/", "")
_TOKENIZER_DIRS = ("tokenizer/", "")


def _named_first(named, fallbacks):
    """What the checkpoint named, then the conventional places, without repeats."""
    out = []
    for x in ([named] if isinstance(named, str) else list(named or [])) + list(fallbacks):
        # An empty directory names the model root.  Tokenizers commonly live there, so it
        # is a real fallback rather than a missing value.  `_index` has already discarded
        # empty paths supplied by a checkpoint; only our explicit fallback can reach here.
        if x is not None and x not in out:
            out.append(x)
    return out


async def _index(src, read_json):
    """What a checkpoint says about its own layout, or {} if it says nothing.

    An index is a `config.json` whose values are paths. It is NOT the encoder config, which
    also lives at that name in some layouts -- told apart by what it contains: an encoder
    config describes an architecture (`hidden_size`, `architectures`), an index describes
    files. Anything ambiguous is treated as not an index, because guessing wrong here means
    reading an architecture as a set of filenames.
    """
    try:
        c = await read_json(src + "/config.json")
    except Exception:
        return {}
    if not isinstance(c, dict) or "hidden_size" in c or "architectures" in c:
        return {}
    out = {}
    for key, value in c.items():
        if not isinstance(value, str) or not value:
            continue
        if key.endswith("_directory"):
            out.setdefault("dirs", {})[key[:-len("_directory")]] = value.rstrip("/") + "/"
        elif key.endswith("_file"):
            out.setdefault("files", {})[key[:-len("_file")]] = value
    return out


async def _rng(path, start, end):
    """Bytes [start, end] inclusive -- the range convention the rest of the SDK reads in."""
    from . import webio
    return await webio.io_read(path, start, end - start + 1)


async def _safetensors_header(path):
    n = int.from_bytes(await _rng(path, 0, 7), "little")
    h = json.loads((await _rng(path, 8, 8 + n - 1)).decode())
    h.pop("__metadata__", None)
    return h, 8 + n


def looks_like_decision(names):
    """Is this checkpoint a decision model? Answered from the tensors it carries.

    An encoder plus something that reduces a position to one number is what makes a decision
    model, and both are visible in the file's own index -- which costs one ranged read, not a
    download. Asking the weights rather than the name means a model nobody has heard of is
    recognised on the first try, and a model that merely has a familiar name is not.
    """
    has_encoder = any(k.startswith("encoder.") for k in names)
    scorers = [k for k in names if k.startswith("scorer.") and k.endswith(".weight")]
    return has_encoder and bool(scorers)


async def _gguf_header(path):
    """Read a GGUF header without knowing how large its embedded metadata is.

    A complete model may carry its tokenizer in metadata, so a fixed small header read is
    not sufficient. `read_header` grows its reads only as far as the header turns out to
    go, and decodes no metadata array until something reads it.
    """
    from . import ggufload as G

    try:
        return await G.read_header(lambda a, b: _rng(path, a, b))
    except ValueError as e:
        if "header limit" in str(e):
            raise ValueError("GGUF metadata exceeds the 128 MiB SDK header limit")
        raise


def _gguf_meta_json(meta, key):
    """JSON metadata in the namespace declared by the file's own architecture.

    The namespace is data, not a model allow-list: a previously unseen architecture name
    works when it publishes the same decision checkpoint contract.
    """
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return None
    raw = meta.get(arch + "." + key)
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except Exception as e:
        raise ValueError("GGUF %s.%s is not valid JSON: %s" % (arch, key, e))
    return value


async def _gguf_tensor(path, data_start, info):
    """Decode one GGUF tensor, bounded by a small conversion workspace.

    F16/F32 tensors keep their stored width. Quantized tensors are expanded to fp16 in
    block-aligned bands, so the 256k-token embedding never also exists as one giant fp32
    temporary. Decision execution already packs or uploads each weight lazily and drops this
    host copy after first use.
    """
    from . import ggufload as G

    shape = tuple(int(d) for d in reversed(info["dims"]))
    count = int(np.prod(shape, dtype=np.int64))
    ttype = info["type"]
    kind = G.GGML_NAMES.get(ttype)
    if not G.is_supported(ttype):
        raise NotImplementedError("decision GGUF tensor %s uses unsupported type %s"
                                  % (info["name"], kind or ttype))
    offset = data_start + int(info["offset"])
    nbytes = G.tensor_nbytes(ttype, count)
    if kind in ("F16", "F32"):
        raw = bytes(await _rng(path, offset, offset + nbytes - 1))
        dtype = np.float16 if kind == "F16" else np.float32
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)

    block, block_bytes = G.tensor_block(ttype)
    if count % block:
        raise ValueError("decision GGUF tensor %s has %d elements, not a whole %s block"
                         % (info["name"], count, kind))
    out = np.empty((count,), dtype=np.float16)
    # At most 8 MiB of expanded fp16 plus the smaller encoded range and fp32 decode scratch.
    band = max(block, ((4 << 20) // block) * block)
    for first in range(0, count, band):
        last = min(count, first + band)
        byte_first = (first // block) * block_bytes
        byte_last = (last // block) * block_bytes
        raw = await _rng(path, offset + byte_first, offset + byte_last - 1)
        out[first:last] = G.dequant(ttype, raw, last - first).astype(np.float16)
    return out.reshape(shape)


async def _gguf_weight(path, data_start, info):
    """Read one GGUF tensor, preserving encoded blocks only for quantized Linear.

    GGUF F16/F32 are already dense values, not quantized blocks. Sending those
    matrices through the generic ggml byte decoder bypasses the backend's dense
    half/f32 matmul and gives identical values a different execution path from
    safetensors. Use the same dense route regardless of which container held them.
    """
    from . import ggufload as G

    shape = tuple(int(d) for d in reversed(info["dims"]))
    kind = G.GGML_NAMES.get(info["type"])
    if (len(shape) == 2 and kind not in ("F16", "F32")
            and wt.ggml_native_supported(kind)
            and shape[1] % wt._GGML_TYPES[kind][2] == 0):
        count = int(np.prod(shape, dtype=np.int64))
        offset = data_start + int(info["offset"])
        nbytes = G.tensor_nbytes(info["type"], count)
        raw = await _rng(path, offset, offset + nbytes - 1)
        return wt.GGMLWeight(raw, kind, shape, type_id=info["type"])
    return await _gguf_tensor(path, data_start, info)


def _tokenizer_from_json(tj):
    from .llm import BPETokenizer, pretok_pattern, pretok_style

    mdl = tj.get("model") or {}
    vocab = dict(mdl.get("vocab") or {})
    control = []
    for a in (tj.get("added_tokens") or []):
        if isinstance(a, dict) and a.get("content") is not None and a.get("id") is not None:
            vocab[a["content"]] = int(a["id"]); control.append(a["content"])
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m
              for m in (mdl.get("merges") or [])]
    tok = BPETokenizer(vocab, merges, control=control, pattern=pretok_pattern(tj),
                       style=pretok_style(tj))
    return tok, vocab


def _warm_decision(model, dec_cfg, webio):
    webio.load_stage("warm")
    model.decide("ready", {"_warm": {"type": dec_cfg.qtypes[0],
                                       "instructions": "warm up",
                                       "criteria": ["a", "b"]}})


async def _from_gguf(src, **kw):
    """Open a self-contained GGUF as a decision model, or return None for another task.

    Not an entry point. `load_decision` picks it once it knows the container; what is a
    decision model and what the bytes are packed in are separate questions, and a caller
    should have to answer neither.

    Recognition is structural, using the tensor names already used for safetensors
    detection. Repository names, filenames, and architecture strings are never allow-listed.
    The file's architecture value only selects its own metadata namespace.
    """
    from . import ggufload as G
    from . import webio

    src = str(src).rstrip("/")
    _version, meta, infos, data_start = await _gguf_header(src)
    by_name = {item["name"]: item for item in infos}
    if not looks_like_decision(by_name):
        return None

    enc_raw = _gguf_meta_json(meta, "encoder_config")
    dec_raw = _gguf_meta_json(meta, "agent_config")
    tj = _gguf_meta_json(meta, "tokenizer_json")
    missing = [name for name, value in (("encoder_config", enc_raw),
                                        ("agent_config", dec_raw),
                                        ("tokenizer_json", tj)) if not isinstance(value, dict)]
    if missing:
        arch = meta.get("general.architecture", "<missing>")
        raise ValueError("decision GGUF architecture %r is missing embedded metadata: %s"
                         % (arch, ", ".join(missing)))

    unsupported = sorted({G.GGML_NAMES.get(item["type"], str(item["type"]))
                          for item in infos if not G.is_supported(item["type"])})
    if unsupported:
        raise NotImplementedError("decision GGUF uses unsupported tensor type(s): %s"
                                  % ", ".join(unsupported))

    weights = {}
    for done, name in enumerate(sorted(by_name), 1):
        weights[name] = await _gguf_weight(src, data_start, by_name[name])
        webio.load_stage("weights", done, len(by_name))

    enc_cfg = EncoderConfig(enc_raw)
    tok, vocab = _tokenizer_from_json(tj)
    mask_id = next((vocab[t] for t in ("[MASK]", "<mask>", "<MASK>", "[mask]")
                    if t in vocab), None)
    if mask_id is None:
        raise ValueError("%s: no marker token in the embedded GGUF tokenizer" % src)
    if "temperature" not in dec_raw and "temperature" in weights:
        dec_raw = dict(dec_raw)
        dec_raw["temperature"] = [float(x) for x in
                                  np.asarray(weights["temperature"]).reshape(-1)]
    dec_cfg = DecisionConfig(dec_raw, type_count=type_rows(weights), weight_names=weights)
    model = DecisionModel(enc_cfg, dec_cfg, weights, tok,
                          mask_id=mask_id, cls_id=enc_cfg.cls_id, sep_id=enc_cfg.sep_id,
                          pad_id=enc_cfg.pad_id)
    _warm_decision(model, dec_cfg, webio)
    return model


async def load_decision(src, container=None, **kw):
    """**Build a decision model from `src`, or return None if it is not one.**

    The one entry point, whatever the weights are packed in. A self-contained GGUF file and
    a served directory of safetensors are the same request -- "is there a decision model
    here, and if so give it to me" -- and which container answers it is this function's
    business, not its caller's. There is deliberately no `load_decision_gguf`: a second,
    format-named entry point would make every caller carry a fact about file layout in order
    to ask a question about models, and would have to be matched by a third the next time
    something is published in a different wrapper.

    What a model IS is still never read off a name. The container is (a format is a naming
    convention), but the answer to "decision model?" comes from the tensors the checkpoint
    carries -- see `looks_like_decision` -- so one nobody has heard of is recognised on the
    first try and one with a familiar name is not.

    `container` is `webio.container_of(src)` when the caller already computed it, so a load
    that has to know the format anyway does not work it out twice.

    Returning None rather than raising is deliberate: this sits in front of the general
    loader, and "not a decision model" is the ordinary case, not a failure.
    """
    from . import webio

    src = str(src).rstrip("/")
    kind = webio.container_of(src) if container is None else container
    if kind == "gguf":
        return await _from_gguf(src, **kw)
    if kind:
        # A lone weights file in some other container: the config and tokenizer a decision
        # model needs are not in it and there is nowhere to look for them.
        return None
    return await _from_directory(src, **kw)


async def _from_directory(src, **kw):
    """Open a served directory as a decision model, or return None for another task."""
    from . import webio
    from . import hfcompat

    src = str(src).rstrip("/")
    said = await _index(src, webio.read_json)
    files, dirs = said.get("files", {}), said.get("dirs", {})

    path = src + "/" + files.get("weights", "model.safetensors")
    try:
        head, base = await _safetensors_header(path)
    except Exception:
        return None
    if not looks_like_decision(head):
        return None

    # Where the checkpoint says its encoder config is, then the usual places.
    enc_cfg = None
    for cand in _named_first(files.get("encoder_config"), ["%sconfig.json" % d for d in _ENCODER_DIRS]):
        try:
            enc_cfg = EncoderConfig(await webio.read_json("%s/%s" % (src, cand)))
            break
        except Exception:
            continue
    if enc_cfg is None:
        return None

    # The decision config is whatever `_config_file` the checkpoint named that is not the
    # encoder's; failing that, the naming habits.
    named = [v for k, v in files.items() if k.endswith("config") and k != "encoder_config"]
    dec_raw = {}
    for cand in _named_first(named, _DECISION_CONFIGS):
        try:
            dec_raw = await webio.read_json("%s/%s" % (src, cand))
            break
        except Exception:
            continue

    tok_dirs = _named_first(dirs.get("tokenizer"), _TOKENIZER_DIRS)
    tj = None
    for d in tok_dirs:
        try:
            tj = await webio.read_json("%s/%stokenizer.json" % (src, d))
            break
        except Exception:
            continue
    if tj is None:
        raise ValueError("%s has a decision checkpoint but no tokenizer.json beside it" % src)
    # Which family of BPE this is comes from the file. A multilingual checkpoint trained
    # with SentencePiece cuts text up by a marker character, not by a regex, and reading one
    # as the other returns different tokens without failing.
    tok, vocab = _tokenizer_from_json(tj)

    # The marker token: whichever of the model's own added tokens marks a position to be
    # filled in. Read from the vocabulary rather than assumed, because the string differs
    # between tokenizers even when the role does not.
    mask_id = next((vocab[t] for t in ("[MASK]", "<mask>", "<MASK>", "[mask]") if t in vocab), None)
    if mask_id is None:
        raise ValueError("%s: no marker token in the vocabulary, so options cannot be scored" % src)

    # Read tensor by tensor. The served readers already fetch ahead and cache whole files
    # (see `webio.prefetch_whole_file`, which they wrap themselves), so these ranges come out
    # of the cache rather than off the wire.
    weights = {}
    done = 0
    for name in sorted(head):
        info = head[name]
        a, z = info["data_offsets"]
        raw = await _rng(path, base + a, base + z - 1)
        weights[name] = hfcompat._decode(bytes(raw), info["dtype"], info["shape"])
        done += 1
        webio.load_stage("weights", done, len(head))

    # Calibration a checkpoint ships in its WEIGHTS rather than its config. Both are ordinary
    # places to keep it -- it is fitted after training, so it lands wherever that script put
    # it -- and a temperature that is not read is not an error anyone sees: the ranking is
    # unchanged and only the confidence beside it is wrong, which is the worst way for a
    # number to be wrong. The config still wins where it says something.
    if "temperature" not in dec_raw and "temperature" in weights:
        dec_raw = dict(dec_raw or {})
        dec_raw["temperature"] = [float(x) for x in
                                  np.asarray(weights["temperature"]).reshape(-1)]

    dec_cfg = DecisionConfig(dec_raw, type_count=type_rows(weights), weight_names=weights)
    model = DecisionModel(enc_cfg, dec_cfg, weights, tok,
                          mask_id=mask_id, cls_id=enc_cfg.cls_id, sep_id=enc_cfg.sep_id,
                          pad_id=enc_cfg.pad_id)
    # Ask it something trivial before handing it over.
    #
    # Weights reach the device on first use, and shaders compile on first dispatch, so
    # without this the FIRST question a reader asks pays for the whole model being uploaded
    # -- measured at 3.5 s against 51 ms for the ones after it. It also makes the loaded
    # model honest about itself: between loading and that first question the page reported
    # nothing on the GPU, because nothing was.
    _warm_decision(model, dec_cfg, webio)
    return model
