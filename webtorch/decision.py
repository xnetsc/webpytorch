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


class DecisionConfig(object):
    """The decision half of the model's own config: how long a sequence may be, how the
    answer distribution is scaled, and which question types exist."""

    def __init__(self, cfg, qtypes=None):
        c = dict(cfg or {})
        self.raw = c
        self.max_len = int(c.get("max_len", 512))
        self.head_max_len = int(c.get("head_max_len", 192))
        self.head_layers = int(c.get("head_layers", 0))
        self.max_questions = int(c.get("max_prefixes", 0) or 0)
        self.option_tokens = int(c.get("option_tokens", 48))
        # Post-hoc calibration, fitted by whoever trained the model. Two levels: a
        # temperature per question type, and a finer one per type AND option count, because
        # a two-way question and a twenty-way one do not need the same scaling.
        self.qtypes, self.shapes = self._read_types(c, qtypes)
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
    def _read_types(c, qtypes=None):
        """The question types this checkpoint has, in the order its type embedding is in.

        Declared, if it says so: `question_types` as a list -- order is meaningful, it is
        the index into the type embedding and the temperatures -- of names, or of
        `{"name", "shape"}`. That is how a model says what it answers, and a model that
        says it needs nothing added here to be supported.

        Otherwise the count is still knowable, from however many temperatures the file
        carries, and only the names and shapes are the convention above. Anything past the
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
        count = len(t) if isinstance(t, (list, tuple)) and t else len(_CONVENTIONAL)
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
    def build_sequence(self, state, qtype, instructions, criteria):
        """`[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`

        The budget matters as much as the order. Options are written first and in full, and
        the state gets what is left, because an option that did not fit has no marker and
        therefore no answer -- while a state that was cut short still answers, just with less
        to go on.
        """
        mask_str = self.tok.dec.get(self.mask_id, "")

        def clean(s):
            return str(s).replace(mask_str, " ") if mask_str else str(s)

        labels, opts = render_options(self.cfg.shape_of(qtype), criteria)
        head_ids = self.tok.encode("%s question: %s" % (qtype, clean(instructions)))
        opt_ids = [[self.mask_id] + self.tok.encode(" " + clean(o))[:self.cfg.option_tokens]
                   for o in opts]
        budget = self.cfg.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:                       # too many, or too long: shorten all of them evenly
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
        ids = ids + self.tok.encode(clean(serialize_state(state)))[:room] + [self.sep_id]
        return ids[:self.cfg.max_len], [m for m in markers if m < self.cfg.max_len], labels

    # ---- the head --------------------------------------------------------------------
    def _head_layer(self, x, i):
        """One pre-norm transformer block, in the layout the checkpoint was saved in.

        The activation is ReLU, not GELU: this head is a stock `TransformerEncoderLayer` and
        that is its default. It is the kind of detail that cannot be read off the weights --
        an activation has no parameters -- so it is read off the reference implementation.
        """
        p = "head.layers.%d." % i
        h, hd, T = self.heads, self.head_dim, x.shape[0]
        a = self._ln(x, p + "norm1")
        qkv = self._lin(a, p + "self_attn.in_proj")
        d = h * hd
        q = wt._slice_last(qkv, 0, d).reshape(T, h, hd).permute(1, 0, 2)
        k = wt._slice_last(qkv, d, 2 * d).reshape(T, h, hd).permute(1, 0, 2)
        v = wt._slice_last(qkv, 2 * d, 3 * d).reshape(T, h, hd).permute(1, 0, 2)
        o = bmm(softmax(bmm(q, transpose_last2(k)) * (1.0 / (hd ** 0.5))), v)
        o = o.permute(1, 0, 2).reshape(T, d)
        x = x + self._lin(o, p + "self_attn.out_proj")
        f = self._ln(x, p + "norm2")
        f = self._lin(wt.ReLU()(self._lin(f, p + "linear1")), p + "linear2")
        return x + f

    # Where one pass over several questions beats one pass each.
    #
    # Measured, four matmuls per layer over 28 layers, separate against batched:
    #
    #     tokens each     x2      x3      x6
    #        32          1.63    2.00    2.28
    #        48          1.49    1.56    1.71
    #        64          1.36    1.32    1.47
    #        96          1.10    1.24    1.24
    #       128          1.05    1.09    1.13
    #       160+         1.04    1.04    1.04
    #
    # It is the same occupancy story as everywhere else here: a short sequence does not give
    # the device enough rows to work on, and putting several together does. By 128 there are
    # already enough and batching is noise when the encoder output crosses the host boundary.
    # Decision heads now consume the padded result on-device; measured at 115--138 tokens,
    # four questions fell from 370.2 ms to 347.5 ms without changing an answer. Keep the
    # boundary at 160: above it padding and the larger attention squares erase that gain.
    _BATCH_MAX_TOKENS = 160

    @classmethod
    def batch_pays(cls, longest, count):
        return count >= 2 and longest <= cls._BATCH_MAX_TOKENS

    def _run_one(self, ids, markers, qtype_idx):
        return self._score(self.enc.encode(ids), markers, qtype_idx)

    def _score(self, h, markers, qtype_idx):
        h = h + wt.embedding(self._t("type_emb.weight"),
                             np.full((h.shape[0],), qtype_idx, dtype=np.int64))
        for i in range(self.n_head_layers):
            h = self._head_layer(h, i)
        # Pick out the rows that matter -- one per option marker, plus position 0 for the
        # action head -- as a matmul with a selector, not by indexing. Two reasons. Indexing
        # a GPU array does not come back as numpy, so a slice of it is a GPU slice and the
        # backend read `[1:]` as the index 1 (it raised "index 3 out of range for shape
        # (3, 1024)"), which is a whole class of bug this sidesteps. And a selector keeps the
        # work where the weights already are: k is a handful of rows against (T, D).
        #
        # Two selectors rather than one and a slice, for the same reason.
        T = h.shape[0]
        pm = np.zeros((len(markers), T), dtype=np.float32)
        for r, mpos in enumerate(markers):
            pm[r, mpos] = 1.0
        m = Tensor(pm).matmul(h)                             # (k, D)
        pp = np.zeros((1, T), dtype=np.float32)
        pp[0, 0] = 1.0
        pooled_t = Tensor(pp).matmul(h)                      # (1, D)
        s = self._ln(m, "scorer.0")
        s = self._lin(gelu(self._lin(s, "scorer.1")), "scorer.3")
        logits = s.numpy().reshape(-1)
        # the action head: the model's own read on whether it should answer at all
        p = np.exp(logits - logits.max()); p = p / p.sum()
        k = max(2, len(markers))
        ent = float(-(p * np.log(np.clip(p, 1e-9, 1))).sum() / math.log(k))
        top2 = np.sort(p)[::-1][:2]
        feats = np.array([top2[0], top2[0] - (top2[1] if len(top2) > 1 else 0.0),
                          ent, k / 255.0], dtype=np.float32)
        pooled = pooled_t.numpy().reshape(-1)
        a = Tensor(np.concatenate([pooled, feats])[None, :])
        a = self._lin(gelu(self._lin(a, "act_head.0")), "act_head.2")
        act = a.numpy().reshape(-1)
        act = np.exp(act - act.max()); act = act / act.sum()
        return logits, float(act[0])

    # ---- the API ---------------------------------------------------------------------
    def _prepare_questions(self, state, questions):
        total = 0
        # Every sequence is built first, because whether to run them together depends on how
        # long the longest one turned out to be.
        built = []
        for qid, q in (questions or {}).items():
            qtype = q["type"]
            if qtype not in self.cfg.qtypes:
                raise ValueError("%r is not a question type this model answers (%s)"
                                 % (qtype, ", ".join(self.cfg.qtypes)))
            ins = q.get("instructions")
            ins = ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)
            ids, markers, labels = self.build_sequence(state, qtype, ins, q.get("criteria"))
            if len(markers) != len(labels):
                raise ValueError("question %r: its options need more than %d tokens, so %d of "
                                 "them have no place in the sequence to be scored at"
                                 % (qid, self.cfg.head_max_len, len(labels) - len(markers)))
            total += len(ids)
            built.append((qid, q, qtype, ids, markers, labels))
        return built, total

    def _raw_questions(self, built, execution=None):
        hs = None
        device_batch = None
        if built and self.batch_pays(max(len(b[3]) for b in built), len(built)):
            try:
                device_batch = self.enc._encode_many_device([b[3] for b in built])
            except Exception:
                try:
                    hs = self.enc.encode_many([b[3] for b in built])
                except Exception:
                    hs = None        # a batched pass is an optimisation, not a step

        encoded = {}
        encoder_tokens = 0
        encoder_passes = 0
        scored = []
        for idx, (qid, q, qtype, ids, markers, labels) in enumerate(built):
            if device_batch is not None:
                batch_h, lengths, padded = device_batch
                # Select this sequence's valid rows while they are still on the device.
                # A tiny one-hot matmul is cheaper than reading B*L*D values to the host and
                # uploading each question again for its decision head.
                selector = np.zeros((lengths[idx], len(lengths) * padded), dtype=np.float32)
                rows = idx * padded + np.arange(lengths[idx])
                selector[np.arange(lengths[idx]), rows] = 1.0
                h = Tensor(selector).matmul(batch_h)
                logits, act = self._score(h, markers, self.cfg.qtypes.index(qtype))
            elif hs is not None:
                logits, act = self._score(hs[idx], markers, self.cfg.qtypes.index(qtype))
            else:
                # Questions can collapse to the exact same encoder input (for example two
                # identical boolean statements under different caller ids). Encoding that
                # sequence twice cannot change the answer, so share it within this call.
                key = tuple(ids)
                if key not in encoded:
                    encoded[key] = self.enc.encode(ids)
                    encoder_tokens += len(ids)
                    encoder_passes += 1
                logits, act = self._score(encoded[key], markers, self.cfg.qtypes.index(qtype))
            scored.append((qid, q, qtype, markers, labels, logits, act))
        if device_batch is not None or hs is not None:
            encoder_passes = 1 if built else 0
            encoder_tokens = len(built) * max(len(b[3]) for b in built) if built else 0
        if execution is not None:
            execution.update({"encoder_tokens": encoder_tokens,
                              "encoder_passes": encoder_passes,
                              "batched": device_batch is not None or hs is not None})
        return scored

    def decide(self, state, questions):
        """Answer every question about this state.

        `questions` is `{id: {"type", "instructions", "criteria"}}`, and the answers come back
        under the same ids. Each carries its whole distribution, not only the winner, because
        a caller that wants to act on "0.51 versus 0.49" has to be able to see it.
        """
        out = {}
        built, total = self._prepare_questions(state, questions)
        execution = {}
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
        usage = {"input_tokens": total, "output_tokens": 0,
                 "questions": len(built),
                 "sequence_tokens": {str(b[0]): len(b[3]) for b in built}}
        usage.update(execution)
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
    not sufficient.  Grow only until the parser says it has the complete header; ordinary
    LLM GGUFs stop on the first read, while self-contained artifacts may need more.
    """
    from . import ggufload as G

    size = 12 << 20
    while True:
        buf = await _rng(path, 0, size - 1)
        try:
            return G.parse_header(buf)
        except EOFError:
            size <<= 1
            if size > (128 << 20):
                raise ValueError("GGUF metadata exceeds the 128 MiB SDK header limit")


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
    try:
        webio.load_stage("warm")
        model.decide("ready", {"_warm": {"type": dec_cfg.qtypes[0],
                                           "instructions": "warm up",
                                           "criteria": ["a", "b"]}})
    except Exception as e:                    # a warm-up is an optimisation, not a step
        try:
            import js
            js.console.warn("webtorch: decision warm-up skipped: " + str(e))
        except Exception:
            pass                              # no browser to tell; the model is still fine


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
        weights[name] = await _gguf_tensor(src, data_start, by_name[name])
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
    dec_cfg = DecisionConfig(dec_raw)
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

    dec_cfg = DecisionConfig(dec_raw)
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
