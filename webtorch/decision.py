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


def render_options(qtype, criteria):
    """The answer texts, in label order.

    Each type has one shape, and the shapes are not interchangeable: `noul` is always
    [false, true] in that order, which is what makes the second probability the answer.
    """
    if qtype == "choice":
        crit = {c: None for c in criteria} if isinstance(criteria, list) else dict(criteria or {})
        return list(crit.keys()), [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if qtype == "score":
        crit = list(criteria or [])
        return [str(i) for i in range(len(crit))], ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = criteria or {}
    return ["false", "true"], [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds")]


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
        self.qtypes = list(qtypes or ["choice", "score", "noul"])
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

        labels, opts = render_options(qtype, criteria)
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
    # already enough and batching is noise, so the line is drawn where the gain stops being
    # worth the padding -- sequences are padded to the longest, and a batch of uneven ones
    # does arithmetic on padding that a separate pass would not.
    _BATCH_MAX_TOKENS = 96

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

    def _raw_questions(self, built):
        hs = None
        if built and self.batch_pays(max(len(b[3]) for b in built), len(built)):
            try:
                hs = self.enc.encode_many([b[3] for b in built])
            except Exception:
                hs = None            # a batched pass is an optimisation, not a step

        scored = []
        for idx, (qid, q, qtype, ids, markers, labels) in enumerate(built):
            if hs is not None:
                logits, act = self._score(hs[idx], markers, self.cfg.qtypes.index(qtype))
            else:
                logits, act = self._run_one(ids, markers, self.cfg.qtypes.index(qtype))
            scored.append((qid, q, qtype, markers, labels, logits, act))
        return scored

    def decide(self, state, questions):
        """Answer every question about this state.

        `questions` is `{id: {"type", "instructions", "criteria"}}`, and the answers come back
        under the same ids. Each carries its whole distribution, not only the winner, because
        a caller that wants to act on "0.51 versus 0.49" has to be able to see it.
        """
        out = {}
        built, total = self._prepare_questions(state, questions)
        for qid, q, qtype, markers, labels, logits, act in self._raw_questions(built):
            z = logits / self.cfg.temp_for(qtype, len(markers))
            p = np.exp(z - z.max()); p = p / p.sum()
            ans = {"type": qtype, "probabilities": {l: round(float(v), 4) for l, v in zip(labels, p)},
                   "confidence": round(confidence(p), 4), "act_probability": round(act, 4)}
            if qtype == "choice":
                ans["choice"] = labels[int(p.argmax())]
            elif qtype == "score":
                ans["score"] = round(float((np.arange(len(p)) * p).sum()), 4)
                ans["legend"] = {str(i): c for i, c in enumerate(q.get("criteria") or [])}
            else:
                ans["noul"] = round(float(p[1]), 4)
            out[qid] = ans
        return {"answers": out, "usage": {"input_tokens": total, "output_tokens": 0}}

    @staticmethod
    def _target_index(qtype, truth, labels):
        if isinstance(truth, dict):
            truth = truth.get(qtype, truth.get("target", truth.get("label")))
        if qtype == "choice":
            if truth in labels:
                return labels.index(truth)
        elif qtype == "noul":
            if isinstance(truth, str):
                v = truth.strip().lower()
                if v in ("true", "yes", "1"): return 1
                if v in ("false", "no", "0"): return 0
            if isinstance(truth, (bool, int, np.integer)) and int(truth) in (0, 1):
                return int(truth)
        else:
            if isinstance(truth, str) and truth in labels:
                return labels.index(truth)
            if isinstance(truth, (int, np.integer)) and 0 <= int(truth) < len(labels):
                return int(truth)
        raise ValueError("label %r is not one of the answers for this %s question (%s)"
                         % (truth, qtype, ", ".join(labels)))

    def calibrate(self, examples, by_options=True, min_samples=20):
        """Fit probability temperatures on separate labelled held-out examples.

        Each item is `{"state": ..., "questions": {...}, "answers": {id: truth}}`. Choice
        truths are option names, score truths are level indices, and noul truths are booleans.
        With `by_options=True` (the default), the same option-count buckets used at inference
        are fitted independently; otherwise one value is fitted per question type.

        The model's chosen class cannot change: positive temperature scaling only repairs
        the numeric probability scale. Fitting on training examples is not calibration.
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
                row["targets"].append(self._target_index(qtype, truths[qid], labels))
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
        # Each type is described in the words someone deciding what to ask would use, not in
        # the model's. `choice`, `score` and `noul` are what the model calls them and what
        # the request must say, but nobody outside knows what "noul" is, and an interface
        # that prints it has handed the reader a puzzle. The plain name and the one-line
        # explanation belong here, with the thing that knows what the type does, rather than
        # in a table kept by whichever application happens to draw the form.
        described = {
            "choice": {"label": "Pick one", "min": 2, "needs": "options", "options": "named",
                       "help": "Name the things it may choose between. It answers with one "
                               "of them and says how likely each was.",
                       "asks_for": "the options to choose between",
                       "answer": "one of the options, with a probability for each"},
            "score":  {"label": "Rate on a scale", "min": 2, "needs": "levels",
                       "options": "ordered",
                       "help": "Describe the levels in order, lowest first. It answers with "
                               "where on that scale this lands.",
                       "asks_for": "the levels of the scale, in order",
                       "answer": "where it lands on the scale"},
            "noul":   {"label": "How likely is this true?", "min": 2, "needs": None,
                       "options": "fixed",
                       "help": "Write a statement. It answers with how likely that statement "
                               "is to hold, from 0 to 100%.",
                       "asks_for": "nothing -- just the statement",
                       "answer": "how likely the statement is to hold, in [0, 1]"},
        }
        types = {}
        for t in self.cfg.qtypes:
            types[t] = dict(described.get(t) or {
                "label": t, "min": 2, "needs": None, "options": "fixed",
                "help": "", "asks_for": "", "answer": ""})
        return {
            "kind": "decision",
            "takes": {"state": {"kinds": ["text", "json"]},
                      "questions": {"types": types,
                                    "max": self.cfg.max_questions or None}},
            "returns": {"per_question": ["probabilities", "confidence", "act_probability"]},
            "calibration": self.cfg.calibration(),
            "limits": {"sequence_tokens": self.cfg.max_len,
                       "question_tokens": self.cfg.head_max_len,
                       "option_tokens": self.cfg.option_tokens},
        }


# ---- loading ---------------------------------------------------------------------------
# Config file names a decision model may carry. There is no way to list a served directory
# over HTTP, so the alternatives are probed. Which one a model uses says nothing about the
# model -- it is a naming habit -- so several are accepted rather than one being required.
_DECISION_CONFIGS = ("decision_config.json", "rl_agent_config.json")
_ENCODER_DIRS = ("encoder/", "")
_TOKENIZER_DIRS = ("tokenizer/", "")


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


async def load_decision(src, **kw):
    """Build a decision model from a served directory, or return None if it is not one.

    Returning None rather than raising is deliberate: this sits in front of the general
    loader, and "not a decision model" is the ordinary case, not a failure.
    """
    from . import webio
    from .llm import BPETokenizer, pretok_pattern, pretok_style
    from . import hfcompat

    src = str(src).rstrip("/")
    path = src + "/model.safetensors"
    try:
        head, base = await _safetensors_header(path)
    except Exception:
        return None
    if not looks_like_decision(head):
        return None

    enc_cfg = None
    for d in _ENCODER_DIRS:
        try:
            enc_cfg = EncoderConfig(await webio.read_json("%s/%sconfig.json" % (src, d)))
            break
        except Exception:
            continue
    if enc_cfg is None:
        return None

    dec_raw = {}
    for name in _DECISION_CONFIGS:
        try:
            dec_raw = await webio.read_json("%s/%s" % (src, name))
            break
        except Exception:
            continue

    tj = None
    for d in _TOKENIZER_DIRS:
        try:
            tj = await webio.read_json("%s/%stokenizer.json" % (src, d))
            break
        except Exception:
            continue
    if tj is None:
        raise ValueError("%s has a decision checkpoint but no tokenizer.json beside it" % src)
    mdl = tj.get("model") or {}
    vocab = dict(mdl.get("vocab") or {})
    control = []
    for a in (tj.get("added_tokens") or []):
        if isinstance(a, dict) and a.get("content") is not None and a.get("id") is not None:
            vocab[a["content"]] = int(a["id"]); control.append(a["content"])
    merges = [" ".join(m) if isinstance(m, (list, tuple)) else m for m in (mdl.get("merges") or [])]
    # Which family of BPE this is comes from the file. A multilingual checkpoint trained
    # with SentencePiece cuts text up by a marker character, not by a regex, and reading one
    # as the other returns different tokens without failing.
    tok = BPETokenizer(vocab, merges, control=control, pattern=pretok_pattern(tj),
                       style=pretok_style(tj))

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
    return model
