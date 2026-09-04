"""Output constraints: restrict what a model may generate next.

Generic by construction. A constraint sees the text produced so far and decides which
continuations remain valid, so it works off decoded text rather than token ids -- the same
constraint drives any tokenizer and any model. `webtorch.generate(..., constraint="json")`
picks a built-in; passing an object with the same three methods uses your own.

The sampler applies a constraint by filtering candidates: it walks tokens in descending
likelihood and takes the first the constraint accepts. Scoring the whole vocabulary would
cost more than the model step itself on a 250k-token vocabulary, and buys nothing -- the
allowed set is what matters, not its exact ordering.
"""
import json


# What `allows` answers, as one type.
#
# A verdict is three independent things, and they are worth keeping independent because a
# caller genuinely needs the combinations:
#
#   allow  may this candidate be used at all
#   take   if it may, does it go into the reply
#   then   what happens from here -- keep ASKING, stop asking and run FREE, or END the reply
#
# Twelve combinations exist; six of them mean something. `allow=True, take=False` with more
# to come cannot be honoured -- every step has to emit a token -- and with nothing to come it
# is just "end without this one", which `END` already is. The six that remain are below, and
# they are the whole space, not a list that grew by need:
#
#     allow  take  then       name
#     no     -     THEN_ASK   DENY          not this one; keep asking
#     no     -     THEN_FREE  DENY_FREE     not this one, and stop asking from here
#     no     -     THEN_END   END           the reply is complete as it stands
#     yes    yes   THEN_ASK   ALLOW         the ordinary answer
#     yes    yes   THEN_FREE  ALLOW_FREE    take it, then stop asking
#     yes    yes   THEN_END   ALLOW_END     take it; it is the last
#
# `True` and `False` are accepted as ALLOW and DENY, because those two are most of the uses
# and a predicate should be allowed to look like one.
# The `then` axis is an int, not a string: it is read a few hundred times per token (once
# per candidate) and an integer compare needs no interning assumption to be cheap. The names
# are what anyone writing or reading a constraint uses; the numbers are what the loop sees.
THEN_ASK = 0
THEN_FREE = 1
THEN_END = 2


class Verdict(object):
    """One answer from a constraint. The six below are the only values -- compare with `is`,
    or read `.allow` / `.take` / `.then`."""

    __slots__ = ("allow", "take", "then", "_name")

    def __init__(self, allow, take, then, name):
        self.allow = allow
        self.take = take
        self.then = then            # THEN_ASK | THEN_FREE | THEN_END
        self._name = name

    def __repr__(self):
        return "Verdict.%s" % self._name

    def __bool__(self):             # so `if verdict:` reads as "was it allowed"
        return bool(self.allow)

    __nonzero__ = __bool__


DENY = Verdict(False, False, THEN_ASK, "DENY")
DENY_FREE = Verdict(False, False, THEN_FREE, "DENY_FREE")
END = Verdict(False, False, THEN_END, "END")
ALLOW = Verdict(True, True, THEN_ASK, "ALLOW")
ALLOW_FREE = Verdict(True, True, THEN_FREE, "ALLOW_FREE")
ALLOW_END = Verdict(True, True, THEN_END, "ALLOW_END")

# The older spellings, kept because they read well in a one-line callback.
_ALIASES = {True: ALLOW, False: DENY, None: DENY,
            "allow": ALLOW, "deny": DENY,
            "last": ALLOW_END, "stop": END, "free": ALLOW_FREE,
            "deny_free": DENY_FREE, "end": END}


def verdict(v):
    """Whatever a callback returned, as a `Verdict`."""
    if isinstance(v, Verdict):
        return v
    try:
        got = _ALIASES.get(v)
    except TypeError:               # unhashable -- certainly not one of ours
        got = None
    if got is not None:
        return got
    raise TypeError(
        "a constraint must answer with a Verdict (or True/False); got %r. The six are "
        "DENY, DENY_FREE, END, ALLOW, ALLOW_FREE, ALLOW_END." % (v,))


class Constraint:
    """Base class. Override `allows`; override `finished` if the output can end early."""

    def reset(self):
        """Called once before each generation."""

    def dormant(self, text):
        """True when this constraint would permit anything at all after `text`.

        Answering yes costs the caller nothing and saves it everything: to ask `allows` or
        `decide` the sampler must first rank the whole vocabulary and decode the candidates,
        and on a 150k vocabulary that is most of what a token costs. Measured on a 27B with
        tool names constrained, a reply ran at 0.8 tok/s against 5.2 with no constraint --
        and in a reply of prose, essentially every one of those questions was asked of a
        constraint that had nothing to say.

        The default is False, so a constraint that does not implement this is asked about
        every token exactly as before. Override it only where the answer can be given from
        the text ALONE and is certainly right: saying "dormant" wrongly does not slow
        anything down, it lets through what should have been refused.
        """
        return False

    def allows(self, text, piece):
        """May `piece` follow `text`?"""
        raise NotImplementedError

    def finished(self, text):
        """True when `text` is a complete output and generation may stop."""
        return False


def _scan_string(s, i):
    """Index just past the JSON string starting at s[i] == '"', or None if unterminated."""
    i += 1
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        if c in "\n\r":                     # control characters are not legal in a JSON string
            return -1
        i += 1
    return None


def _scan_number(s, i):
    """Index just past the JSON number at s[i], or -1 if it cannot be one.

    Returns len(s) when the number runs to the end of the input -- it may still be growing.
    Strict about the grammar, which loose scanning is not: `--1` and `01` are not numbers,
    while `1e` and `-` are valid prefixes of one.
    """
    n = len(s); j = i
    if j < n and s[j] == "-":
        j += 1
    if j >= n:
        return n
    if s[j] == "0":
        j += 1
    elif s[j].isdigit():
        while j < n and s[j].isdigit():
            j += 1
    else:
        return -1
    if j >= n:
        return n
    if s[j] == ".":
        j += 1
        if j >= n:
            return n
        if not s[j].isdigit():
            return -1
        while j < n and s[j].isdigit():
            j += 1
        if j >= n:
            return n
    if j < n and s[j] in "eE":
        j += 1
        if j < n and s[j] in "+-":
            j += 1
        if j >= n:
            return n
        if not s[j].isdigit():
            return -1
        while j < n and s[j].isdigit():
            j += 1
    return j


_LITERALS = ("true", "false", "null")


def json_prefix_ok(s):
    """True when `s` could still become valid JSON once more characters are appended.

    A prefix checker, not a validator: an unterminated string or a half-typed `tru` is fine,
    a `,` where a value belongs is not. That is exactly what constrained decoding needs --
    reject a token only when no continuation could rescue it.
    """
    stack = []                              # '{' / '[' currently open
    state = "value"                         # value | key | colon | comma | value_or_close
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n\r":
            i += 1
            continue
        if state in ("value", "value_or_close"):
            if c == "{":
                stack.append("{"); state = "key_or_close"; i += 1; continue
            if c == "[":
                stack.append("["); state = "value_or_close"; i += 1; continue
            if c == "]" and state == "value_or_close" and stack and stack[-1] == "[":
                stack.pop(); state = "comma"; i += 1; continue
            if c == '"':
                j = _scan_string(s, i)
                if j is None:
                    return True             # still inside a string
                if j < 0:
                    return False
                i = j; state = "comma"; continue
            if c == "-" or c.isdigit():
                j = _scan_number(s, i)
                if j < 0:
                    return False
                if j >= n:
                    return True             # number may still be growing
                i = j; state = "comma"; continue
            for lit in _LITERALS:
                if s.startswith(lit, i):
                    i += len(lit); state = "comma"; break
                if lit.startswith(s[i:]):
                    return True             # a literal still being spelled out
            else:
                return False
            continue
        if state in ("key", "key_or_close"):
            if c == "}" and state == "key_or_close" and stack and stack[-1] == "{":
                stack.pop(); state = "comma"; i += 1; continue
            if c != '"':
                return False
            j = _scan_string(s, i)
            if j is None:
                return True
            if j < 0:
                return False
            i = j; state = "colon"; continue
        if state == "colon":
            if c != ":":
                return False
            i += 1; state = "value"; continue
        if state == "comma":
            if c == ",":
                if not stack:
                    return False
                state = "key" if stack[-1] == "{" else "value"
                i += 1; continue
            if c == "}" and stack and stack[-1] == "{":
                stack.pop(); i += 1; continue
            if c == "]" and stack and stack[-1] == "[":
                stack.pop(); i += 1; continue
            return False
    return True


class JsonConstraint(Constraint):
    """Emit only well-formed JSON."""

    def allows(self, text, piece):
        return json_prefix_ok(text + piece)

    def finished(self, text):
        t = text.strip()
        if not t:
            return False
        try:
            json.loads(t)
            return True
        except Exception:
            return False


class RegexConstraint(Constraint):
    """Emit only text that matches a regular expression.

    Needs the `regex` package, for its PARTIAL matching: deciding what may come next means
    asking "can this prefix still grow into a match", and the standard library's `re` cannot
    answer that. It only reports whether something already matches.

    So there is no fallback, and that is deliberate. The old one used `re.match`, which
    rejects every incomplete prefix -- including the correct ones. Every candidate was
    refused, the sampler widened to the whole vocabulary, found nothing, and fell back to an
    unconstrained pick: the constraint silently did NOTHING, and asking for a date pattern
    returned "Sure! What would you like to d". A constraint that quietly does not apply is
    worse than one that refuses to start.
    """

    def __init__(self, pattern):
        try:
            import regex as _re
        except ImportError:
            raise ImportError(
                "a regex constraint needs the `regex` package (the standard library's `re` "
                "cannot test whether a prefix could still become a match, so a constraint "
                "built on it silently allows everything). In a browser: "
                "await micropip.install('regex'). Or use a callback constraint -- "
                "constraint=lambda text, piece: ... -- which needs nothing.")
        self.re = _re
        self.rx = _re.compile(pattern)

    def allows(self, text, piece):
        return self.rx.fullmatch(text + piece, partial=True) is not None

    def finished(self, text):
        return self.rx.fullmatch(text) is not None


class ToolNameConstraint(Constraint):
    """Inside a tool call, the name must be one the caller actually registered.

    A model that has talked itself into the wrong name is CERTAIN of it by the time it
    writes the call -- measured at p=0.9998 on the wrong token -- so nothing that reweights
    the distribution repairs it: not a lower temperature, not top-p. Removing the token from
    the candidate set does, which is what a constraint is for.

    Everything outside a call is left alone; prose is prose. The set of names is whatever
    was passed to this generation, and the delimiter comes from the model's own template
    (see `tool_call_format`), so nothing here is written down in advance.

    `payload` says where the name sits: a JSON call writes `"name": "X"`, an XML one writes
    `<function=X>`. Both are located relative to the opening delimiter, so an opener that
    could not be derived means no constraint rather than a guessed one.
    """

    # Only the tail is scanned. The name follows its opening delimiter within a few dozen
    # characters, and `allows` runs per candidate per token -- scanning the whole reply each
    # time would make the cost of a constraint grow with the length of the thing it guards.
    WINDOW = 400

    def __init__(self, names, open_tag=None, payload=None):
        self.names = sorted({str(n) for n in (names or []) if n})
        self.open = str(open_tag or "")
        self.payload = payload or "flat"

    def _pending(self, s):
        """What has been written of a name so far, and whether it is finished -- or None
        when the text is not inside a name."""
        if not self.names or not self.open:
            return None
        tail = s[-self.WINDOW:]
        i = tail.rfind(self.open)
        if i < 0:
            return None
        rest = tail[i + len(self.open):]
        if self.payload == "xml":
            j = rest.rfind("<function=")
            if j < 0:
                return None
            got = rest[j + len("<function="):]
            end = got.find(">")
        else:
            j = rest.rfind('"name"')
            if j < 0:
                return None
            got = rest[j + len('"name"'):]
            k = got.find(":")
            if k < 0:
                return None
            got = got[k + 1:].lstrip()
            if not got.startswith('"'):
                # the opening quote has not been written yet; nothing to check, but do not
                # let anything other than the quote start the value
                return ("", False) if got == "" else None
            got = got[1:]
            end = got.find('"')
        if end >= 0:
            return (got[:end], True)
        return (got, False)

    def dormant(self, text):
        """Nothing to say unless a call is being opened or a name is being written.

        `allows` judges `text + piece`, so it is not enough that `text` is outside a name --
        the piece could put it inside one. What bounds that is the opener: a name only ever
        follows it, so this stays awake while the opener is present in the window, and also
        while the tail is a partial opener, which is the only way the next piece can complete
        one. A piece that carries the WHOLE opener at once leaves nothing after it for a name
        to be, so `allows` would have permitted that step anyway.
        """
        if not self.names or not self.open:
            return True                      # nothing to constrain against
        tail = text[-self.WINDOW:]
        if self.open in tail:
            return False
        o = self.open
        for i in range(min(len(o) - 1, len(tail)), 0, -1):
            if tail.endswith(o[:i]):
                return False                 # the opener is being written right now
        return True

    def allows(self, text, piece):
        got = self._pending(text + piece)
        if got is None:
            return True
        name, done = got
        if done:
            return name in self.names
        return any(n.startswith(name) for n in self.names)


class StopConstraint(Constraint):
    """Stop at any of the given strings (in addition to the model's end-of-turn token)."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]

    def dormant(self, text):
        return True                          # this one never refuses a piece, it only ends

    def allows(self, text, piece):
        return True

    def finished(self, text):
        return any(s in text for s in self.stops)


# Two ways to say what may come next.
#
# `allows(text, piece)` is asked about ONE candidate at a time, walking the model's ranking
# until something is accepted. Easy to write -- a predicate, no knowledge of the tokenizer --
# and bounded at 64 questions per token while the model's own top choices are acceptable.
# When they are not, the sampler widens, and a constraint that permits only something rare
# can be asked about the whole vocabulary: measured at 153,408 questions for one token.
#
# `decide(text, candidates)` is asked ONCE per token. It is handed everything on offer, best
# first, and answers with a verdict AND the set that is permitted:
#
#     def decide(text, candidates):
#         return webtorch.ALLOW, ["python", "javascript"]   # only these
#         return webtorch.ALLOW, None                       # no opinion; keep the ranking
#         return webtorch.ALLOW, []                         # none of these -- widen and ask again
#         return webtorch.ALLOW_END, ["}"]                   # exactly this, and it is the last
#
# The permitted set is TEXT, and a candidate counts when its text is a prefix of one of them
# or one of them is a prefix of it -- so a name that takes three tokens to spell is reachable
# without the caller knowing how it splits. A permitted string that is nowhere in the
# candidates is looked up in the vocabulary and offered anyway, which is how a caller forces
# a token the model ranked nowhere.
#
# One call instead of sixty-four, and no widening loop when the set is small and known: a
# tool name, an enum value, the terminals a grammar accepts next.
class _Decider(Constraint):
    """Wraps a `decide(text, candidates) -> (verdict, allowed)` callback."""

    def __init__(self, decide, finished=None, reset=None):
        if not callable(decide):
            raise TypeError("needs a callable decide(text, candidates)")
        self._decide = decide
        self._finished = finished
        self._reset = reset

    def reset(self):
        if self._reset is not None:
            self._reset()

    def decide(self, text, candidates):
        got = self._decide(text, candidates)
        if isinstance(got, tuple):
            v, allowed = (list(got) + [None])[:2]
            return verdict(v), allowed
        # A bare list is a permitted set with nothing to say about what happens next; a bare
        # verdict is a decision with no opinion on which candidate.
        if isinstance(got, Verdict) or got is True or got is False or isinstance(got, str):
            return verdict(got), None
        return ALLOW, got

    def allows(self, text, piece):
        # The per-candidate form, derived, so a constraint written as `decide` still works
        # everywhere a constraint is accepted -- composed with another one, or reached by a
        # caller that only knows the predicate protocol.
        v, opts = self.decide(text, [piece])
        if not v.allow:
            return v
        return v if piece_ok(piece, opts) else DENY

    def finished(self, text):
        return bool(self._finished(text)) if self._finished is not None else False


def piece_ok(piece, options):
    """Is `piece` a step towards one of `options`? `None` means everything is.

    Either direction counts. A piece shorter than an option is a prefix of it and is how a
    multi-token option gets spelled out; a piece longer than an option overshoots but still
    starts with it, which happens when one token covers a whole option and some of what
    follows.
    """
    if options is None:
        return True
    for o in options:
        o = str(o)
        if o.startswith(piece) or piece.startswith(o):
            return True
    return False


class CallbackConstraint(Constraint):
    """A plain function decides what may come next: `allows(text, piece) -> bool`.

    The simplest form of the whole idea, and the one that needs no class: given everything
    generated so far and one candidate continuation, say whether it is allowed. The decision
    is made per token against the CURRENT text, so it can depend on everything already
    written -- a bracket depth, a field the schema has not seen yet, a state machine of the
    caller's own.

    `allows` may answer with more than yes or no. The answer is a `Verdict`, which says
    three separate things -- whether the candidate is allowed, whether it goes into the
    reply, and whether to keep asking, run free, or end -- and the six that mean anything
    are named at the top of this module. `True` and `False` still work, as ALLOW and DENY.

    `finished(text)` is optional and says the output is complete, which lets generation stop
    on the constraint rather than on a token budget. Returning `"stop"` from `allows` is the
    same statement made at the moment the decision is actually available.

    Both see decoded TEXT, never token ids. That is what makes a constraint portable across
    tokenizers and expressible in the caller's own terms: "a digit may follow" is a statement
    about characters, and which token ids happen to spell them is not the caller's problem.
    """

    def __init__(self, allows, finished=None, reset=None):
        if not callable(allows):
            raise TypeError("a callback constraint needs a callable allows(text, piece)")
        self._allows = allows
        self._finished = finished
        self._reset = reset

    def reset(self):
        if self._reset is not None:
            self._reset()

    def allows(self, text, piece):
        return verdict(self._allows(text, piece))

    def finished(self, text):
        return bool(self._finished(text)) if self._finished is not None else False


class AllOf(Constraint):
    """Every part must allow a piece, and any part may end the generation. Lets `stop=`
    compose with a structural constraint instead of one silently replacing the other."""

    def __init__(self, parts):
        self.parts = [p for p in parts if p is not None]

    def reset(self):
        for p in self.parts:
            p.reset()

    def dormant(self, text):
        """Only when every part is. A part that does not implement it is not dormant --
        the same reading the sampler gives a constraint of a caller's own."""
        for p in self.parts:
            d = getattr(p, "dormant", None)
            if not (callable(d) and d(text)):
                return False
        return True

    def allows(self, text, piece):
        return all(p.allows(text, piece) for p in self.parts)

    def finished(self, text):
        return any(p.finished(text) for p in self.parts)


_BUILTIN = {"json": JsonConstraint}


def build(spec):
    """Turn a constraint spec into a Constraint: a name ("json"), a compiled pattern or
    {"regex": ...}, a list of stop strings, or an object that already is one."""
    if spec is None or isinstance(spec, Constraint):
        return spec
    if isinstance(spec, str):
        if spec in _BUILTIN:
            return _BUILTIN[spec]()
        raise ValueError("unknown constraint %r; built in: %s"
                         % (spec, ", ".join(sorted(_BUILTIN))))
    if isinstance(spec, dict):
        if "regex" in spec:
            return RegexConstraint(spec["regex"])
        if "stop" in spec:
            return StopConstraint(spec["stop"])
        if callable(spec.get("decide")):
            return _Decider(spec["decide"], spec.get("finished"), spec.get("reset"))
        if callable(spec.get("allows")):
            return CallbackConstraint(spec["allows"], spec.get("finished"),
                                      spec.get("reset"))
        raise ValueError("constraint dict needs an 'allows', 'regex' or 'stop' key, got %s"
                         % sorted(spec))
    if isinstance(spec, (list, tuple)):
        return StopConstraint(spec)
    if callable(getattr(spec, "allows", None)):
        return spec
    # A bare function is the most direct spelling of the idea, so it is accepted as one.
    if callable(spec):
        return CallbackConstraint(spec)
    raise TypeError("constraint must be a name, a callable allows(text, piece), a dict, "
                    "a list of stops, or a Constraint")
