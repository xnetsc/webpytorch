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


class Constraint:
    """Base class. Override `allows`; override `finished` if the output can end early."""

    def reset(self):
        """Called once before each generation."""

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

    Uses `regex`'s partial matching where available so a prefix is accepted while it can
    still grow into a match; without it, the check falls back to matching what is complete.
    """

    def __init__(self, pattern):
        try:
            import regex as _re
            self._partial = True
        except ImportError:
            import re as _re
            self._partial = False
        self.re = _re
        self.rx = _re.compile(pattern)

    def allows(self, text, piece):
        s = text + piece
        if self._partial:
            return self.rx.fullmatch(s, partial=True) is not None
        return self.rx.match(s) is not None

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

    def allows(self, text, piece):
        return True

    def finished(self, text):
        return any(s in text for s in self.stops)


class AllOf(Constraint):
    """Every part must allow a piece, and any part may end the generation. Lets `stop=`
    compose with a structural constraint instead of one silently replacing the other."""

    def __init__(self, parts):
        self.parts = [p for p in parts if p is not None]

    def reset(self):
        for p in self.parts:
            p.reset()

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
        raise ValueError("constraint dict needs a 'regex' or 'stop' key, got %s"
                         % sorted(spec))
    if isinstance(spec, (list, tuple)):
        return StopConstraint(spec)
    if callable(getattr(spec, "allows", None)):
        return spec
    raise TypeError("constraint must be a name, a dict, a list of stops, or a Constraint")
