"""Reading and writing tool calls in whatever form a model uses.

The delimiters a model wraps a call in, and whether it writes JSON or XML inside them, are
ITS convention and not a standard -- so they are read off its own template (see
`BPETokenizer.tool_call_format`) and passed in here as `fmt`. The fallbacks below are only
what to try when a template could not be read at all; a last resort, never the rule.

Everything here is a pure function of (text, the tools this call was given, that format).
Nothing knows any model family by name.
"""
import json
import re

# Tried only when the template could not be read.
FALLBACK_DELIMS = [("<tool_call>", "</tool_call>"),
                   ("<|tool_call|>", "<|/tool_call|>"),
                   ("[TOOL_CALLS]", "")]

# A second wire format, not a special case for one model: some templates write
# <function=NAME><parameter=KEY>value</parameter></function> instead of JSON, the way
# `nested` and `flat` are two shapes of the same JSON thing.
_RE_XML_CALL = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function\s*>", re.S)
_RE_XML_PARAM = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter\s*>", re.S)


def tool_names(tools):
    """The names in a `tools` list, whichever of the two shapes it uses."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        f = t.get("function")
        n = (f.get("name") if isinstance(f, dict) else None) or t.get("name")
        if n:
            out.append(str(n))
    return out


def match_name(name, tools):
    """The registered tool `name` refers to, or None.

    Whitespace, punctuation and case carry no meaning in a tool name: "run_ Python" is
    "run_python" with noise in it. An exact match first; a name that is identical once the
    noise is stripped is the model's intent stated plainly, so it counts -- that is reading
    what it said, not guessing what it meant. Anything further apart than that is not
    matched, because a wrong tool run confidently is worse than one not run.
    """
    names = tool_names(tools)
    if name in names:
        return name
    want = _norm(name)
    for n in names:
        if _norm(n) == want:
            return n
    return None


def _norm(s):
    return re.sub(r"[\s_\-.]+", "", str(s or "").lower())


def _schema(tools, name, key):
    """The declared type of one argument, or None."""
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        f = t.get("function") if isinstance(t.get("function"), dict) else t
        if f.get("name") != name:
            continue
        props = (f.get("parameters") or {}).get("properties") or {}
        return (props.get(key) or {}).get("type")
    return None


def _delims(fmt):
    o = (fmt or {}).get("open")
    if o:
        return [(o, (fmt or {}).get("close") or "")]
    return list(FALLBACK_DELIMS)


def _xml_value(tools, name, key, raw):
    """How a value was encoded is stated by the templates that write this form:

        args_value | string if args_value is string else args_value | tojson

    so a string went in as itself and everything else went in as JSON. Read it back the same
    way, using the tool's own schema to say which -- guessing from the text would turn the
    string "12" into a number, and a code argument that happens to be `[1, 2]` into a list.
    """
    body = str(raw)
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n"):
        body = body[:-1]
    t = _schema(tools, name, key)
    if t == "string":
        return body
    if t:
        try:
            return json.loads(body)
        except Exception:
            return body
    # No schema to consult: JSON if it parses as something structured, text otherwise. A bare
    # word is not JSON and stays a word.
    try:
        v = json.loads(body)
    except Exception:
        return body
    return v if isinstance(v, (dict, list, bool)) else body


def _as_call(o, wrapped, tools):
    """One parsed object as a call, or None."""
    if not isinstance(o, dict):
        return None
    # The nested form is recognised HERE, at the outer object, not left to be found as the
    # inner one. Both are calls and both parse -- but the span differs, and the span is what
    # gets removed: matching the inner object strips it out of the middle of its wrapper and
    # leaves `{"type":"function","function":}` in the reply. Parse and strip have to agree on
    # the same characters, which is the whole reason they share one scan.
    inner = o["function"] if (o.get("type") == "function"
                              and isinstance(o.get("function"), dict)) else o
    name = inner.get("name")
    if not isinstance(name, str):
        return None
    known = name in tool_names(tools)
    if not known and not wrapped:
        return None
    # An id a model gives its own call is kept: a template that ties results to calls by id
    # must see the SAME one back, not one invented here. It sits on the wrapper when there is
    # one, so both are looked at.
    cid = None
    for k in ("id", "tool_call_id", "call_id"):
        v = o.get(k, inner.get(k))
        if isinstance(v, str) and v:
            cid = v
            break
    args = inner.get("arguments")
    if args is None:
        args = inner.get("parameters")
    return {"name": name, "args": args if args is not None else {},
            "known": known, "id": cid}


def _as_xml(payload, tools):
    m = _RE_XML_CALL.search(str(payload))
    if not m:
        return None
    name = m.group(1)
    args = {}
    for p in _RE_XML_PARAM.finditer(m.group(2)):
        args[p.group(1)] = _xml_value(tools, name, p.group(1), p.group(2))
    return {"name": name, "args": args, "known": name in tool_names(tools), "id": None}


def parse(text, tools=None, fmt=None):
    """Every tool call in `text`, each with the span it occupies:
    [{"name", "args", "id", "known", "start", "end"}], in order.

    ONE scan, used both for running the calls and for removing them from what is shown. Two
    separate readers drift: the reply ends up with a call the loop did not run, printed raw
    as prose -- which is what happened, a bare call at the end of a message neither executed
    nor hidden.

    A span counts as a call when it is inside the model's call markup (that is protocol,
    whatever name it carries) or when it is a bare JSON object naming a tool that ACTUALLY
    EXISTS in `tools`. Bare JSON with an unknown name is left alone: a reply may legitimately
    show a JSON object, and guessing would eat the answer.
    """
    src = str(text)
    found = []
    form = (fmt or {}).get("payload")          # 'flat' | 'nested' | 'xml' | None

    def overlaps(a, b):
        return any(a < f["end"] and f["start"] < b for f in found)

    def as_json(body):
        try:
            return _as_call(json.loads(body), True, tools)
        except Exception:
            return None

    # Inside the model's own delimiters the question is no longer WHETHER this is a call --
    # the wrapper settled that -- only which encoding it used. So the declared form is tried
    # first and the other one after: a template that says JSON while the model writes XML is
    # a real disagreement, and refusing to read it puts the markup back in the reply for the
    # sanitiser to strip into a bare fragment. Outside the delimiters nothing is that
    # generous.
    def read_payload(body):
        if form == "xml":
            return _as_xml(body, tools) or as_json(body)
        return as_json(body) or _as_xml(body, tools)

    for o, c in _delims(fmt):
        pat = re.escape(o) + r"\s*(.*?)\s*" + (re.escape(c) if c else r"(?=\n|$)")
        for m in re.finditer(pat, src, re.S):
            call = read_payload(m.group(1))
            if call and not overlaps(m.start(), m.end()):
                found.append({"start": m.start(), "end": m.end(), "call": call})

    # The same payload with its wrapper missing. Only for a model whose form this IS (or one
    # we could not ask): a `<function=...>` block is markup nothing else writes, and left as
    # prose the sanitiser eats the tags as unknown elements and prints the arguments as if
    # they were the answer.
    if form == "xml" or form is None:
        for m in _RE_XML_CALL.finditer(src):
            if overlaps(m.start(), m.end()):
                continue
            call = _as_xml(m.group(0), tools)
            if call:
                found.append({"start": m.start(), "end": m.end(), "call": call})

    # Bare objects, wherever they sit. Brace-matched rather than regexed: the arguments are
    # themselves an object, and a pattern stopping at the first `}` would cut a call in half.
    k = 0
    while k < len(src):
        if src[k] != "{" or any(f["start"] <= k < f["end"] for f in found):
            k += 1
            continue
        depth, in_str, esc, end = 0, False, False, -1
        for q in range(k, len(src)):
            ch = src[q]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = q + 1
                    break
        if end < 0:
            break
        call = None
        try:
            call = _as_call(json.loads(src[k:end]), False, tools)
        except Exception:
            pass
        if call:
            found.append({"start": k, "end": end, "call": call})
            k = end
        else:
            k += 1

    found.sort(key=lambda f: f["start"])
    return [dict(f["call"], start=f["start"], end=f["end"]) for f in found]


def calls(text, tools=None, fmt=None):
    """Just the calls, without their spans."""
    return [{k: v for k, v in c.items() if k not in ("start", "end")}
            for c in parse(text, tools, fmt)]


def strip(text, tools=None, fmt=None):
    """`text` with every call span removed. The markup is protocol, not prose -- and so is a
    bare call to a tool that exists."""
    src = str(text)
    spans = parse(src, tools, fmt)
    if not spans:
        return src.strip()
    out, at = [], 0
    for s in spans:
        out.append(src[at:s["start"]])
        at = s["end"]
    out.append(src[at:])
    return "".join(out).strip()


def render(name, args, fmt=None):
    """One call written the way this model writes them. Used when showing a model the call it
    should have made: the example has to be something it can send back verbatim."""
    f = fmt or {}
    o = f.get("open") or "<tool_call>"
    c = f.get("close") or "</tool_call>"
    if f.get("payload") == "xml":
        body = "<function=%s>\n" % name
        for k, v in (args or {}).items():
            body += "<parameter=%s>\n%s\n</parameter>\n" % (
                k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        body += "</function>"
        return "%s\n%s\n%s" % (o, body, c)
    if f.get("payload") == "nested":
        payload = {"type": "function", "function": {"name": name, "arguments": args or {}}}
    else:
        payload = {"name": name, "arguments": args or {}}
    return "%s\n%s\n%s" % (o, json.dumps(payload, ensure_ascii=False), c)


def result_message(call, content, keeps_name=False, id_field=None, via=None):
    """The message that carries one tool's result back to the model.

    A template that drops the tool's name and has no id field gives the model no way to tell
    which call a result belongs to, so the name is folded into the body for those -- and only
    for those, since doing it unconditionally puts the name in twice.
    """
    carries = bool(keeps_name) or bool(id_field)
    if carries:
        body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    else:
        try:
            inner = json.loads(content) if isinstance(content, str) else content
        except Exception:
            inner = content
        body = json.dumps({"tool": call.get("name"), "result": inner}, ensure_ascii=False)
    if via == "user":
        return {"role": "user", "content": "Result of the tool you called:\n" + body}
    msg = {"role": "tool", "name": call.get("name"), "content": body}
    if id_field:
        msg[id_field] = call.get("id")
    return msg
