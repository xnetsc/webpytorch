#!/usr/bin/env python3
"""Refuse a commit whose documentation contradicts the code it documents.

A document does not go wrong by getting old. It goes wrong when it ASSERTS something about
the code that is no longer true -- and a reader who follows it is then worse off than if it
had said nothing. Every documentation error found in the audit of 2026-09-04 was that shape:
`max_parallel=8` when the parameter defaults to 16, `bits=4` when it defaults to None,
`generate() -> str` when it returns a GenResult, a whole public API (constraints, tool
calling) that no document mentioned at all.

So the rule this enforces is not "you touched code, therefore touch a document". That rule
is satisfied by editing a space, catches none of the above, and would stop commits that
genuinely need no documentation change. It is:

  A document may not state a fact about the code that the commit makes false.

Two such facts are checkable without judgement, and those two are refused:

  A. A signature written in docs/API.md as a definition entry -- a bullet whose first thing
     is `name(param=default, ...)` -- must agree with the real signature. Usage examples
     mid-sentence are NOT definitions and are ignored; `stream(channels=True)` in a sentence
     about live streams is illustrating a call, not declaring the default.
  B. A name this commit ADDS to `webtorch.__all__` must appear somewhere in docs/API.md.
     Only names the commit adds: 22 public names were already undocumented when this check
     was written, and a hook that refuses every commit until an existing backlog is cleared
     is a hook that teaches everyone to pass --no-verify.

Everything else about documentation needs a human. Prose claims -- what a backend does, how
fast something is, which formats are supported -- cannot be checked against source, so when
the public surface moves and no document does, this prints what changed and lets the commit
through. That is the "if necessary" part, and it belongs to the person, not the hook.

This is a `pre-commit` check: it reads the STAGED tree, so it judges the commit being made.

Enable it with:  git config core.hooksPath .githooks
Bypass it with:  git commit --no-verify
"""
import ast
import re
import subprocess
import sys

API = 'docs/API.md'
INIT = 'webtorch/__init__.py'

# A definition entry: a list bullet whose first content is a backticked call with defaults.
# `- `webtorch.use_default_io(cache=True, ...)`` declares; "...as `stream(channels=True)`
# labels the pieces" illustrates. Only the first is a statement about the signature.
DEFN = re.compile(r'^\s*[-*]\s+`(?:await\s+)?'
                  r'([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)'
                  r'\(([^`]*)\)`', re.M)
# Split on commas that are not inside brackets, so `f(a, b={1: 2})` stays one parameter.
SPLIT = re.compile(r',(?![^()\[\]{}]*[)\]}])')
PLACEHOLDER = ('…', '...')


def run(*args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return p.stdout if p.returncode == 0 else None


def staged(path):
    """`path` as this commit will record it, or None if the commit does not have it."""
    return run('git', 'show', ':' + path)


def head(path):
    """`path` as the previous commit recorded it, or None (a new file, or no HEAD yet)."""
    return run('git', 'show', 'HEAD:' + path)


def definitions(source, filename):
    """{'name': [node], 'Class.name': [node]} for every def in one module.

    A class member is not always a `def`. `Quantizer.stream = staticmethod(stream_quantize)`
    is the real signature of `Quantizer.stream`, and reading only `def` statements sends the
    lookup off to some unrelated `stream` elsewhere -- which is exactly the false alarm this
    check must not raise.
    """
    out = {}
    try:
        tree = ast.parse(source, filename)
    except SyntaxError:
        return out                          # not our business to report; Python will
    def add(key, node):
        out.setdefault(key, []).append(node)

    top = {}                                # module-level defs, for resolving aliases
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top[node.name] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(node.name + '.' + sub.name, sub)
                elif isinstance(sub, ast.Assign) and len(sub.targets) == 1 and \
                        isinstance(sub.targets[0], ast.Name):
                    target = alias_target(sub.value, top)
                    if target is not None:
                        add(node.name + '.' + sub.targets[0].id, target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node)
    return out


def alias_target(value, top):
    """`f`, `staticmethod(f)` or `classmethod(f)` -> the def of f, when f is in this module."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and \
            value.func.id in ('staticmethod', 'classmethod') and len(value.args) == 1:
        value = value.args[0]
    if isinstance(value, ast.Name):
        return top.get(value.id)
    return None


def defaults_of(node):
    """param -> default, as an AST node."""
    a = node.args
    out = {}
    positional = a.posonlyargs + a.args
    if a.defaults:
        for arg, d in zip(positional[len(positional) - len(a.defaults):], a.defaults):
            out[arg.arg] = d
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[arg.arg] = d
    return out


def same(doc_text, node):
    """Does the documented default mean the same as the real one?

    Compared as VALUES where both are literals, so `"auto"` and 'auto' agree -- the doc is
    not wrong for choosing different quotes. Anything not a literal (a call, a constant from
    elsewhere) is compared as normalised source text.
    """
    real_src = ast.unparse(node)
    if doc_text == real_src:
        return True
    try:
        return ast.literal_eval(doc_text) == ast.literal_eval(real_src)
    except (ValueError, SyntaxError):
        return doc_text.replace(' ', '') == real_src.replace(' ', '')


def public_names(source):
    if not source:
        return set()
    try:
        tree = ast.parse(source, INIT)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
            return {e.value for e in node.value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def main():
    names = (run('git', 'diff', '--cached', '--name-only') or '').split()
    touched_sdk = [n for n in names if n.startswith('webtorch/') and n.endswith('.py')]
    touched_doc = [n for n in names if n.endswith('.md')]
    if not touched_sdk and API not in names:
        return 0                            # nothing here can have moved

    # The SDK exactly as this commit records it.
    defs = {}
    for path in (run('git', 'ls-files', 'webtorch/*.py') or '').split():
        src = staged(path)
        if src is None:
            continue
        for key, nodes in definitions(src, path).items():
            defs.setdefault(key, []).extend(nodes)

    doc = staged(API) or ''
    problems = []

    # ---- A. a documented signature that the code contradicts ----------------------------
    for m in DEFN.finditer(doc):
        dotted, argstr = m.group(1), m.group(2)
        parts = dotted.split('.')
        # A dotted name names ONE thing. If `Class.method` does not resolve, that is this
        # checker failing to find it, not licence to grade the entry against a same-named
        # function from an unrelated class -- guessing there is how a correct document gets
        # reported as wrong, and one false alarm is enough to make the whole hook ignorable.
        if len(parts) >= 2 and parts[-2] != 'webtorch' and parts[-2][:1].isupper():
            candidates = defs.get(parts[-2] + '.' + parts[-1])
        else:
            candidates = defs.get(parts[-1])
        if not candidates:
            continue                        # not defined in the SDK: a shim, or not ours
        documented = {}
        for piece in SPLIT.split(argstr):
            if '=' not in piece:
                continue
            key, value = piece.split('=', 1)
            key, value = key.strip(), value.strip()
            if key.startswith('*') or not value or any(p in value for p in PLACEHOLDER):
                continue                    # `**kw`, or a deliberately elided default
            documented[key] = value
        if not documented:
            continue
        # An overloaded name is fine as long as SOME definition matches what is written.
        best = None
        for node in candidates:
            real = defaults_of(node)
            wrong = {k: v for k, v in documented.items()
                     if k not in real or not same(v, real[k])}
            if not wrong:
                best = None
                break
            if best is None or len(wrong) < len(best[1]):
                best = (node, wrong, real)
        if best is None:
            continue
        node, wrong, real = best
        for key, written in sorted(wrong.items()):
            actual = ast.unparse(real[key]) if key in real else 'no such parameter'
            problems.append('%s: %s(%s=%s) -- the code says %s'
                            % (API, dotted, key, written, actual))

    # ---- B. a public name this commit adds, documented nowhere ---------------------------
    if INIT in names:
        added = public_names(staged(INIT)) - public_names(head(INIT))
        for name in sorted(added):
            if not re.search(r'(?<![A-Za-z_0-9])%s(?![A-Za-z_0-9])' % re.escape(name), doc):
                problems.append('%s: `%s` is public now and appears in no reference entry'
                                % (INIT, name))

    if problems:
        sys.stderr.write('\ncommit refused: the documentation would contradict the code\n\n')
        for p in problems:
            sys.stderr.write('  ' + p + '\n')
        sys.stderr.write("""
A wrong document is worse than a missing one: someone follows it and is wrong with
confidence. Correct the entry (or the code, if the document was right), stage it, and
commit again.

Only provable contradictions are refused. Prose -- what a backend does, how fast something
is -- is not checked, and is still yours to keep true.

Deliberate?  git commit --no-verify
""")
        return 1

    # ---- C. the part no hook can judge: say what moved, do not block ----------------------
    if touched_sdk and not touched_doc:
        moved = []
        for path in touched_sdk:
            before, after = head(path), staged(path)
            if before is None or after is None:
                continue
            b, a = definitions(before, path), definitions(after, path)
            moved += ['+ ' + k for k in sorted(set(a) - set(b))]
            moved += ['- ' + k for k in sorted(set(b) - set(a))]
        if moved:
            sys.stderr.write('\nnote: this commit changes the shape of the SDK and no '
                             'document with it.\n')
            for line in moved[:12]:
                sys.stderr.write('  ' + line + '\n')
            if len(moved) > 12:
                sys.stderr.write('  ... and %d more\n' % (len(moved) - 12))
            sys.stderr.write('If any of it is a public behaviour someone could rely on, '
                             'docs/API.md is where\nit belongs. Committing anyway.\n\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
