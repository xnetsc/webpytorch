"""Translate the restricted WGSL the quantized decoders are written in to GLSL ES 3.00.

Why a translator and not a second set of kernels: the dequantization math for the 28 ggml
formats is the same arithmetic on both backends -- only the shading language differs. Kept
as two hand-written copies they would diverge on the first format added, and the one nobody
ran would be the one that broke. There is one source of truth (the WGSL decoder bodies in
`_core`), and WebGL reads it through here.

The subset is small and closed, because the decoders were written against it: scalar `let`
and `var`, `for`, `if`/`else`, function definitions, calls, member access on vectors, and
the bit/arithmetic operators. No arrays, no pointers, no structs, no loops other than `for`.

Type inference is exact rather than heuristic. WGSL permits NO implicit conversion between
u32/i32/f32, so every expression's type is already pinned by the source; this reproduces
that reasoning to recover the type each `let` needs spelled out in GLSL. The one wrinkle is
WGSL's abstract integer literal (`15` in `f32(i32(e) - 15)`), which adopts the type of what
it meets -- represented here as ABSTRACT and resolved at the operator.

GLSL ES 3.00 is more permissive than WGSL at the leaves (it converts int->uint->float
implicitly), so a type recovered correctly always compiles; the risk is the reverse, and it
is a loud one -- `>>` is arithmetic on a signed type and logical on an unsigned one, so a
u32 mistaken for an i32 changes results rather than failing to build. That is what the ggml
self-check compares against the numpy reference, format by format.
"""

import re

# ---- types -----------------------------------------------------------------------------
F, U, I, B = "float", "uint", "int", "bool"
V4, V3, V2 = "vec4", "vec3", "vec2"
ABSTRACT = "@int"          # an unsuffixed integer literal, before it meets an operator

_WGSL_TYPE = {"f32": F, "u32": U, "i32": I, "bool": B,
              "vec4<f32>": V4, "vec3<f32>": V3, "vec2<f32>": V2,
              "vec4<u32>": "uvec4", "vec2<u32>": "uvec2",
              "vec4<i32>": "ivec4", "vec2<i32>": "ivec2"}

_VEC_ELEM = {V4: F, V3: F, V2: F, "uvec4": U, "uvec2": U, "ivec4": I, "ivec2": I}

# Struct member types, for the one struct the kernels use. `gm.K` has to come back as uint,
# not as the float a vector swizzle would give -- the difference decides whether `gm.K /
# BLKVALS` is an integer division or a float one, and the float one is off by a block.
STRUCTS = {"GM": {"M": U, "N": U, "K": U, "rowb": U, "estride": U,
                  "eslot": U, "xper": U, "pad": U}}

# Return types of the builtins the decoders use. Anything not listed is either translated
# structurally (constructors) or is a user function whose signature is read from its `fn`.
_BUILTIN = {
    "dot": F, "exp2": F, "sqrt": F, "inverseSqrt": F, "log2": F, "pow": F,
    "floor": F, "ceil": F, "round": F, "fract": F, "sin": F, "cos": F,
    "countOneBits": U, "firstLeadingBit": U, "reverseBits": U,
    "unpack4x8unorm": V4, "unpack2x16float": V2,
}
# min/max/abs/sign/clamp/select take their type from an argument rather than having one.
_FROM_ARG0 = {"min", "max", "abs", "sign", "clamp", "select", "mix", "step"}


def _rank(t):
    return {ABSTRACT: 0, B: 0, I: 1, U: 2, F: 3}.get(t, 4)


def _join(a, b):
    """Result type of a binary arithmetic/bit operator on operand types `a` and `b`."""
    if a == ABSTRACT:
        return b
    if b == ABSTRACT:
        return a
    if a in _VEC_ELEM:
        return a
    if b in _VEC_ELEM:
        return b
    return a if _rank(a) >= _rank(b) else b


# Words GLSL reserves that a WGSL author is free to use as a name -- `half` is a loop
# variable in three of the K-quant decoders -- renamed here rather than left for a driver to
# report as a syntax error on a line that looks fine.
#
# Deliberately NOT listed: anything that is a keyword in BOTH languages (`for`, `if`,
# `return`, `struct`, `const`, `true`, `bool`) or that WGSL spells structurally (`vec2`..
# `vec4`). Those can never appear as an identifier in the input, so renaming them can only
# break the translation -- which it did: `vec4<f32>` became `vec4_<f32>`.
_RESERVED = set("""
attribute varying uniform buffer shared coherent volatile restrict readonly writeonly
atomic_uint layout centroid flat smooth noperspective patch sample precise
do while switch case default in out inout discard
mat2 mat3 mat4 ivec2 ivec3 ivec4 bvec2 bvec3 bvec4 uvec2 uvec3 uvec4
lowp mediump highp precision invariant
common partition active asm class union enum typedef template this resource goto inline
noinline public static extern external interface long short double half fixed unsigned
superp input output hvec2 hvec3 hvec4 fvec2 fvec3 fvec4 dvec2 dvec3 dvec4
sampler2DRect sampler3DRect samplerBuffer filter sizeof cast namespace using row_major
subroutine packed float int uint void
""".split())


def _derserve(src):
    """Rename identifiers that GLSL reserves. Comments are left alone."""
    out = []
    for line in src.split("\n"):
        code, sep, comment = line.partition("//")
        out.append(re.sub(r"\b(%s)\b" % "|".join(_RESERVED),
                          lambda m: m.group(1) + "_", code) + sep + comment)
    return "\n".join(out)


# ---- expression tokenizer / type inference ---------------------------------------------
_TOK = re.compile(r"""
    (?P<float>  \d+\.\d*(?:[eE][-+]?\d+)?f? | \.\d+(?:[eE][-+]?\d+)?f? | \d+[eE][-+]?\d+f? )
  | (?P<uint>   (?:0[xX][0-9a-fA-F]+|\d+)u )
  | (?P<int>    0[xX][0-9a-fA-F]+ | \d+ )
  | (?P<name>   (?:vec[234]|bitcast|array)\s*<[^>]*>  |  [A-Za-z_]\w* )
  | (?P<op>     <<|>>|<=|>=|==|!=|&&|\|\| | [-+*/%&|^~!<>(),.;?:\[\]{}=] )
  | (?P<ws>     \s+ )
""", re.X)


def _tokens(s):
    out, i = [], 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m:
            raise ValueError("wgsl2glsl: cannot tokenize at %r" % s[i:i + 40])
        i = m.end()
        if m.lastgroup == "ws":
            continue
        tok = m.group()
        if m.lastgroup == "name" and "<" in tok:
            tok = re.sub(r"\s+", "", tok)       # vec4 < f32 > and vec4<f32> are one name
        out.append((m.lastgroup, tok))
    return out


class _Types:
    """Names in scope -> type, plus the signatures of functions already translated."""

    def __init__(self):
        self.var = {}
        self.fn = dict(_BUILTIN)

    def copy(self):
        t = _Types()
        t.var = dict(self.var)
        t.fn = self.fn          # shared: functions are module-wide
        return t


def _infer(toks, ty, i=0, stop=None):
    """Type of the expression starting at token `i`. Returns (type, next_index).

    A full parser is unnecessary: the type of an expression in this subset is decided by
    its leaves and the join rule, and precedence never changes a type -- so this walks the
    token run, joining as it goes, and only needs real structure for calls and indexing."""
    cur = None
    n = len(toks)
    while i < n:
        kind, tok = toks[i]
        if stop and tok in stop and kind == "op":
            break
        if kind == "op" and tok in "),]":
            break
        if kind == "float":
            cur = _join(cur, F) if cur else F; i += 1
        elif kind == "uint":
            cur = _join(cur, U) if cur else U; i += 1
        elif kind == "int":
            cur = _join(cur, ABSTRACT) if cur else ABSTRACT; i += 1
        elif kind == "name":
            # a call?
            if i + 1 < n and toks[i + 1][1] == "(":
                name = tok
                j = _skip_call(toks, i + 1)
                if name.startswith("bitcast<"):
                    t = _WGSL_TYPE[name[len("bitcast<"):-1]]
                elif name in _WGSL_TYPE:
                    t = _WGSL_TYPE[name]
                elif name in _FROM_ARG0:
                    t, _ = _infer(toks, ty, i + 2, stop={",", ")"})
                    if name == "select":
                        pass                     # select(false, true, cond): arg0 has the type
                elif name in ty.fn:
                    t = ty.fn[name]
                else:
                    raise ValueError("wgsl2glsl: unknown function %r" % name)
                cur = _join(cur, t) if cur else t
                i = j
            elif tok in ("true", "false"):
                cur = _join(cur, B) if cur else B; i += 1
            else:
                if tok not in ty.var:
                    raise ValueError("wgsl2glsl: unknown name %r" % tok)
                cur = _join(cur, ty.var[tok]) if cur else ty.var[tok]
                i += 1
        elif kind == "op" and tok == ".":
            # swizzle: .x on a vector is its element type; .xy is a shorter vector
            base = cur
            sw = toks[i + 1][1]
            if base in STRUCTS:
                cur = STRUCTS[base][sw]; i += 2; continue
            elem = _VEC_ELEM.get(base, F)
            cur = elem if len(sw) == 1 else {2: V2, 3: V3, 4: V4}[len(sw)]
            i += 2
        elif kind == "op" and tok == "(":
            t, j = _infer(toks, ty, i + 1)
            cur = _join(cur, t) if cur else t
            i = j + 1
        elif kind == "op" and tok in ("<", ">", "<=", ">=", "==", "!=", "&&", "||"):
            cur = B; i += 1
            # the right-hand side cannot change a comparison's type; skip it
            depth = 0
            while i < n:
                k2, t2 = toks[i]
                if k2 == "op" and t2 in "([":
                    depth += 1
                elif k2 == "op" and t2 in ")]":
                    if depth == 0:
                        break
                    depth -= 1
                elif k2 == "op" and t2 in (",", ";") and depth == 0:
                    break
                i += 1
        elif kind == "op" and tok == "?":
            t, j = _infer(toks, ty, i + 1, stop={":"})
            cur = t
            i = j
            depth = 0
            while i < n and not (toks[i][1] == ";" and depth == 0):
                i += 1
        elif kind == "op":
            i += 1
        else:
            i += 1
    return (cur or ABSTRACT), i


def _skip_call(toks, i):
    """`i` is the '(' of a call; return the index just past its ')'."""
    depth = 0
    n = len(toks)
    while i < n:
        t = toks[i][1]
        if t in "([":
            depth += 1
        elif t in ")]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


# ---- expression rewriting ---------------------------------------------------------------
def _args(s, i):
    """`i` is the index of '('; return (list of argument source strings, index past ')')."""
    assert s[i] == "("
    depth, start, out = 0, i + 1, []
    j = i
    while j < len(s):
        c = s[j]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth == 0:
                out.append(s[start:j])
                return out, j + 1
        elif c == "," and depth == 1:
            out.append(s[start:j]); start = j + 1
        j += 1
    raise ValueError("wgsl2glsl: unbalanced parentheses in %r" % s[i:i + 60])


def _rewrite_buffers(s, buffers):
    """`name[expr]` -> `fetch(int(expr))`.

    A storage buffer indexes like an array in WGSL and is a texture in GLSL, so every read
    of one has to become a fetch call. Scanned rather than matched by regex for the same
    reason as the constructors: the index expression contains brackets of its own.
    """
    if not buffers:
        return s
    out, i, n = [], 0, len(s)
    pat = re.compile(r"\b(%s)\s*\[" % "|".join(map(re.escape, buffers)))
    while i < n:
        m = pat.match(s, i)
        if not m:
            out.append(s[i]); i += 1; continue
        depth, j = 0, m.end() - 1
        while j < n:
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = _rewrite_buffers(s[m.end():j], buffers)
        out.append("%s(int(%s))" % (buffers[m.group(1)], inner))
        i = j + 1
    return "".join(out)


def _rewrite_expr(s):
    """Constructors, `select`, and `bitcast` -- the forms whose GLSL spelling is structural.

    Done as a scan rather than a regular expression because every one of them nests: the
    arguments of a `select` are routinely another `select`, and a regex that matched to the
    first ')' would cut them in half.
    """
    out, i, n = [], 0, len(s)
    while i < n:
        m = re.compile(r"\b(select|bitcast\s*<\s*(f32|u32|i32)\s*>|"
                       r"vec4<f32>|vec3<f32>|vec2<f32>|vec4<u32>|vec2<u32>|"
                       r"vec4<i32>|vec2<i32>|f32|u32|i32)\s*\(").match(s, i)
        if not m:
            out.append(s[i]); i += 1; continue
        head = m.group(1)
        args, j = _args(s, m.end() - 1)
        args = [_rewrite_expr(a) for a in args]
        if head == "select":
            # WGSL orders it (if_false, if_true, condition); the ternary is the other way.
            out.append("((%s) ? (%s) : (%s))" % (args[2], args[1], args[0]))
        elif head.startswith("bitcast"):
            to = m.group(2)
            fn = {"f32": "uintBitsToFloat", "u32": "floatBitsToUint",
                  "i32": "floatBitsToInt"}[to]
            out.append("%s(%s)" % (fn, args[0]))
        else:
            out.append("%s(%s)" % (_WGSL_TYPE.get(head, head), ", ".join(args)))
        i = j
    return "".join(out)


# ---- statement translation --------------------------------------------------------------
_FN_RE = re.compile(r"^\s*fn\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([\w<>]+)\s*)?\{")
_LET_RE = re.compile(r"^\s*(let|var)\s+(?:<\s*private\s*>\s*)?(\w+)\s*(?::\s*([\w<>]+))?\s*=\s*(.*);\s*$")
_DECL_RE = re.compile(r"^\s*var\s*(?:<\s*private\s*>\s*)?(\w+)\s*:\s*([\w<>]+)\s*;\s*$")
_FOR_RE = re.compile(r"^(\s*)for\s*\(\s*var\s+(\w+)\s*:\s*([\w<>]+)\s*=\s*([^;]+);(.*)$")


def _ty(w):
    if w in _WGSL_TYPE:
        return _WGSL_TYPE[w]
    raise ValueError("wgsl2glsl: unsupported type %r" % w)


def translate(src, types=None, decls=None, buffers=None):
    """WGSL -> GLSL ES 3.00 for the decoder subset.

    `buffers` maps a storage-buffer name to the fetch function that replaces it.

    `types` seeds names already in scope (the GEMV scaffolding's own variables and the
    signatures of the helpers it provides); `decls` collects the types of `var<private>`
    declarations so the caller can hoist them to global scope.
    """
    ty = types.copy() if types else _Types()
    src = _derserve(src)
    out = []
    protos = []
    rw = (lambda e: _rewrite_expr(_rewrite_buffers(e, buffers))) if buffers else _rewrite_expr
    work = list(_statements(src))
    while work:
        raw = work.pop(0)
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("//"):
            out.append(line); continue
        # Bindings are not translated: a storage buffer has no GLSL counterpart, and the
        # assembler declares the sampler that replaces it. Workgroup memory has no
        # counterpart at all, so it is an error rather than something to drop quietly --
        # a shared staging array silently removed still compiles and reads as zeros.
        if s.startswith("@group") or s.startswith("@binding") or "var<storage" in s:
            continue
        if "var<workgroup" in s or "workgroupBarrier" in s:
            raise ValueError("wgsl2glsl: no workgroup memory in GLSL -- %r" % s[:60])

        m = _FN_RE.match(line)
        if m:
            name, params, ret = m.group(1), m.group(2), m.group(3)
            ps = []
            for p in [p for p in params.split(",") if p.strip()]:
                pn, pt = [q.strip() for q in p.split(":")]
                ps.append("%s %s" % (_ty(pt), pn))
                ty.var[pn] = _ty(pt)
            rt = _ty(ret) if ret else "void"
            ty.fn[name] = rt
            # WGSL lets a module-scope function call one declared later -- `SGN4` is written
            # above `SGN` in the i-quant helpers -- and GLSL does not. Emitting a prototype
            # for every function makes the order in the source irrelevant, which is better
            # than reordering: a reordering has to understand the call graph to be right.
            protos.append("%s %s(%s);" % (rt, name, ", ".join(ps)))
            out.append("%s %s(%s) {" % (rt, name, ", ".join(ps)))
            # A one-line helper carries its whole body after the brace. Emitting only the
            # header dropped it, and a function whose body is gone still compiles -- it
            # returns nothing and every value that came through it reads as zero.
            rest = line[m.end():].strip()
            if rest:
                work[:0] = _statements(rest)
            continue

        m = _DECL_RE.match(line)
        if m:
            t = _ty(m.group(2))
            ty.var[m.group(1)] = t
            if decls is not None and "private" in line:
                decls.append((t, m.group(1))); out.append("")
            else:
                out.append("%s%s %s;" % (line[:len(line) - len(line.lstrip())], t, m.group(1)))
            continue

        m = _LET_RE.match(line)
        if m:
            kw, name, decl, expr = m.groups()
            if decl:
                t = _ty(decl)
            else:
                t, _ = _infer(_tokens(expr), ty)
                if t == ABSTRACT:
                    t = I
            ty.var[name] = t
            ind = line[:len(line) - len(line.lstrip())]
            if decls is not None and "private" in line:
                decls.append((t, name))
                out.append("%s%s = %s;" % (ind, name, rw(expr)))
            else:
                out.append("%s%s %s = %s;" % (ind, t, name, rw(expr)))
            continue

        m = _FOR_RE.match(line)
        if m:
            ind, name, dt, init, rest = m.groups()
            ty.var[name] = _ty(dt)
            out.append("%sfor (%s %s = %s;%s" % (ind, _ty(dt), name,
                                                 rw(init), rw(rest)))
            continue

        out.append(rw(line))
    return "\n".join(protos + out)


def _statements(src):
    """Logical statements, one per returned string.

    Two things the line-by-line form got wrong on the real decoders. A `return vec2<f32>(a,
    b)` is routinely split across two lines, and rewriting half of it counts parentheses
    that do not balance. And several lines carry two statements (`let so = o + 4u; let qo =
    o + 16u;`), so a regex anchored to the end of the line captures the second one as part
    of the first one's expression. Joining by depth and then splitting on top-level `;`
    handles both, and leaves comments and braces where they were.
    """
    out, buf, depth = [], "", 0
    for raw in src.split("\n"):
        code = re.sub(r"//.*$", "", raw)
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("//")):
            out.append(raw); continue
        buf = (buf + " " + raw.strip()) if buf else raw
        depth += code.count("(") - code.count(")")
        if depth > 0:
            continue                       # expression still open; keep gathering
        depth = 0
        # split on `;` that is not inside parentheses
        piece, d, parts = "", 0, []
        for ch in buf:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
            if ch == ";" and d == 0:
                parts.append(piece + ";"); piece = ""
            else:
                piece += ch
        if piece.strip():
            parts.append(piece)
        ind = buf[:len(buf) - len(buf.lstrip())]
        for i, pp in enumerate(parts):
            out.append(pp if i == 0 else ind + pp.strip())
        buf = ""
    if buf:
        out.append(buf)
    return out


# GLSL has no `unpack4x8unorm`; the decoders use it to pull four bytes out of a word.
GLSL_PRELUDE = """
vec4 unpack4x8unorm(uint v) {
  return vec4(float(v & 255u), float((v >> 8u) & 255u),
              float((v >> 16u) & 255u), float((v >> 24u) & 255u)) * 0.00392156862745098;
}
"""
