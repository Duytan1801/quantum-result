"""Render LaTeX math so it is readable in Discord.

Discord does not render LaTeX in chat messages, so this does two things:

* Inline math (``$...$`` / ``\\(...\\)``) is converted to Unicode text, so
  ``x^2`` becomes ``x²`` and ``\\frac{a}{b}`` becomes ``a/b`` right in the
  message.
* Display math (``$$...$$`` / ``\\[...\\]``) is rendered to a PNG with
  matplotlib and sent as an attachment.

Fenced code blocks are left completely alone - Python source in a code block
may legitimately contain ``$`` or ``\\frac``.
"""

import os
import re

# ---------------------------------------------------------------------------
# LaTeX -> Unicode
# ---------------------------------------------------------------------------
_SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
    "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ",
    "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
    "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
}

_SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ",
    "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ",
    "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

# LaTeX commands that map to a single Unicode character.
_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "omicron": "ο", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "ϕ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "simeq": "≃",
    "propto": "∝", "ll": "≪", "gg": "≫",
    "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀",
    "exists": "∃", "nexists": "∄", "in": "∈", "notin": "∉",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "sum": "∑", "prod": "∏", "int": "∫", "iint": "∬", "oint": "∮",
    "sqrt": "√", "cdot": "·", "cdots": "⋯", "ldots": "…",
    "dots": "…", "vdots": "⋮", "ddots": "⋱",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "gets": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "Leftrightarrow": "⇔", "mapsto": "↦", "implies": "⟹",
    "longrightarrow": "⟶", "longleftarrow": "⟵",
    "longleftrightarrow": "⟷", "Longrightarrow": "⟹",
    "Longleftarrow": "⟸", "Longleftrightarrow": "⟺",
    "longmapsto": "⟼", "hookrightarrow": "↪", "hookleftarrow": "↩",
    "uparrow": "↑", "downarrow": "↓", "updownarrow": "↕",
    "nearrow": "↗", "searrow": "↘", "swarrow": "↙", "nwarrow": "↖",
    "angle": "∠", "perp": "⊥", "parallel": "∥", "circ": "∘",
    "star": "★", "dagger": "†", "degree": "°", "prime": "′",
    "ell": "ℓ", "hbar": "ℏ", "Re": "ℜ", "Im": "ℑ",
    "aleph": "ℵ", "wp": "℘", "top": "⊤", "bot": "⊥",
    "llcorner": "⌊", "lrcorner": "⌋", "lceil": "⌈", "rceil": "⌉",
    "therefore": "∴", "because": "∵", "checkmark": "✓",
    "notin": "∉", "land": "∧", "lor": "∨", "neg": "¬",
    "mid": "∣", "parallel": "∥", "bigoplus": "⊕", "otimes": "⊗",
    "langle": "⟨", "rangle": "⟩", "lbrace": "{", "rbrace": "}",
    "lbrack": "[", "rbrack": "]", "backslash": "\\",
    "lvert": "|", "rvert": "|", "lVert": "‖", "rVert": "‖",
    "vert": "|", "Vert": "‖", "|": "‖",
    # Relations models use often that were missing before.
    "iff": "⇔", "cong": "≅", "doteq": "≐", "triangleq": "≜",
    "models": "⊨", "vdash": "⊢", "dashv": "⊣", "asymp": "≍",
    "prec": "≺", "succ": "≻", "preceq": "⪯", "succeq": "⪰",
    "lt": "<", "gt": ">",
    "dotsc": "…", "dotsb": "…", "dotsm": "…", "dotso": "…",
}

# Accents that compose onto a letter. The value is a combining character, so
# \dot{x} becomes "ẋ" and \tilde{A} becomes "Ã" - dropping them used to turn
# "ẋ = Ax" into "x = Ax", which is a different (false) statement.
_ACCENTS = {
    "dot": "\u0307", "ddot": "\u0308", "dddot": "\u20db",
    "tilde": "\u0303", "widetilde": "\u0303",
    "hat": "\u0302", "widehat": "\u0302",
    "bar": "\u0304", "check": "\u030c", "breve": "\u0306",
    "acute": "\u0301", "grave": "\u0300",
    "vec": "\u20d7", "mathring": "\u030a",
}

# Environments whose contents should be shown as a bracketed matrix. The
# fallback used to drop the environment entirely, so
# \begin{pmatrix} A & B \\ C & D \end{pmatrix} became "A B\nC D" with the
# brackets - and the structure - gone.
_MATRIX_ENVS = {
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "smallmatrix",
    "vmatrix", "Vmatrix",
    "array", "cases", "dcases", "rcases",
}

# Small fractions get their proper single glyph: \frac12 -> "½".
_NICE_FRACTIONS = {
    ("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔",
    ("1", "4"): "¼", ("3", "4"): "¾",
    ("1", "5"): "⅕", ("2", "5"): "⅖", ("3", "5"): "⅗", ("4", "5"): "⅘",
    ("1", "6"): "⅙", ("5", "6"): "⅚",
    ("1", "8"): "⅛", ("3", "8"): "⅜", ("5", "8"): "⅝", ("7", "8"): "⅞",
}

# Commands whose braced argument is plain text.
_TEXT_COMMANDS = (
    "text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "operatorname",
    "textbf", "textit", "textrm", "textsf", "texttt", "mathcal", "mathbb",
    "mathfrak", "mathscr", "boldsymbol",
)

# Blackboard-bold and calligraphic letter maps.
_BLACKBOARD = {
    "R": "ℝ", "C": "ℂ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "P": "ℙ",
    "A": "𝔸", "B": "𝔹", "C": "ℂ", "D": "𝔻", "E": "𝔼", "F": "𝔽",
    "G": "𝔾", "H": "ℍ", "I": "𝕀", "J": "𝕁", "K": "𝕂", "L": "𝕃",
    "M": "𝕄", "N": "ℕ", "O": "𝕆", "P": "ℙ", "Q": "ℚ", "R": "ℝ",
    "S": "𝕊", "T": "𝕋", "U": "𝕌", "V": "𝕍", "W": "𝕎", "X": "𝕏",
    "Y": "𝕐", "Z": "ℤ",
}

_CALLIGRAPHIC = {
    "A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ", "F": "ℱ",
    "G": "𝒢", "H": "ℋ", "I": "ℐ", "J": "𝒥", "K": "𝒦", "L": "ℒ",
    "M": "ℳ", "N": "𝒩", "O": "𝒪", "P": "𝒫", "Q": "𝒬", "R": "ℛ",
    "S": "𝒮", "T": "𝒯", "U": "𝒰", "V": "𝒱", "W": "𝒲", "X": "𝒳",
    "Y": "𝒴", "Z": "𝒵",
}


def _match_brace(text: str, start: int):
    """Given text[start] == '{', return (content, index_after_closing_brace)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


def _script_text(value: str, table: dict, prefix: str) -> str:
    """Typeset a super/subscript in plain text.

    Unicode has super/subscript forms for digits and many lowercase letters,
    but not for everything. When the run cannot be mapped (e.g. ``e^{At}``,
    where there is no superscript "A"), fall back to an explicit ``^(At)``
    instead of mapping only part of it - "eAᵗ" reads as e times A times t,
    which is a different expression.
    """
    if not value:
        return ""
    mapped = [table.get(ch) for ch in value]
    if all(m is not None for m in mapped):
        return "".join(mapped)
    if len(value) == 1:
        # A single letter that Unicode cannot raise/lower still has to keep
        # its position: s^\gamma must not become "sgamma" (which reads as
        # s times gamma). Plain punctuation is fine loose: Q^* -> "Q*".
        if value.isalpha():
            return f"{prefix}({value})"
        return value
    return f"{prefix}({value.replace(' ', '')})"


def _read_token(text: str, i: int):
    """Read one LaTeX argument that may be written without braces.

    ``\\frac12`` and ``\\frac\\pi2`` are common model output. Without this
    they collapsed - ``\\frac12`` produced "12" instead of "1/2".
    """
    if i >= len(text):
        return "", i
    if text[i] == "{":
        return _match_brace(text, i)
    if text[i] == "\\":  # a command such as \pi
        j = i + 1
        while j < len(text) and text[j].isalpha():
            j += 1
        if j > i + 1:
            return text[i:j], j
    return text[i], i + 1


def _read_env_body(text: str, i: int, env: str):
    r"""Read the body of an environment that starts at i.

    Returns (body, index_after_end), balancing nested environments of the
    same name.
    """
    opener = f"\\begin{{{env}}}"
    closer = f"\\end{{{env}}}"
    depth = 1
    pos = i
    while pos < len(text):
        end = text.find(closer, pos)
        if end == -1:
            return text[i:], len(text)
        while True:
            begin = text.find(opener, pos)
            if begin != -1 and begin < end:
                depth += 1
                pos = begin + len(opener)
                continue
            break
        depth -= 1
        if depth == 0:
            return text[i:end], end + len(closer)
        pos = end + len(closer)
    return text[i:], len(text)


def _skip_command(text: str, i: int) -> int:
    """Move past a backslash command (or escaped character) starting at i."""
    j = i + 1
    while j < len(text) and text[j].isalpha():
        j += 1
    return j if j > i + 1 else i + 2


def _split_rows(text: str) -> list[str]:
    """Split an environment body on its \\\\ row separators."""
    parts, depth, start = [], 0, 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 < n and text[i + 1] == "\\" and depth == 0:
                row = text[start:i]
                i += 2
                # \\[2mm] is a row-spacing argument, not content.
                if i < n and text[i] == "[":
                    close = text.find("]", i)
                    if close != -1:
                        i = close + 1
                parts.append(row)
                start = i
                continue
            i = _skip_command(text, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    parts.append(text[start:])
    return parts


def _split_cells(text: str) -> list[str]:
    """Split one matrix row on its & column separators."""
    parts, depth, start = [], 0, 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i = _skip_command(text, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return parts


def _matrix_to_text(env: str, body: str) -> str:
    r"""Show a matrix environment as bracketed plain text.

    \begin{pmatrix} A & B \\ C & D \end{pmatrix} becomes "[ A  B ; C  D ]",
    which keeps the structure that used to be lost entirely.
    """
    lines = []
    for row in _split_rows(body):
        cells = [latex_to_unicode(c).strip() for c in _split_cells(row)]
        lines.append("  ".join(c for c in cells if c))
    joined = " ; ".join(line for line in lines if line.strip())
    if env in ("cases", "dcases", "rcases"):
        return "{ " + joined
    if env in ("vmatrix", "Vmatrix"):
        return f"| {joined} |"
    return f"[ {joined} ]"


def latex_to_unicode(latex: str) -> str:
    """Convert a LaTeX math expression to readable Unicode text."""
    if not latex:
        return ""
    out = []
    i = 0
    n = len(latex)

    while i < n:
        ch = latex[i]

        # A doubled backslash is a row separator in an aligned environment.
        # This has to be checked before the escaped-character case below,
        # which would otherwise swallow it and leave a stray backslash.
        if ch == "\\" and latex.startswith("\\\\", i):
            i += 2
            # \\[2mm] is a row-spacing argument, not content; it used to
            # leak into the message as a literal "[2mm]".
            if i < n and latex[i] == "[":
                close = latex.find("]", i)
                if close != -1:
                    i = close + 1
            out.append("\n")
            continue

        # Escaped characters
        if ch == "\\" and i + 1 < n and latex[i + 1] in "\\{}%&_#$":
            out.append(latex[i + 1])
            i += 2
            continue

        # An empty group contributes nothing, but printing it used to leave
        # stray "{}" behind: {}_{C}D_{\alpha} became "{}CDα".
        if ch == "{" and i + 1 < n and latex[i + 1] == "}":
            i += 2
            continue

        if ch == "\\":
            # Read the command name - letters only. A trailing "_" belongs to
            # a subscript, not to the command name.
            j = i + 1
            while j < n and latex[j].isalpha():
                j += 1
            name = latex[i + 1:j]
            i = j

            # Environment markers: drop the name, keep the contents.
            if name in ("begin", "end"):
                if i < n and latex[i] == "{":
                    env, i = _match_brace(latex, i)
                    if name == "begin" and env in _MATRIX_ENVS:
                        if env == "array" and i < n and latex[i] == "{":
                            _colspec, i = _match_brace(latex, i)
                        body, i = _read_env_body(latex, i, env)
                        out.append(_matrix_to_text(env, body))
                continue

            # \frac{a}{b}, \tfrac{a}{b}, \dfrac{a}{b}
            if name in ("frac", "tfrac", "dfrac"):
                num, i = _read_token(latex, i)
                den, i = _read_token(latex, i)
                num_u = latex_to_unicode(num)
                den_u = latex_to_unicode(den)
                if (num_u, den_u) in _NICE_FRACTIONS:
                    out.append(_NICE_FRACTIONS[(num_u, den_u)])
                elif not num_u or not den_u:
                    out.append(f"({num_u})/({den_u})" if num_u or den_u else "")
                elif len(num_u) <= 1 and len(den_u) <= 1:
                    out.append(f"{num_u}/{den_u}")
                else:
                    out.append(f"({num_u})/({den_u})")
                continue

            # \binom{n}{k} - there is no Unicode binomial, so name it.
            if name in ("binom", "tbinom", "dbinom"):
                top, i = _read_token(latex, i)
                bottom, i = _read_token(latex, i)
                out.append(f"C({latex_to_unicode(top)}, {latex_to_unicode(bottom)})")
                continue

            # \sqrt{x} or \sqrt[n]{x}
            if name == "sqrt":
                root = ""
                if i < n and latex[i] == "[":
                    close = latex.find("]", i)
                    root = latex[i + 1:close]
                    i = close + 1
                arg, i = _read_token(latex, i)
                arg_u = latex_to_unicode(arg)
                prefix = f"√[{root}]" if root else "√"
                out.append(f"{prefix}({arg_u})" if len(arg_u) > 1 else f"{prefix}{arg_u}")
                continue

            # \mathbb{R} / \mathcal{L}
            if name in ("mathbb", "mathcal", "mathfrak", "mathscr"):
                arg, i = _match_brace(latex, i) if i < n and latex[i] == "{" else ("", i)
                table = {"mathbb": _BLACKBOARD, "mathcal": _CALLIGRAPHIC}.get(name)
                if table:
                    out.append("".join(table.get(c, c) for c in arg))
                else:
                    out.append(arg)
                continue

            # Text-like commands: keep the content
            if name in _TEXT_COMMANDS:
                arg, i = _match_brace(latex, i) if i < n and latex[i] == "{" else ("", i)
                out.append(arg.replace("\\ ", " "))
                continue

            # Accents: keep the letter underneath. \dot{x} -> ẋ. Dropping the
            # accent used to corrupt the mathematics ("ẋ = Ax" -> "x = Ax").
            if name in _ACCENTS:
                arg, i = _match_brace(latex, i) if i < n and latex[i] == "{" else ("", i)
                arg_u = latex_to_unicode(arg)
                if name == "overline":
                    out.append("".join(c + "\u0305" for c in arg_u))
                elif arg_u:
                    out.append(arg_u + _ACCENTS[name])
                continue

            # Arrows with a label above them. \xrightarrow{d} used to lose
            # the arrow and leave a bare "d": "Zₙ d N(0,1)".
            if name in ("xrightarrow", "xleftarrow"):
                label, i = _read_group(latex, i)
                arrow = "←" if name == "xleftarrow" else "→"
                label_u = _script_text(latex_to_unicode(label), _SUPERSCRIPTS, "^")
                out.append(f"{arrow}{label_u}")
                continue

            if name in ("stackrel", "overset", "underset"):
                first, i = _read_group(latex, i)
                base, i = _read_group(latex, i)
                if name == "underset":
                    out.append(latex_to_unicode(base)
                               + _script_text(latex_to_unicode(first), _SUBSCRIPTS, "_"))
                else:
                    out.append(latex_to_unicode(base)
                               + _script_text(latex_to_unicode(first), _SUPERSCRIPTS, "^"))
                continue

            # Horizontal space. \quad used to vanish entirely, joining its
            # neighbours into one word.
            if name in ("quad", "qquad", "thickspace", "medspace"):
                out.append(" ")
                continue

            # Spacing and sizing commands that contribute nothing.
            if name in ("left", "right", "displaystyle", "limits", "big", "Big",
                        "bigg", "Bigg", "bigl", "Bigl", "bigr", "Bigr",
                        "thinspace", "negthinspace", "mspace",
                        "lbrace", "rbrace", "substack", "not"):
                if name in ("lbrace", "rbrace"):
                    out.append({"lbrace": "{", "rbrace": "}"}[name])
                continue

            # A command we do not know but that takes a braced argument:
            # keep the argument, drop the command name (e.g. \vec{v} -> v).
            if i < n and latex[i] == "{":
                arg, i = _match_brace(latex, i)
                out.append(latex_to_unicode(arg))
                continue

            # Bare spacing macros: \, \; \: \! \{space}
            if name == "" and i < n and latex[i] in ",;! :":
                if latex[i] != "!":
                    out.append(" ")
                i += 1
                continue

            if name in _SYMBOLS:
                out.append(_SYMBOLS[name])
            elif name:
                out.append(name)
            continue

        # Superscript
        if ch == "^":
            arg, i = _read_group(latex, i + 1)
            out.append(_script_text(latex_to_unicode(arg), _SUPERSCRIPTS, "^"))
            continue

        # Subscript
        if ch == "_":
            arg, i = _read_group(latex, i + 1)
            out.append(_script_text(latex_to_unicode(arg), _SUBSCRIPTS, "_"))
            continue

        # Alignment marker
        if ch == "&":
            i += 1
            continue

        out.append(ch)
        i += 1

    # Compose accents into their precomposed forms where one exists, so
    # x + combining dot shows as ẋ.
    import unicodedata

    return unicodedata.normalize("NFC", "".join(out))


def _read_group(text: str, i: int):
    """Read a group starting at i: {...}, a command, or a single character.

    ``s^\\gamma`` used to read only the backslash as the group and emit the
    rest - "gamma" - as ordinary text: "sgamma".
    """
    if i >= len(text):
        return "", i
    if text[i] == "{":
        return _match_brace(text, i)
    if text[i] == "\\":
        j = i + 1
        while j < len(text) and text[j].isalpha():
            j += 1
        if j > i + 1:
            return text[i:j], j
    return text[i], i + 1


# ---------------------------------------------------------------------------
# Rendering display math to PNG
# ---------------------------------------------------------------------------
# Discord's default dark theme background, so a rendered equation blends in
# rather than floating as a white rectangle.
DISCORD_DARK_BG = "#313338"


def render_latex_png(latex: str, path: str, fontsize: int = 22,
                      color: str = "white", background: str = DISCORD_DARK_BG) -> bool:
    """Render a LaTeX expression to a PNG. Returns True on success.

    This is the fallback used when the KaTeX + MathJax renderer is not
    available. matplotlib's mathtext is a small subset of LaTeX and its
    failure mode is silent, so anything it cannot typeset is rejected here
    and the caller falls back to readable Unicode text:

      * an environment (\\begin{aligned}) - mathtext accepts it without
        complaint but draws the LaTeX *source* instead of typesetting it;
      * a newline - mathtext gives up on the whole expression and draws the
        source, including the $ delimiters, as ordinary text;
      * a $ inside the expression - the $$...$$ wrapper would then have an
        odd number of dollars, so matplotlib stops treating it as maths and
        draws it literally.

    A picture of LaTeX source is worse than no picture at all.
    """
    if "\\begin{" in latex:
        return False
    if "\n" in latex or "\r" in latex:
        return False
    if "$" in latex:
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import mathtext

        # Validate first so a bad expression fails fast.
        mathtext.MathTextParser("agg").parse(latex, dpi=100, prop=None)

        fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
        if background:
            fig.patch.set_facecolor(background)
        else:
            fig.patch.set_alpha(0.0)
        fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=color)
        fig.savefig(path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), transparent=not background,
                    pad_inches=0.16)
        plt.close(fig)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Message-level processing
# ---------------------------------------------------------------------------
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$([^$\n]+?)\$(?!\$)|\\\((.+?)\\\)", re.DOTALL)
_ESCAPED_DOLLAR = re.compile(r"\\\$")

# Stand-in for an escaped dollar while inline math is being matched.
_DOLLAR_SENTINEL = "\x00DOLLAR\x00"

# LaTeX environments that models emit for display math. They are matched
# without their surrounding $$ or \[...\] delimiters too, because models
# routinely write a bare \begin{aligned}...\end{aligned} block.
_ENV_TOKEN = re.compile(r"\\(begin|end)\{([A-Za-z]+)\*?\}")


def _find_env_blocks(text: str) -> list[tuple[int, int, str]]:
    r"""Locate \begin{env}...\end{env} blocks, allowing nested environments.

    Returns (start, end, env_name) for each block, in source order.
    """
    blocks = []
    stack: list[tuple[str, int]] = []
    for m in _ENV_TOKEN.finditer(text):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append((env, m.start()))
        else:
            # Match the nearest open environment with the same name.
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx][0] == env:
                    start = stack[idx][1]
                    del stack[idx:]
                    blocks.append((start, m.end(), env))
                    break
    blocks.sort()
    return blocks


def _display_regions(chunk: str) -> list[tuple[int, int, str]]:
    """Every display-math region in a chunk, in source order.

    Both the delimited forms ($$...$$, \\[...\\]) and bare environment blocks
    (\\begin{aligned}...\\end{aligned}) are found. A bare block that already sits
    inside a delimited one is not counted twice.
    """
    regions: list[tuple[int, int, str]] = []

    for m in _DISPLAY_MATH.finditer(chunk):
        latex = m.group(1) if m.group(1) is not None else m.group(2)
        regions.append((m.start(), m.end(), latex.strip()))

    for start, end, _env in _find_env_blocks(chunk):
        if any(s <= start and end <= e for s, e, _ in regions):
            continue  # already covered by a $$ or \[...\] region
        regions.append((start, end, chunk[start:end].strip()))

    regions.sort()
    return regions


def split_code_blocks(text: str):
    """Yield (is_code, chunk) so code blocks can be left untouched."""
    pos = 0
    for m in _CODE_BLOCK.finditer(text):
        if m.start() > pos:
            yield False, text[pos:m.start()]
        yield True, m.group(0)
        pos = m.end()
    if pos < len(text):
        yield False, text[pos:]


def extract_display_math(text: str):
    """Pull display math blocks out, returning (text_without_them, list_of_latex)."""
    blocks = []

    def grab(m):
        latex = m.group(1) if m.group(1) is not None else m.group(2)
        blocks.append(latex.strip())
        return "\n"

    return _DISPLAY_MATH.sub(grab, text), blocks


def _inline_to_unicode(text: str) -> str:
    """Convert inline math to Unicode. Escaped dollars are protected first so a
    literal "$100" is never treated as the start of a formula."""
    text = _ESCAPED_DOLLAR.sub(_DOLLAR_SENTINEL, text)

    def inline_sub(m):
        latex = m.group(1) if m.group(1) is not None else m.group(2)
        return latex_to_unicode(latex)

    text = _INLINE_MATH.sub(inline_sub, text)
    return text.replace(_DOLLAR_SENTINEL, "$")


def process_math(text: str, render_dir: str = None, max_images: int = 5):
    """Convert math in a message for Discord.

    Returns an ordered list of parts, so a rendered equation stays next to the
    text it belongs to instead of all the images piling up at the end.

    Each part is ("text", str) or ("image", path).
    """
    if not text:
        return []

    parts = []

    for is_code, chunk in split_code_blocks(text):
        if is_code:
            parts.append(("text", chunk))
            continue

        # Split the chunk into alternating text and equation sections, keeping
        # the original order.
        pos = 0
        for start, end, latex in _display_regions(chunk):
            if start > pos:
                parts.append(("text", _inline_to_unicode(chunk[pos:start])))

            image = None
            made = sum(1 for p in parts if p[0] == "image")
            if render_dir and made < max_images:
                os.makedirs(render_dir, exist_ok=True)
                path = os.path.join(render_dir, f"equation_{made + 1}.png")
                try:
                    from latex_render import render_latex_png

                    if render_latex_png(latex, path):
                        image = path
                except Exception as exc:
                    print(f"could not render equation: {exc}")

            if image is not None:
                parts.append(("image", image))
            else:
                # Fall back to readable Unicode text.
                parts.append(("text", f"\n{latex_to_unicode(latex)}\n"))
            pos = end

        if pos < len(chunk):
            parts.append(("text", _inline_to_unicode(chunk[pos:])))

    return parts
