"""LaTeX to PNG, typeset properly with KaTeX and MathJax.

matplotlib's mathtext is a small subset of LaTeX: it silently renders
unsupported commands as literal text, so images came out full of
"\\begin{aligned}" instead of equations. This module uses the real thing:

* KaTeX  - parses and validates the LaTeX, so broken expressions are caught
* MathJax - typesets it, correctly laying out aligned/matrix/cases etc.

If node or the npm packages are unavailable, it falls back to matplotlib's
mathtext so maths still renders (just less well).
"""

import json
import os
import shutil
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "render")
SCRIPT = os.path.join(RENDER_DIR, "render_math.js")

# Colours that match Discord's dark theme.
BACKGROUND = "#313338"
FOREGROUND = "#f1f3f5"

# Where node might live.
NODE_CANDIDATES = [
    os.environ.get("NODE_BINARY"),
    shutil.which("node"),
    "/usr/local/bin/node",
    "/usr/bin/node",
    "/opt/homebrew/bin/node",
    os.path.expanduser("~/.nvm/current/bin/node"),
    "/usr/local/nvm/current/bin/node",
]

_render_lock = threading.Lock()
_proc = None
_node = None
_modules = None


def _find_node():
    global _node, _modules
    if _node is not None:
        return _node
    for candidate in NODE_CANDIDATES:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            _node = candidate
            break
    if _node is None:
        return None
    # The npm packages live next to the script (or in a parent node_modules).
    for base in (RENDER_DIR, HERE, os.path.dirname(HERE)):
        modules = os.path.join(base, "node_modules")
        if os.path.isdir(os.path.join(modules, "katex")):
            _modules = modules
            break
    return _node


def _start_process():
    global _proc
    node = _find_node()
    if node is None:
        raise RuntimeError("node is not installed")

    env = dict(os.environ)
    if _modules:
        env["NODE_PATH"] = _modules
    env["PATH"] = os.path.dirname(node) + os.pathsep + env.get("PATH", "")

    _proc = subprocess.Popen(
        [node, SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, env=env, cwd=RENDER_DIR,
    )


def _render_svg(latex: str) -> str:
    """Render LaTeX to an SVG string using the node renderer."""
    global _proc
    with _render_lock:
        if _proc is None or _proc.poll() is not None:
            _start_process()
        try:
            _proc.stdin.write(json.dumps({"latex": latex}) + "\n")
            _proc.stdin.flush()
            line = _proc.stdout.readline()
        except Exception as exc:
            _proc = None
            raise RuntimeError(f"renderer process failed: {exc}")
    if not line:
        _proc = None
        raise RuntimeError("renderer produced no output")
    out = json.loads(line)
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["svg"]


def _svg_to_png(svg: str, path: str) -> bool:
    """Rasterise SVG to PNG on Discord's dark background."""
    svg = svg.replace("currentColor", FOREGROUND)
    start = svg.find("<svg")
    end = svg.rfind("</svg>")
    if start == -1 or end == -1:
        raise RuntimeError("no svg element in renderer output")
    inner = svg[start:end + 6]
    if "xmlns=" not in inner[:200]:
        inner = inner.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)

    try:
        import cairosvg
        cairosvg.svg2png(bytestring=inner.encode(), write_to=path,
                         background_color=BACKGROUND, output_width=1400)
    except ImportError as exc:
        raise RuntimeError(f"cairosvg is not installed: {exc}")
    return os.path.exists(path) and os.path.getsize(path) > 0


def render_latex_png(latex: str, path: str) -> bool:
    """Render LaTeX to a PNG. Returns True on success.

    Falls back to matplotlib's mathtext if node, the npm packages, or
    cairosvg are unavailable.
    """
    latex = (latex or "").strip()
    if not latex:
        return False

    try:
        svg = _render_svg(latex)
        return _svg_to_png(svg, path)
    except Exception as exc:
        print(f"katex/mathjax render failed ({exc}); falling back to mathtext")

    # Fallback: matplotlib mathtext handles simple expressions.
    try:
        from mathrender import render_latex_png as mathtext_render

        return mathtext_render(latex, path)
    except Exception:
        return False


def renderer_available() -> bool:
    """Is the proper KaTeX/MathJax renderer usable on this host?"""
    return _find_node() is not None and _modules is not None
