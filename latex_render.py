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
import platform
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(HERE, "render")
SCRIPT = os.path.join(RENDER_DIR, "render_math.js")

# Colours that match Discord's dark theme.
BACKGROUND = "#313338"
FOREGROUND = "#f1f3f5"

# The Node runtime fetched when none is installed. It matches the version
# the render.yaml build command installs, so both paths agree.
NODE_VERSION = "v20.18.0"

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

# One auto-install attempt per process: if nodejs.org is unreachable the bot
# must keep answering, not stall on every message.
_bootstrap_lock = threading.Lock()
_node_bootstrap_done = False
_npm_bootstrap_done = False


def _bootstrap_node():
    """Download a self-contained Node.js runtime when none is installed.

    Render rebuilds the container on every deploy, so node installed by the
    build command only exists when the service actually uses render.yaml.
    Node ships as a single tarball, so it can be fetched at runtime instead -
    and then KaTeX + MathJax work no matter how the service is configured.

    Returns the path to the node binary, or None on any failure.
    """
    global _node_bootstrap_done
    with _bootstrap_lock:
        if _node_bootstrap_done:
            return None
        _node_bootstrap_done = True

    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        return None

    import tarfile
    import tempfile
    import urllib.request

    base = os.path.expanduser(os.path.join("~", ".lain2", "node"))
    dirname = f"node-{NODE_VERSION}-linux-x64"
    node_bin = os.path.join(base, dirname, "bin", "node")

    if not os.path.exists(node_bin):
        url = f"https://nodejs.org/dist/{NODE_VERSION}/{dirname}.tar.xz"
        print(f"latex renderer: no node found; downloading {url} ...")
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                data = resp.read()
            fd, tmp = tempfile.mkstemp(suffix=".tar.xz")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            with tarfile.open(tmp, "r:xz") as tar:
                try:
                    tar.extractall(base, filter="data")
                except TypeError:  # python < 3.12 has no filter argument
                    tar.extractall(base)
            os.remove(tmp)
            os.chmod(node_bin, 0o755)
        except Exception as exc:
            print(f"latex renderer: could not auto-install node: {exc}")
            return None
    print(f"latex renderer: using auto-installed node at {node_bin}")
    return node_bin


def _npm_install():
    """Install the renderer's npm packages if node_modules is missing."""
    global _npm_bootstrap_done
    with _bootstrap_lock:
        if _npm_bootstrap_done:
            return False
        _npm_bootstrap_done = True

    if _node is None:
        return False
    # The node tarball bundles npm; a system node needs npm on the PATH.
    root = os.path.dirname(os.path.dirname(_node))
    npm_cli = os.path.join(root, "lib", "node_modules", "npm", "bin", "npm-cli.js")
    if os.path.exists(npm_cli):
        cmd = [_node, npm_cli]
    else:
        npm = shutil.which("npm")
        if not npm:
            return False
        cmd = [npm]
    print("latex renderer: node_modules missing; running npm install ...")
    try:
        subprocess.run(
            cmd + ["install", "--prefix", RENDER_DIR, "--no-audit", "--no-fund"],
            cwd=RENDER_DIR, timeout=420, capture_output=True,
        )
    except Exception as exc:
        print(f"latex renderer: npm install failed: {exc}")
        return False
    return os.path.isdir(os.path.join(RENDER_DIR, "node_modules", "katex"))


def _find_node():
    global _node, _modules
    if _node is None:
        for candidate in NODE_CANDIDATES:
            if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
                _node = candidate
                break
        if _node is None:
            # Nothing installed: fetch a runtime so maths still renders
            # properly on hosts where the build command never ran.
            _node = _bootstrap_node()
    if _node is None:
        return None
    # The npm packages live next to the script (or in a parent node_modules).
    if _modules is None:
        bases = (RENDER_DIR, HERE, os.path.dirname(HERE))
        for base in bases:
            modules = os.path.join(base, "node_modules")
            if os.path.isdir(os.path.join(modules, "katex")):
                _modules = modules
                break
        if _modules is None and _npm_install():
            for base in bases:
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
