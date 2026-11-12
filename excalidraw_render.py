"""Render Excalidraw sketches to PNG so they can be sent on Discord.

`Excalidraw_Interface` produces `.excalidraw` JSON, which Discord cannot
display. This draws those elements with matplotlib instead: rectangles,
diamonds and ellipses as shapes, text as labels, and arrows as lines with
arrowheads.

Colours are remapped for Discord's dark theme, so a sketch that would be
black-on-white becomes light-on-dark.
"""

import os

# Excalidraw's named palette -> hex.
EXCALIDRAW_COLORS = {
    "black": "#1e1e1e",
    "red": "#e03131",
    "pink": "#c2255c",
    "violet": "#6741d9",
    "blue": "#1971c2",
    "cyan": "#0c8599",
    "green": "#2f9e44",
    "yellow": "#f08c00",
    "orange": "#e8590c",
    "gray": "#868e96",
    "brown": "#8f5c2c",
    "light": "#f8f9fa",
    "white": "#ffffff",
}

# On Discord's dark background, near-black strokes are invisible.
_DARK_BG_STROKE = "#e9ecef"
_DARK_BG_TEXT = "#f1f3f5"
_DARK_BG = "#313338"


def _to_hex(color, default="#e9ecef"):
    """Normalise an Excalidraw colour to a hex string."""
    if not color or color == "transparent":
        return None
    if color.startswith("#"):
        # Very dark colours are lifted so they show on Discord's dark theme.
        if color.lower() in ("#000000", "#1e1e1e"):
            return _DARK_BG_STROKE
        return color
    return EXCALIDRAW_COLORS.get(color.lower(), default)


def _shape_centre(el):
    return el["x"] + el["width"] / 2, el["y"] + el["height"] / 2


def _bound_point(el, towards):
    """The point on a shape's edge closest to `towards` (centre or point)."""
    x, y = el["x"], el["y"]
    w, h = el["width"], el["height"]
    if towards is None:
        return x + w / 2, y + h / 2
    tx, ty = towards
    cx, cy = x + w / 2, y + h / 2
    if tx == cx and ty == cy:
        return cx, cy
    # Scale the direction vector so it reaches the rectangle border.
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def render_excalidraw(data: dict, path: str, background: str = _DARK_BG,
                      scale: float = 1.6) -> bool:
    """Render Excalidraw JSON to a PNG. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

        elements = [e for e in data.get("elements", []) if not e.get("isDeleted")]
        if not elements:
            return False

        # Bounds
        min_x = min(e["x"] for e in elements)
        min_y = min(e["y"] for e in elements)
        max_x = max(e["x"] + e.get("width", 0) for e in elements)
        max_y = max(e["y"] + e.get("height", 0) for e in elements)
        pad = 30
        w = (max_x - min_x) + pad * 2
        h = (max_y - min_y) + pad * 2

        fig = plt.figure(figsize=(w / 60, h / 60), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(min_x - pad, max_x + pad)
        ax.set_ylim(max_y + pad, min_y - pad)  # screen coords, flipped
        ax.axis("off")
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)

        by_id = {e["id"]: e for e in elements}

        # Arrows first so shapes sit on top of their edges.
        for el in elements:
            if el["type"] not in ("arrow", "line"):
                continue
            points = el.get("points") or [[0, 0]]
            pts = [(el["x"] + px, el["y"] + py) for px, py in points]

            # Trim the endpoints to the bound shapes so arrows do not poke
            # through the boxes they connect.
            start_shape = el.get("startBinding", {}).get("elementId")
            end_shape = el.get("endBinding", {}).get("elementId")
            if len(pts) >= 2:
                if start_shape in by_id:
                    pts[0] = _bound_point(by_id[start_shape], pts[1])
                if end_shape in by_id:
                    pts[-1] = _bound_point(by_id[end_shape], pts[-2])

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            stroke = _to_hex(el.get("strokeColor")) or _DARK_BG_STROKE
            style = "-" if el.get("strokeStyle") == "solid" else "--"

            if el["type"] == "arrow":
                ax.annotate(
                    "",
                    xy=pts[-1], xytext=pts[-2] if len(pts) >= 2 else pts[0],
                    arrowprops=dict(arrowstyle="-|>", color=stroke, lw=1.8,
                                    shrinkA=0, shrinkB=0, linestyle=style),
                )
                if len(pts) > 2:
                    ax.plot(xs[:-1], ys[:-1], color=stroke, lw=1.8, linestyle=style)
            else:
                ax.plot(xs, ys, color=stroke, lw=1.8, linestyle=style)

            # Edge label
            label = (el.get("text") or "").strip()
            if label:
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                ax.text(mx, my, label, fontsize=11 * scale, color=_DARK_BG_TEXT,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc=background, ec="none"))

        # Shapes
        for el in elements:
            t = el["type"]
            if t not in ("rectangle", "diamond", "ellipse"):
                continue
            stroke = _to_hex(el.get("strokeColor")) or _DARK_BG_STROKE
            fill = _to_hex(el.get("backgroundColor"))
            x, y = el["x"], el["y"]
            w, h = el["width"], el["height"]

            if t == "rectangle":
                patch = FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0,rounding_size=8",
                    linewidth=1.8, edgecolor=stroke,
                    facecolor=fill if fill else "none",
                )
            elif t == "ellipse":
                patch = Ellipse((x + w / 2, y + h / 2), w, h,
                                linewidth=1.8, edgecolor=stroke,
                                facecolor=fill if fill else "none")
            else:  # diamond
                patch = Polygon(
                    [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)],
                    closed=True, linewidth=1.8, edgecolor=stroke,
                    facecolor=fill if fill else "none",
                )
            ax.add_patch(patch)

        # Text
        for el in elements:
            if el["type"] != "text":
                continue
            text = (el.get("text") or "").strip()
            if not text:
                continue
            container = el.get("containerId")
            colour = _DARK_BG_TEXT
            if container and container in by_id:
                cx, cy = _shape_centre(by_id[container])
            else:
                cx = el["x"] + el.get("width", 0) / 2
                cy = el["y"] + el.get("height", 0) / 2

            fs = (el.get("fontSize") or 20) * scale * 0.62
            ax.text(cx, cy, text, fontsize=fs, color=colour,
                    ha="center", va="center", wrap=True)

        fig.savefig(path, dpi=200, facecolor=background, bbox_inches="tight",
                    pad_inches=0.1)
        plt.close(fig)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False
