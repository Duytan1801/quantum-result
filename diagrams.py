"""Build Excalidraw diagrams from a simple, layout-free specification.

The model describes a diagram as nodes on a grid plus edges between them; the
coordinates are computed here. That keeps the model away from pixel arithmetic
and produces a tidy, readable layout every time.

The sketch itself is built with `Excalidraw_Interface` and rendered to PNG by
`excalidraw_render` so it can be sent on Discord.
"""

import os

# Grid spacing (centre-to-centre).
COL_SPACING = 300
ROW_SPACING = 170

# Node sizing.
MIN_WIDTH = 150
MAX_WIDTH = 320
NODE_HEIGHT = 62

VALID_SHAPES = ("rectangle", "diamond", "ellipse")
VALID_COLORS = (
    "red", "pink", "violet", "blue", "cyan", "green", "yellow", "orange",
    "gray", "brown", "black",
)


def _node_size(text: str):
    """Width/height for a node, based on the length of its label."""
    longest = max((len(line) for line in (text or "").split("\n")), default=1)
    width = min(MAX_WIDTH, max(MIN_WIDTH, int(longest * 9.5) + 40))
    height = NODE_HEIGHT if "\n" not in (text or "") else NODE_HEIGHT + 18
    return width, height


def build_diagram(spec: dict) -> dict:
    """Build Excalidraw JSON from a spec.

    spec = {
        "title": "optional title",
        "nodes": [{"id": "a", "text": "Start", "shape": "rectangle",
                   "row": 0, "col": 0, "color": "blue"}],
        "edges": [{"from": "a", "to": "b", "label": "yes", "style": "solid"}],
    }
    """
    from Excalidraw_Interface import SketchBuilder

    sb = SketchBuilder()

    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    if not nodes:
        raise ValueError("a diagram needs at least one node")

    # Lay the nodes out on a grid.
    placed = {}
    for index, node in enumerate(nodes):
        node_id = str(node.get("id") or f"n{index}")
        text = str(node.get("text") or node_id)
        shape = str(node.get("shape") or "rectangle").lower()
        if shape not in VALID_SHAPES:
            shape = "rectangle"
        color = str(node.get("color") or "").lower()
        if color not in VALID_COLORS:
            color = ""

        row = node.get("row")
        col = node.get("col")
        row = index if row is None else int(row)
        col = 0 if col is None else int(col)

        cx = col * COL_SPACING
        cy = row * ROW_SPACING
        width, height = _node_size(text)

        # Shape
        shape_kwargs = {}
        if color:
            shape_kwargs["strokeColor"] = color
            shape_kwargs["backgroundColor"] = color
        if shape == "rectangle":
            element = sb.Rectangle(cx, cy, width=width, height=height, **shape_kwargs)
        elif shape == "diamond":
            element = sb.Diamond(cx, cy, width=width, height=height, **shape_kwargs)
        else:
            element = sb.Ellipse(cx, cy, width=width, height=height, **shape_kwargs)

        # Label, centred on the shape.
        label = sb.Text(text, cx, cy, fontSize=16)
        try:
            label._element["containerId"] = element._element["id"]
        except Exception:
            pass

        placed[node_id] = (element, cx, cy, width, height)

    # Title, above the first row.
    title = str(spec.get("title") or "").strip()
    if title:
        sb.Text(title, 0, -ROW_SPACING + 20, fontSize=26)

    # Edges
    for edge in edges:
        src = placed.get(str(edge.get("from")))
        dst = placed.get(str(edge.get("to")))
        if not src or not dst:
            continue
        label = str(edge.get("label") or "").strip()
        style = str(edge.get("style") or "solid").lower()
        edge_kwargs = {}
        if style == "dashed":
            edge_kwargs["strokeStyle"] = "dashed"
        if edge.get("color"):
            edge_kwargs["strokeColor"] = str(edge["color"]).lower()

        arrow = sb.create_binding_arrows(
            src[0], dst[0], sb.DoubleArrow if edge.get("double") else None, **edge_kwargs
        )

        if label:
            # Offset the label perpendicular to the edge so it does not sit
            # on top of the line.
            dx = dst[1] - src[1]
            dy = dst[2] - src[2]
            length = (dx * dx + dy * dy) ** 0.5 or 1.0
            mid_x = (src[1] + dst[1]) / 2 - dy / length * 16
            mid_y = (src[2] + dst[2]) / 2 + dx / length * 16
            sb.Text(label, mid_x, mid_y, fontSize=14)

    return sb.export_to_json()


def diagram_to_png(spec: dict, path: str) -> str:
    """Build a diagram from a spec and render it to a PNG. Returns the path."""
    from excalidraw_render import render_excalidraw

    data = build_diagram(spec)
    if not render_excalidraw(data, path):
        raise RuntimeError("could not render the diagram to PNG")
    return path
