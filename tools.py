"""Tool definitions and implementations.

Each tool is an async function plus a JSON-schema description the model sees.
"""

import contextvars
import os

import httpx

from config import (
    LANGSEARCH_API_KEY,
    LANGSEARCH_API_URL,
    WOLFRAM_APP_ID,
    WOLFRAM_API_URL,
)

# ---------------------------------------------------------------------------
# File system workspace (sandboxed)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.join(os.getcwd(), "bot_files")
os.makedirs(BASE_DIR, exist_ok=True)


def safe_path(file_path: str) -> str:
    """Resolve and validate that the path is within BASE_DIR."""
    full_path = os.path.realpath(os.path.join(BASE_DIR, file_path))
    base_real = os.path.realpath(BASE_DIR)
    if full_path != base_real and not full_path.startswith(base_real + os.sep):
        raise ValueError(f"Access denied: {file_path} is outside the workspace")
    return full_path


# Files written during the current agent run. Used so a long job can attach
# the artifacts it produced to its final Discord message.
written_files: contextvars.ContextVar = contextvars.ContextVar("written_files", default=None)


def track_file(path: str):
    """Record a file written by a tool, if we are inside a tracked run."""
    tracker = written_files.get()
    if tracker is not None:
        tracker.append(path)


# ---------------------------------------------------------------------------
# Wolfram Alpha
# ---------------------------------------------------------------------------
_WA_SKIP_PODS = {
    "number line", "manipulatives illustration", "visual representation of the integral",
    "plots", "plot", "diagram", "web results", "interesting facts", "wikipedia entry",
    "related questions", "time plot", "polar plot", "3d plot", "contour plot",
}


def _wa_pod_text(pod: dict) -> str:
    """Concatenate the plaintext subpods of a Wolfram Alpha pod."""
    parts = []
    for sub in pod.get("subpods", []) or []:
        text = (sub.get("plaintext") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


async def wolfram_alpha(query: str) -> str:
    """Query the Wolfram Alpha computational engine and format the result."""
    if not WOLFRAM_APP_ID:
        return "Error: WOLFRAM_APP_ID is not configured."

    params = {"input": query, "appid": WOLFRAM_APP_ID, "output": "json"}

    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            # Wolfram Alpha rejects the default httpx User-Agent with a 403,
            # so send a normal-looking one.
            resp = await client.get(
                WOLFRAM_API_URL,
                params=params,
                headers={"User-Agent": "Mozilla/5.0 (compatible; DiscordBot/1.0)"},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Wolfram Alpha API error ({e.response.status_code}): {e.response.text[:300]}"
        except Exception as exc:
            return f"Wolfram Alpha error: {exc}"

    result = data.get("queryresult", {})

    if not result.get("success"):
        # Surface the "did you mean" hints so the model can retry.
        tips = result.get("tips")
        if tips:
            return "Wolfram Alpha could not evaluate that query. Suggestions: " + "; ".join(tips)
        return "Wolfram Alpha could not evaluate that query. Try rephrasing it more specifically."

    lines = []

    # Show how the query was interpreted first - this is often the key detail.
    for pod in result.get("pods", []) or []:
        if (pod.get("title") or "").lower() == "input interpretation":
            text = _wa_pod_text(pod)
            if text:
                lines.append(f"Interpreted as: {text}")
            break

    # Primary results carry the actual answer.
    primary = []
    secondary = []
    for pod in result.get("pods", []) or []:
        title = (pod.get("title") or "").strip()
        if title.lower() in _WA_SKIP_PODS:
            continue
        text = _wa_pod_text(pod)
        if not text:
            continue
        (primary if pod.get("primary") else secondary).append((title, text))

    for title, text in primary:
        lines.append(f"\n{title}:\n{text}")

    # A couple of secondary pods add useful context without bloating context.
    for title, text in secondary[:2]:
        lines.append(f"\n{title}:\n{text}")

    if not lines:
        return "Wolfram Alpha returned no readable result for that query."

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
async def web_search(query: str, count: int = 5, freshness: str = "noLimit",
                     deep: bool = True) -> str:
    """Search the web via LangSearch and format the results."""
    if not LANGSEARCH_API_KEY:
        return "Error: LANGSEARCH_API_KEY not configured."

    payload = {
        "query": query,
        "count": min(max(count, 1), 10),
        "freshness": freshness,
    }
    if deep:
        # Ask for page text, not just snippets. Snippets alone are too thin
        # for detailed research questions (e.g. specific financial figures).
        payload["contents"] = {"text": {"maxCharacters": 3000}}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                LANGSEARCH_API_URL,
                headers={
                    "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"LangSearch API error ({e.response.status_code}): {e.response.text}"
        except Exception as exc:
            return f"LangSearch error: {exc}"

    sources = data.get("data", {}).get("webPages", {}).get("value", [])
    if not sources:
        return "No results found."

    lines = []
    for src in sources:
        name = src.get("name", "")
        url = src.get("url", "")
        # "text" replaces "snippet" when contents.text is requested.
        body = src.get("text") or src.get("snippet", "")
        lines.append(f"### {name}\n{url}\n{body}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Figure verification
# ---------------------------------------------------------------------------
# The model must not assert arithmetic it has not actually performed. This
# evaluates the expressions behind its claims so the numbers can be checked
# against a real engine rather than trusted from the model's head.
#
# Expressions are evaluated with a whitelisted AST walker - no names, no
# attribute access, no arbitrary calls - so nothing can escape the sandbox.

import ast
import math

_SAFE_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "fabs": math.fabs, "exp": math.exp, "log": math.log,
    "log2": math.log2, "log10": math.log10, "pow": pow, "min": min, "max": max,
    "sum": sum, "factorial": math.factorial, "gcd": math.gcd, "log1p": math.log1p,
}

_SAFE_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_MAX_POW = 10000  # stop 10**1000000 style blow-ups


def _safe_eval(expr: str):
    """Evaluate an arithmetic expression using only whitelisted operations."""
    tree = ast.parse(expr, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numbers are allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
                raise ValueError("exponent too large")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            operand = walk(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError("unary operator not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise ValueError("function not allowed")
            if node.keywords:
                raise ValueError("keyword arguments not allowed")
            return _SAFE_FUNCS[node.func.id](*[walk(a) for a in node.args])
        if isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTS:
                return _SAFE_CONSTS[node.id]
            raise ValueError(f"name '{node.id}' is not allowed")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        raise ValueError(f"{type(node).__name__} not allowed")

    return walk(tree)


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"
    return str(value)


async def verify_claims(checks: list) -> str:
    """Recompute the arithmetic behind claimed figures and report mismatches.

    Each check is {"expression": "<arithmetic>", "claimed": "<value>"} where
    `claimed` is what the answer asserts. The expression is evaluated here and
    compared, so any figure the model states can be proven rather than trusted.
    """
    if not checks:
        return "Error: no checks supplied."

    lines = []
    mismatches = 0
    for i, check in enumerate(checks, 1):
        if not isinstance(check, dict):
            lines.append(f"{i}. ERROR: check must be an object")
            mismatches += 1
            continue
        expr = str(check.get("expression", "")).strip()
        claimed = check.get("claimed")
        if not expr:
            lines.append(f"{i}. ERROR: missing expression")
            mismatches += 1
            continue
        try:
            actual = _safe_eval(expr)
        except Exception as exc:
            lines.append(f"{i}. ERROR: could not evaluate {expr!r}: {exc}")
            mismatches += 1
            continue

        actual_s = _fmt(actual)
        if claimed is None:
            lines.append(f"{i}. OK  {expr} = {actual_s}")
            continue
        claimed_s = str(claimed).strip()
        try:
            claimed_num = float(claimed_s.replace(",", "").replace("$", "").rstrip("%"))
        except ValueError:
            lines.append(f"{i}. OK  {expr} = {actual_s} (claimed: {claimed_s})")
            continue
        try:
            actual_num = float(actual_s)
        except (TypeError, ValueError):
            lines.append(f"{i}. OK  {expr} = {actual_s} (claimed: {claimed_s})")
            continue

        # Allow a small relative tolerance for rounding in the claim.
        tolerance = max(abs(claimed_num) * 0.005, 0.005)
        if abs(actual_num - claimed_num) <= tolerance:
            lines.append(f"{i}. OK  {expr} = {actual_s} (claimed {claimed_s})")
        else:
            mismatches += 1
            lines.append(
                f"{i}. MISMATCH  {expr} = {actual_s}, but the answer claims "
                f"{claimed_s}. Fix the number before presenting it."
            )

    header = f"Recomputed {len(checks)} figure(s); {mismatches} mismatch(es)."
    return header + "\n" + "\n".join(lines)


async def draw_diagram(spec: dict, filename: str = "diagram.png") -> str:
    """Draw a diagram and save it as a PNG in the workspace.

    `spec` is {"title": ..., "nodes": [...], "edges": [...]}. The layout is
    computed from each node's row/column, so the caller never has to work out
    coordinates.
    """
    if not isinstance(spec, dict):
        return "Error: draw_diagram needs a specification object."
    nodes = spec.get("nodes")
    if not nodes:
        return "Error: a diagram needs at least one node."

    name = os.path.basename(str(filename or "diagram.png"))
    if not name.lower().endswith(".png"):
        name += ".png"
    path = safe_path(name)

    try:
        from diagrams import diagram_to_png

        diagram_to_png(spec, path)
    except Exception as exc:
        return f"Error: could not draw the diagram: {exc}"

    track_file(path)
    return (
        f"Drew the diagram '{name}' ({len(nodes)} nodes) and saved it in the "
        f"workspace. It will be attached to your reply automatically."
    )


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------
async def read_file(file_path: str) -> str:
    """Read the contents of a file from the workspace."""
    try:
        path = safe_path(file_path)
    except ValueError as e:
        return str(e)
    if not os.path.exists(path):
        return f"Error: File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"Error reading file: {exc}"


async def write_file(file_path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates/overwrites the file."""
    try:
        path = safe_path(file_path)
    except ValueError as e:
        return str(e)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        track_file(path)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


async def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string (first occurrence only)."""
    try:
        path = safe_path(file_path)
    except ValueError as e:
        return str(e)
    if not os.path.exists(path):
        return f"Error: File not found: {file_path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return f"Error: old_string not found in {file_path}"
        new_content = content.replace(old_string, new_string, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        track_file(path)
        return f"Successfully edited {file_path}"
    except Exception as exc:
        return f"Error editing file: {exc}"


# ---------------------------------------------------------------------------
# Tool schemas shown to the model
# ---------------------------------------------------------------------------
# Fill colours a diagram node may use.
VALID_COLORS = (
    "red", "pink", "violet", "blue", "cyan", "green", "yellow", "orange",
    "gray", "brown", "black",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wolfram_alpha",
            "description": (
                "Compute with WolframAlpha's computational engine. Use this FIRST for any "
                "math, algebra, calculus, statistics, physics, chemistry, unit conversion, "
                "or date question: derivatives, integrals, equations, simplification, "
                "factoring, matrices, probabilities, conversions. Prefer this over writing "
                "code for arithmetic and symbolic math. Returns the primary result and "
                "related details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The query in natural language or Wolfram Language notation. "
                            "Be specific: 'integrate x^2 from 0 to 3', 'simplify (x^2-1)/ln(x)', "
                            "'solve x^2 + 3x - 4 = 0', '1500 kg to pounds'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": (
                "Execute code in a sandbox. Use for data processing, string/file work, or "
                "algorithms. For pure math and symbolic computation, use the wolfram_alpha "
                "tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The code to execute"},
                    "language": {"type": "string", "description": "Programming language (python, javascript, bash)", "default": "python"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, facts, or anything not in your training data. Returns titles, URLs, and the page text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Number of results (default 5, max 10)", "default": 5},
                    "freshness": {"type": "string", "description": "Time filter: 'noLimit' (default), 'oneDay', 'oneWeek', 'oneMonth', 'oneYear', or a date range 'YYYY-MM-DD..YYYY-MM-DD'", "default": "noLimit"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to read (relative to workspace)"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace. Creates or overwrites the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to write (relative to workspace)"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old_string with new_string (first occurrence only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file to edit"},
                    "old_string": {"type": "string", "description": "Text to find and replace"},
                    "new_string": {"type": "string", "description": "Text to replace old_string with"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_diagram",
            "description": (
                "Draw a diagram - a flowchart, architecture sketch, pipeline, or "
                "concept map - and send it as an image. Describe the nodes and the "
                "connections between them; the layout is worked out for you, so you "
                "only need to say what each box says and how the arrows run. Use this "
                "whenever a picture would explain something better than prose."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Optional title shown above the diagram.",
                    },
                    "nodes": {
                        "type": "array",
                        "description": "The boxes in the diagram.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": (
                                        "A short unique id for this node, used by the "
                                        "edges to refer to it (e.g. 'start', 'search')."
                                    ),
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The label inside the node.",
                                },
                                "shape": {
                                    "type": "string",
                                    "enum": ["rectangle", "diamond", "ellipse"],
                                    "description": (
                                        "rectangle for a step, diamond for a decision, "
                                        "ellipse for a start/end."
                                    ),
                                    "default": "rectangle",
                                },
                                "row": {
                                    "type": "integer",
                                    "description": (
                                        "Vertical position, 0 for the top. Nodes in "
                                        "the same row sit side by side."
                                    ),
                                },
                                "col": {
                                    "type": "integer",
                                    "description": (
                                        "Horizontal position. Use different columns to "
                                        "place nodes side by side; negative columns "
                                        "are allowed."
                                    ),
                                },
                                "color": {
                                    "type": "string",
                                    "enum": list(VALID_COLORS),
                                    "description": "Fill colour of the node.",
                                },
                            },
                            "required": ["id", "text"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "description": "The arrows between nodes.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string", "description": "Node id to draw from."},
                                "to": {"type": "string", "description": "Node id to draw to."},
                                "label": {"type": "string", "description": "Optional text on the arrow."},
                                "style": {
                                    "type": "string",
                                    "enum": ["solid", "dashed"],
                                    "description": "Line style of the arrow.",
                                },
                                "double": {
                                    "type": "boolean",
                                    "description": "Draw arrows at both ends.",
                                },
                            },
                            "required": ["from", "to"],
                        },
                    },
                },
                "required": ["nodes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_claims",
            "description": (
                "Recompute the arithmetic behind figures you are about to state and "
                "check them against what you claimed. Each check needs the expression "
                "that produces the number (e.g. '(127.6 - 90.2) / 90.2 * 100') and the "
                "value you intend to publish (e.g. '41.6'). Use this for EVERY derived "
                "figure you are about to give - growth rates, percentages, totals, "
                "differences, ratios, sums, averages - and for any figure you computed "
                "mentally rather than reading it off a source. It returns a pass/mismatch "
                "verdict per figure so you can correct anything that does not add up. "
                "Do not present a derived number you have not run through this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "description": "The figures to recompute and check.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": (
                                        "Arithmetic expression producing the figure, "
                                        "e.g. '(127.6 - 90.2) / 90.2 * 100'. Plain "
                                        "arithmetic only - numbers, + - * / // % **, "
                                        "parentheses, and functions like sqrt/log/abs."
                                    ),
                                },
                                "claimed": {
                                    "type": "string",
                                    "description": "The value you intend to publish.",
                                },
                            },
                            "required": ["expression", "claimed"],
                        },
                    },
                },
                "required": ["checks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_agent_task",
            "description": (
                "Delegate a task to a specialised background agent. The agent runs "
                "in the background while the user keeps chatting, reports its "
                "progress, and posts its result (with any files it produced) when "
                "it finishes.\n"
                "Pick the agent that fits the work:\n"
                "AGENTS:\n"
                "{AGENTS}\n"
                "Use start_agent_task when the task needs several tools, several "
                "sources, or a substantial deliverable - not just a single lookup. "
                "A specialist with its own budget and workspace will do it better "
                "and more thoroughly than you can in one turn, so prefer delegating "
                "when it fits. Do not duplicate the work yourself once delegated.\n"
                "If the work splits into more than one piece, call this tool once "
                "per piece in the same turn - the agents then run in parallel.\n"
                "After starting a task, reply with one short line naming the agent "
                "you started and saying results will be posted here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["research", "analyze", "code", "explore"],
                        "description": "Which specialist agent should handle this task.",
                    },
                    "topic": {
                        "type": "string",
                        "description": (
                            "A precise, self-contained description of the task, "
                            "including the specific questions to answer, the "
                            "constraints, and what the deliverable should be."
                        ),
                    },
                },
                "required": ["agent", "topic"],
            },
        },
    },
]

def build_tool_list() -> list:
    """Assemble the full tool list, filling in the agent catalogue.

    Kept as a function (rather than a module constant) so the catalogue can be
    built from the registry without a circular import.
    """
    from agents import AGENTS, agent_catalogue  # local import avoids a cycle

    delegation = next(t for t in TOOLS if t["function"]["name"] == "start_agent_task")
    fn = delegation["function"]
    fn["description"] = fn["description"].replace("{AGENTS}", agent_catalogue())
    fn["parameters"]["properties"]["agent"]["enum"] = list(AGENTS)
    return TOOLS


# Jobs must not spawn further jobs.
JOB_TOOLS = [t for t in TOOLS if t["function"]["name"] != "start_agent_task"]
