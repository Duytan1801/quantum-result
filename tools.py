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
from sandbox import run_code_in_sandbox

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
