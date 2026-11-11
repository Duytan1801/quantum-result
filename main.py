import os
import asyncio
import json
import contextvars
import time
import discord
from discord.ext import commands
from flask import Flask
from threading import Thread
from supabase import create_client, Client
from openai import AsyncOpenAI
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MAX_HISTORY = 3

# Groq is an optional fallback provider (used when NVIDIA is unavailable).
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Daytona provides the sandboxed code execution backend (replaces Piston,
# whose public API became whitelist-only in February 2026).
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
DAYTONA_API_URL = os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")

_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")
if not _supabase_url or not _supabase_key:
    raise SystemExit(
        "SUPABASE_URL and SUPABASE_KEY must be set (see .env.example). "
        "Copy .env.example to .env and fill in the values."
    )
supabase: Client = create_client(_supabase_url, _supabase_key)

if not NVIDIA_API_KEY and not GROQ_API_KEY:
    raise SystemExit(
        "NVIDIA_API_KEY (or GROQ_API_KEY) must be set. Copy .env.example to .env "
        "and fill in the values."
    )
nvidia_client = AsyncOpenAI(base_url=NVIDIA_API_URL, api_key=NVIDIA_API_KEY or "not-used")
groq_client = AsyncOpenAI(base_url=GROQ_API_URL, api_key=GROQ_API_KEY or "not-used") if GROQ_API_KEY else None

LANGSEARCH_API_URL = os.getenv("LANGSEARCH_API_URL", "https://api.langsearch.com/v1/web-search")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")

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

# ---------------------------------------------------------------------------
# Tools: code execution + web search
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
]

# ---------------------------------------------------------------------------
# Code execution (Daytona sandbox)
# ---------------------------------------------------------------------------
# A single sandbox is reused across calls so the model keeps state (files,
# variables) between tool invocations, and so we don't pay sandbox startup on
# every call. The sandbox is recreated automatically if it dies.
_daytona_client = None
_daytona_sandbox = None
_daytona_lock = asyncio.Lock()


async def _get_daytona_sandbox():
    """Return a live Daytona sandbox, creating one if needed."""
    global _daytona_client, _daytona_sandbox

    if _daytona_client is None:
        from daytona import AsyncDaytona, DaytonaConfig

        _daytona_client = AsyncDaytona(DaytonaConfig())

    if _daytona_sandbox is None:
        _daytona_sandbox = await _daytona_client.create()

    return _daytona_sandbox


async def _reset_daytona_sandbox():
    global _daytona_sandbox
    sandbox, _daytona_sandbox = _daytona_sandbox, None
    if sandbox is not None:
        try:
            await _daytona_client.delete(sandbox)
        except Exception:
            pass


_LANGUAGE_ALIASES = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
}


async def run_code_in_sandbox(code: str, language: str = "python") -> str:
    """Execute code in an isolated Daytona sandbox and return its output."""
    if not DAYTONA_API_KEY:
        return "Error: DAYTONA_API_KEY is not configured; code execution is unavailable."

    lang = _LANGUAGE_ALIASES.get((language or "python").lower())
    if lang is None:
        return (
            f"Error: unsupported language '{language}'. "
            "Supported: python, javascript, typescript, bash."
        )

    async with _daytona_lock:
        for attempt in (1, 2):
            try:
                sandbox = await _get_daytona_sandbox()
            except Exception as exc:
                if attempt == 2:
                    return f"Error: could not start sandbox: {exc}"
                continue

            try:
                if lang == "python":
                    response = await sandbox.process.code_run(code, timeout=60)
                else:
                    # For non-Python runtimes, pipe the source through a shell
                    # heredoc into the interpreter.
                    interpreter = {"javascript": "node", "typescript": "ts-node", "bash": "bash"}[lang]
                    heredoc = f"{interpreter} <<'__CODE_EOF__'\n{code}\n__CODE_EOF__"
                    response = await sandbox.process.exec(heredoc, timeout=60)

                output = (response.result or "").rstrip()
                exit_code = getattr(response, "exit_code", 0)
                if exit_code:
                    output += f"\n[exit code: {exit_code}]"
                return output or "(no output)"
            except Exception as exc:
                # The sandbox may have died between calls; rebuild it and retry.
                await _reset_daytona_sandbox()
                if attempt == 2:
                    return f"Code execution error: {exc}"
        return "Code execution error: unknown failure"

# Backwards-compatible alias for the old Piston-based name.
run_code_in_piston = run_code_in_sandbox

# ---------------------------------------------------------------------------
# Wolfram Alpha computational engine
# ---------------------------------------------------------------------------
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")
WOLFRAM_API_URL = "https://api.wolframalpha.com/v2/query"

# Pods that never carry useful text for the model.
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

    params = {
        "input": query,
        "appid": WOLFRAM_APP_ID,
        "output": "json",
    }

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
        if pod.get("primary"):
            primary.append((title, text))
        else:
            secondary.append((title, text))

    for title, text in primary:
        lines.append(f"\n{title}:\n{text}")

    # A couple of secondary pods add useful context without bloating context.
    for title, text in secondary[:2]:
        lines.append(f"\n{title}:\n{text}")

    if not lines:
        return "Wolfram Alpha returned no readable result for that query."

    return "\n".join(lines).strip()

# ---------------------------------------------------------------------------
# LangSearch web search
# ---------------------------------------------------------------------------
async def web_search(query: str, count: int = 5, freshness: str = "noLimit",
                     deep: bool = True) -> str:
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

            # Format results
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

        except httpx.HTTPStatusError as e:
            return f"LangSearch API error ({e.response.status_code}): {e.response.text}"
        except Exception as exc:
            return f"LangSearch error: {exc}"

# ---------------------------------------------------------------------------
# File system operations
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
        _track_file(path)
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
        return f"Successfully edited {file_path}"
    except Exception as exc:
        return f"Error editing file: {exc}"


# ---------------------------------------------------------------------------
# Chat mode database helpers
# ---------------------------------------------------------------------------
async def db_is_chat_mode_enabled(channel_id: str) -> bool:
    """Check if chat mode is enabled for a channel."""
    def _fetch():
        try:
            resp = (
                supabase.table("chat_mode")
                .select("enabled")
                .eq("channel_id", str(channel_id))
                .execute()
            )
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return False
            raise
        return bool(resp.data and resp.data[0].get("enabled"))
    result = await asyncio.to_thread(_fetch)
    return result


async def db_set_chat_mode(channel_id: str, enabled: bool):
    """Enable or disable chat mode for a channel."""
    def _upsert():
        try:
            supabase.table("chat_mode").upsert({
                "channel_id": str(channel_id),
                "enabled": enabled,
            }, on_conflict="channel_id").execute()
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return
            raise
    await asyncio.to_thread(_upsert)


# ---------------------------------------------------------------------------
# Supabase helpers (with graceful fallback for missing tool_calls columns)
# ---------------------------------------------------------------------------
async def db_get_history(user_id, limit=MAX_HISTORY * 6):
    def _fetch():
        try:
            resp = (
                supabase.table("conversation_history")
                .select("role, content, tool_calls, tool_call_id, name")
                .eq("user_id", str(user_id))
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                resp = (
                    supabase.table("conversation_history")
                    .select("role, content")
                    .eq("user_id", str(user_id))
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
            else:
                raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return list(reversed(rows))


async def db_get_user_facts(user_id):
    def _fetch():
        try:
            resp = (
                supabase.table("user_facts")
                .select("fact_key, fact_value")
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower() or "404" in str(e):
                return []
            raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return {r["fact_key"]: r["fact_value"] for r in rows}


async def db_get_fact(user_id, key):
    """Return a single stored fact value, or None if it does not exist."""
    facts = await db_get_user_facts(user_id)
    return facts.get(key)


async def db_add_message(user_id, role, content, tool_calls=None, tool_call_id=None, name=None):
    content = content or ""
    def _insert():
        payload = {
            "user_id": str(user_id),
            "role": role,
            "content": content,
        }
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        if tool_call_id is not None:
            payload["tool_call_id"] = tool_call_id
        if name is not None:
            payload["name"] = name
        try:
            supabase.table("conversation_history").insert(payload).execute()
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                fallback_payload = {
                    "user_id": str(user_id),
                    "role": role,
                    "content": content,
                }
                supabase.table("conversation_history").insert(fallback_payload).execute()
            else:
                raise
    await asyncio.to_thread(_insert)


async def db_set_fact(user_id, key, value):
    def _upsert():
        supabase.table("user_facts").upsert({
            "user_id": str(user_id),
            "fact_key": key,
            "fact_value": value,
        }, on_conflict="user_id,fact_key").execute()
    await asyncio.to_thread(_upsert)


async def db_delete_fact(user_id, key):
    def _delete():
        supabase.table("user_facts").delete().eq("user_id", str(user_id)).eq("fact_key", key).execute()
    await asyncio.to_thread(_delete)

# ---------------------------------------------------------------------------
# Discord bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------------------------------------
# Keep-alive web server
# ---------------------------------------------------------------------------
app = Flask(__name__)
@app.route("/")
def home(): return "Bot is alive"
def run_flask(): app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
def keep_alive():
    Thread(target=run_flask, daemon=True).start()

# ---------------------------------------------------------------------------
# NVIDIA NIM streaming with tool calling
# ---------------------------------------------------------------------------
MAX_AGENT_STEPS = 5

# Files written during the current agent run. Used so a long job can attach
# the artifacts it produced to its final Discord message.
_written_files: contextvars.ContextVar = contextvars.ContextVar("written_files", default=None)


def _track_file(path: str):
    """Record a file written by a tool, if we are inside a tracked run."""
    tracker = _written_files.get()
    if tracker is not None:
        tracker.append(path)


def build_system_prompt(user_facts: dict) -> str:
    """Build the system prompt, including any remembered facts."""
    facts_text = ""
    if user_facts:
        facts_lines = [f"- {k}: {v}" for k, v in user_facts.items()]
        facts_text = "\n\nUser facts you should remember:\n" + "\n".join(facts_lines)

    return (
        "You are a helpful assistant with a computational engine (WolframAlpha), code "
        "execution, web search, and file management abilities. "
        "Use the wolfram_alpha tool FIRST for any mathematical question: arithmetic, algebra, "
        "calculus (derivatives, integrals), equations, simplification, factoring, matrices, "
        "eigenvalues, statistics and probability, physics, chemistry, unit conversions, and "
        "dates. Do not write code to do arithmetic or symbolic math - WolframAlpha is faster "
        "and more accurate. Use the code tool for data processing, string/file manipulation, "
        "and algorithms, not for math. "
        "Use the web_search tool for current information, news, facts, or anything not in your training data. "
        "Use the read_file, write_file, and edit_file tools to read, write, and edit files in the workspace. Only call tools when necessary. Once you have gathered enough information to answer the user, provide the final answer directly without calling more tools. Do not call the same tool repeatedly - if one search or execution doesn\t give useful results, answer with what you have or try a different approach. "
        "Files are stored in a sandboxed workspace directory."
        "\n\n"
        "How to answer well:\n"
        "- Ground every factual claim in what a tool actually returned. If a tool did not "
        "give you the number, date, or name, do not state it from memory. Say what is missing.\n"
        "- Distinguish what a source reported from what you inferred. Flag estimates "
        "explicitly, and flag it when two sources disagree rather than silently picking one.\n"
        "- Prefer showing the work over asserting the conclusion: include the key figures, "
        "the units, and the source.\n"
        "- If you could not verify something, say so plainly. Never fill a gap with a "
        "plausible-sounding number.\n"
        "- When you finish, offer one concrete next step you could take (a specific search, "
        "a specific analysis, or a specific file) so the user can push the work forward."
        + facts_text
    )


def history_to_messages(history: list) -> list:
    """Rebuild API messages from stored conversation rows.

    A "tool" message is only valid if the assistant turn that requested it is
    still present, so drop orphaned tool rows (which can happen when history
    is truncated mid tool-call sequence).
    """
    messages = []
    pending_tool_ids = set()
    for row in history:
        msg = {"role": row["role"], "content": row["content"] or ""}
        if row.get("tool_calls"):
            msg["tool_calls"] = row["tool_calls"]
            if row["role"] == "assistant":
                pending_tool_ids = {tc.get("id") for tc in row["tool_calls"] if tc.get("id")}
        if row.get("tool_call_id"):
            if row["tool_call_id"] not in pending_tool_ids:
                continue  # orphaned tool result, skip it
            pending_tool_ids.discard(row["tool_call_id"])
            msg["tool_call_id"] = row["tool_call_id"]
            msg["name"] = row.get("name")
        messages.append(msg)
    return messages


async def get_nvidia_response(user_id, user_message, channel=None):
    """Answer a message and persist the exchange. Returns the answer text."""
    result = await get_nvidia_response_full(user_id, user_message, channel)
    return result["answer"]


def _footer_for(result: dict) -> str:
    """Build the small grey footer line: model + tools used."""
    parts = [NVIDIA_MODEL]
    if result.get("tools_used"):
        parts.append(" · ".join(result["tools_used"]))
    return " · ".join(parts)


async def get_nvidia_response_full(user_id, user_message, channel=None):
    """Like get_nvidia_response, but also returns tool/footer metadata."""
    history = await db_get_history(user_id)
    user_facts = await db_get_user_facts(user_id)

    messages = [{"role": "system", "content": build_system_prompt(user_facts)}]

    keep_count = MAX_HISTORY * 6
    if len(history) > keep_count:
        messages.append({
            "role": "system",
            "content": f"[Earlier conversation truncated. Only the last {keep_count} messages are shown.]"
        })
        history_to_use = history[-keep_count:]
    else:
        history_to_use = history

    messages.extend(history_to_messages(history_to_use))
    messages.append({"role": "user", "content": user_message})
    await db_add_message(user_id, "user", user_message)

    result = await run_agent_loop(
        messages,
        user_id=user_id,
        channel=channel,
        max_steps=MAX_AGENT_STEPS,
    )
    result["footer"] = _footer_for(result)
    return result


async def run_agent_loop(messages, user_id=None, channel=None, max_steps=MAX_AGENT_STEPS,
                         tools=None, persist=True, on_step=None, max_tokens=4096):
    """Run the agent tool-calling loop until the model produces a final answer.

    Returns a dict with the answer, which tools were used, and how many
    steps it took.
    """
    if tools is None:
        tools = TOOLS

    final_answer = ""
    tools_used = []
    step_count = 0
    failed_count = 0

    while True:
        if step_count >= max_steps:
            if channel:
                await channel.send(f"⚠️ Reached maximum steps ({max_steps}). Stopping.")
            break
        step_count += 1

        # Try NVIDIA first, fall back to Groq if NVIDIA is unavailable.
        providers = [("NVIDIA", nvidia_client, NVIDIA_MODEL)]
        if groq_client is not None:
            providers.append(("Groq", groq_client, GROQ_MODEL))

        stream = None
        provider_name = ""
        last_error = None
        for name, client, mdl in providers:
            try:
                stream = await client.chat.completions.create(
                    model=mdl,
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=max_tokens,
                    top_p=0.95,
                    stream=True,
                    tools=tools,
                    tool_choice="auto",
                )
                provider_name = name
                break
            except Exception as e:
                last_error = e
                continue

        if stream is None:
            if channel:
                await channel.send(f"⚠️ LLM API error: {str(last_error)[:150]}")
        
        step_content = ""
        tool_calls_accum = {}
        finish_reason = None
        
        try:
            async for chunk in stream:
                # The provider sends a final usage-only chunk with an empty
                # "choices" list. Skip it instead of crashing on choices[0].
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    step_content += delta.content
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_accum[idx]["id"] = tc.id
                        if tc.function.name:
                            tool_calls_accum[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_accum[idx]["arguments"] += tc.function.arguments
        except Exception as exc:
            # A stream can fail mid-flight (rate limit, network drop).
            if channel:
                await channel.send(f"⚠️ LLM stream error: {str(exc)[:150]}")
            if final_answer:
                return {"answer": final_answer, "tools_used": tools_used, "steps": step_count}
            return {"answer": "LLM stream error", "tools_used": tools_used, "steps": step_count}
        
        if tool_calls_accum:
            tool_calls_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_accum.values()
            ]
            if persist and user_id is not None:
                await db_add_message(user_id, "assistant", step_content or "", tool_calls=tool_calls_list)
            # The API requires the assistant turn that requested the tool calls
            # to precede the tool results, otherwise the next request is rejected.
            messages.append({
                "role": "assistant",
                "content": step_content or None,
                "tool_calls": tool_calls_list,
            })
            # Text emitted before a tool call is a preamble ("Let me gather
            # more data..."), not part of the answer. Keep it out of the
            # final answer but preserve it for the model's own context.
            step_content = ""
        else:
            # No tool calls: this is the real answer. Discard any preamble
            # collected during earlier tool-calling steps.
            if finish_reason == "length":
                # The answer was cut off by the token limit and will be
                # continued, so append rather than replace - otherwise the
                # earlier part of the answer is lost.
                final_answer += step_content
            else:
                final_answer = step_content
            if persist and user_id is not None:
                await db_add_message(user_id, "assistant", step_content)
        
        if not tool_calls_accum:
            if finish_reason == "length":
                messages.append({"role": "assistant", "content": step_content})
                messages.append({"role": "user", "content": "Continue your response from where you left off."})
                continue
            break
        
        # Send one status message per step, not per tool call. The model often
        # requests several searches in one step, which produced duplicate
        # "Step N - Searching the web..." lines.
        status_messages = {
            "code": "💻 Running code...",
            "wolfram_alpha": "🧮 Computing...",
            "web_search": "🔍 Searching the web...",
            "read_file": "📖 Reading file...",
            "write_file": "📝 Writing file...",
            "edit_file": "✏️ Editing file...",
        }
        step_labels = []
        for tc in tool_calls_accum.values():
            label = status_messages.get(tc["name"], f"Running {tc['name']}...")
            if label not in step_labels:
                step_labels.append(label)
            if tc["name"] not in tools_used:
                tools_used.append(tc["name"])

        if channel:
            await channel.send(f"**Step {step_count}** - {' '.join(step_labels)}")
        if on_step is not None:
            await on_step(step_count, max_steps, step_labels)

        step_failed = False
        for idx, tc in tool_calls_accum.items():
            tool_name = tc["name"]

            if tc["name"] == "wolfram_alpha":
                try:
                    args = json.loads(tc["arguments"])
                    query = args.get("query", "")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for wolfram_alpha tool"
                else:
                    result = await wolfram_alpha(query)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "wolfram_alpha",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="wolfram_alpha")

            elif tc["name"] == "code":
                try:
                    args = json.loads(tc["arguments"])
                    code = args.get("code", "")
                    language = args.get("language", "python")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for code tool"
                else:
                    result = await run_code_in_sandbox(code, language)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "code",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="code")
            
            elif tc["name"] == "web_search":
                try:
                    args = json.loads(tc["arguments"])
                    query = args.get("query", "")
                    count = args.get("count", 5)
                    freshness = args.get("freshness", "noLimit")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for web_search tool"
                else:
                    result = await web_search(query, count, freshness)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "web_search",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="web_search")
            
            elif tc["name"] == "read_file":
                try:
                    args = json.loads(tc["arguments"])
                    file_path = args.get("file_path", "")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for read_file tool"
                else:
                    result = await read_file(file_path)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "read_file",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="read_file")
            
            elif tc["name"] == "write_file":
                try:
                    args = json.loads(tc["arguments"])
                    file_path = args.get("file_path", "")
                    content = args.get("content", "")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for write_file tool"
                else:
                    result = await write_file(file_path, content)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "write_file",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="write_file")
            
            elif tc["name"] == "edit_file":
                try:
                    args = json.loads(tc["arguments"])
                    file_path = args.get("file_path", "")
                    old_string = args.get("old_string", "")
                    new_string = args.get("new_string", "")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for edit_file tool"
                else:
                    result = await edit_file(file_path, old_string, new_string)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "edit_file",
                    "content": result,
                })
                if persist and user_id is not None:
                    await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="edit_file")
            
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": f"Error: Unknown tool {tc['name']}",
                })

            # Detect tool-level failures so the model doesn't loop on a
            # broken tool until it exhausts the step budget.
            if result.startswith("Error:") or "API error" in result[:120] or result.startswith("Wolfram Alpha could not"):
                step_failed = True

        tool_calls_accum.clear()

        if step_failed:
            failed_count += 1
            if failed_count >= 2:
                # The tool is not working. Tell the model to stop calling
                # tools and answer with whatever it has.
                messages.append({
                    "role": "user",
                    "content": (
                        "The tools are returning errors and are not working right now. "
                        "Do not call any more tools. Answer the user directly with the "
                        "information you already have, and mention briefly that the "
                        "tool was unavailable if relevant."
                    ),
                })
                failed_count = 0
        else:
            failed_count = 0
    
    return {
        "answer": final_answer or "(no response)",
        "tools_used": tools_used,
        "steps": step_count,
    }


# ---------------------------------------------------------------------------
# Background research jobs
# ---------------------------------------------------------------------------
# A job is a long-running research task that runs in the background while the
# bot stays responsive. It reports progress as it goes and delivers the final
# answer (plus any files it produced) when it finishes.
#
# Jobs are tracked in memory and their counter is mirrored to disk so IDs keep
# increasing across restarts.

JOB_MAX_STEPS = 30
JOB_TICK_SECONDS = 100          # how often to report progress
JOB_STATE_FILE = os.path.join(BASE_DIR, ".jobs.json")

_jobs: dict[str, dict] = {}
_job_counter = 0
_job_lock = asyncio.Lock()


def _load_job_state():
    """Restore the job counter from disk so IDs keep increasing across restarts."""
    global _job_counter
    try:
        with open(JOB_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        _job_counter = max(_job_counter, int(data.get("counter", 0)))
    except Exception:
        pass


def _save_job_state():
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"counter": _job_counter}, f)
    except Exception:
        pass


async def _next_job_id() -> str:
    global _job_counter
    async with _job_lock:
        _job_counter += 1
        _save_job_state()
        return f"job_{_job_counter}"


async def run_research_job(job_id: str, channel, user_id: str, topic: str,
                           max_steps: int = JOB_MAX_STEPS):
    """Run a long research job in the background and post progress + result.

    The job works through the topic with the normal tool-calling loop, but
    with a much larger step budget than an interactive reply, and it can save
    intermediate artifacts to files that get attached at the end.
    """
    tracker: list[str] = []
    token = _written_files.set(tracker)

    started = time.monotonic()

    async def report(step, total, labels):
        elapsed = int(time.monotonic() - started)
        mins, secs = divmod(elapsed, 60)
        await channel.send(
            f"-# step {step} of {total} · {mins}m{secs:02d}s elapsed · "
            f"{' '.join(labels)}"
        )

    try:
        messages = [{
            "role": "system",
            "content": (
                "You are a research agent working on a long, multi-step task. Work "
                "autonomously: plan, gather evidence with tools, analyse it, and "
                "produce a thorough, well-structured report.\n"
                "Ground every figure in what a tool actually returned. If a tool did "
                "not give you a number, do not state it - flag it as unverified. If "
                "sources disagree, show the conflict rather than resolving it "
                "silently. Distinguish reported figures from your own estimates.\n"
                "IMPORTANT: write your deliverable to files as you go, not only at the "
                "end. Use write_file to save each major section as its own markdown "
                "file (e.g. report_1.md, report_2.md) as soon as you have finished "
                "that section, and revise it later if new evidence arrives. The files "
                "are what gets delivered to the user, so they must contain the real "
                "work. Keep your final chat summary short and point at the files.\n"
                f"The task: {topic}"
            ),
        }]

        result = await run_agent_loop(
            messages,
            user_id=None,
            channel=channel,
            max_steps=max_steps,
            persist=False,
            on_step=report,
            max_tokens=8192,
        )

        answer = result["answer"]
        files = [p for p in tracker if os.path.exists(p)]

        header = f"**Job {job_id.replace('job_', '#')} is done** — {topic}\n"
        if files:
            header += (
                f"Delivered as {len(files)} file"
                f"{'s' if len(files) != 1 else ''}: "
                + ", ".join(os.path.basename(p) for p in files)
                + "\n"
            )

        summary = answer if answer.strip() else "(job finished with no summary)"
        await send_long(channel, header + summary)

        for path in files:
            try:
                await channel.send(file=discord.File(path))
            except Exception as exc:
                await channel.send(f"⚠️ Could not attach {os.path.basename(path)}: {exc}")

    except Exception as exc:
        try:
            await channel.send(f"⚠️ **Job {job_id.replace('job_', '#')} failed:** {exc}")
        except Exception:
            pass
    finally:
        _written_files.reset(token)

# ---------------------------------------------------------------------------
# Discord message helpers
# ---------------------------------------------------------------------------
# Discord rejects messages over 2000 characters, so long replies are split
# into several messages instead of being cut off mid-sentence.
DISCORD_MAX_LEN = 2000


def split_message(text: str, max_len: int = DISCORD_MAX_LEN) -> list[str]:
    """Split text into Discord-sized chunks, preferring natural break points.

    Tries to break on a paragraph, then a line, then a word boundary, so we
    don't cut words in half. Very long unbroken tokens (e.g. a base64 blob)
    are hard-split as a last resort.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_len:
        window = remaining[:max_len]

        # Prefer a paragraph break, then a newline, then a space.
        cut = window.rfind("\n\n")
        if cut < max_len // 2:
            cut = window.rfind("\n")
        if cut < max_len // 2:
            cut = window.rfind(" ")
        if cut < max_len // 2:
            cut = max_len  # no usable break: hard split

        # Don't leave a tiny fragment behind.
        if cut < 1:
            cut = max_len

        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")

    if remaining:
        chunks.append(remaining)

    return chunks


async def send_long(target, content: str, code_block: bool = False, footer: str = None):
    """Send content to a channel/context, splitting it across messages if needed.

    `target` may be a CommandContext (uses .send) or a Message (first chunk is
    a reply, continuations are plain channel messages).
    """
    # Reserve room for the continuation marker so no message exceeds the limit.
    marker_room = 16
    footer_text = f"\n-# {footer}" if footer else ""
    footer_room = len(footer_text) + 4

    if code_block:
        chunks = split_message(content, DISCORD_MAX_LEN - marker_room - 10 - footer_room)
        formatted = [
            f"```\n{chunk}{f'\n… ({i + 1}/{len(chunks)})' if len(chunks) > 1 else ''}\n```"
            for i, chunk in enumerate(chunks)
        ]
    else:
        chunks = split_message(content, DISCORD_MAX_LEN - marker_room - footer_room)
        formatted = [
            f"{chunk}{f'… ({i + 1}/{len(chunks)})' if len(chunks) > 1 else ''}"
            for i, chunk in enumerate(chunks)
        ]

    # The footer goes on the last chunk so the whole reply reads as one unit.
    if footer_text and formatted:
        formatted[-1] = formatted[-1] + footer_text

    for i, text in enumerate(formatted):
        if i == 0 and hasattr(target, "reply"):
            await target.reply(text)
        elif hasattr(target, "send"):
            await target.send(text)
        else:
            # A Context has .send; a Message has .channel.send.
            await target.channel.send(text)


# ---------------------------------------------------------------------------
# Discord commands
# ---------------------------------------------------------------------------
@bot.command(name="ask")
async def ask(ctx, *, question: str):
    async with ctx.typing():
        result = await get_nvidia_response_full(ctx.author.id, question, ctx.channel)
    await send_long(ctx, result["answer"], footer=result["footer"])

@bot.command(name="run")
async def run_code(ctx, language: str = "python", *, code: str):
    async with ctx.typing():
        result = await run_code_in_sandbox(code, language)
    await send_long(ctx, result, code_block=True)

@bot.command(name="search")
async def search_web(ctx, *, query: str):
    """Search the web. Usage: !search latest AI news"""
    async with ctx.typing():
        result = await web_search(query)
    await send_long(ctx, result)

@bot.command(name="calc")
async def calc(ctx, *, query: str):
    """Compute with WolframAlpha. Usage: !calc integrate x^2 from 0 to 3"""
    async with ctx.typing():
        result = await wolfram_alpha(query)
    await send_long(ctx, result)

@bot.command(name="research")
async def research(ctx, *, topic: str):
    """Start a long background research job. Usage: !research AI capex financing"""
    job_id = await _next_job_id()
    label = job_id.replace("job_", "#")
    _jobs[job_id] = {
        "id": job_id,
        "topic": topic,
        "user_id": ctx.author.id,
        "channel_id": ctx.channel.id,
        "status": "running",
    }
    await ctx.send(
        f"**⏳ Job {label} started** — {topic}\n"
        f"-# researching in the background · up to {JOB_MAX_STEPS} steps · "
        f"progress updates every {JOB_TICK_SECONDS}s · results will post here"
    )

    async def runner():
        try:
            await run_research_job(job_id, ctx.channel, ctx.author.id, topic)
            _jobs[job_id]["status"] = "done"
        except Exception as exc:
            _jobs[job_id]["status"] = "failed"
            try:
                await ctx.send(f"⚠️ Job {label} failed: {exc}")
            except Exception:
                pass

    # Detach so the bot stays responsive while the job runs.
    asyncio.create_task(runner())

@bot.command(name="jobs")
async def list_jobs(ctx):
    """List background research jobs. Usage: !jobs"""
    if not _jobs:
        await ctx.send("No research jobs have run yet.")
        return
    lines = []
    for job in list(_jobs.values())[-10:]:
        label = job["id"].replace("job_", "#")
        lines.append(f"`{label}` [{job['status']}] {job['topic'][:80]}")
    await ctx.send("**Research jobs**\n" + "\n".join(lines))

@bot.command(name="remember")
async def remember(ctx, key: str, *, value: str):
    await db_set_fact(ctx.author.id, key, value)
    await ctx.send(f"Remembered: {key} = {value}")

@bot.command(name="recall")
async def recall(ctx, key: str):
    val = await db_get_fact(ctx.author.id, key)
    await ctx.send(f"{key}: {val}" if val else "I don't remember that.")

@bot.command(name="forget")
async def forget(ctx, key: str):
    await db_delete_fact(ctx.author.id, key)
    await ctx.send(f"Forgot {key}.")

@bot.command(name="sendfile")
async def send_file_cmd(ctx, file_path: str):
    """Send a file from the workspace as a Discord attachment. Usage: !sendfile path/to/file.txt"""
    try:
        path = safe_path(file_path)
    except ValueError as e:
        await ctx.send(str(e))
        return
    if not os.path.exists(path):
        await ctx.send(f"File not found: {file_path}")
        return
    try:
        await ctx.send(file=discord.File(path))
    except Exception as e:
        await ctx.send(f"Error sending file: {e}")

@bot.command(name="listfiles")
async def list_files_cmd(ctx, directory: str = ""):
    """List files in the workspace directory. Usage: !listfiles [subdir]"""
    try:
        target_dir = safe_path(directory) if directory else BASE_DIR
    except ValueError as e:
        await ctx.send(str(e))
        return
    if not os.path.isdir(target_dir):
        await ctx.send(f"Not a directory: {directory}")
        return
    try:
        entries = os.listdir(target_dir)
        if not entries:
            await ctx.send("Directory is empty")
            return
        lines = []
        for entry in sorted(entries):
            full = os.path.join(target_dir, entry)
            if os.path.isdir(full):
                lines.append(f"📁 {entry}/")
            else:
                size = os.path.getsize(full)
                lines.append(f"📄 {entry} ({size} bytes)")
        response = "\n".join(lines[:50])
        if len(entries) > 50:
            response += f"\n...and {len(entries) - 50} more"
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"Error listing files: {e}")


@bot.command(name="chatmode")
async def chatmode(ctx, action: str):
    """Toggle chat mode for this channel. Usage: !chatmode on|off"""
    action = action.lower()
    if action not in ("on", "off"):
        await ctx.send("Usage: !chatmode on|off")
        return
    enabled = action == "on"
    await db_set_chat_mode(str(ctx.channel.id), enabled)
    status = "enabled" if enabled else "disabled"
    await ctx.send(f"Chat mode {status} for this channel. The bot will now respond to all messages{' (no mention needed)' if enabled else ''}.")

@bot.command(name="test_piston")
async def test_piston(ctx):
    """Test if the code execution sandbox is working. Usage: !test_piston"""
    async with ctx.typing():
        result = await run_code_in_sandbox("print('Code sandbox is working!')", "python")
    await send_long(ctx, result, code_block=True)

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    _load_job_state()
    print("------")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    # Ignore other bots to avoid feedback loops.
    if message.author.bot:
        return

    # A prefix command (e.g. "!ask ...") is handled by process_commands.
    # Don't also feed it to the chat handler, or it would run twice.
    is_command = False
    if isinstance(bot.command_prefix, str):
        is_command = message.content.startswith(bot.command_prefix)
    elif isinstance(bot.command_prefix, (list, tuple)):
        is_command = any(message.content.startswith(p) for p in bot.command_prefix)

    await bot.process_commands(message)
    if is_command:
        return

    try:
        chat_on = await db_is_chat_mode_enabled(str(message.channel.id))
    except Exception as exc:
        print(f"chat mode lookup failed: {exc}")
        chat_on = False

    mentioned = bot.user.mentioned_in(message)
    is_dm = isinstance(message.channel, discord.DMChannel)

    if chat_on or mentioned or is_dm:
        content = message.content
        if mentioned:
            content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not content:
            content = "Hello!"
        async with message.channel.typing():
            result = await get_nvidia_response_full(message.author.id, content, message.channel)
        await send_long(message, result["answer"], footer=result["footer"])

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
keep_alive()
bot.run(os.environ["DISCORD_TOKEN"])