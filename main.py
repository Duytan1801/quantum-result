import os
import asyncio
import json
import discord
from discord.ext import commands
from flask import Flask
from threading import Thread
from supabase import create_client, Client
from groq import Groq
import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
MAX_HISTORY = 3

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
            "name": "code",
            "description": "Execute code in a sandbox via Piston API. Use for calculations, data processing, API calls, etc.",
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
            "description": "Search the web for current information, news, facts, or anything not in your training data. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Number of results (default 5, max 10)", "default": 5},
                    "freshness": {"type": "string", "description": "Time filter: 'day', 'week', 'month', 'year', 'noLimit'", "default": "noLimit"},
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
# Piston code execution
# ---------------------------------------------------------------------------
async def run_code_in_piston(code: str, language: str = "python") -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                "https://emkc.org/api/v2/piston/execute",
                json={
                    "language": language,
                    "version": "3.10.0" if language == "python" else "18.15.0",
                    "files": [{"content": code}],
                },
            )
            resp.raise_for_status()
            result = resp.json()
            output = result.get("output", "")
            if result.get("stderr"):
                output += f"\n[stderr]: {result['stderr']}"
            if result.get("status", 0) != 0:
                output += f"\n[exit code: {result['status']}]"
            return output or "(no output)"
        except httpx.HTTPStatusError as e:
            return f"Piston API error ({e.response.status_code}): {e.response.text[:200]}"
        except Exception as exc:
            return f"Piston execution error: {exc}"

# ---------------------------------------------------------------------------
# LangSearch web search
# ---------------------------------------------------------------------------
async def web_search(query: str, count: int = 5, freshness: str = "noLimit") -> str:
    if not LANGSEARCH_API_KEY:
        return "Error: LANGSEARCH_API_KEY not configured."

    payload = {
        "query": query,
        "count": min(count, 10),
        "freshness": freshness,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
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
                snippet = src.get("snippet", "")
                lines.append(f"{name}\n{url}\n{snippet}")

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
# Groq streaming with tool calling
# ---------------------------------------------------------------------------
MAX_AGENT_STEPS = 5

async def get_groq_response(user_id, user_message, channel=None):
    history = await db_get_history(user_id)
    user_facts = await db_get_user_facts(user_id)

    # Build system prompt with user facts
    facts_text = ""
    if user_facts:
        facts_lines = [f"- {k}: {v}" for k, v in user_facts.items()]
        facts_text = "\n\nUser facts you should remember:\n" + "\n".join(facts_lines)

    system_prompt = (
        "You are a helpful assistant with code execution, web search, and file management abilities. "
        "Use the code tool for calculations, data tasks, or when the user asks you to run code. "
        "Use the web_search tool for current information, news, facts, or anything not in your training data. "
        "Use the read_file, write_file, and edit_file tools to read, write, and edit files in the workspace. Only call tools when necessary. Once you have gathered enough information to answer the user, provide the final answer directly without calling more tools. Do not call the same tool repeatedly - if one search or execution doesn\t give useful results, answer with what you have or try a different approach. "
        "Files are stored in a sandboxed workspace directory."
        + facts_text
    )
    messages = [{"role": "system", "content": system_prompt}]

    # Intelligent history truncation: keep recent messages, summarize older ones
    keep_count = MAX_HISTORY * 4  # ~40 exchanges
    if len(history) > keep_count:
        messages.append({
            "role": "system",
            "content": f"[Earlier conversation truncated. Only the last {keep_count} messages are shown.]"
        })
        history_to_use = history[-keep_count:]
    else:
        history_to_use = history

    for row in history_to_use:
        msg = {"role": row["role"], "content": row["content"] or ""}
        if row.get("tool_calls"):
            msg["tool_calls"] = row["tool_calls"]
        if row.get("tool_call_id"):
            msg["tool_call_id"] = row["tool_call_id"]
            msg["name"] = row.get("name")
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})
    await db_add_message(user_id, "user", user_message)

    final_answer = ""
    
    step_count = 0
    
    while True:
        if step_count >= MAX_AGENT_STEPS:
            if channel:
                await channel.send(f"⚠️ Reached maximum steps ({MAX_AGENT_STEPS}). Stopping.")
            break
        step_count += 1
        
        if channel and step_count > 1:
            await channel.send(f"🔄 **Step {step_count}** - Continuing...")
        
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=1000,
            top_p=0.95,
            stream=True,
            tools=TOOLS,
            tool_choice="auto",
        )
        
        step_content = ""
        tool_calls_accum = {}
        finish_reason = None
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                step_content += delta.content
                final_answer += delta.content
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
        
        if tool_calls_accum:
            tool_calls_list = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls_accum.values()
            ]
            await db_add_message(user_id, "assistant", step_content or "", tool_calls=tool_calls_list)
        else:
            await db_add_message(user_id, "assistant", step_content)
        
        if not tool_calls_accum:
            if finish_reason == "length":
                messages.append({"role": "assistant", "content": step_content})
                messages.append({"role": "user", "content": "Continue your response from where you left off."})
                continue
            break
        
        for idx, tc in tool_calls_accum.items():
            tool_name = tc["name"]
            
            # Send status message for each tool
            if channel:
                status_messages = {
                    "code": "💻 Running code...",
                    "web_search": "🔍 Searching the web...",
                    "read_file": "📖 Reading file...",
                    "write_file": "📝 Writing file...",
                    "edit_file": "✏️ Editing file...",
                }
                await channel.send(f"**Step {step_count}** - {status_messages.get(tool_name, f'Running {tool_name}...')}")
            
            if tc["name"] == "code":
                try:
                    args = json.loads(tc["arguments"])
                    code = args.get("code", "")
                    language = args.get("language", "python")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for code tool"
                else:
                    result = await run_code_in_piston(code, language)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": "code",
                    "content": result,
                })
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
                await db_add_message(user_id, "tool", result, tool_call_id=tc["id"], name="edit_file")
            
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": f"Error: Unknown tool {tc['name']}",
                })
        
        tool_calls_accum.clear()
    
    return final_answer or "(no response)"

# ---------------------------------------------------------------------------
# Discord commands
# ---------------------------------------------------------------------------
@bot.command(name="ask")
async def ask(ctx, *, question: str):
    async with ctx.typing():
        answer = await get_groq_response(ctx.author.id, question, ctx.channel)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n...(truncated)"
    await ctx.send(answer)

@bot.command(name="run")
async def run_code(ctx, language: str = "python", *, code: str):
    async with ctx.typing():
        result = await run_code_in_piston(code, language)
    if len(result) > 1900:
        result = result[:1900] + "\n...(truncated)"
    await ctx.send(f"```\n{result}\n```")

@bot.command(name="search")
async def search_web(ctx, *, query: str):
    """Search the web. Usage: !search latest AI news"""
    async with ctx.typing():
        result = await web_search(query)
    if len(result) > 1900:
        result = result[:1900] + "\n...(truncated)"
    await ctx.send(result)

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
    """Test if Piston sandbox is working. Usage: !test_piston"""
    async with ctx.typing():
        result = await run_code_in_piston("print('Piston sandbox is working!')", "python")
    if len(result) > 1900:
        result = result[:1900] + "\n...(truncated)"
    await ctx.send(f"```\n{result}\n```")

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

    chat_on = await db_is_chat_mode_enabled(str(message.channel.id))
    mentioned = bot.user.mentioned_in(message)
    is_dm = isinstance(message.channel, discord.DMChannel)

    if chat_on or mentioned or is_dm:
        content = message.content
        if mentioned:
            content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not content:
            content = "Hello!"
        async with message.channel.typing():
            answer = await get_groq_response(message.author.id, content, message.channel)
        if len(answer) > 1900:
            answer = answer[:1900] + "\n...(truncated)"
        await message.reply(answer)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
keep_alive()
bot.run(os.environ["DISCORD_TOKEN"])