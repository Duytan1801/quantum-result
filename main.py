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
MAX_HISTORY = 10

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DAYTONA_API_URL = os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")

LANGSEARCH_API_URL = os.getenv("LANGSEARCH_API_URL", "https://api.langsearch.com/v1/web-search")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")

# ---------------------------------------------------------------------------
# Tools: code execution + web search
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "code",
            "description": "Execute code in a sandboxed Daytona environment. Use for calculations, data processing, API calls, etc.",
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
]

# ---------------------------------------------------------------------------
# Daytona sandbox execution
# ---------------------------------------------------------------------------
async def run_code_in_daytona(code: str, language: str = "python") -> str:
    if not DAYTONA_API_KEY:
        return "Error: DAYTONA_API_KEY not configured."

    lang_cmd = {
        "python": "python3 -c",
        "python3": "python3 -c",
        "javascript": "node -e",
        "js": "node -e",
        "bash": "bash -c",
        "sh": "sh -c",
    }
    cmd_prefix = lang_cmd.get(language.lower(), "python3 -c")

    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {"Authorization": f"Bearer {DAYTONA_API_KEY}"}
        sandbox_id = None
        try:
            create_resp = await client.post(
                f"{DAYTONA_API_URL}/sandbox",
                headers=headers,
                json={"image": "daytonaio/sandbox:latest"},
            )
            create_resp.raise_for_status()
            sandbox_id = create_resp.json()["id"]

            try:
                run_resp = await client.post(
                    f"{DAYTONA_API_URL}/sandbox/{sandbox_id}/exec",
                    headers=headers,
                    json={"command": f'{cmd_prefix} {json.dumps(code)}'},
                )
                run_resp.raise_for_status()
                result = run_resp.json()
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                exit_code = result.get("exit_code", 0)
                output = stdout
                if stderr:
                    output += f"\n[stderr]: {stderr}"
                if exit_code != 0:
                    output += f"\n[exit code: {exit_code}]"
                return output or "(no output)"
            except httpx.HTTPStatusError as e:
                return f"Daytona API error ({e.response.status_code}): {e.response.text}"
            except Exception as exc:
                return f"Daytona execution error: {exc}"
        finally:
            if sandbox_id:
                try:
                    await client.delete(f"{DAYTONA_API_URL}/sandbox/{sandbox_id}", headers=headers)
                except Exception:
                    pass

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
async def get_groq_response(user_id, user_message):
    history = await db_get_history(user_id)
    user_facts = await db_get_user_facts(user_id)

    # Build system prompt with user facts
    facts_text = ""
    if user_facts:
        facts_lines = [f"- {k}: {v}" for k, v in user_facts.items()]
        facts_text = "\n\nUser facts you should remember:\n" + "\n".join(facts_lines)

    system_prompt = (
        "You are a helpful assistant with code execution and web search abilities. "
        "Use the code tool for calculations, data tasks, or when the user asks you to run code. "
        "Use the web_search tool for current information, news, facts, or anything not in your training data."
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
    
    while True:
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
            await db_add_message(user_id, "assistant", step_content or None, tool_calls=tool_calls_list)
        else:
            await db_add_message(user_id, "assistant", step_content)
        
        if not tool_calls_accum:
            if finish_reason == "length":
                messages.append({"role": "assistant", "content": step_content})
                messages.append({"role": "user", "content": "Continue your response from where you left off."})
                continue
            break
        
        for idx, tc in tool_calls_accum.items():
            if tc["name"] == "code":
                try:
                    args = json.loads(tc["arguments"])
                    code = args.get("code", "")
                    language = args.get("language", "python")
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments for code tool"
                else:
                    result = await run_code_in_daytona(code, language)
                
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
        answer = await get_groq_response(ctx.author.id, question)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n...(truncated)"
    await ctx.send(answer)

@bot.command(name="run")
async def run_code(ctx, language: str = "python", *, code: str):
    async with ctx.typing():
        result = await run_code_in_daytona(code, language)
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

    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        content = (
            message.content.replace(f"<@{bot.user.id}>", "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )
        if not content:
            content = "Hello!"
        async with message.channel.typing():
            answer = await get_groq_response(message.author.id, content)
        if len(answer) > 1900:
            answer = answer[:1900] + "\n...(truncated)"
        await message.reply(answer)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
keep_alive()
bot.run(os.environ["DISCORD_TOKEN"])