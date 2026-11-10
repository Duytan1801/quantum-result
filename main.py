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

# ---------------------------------------------------------------------------
# Tool: code execution via Daytona
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
    }
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
# Supabase helpers
# ---------------------------------------------------------------------------
async def db_get_history(user_id, limit=MAX_HISTORY):
    def _fetch():
        resp = (
            supabase.table("conversation_history")
            .select("role, content, tool_calls, tool_call_id, name")
            .eq("user_id", str(user_id))
            .order("created_at", desc=True)
            .limit(limit * 4)
            .execute()
        )
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return list(reversed(rows))

async def db_add_message(user_id, role, content, tool_calls=None, tool_call_id=None, name=None):
    def _insert():
        supabase.table("conversation_history").insert({
            "user_id": str(user_id),
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
            "name": name,
        }).execute()
    await asyncio.to_thread(_insert)

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
    messages = [{"role": "system", "content": "You are a helpful assistant with code execution ability. Use the code tool for calculations, data tasks, or when the user asks you to run code."}]
    for row in history:
        msg = {"role": row["role"], "content": row["content"] or ""}
        if row.get("tool_calls"):
            msg["tool_calls"] = row["tool_calls"]
        if row.get("tool_call_id"):
            msg["tool_call_id"] = row["tool_call_id"]
            msg["name"] = row.get("name")
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})
    await db_add_message(user_id, "user", user_message)

    full_answer = ""
    tool_calls_accum = {}

    while True:
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            tools=TOOLS,
            tool_choice="auto",
        )

        full_answer = ""
        tool_calls_accum = {}

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_answer += delta.content

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
            await db_add_message(user_id, "assistant", full_answer or None, tool_calls=tool_calls_list)
        else:
            await db_add_message(user_id, "assistant", full_answer)

        if not tool_calls_accum:
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
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": f"Error: Unknown tool {tc['name']}",
                })

        tool_calls_accum.clear()

    return full_answer or "(no response)"

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
# Relational memory helpers (local JSON fallback)
# ---------------------------------------------------------------------------
RELATIONAL_FILE = "relational.json"

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def db_get_fact(user_id, key):
    rel = load_json(RELATIONAL_FILE)
    return rel.get(str(user_id), {}).get(key)

async def db_set_fact(user_id, key, value):
    rel = load_json(RELATIONAL_FILE)
    rel.setdefault(str(user_id), {})[key] = value
    save_json(RELATIONAL_FILE, rel)

async def db_delete_fact(user_id, key):
    rel = load_json(RELATIONAL_FILE)
    if str(user_id) in rel and key in rel[str(user_id)]:
        del rel[str(user_id)][key]
        save_json(RELATIONAL_FILE, rel)

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