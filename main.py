import os
import asyncio
import discord
from discord.ext import commands
from flask import Flask
from threading import Thread
import requests
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
MAX_HISTORY = 8

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ---------------------------------------------------------------------------
# Discord bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------------------------------------
# Keep-alive web server (Render free tier)
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Bot is alive"

def run_flask():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))

def keep_alive():
    t = Thread(target=run_flask, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Supabase helpers (run sync calls in a thread so they don't block the bot)
# ---------------------------------------------------------------------------
async def db_get_history(user_id, limit=MAX_HISTORY):
    def _fetch():
        resp = (
            supabase.table("conversation_history")
            .select("role, content")
            .eq("user_id", str(user_id))
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return list(reversed(rows))

async def db_add_history(user_id, role, content):
    def _insert():
        supabase.table("conversation_history").insert({
            "user_id": str(user_id),
            "role": role,
            "content": content,
        }).execute()
    await asyncio.to_thread(_insert)

async def db_get_fact(user_id, key):
    def _fetch():
        resp = (
            supabase.table("user_facts")
            .select("fact_value")
            .eq("user_id", str(user_id))
            .eq("fact_key", key)
            .execute()
        )
        return resp.data
    rows = await asyncio.to_thread(_fetch)
    return rows[0]["fact_value"] if rows else None

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
# Groq API helper with permanent memory
# ---------------------------------------------------------------------------
async def get_groq_response(user_id, user_message):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set."

    history = await db_get_history(user_id)

    messages = [
        {"role": "system", "content": "You are a helpful assistant on Discord. Remember what the user tells you."}
    ]
    messages.extend({"role": r["role"], "content": r["content"]} for r in history)
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]

        await db_add_history(user_id, "user", user_message)
        await db_add_history(user_id, "assistant", answer)

        return answer
    except Exception as exc:
        return f"Error contacting Groq: {exc}"

# ---------------------------------------------------------------------------
# Relational memory commands
# ---------------------------------------------------------------------------
@bot.command(name="remember")
async def remember(ctx, key: str, *, value: str):
    """Save a fact. Usage: !remember favorite_color blue"""
    await db_set_fact(ctx.author.id, key, value)
    await ctx.send(f"Remembered: {key} = {value}")

@bot.command(name="recall")
async def recall(ctx, key: str):
    """Recall a saved fact. Usage: !recall favorite_color"""
    val = await db_get_fact(ctx.author.id, key)
    if val:
        await ctx.send(f"{key}: {val}")
    else:
        await ctx.send("I don't remember that.")

@bot.command(name="forget")
async def forget(ctx, key: str):
    """Delete a saved fact. Usage: !forget favorite_color"""
    await db_delete_fact(ctx.author.id, key)
    await ctx.send(f"Forgot {key}.")

# ---------------------------------------------------------------------------
# Discord events & commands
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.command(name="ask")
async def ask(ctx, *, question: str):
    """Ask Groq anything."""
    async with ctx.typing():
        answer = await get_groq_response(ctx.author.id, question)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n...(truncated)"
    await ctx.send(answer)

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