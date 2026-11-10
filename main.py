import os
import discord
from discord.ext import commands
from flask import Flask
from threading import Thread
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ---------------------------------------------------------------------------
# Discord bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.direct_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------------------------------------
# Keep-alive web server (Render free tier sleeps without this)
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
# Groq API helper
# ---------------------------------------------------------------------------
async def get_groq_response(user_message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant on Discord."},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"Error contacting Groq: {exc}"


# ---------------------------------------------------------------------------
# Discord events & commands
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")


@bot.command(name="ask")
async def ask(ctx, *, question: str):
    """Ask Groq anything. Usage: !ask What is quantum computing?"""
    async with ctx.typing():
        answer = await get_groq_response(question)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n...(truncated)"
    await ctx.send(answer)


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Handle !ask and other commands first
    await bot.process_commands(message)

    # Also reply when mentioned or in a DM
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        content = (
            message.content.replace(f"<@{bot.user.id}>", "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )
        if not content:
            content = "Hello!"

        async with message.channel.typing():
            answer = await get_groq_response(content)

        if len(answer) > 1900:
            answer = answer[:1900] + "\n...(truncated)"
        await message.reply(answer)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
keep_alive()
bot.run(os.environ["DISCORD_TOKEN"])