"""Discord bot entrypoint: bot setup, commands, and event handlers."""

import os

import discord
from discord.ext import commands
from flask import Flask
from threading import Thread

from agent import get_nvidia_response_full
from discord_utils import send_long
from agents import AGENTS, get_agent
from jobs import cancel_job, job_label, list_jobs, start_agent_task
from sandbox import run_code_in_sandbox
from storage import (
    db_delete_fact,
    db_get_fact,
    db_is_chat_mode_enabled,
    db_set_chat_mode,
    db_set_fact,
)
from tools import BASE_DIR, safe_path, web_search, wolfram_alpha

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
def home():
    return "Bot is alive"


def run_flask():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))


def keep_alive():
    Thread(target=run_flask, daemon=True).start()


# ---------------------------------------------------------------------------
# Commands
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
    """Start a background research job. Usage: !research AI capex financing"""
    await start_agent_task("research", topic, ctx.channel, ctx.author.id)


@bot.command(name="agent")
async def agent_cmd(ctx, agent_name: str, *, topic: str):
    """Delegate to a specialist agent. Usage: !agent analyze AI capex model"""
    if get_agent(agent_name) is None:
        await ctx.send(f"Unknown agent '{agent_name}'. Available: {', '.join(AGENTS)}.")
        return
    await start_agent_task(agent_name, topic, ctx.channel, ctx.author.id)


@bot.command(name="agents")
async def agents_cmd(ctx):
    """List the specialist agents. Usage: !agents"""
    lines = [f"`{name}` ({spec['label']}) — {spec['description']}" for name, spec in AGENTS.items()]
    await ctx.send("**Agents**\n" + "\n".join(lines))


@bot.command(name="jobs")
async def jobs_cmd(ctx):
    """List background jobs. Usage: !jobs"""
    jobs = list_jobs()
    if not jobs:
        await ctx.send("No jobs have run yet.")
        return
    lines = []
    for job in jobs[-10:]:
        lines.append(
            f"`{job_label(job['id'])}` [{job['status']}] "
            f"({job.get('agent', 'research')}) {job['topic'][:70]}"
        )
    await ctx.send("**Jobs**\n" + "\n".join(lines))


@bot.command(name="cancel")
async def cancel_job_cmd(ctx, job_id: str):
    """Cancel a running research job. Usage: !cancel #3"""
    await ctx.send(cancel_job(job_id))


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
    await ctx.send(
        f"Chat mode {status} for this channel. The bot will now respond to all messages"
        f"{' (no mention needed)' if enabled else ''}."
    )


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
    from jobs import _load_job_state
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
