"""Discord bot entrypoint: bot setup, commands, and event handlers."""

import asyncio
import os

import discord
from discord.ext import commands
from flask import Flask
from threading import Thread

from agent import get_nvidia_response_full
from discord_utils import send_long
from agents import AGENTS, get_agent
from jobs import cancel_job, job_label, list_jobs, start_agent_task
from people import (
    bot_name_variants,
    display_name,
    person as person_entry,
    roster_from_messages,
    roster_line,
)
from presence import ChannelTracker, channel_scope, decide
from sandbox import run_code_in_sandbox
from storage import (
    db_delete_fact,
    db_get_user_facts_multi,
    db_is_chat_mode_enabled,
    db_scope_has_rows,
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

# Who has spoken in each channel. A channel with two or more people in it is
# a room, not a private conversation: the bot waits to be spoken to, and the
# context of everybody in the room is shared.
TRACKER = ChannelTracker()

# One turn at a time per channel. Without this, two messages arriving
# together produced two answers that landed on top of each other.
CHANNEL_LOCKS: dict[str, asyncio.Lock] = {}

# The ids of the bot's own recent messages, so a reply to one of them can be
# recognised as being for the bot.
BOT_MESSAGE_IDS: dict[str, set] = {}
BOT_MESSAGE_ID_LIMIT = 200

# Which channels are rooms. Kept across the life of the process, and seeded
# from the database and the channel's own messages the first time a channel
# is seen, so a restart does not make a busy channel look like a private
# chat.
SHARED_CHANNELS: dict[str, bool] = {}

# Who is in each channel, read from the channel's own messages. Nobody is
# listed in the code: the bot learns the room from the room.
ROSTERS: dict[str, list] = {}
ROSTER_TTL = 300


async def _channel_roster(channel_id: str, channel=None, force: bool = False) -> list:
    """The people in this channel, with the names they go by."""
    cached = ROSTERS.get(str(channel_id))
    if cached and not force:
        return cached

    people = []
    if channel is not None:
        try:
            old_messages = []
            async for old in channel.history(limit=100):
                old_messages.append(old)
                if len(old_messages) >= 100:
                    break
            people = roster_from_messages(old_messages)
        except Exception as exc:
            print(f"could not read the messages in {channel_id}: {exc}")

    ROSTERS[str(channel_id)] = people
    return people


async def _channel_is_shared(channel_id: str, channel=None) -> bool:
    """Has this channel got more than one person in it?"""
    if channel_id in SHARED_CHANNELS:
        return SHARED_CHANNELS[channel_id]
    shared = False
    try:
        # If the channel already has rows under its shared scope, it has been
        # a room before - even if the tracker has been reset by a restart.
        shared = await db_scope_has_rows(channel_scope(channel_id))
    except Exception as exc:
        print(f"could not check whether {channel_id} is shared: {exc}")
    if not shared:
        # Otherwise ask Discord who has been talking in here.
        roster = await _channel_roster(channel_id, channel)
        if len(roster) >= 2:
            shared = True
            for entry in roster:
                TRACKER.note(channel_id, entry["id"])
    if shared:
        TRACKER.mark_shared(channel_id)
    SHARED_CHANNELS[channel_id] = shared
    return shared


def _channel_lock(channel_id: str) -> asyncio.Lock:
    """The lock that serialises the turns in one channel."""
    lock = CHANNEL_LOCKS.get(channel_id)
    if lock is None:
        lock = asyncio.Lock()
        CHANNEL_LOCKS[channel_id] = lock
    return lock


def _remember_bot_messages(channel_id: str, messages) -> None:
    """Record the bot's own message ids, so replies to them are noticed."""
    ids = BOT_MESSAGE_IDS.setdefault(channel_id, set())
    for message in messages or []:
        message_id = getattr(message, "id", None)
        if message_id:
            ids.add(message_id)
    if len(ids) > BOT_MESSAGE_ID_LIMIT:
        # Keep the most recent ones.
        for message_id in list(ids)[:-BOT_MESSAGE_ID_LIMIT]:
            ids.discard(message_id)


def _scopes_for(channel, user_id) -> tuple[list, str, bool]:
    """The context keys for one person in one channel.

    In a room the conversation and the memory are shared, so the channel's
    scope is read as well as the person's own, and new rows are written to
    the channel's scope.
    """
    scopes = [str(user_id)]
    scope = TRACKER.scope_for(getattr(channel, "id", None))
    if not scope:
        return scopes, str(user_id), False
    scopes.append(scope)
    return scopes, scope, True



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
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


async def attach_images(target, files: list, caption: str = None):
    """Attach images the agent produced during its turn."""
    images = [f for f in (files or []) if f.lower().endswith(IMAGE_SUFFIXES)]
    if not images:
        return
    sender = target.send if hasattr(target, "send") else target.channel.send
    for path in images:
        try:
            await sender(file=discord.File(path))
        except Exception as exc:
            await sender(f"⚠️ Could not attach {os.path.basename(path)}: {exc}")


@bot.command(name="ask")
async def ask(ctx, *, question: str):
    scopes, write_scope, shared = _scopes_for(ctx.channel, ctx.author.id)
    roster = await _channel_roster(str(ctx.channel.id), ctx.channel)
    async with ctx.typing():
        result = await get_nvidia_response_full(
            ctx.author.id,
            question,
            ctx.channel,
            attachments=ctx.message.attachments,
            scopes=scopes,
            write_scope=write_scope,
            shared=shared,
            speaker=person_entry(ctx.author),
            roster=roster,
        )
    await send_long(ctx, result["answer"], footer=result["footer"])
    await attach_images(ctx, result.get("files") or [])


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
    # In a room, everybody should know the fact - so it is stored against
    # the channel as well as against the person who said it.
    scope = TRACKER.scope_for(ctx.channel.id)
    if scope:
        await db_set_fact(scope, key, value)
    await ctx.send(f"Remembered: {key} = {value}")


@bot.command(name="recall")
async def recall(ctx, key: str):
    scopes = [str(ctx.author.id)]
    scope = TRACKER.scope_for(ctx.channel.id)
    if scope:
        scopes.append(scope)
    facts = await db_get_user_facts_multi(scopes)
    val = facts.get(key)
    await ctx.send(f"{key}: {val}" if val else "I don't remember that.")


@bot.command(name="forget")
async def forget(ctx, key: str):
    """Forget a stored fact, and any long-term memory that mentions it."""
    scopes = [str(ctx.author.id)]
    scope = TRACKER.scope_for(ctx.channel.id)
    if scope:
        scopes.append(scope)

    for scope_key in scopes:
        await db_delete_fact(scope_key, key)

    forgotten = 0
    try:
        from memory import forget_memories, is_available

        if await is_available():
            for scope_key in scopes:
                forgotten += await forget_memories(scope_key, key)
    except Exception as exc:
        print(f"could not forget semantic memories: {exc}")

    if forgotten:
        await ctx.send(
            f"Forgot {key}, and {forgotten} remembered "
            f"{'detail' if forgotten == 1 else 'details'} that mentioned it."
        )
    else:
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
        f"Chat mode {status} for this channel. The bot will now join in when "
        f"there is something useful to add - a request, or a question about "
        f"something it is good at - instead of answering every message."
    )


@bot.command(name="test_piston")
async def test_piston(ctx):
    """Test if the code execution sandbox is working. Usage: !test_piston"""
    async with ctx.typing():
        result = await run_code_in_sandbox("print('Code sandbox is working!')", "python")
    await send_long(ctx, result, code_block=True)


@bot.command(name="kaggle")
async def kaggle_cmd(ctx):
    """Check whether Kaggle cloud compute is configured. Usage: !kaggle"""
    from kaggle_backend import force_refresh, kaggle_status

    force_refresh()
    status = kaggle_status()
    lines = [f"**Kaggle status**"]
    lines.append(f"- Env var `KAGGLE_API_TOKEN` set: {status['env_KAGGLE_API_TOKEN_set']}")
    lines.append(f"- Credentials source: {status['credentials_source'] or 'none'}")
    lines.append(f"- Hardcoded fallback present: {status['hardcoded_fallback_present']}")
    lines.append(f"- Legacy key pair present: {status['legacy_key_present']}")
    lines.append(f"- Config file present: {status['config_file_present']}")
    if status["authenticated"]:
        lines.append("- Live authentication: **OK** — Kaggle is ready to run scripts.")
    else:
        lines.append(f"- Live authentication: **failed** — {status['error']}")
    await send_long(ctx, "\n".join(lines))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
async def _reembed_if_needed(model: str):
    """Re-embed memories if the embedding model has changed."""
    try:
        from memory import reembed_stale_memories

        count = await reembed_stale_memories()
        if count:
            print(f"long-term memory: re-embedded {count} memories for {model}")
    except Exception as exc:
        print(f"long-term memory: re-embed failed ({exc})")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")

    # Say whether the proper math renderer is in use. If Node is missing the
    # bot tries to install it at runtime; if that fails too, it falls back to
    # matplotlib's mathtext, which cannot typeset \begin{aligned} and
    # friends - and it does so silently, which makes it look like the
    # renderer is broken rather than absent.
    try:
        from latex_render import renderer_available

        if renderer_available():
            print("math renderer: KaTeX + MathJax")
        else:
            print("math renderer: matplotlib fallback (no node, and the "
                  "auto-install did not succeed - multi-line environments "
                  "will not typeset)")
    except Exception as exc:
        print(f"math renderer: unavailable ({exc})")

    # Say whether long-term memory is on. It stays off until
    # supabase_memory.sql has been run, and it says so rather than failing
    # quietly on every turn.
    try:
        from memory import active_model, is_available

        if await is_available():
            model = await active_model()
            print(f"long-term memory: on ({model})")
            # The embedding model can change (the preferred one sometimes
            # stops answering), and vectors from different models cannot be
            # compared, so the old ones are re-embedded in the background.
            asyncio.create_task(_reembed_if_needed(model))
        else:
            print("long-term memory: OFF - run supabase_memory.sql in the "
                  "Supabase SQL editor")
    except Exception as exc:
        print(f"long-term memory: unavailable ({exc})")

    # Say whether Kaggle is on. Report exactly where the credentials came
    # from and whether a live authentication worked, so you can tell at a
    # glance whether the token reached the process.
    try:
        from kaggle_backend import credentials_source, kaggle_status, status_hint

        status = kaggle_status()
        # Warm the status cache here, so the first message after a deploy
        # does not have to wait for a live Kaggle check.
        status_hint(report=status)
        if status["authenticated"]:
            print(f"kaggle: on (credentials from {status['credentials_source']})")
        else:
            src = status["credentials_source"] or "none"
            print(f"kaggle: OFF - credentials from {src}, but authentication failed:")
            print(f"  {status['error']}")
            print("  Set KAGGLE_API_TOKEN in Render's env vars and redeploy, or "
                  "the built-in fallback should already be in kaggle_backend.py.")
    except Exception as exc:
        print(f"kaggle: unavailable ({exc})")

    from jobs import _load_job_state, report_interrupted_jobs
    _load_job_state()

    # Jobs that were running when the process stopped are gone. Say so,
    # rather than leaving the user waiting for a result that never comes.
    for job in report_interrupted_jobs():
        try:
            channel = bot.get_channel(int(job["channel_id"])) if job.get("channel_id") else None
            if channel is None:
                continue
            await channel.send(
                f"⚠️ **Job {job_label(job['id'])} was interrupted** — the bot "
                f"restarted while it was running, so its results were lost.\n"
                f"-# task was: {job['topic'][:120]}\n"
                f"-# start it again with: !research {job['topic'][:120]}"
            )
        except Exception as exc:
            print(f"could not report interrupted job: {exc}")
    print("------")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    # Ignore other bots to avoid feedback loops.
    if message.author.bot:
        return

    channel_id = str(message.channel.id)

    # Remember that this person is in this channel. It is done for every
    # message, including the ones the bot stays quiet for, because that is
    # what tells a one-to-one channel apart from a room.
    TRACKER.note(channel_id, message.author.id)

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
        chat_on = await db_is_chat_mode_enabled(channel_id)
    except Exception as exc:
        print(f"chat mode lookup failed: {exc}")
        chat_on = False

    mentioned = bot.user.mentioned_in(message)
    is_dm = isinstance(message.channel, discord.DMChannel)

    # A channel that has other people in it is a room, not a private chat.
    if not is_dm:
        await _channel_is_shared(channel_id, message.channel)

    # Who is in the room, and who is talking. Read from the channel, so the
    # bot always has the names it needs without anybody being listed in the
    # code.
    roster = await _channel_roster(channel_id, message.channel)
    speaker = person_entry(message.author)
    bot_names = bot_name_variants(bot.user)

    async def _decide():
        return await decide(
            message,
            is_dm=is_dm,
            shared=TRACKER.is_shared(channel_id) if not is_dm else False,
            bot_user_id=bot.user.id,
            bot_names=bot_names,
            bot_message_ids=BOT_MESSAGE_IDS.get(channel_id),
            chat_on=chat_on,
            roster=roster,
            author_name=speaker.get("name") or "",
            roster_line=roster_line(roster),
            bot_name=display_name(bot.user),
        )

    decision = await _decide()
    if not decision.reply:
        print(f"staying quiet in {channel_id}: {decision.reason}")
        return

    content = message.content
    if mentioned:
        content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    if not content:
        content = "Hello!"

    async with _channel_lock(channel_id):
        # The room may have changed while this turn waited for its turn -
        # somebody else may have spoken, which makes it a shared channel. The
        # decision is taken again now, so the bot does not answer a message
        # that was plainly meant for somebody else. The classifier's answer
        # is cached, so the second look costs nothing.
        decision = await _decide()
        if not decision.reply:
            print(f"staying quiet in {channel_id}: {decision.reason}")
            return

        shared = TRACKER.is_shared(channel_id) if not is_dm else False
        scopes = [str(message.author.id)]
        write_scope = str(message.author.id)
        if shared:
            # A room: everybody's messages belong to the same thread, and the
            # long-term memory is shared too.
            scope = channel_scope(channel_id)
            scopes.append(scope)
            write_scope = scope
            SHARED_CHANNELS[channel_id] = True
            TRACKER.mark_shared(channel_id)

        async with message.channel.typing():
            result = await get_nvidia_response_full(
                message.author.id,
                content,
                message.channel,
                attachments=message.attachments,
                scopes=scopes,
                write_scope=write_scope,
                shared=shared,
                speaker=speaker,
                roster=roster,
            )
        sent = await send_long(message, result["answer"], footer=result["footer"])
        _remember_bot_messages(channel_id, sent)
        await attach_images(message, result.get("files") or [])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# Guarded, so that the module can be imported by a test without the bot
# trying to log in.
if __name__ == "__main__":
    keep_alive()
    bot.run(os.environ["DISCORD_TOKEN"])
