"""Background agent tasks.

A job is a long-running task handed to a specialist agent from the registry.
It runs in the background while the bot stays responsive, reports progress as
it goes, and delivers the final answer (plus any files it produced) when it
finishes.
"""

import asyncio
import json
import os
import re
import time
from difflib import SequenceMatcher

import discord

from agents import AGENTS, get_agent
from config import JOB_TICK_SECONDS
from discord_utils import DISCORD_MAX_LEN, fit_discord, send_long
from agent import run_agent_loop
from storage import db_add_message
from tools import written_files

JOBS: dict[str, dict] = {}
_job_counter = 0
_job_lock = asyncio.Lock()

# How many background jobs may run at the same time. Each one drives a full
# agent loop with its own tool budget, so a pile of them at once starves the
# interactive replies.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

# Two topics that are this alike are the same task.
_SAME_TOPIC_RATIO = 0.85

JOB_STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bot_files", ".jobs.json"
)


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
        os.makedirs(os.path.dirname(JOB_STATE_FILE), exist_ok=True)
        with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"counter": _job_counter}, f)
    except Exception:
        pass


def _save_running_jobs():
    """Record which jobs were running, so a restart can report them lost."""
    try:
        os.makedirs(os.path.dirname(JOB_STATE_FILE), exist_ok=True)
        running = [
            {"id": j["id"], "topic": j["topic"], "agent": j.get("agent", "research"),
             "channel_id": j.get("channel_id")}
            for j in JOBS.values() if j.get("status") == "running"
        ]
        with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"counter": _job_counter, "running": running}, f)
    except Exception:
        pass


def _load_running_jobs() -> list[dict]:
    """Jobs that were running when the process last stopped."""
    try:
        with open(JOB_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("running", [])
    except Exception:
        return []


async def next_job_id() -> str:
    global _job_counter
    async with _job_lock:
        _job_counter += 1
        _save_job_state()
        return f"job_{_job_counter}"


def job_label(job_id: str) -> str:
    """Human-facing label for a job id, e.g. job_3 -> #3."""
    return job_id.replace("job_", "#")


def _set_status(job_id: str, status: str):
    """Update a job's status, tolerating the job having been removed."""
    job = JOBS.get(job_id)
    if job is not None:
        job["status"] = status


# How much of an agent's summary to keep in the conversation history.
_SUMMARY_LIMIT = 3000


async def record_job_result(user_id, topic: str, agent_name: str, summary: str,
                             files: list[str], scopes: list[str] = None):
    """Store an agent's outcome in the conversation history.

    The agent runs detached from the conversation, so without this the main
    model has no idea the work happened - it will happily contradict its own
    report on the next question.

    In a room the result is stored against the channel as well, so the other
    people there know about it too.
    """
    if not user_id and not scopes:
        return
    try:
        file_names = ", ".join(os.path.basename(p) for p in files) or "none"
        body = (
            f"[background {agent_name} agent finished]\n"
            f"Task: {topic}\n"
            f"Files produced: {file_names}\n\n"
            f"Findings:\n{summary[:_SUMMARY_LIMIT]}"
        )
        for scope in (scopes or [user_id]):
            if scope:
                await db_add_message(scope, "assistant", body)
    except Exception:
        pass


async def _run_job(job_id: str, channel, user_id: str, topic: str, agent_name: str):
    """Run a background agent task and post progress + result."""
    spec = get_agent(agent_name) or AGENTS["research"]
    tracker: list[str] = []
    token = written_files.set(tracker)

    started = time.monotonic()
    last_progress = 0.0

    async def report(step, total, labels):
        """Post a progress line, at most once every JOB_TICK_SECONDS.

        The agent loop also reports each step, but for a long job that is
        one message per step - dozens of lines saying the same thing. This is
        the heartbeat that says how far along the job is.
        """
        nonlocal last_progress
        now = time.monotonic()
        if last_progress and now - last_progress < JOB_TICK_SECONDS:
            return
        last_progress = now
        elapsed = int(now - started)
        mins, secs = divmod(elapsed, 60)
        try:
            await channel.send(
                f"-# step {step} of {total} · {mins}m{secs:02d}s elapsed · "
                f"{' '.join(labels)}"
            )
        except Exception:
            pass

    try:
        system_prompt = spec["prompt"] + f"\n\nThe task: {topic}"

        # A delegated agent can call kaggle_run too, so it gets the live
        # status as well. Without it, an agent that was told to use Kaggle
        # may refuse on the strength of an old remembered failure.
        has_kaggle = any(
            t.get("function", {}).get("name", "").startswith("kaggle")
            for t in spec.get("tools", [])
        )
        if has_kaggle:
            try:
                import kaggle_backend

                hint = await asyncio.to_thread(kaggle_backend.status_hint)
                if hint:
                    system_prompt += (
                        "\n\nLive check of Kaggle just now: " + hint +
                        " This status is current and authoritative: if it says "
                        "Kaggle is working, use kaggle_run for heavy compute, "
                        "and do not refuse it because of an older remembered "
                        "note about failed runs."
                    )
            except Exception as exc:
                print(f"kaggle status check failed: {exc}")

        messages = [{
            "role": "system",
            "content": system_prompt,
        }]

        result = await run_agent_loop(
            messages,
            user_id=None,
            channel=channel,
            max_steps=spec["max_steps"],
            tools=spec["tools"],
            persist=False,
            on_step=report,
            max_tokens=spec["max_tokens"],
        )

        answer = result["answer"]
        files = [p for p in tracker if os.path.exists(p)]

        header = f"**Job {job_label(job_id)} is done** — {fit_discord(topic, 400)}\n"
        if files:
            header += (
                f"Delivered as {len(files)} file"
                f"{'s' if len(files) != 1 else ''}: "
                + ", ".join(os.path.basename(p) for p in files)
                + "\n"
            )

        summary = answer if answer.strip() else "(job finished with no summary)"
        await send_long(channel, header + summary)

        # Record the outcome in the user's conversation history, so the main
        # model knows the agent ran and what it found. Without this the bot
        # forgets its own report and contradicts it on the next question.
        scopes = []
        if user_id:
            scopes.append(user_id)
        if channel is not None:
            try:
                from presence import channel_scope
                from storage import db_scope_has_rows

                scope = channel_scope(channel.id)
                if await db_scope_has_rows(scope):
                    scopes.append(scope)
            except Exception:
                pass
        await record_job_result(user_id, topic, agent_name, summary, files,
                                scopes=scopes)

        for path in files:
            try:
                await channel.send(file=discord.File(path))
            except Exception as exc:
                await channel.send(f"⚠️ Could not attach {os.path.basename(path)}: {exc}")

    except asyncio.CancelledError:
        try:
            await channel.send(f"🛑 **Job {job_label(job_id)} cancelled.**")
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            await channel.send(f"⚠️ **Job {job_label(job_id)} failed:** {exc}")
        except Exception:
            pass
    finally:
        written_files.reset(token)


def _running_jobs() -> list[dict]:
    """The jobs that are running right now."""
    return [job for job in JOBS.values() if job.get("status") == "running"]


def _topic_key(topic: str) -> str:
    return " ".join(re.findall(r"\w+", (topic or "").lower()))


def _same_topic(a: str, b: str) -> bool:
    """Are these two topics the same task?"""
    key_a, key_b = _topic_key(a), _topic_key(b)
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return True
    return SequenceMatcher(None, key_a, key_b).ratio() >= _SAME_TOPIC_RATIO


async def start_agent_task(agent_name: str, topic: str, channel=None,
                            user_id=None) -> str:
    """Start a background agent task and return a short status line.

    This is the handler behind the `start_agent_task` tool, so the model can
    decide on its own that a request needs a specialist.
    """
    if not topic.strip():
        return "Error: no topic given."

    spec = get_agent(agent_name)
    if spec is None:
        return f"Error: unknown agent '{agent_name}'. Available: {', '.join(AGENTS)}."

    # Two jobs doing the same work at once is wasted compute and a doubled
    # answer in the channel. One is already running: point at it instead.
    for job in _running_jobs():
        if job.get("agent") == agent_name and _same_topic(job.get("topic", ""), topic):
            return (
                f"Job {job_label(job['id'])} is already running this exact task "
                f"({fit_discord(job.get('topic', ''), 160)}). It was NOT started "
                f"again - a second copy would only repeat the work and post a "
                f"duplicate answer. Tell the user it is already running and "
                f"that the result will appear in this channel."
            )

    running = _running_jobs()
    if len(running) >= MAX_CONCURRENT_JOBS:
        return (
            f"There are already {len(running)} background jobs running and the "
            f"limit is {MAX_CONCURRENT_JOBS}. The new task was NOT started, "
            f"because running it now would slow down the jobs already in "
            f"flight. Tell the user to wait for one of them to finish, then "
            f"start it again."
        )

    job_id = await next_job_id()
    JOBS[job_id] = {
        "id": job_id,
        "agent": agent_name,
        "topic": topic,
        "user_id": user_id,
        "channel_id": getattr(channel, "id", None),
        "status": "running",
        "started_at": time.time(),
    }

    if channel is not None:
        # The topic can be thousands of characters long, and Discord rejects
        # any message over 2000 characters ("Invalid Form Body"). That used to
        # kill the whole tool call, so the job was created but never ran and
        # the model had to retry.
        title = f"**⏳ Job {job_label(job_id)} started** — "
        footer = (
            f"\n-# {spec['label']} · up to {spec['max_steps']} steps · "
            f"progress updates every {JOB_TICK_SECONDS}s · results will post here"
        )
        room = DISCORD_MAX_LEN - len(title) - len(footer) - 1
        try:
            await channel.send(title + fit_discord(topic, room) + footer)
        except Exception as exc:
            # A failure to announce the job must not stop the job itself.
            print(f"could not announce job {job_id}: {exc}")

    async def runner():
        try:
            await _run_job(job_id, channel, user_id, topic, agent_name)
            _set_status(job_id, "done")
        except asyncio.CancelledError:
            _set_status(job_id, "cancelled")
            raise
        except Exception as exc:
            _set_status(job_id, "failed")
            try:
                await channel.send(f"⚠️ Job {job_label(job_id)} failed: {exc}")
            except Exception:
                pass
        finally:
            _save_running_jobs()

    task = asyncio.create_task(runner())
    JOBS[job_id]["task"] = task
    task.add_done_callback(lambda t: JOBS.get(job_id, {}).pop("task", None))
    _save_running_jobs()

    return (
        f"Background {spec['label'].lower()} task {job_label(job_id)} started. It is "
        f"running now and will post its result (and any files) in this channel when "
        f"it finishes. Tell the user it has started and that results will appear "
        f"here - do not do the work yourself."
    )


def report_interrupted_jobs() -> list[dict]:
    """Jobs that were running when the process last stopped.

    Their work is gone, so the caller should tell the user rather than
    leaving them waiting for a result that will never arrive.
    """
    running = _load_running_jobs()
    if running:
        try:
            with open(JOB_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"counter": _job_counter, "running": []}, f)
        except Exception:
            pass
    return running


def list_jobs() -> list[dict]:
    """All known jobs, newest last."""
    return list(JOBS.values())


def cancel_job(job_id: str) -> str:
    """Cancel a running job. Returns a status message for the user."""
    job = JOBS.get(job_id)
    if job is None:
        # Accept "#3" as well as "job_3".
        for key in JOBS:
            if job_label(key) == job_id:
                job = JOBS[key]
                job_id = key
                break
    if job is None:
        return f"No such job: {job_id}"

    task = job.get("task")
    if task is None:
        return f"Job {job_label(job_id)} is not running (status: {job.get('status')})."

    task.cancel()
    job["status"] = "cancelled"
    return f"Job {job_label(job_id)} cancelled."
