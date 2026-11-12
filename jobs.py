"""Background agent tasks.

A job is a long-running task handed to a specialist agent from the registry.
It runs in the background while the bot stays responsive, reports progress as
it goes, and delivers the final answer (plus any files it produced) when it
finishes.
"""

import asyncio
import json
import os
import time

import discord

from agents import AGENTS, get_agent
from config import JOB_TICK_SECONDS
from discord_utils import send_long
from agent import run_agent_loop
from storage import db_add_message
from tools import written_files

JOBS: dict[str, dict] = {}
_job_counter = 0
_job_lock = asyncio.Lock()

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
                             files: list[str]):
    """Store an agent's outcome in the user's conversation history.

    The agent runs detached from the conversation, so without this the main
    model has no idea the work happened - it will happily contradict its own
    report on the next question.
    """
    if not user_id:
        return
    try:
        file_names = ", ".join(os.path.basename(p) for p in files) or "none"
        body = (
            f"[background {agent_name} agent finished]\n"
            f"Task: {topic}\n"
            f"Files produced: {file_names}\n\n"
            f"Findings:\n{summary[:_SUMMARY_LIMIT]}"
        )
        await db_add_message(user_id, "assistant", body)
    except Exception:
        pass


async def _run_job(job_id: str, channel, user_id: str, topic: str, agent_name: str):
    """Run a background agent task and post progress + result."""
    spec = get_agent(agent_name) or AGENTS["research"]
    tracker: list[str] = []
    token = written_files.set(tracker)

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
            "content": spec["prompt"] + f"\n\nThe task: {topic}",
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

        header = f"**Job {job_label(job_id)} is done** — {topic}\n"
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
        await record_job_result(user_id, topic, agent_name, summary, files)

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
        await channel.send(
            f"**⏳ Job {job_label(job_id)} started** — {topic}\n"
            f"-# {spec['label']} · up to {spec['max_steps']} steps · "
            f"progress updates every {JOB_TICK_SECONDS}s · results will post here"
        )

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
