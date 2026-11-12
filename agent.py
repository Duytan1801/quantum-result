"""The agent loop: prompt building, context assembly, and tool-calling."""

import asyncio
import json
import os

from config import (
    CONTEXT_CHAR_BUDGET,
    HISTORY_ROWS,
    MAX_AGENT_STEPS,
    NVIDIA_MODEL,
    provider_chain,
)
from agents import AGENTS, TOOLS, agent_catalogue
from storage import db_add_message, db_get_history, db_get_user_facts
from tools import written_files
from vision import build_image_content
from tools import (
    draw_diagram,
    edit_file,
    read_file,
    run_code_in_sandbox,
    verify_claims,
    web_search,
    wolfram_alpha,
    write_file,
)

# Status label shown in Discord for each tool.
STATUS_LABELS = {
    "code": "💻 Running code...",
    "wolfram_alpha": "🧮 Computing...",
    "web_search": "🔍 Searching the web...",
    "read_file": "📖 Reading file...",
    "write_file": "📝 Writing file...",
    "edit_file": "✏️ Editing file...",
    "verify_claims": "🧮 Recomputing figures...",
    "draw_diagram": "🎨 Drawing a diagram...",
    "start_agent_task": "⏳ Starting a background agent...",
}

# Words that signal the answer depends on the world as it is *now*. If the
# user asks anything matching these, the model is told to search before it
# answers rather than trusting its training data.
_FRESHNESS_MARKERS = (
    "latest", "recent", "recently", "today", "tonight", "now", "right now",
    "currently", "current", "these days", "nowadays", "this week", "this month",
    "this year", "this morning", "this afternoon", "this evening", "so far",
    "yesterday", "tomorrow", "up to date", "up-to-date", "as of",
    "news", "price", "prices", "share price", "stock price", "market cap",
    "market", "earnings", "revenue", "guidance", "forecast", "outlook",
    "who is the ceo", "who is the cto", "who is the founder", "still alive",
    "alive", "dead", "died", "announced", "announced", "released", "launched",
    "acquired", "merger", "ipo", "layoffs", "version", "release",
)

# Explicit year mentions - a question pinned to a year is about that period.
_RECENT_YEARS = ("2024", "2025", "2026", "2027")


def needs_fresh_data(message: str) -> bool:
    """Does this question depend on information that may be out of date?"""
    text = (message or "").lower()
    if any(marker in text for marker in _FRESHNESS_MARKERS):
        return True
    return any(year in text for year in _RECENT_YEARS)


FRESHNESS_REMINDER = (
    "This question may depend on information that has changed since your training "
    "data. Before you answer:\n"
    "1. Use web_search to get current, authoritative sources. Do not answer from "
    "memory.\n"
    "2. Prefer a primary source (the company's own filings, the official "
    "announcement, the regulator) over a summary of one.\n"
    "3. If sources disagree, report the disagreement rather than picking one "
    "silently.\n"
    "4. Run every derived figure through verify_claims before you state it, and "
    "correct anything that does not add up.\n"
    "5. Say so plainly if you could not verify something."
)


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
        "When to delegate to a background agent:\n"
        "- You have specialist agents available. When a request needs more than a "
        "couple of steps, pick the agent that fits and call start_agent_task with "
        "its name and a precise description of the task.\n"
        "AGENTS:\n"
        f"{agent_catalogue()}\n"
        "- Prefer delegating when it fits: a specialist with its own budget and "
        "workspace will do the job better and more thoroughly than you can in one "
        "turn, and the user keeps chatting while it works.\n"
        "- Your own code tool is for running a short snippet to compute an answer. "
        "If the user wants a program, a script, a tool or a file that they will "
        "run, delegate to the code agent instead of writing it yourself.\n"
        "- Your own file tools are for reading and for small notes. If the "
        "deliverable is a report, a model, a dataset or a program, delegate.\n"
        "- The agent runs in the background and posts its result with any files "
        "attached when it finishes. Do not do that work yourself in this turn.\n"
        "- After delegating, reply with one short line naming the agent you started "
        "and saying results will be posted here.\n"
        "- Answer directly instead of delegating only for single lookups, quick "
        "facts, and maths - anything one tool call can settle."
        "\n\n"
        "Images:\n"
        "- You can see any pictures attached to the message. Describe them "
        "accurately: read charts, tables, screenshots, diagrams and text in "
        "images, and quote the numbers you can actually make out.\n"
        "- If an image is unclear or cut off, say so rather than guessing what "
        "it says.\n"
        "- If a picture shows figures, read the axis labels, units and scale "
        "before you interpret the shape of the data, and recompute any "
        "derived quantity with verify_claims."
        "\n\n"
        "Sourcing rules - these are not optional:\n"
        "- If a question could have a different answer today than it did a year "
        "ago, you MUST search before answering. Prices, financials, earnings, "
        "headcounts, prices, product versions, who holds a role, what was "
        "announced, market conditions, anything 'latest' or 'current'. Your "
        "training data is not a source for these.\n"
        "- Prefer primary sources: the company's own filings, its press release, "
        "the regulator, the official documentation. Use secondary coverage to "
        "interpret, not to establish the fact.\n"
        "- Date every figure. If you do not know when a source was published, "
        "say that instead of presenting it as current.\n"
        "- When sources conflict, show the conflict and both numbers rather "
        "than silently picking one.\n"
        "\n"
        "Verification rules - these are not optional:\n"
        "- Never state a derived figure you have not actually computed. Growth "
        "rates, percentages, totals, differences, ratios, sums, averages, "
        "differences between two numbers, anything you did to get from a source "
        "figure: compute it with verify_claims, code or wolfram_alpha, and use "
        "the computed value.\n"
        "- Use verify_claims to check the figures you are about to publish: pass "
        "the expression and the value you intend to state, and correct any "
        "mismatch before you write the answer.\n"
        "- When you compute a figure from a source's raw numbers, recompute it "
        "rather than trusting a percentage the source printed - sources are "
        "often wrong, and sometimes deliberately round.\n"
        "- If a recomputation contradicts what a source claimed, trust the "
        "recomputation and flag the discrepancy.\n"
        "- Keep the arithmetic visible in the answer so the user can check it."
        "\n\n"
        "Honesty rules:\n"
        "- Ground every factual claim in what a tool actually returned. If a tool "
        "did not give you the number, date, or name, do not state it from memory. "
        "Say what is missing.\n"
        "- Distinguish what a source reported from what you inferred. Flag estimates "
        "explicitly, and flag it when two sources disagree rather than silently "
        "picking one.\n"
        "- If you could not verify something, say so plainly. Never fill a gap with a "
        "plausible-sounding number.\n"
        "- When you finish, offer one concrete next step you could take (a specific search, "
        "a specific analysis, or a specific file) so the user can push the work forward."
        + facts_text
    )


def history_to_messages(history: list) -> list:
    """Rebuild API messages from stored conversation rows.

    Two things can break a request:
      * a "tool" row whose assistant turn is missing - the API rejects a tool
        result with no matching call;
      * an assistant turn whose tool_calls have no results, which happens when
        the fetched window cuts the conversation in the middle of a tool-call
        group. The API rejects that too, so those calls are stripped.
    """
    # Which tool calls actually have a result stored?
    answered = {row.get("tool_call_id") for row in history if row.get("tool_call_id")}

    messages = []
    pending_tool_ids = set()
    for row in history:
        role = row.get("role")
        msg = {"role": role, "content": row.get("content") or ""}

        if role == "assistant" and row.get("tool_calls"):
            calls = [tc for tc in row["tool_calls"] if tc.get("id") in answered]
            if calls:
                msg["tool_calls"] = calls
                pending_tool_ids = {tc["id"] for tc in calls}

        if row.get("tool_call_id"):
            if row["tool_call_id"] not in pending_tool_ids:
                continue  # orphaned tool result, skip it
            pending_tool_ids.discard(row["tool_call_id"])
            msg["tool_call_id"] = row["tool_call_id"]
            msg["name"] = row.get("name")

        messages.append(msg)
    return messages


def _footer_for(result: dict) -> str:
    """Build the small grey footer line: model + tools used."""
    parts = [NVIDIA_MODEL]
    if result.get("tools_used"):
        parts.append(" · ".join(result["tools_used"]))
    return " · ".join(parts)


def _check_delegation_tool() -> bool:
    """Warn loudly if the delegation tool and its handler disagree.

    A mismatch here previously killed every turn silently: the tool raised
    ImportError, nothing caught it, and the bot went quiet mid-conversation.
    We only warn rather than exiting, so a bad deploy degrades one feature
    instead of taking the whole bot offline.
    """
    tool_names = {t["function"]["name"] for t in TOOLS}
    if "start_agent_task" not in tool_names:
        print("ERROR: start_agent_task is missing from the tool list - the model "
              "would call a tool that does not exist.")
        return False
    try:
        from jobs import start_agent_task  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: jobs.start_agent_task could not be imported ({exc}). "
              "The model calls start_agent_task, so this must exist.")
        return False
    return True


async def _run_tool(name: str, args: dict, channel=None, user_id=None) -> str:
    """Execute one tool by name with already-parsed arguments."""
    if name == "wolfram_alpha":
        return await wolfram_alpha(args.get("query", ""))
    if name == "code":
        return await run_code_in_sandbox(args.get("code", ""), args.get("language", "python"))
    if name == "web_search":
        return await web_search(
            args.get("query", ""),
            args.get("count", 5),
            args.get("freshness", "noLimit"),
        )
    if name == "read_file":
        return await read_file(args.get("file_path", ""))
    if name == "write_file":
        return await write_file(args.get("file_path", ""), args.get("content", ""))
    if name == "edit_file":
        return await edit_file(
            args.get("file_path", ""),
            args.get("old_string", ""),
            args.get("new_string", ""),
        )
    if name == "draw_diagram":
        return await draw_diagram(
            args.get("spec") or args.get("diagram") or args,
            args.get("filename", "diagram.png"),
        )
    if name == "verify_claims":
        return await verify_claims(args.get("checks") or [])
    if name == "start_agent_task":
        # Imported lazily: jobs imports this module, so a top-level import
        # would be circular.
        from jobs import start_agent_task

        return await start_agent_task(
            args.get("agent", "research"),
            args.get("topic", ""),
            channel,
            user_id,
        )
    return f"Error: Unknown tool {name}"


def _is_failure(result: str) -> bool:
    """Did this tool call fail? Used to break out of retry loops."""
    return (
        result.startswith("Error:")
        or "API error" in result[:120]
        or result.startswith("Wolfram Alpha could not")
    )


def _message_size(m) -> int:
    """Rough size of a message for the context budget.

    Image parts are counted as a flat estimate, not their base64 length - a
    data URL can be megabytes and would blow the budget on its own.
    """
    content = m.get("content")
    if isinstance(content, list):
        size = sum(
            len(str(p.get("text") or "")) + 2000
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
        size += sum(2000 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        return size + len(str(m.get("tool_calls") or ""))
    return len(str(content or "")) + len(str(m.get("tool_calls") or ""))


def _transcript(messages: list, limit: int = 60000) -> str:
    """Render messages as plain text so they can be summarised."""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = str(m.get("content") or "")
        if role == "tool":
            content = f"[{m.get('name') or 'tool'}] {content}"
        elif role == "assistant" and m.get("tool_calls"):
            names = ", ".join(t["function"]["name"] for t in m["tool_calls"])
            content = f"(calling {names}) {content}".strip()
        if content.strip():
            lines.append(f"{role.upper()}: {content[:4000]}")
    text = "\n\n".join(lines)
    return text[:limit] + ("\n\n[...truncated]" if len(text) > limit else "")


async def summarize_messages(messages: list) -> str:
    """Summarise a block of conversation so the model can keep going without it."""
    if not messages:
        return ""
    transcript = _transcript(messages)
    if not transcript.strip():
        return ""

    request = {
        "messages": [
            {
                "role": "system",
                # Keep this prompt short. A long, heavily-specified prompt makes
                # the reasoning model think for thousands of tokens and return
                # an empty summary when the budget runs out.
                "content": (
                    "List every distinct thing the user asked for (numbered, one per "
                    "line). Then key findings with sources. Then decisions and "
                    "preferences. Then open items. Terse, no preamble."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.2,
    }

    last_error = None
    for _name, client, model in provider_chain():
        # The model's max output is 131072. Give the summary that much room:
        # it is a reasoning model, and a smaller budget makes it spend the
        # tokens thinking and return an empty summary.
        try:
            response = await client.chat.completions.create(
                model=model, max_completion_tokens=131072, **request
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            last_error = "empty summary"
        except Exception as exc:
            last_error = exc
            continue
    print(f"could not summarise conversation: {last_error}")
    return ""


async def compact_history(messages: list, budget: int = None,
                          keep_recent: int = None) -> list:
    """Summarise older turns instead of dropping them (Claude-style).

    The most recent part of the conversation is kept verbatim; everything
    before it is replaced by a single summary. The summary is carried forward
    on later turns, so the model never loses the thread.
    """
    if budget is None:
        from config import CONTEXT_CHAR_BUDGET
        budget = CONTEXT_CHAR_BUDGET
    if keep_recent is None:
        from config import KEEP_RECENT_CHARS
        keep_recent = KEEP_RECENT_CHARS

    if sum(_message_size(m) for m in messages) <= budget:
        return messages

    # Leading system messages are always kept.
    head = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        head.append(messages[i])
        i += 1
    body = messages[i:]

    # Walk back from the end, keeping the most recent turns verbatim.
    keep_from = len(body)
    acc = 0
    for idx in range(len(body) - 1, -1, -1):
        acc += _message_size(body[idx])
        if acc >= keep_recent:
            keep_from = idx
            break
    older = body[:keep_from]
    recent = body[keep_from:]

    if not older:
        return messages

    summary = await summarize_messages(older)
    if not summary:
        # Could not summarise - fall back to dropping the oldest tool groups.
        return trim_context(messages, budget)

    compacted = head + [
        {"role": "system", "content": f"[Earlier conversation, summarised]\n{summary}"}
    ] + recent
    return compacted


def trim_context(messages: list, budget: int = None) -> list:
    """Drop the oldest tool-calling turns when the context grows too large.

    With a high step limit a single turn can accumulate megabytes of tool
    output and blow the model's context window. Assistant turns that requested
    tools are dropped together with their results, so a tool result is never
    separated from the call that produced it.
    """
    if budget is None:
        from config import CONTEXT_CHAR_BUDGET
        budget = CONTEXT_CHAR_BUDGET

    if sum(_message_size(m) for m in messages) <= budget:
        return messages

    # Leading system messages are always kept.
    head = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        head.append(messages[i])
        i += 1
    body = messages[i:]

    # Find each assistant turn that requested tools, and the results after it.
    groups = []
    j = 0
    while j < len(body):
        if body[j].get("role") == "assistant" and body[j].get("tool_calls"):
            k = j + 1
            while k < len(body) and body[k].get("role") == "tool":
                k += 1
            groups.append((j, k))
            j = k
        else:
            j += 1

    if not groups:
        return messages

    # Drop the oldest groups until the context fits. Collect the indices
    # first - deleting as we go would shift the later group positions.
    dropped = set()
    for start, end in groups:
        kept = [m for i, m in enumerate(body) if i not in dropped]
        if sum(_message_size(m) for m in head + kept) <= budget:
            break
        dropped.update(range(start, end))

    kept = [m for i, m in enumerate(body) if i not in dropped]
    return head + kept


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

        # Keep the request inside the model's context window.
        messages = trim_context(messages)

        # Try NVIDIA first, fall back to Groq if NVIDIA is unavailable.
        stream = None
        last_error = None
        for _name, client, model in provider_chain():
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=max_tokens,
                    top_p=0.95,
                    stream=True,
                    tools=tools,
                    tool_choice="auto",
                )
                break
            except Exception as exc:
                last_error = exc
                continue

        if stream is None:
            if channel:
                await channel.send(f"⚠️ LLM API error: {str(last_error)[:150]}")
            return {
                "answer": final_answer or "LLM API error",
                "tools_used": tools_used,
                "steps": step_count,
            }

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
            return {
                "answer": final_answer or "LLM stream error",
                "tools_used": tools_used,
                "steps": step_count,
            }

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
                await db_add_message(user_id, "assistant", step_content or "",
                                     tool_calls=tool_calls_list)
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
                messages.append({
                    "role": "user",
                    "content": "Continue your response from where you left off.",
                })
                continue
            break

        # Send one status message per step, not per tool call. The model often
        # requests several searches in one step, which produced duplicate
        # "Step N - Searching the web..." lines.
        step_labels = []
        for tc in tool_calls_accum.values():
            label = STATUS_LABELS.get(tc["name"], f"Running {tc['name']}...")
            if label not in step_labels:
                step_labels.append(label)
            if tc["name"] not in tools_used:
                tools_used.append(tc["name"])

        if channel:
            await channel.send(f"**Step {step_count}** - {' '.join(step_labels)}")
        if on_step is not None:
            await on_step(step_count, max_steps, step_labels)

        step_failed = False
        for tc in tool_calls_accum.values():
            try:
                args = json.loads(tc["arguments"])
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
                result = f"Error: Invalid JSON arguments for {tc['name']} tool"
            else:
                # A tool that raises must not kill the whole turn - the model
                # needs to see the error so it can recover or answer directly.
                try:
                    result = await _run_tool(tc["name"], args, channel, user_id)
                except Exception as exc:
                    result = f"Error: {tc['name']} tool failed: {exc}"
                    if channel:
                        await channel.send(
                            f"⚠️ The `{tc['name']}` tool failed: {str(exc)[:150]}"
                        )

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": result,
            })
            if persist and user_id is not None:
                await db_add_message(user_id, "tool", result,
                                     tool_call_id=tc["id"], name=tc["name"])

            if _is_failure(result):
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


async def get_nvidia_response_full(user_id, user_message, channel=None, attachments=None):
    """Answer a message, persist the exchange, and return the answer plus metadata."""
    _check_delegation_tool()
    history = await db_get_history(user_id, limit=HISTORY_ROWS)
    user_facts = await db_get_user_facts(user_id)

    messages = [{"role": "system", "content": build_system_prompt(user_facts)}]
    messages.extend(history_to_messages(history))

    # If the user attached pictures, hand them to the vision model.
    content = user_message
    stored = user_message
    if attachments:
        try:
            content, note = await build_image_content(attachments, user_message)
            stored = user_message + note
        except Exception as exc:
            print(f"could not prepare images: {exc}")
            content = user_message
            stored = user_message

    messages.append({"role": "user", "content": content})
    await db_add_message(user_id, "user", stored)

    # If the question depends on the world as it is now, remind the model to
    # source and verify before answering. Cheap to add, and it stops the model
    # confidently answering from stale training data.
    if needs_fresh_data(user_message):
        messages.insert(1, {"role": "system", "content": FRESHNESS_REMINDER})

    # Pull in anything worth remembering from earlier conversations.
    try:
        from memory import format_memories, is_available, retrieve_memories

        if await is_available():
            memories = await retrieve_memories(user_id, user_message, top_k=6)
            memories_text = format_memories(memories)
            if memories_text:
                messages.insert(1, {"role": "system", "content": memories_text})
    except Exception as exc:
        print(f"memory retrieval failed: {exc}")

    # Keep the whole conversation, but compact it if it has grown too large.
    # Older turns are summarised rather than dropped, so the model keeps the
    # thread of what the user wants.
    messages = await compact_history(messages)

    # Track any files written during this turn so generated images (diagrams,
    # plots) can be attached to the reply automatically.
    tracker: list[str] = []
    token = written_files.set(tracker)
    try:
        result = await run_agent_loop(
            messages,
            user_id=user_id,
            channel=channel,
            max_steps=MAX_AGENT_STEPS,
        )
    finally:
        written_files.reset(token)

    result["files"] = [p for p in tracker if os.path.exists(p)]
    result["footer"] = _footer_for(result)

    # Learn from this exchange in the background, so it never delays the reply.
    _schedule_memory_update(user_id, user_message, result["answer"])

    return result


def _schedule_memory_update(user_id, user_message: str, answer: str):
    """Extract durable facts from the exchange and store them."""
    async def run():
        try:
            from memory import extract_memories, is_available, store_memories

            if not await is_available():
                return
            memories = await extract_memories(
                f"USER: {user_message}\n\nASSISTANT: {answer}"
            )
            if memories:
                await store_memories(user_id, memories)
        except Exception as exc:
            print(f"memory update failed: {exc}")

    try:
        asyncio.create_task(run())
    except RuntimeError:
        pass


async def get_nvidia_response(user_id, user_message, channel=None):
    """Answer a message and persist the exchange. Returns the answer text."""
    result = await get_nvidia_response_full(user_id, user_message, channel)
    return result["answer"]
