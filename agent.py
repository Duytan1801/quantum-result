"""The agent loop: prompt building, context assembly, and tool-calling."""

import json

from config import MAX_AGENT_STEPS, MAX_HISTORY, NVIDIA_MODEL, provider_chain
from agents import AGENTS, TOOLS, agent_catalogue
from storage import db_add_message, db_get_history, db_get_user_facts
from tools import (
    edit_file,
    read_file,
    run_code_in_sandbox,
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
    "start_research_job": "⏳ Starting a background research job...",
}


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
        "How to answer well:\n"
        "- Ground every factual claim in what a tool actually returned. If a tool did not "
        "give you the number, date, or name, do not state it from memory. Say what is missing.\n"
        "- Distinguish what a source reported from what you inferred. Flag estimates "
        "explicitly, and flag it when two sources disagree rather than silently picking one.\n"
        "- Prefer showing the work over asserting the conclusion: include the key figures, "
        "the units, and the source.\n"
        "- If you could not verify something, say so plainly. Never fill a gap with a "
        "plausible-sounding number.\n"
        "- When you finish, offer one concrete next step you could take (a specific search, "
        "a specific analysis, or a specific file) so the user can push the work forward."
        + facts_text
    )


def history_to_messages(history: list) -> list:
    """Rebuild API messages from stored conversation rows.

    A "tool" message is only valid if the assistant turn that requested it is
    still present, so drop orphaned tool rows (which can happen when history
    is truncated mid tool-call sequence).
    """
    messages = []
    pending_tool_ids = set()
    for row in history:
        msg = {"role": row["role"], "content": row["content"] or ""}
        if row.get("tool_calls"):
            msg["tool_calls"] = row["tool_calls"]
            if row["role"] == "assistant":
                pending_tool_ids = {tc.get("id") for tc in row["tool_calls"] if tc.get("id")}
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
                result = await _run_tool(tc["name"], args, channel, user_id)

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


async def get_nvidia_response_full(user_id, user_message, channel=None):
    """Answer a message, persist the exchange, and return the answer plus metadata."""
    history = await db_get_history(user_id)
    user_facts = await db_get_user_facts(user_id)

    messages = [{"role": "system", "content": build_system_prompt(user_facts)}]

    keep_count = MAX_HISTORY * 6
    if len(history) > keep_count:
        messages.append({
            "role": "system",
            "content": f"[Earlier conversation truncated. Only the last {keep_count} messages are shown.]"
        })
        history_to_use = history[-keep_count:]
    else:
        history_to_use = history

    messages.extend(history_to_messages(history_to_use))
    messages.append({"role": "user", "content": user_message})
    await db_add_message(user_id, "user", user_message)

    result = await run_agent_loop(
        messages,
        user_id=user_id,
        channel=channel,
        max_steps=MAX_AGENT_STEPS,
    )
    result["footer"] = _footer_for(result)
    return result


async def get_nvidia_response(user_id, user_message, channel=None):
    """Answer a message and persist the exchange. Returns the answer text."""
    result = await get_nvidia_response_full(user_id, user_message, channel)
    return result["answer"]
