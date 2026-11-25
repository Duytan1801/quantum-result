"""The agent loop: prompt building, context assembly, and tool-calling."""

import asyncio
import difflib
import json
import logging
import os
import re
import time

import config
from cache import CACHES
from config import (
    HISTORY_ROWS,
    MAX_AGENT_STEPS,
    NVIDIA_MODEL,
    extra_body_for,
    provider_chain,
    vision_models,
)
from agents import TOOLS, agent_catalogue
from context import (
    compact_history,
    history_to_messages,
    sanitize_history,
    trim_context,
)
from sandbox import run_code_in_sandbox
from prompts import build_system_prompt
from storage import (
    db_add_message,
    db_get_history_multi,
    db_get_user_facts_multi,
)
from tools import (
    BASE_DIR,
    draw_diagram,
    edit_file,
    hf_dataset_info,
    hf_download,
    hf_model_info,
    hf_read_file,
    hf_search_datasets,
    hf_search_models,
    paper_details,
    read_file,
    read_pdf,
    search_papers,
    verify_claims,
    web_search,
    wolfram_alpha,
    write_file,
    written_files,
)
from vision import build_image_content

# Configure logging
logger = logging.getLogger(__name__)

# Status label shown in Discord for each tool.
STATUS_LABELS = {
    "code": "💻 Running code...",
    "wolfram_alpha": "🧮 Computing...",
    "web_search": "🔍 Searching the web...",
    "search_papers": "📚 Searching academic papers...",
    "paper_details": "📚 Looking up the paper...",
    "read_pdf": "📄 Reading the PDF...",
    "hf_search_models": "🤗 Searching Hugging Face models...",
    "hf_search_datasets": "🤗 Searching Hugging Face datasets...",
    "hf_model_info": "🤗 Reading the model card...",
    "hf_dataset_info": "🤗 Reading the dataset card...",
    "hf_read_file": "🤗 Reading the repository file...",
    "hf_download": "⬇️ Downloading from Hugging Face...",
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


# The model sometimes answers a question that needs a source straight out of
# its training data. The prompt already says it must search, and most of the
# time it does - but not every time, and a reply from memory is exactly the
# reply that is wrong six months later.
#
# So the answer is checked: if the turn was a real job and nothing was
# looked up, the model is asked once more, and this time it is told what
# the choice is. It can still decide that the question was conversation,
# in which case it just answers again.
TOOL_RETRY_NOTE = (
    "That answer came from memory. Look it up before you answer: call "
    "web_search, then answer again with what you find. Reply to the person, "
    "in their language, in the tone of the conversation you are already in - "
    "not to this note - and do not mention the search, the tools, or that you "
    "checked anything."
)


def _has_image(messages) -> bool:
    """Does this turn carry a picture?

    The picture is in the user message as a content list with an
    `image_url` part. MiniMax M3.1 Flash cannot look at one, so a turn
    that has one goes to Nemotron instead.
    """
    for message in messages or []:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


# ---------------------------------------------------------------------------
# The answer cache
# ---------------------------------------------------------------------------
#
# In a room, the same question comes up again and again. "What is filtering
# theory?" gets asked once on Monday and again on Wednesday, by a different
# person, expecting the same answer. Each of those is a full agent loop: a
# search, a couple of model calls, an embedding, and ten seconds of waiting.
#
# So the answer is kept for ten minutes and handed back when the same
# person asks the same thing in the same room.
#
# Four things are deliberately NOT cached, because getting them wrong would
# be worse than being slow:
#
# - A personal reply. If somebody says they are upset and says it again
#   two minutes later, they want a fresh answer, not the same words. Only
#   a reply that came out of a tool is kept, because that one was a
#   factual answer rather than a conversation.
# - A question about something that changes. "What is the share price
#   today" is never served from a cache.
# - A reply that produced a file. The file was attached to the first
#   reply; a cached reply would arrive with the text and nothing else.
# - A question that started a background agent. Its result arrives later,
#   on its own, and must not be pretended into a reply that is not there.
#
# The key carries the speaker and the room, so an answer given to one
# person is never handed to another.

def _answer_cache_key(user_message: str, speaker, write_scope: str,
                      user_id) -> str:
    """The key for the answer cache, or "" if this turn cannot be cached."""
    if not (user_message or "").strip():
        return ""
    who = str((speaker or {}).get("id") or user_id or "")
    # The message, tidied so that "what is x ?" and "what is x" are the
    # same question.
    text = " ".join((user_message or "").split()).lower()
    return f"{who}|{write_scope}|{text}"


def _cacheable(result: dict, user_message: str) -> bool:
    """Should this turn be kept for ten minutes?"""
    answer = (result.get("answer") or "").strip()
    if not answer or answer.startswith("LLM "):
        return False
    if result.get("files"):
        # The file was attached to the first reply. A cached reply would
        # come with the words and nothing else.
        return False
    tools = set(result.get("tools_used") or [])
    if "start_agent_task" in tools:
        # The real result is posted later, by the agent.
        return False
    if not tools:
        # No tool means this was a conversation, not an answer. Repeating a
        # conversation verbatim is worse than repeating nothing.
        return False
    if needs_fresh_data(user_message):
        # The answer depends on the world as it is now.
        return False
    return True


def _footer_for(result: dict) -> str:
    """Build the small grey footer line: the model that answered.

    The model is the one that actually answered, not the one that was
    configured first - so the footer tells the truth about which model
    produced the reply.

    Only the model. The tools that were used are not in the footer: they
    are already in the room as the status line, and "web_search" in the
    middle of a conversation is the bot talking about its own insides.
    """
    return result.get("model") or NVIDIA_MODEL


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
        code = args.get("code", "")
        language = args.get("language", "python")
        # CODE_BACKEND decides whether the plain code tool goes to the fast
        # sandbox or to Kaggle's cloud compute.
        backend = getattr(config, "CODE_BACKEND", "both")
        if backend == "kaggle" and language == "python":
            import kaggle_backend

            # The same per-user state directory as kaggle_run, so a script
            # written with the code tool leaves its files where the next run
            # can pick them up and where read_file can find them.
            state_dir = os.path.join(BASE_DIR, "kaggle", str(user_id or "default"))

            result = await kaggle_backend.run_code(
                code, timeout=600, state_dir=state_dir)
            return kaggle_backend.format_output(
                result,
                state_rel=os.path.relpath(state_dir, BASE_DIR),
            )
        return await run_code_in_sandbox(code, language)
    if name == "kaggle_run":
        import kaggle_backend

        # The run's files are kept per user and pushed into the next run, so
        # a pipeline can be built over several runs instead of being redone.
        state_dir = os.path.join(BASE_DIR, "kaggle", str(user_id or "default"))

        result = await kaggle_backend.run_code(
            args.get("code", ""),
            timeout=int(args.get("timeout") or 600),
            accelerator=args.get("accelerator") or "",
            datasets=args.get("datasets") or [],
            state_dir=state_dir,
        )
        text = kaggle_backend.format_output(
            result,
            state_rel=os.path.relpath(state_dir, BASE_DIR),
        )
        # A script that wrote files gets them attached to the reply.
        for f in result.get("files") or []:
            path = f.get("path")
            if path and os.path.exists(path) and not path.endswith(".log"):
                try:
                    tracker = written_files.get()
                    if tracker is not None:
                        tracker.append(path)
                except Exception:
                    pass
        return text
    if name == "kaggle_datasets":
        import kaggle_backend

        items = await kaggle_backend.list_datasets(args.get("query") or args.get("search") or "",
                                                    int(args.get("limit") or 15))
        if not items:
            return "No datasets found."
        lines = [f"- {d['ref']} ({d['size']} bytes) - {d['title']}" for d in items]
        return "Kaggle datasets:\n" + "\n".join(lines)
    if name == "kaggle_models":
        import kaggle_backend

        items = await kaggle_backend.list_models(args.get("query") or args.get("search") or "",
                                                  int(args.get("limit") or 15))
        if not items:
            return "No models found."
        lines = [f"- {m['ref']} - {m['title']}" for m in items]
        return "Kaggle models:\n" + "\n".join(lines)
    if name == "web_search":
        return await web_search(
            args.get("query", ""),
            args.get("count", 5),
            args.get("freshness", "noLimit"),
        )
    if name == "read_file":
        return await read_file(args.get("file_path", ""))
    if name == "search_papers":
        return await search_papers(
            args.get("query", ""),
            int(args.get("limit") or 5),
            int(args.get("year") or 0),
        )
    if name == "paper_details":
        return await paper_details(args.get("title", ""))
    if name == "read_pdf":
        return await read_pdf(
            args.get("source", ""),
            args.get("pages", "") or "",
            int(args.get("max_chars") or 60000),
        )
    if name == "hf_search_models":
        return await hf_search_models(
            args.get("query", ""),
            int(args.get("limit") or 10),
            args.get("sort") or "downloads",
        )
    if name == "hf_search_datasets":
        return await hf_search_datasets(
            args.get("query", ""),
            int(args.get("limit") or 10),
            args.get("sort") or "downloads",
        )
    if name == "hf_model_info":
        return await hf_model_info(args.get("model_id", ""))
    if name == "hf_dataset_info":
        return await hf_dataset_info(args.get("dataset_id", ""))
    if name == "hf_read_file":
        return await hf_read_file(
            args.get("repo_id", ""),
            args.get("path", "") or "README.md",
            args.get("repo_type", "") or "model",
        )
    if name == "hf_download":
        return await hf_download(
            args.get("repo_id", ""),
            args.get("repo_type", "") or "model",
            args.get("filename", "") or "",
            int(args.get("max_bytes") or 2147483648),
        )
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


def _normalize_tool_args(args: dict) -> str:
    """Normalize tool arguments for deduplication comparison.
    
    Creates a stable representation of arguments that can be compared
    for similarity, ignoring formatting differences.
    """
    if not args:
        return ""
    
    # Sort keys and create a stable string representation
    try:
        # Handle common text normalization
        parts = []
        for key in sorted(args.keys()):
            value = args[key]
            if isinstance(value, str):
                # Normalize whitespace and strip
                value = " ".join(value.split()).strip()[:200]
            elif isinstance(value, (list, dict)):
                # Truncate complex structures
                value_str = str(value)[:200]
            else:
                value_str = str(value)
            parts.append(f"{key}={value_str}")
        return "|".join(parts)
    except Exception:
        return str(args)[:200]


def _near_duplicates(call_counts: dict, name: str) -> bool:
    """Has this tool been called with several different sets of arguments?

    The log showed the model running the same search over and over with
    the wording changed slightly each time. The results are the same, so
    the second and third are worth calling out.
    """
    return sum(1 for k, v in call_counts.items()
               if k[0] == name and v >= 1) >= 2


def _is_failure(result: str) -> bool:
    """Did this tool call fail? Used to break out of retry loops."""
    return (
        result.startswith("Error:")
        or "API error" in result[:120]
        or result.startswith("Wolfram Alpha could not")
    )


# Matches one sentence: a run of text ending in a full stop, a question
# mark or an exclamation mark, or any run without one.
_SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?]+|[^.!?\n]+")


def _collapse_blocks(text: str, spam_at: int = 3) -> str:
    """Collapse a whole block (paragraph) that has been repeated."""
    blocks = text.split("\n\n")
    keys = [" ".join(b.split()) for b in blocks]
    counts = {}
    for key in keys:
        if key:
            counts[key] = counts.get(key, 0) + 1

    kept = []
    shown = {}
    for block, key in zip(blocks, keys):
        if not key or counts.get(key, 0) < spam_at:
            kept.append(block)
            continue
        seen = shown.get(key, 0)
        shown[key] = seen + 1
        if seen == 0:
            kept.append(block)
        elif seen == 1:
            kept.append(f"(the same paragraph was repeated {counts[key]} times; "
                        f"it is shown once)")
    return "\n\n".join(kept)


def _collapse_lines(text: str, spam_at: int = 3) -> str:
    """Collapse a line that has been repeated word for word."""
    lines = text.split("\n")
    keys = [" ".join(l.split()) for l in lines]
    counts = {}
    for key in keys:
        if key:
            counts[key] = counts.get(key, 0) + 1

    kept = []
    shown = {}
    for line, key in zip(lines, keys):
        # Short lines are structure ("---", "-", "|"), not a repeated
        # thought, so they are never touched.
        if len(key) < 40 or counts.get(key, 0) < spam_at:
            kept.append(line)
            continue
        seen = shown.get(key, 0)
        shown[key] = seen + 1
        if seen == 0:
            kept.append(line)
        elif seen == 1:
            kept.append(f"(the same line was repeated {counts[key]} times; "
                        f"it is shown once)")
    return "\n".join(kept)


def _split_sentences(line: str) -> list[tuple[str, str]]:
    """Split a line into sentences, keeping the space before each one."""
    out = []
    for m in _SENTENCE_RE.finditer(line):
        raw = m.group()
        stripped = raw.lstrip()
        out.append((stripped, raw[:len(raw) - len(stripped)]))
    return out


def _collapse_sentences(text: str, spam_at: int = 3) -> str:
    """Collapse a sentence that has been repeated across the answer.

    The model loops by writing the same sentence again and again, in the
    same line or in the next one. Matching the sentences rather than the
    lines catches both, and it is safe: two lines that merely look alike
    (three cluster bullets, say) carry different sentences and are left
    alone.
    """
    lines = text.split("\n")
    parsed = [_split_sentences(line) for line in lines]

    # How often does each sentence appear in the whole answer?
    counts = {}
    for line in parsed:
        for sent, _ in line:
            key = " ".join(sent.split())
            if len(key) >= 40:
                counts[key] = counts.get(key, 0) + 1
    repeated = {key for key, n in counts.items() if n >= spam_at}

    if not repeated:
        return text

    out_lines = []
    seen = set()
    for line in parsed:
        kept = []
        dropped = 0
        for sent, gap in line:
            key = " ".join(sent.split())
            if key not in repeated:
                kept.append(gap + sent)
                continue
            if key in seen:
                # A copy of a sentence already shown - drop it.
                dropped += 1
                continue
            seen.add(key)
            kept.append(gap + sent)
        if dropped:
            kept.append(f" ({dropped} repeated sentence"
                        f"{'s were' if dropped > 1 else ' was'} removed)")
        out_lines.append("".join(kept))
    return "\n".join(out_lines)


def _collapse_repeats(text: str, spam_at: int = 3) -> str:
    """Collapse a passage that has been repeated over and over.

    A model that loops can write the same sentence many times in a row, and
    the user ends up with several screens of the same paragraph. A passage
    that appears a few times is left alone - that can be deliberate - but
    one that is repeated over and over is shown once, with a note.

    The repeat is looked for at three levels, because the junk that reached
    Discord came in all three shapes: a whole paragraph repeated with a
    blank line between the copies, the same paragraph repeated with only a
    single newline between the copies, and the same sentence repeated with
    no separator at all.
    """
    if not text:
        return text
    text = _collapse_blocks(text, spam_at)
    text = _collapse_lines(text, spam_at)
    text = _collapse_sentences(text, spam_at)
    return text


# --- Answer-level loop breaker ---------------------------------------------
#
# Cleaning the stored history helps, but the model can still fall into the
# same groove from the conversation happening right now: it answers "Hi" with
# the same sentence it gave fourteen times before. So the finished answer is
# compared with the recent answers, and if it is a copy the model is asked
# for a different one - once.

_REPEAT_RATIO = 0.75
_REPEAT_MIN_CHARS = 40


def _norm_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def repeats_history(answer: str, prior: list) -> bool:
    """Does this answer copy one of the recent answers?"""
    a = _norm_answer(answer)
    if len(a) < _REPEAT_MIN_CHARS:
        return False
    for other in reversed(list(prior)[-6:]):
        b = _norm_answer(other)
        if not b:
            continue
        if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= _REPEAT_RATIO:
            return True
    return False


async def _fresh_answer(messages: list, max_tokens: int) -> str:
    """Ask for the answer again, with an instruction not to repeat itself."""
    request = list(messages) + [{
        "role": "user",
        "content": (
            "Your previous reply was almost word for word the same as an "
            "earlier reply in this conversation. Do not repeat it. Answer the "
            "message you were given directly and specifically. If the answer "
            "is the same information, say it differently and add whatever is "
            "new. Do not announce what you are about to do - just do it or "
            "answer."
        ),
    }]
    request = trim_context(request)
    for _name, client, model in provider_chain():
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=request,
                temperature=0.7,
                max_completion_tokens=max_tokens,
                extra_body=extra_body_for(_name),
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as exc:
            print(f"could not get a fresh answer: {exc}")
            continue
    return ""


async def _voice_answer(messages: list, max_tokens: int, problem: str,
                        expected_term: str = "") -> str:
    """Ask for the same answer again, in the register this room uses.

    The voice is not fixed by rewriting the words - the bot might be
    writing a letter, or quoting somebody, and a mechanical substitution
    would break it. So the model is asked again, with the problem named,
    and it keeps the content and changes the voice.
    """
    term_note = ""
    if expected_term:
        term_note = (
            f" The person you are talking to is '{expected_term}' - that is "
            f"the word for them, and you are 'mình'."
        )
    request = list(messages) + [{
        "role": "user",
        "content": (
            f"Your reply used {problem}, which is the wrong register for "
            "this room. The people here talk as anh/em/a/e and say 'mình' "
            "for themselves; 'Bạn' and 'Tôi' make it sound like customer "
            f"service.{term_note} Say the same thing again, in their voice, "
            "and do not change what you are saying - only how you say it."
        ),
    }]
    request = trim_context(request)
    for _name, client, model in provider_chain():
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=request,
                temperature=0.5,
                max_completion_tokens=max_tokens,
                extra_body=extra_body_for(_name),
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as exc:
            print(f"could not get an answer in the right register: {exc}")
            continue
    return ""


async def _summarise_at_limit(messages: list, max_tokens: int) -> str:
    """One last tool-free call, asking the model to report what it found.

    Used when the agent hits its step limit: the tool calls it already made
    hold real work, and throwing that away because the budget ran out is
    exactly the wrong outcome.
    """
    request = list(messages) + [{
        "role": "user",
        "content": (
            "You have run out of steps. Do not call any more tools. Write the "
            "final answer now, using only what you have already gathered: "
            "state what you found, what you did not get to, and point at any "
            "files you wrote. Be concrete and cite the sources you actually "
            "used."
        ),
    }]
    last_error = None
    for _name, client, model in provider_chain():
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=request,
                temperature=0.4,
                max_completion_tokens=max_tokens,
                extra_body=extra_body_for(_name),
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            last_error = "empty summary"
        except Exception as exc:
            last_error = exc
            continue
    print(f"could not summarise at the step limit: {last_error}")
    return ""


async def run_agent_loop(messages, user_id=None, channel=None, max_steps=MAX_AGENT_STEPS,
                         tools=None, persist=True, on_step=None, max_tokens=8192,
                         thinking: bool = True, require_tool: bool = False,
                         speaker_id: str = ""):
    """Run the agent tool-calling loop until the model produces a final answer.

    `max_tokens` is the allowance for each step. The reasoning and the
    answer come out of the same allowance - the model thinks first and
    then answers, and both are counted - so the number has to leave room
    for both, or a hard question ends in a step that thinks for eight
    thousand tokens and says nothing.

    `thinking` switches the reasoning off for the turn. It is off for a
    passing remark: the answer is a line or two, and the reasoning is a
    large part of why a reply can take a minute to arrive.

    `require_tool` says the turn is a real job that should have looked
    something up. If the model answers without calling a tool, it is
    asked once more - not forced, but told what the choice is.

    `speaker_id` is who the reply is for. It is what lets the register
    check know which kinship term belongs in the answer.

    Returns a dict with the answer, which tools were used, and how many
    steps it took.
    """
    if tools is None:
        tools = TOOLS

    # A turn that carries a picture needs a model that can see one. Laguna
    # and MiniMax cannot, so they are skipped for those turns and Nemotron
    # goes first; everything else starts with Laguna, which is the fastest
    # of the three.
    chain = (vision_models() if _has_image(messages)
             else provider_chain())

    # The request as it was before this turn started, and the answers the
    # model has already given in this conversation. The finished answer is
    # checked against them, so a reply that is a copy can be caught.
    base_messages = list(messages)
    prior_answers = [
        m.get("content") for m in messages
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]

    # The message this turn is about. The retry below puts it back in front
    # of the model: after the nudge the last thing in the conversation is
    # the nudge itself, and the model has been seen to answer an older
    # question instead of the one it was given.
    original_request = ""
    for _message in reversed(base_messages):
        if _message.get("role") == "user":
            _content = _message.get("content")
            if isinstance(_content, str):
                original_request = _content
            break

    final_answer = ""
    tools_used = []
    step_count = 0
    failed_count = 0
    continued = False
    # Which model actually produced the answer. The footer shows this, so
    # it has to be the model that really answered rather than the one that
    # was configured first.
    used_model = ""
    # How many times each exact tool call has been made, and how many times
    # each tool has been called at all.
    call_counts = {}
    tool_counts = {}
    warned_tool_count = set()
    # How many times the model has been asked to go and look something up
    # after answering from memory. It is asked once: a second ask costs
    # another full round of thinking for very little gain.
    nudged = 0
    # Whether the next step has to call a tool. The model is told to, and
    # the step is not allowed to end without one.
    force_tool = False

    while True:
        if step_count >= max_steps:
            if channel:
                await channel.send(f"⚠️ Reached maximum steps ({max_steps}). Stopping.")
            if not final_answer.strip():
                # The agent ran out of budget before producing an answer.
                # Rather than posting "(no response)", make one last
                # tool-free call so the work it did is not thrown away.
                final_answer = await _summarise_at_limit(messages, max_tokens)
            break
        step_count += 1

        # Keep the request inside the model's context window.
        messages = trim_context(messages)

        # Try the models in order: Laguna, MiniMax, Nemotron, then Groq.
        # A turn with a picture in it skips the two that cannot look at
        # images, and goes to Nemotron first.
        #
        # A step that has to call a tool starts one model later: Laguna
        # answers straight out of its memory even when it is told to call
        # a tool, so a forced step that starts there is not forced at all.
        step_chain = chain
        force_now = force_tool
        if force_now:
            forced = [c for c in chain if c[0] != "Laguna"]
            step_chain = forced or chain
        force_tool = False

        stream = None
        last_error = None
        for _name, client, model in step_chain:
            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=max_tokens,
                    top_p=0.95,
                    stream=True,
                    tools=tools,
                    tool_choice="required" if force_now else "auto",
                    extra_body=extra_body_for(_name, thinking=thinking),
                )
                used_model = model
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
                "model": used_model,
            }

        step_content = ""
        tool_calls_accum = {}
        finish_reason = None
        # How much thinking this step produced. The reasoning and the answer
        # come out of the same token allowance, so a step can spend all of
        # it thinking and answer nothing - and then the bot has nothing to
        # send. This is how much thinking was spent, so that case can be
        # spotted.
        step_reasoning = 0

        try:
            async for chunk in stream:
                # The provider sends a final usage-only chunk with an empty
                # "choices" list. Skip it instead of crashing on choices[0].
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    step_content += delta.content
                # The model's reasoning arrives in its own field. It is not
                # part of the answer, so it is read and dropped rather than
                # shown to the user.
                if getattr(delta, "reasoning_content", None):
                    step_reasoning += 1
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
                "model": used_model,
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
                continued = True
            elif continued:
                # The end of an answer that had to be continued: keep the
                # earlier parts too. Replacing them here used to throw away
                # the whole first half of the reply.
                final_answer += step_content
            else:
                final_answer = step_content

        if not tool_calls_accum:
            if finish_reason == "length":
                if persist and user_id is not None:
                    await db_add_message(user_id, "assistant", step_content)
                messages.append({"role": "assistant", "content": step_content})
                if not step_content.strip() and step_reasoning:
                    # The whole allowance went on thinking and nothing was
                    # said. The reasoning and the answer come out of the
                    # same budget, so asking it to think harder is the one
                    # thing that will not help: ask it to answer now.
                    messages.append({
                        "role": "user",
                        "content": (
                            "You used the whole token budget thinking and "
                            "produced no answer. Stop deliberating and give "
                            "the answer now, in a few sentences. If you "
                            "cannot say more, say what you know."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": "Continue your response from where you left off.",
                    })
                continue

            if require_tool and not tools_used and nudged < 1:
                # The model answered a real job out of its memory. Before
                # making it do the work again, ask whether this message
                # asked for something that has to come from a source: a
                # question that needs a lookup, or a person talking about
                # their day. The model tells them apart, and a personal
                # message is left alone.
                nudged += 1
                wanted = False
                try:
                    from presence import needs_lookup

                    wanted = await needs_lookup(original_request)
                except Exception as exc:
                    print(f"the lookup check failed: {exc}")

                if not wanted:
                    # It was conversation, or something that needs no
                    # source. The answer stands.
                    if persist and user_id is not None:
                        await db_add_message(user_id, "assistant", step_content)
                    break

                # It asked for something with an answer, so the answer has
                # to come from a source. The next step is not allowed to
                # end without a tool call.
                force_tool = True
                retry = TOOL_RETRY_NOTE
                if original_request:
                    retry += (
                        "\n\nThe message to answer is this one, and no "
                        f"other:\n\"{original_request[:1500]}\""
                    )
                messages.append({"role": "assistant", "content": step_content})
                messages.append({"role": "user", "content": retry})
                continue

            if persist and user_id is not None:
                await db_add_message(user_id, "assistant", step_content)
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

        if channel and on_step is None:
            # When a caller supplies on_step it does its own reporting (a
            # background job posts a throttled heartbeat). Sending both
            # meant two messages per step saying the same thing.
            #
            # No "Step 1 -" in front of it. The step number means nothing to
            # the person reading the room, and the bold formatting is the
            # loudest thing about it - it is the single most obviously
            # machine-written line the bot posts. A person waiting for an
            # answer says what they are doing, not which step they are on.
            await channel.send(" ".join(step_labels))
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
                # The model can get stuck calling the same thing over and over
                # - the log showed eighteen identical Kaggle runs in a row,
                # nearly two hours, because each one started from nothing.
                # A call that has already succeeded does not need doing again.
                # Enhanced deduplication: track by normalized args for smarter detection
                normalized_args = _normalize_tool_args(args)
                key = (tc["name"], normalized_args[:200])
                tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1

                # Warn after 6 calls to same tool type
                if tool_counts[tc["name"]] >= 6 and tc["name"] not in warned_tool_count:
                    warned_tool_count.add(tc["name"])
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called `{tc['name']}` {tool_counts[tc['name']]} "
                            f"times in this turn. Each call is expensive. Work with "
                            f"the results you have rather than running it again."
                        ),
                    })

                # The exact same call: it has already been made twice and
                # worked twice, so a third one would cost a minute and
                # change nothing.
                if call_counts.get(key, 0) >= 2:
                    result = (
                        f"This exact `{tc['name']}` call has already been made "
                        f"twice in this turn and produced a result both times. "
                        f"It was not run again. Work with the result you already "
                        f"have, change the arguments, or answer the user."
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called `{tc['name']}` with exactly the same "
                            f"arguments three times. Repeating it will not change "
                            f"the result, so the last one was not run. Do something "
                            f"different: change the script, work with the files "
                            f"the earlier runs produced, or answer the user with "
                            f"what you already have."
                        ),
                    })
                # The same tool, different arguments, several times over. The
                # results are nearly always the same.
                elif tool_counts[tc["name"]] >= 3 and _near_duplicates(
                        call_counts, tc["name"]):
                    result = (
                        f"You are calling `{tc['name']}` multiple times with "
                        f"similar arguments in this turn. Use the tool results "
                        f"you already have rather than making more calls."
                    )
                else:
                    # A tool that raises must not kill the whole turn - the
                    # model needs to see the error so it can recover or answer
                    # directly.
                    try:
                        result = await _run_tool(tc["name"], args, channel, user_id)
                    except Exception as exc:
                        result = f"Error: {tc['name']} tool failed: {exc}"
                        if channel:
                            await channel.send(
                                f"⚠️ The `{tc['name']}` tool failed: {str(exc)[:150]}"
                            )
                    # Only a call that worked counts towards the repeat limit:
                    # retrying a call that failed is exactly what should
                    # happen.
                    if not _is_failure(result):
                        call_counts[key] = call_counts.get(key, 0) + 1

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

    answer = _collapse_repeats(final_answer or "(no response)")

    # The loop breaker. A reply that copies an earlier one is almost always
    # a model stuck in a groove rather than an answer, so it is asked for a
    # different one. If the second attempt is a copy too, the original stands.
    if final_answer and repeats_history(final_answer, prior_answers):
        logger.warning(f"answer repeat detected, requesting fresh response")
        fresh = await _fresh_answer(base_messages, max_tokens)
        if fresh and not repeats_history(fresh, prior_answers):
            answer = _collapse_repeats(fresh)
            logger.info("received non-repeating answer, using it")

    # The voice check. A reply in the wrong register is asked for again,
    # because the bot's own old replies are in its context and a model
    # copies what it reads: one reply in the wrong voice pulls the next
    # one into the same voice, and the room gets a formal bot that cannot
    # stop being formal.
    expected_term = ""
    try:
        from config import kinship_for

        found = kinship_for(speaker_id)
        expected_term = found[0] if found else ""
    except Exception:
        expected_term = ""

    try:
        from guard import register_problem

        problem = register_problem(answer, expected_term)
    except Exception as exc:
        logger.error(f"the register check failed: {exc}")
        problem = ""

    if problem:
        logger.warning(f"wrong register ({problem}), asking again")
        again = await _voice_answer(base_messages, max_tokens, problem,
                                    expected_term)
        if again and not register_problem(again, expected_term):
            answer = _collapse_repeats(again)
            logger.info("got an answer in the right register")
        elif again:
            # Still wrong. The first answer stands: a wrong-voice answer
            # is better than no answer, and the guard still has its say.
            logger.info("the answer came back in the wrong register again")

    # The guard. The prompt asks for a particular voice; this is what makes
    # sure the voice that actually arrives is that one. It strips the
    # bot-speak, the prompt echoes and the "let me know if you need anything
    # else" - and it leaves the substance of the answer alone.
    try:
        from guard import sanitize_reply

        answer = sanitize_reply(answer)
    except Exception as exc:
        logger.error(f"the guard could not check the reply: {exc}")

    # Log completion metrics
    logger.debug(f"turn completed: {step_count} steps, {len(tools_used)} tools, "
                 f"model={used_model}, answer_len={len(answer)}")

    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": step_count,
        "model": used_model,
    }


async def get_nvidia_response_full(user_id, user_message, channel=None,
                                   attachments=None, scopes=None,
                                   write_scope=None, shared: bool = False,
                                   speaker=None, roster=None, shape: str = "",
                                   bot_name: str = "", bot_aliases=(),
                                   write_scopes=None):
    """Answer a message, persist the exchange, and return the answer plus metadata.

    `scopes` are the context keys to read from. A channel with more than one
    person passes the person's own key and the channel's shared key, so the
    bot knows what everybody in the room has been talking about.

    `write_scope` is where the new rows are stored - the shared channel key
    when the room has more than one person, so the thread stays together.

    `write_scopes` is where a *memory* learned in this turn is stored, which
    is not always the same thing as where the conversation goes. It is a
    list because a shared room stores what it learns under both the room's
    key and the person's, so the fact is still known when somebody else
    asks. A detached room - one that knows the whole picture but shares
    nothing - writes only to its own key, so what is learned there stays
    there. It defaults to `scopes`, which is what an ordinary room wants.

    `speaker` and `roster` say who is being talked to and who else is in
    the room. They are read from the channel, not from a list in the code,
    so the bot always knows the names it needs.

    `shape` is what the presence classifier made of the message: "aside",
    "question" or "deep". It decides how long the reply should be, and
    whether the model needs to think before it answers.

    `bot_name` and `bot_aliases` are the bot's own names, so it knows what
    it is called.
    """
    _check_delegation_tool()
    scopes = [str(s) for s in (scopes or [user_id]) if s]
    write_scope = str(write_scope or user_id)
    write_scopes = [str(s) for s in (write_scopes or scopes) if s]

    # The answer cache. A question that has already been answered in this
    # room, by this person, does not need answering again.
    cache_key = _answer_cache_key(user_message, speaker, write_scope, user_id)
    if cache_key:
        cached = CACHES.get("answer", cache_key)
        if cached is not None:
            print(f"answer cache: served '{user_message[:60]}' from the cache")
            # The cached answer keeps its text and its footer, but it is not
            # a fresh run: it took no steps and no tools.
            served = dict(cached)
            served["steps"] = 0
            served["tools_used"] = []
            served["cached"] = True
            return served

    history = await db_get_history_multi(scopes, limit=HISTORY_ROWS)
    # The stored conversation can contain a wall of near-identical answers
    # from an earlier bug. Feeding that back is what taught the model to
    # repeat itself, so it is cleaned before it reaches the prompt.
    history = sanitize_history(history)
    user_facts = await db_get_user_facts_multi(scopes)

    # The live Kaggle status goes into the prompt. The model has to know that
    # the tool works right now, not what a remembered note says about it, so
    # the check is made against the real API before the prompt is built.
    kaggle_hint = ""
    try:
        import kaggle_backend

        kaggle_hint = await asyncio.to_thread(kaggle_backend.status_hint)
    except Exception as exc:
        print(f"kaggle status check failed: {exc}")

    messages = [
        {"role": "system",
         "content": build_system_prompt(user_facts, kaggle_hint, shared=shared,
                                        speaker=speaker, roster=roster,
                                        bot_name=bot_name,
                                        bot_aliases=bot_aliases)}
    ]
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
    await db_add_message(write_scope, "user", stored)

    # The register decision, as the last thing before the question. Without
    # it, a conversation that has been personal for a while pulls the next
    # message into the same register - and a technical question gets
    # answered out of the model's memory instead of with a tool.
    try:
        from prompts import turn_note

        messages.insert(len(messages) - 1, {"role": "system",
                                            "content": turn_note()})
    except Exception as exc:
        print(f"could not add the turn note: {exc}")

    # How big a reply this turn wants. A person answers in proportion: a
    # remark made in passing gets a line back, and the same note applied
    # to a question makes the voice worse, so only a passing remark gets
    # one. The note sits beside the message, where a note is read.
    if shape == "aside":
        messages.insert(len(messages) - 1, {
            "role": "system",
            "content": (
                "[This is a passing remark, not a prompt for an essay. "
                "Answer it at its own size: a line or two, then stop. No "
                "closing reflection, no summary of what they said.]"
            ),
        })

    # How the room actually talks. Examples beat adjectives: a model told
    # "be casual" writes a generic kind of casual, while a model shown three
    # real lines writes in the register of those lines. Neuro-sama gets this
    # for free from a live stream of chat; here it comes from the room's
    # own history, so the bot stops saying "Bạn" to people who have been
    # talking like friends all evening.
    try:
        from prompts import voice_examples, voice_note

        anchor = voice_note(voice_examples(history))
        if anchor:
            messages.insert(len(messages) - 1, {"role": "system",
                                                "content": anchor})
    except Exception as exc:
        print(f"could not add the voice anchor: {exc}")

    # The reasoning is what makes a reply slow, and a passing remark does
    # not need it. Everything else gets the full treatment.
    thinking = shape != "aside"

    # If the question depends on the world as it is now, remind the model to
    # source and verify before answering. Cheap to add, and it stops the model
    # confidently answering from stale training data.
    if needs_fresh_data(user_message):
        messages.insert(1, {"role": "system", "content": FRESHNESS_REMINDER})

    # Pull in anything worth remembering from earlier conversations.
    try:
        from memory import format_memories, is_available, retrieve_memories

        if await is_available():
            # The relationship impressions come first, because they decide
            # the tone of the reply and everything else is easier to get
            # wrong once the wrong voice is set.
            try:
                from memory import build_impressions

                impressions = await build_impressions(
                    scopes, speaker=speaker, roster=roster)
                if impressions:
                    messages.insert(1, {"role": "system", "content": impressions})
            except Exception as exc:
                print(f"relationship impressions failed: {exc}")

            # The query carries the names of the people in the room, so that
            # what the bot knows about *them* comes back - not just what is
            # about the subject of the message. In a room, that is the
            # difference between knowing how the people relate and knowing
            # nothing about them.
            query = user_message
            if shared and roster:
                names = " ".join(
                    (entry.get("plain") or entry.get("name") or "")
                    for entry in roster
                ).strip()
                if names:
                    query = f"{user_message} {names}"

            memories = await retrieve_memories(
                user_id, query, top_k=6, scopes=scopes
            )
            memories_text = format_memories(memories, shared=shared)
            if memories_text:
                messages.insert(1, {"role": "system", "content": memories_text})
    except Exception as exc:
        print(f"memory retrieval failed: {exc}")

    # Keep the whole conversation, but compact it if it has grown too large.
    # Older turns are summarised rather than dropped, so the model keeps the
    # thread of what the user wants. The summary is cached per scope, so in a
    # room both people get the same summary.
    messages = await compact_history(messages, user_id=write_scope)

    # Track any files written during this turn so generated images (diagrams,
    # plots) can be attached to the reply automatically.
    tracker: list[str] = []
    token = written_files.set(tracker)

    # Whether this turn should have looked something up. The presence
    # classifier has already said whether the message is a real job ("deep"),
    # so this is a model's opinion and not a list of words; the freshness
    # check is the older one, kept for the turns that never reach a room.
    require_tool = shape == "deep" or needs_fresh_data(user_message)

    try:
        result = await run_agent_loop(
            messages,
            user_id=write_scope,
            channel=channel,
            max_steps=MAX_AGENT_STEPS,
            thinking=thinking,
            require_tool=require_tool,
            speaker_id=str((speaker or {}).get("id") or ""),
        )
    finally:
        written_files.reset(token)

    result["files"] = [p for p in tracker if os.path.exists(p)]
    result["footer"] = _footer_for(result)

    # Keep the answer, if this is the kind of answer worth keeping.
    if cache_key and _cacheable(result, user_message):
        CACHES.put("answer", cache_key, dict(result))

    # Learn from this exchange in the background, so it never delays the reply.
    _schedule_memory_update(user_id, user_message, result["answer"],
                             scopes=write_scopes,
                             person_name=(speaker or {}).get("plain")
                             or (speaker or {}).get("name"))

    return result

def _schedule_memory_update(user_id, user_message: str, answer: str,
                            scopes: list[str] = None, person_name: str = None):
    """Extract durable facts from the exchange and store them.

    Runs in the background so it never delays the reply, and skips
    exchanges that cannot contain a durable fact - otherwise every "hi"
    would cost an LLM call and an embedding call.

    `person_name` is who was talking. Memories are written with that name
    in them rather than with "the user", because in a room with more than
    one person "the user" is ambiguous - and a memory the model cannot
    attribute is a memory it will get wrong.
    """
    async def run():
        try:
            from memory import (
                extract_memories,
                is_available,
                store_memories,
                worth_extracting,
            )

            if not worth_extracting(user_message, answer):
                return
            if not await is_available():
                return
            memories = await extract_memories(
                f"USER: {user_message}\n\nASSISTANT: {answer}",
                person_name=person_name,
            )
            if memories:
                await store_memories(user_id, memories, source="conversation",
                                     scopes=scopes)
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
