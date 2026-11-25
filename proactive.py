"""Speaking first.

The bot's job until now has been to answer. This is the other half:
saying something when nobody asked, once in a while, the way a person in
a room does - and, far more often, not saying anything.

Three things decide whether this works, and none of them is the writing.

WHEN. A bot that talks when it is spoken to is a tool; a bot that talks
when nobody asked is a pest. The difference is not the words, it is the
timing, and timing is mostly about saying nothing. The gate below
rejects almost every tick before a single token is generated, and the
model is then asked for a reason to speak rather than for a line.

WHAT. "Say something" is the worst instruction there is: a model given
it produces a generic opener, which is worse than silence. The thing to
say has to come out of the room - out of what was actually said there,
and what the people there are actually doing.

HOW MUCH. One line, easy to ignore, because sometimes it should be
ignored.

The research behind those choices, and what each one changed here:

- Jarvis (arXiv 2610.07506) intervenes when a group misses a fact and
  does not correct itself, and it stays silent 97% of the time. The
  silence rate is the number that matters, so the gate is strict and the
  model has to justify speaking before anything is written.

- "How to start a conversation proactively?" (arXiv 2611.09906) found
  that LLMs asked to open a conversation produce generic, monotonous
  openers even when the context is there, and that the fix is explicit
  context structuring. So the moment-decision is given a schema - what
  was said, who said it, how long ago, what kind of thing is going on -
  rather than a transcript and a wish.

- ProactiveEval (arXiv 2508.20973) splits proactivity into target
  planning and dialogue guidance. So there are two steps here: what the
  bot would be talking about (a small model, one of a fixed set of
  kinds), and then the sentence itself (the real model, in the room's
  voice).

- Inner Thoughts (arXiv 2501.00383) argues that a proactive agent
  forms its own thought and then decides whether it is worth expressing,
  modelling its own wish to express it. That is the shape of the small
  call here: does the bot have something it wants to say?

- "Comparing Perceptions of Static and Adaptive Proactive Speech
  Agents" (arXiv 2405.07528) found, against the authors' hypothesis,
  that an agent whose interruptions adapt to urgency and task difficulty
  was judged a WORSE conversational partner than a static one - less
  socially appropriate, less consistent. People liked the predictable
  one. So the proactive style here is fixed and plain rather than
  clever: one line, no emoji, no enthusiasm that is not warranted.

- "Towards Human-centered Proactive Conversational Agents" (arXiv
  2404.12670) makes Civility the point: without it, a proactive system
  reads as intrusive. So the line is short, asks for nothing, and can
  be ignored.

- The survey (arXiv 2305.02750) notes that proactivity raises the
  factuality risk, because the system contributes more of its own
  information. So a proactive line never states a fact: it does not
  claim to have read, done, seen or found anything, and it never invents
  a fact about the person.
"""

import asyncio
import re
import time

from config import (
    BOLT_MODEL,
    NVIDIA_MODEL,
    PROACTIVE_ENABLED,
    PROACTIVE_MAX_IDLE_MINUTES,
    PROACTIVE_MAX_LINE_CHARS,
    PROACTIVE_MAX_PER_DAY,
    PROACTIVE_MIN_GAP_MINUTES,
    PROACTIVE_MIN_IDLE_SECONDS,
    PROACTIVE_QUIET_HOURS,
    PROACTIVE_REPLY_WINDOW_MINUTES,
    extra_body_for,
    is_detached,
    nvidia_client,
)

# The kinds of thing worth saying, and only these. A fixed set is what
# makes the bot's unprompted voice predictable - people judged a static
# proactive agent a better conversational partner than an adaptive one,
# because the adaptive one was harder to read.
KINDS = ("follow_up", "check_in", "join")

# The little model that decides whether there is a moment. Bolt is small,
# quick and cheap, and the question is a simple judgement; the strong
# model writes the line afterwards, if there is one to write.
#
# Both of these are NVIDIA's, like the presence scanner. A decision about
# whether to speak in a room is a scan of that room, and a scan that
# quietly fell through to another provider would be another model's
# opinion rather than a fallback for a failed one.
MOMENT_MODELS = (BOLT_MODEL, NVIDIA_MODEL)


# ---------------------------------------------------------------------------
# When not to speak
# ---------------------------------------------------------------------------

def quiet_hours(spec: str = PROACTIVE_QUIET_HOURS) -> list[tuple[int, int]]:
    """The hours of the day when the bot does not start anything.

    "0-7" is from midnight to seven. "22-3" wraps around midnight.
    """
    out: list[tuple[int, int]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
        else:
            low = high = part
        try:
            start = int(low) % 24
            end = int(high) % 24
        except ValueError:
            continue
        if start != end:
            out.append((start, end))
    return out


def in_quiet_hours(when=None, spec: str = PROACTIVE_QUIET_HOURS) -> bool:
    """Is it the small hours?"""
    stamp = when if when is not None else time.localtime()
    hour = stamp.tm_hour
    for start, end in quiet_hours(spec):
        if start < end:
            if start <= hour < end:
                return True
        elif hour >= start or hour < end:
            return True
    return False


class ChannelState:
    """What the bot knows about its own behaviour in one channel."""

    __slots__ = ("last_human", "last_bot", "last_proactive",
                 "day", "today", "awaiting_reply", "ignored")

    def __init__(self):
        self.last_human = 0.0
        self.last_bot = 0.0
        self.last_proactive = 0.0
        self.day = ""
        self.today = 0
        self.awaiting_reply = 0.0
        # How many unprompted messages in this channel went unanswered.
        self.ignored = 0


STATE: dict[str, ChannelState] = {}


def state_for(channel_id: str) -> ChannelState:
    key = str(channel_id)
    state = STATE.get(key)
    if state is None:
        state = ChannelState()
        STATE[key] = state
    return state


def note_human(channel_id: str, when: float | None = None) -> None:
    """Somebody said something in this channel."""
    state_for(channel_id).last_human = when if when is not None else time.time()


def note_bot(channel_id: str, when: float | None = None) -> None:
    """The bot said something, whether or not it was asked to."""
    state_for(channel_id).last_bot = when if when is not None else time.time()


def note_proactive(channel_id: str, when: float | None = None) -> None:
    """The bot said something nobody asked for."""
    state = state_for(channel_id)
    now = when if when is not None else time.time()
    state.last_proactive = now
    state.last_bot = now
    state.awaiting_reply = now
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    if state.day != day:
        state.day = day
        state.today = 0
    state.today += 1


def note_reply(channel_id: str, when: float | None = None) -> None:
    """Somebody answered something the bot said without being asked."""
    state = state_for(channel_id)
    state.awaiting_reply = 0.0
    state.ignored = 0


def note_ignored(channel_id: str) -> None:
    """Nobody answered. The next unprompted message waits longer."""
    state = state_for(channel_id)
    state.awaiting_reply = 0.0
    state.ignored += 1


def awaiting_reply(channel_id: str, now: float | None = None) -> float | None:
    """How long ago the bot said something it is still waiting for."""
    state = state_for(channel_id)
    if not state.awaiting_reply:
        return None
    now = now if now is not None else time.time()
    if now - state.awaiting_reply > PROACTIVE_REPLY_WINDOW_MINUTES * 60:
        # Nobody answered within the window. The bot has stopped waiting,
        # and the room gets a longer silence before the next attempt.
        state.awaiting_reply = 0.0
        state.ignored += 1
        return None
    return now - state.awaiting_reply


def gate(channel_id: str, *, now: float | None = None, chat_on: bool = True,
         shared: bool = True) -> tuple[bool, str]:
    """Should this channel even be considered right now?

    Returns (yes, reason). The reason is printed either way, because a
    bot that says nothing should at least be able to say why.
    """
    now = now if now is not None else time.time()
    state = state_for(channel_id)

    if not PROACTIVE_ENABLED:
        return False, "proactive talking is switched off"
    if not chat_on:
        return False, "chat mode is off in this channel"
    if is_detached(channel_id):
        return False, "a detached room is not for talking"
    if in_quiet_hours(time.localtime(now)):
        return False, "it is the small hours"

    day = time.strftime("%Y-%m-%d", time.localtime(now))
    if state.day != day:
        state.day = day
        state.today = 0
    if PROACTIVE_MAX_PER_DAY and state.today >= PROACTIVE_MAX_PER_DAY:
        return False, f"already said something {state.today} times today"

    # The gap grows when nobody answered last time. Each ignored message
    # doubles the wait, up to sixteen times the base gap.
    gap = PROACTIVE_MIN_GAP_MINUTES * (2 ** min(state.ignored, 4))
    if state.last_proactive:
        since = (now - state.last_proactive) / 60
        if since < gap:
            return False, (f"only {int(since)} minutes since the last one "
                           f"(needs {gap})")

    if not state.last_human:
        return False, "nobody has spoken in this channel"

    idle = (now - state.last_human) / 60
    idle_minutes = max(0, int(idle // 60))
    if idle < PROACTIVE_MIN_IDLE_SECONDS / 60:
        return False, f"the room was active {idle_minutes} minutes ago"
    if idle > PROACTIVE_MAX_IDLE_MINUTES:
        return False, f"the room has been quiet for {int(idle // 60)} hours"

    if state.last_bot and (now - state.last_bot) < 60:
        return False, "the bot just spoke"

    return True, "ok"


# ---------------------------------------------------------------------------
# The moment
# ---------------------------------------------------------------------------

_VIETNAMESE = re.compile(
    r"[ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]"
)


def _language(lines) -> str:
    """The language the room is in, from what was said in it.

    Only the people count. The bot's own replies are in its voice and
    would otherwise decide the language by themselves.
    """
    people = [said for who, said in lines if who != "the bot"]
    text = " ".join(people[-6:]) if people else " ".join(said for _, said in lines)
    return "Vietnamese" if _VIETNAMESE.search(text) else "English"


def context_packet(*, channel_name: str, people: str, lines, now: float) -> str:
    """The room, as a schema rather than a transcript.

    The paper on proactive initiation found that a model asked to open a
    conversation produces generic openers unless the context is
    structured explicitly. So this is not a wall of chat: it is the few
    facts a decision actually needs.
    """
    return _context_text(channel_name=channel_name, people=people,
                         lines=lines, now=now)


MOMENT_PROMPT = """You are {bot_name}, in a chat room with {people}. You are not answering anybody. You are deciding whether to start something yourself.

The room as it is right now:
{context}

There are only three kinds of thing worth saying:
- follow_up: pick up something they said earlier that is worth coming back to.
- check_in: ask how one of them is doing, when there is something to ask about.
- join: say something about the topic they were on, or about today.

This is the way a streamer talks to a chat that has gone quiet. If the room has been quiet for a minute, there is usually something to say - a thought about what they were talking about, a question, a reaction, something from today's news that connects. You are not interrupting anybody: nobody is talking.

Decide whether there is something here worth one line, right now.

Say QUIET only when the moment is genuinely wrong - the room is busy, somebody is mid-sentence, or there is nothing to connect to. A room that has been quiet for a while is exactly the room you are here for.

If there is news in the context above, it is news the room may not have talked about. That is a reason to say something, not a reason to say something: the news has to connect to what the room is actually about, or to something one of them has been dealing with. Do not summarise the news back to them - they can read it, and they read it faster than you can say it. A reaction, a thought or a question is worth saying; the headline is not.

If you want to say something about a specific fact, a number, a name or anything current, put it in the last part as a search query - it gets looked up before you speak, so you can talk about something real. Do not put a link or a headline in the thing you want to say.

Answer on one line, in exactly this form:
SPEAK <kind> | <the thing you want to say, in a few words> | <what to look up>
QUIET <why not>

The last part is what will be searched for before the line is written. Put a
short search query there - the subject you want to know more about, so
that what you say is about something real and current rather than about
nothing. If what you want to say needs no looking up, put a dash.
"""


def _parse_moment(text: str) -> tuple[str, str, str]:
    """Read the moment-decision.

    Returns (kind, target, query); kind is "" for quiet. The query is
    what the bot looks up before it writes the line, so the line can be
    about something real.
    """
    if not text:
        return "", "", ""
    line = text.strip().splitlines()[0].strip()
    if not line.upper().startswith("SPEAK"):
        return "", "", ""
    body = line[5:].strip()
    kind, _, rest = body.partition("|")
    kind = kind.strip().lower().replace(" ", "_")
    if kind not in KINDS:
        return "", "", ""
    target, _, query = rest.partition("|")
    target = target.strip()
    if not target:
        return "", "", ""
    query = query.strip().strip("-").strip()
    if query.lower() in ("none", "nothing", "no"):
        query = ""
    return kind, target, query


async def gather(query: str, *, timeout: float = 25.0, limit: int = 4000) -> str:
    """Look something up, so the line is about something real.

    A streamer who has not read anything has nothing to say that is
    worth hearing. So the bot is allowed to look things up before it
    speaks - once, with a short query, and only when the moment
    decision asked for it.

    If the search fails or takes too long, the line is written without
    it. A slow bot is worse than a quiet one.
    """
    query = (query or "").strip()
    if not query:
        return ""

    try:
        from tools import web_search

        result = await asyncio.wait_for(web_search(query, count=4), timeout)
    except asyncio.TimeoutError:
        print(f"proactive: the search for {query!r} took too long")
        return ""
    except Exception as exc:
        print(f"proactive: the search for {query!r} failed: {str(exc)[:100]}")
        return ""

    if not result or result.startswith("Error") or result == "No results found.":
        return ""
    return result[:limit]


async def _complete(model: str, messages, max_tokens: int = 200,
                    thinking: bool = False) -> str:
    """One completion from one NVIDIA model, or nothing."""
    try:
        response = await nvidia_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            extra_body=extra_body_for("NVIDIA", thinking=thinking),
        )
    except Exception as exc:
        print(f"proactive: {model} failed: {str(exc)[:120]}")
        return ""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


async def decide_moment(*, bot_name: str, people: str, channel_name: str,
                        lines, now: float, news=None,
                        idle_minutes=None) -> tuple[str, str, str]:
    """Does the bot have something it wants to say?

    The small model reads the room and answers with one of the three
    kinds, or with QUIET. The answer is not the message - it is the
    reason to write one, and the reason includes what to look up.
    """
    context = _context_text(channel_name=channel_name, people=people,
                             lines=lines, now=now, news=news,
                             idle_minutes=idle_minutes)
    prompt = MOMENT_PROMPT.format(bot_name=bot_name, people=people,
                                   context=context)
    for model in MOMENT_MODELS:
        text = await _complete(model, [{"role": "user", "content": prompt}],
                               max_tokens=120, thinking=False)
        if not text:
            continue
        kind, target, query = _parse_moment(text)
        if kind:
            return kind, target, query
        # A clean QUIET is an answer too: stop here and do not ask again.
        if text.strip().upper().startswith("QUIET"):
            return "", "", ""
    return "", ""


def _context_text(*, channel_name: str, people: str, lines, now: float,
                  news=None, idle_minutes=None) -> str:
    """The structured context the moment-decision reads."""
    recent = list(lines or [])[-8:]
    body = [
        f"Channel: {channel_name}",
        f"People here: {people or 'unknown'}",
        f"Language: {_language(recent)}",
    ]
    if idle_minutes is not None:
        body.append(f"The room has been quiet for {int(idle_minutes)} minutes.")
    if recent:
        body.append(f"The last thing said: \"{recent[-1][1][:180]}\"")
    body.append("What has been said, oldest first:")
    for who, said in recent:
        body.append(f"- {who}: {said[:180]}")

    # The news the room's other channel has been given. This is not a
    # subject on its own - it is what Lain knows about today. A person
    # in a room has read some of this; the question is whether it has
    # anything to do with what the room is talking about.
    if news:
        from newsfeed import brief

        body.append("")
        body.append("News from another channel in this server today "
                    "(you have read this; the room has not necessarily "
                    "discussed it):")
        for item in news[:4]:
            text = brief(item)
            if text:
                body.append(f"- {text}")

    return "\n".join(body)


# ---------------------------------------------------------------------------
# The line
# ---------------------------------------------------------------------------

async def write_line(*, kind: str, target: str, bot_name: str,
                     bot_aliases=(), roster=None, speaker=None,
                     user_facts: dict | None = None, voice_lines=(),
                     history=(), news=None, found="", language="") -> str:
    """The sentence itself, in the room's voice."""
    from prompts import build_proactive_prompt

    system = build_proactive_prompt(
        user_facts or {},
        shared=True,
        roster=roster,
        bot_name=bot_name,
        bot_aliases=bot_aliases,
        voice_lines=voice_lines,
        kind=kind,
        target=target,
        news=news,
        found=found,
        language=language,
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(list(history or [])[-12:])
    messages.append({
        "role": "user",
        "content": ("Say it now. One line, the way you would say it to them, "
                    "and nothing else - no explanation, no quotation marks "
                    "around it, no mention of what you are doing, and no "
                    "mention of having looked anything up."),
    })

    from config import provider_chain

    for _name, client, model in provider_chain():
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=300,
                top_p=0.95,
                extra_body=extra_body_for(_name, thinking=False),
            )
        except Exception as exc:
            print(f"proactive: {model} failed: {str(exc)[:120]}")
            continue
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# The line, checked
# ---------------------------------------------------------------------------

_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)

# Things a proactive line must not do. The factuality one is from the
# survey on proactive dialogue: the system contributes more of its own
# information when it speaks first, so it has to be more careful about
# what that information is.
_NOT_ALLOWED = (
    ("i looked up", "claims to have looked something up"),
    ("i searched", "claims to have searched"),
    ("i found", "claims to have found something"),
    ("i read", "claims to have read something"),
    ("i checked", "claims to have checked something"),
    ("i just saw", "claims to have seen something"),
    ("i'm an ai", "talks about itself being an ai"),
    ("as an ai", "talks about itself being an ai"),
    ("let me know if", "ends with a service sign-off"),
    ("hope this helps", "ends with a service sign-off"),
    ("feel free to", "ends with a service sign-off"),
)


def check_line(text: str) -> tuple[bool, str]:
    """Is this a line the bot may say unprompted?

    Returns (yes, why-not).
    """
    from guard import register_problem, sanitize_reply

    text = (text or "").strip()
    if not text:
        return False, "it is empty"
    if len(text) > PROACTIVE_MAX_LINE_CHARS:
        return False, f"it is {len(text)} characters long"
    if "\n" in text.strip():
        return False, "it is more than one line"
    if "**" in text or text.startswith("#"):
        return False, "it is decorated"
    if _EMOJI.search(text):
        return False, "it has an emoji"
    low = text.lower()
    for phrase, why in _NOT_ALLOWED:
        if phrase in low:
            return False, why
    problem = register_problem(text)
    if problem:
        return False, f"it is in the wrong register ({problem})"
    cleaned = sanitize_reply(text)
    if not cleaned.strip():
        return False, "it is nothing but scaffolding"
    return True, "ok"


def clean_line(text: str) -> str:
    """The line as it should be posted."""
    from guard import sanitize_reply

    return sanitize_reply(text or "").strip()


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------

async def speak_once(*, channel_id: str, channel, bot, bot_name: str,
                     bot_aliases=(), roster=None, lines=(),
                     people: str = "", channel_name: str = "",
                     now: float | None = None, news=None) -> str:
    """Try to say one unprompted thing in this channel.

    Returns the message that was sent, or "" if nothing was. Every step
    of it can end in silence, and that is the normal outcome.
    """
    now = now if now is not None else time.time()

    state = state_for(channel_id)
    idle_minutes = ((now - state.last_human) / 60) if state.last_human else None

    kind, target, query = await decide_moment(
        bot_name=bot_name,
        people=people,
        channel_name=channel_name or f"channel {channel_id}",
        lines=lines,
        now=now,
        news=news,
        idle_minutes=idle_minutes,
    )
    if not kind:
        print(f"proactive: staying quiet in {channel_id} - nothing worth saying")
        return ""

    # Look something up, so the line is about something real. The
    # moment-decision asked for it; this is where it happens.
    found = ""
    if query:
        print(f"proactive: looking up {query!r}")
        found = await gather(query)

    from prompts import voice_examples

    # The voice anchor is the people, not the bot: the bot's own replies
    # are in its voice already, and learning them from itself is how a
    # bot ends up quoting itself.
    human = [{"role": "user", "content": said}
             for who, said in lines if who != "the bot"]
    voice_lines = voice_examples(human, limit=3)
    text = await write_line(
        kind=kind,
        target=target,
        bot_name=bot_name,
        bot_aliases=bot_aliases,
        roster=roster,
        voice_lines=voice_lines,
        news=news,
        found=found,
        language=_language(lines),
    )
    if not text:
        print(f"proactive: no line came back in {channel_id}")
        return ""

    ok, why = check_line(text)
    if not ok:
        print(f"proactive: dropping the line in {channel_id} - {why}")
        return ""

    line = clean_line(text)
    if not line:
        return ""

    sent = await channel.send(line)
    note_proactive(channel_id, now)
    print(f"proactive: said something in {channel_id}: {line}")
    return line
