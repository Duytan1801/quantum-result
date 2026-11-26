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
                 "day", "today", "awaiting_reply", "ignored", "subjects")

    def __init__(self):
        self.last_human = 0.0
        self.last_bot = 0.0
        self.last_proactive = 0.0
        self.day = ""
        self.today = 0
        self.awaiting_reply = 0.0
        # How many unprompted messages in this channel went unanswered.
        self.ignored = 0
        # The subjects she has already brought up. Not the sentences - the
        # sentences change and the subject does not, and it is the subject
        # she keeps coming back to.
        self.subjects: list[str] = []


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


def note_subject(channel_id: str, subject: str) -> None:
    """The subject she has just brought up in this room.

    The sentences change and the subject does not. A bot that brings up
    "AI and reality perception" three evenings running is not having a
    conversation, it is rereading its own notes - so the subject is
    remembered separately from the line, and the next decision is told
    to pick a different one.
    """
    subject = (subject or "").strip()
    if not subject:
        return
    state = state_for(channel_id)
    state.subjects = [s for s in state.subjects
                      if s.lower() != subject.lower()]
    state.subjects.append(subject)
    del state.subjects[:-3]


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
    if PROACTIVE_MAX_IDLE_MINUTES and idle > PROACTIVE_MAX_IDLE_MINUTES:
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
- join: say something about the topic they were on, or about today - and "today" means the news you have read, which is usually more interesting than asking how somebody is.

This is the way a streamer talks to a chat that has gone quiet. If the room has been quiet for a minute, there is usually something to say - a thought about what they were talking about, a question, a reaction, something from today's news that connects. You are not interrupting anybody: nobody is talking.

Decide whether there is something here worth one line, right now.

Say QUIET only when the moment is genuinely wrong - the room is busy, somebody is mid-sentence, or there is nothing to connect to. A room that has been quiet for a while is exactly the room you are here for.

If there is news in the context above, it is news the room may not have talked about. That is a reason to say something, not a reason to say something: the news has to connect to what the room is actually about, or to something one of them has been dealing with. Do not summarise the news back to them - they can read it, and they read it faster than you can say it. A reaction, a thought or a question is worth saying; the headline is not.

If you want to say something about a specific fact, a number, a name or anything current, put it in the last part as a search query - it gets looked up before you speak, so you can talk about something real. Do not put a link or a headline in the thing you want to say.

The lines under "What you already said" are yours. If nobody answered them, do not repeat them and do not pick the same subject again - move on and change the topic, or say something about today's news instead. A different remark about the same thing is still the same remark.

The list under "Subjects you have already brought up" is the same thing one level up: those are the topics you have already had your say about. Do not pick one of them again - even from a different angle, even with different words. Pick something else, or say nothing.

Do not ask the same kind of question twice in a row either. If you already asked how somebody is doing, do not ask again - bring up something from today's news, or something else they were talking about. A room where every line is the same question gets muted, and then you are talking to nobody.

Answer on one line, in exactly this form:
SPEAK <kind> | <the subject, in a few words> | <what to look up>
QUIET <why not>

The middle part is the subject, not the sentence - the line itself is written afterwards, and it will not be a version of what you write here. Name the thing: the person, the company, the topic. "Tidal removing AI songs" is a subject; "tidal is removing ai songs, how weird" is a sentence, and sentences are what the bot ends up sounding like.

The last part is what will be searched for before the line is written. Put a short search query there - the subject you want to know more about, so that what you say is about something real and current rather than about nothing. If the subject needs no looking up, put a dash.
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


# The words that are in every subject: the kind of thing, rather than
# the thing. "news about the breach" and "the breach" are the same
# subject, and "the latest news" is not a subject at all.
_SUBJECT_STOP = {
    "news", "story", "thing", "things", "talk", "update", "article",
    "report", "latest", "today", "connection", "asset", "stuff",
    "people", "time", "day", "way", "part", "case", "issue", "rumor",
    "rumour", "claim", "claims", "allegation", "about", "over", "with",
    "from", "that", "this",
}


def _subject_words(text: str) -> set:
    """The words of a subject that name something."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text or "")
    return {w.lower() for w in words if w.lower() not in _SUBJECT_STOP}


def _same_subject(new: str, used) -> bool:
    """Is this the subject she has already brought up?

    The decision is asked not to repeat a subject, and a small model
    asked a question like that does sometimes answer with the same
    thing anyway - so the answer is checked here rather than trusted.
    Two subjects are the same when they name the same thing: "MSS
    asset connection" and "MSS asset Kang Wei" are one story, however
    they are worded.
    """
    new_words = _subject_words(new)
    if not new_words:
        return False
    for old in (used or []):
        old_words = _subject_words(str(old))
        if not old_words:
            continue
        shared = new_words & old_words
        if not shared:
            continue
        # One shared name is enough when the subject is short - a short
        # subject is mostly the name of the thing, and that is what
        # makes two short subjects the same story.
        if len(new_words) <= 3 or len(old_words) <= 3:
            return True
        if len(shared) / len(new_words) >= 0.5:
            return True
    return False


async def decide_moment(*, bot_name: str, people: str, channel_name: str,
                        lines, now: float, news=None,
                        idle_minutes=None, subjects=()) -> tuple[str, str, str]:
    """Does the bot have something it wants to say?

    The small model reads the room and answers with one of the three
    kinds, or with QUIET. The answer is not the message - it is the
    reason to write one, and the reason includes what to look up.

    `subjects` are the subjects she has already brought up in this
    room. The prompt says not to repeat them and the answer is checked
    afterwards, because a small model asked a question like that
    sometimes answers with the same thing anyway. A repeated subject is
    asked about once more, with the subject named at it; if it comes
    back the same, she says nothing.
    """
    context = _context_text(channel_name=channel_name, people=people,
                             lines=lines, now=now, news=news,
                             idle_minutes=idle_minutes,
                             subjects=subjects)
    prompt = MOMENT_PROMPT.format(bot_name=bot_name, people=people,
                                   context=context)
    used = [str(s).strip() for s in (subjects or []) if str(s).strip()]

    for attempt in range(2):
        ask = prompt
        if attempt and used:
            ask += (
                "\nYou have already brought up these subjects in this room: "
                + ", ".join(used[-3:])
                + ". Do not pick one of them, and do not pick another angle "
                "on the same story. Pick something else entirely, or say "
                "QUIET.\n"
            )
        repeated = False
        for model in MOMENT_MODELS:
            text = await _complete(model, [{"role": "user", "content": ask}],
                                   max_tokens=120, thinking=False)
            if not text:
                continue
            kind, target, query = _parse_moment(text)
            if kind:
                if used and _same_subject(target, used):
                    # The same story again. Ask once more, more plainly.
                    repeated = True
                    continue
                return kind, target, query
            # A clean QUIET is an answer too: stop here and do not ask again.
            if text.strip().upper().startswith("QUIET"):
                return "", "", ""
        if not repeated:
            break
    return "", "", ""


def _balanced(lines, humans: int = 8, own: int = 2):
    """The room as it should be read: the people, and a little of her.

    In a room that has gone quiet, the last dozen messages are the
    bot's own - it was talking to itself. A transcript like that tells
    the model the room is discussing whatever it has been saying, which
    is how a bot ends up having a conversation with itself. So the
    people come first, and only a couple of her own lines are kept, to
    remind her what she already said.
    """
    items = list(lines or [])
    people = [(i, pair) for i, pair in enumerate(items) if pair[0] != "the bot"]
    hers = [(i, pair) for i, pair in enumerate(items) if pair[0] == "the bot"]
    picked = people[-humans:] + hers[-own:] if own else people[-humans:]
    picked.sort(key=lambda item: item[0])
    return [pair for _i, pair in picked]


def _context_text(*, channel_name: str, people: str, lines, now: float,
                  news=None, idle_minutes=None, subjects=()) -> str:
    """The structured context the moment-decision reads."""
    recent = _balanced(lines)
    body = [
        f"Channel: {channel_name}",
        f"People here: {people or 'unknown'}",
        f"Language: {_language(recent)}",
    ]
    if idle_minutes is not None:
        body.append(f"The room has been quiet for {int(idle_minutes)} minutes.")
    if recent:
        body.append(f"The last thing said: \"{recent[-1][1][:180]}\"")

    # What the bot has already said in this room. It is listed apart from
    # the rest, because the rule is simple: if nobody answered it, say
    # something else. A bot that keeps returning to the same remark every
    # two minutes is not a person, it is a stuck record.
    said_before = [said for who, said in recent if who == "the bot"]
    if said_before:
        body.append("")
        body.append("What you already said in this room (do not say it again):")
        for said in said_before[-3:]:
            body.append(f"- {said[:180]}")

    # The subjects she has already brought up. Not the sentences - the
    # sentences change and the subject does not, and it is the subject
    # she keeps coming back to, night after night. Saying it again in
    # different words is still the same remark.
    used = [str(s).strip() for s in (subjects or []) if str(s).strip()]
    if used:
        body.append("")
        body.append("Subjects you have already brought up in this room "
                    "(pick a different one - a new angle on the same "
                    "subject is still the same subject):")
        for subject in used[-3:]:
            body.append(f"- {subject[:120]}")

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
                     history=(), news=None, found="", language="",
                     said_before=(), lines=(), model: str = "") -> str:
    """The sentence itself, in the room's voice.

    This is Neuro-sama's shape: one prompt - the character, what she
    knows, the room, and then her name - and the model continues as
    her. There is no instruction telling it what to say, because that
    is what produced the caption voice: handed a sentence to
    paraphrase, a model writes a caption.

    `model` is the model this room chose with !model. It is put first
    in the chain, with the rest kept behind it as fallbacks - the same
    choice that decides who answers a message decides who writes the
    unprompted ones.
    """
    from prompts import build_proactive_prompt, build_proactive_turn

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
        said_before=list(said_before or [])[-3:],
    )
    turn = build_proactive_turn(
        bot_name=bot_name,
        news=news,
        found=found,
        target=target,
        lines=lines,
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": turn}]

    from config import chain_with_model, provider_chain

    chain = chain_with_model(model) if model else provider_chain()

    for _name, client, chosen in chain:
        try:
            response = await client.chat.completions.create(
                model=chosen,
                messages=messages,
                temperature=0.8,
                max_completion_tokens=300,
                top_p=0.95,
                extra_body=extra_body_for(_name, thinking=False),
            )
        except Exception as exc:
            print(f"proactive: {chosen} failed: {str(exc)[:120]}")
            continue
        text = (response.choices[0].message.content or "").strip()
        text = _clean_generated(text, bot_name)
        if text:
            return text
    return ""


def _clean_generated(text: str, bot_name: str = "Lain") -> str:
    """The first line of what came back, without the name on the front.

    The model is continuing from "Lain:", so sometimes it writes the
    name again, and sometimes it writes a line and then a thought about
    the line. Only the first line is the line.
    """
    text = (text or "").strip()
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        prefix = f"{bot_name}:"
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
        if line:
            return line
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
    from prompts import proactive_examples

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

    # The examples in the prompt are there for the tone. A model asked to
    # continue a transcript will sometimes continue it with one of them
    # instead of with its own words, and a bot that says the same three
    # sentences on rotation is worse than one that says nothing.
    #
    # It is not only the example word for word. A model asked for a line
    # about a thing will often reach for an example and swap the subject
    # out - which is the same sentence with a new noun in it, and reads
    # exactly as canned as the original.
    words = _words(text)
    for example in proactive_examples("") + proactive_examples("English"):
        example_words = _words(example)
        if not words or not example_words:
            continue
        if words == example_words:
            return False, "it is one of the example lines, word for word"
        shared = words & example_words
        if len(shared) / len(example_words) >= 0.6:
            return False, "it is an example line with the subject swapped out"

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


def _words(text: str) -> set:
    """The words of a line, for comparing one line with another.

    The short ones are left out. "có", "là", "và" and "of" are in half
    the sentences ever written, so a line that shares one of them with
    another line has not really said the same thing.
    """
    return {w for w in re.findall(r"\w+", (text or "").lower())
            if len(w) > 2}


def repeats(text: str, lines, threshold: float = 0.8) -> bool:
    """Is this line the same thing as one the bot already said?

    The rule is the Neuro one: if nobody answered what you said, move
    on and change the subject. A bot that says the same sentence every
    two minutes is not talking, it is stuck - and it reads worse than
    saying nothing at all.
    """
    new = _words(text)
    if not new:
        return False
    for who, said in lines or ():
        if who != "the bot":
            continue
        old = _words(said)
        if not old:
            continue
        # How much of the new line was already in the old one. Containment
        # rather than overlap: a longer old line that contains the new
        # one is still the same remark.
        if len(new & old) / len(new) >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------

async def speak_once(*, channel_id: str, channel, bot, bot_name: str,
                     bot_aliases=(), roster=None, lines=(),
                     people: str = "", channel_name: str = "",
                     now: float | None = None, news=None,
                     model: str = "") -> str:
    """Try to say one unprompted thing in this channel.

    Returns the message that was sent, or "" if nothing was. Every step
    of it can end in silence, and that is the normal outcome.

    `model` is the model this room chose with !model. The decision
    about whether to speak stays on NVIDIA - it is a scan of the room,
    and a scan that quietly fell through to another provider would be
    another model's opinion rather than a fallback for a failed one.
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
        subjects=state.subjects,
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
    said_before = [said for who, said in lines if who == "the bot"]
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
        said_before=said_before,
        lines=lines,
        model=model,
    )
    if not text:
        print(f"proactive: no line came back in {channel_id}")
        return ""

    ok, why = check_line(text)
    if not ok:
        print(f"proactive: dropping the line in {channel_id} - {why}")
        return ""

    # The same remark twice is not a remark. If the model ignored the
    # instruction to move on, the line is dropped rather than posted.
    if repeats(text, lines):
        print(f"proactive: dropping the line in {channel_id} - it repeats "
              f"what was already said")
        return ""

    line = clean_line(text)
    if not line:
        return ""

    sent = await channel.send(line)
    note_proactive(channel_id, now)
    # Remember the subject, not just the sentence. The sentences change
    # and the subject does not - it is what she keeps coming back to.
    note_subject(channel_id, target)
    print(f"proactive: said something in {channel_id}: {line}")
    return line
