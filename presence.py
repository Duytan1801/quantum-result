"""Deciding when the bot should speak.

This used to be a list of keywords. The bot had a table of "task verbs" and a
table of "subjects it knows about", and it decided from those whether a
message was for it. That is brittle in a way that is easy to see once you
look at two real messages:

    "can you look up the Nvidia earnings date?"
    "did you finish the report?"

Those differ by a couple of words, and no keyword list gets both right at the
same time - every list either misses the first or answers the second.
So the list is gone.

What is left is two layers:

**Structure**, which is free and exact. A real mention, a reply to one of
the bot's own messages, the bot's own name (derived from its Discord user,
not written down). Those are unambiguous, and they are how people actually
ask a bot something.

**Reading the room**, for everything else. A small call to a model,
asked one question - should I reply to this? - and answered with a single
character. It sees the message, who wrote it, and who else is in the
chat, so it can tell a question aimed at the bot from a question aimed
at the person next to it.

That scanner is MiniMax, while the answering model is Laguna. They are
not the same on purpose: the scanner has to be quick and literal about
two characters, the answer has to sound like a person, and a bot that
read the room with the same model that then wrote the reply would be
marking its own homework.

Since then the log showed two more things, and both are handled here.

**The message may not be finished.** People type in pieces. Answering the
first half of a half-typed sentence is the most obviously robotic thing a
bot can do, so the classifier can now say "wait" instead of "answer", and
the bot waits a few seconds to see whether more is coming. This is the
same idea as the turn detectors used for voice agents (Pype's Cue v1 asks
whether the speaker is DONE, LIKELY_DONE or MID_THOUGHT, and gives each
state a different silence budget): a reply is not a yes or a no, it is a
decision about *when*.

**The bot has to know how big a reply the message wants.** A person
answers in proportion - a remark made in passing gets a line back, a
question gets its answer. So the classifier also says what kind of turn
this is, and that decides the length of the reply.

Both are asked in the same call, so the bot still pays for exactly one
small call per message. The answer is two characters: a decision and a
shape.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass

from people import (addresses_bot, mentioned, name_collisions, names_in,
                    replies_to_bot)

# The classifier's answer is cached by message id: the same message is
# looked at twice - once before the channel lock and once after it - and
# the classifier is a network call that must not be paid for twice.
_DECISIONS: dict = {}
_CACHE_MAX = 500

# How many tokens the classifier may use. It answers two characters, so
# this is a safety net rather than a budget.
_CLASSIFIER_TOKENS = int(os.getenv("PRESENCE_CLASSIFIER_TOKENS", "4"))

# The endpoint rate-limits the scanner model: NVIDIA serves several
# models from one pool of requests per minute, and the scanner - one
# small call per message - is enough to hit the limit on a busy room.
# Measured on the live endpoint: back-to-back calls come back 429 about
# half the time, while a call two seconds later usually goes through.
#
# So the scanner waits a moment and asks again before it gives up on the
# model and falls back. The fallback is there for a provider that is
# down, not for a provider that is busy for one second, and the wait is
# short enough that the person has not noticed it.
_RETRY_DELAY = float(os.getenv("PRESENCE_RETRY_DELAY", "2.0"))

# How long to wait when a message looks unfinished. Long enough for the
# rest of a sentence to arrive, short enough that the person does not
# think the bot ignored them.
WAIT_SECONDS = float(os.getenv("PRESENCE_WAIT_SECONDS", "4"))

# The shapes a turn can be. They decide how long the reply should be.
#
#   aside    - a remark made in passing. A line or two back, not an essay.
#   question - something with an answer. Answer it properly.
#   deep     - a real job: a fact to look up, code, a file, a calculation.
#
# The rule about length is taken from a project that measured it: a note
# that told the model a passing remark wanted a short reply took the
# median reply to those remarks from 52 words down to 16, and the same
# note applied to questions made the voice worse, so questions do not get
# one.
SHAPES = {
    "1": "aside",
    "2": "question",
    "3": "deep",
}

# The decisions. A = answer now, W = wait, Q = stay quiet.
DECISIONS = {
    "A": "answer",
    "W": "wait",
    "Q": "quiet",
}


@dataclass
class Decision:
    """Should the bot answer, and why."""

    reply: bool
    reason: str
    from_classifier: bool = False
    # True when the message looks unfinished and the bot should wait for
    # the rest of it rather than answer now.
    wait: bool = False
    # "aside", "question" or "deep" - how big a reply this turn wants.
    shape: str = ""


def channel_scope(channel_id) -> str:
    """The key a shared channel's context is stored under."""
    return f"channel:{channel_id}"


def _cache_put(key, value) -> None:
    _DECISIONS[key] = value
    if len(_DECISIONS) > _CACHE_MAX:
        for old in list(_DECISIONS)[:-_CACHE_MAX]:
            _DECISIONS.pop(old, None)


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_RULE_CHAT = (
    "Chat mode is ON for this chat: the bot is here to JOIN IN, not only to "
    "be summoned. Work through this in order and stop at the first one that "
    "fits:\n"
    "1. The message is not finished - the person is still typing. Wait (W).\n"
    "2. The message is addressed to the bot - its name, a mention, a reply to "
    "one of its own messages. Answer it (A).\n"
    "3. The message asks the bot to do something, or asks a question that has "
    "a real answer - a fact, a date, a price, the weather, a paper, code, "
    "maths, a file, a calculation, or 'can you check X'. Answer it (A).\n"
    "4. The message is somebody telling the bot how they are, or talking "
    "about their day. Answer it (A) - a line or two, not a lecture.\n"
    "5. The message is finished, but it is banter, an acknowledgement, a "
    "laugh, a remark in passing, or a reply to the other person in the room. "
    "Stay quiet (Q) and let them have the conversation.\n"
    "A message that only mentions the bot in passing ('the bot is slow "
    "today', 'Mốt chắc lain2 cướp luôn e từ lain', 'I love lain2 more than "
    "you') is not addressed to it - that is (Q) unless it also asks for "
    "something."
)

_CLASSIFIER_RULE_QUIET = (
    "Work through this in order and stop at the first one that fits:\n"
    "1. The message is not finished - the person is still typing. Wait (W).\n"
    "2. The message is clearly addressed to the bot: a mention, its name, a "
    "reply to one of its own messages, or a direct request to the bot ('can "
    "you look up X'). Answer it (A).\n"
    "3. The message is finished, but it is for somebody else, or it is "
    "banter, or a remark in passing. Stay quiet (Q).\n"
    "Anything that looks like a task but is aimed at somebody else is (Q)."
)

# Four examples of each answer. The classifier answers two characters, so
# the examples are what decide the character - without them it drifts
# towards "Q" and the bot goes quiet in a room it should be joining, and
# it also drifts the other way and answers messages meant for the person
# next to it.
_CLASSIFIER_EXAMPLES = (
    "Examples:\n"
    'Message from someone in the chat: "can you look up the Nvidia earnings date?" -> A3\n'
    'Message from someone in the chat: "can you look up the Nvidia" -> W3\n'
    'Message from someone in the chat: "did you finish the report?" -> Q2\n'
    'Message from someone in the chat: "what is the weather in Hanoi right now?" -> A3\n'
    'Message from someone in the chat: "mệt quá đi, hôm nay học nhiều quá" -> A1\n'
    'Message from someone in the chat: "i was thinking, like," -> W1\n'
    'Message from someone in the chat: "the thing we talked about yesterday, the one where" -> W2\n'
    'Message from someone in the chat: "what do you think about the new model?" -> A2\n'
    'Message from someone in the chat: "cai gi vậy trời" -> Q1\n'
    'Message from someone in the chat: "hows jade" -> Q2\n'
    'Message from someone in the chat: "hello?" -> Q1\n'
    'Message from someone in the chat: "can you check on the plots?" -> A3\n'
    'Message from someone in the chat: "is there any research that shows '
    'something?" -> A3\n'
    'Message from someone in the chat: "U hear me, lain?" -> A1\n'
    'Message from someone in the chat: "Lainy, what do you think?" -> A2\n'
    'Message from someone in the chat: "E yeu lain2 hon ca toi" -> Q1\n'
    'Message from someone in the chat: "Mốt chắc lain2 cướp luôn e từ lain" -> Q1\n'
    'Message from someone in the chat, replying to the other person: '
    '"the new model is slower but smarter" -> Q2\n'
)

# When the message is unfinished. The examples are the point: a sentence
# that trails off is the one case where answering now is worse than
# waiting, and a model asked "should I answer this?" will usually say
# yes to a question even when the question is not finished yet.
_CLASSIFIER_WAIT_RULE = (
    "What counts as not finished: the sentence trails off, it ends in 'and' "
    "or 'so' or 'because' with nothing after it, a number or a name is only "
    "half written, or the person is clearly still typing. A complete thought "
    "is not unfinished, however short it is, and a message that ends in a "
    "question mark is finished - however long it is. Examples of unfinished: "
    "'can you look up the Nvidia' / 'the thing we talked about yesterday, "
    "the one where' / 'i was thinking, like,'. Examples of finished: 'can "
    "you look up the Nvidia earnings date?' / 'mệt quá đi' / 'ok'."
)

# The cooldown. A bot that answers every thirty seconds turns a room into
# a monologue. When the bot has just spoken, it only answers again if the
# new message is clearly for it.
_CLASSIFIER_COOLDOWN_RULE = (
    "You spoke {seconds} seconds ago in this chat, so the room is not yours "
    "yet. This does not override the rules above: if the message names the "
    "bot or asks the bot for something, answer it. But a remark, a joke or "
    "a message aimed at somebody else is theirs, not yours, however "
    "interesting it is - stay quiet (Q)."
)


def _name_note(hits, collisions) -> str:
    """What to tell the classifier about the bot's name in the message.

    Two things can be true at once: the message uses the bot's name, and
    that name also belongs to somebody else in the room. The bot is
    "lain2"; one of the people in the chat is called "Lain". The model
    has to be told both, or it cannot tell "U hear me, lain?" (the bot)
    from "Lain" (Damian giving his name).
    """
    if not hits:
        return ""
    names = ", ".join(f"'{h}'" for h in hits)
    clashes = {name: who for name, who in collisions.items() if name in hits}
    if clashes:
        bits = ", ".join(
            f"'{name}' (also {who})" for name, who in clashes.items()
        )
        return (
            f"\nThe bot's name appears in this message: {names}. But {bits} "
            f"is also the name of somebody else in this chat. A message that "
            f"is only that name, with nothing else, is probably that person "
            f"giving their name - stay quiet (Q). A message that talks to "
            f"the bot ('U hear me, lain?', 'yo lain') is for the bot - "
            f"answer it (A).\n"
        )
    return (
        f"\nThe bot's name appears in this message: {names}. If the bot is "
        f"being addressed - 'U hear me, lain?', 'yo lain' - answer (A). If "
        f"the bot is only being talked about, in the third person, stay "
        f"quiet (Q).\n"
    )


def _classifier_prompt(text: str, *, author_name: str, roster_line: str,
                       bot_name: str, chat_on: bool,
                       recent=(), reply_to: str = "",
                       bot_spoke_seconds: float | None = None,
                       final: bool = False, name_note: str = "",
                       bot_aliases: str = "") -> str:
    rule = _CLASSIFIER_RULE_CHAT if chat_on else _CLASSIFIER_RULE_QUIET
    people = roster_line or author_name

    context = ""
    if recent:
        lines = []
        for name, said in recent:
            lines.append(f"{name}: {said}")
        context = (
            "\nThe last few messages in the chat, oldest first:\n"
            + "\n".join(lines)
            + "\n"
        )

    reply_note = ""
    if reply_to:
        reply_note = (
            f"\nThis message is a reply to {reply_to}'s message, so it is "
            f"aimed at {reply_to} and not at the bot. Answer it only if it "
            f"also asks the bot for something.\n"
        )

    cooldown = ""
    # The cooldown only matters when the bot has just spoken. Told about a
    # reply from ten minutes ago it is just a nudge towards silence, and
    # the classifier is already too ready to stay quiet.
    if bot_spoke_seconds is not None and bot_spoke_seconds < 180:
        cooldown = "\n" + _CLASSIFIER_COOLDOWN_RULE.format(
            seconds=int(bot_spoke_seconds))

    final_note = ""
    if final:
        final_note = (
            "\nThe person has stopped typing now: whatever they sent is the "
            "whole message. If it is still unfinished, answer it as it is - "
            "do not wait again.\n"
        )

    alias_note = ""
    if bot_aliases:
        alias_note = f" (people also call it: {bot_aliases})"

    return (
        f'A bot named "{bot_name}"{alias_note} is in a group chat with: '
        f"{people}.\n"
        + context
        + f"\nMessage from {author_name}:\n"
        + f'"{text[:1500]}"\n'
        + reply_note
        + name_note
        + final_note
        + "\n"
        + _CLASSIFIER_WAIT_RULE
        + "\n"
        + f"{rule}\n"
        + (_CLASSIFIER_EXAMPLES if chat_on else "")
        + cooldown
        + "\nAnswer with two characters and nothing else.\n"
        + "First character - what to do with the message:\n"
        + "  A = answer it now\n"
        + "  W = wait, the message is not finished and more is coming\n"
        + "  Q = stay quiet, it is not for you\n"
        + "Second character - what kind of turn this is:\n"
        + "  1 = a passing remark; a line or two back is right\n"
        + "  2 = a question that has an answer\n"
        + "  3 = something that needs a real job done\n"
    )


def _parse_verdict(content: str) -> tuple[str | None, str]:
    """Read the classifier's two-character answer.

    Returns (decision, shape). The decision is "answer", "wait" or
    "quiet"; None means the answer was not usable - and then the bot
    stays quiet, because a bot that cannot tell is better quiet than
    wrong.
    """
    decision = None
    shape = ""
    for char in (content or "").strip():
        upper = char.upper()
        if decision is None and upper in DECISIONS:
            decision = DECISIONS[upper]
            continue
        if not shape and char in SHAPES:
            shape = SHAPES[char]
    return decision, shape


def scanner_models() -> tuple[str, ...]:
    """The models that read the room, in order.

    Bolt scans first - it is smaller and a fraction of the cost, and it's
    good at the presence question. Nemotron is behind it for when Bolt
    or the endpoint is rate-limited. Laguna is deliberately not here: it
    answers. The scanner and the answer are two different jobs - one
    wants a literal two-character verdict, the other wants to sound
    like a person - and running both on the same model would mean the
    bot marks its own homework.

    Groq is not here either. It is the answering chain's last resort,
    and a scan that quietly fell through to another provider would be
    another model's opinion, not a fallback for a failed one.
    """
    from config import BOLT_MODEL, NVIDIA_MODEL

    return (BOLT_MODEL, NVIDIA_MODEL)


def _rate_limited(exc: Exception) -> bool:
    """Was this call turned away for being too fast?

    The OpenAI client raises a RateLimitError for a 429, but the check is
    done on the text as well, because a provider can also put the code
    in the body of an ordinary error.
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()


async def classify_reply(text: str, *, author_name: str, roster_line: str,
                          bot_name: str, chat_on: bool,
                          recent=(), reply_to: str = "",
                          bot_spoke_seconds: float | None = None,
                          final: bool = False, name_note: str = "",
                          bot_aliases: str = ""
                          ) -> tuple[bool, bool, str] | None:
    """Ask the model whether this message is for the bot.

    Returns (reply, wait, shape), or None if the call failed - in which
    case the bot stays quiet, because a bot that cannot tell is better
    quiet than wrong.

    The scanner is Bolt (Nemotron 3.5 bolt), with Nemotron 3.5 Super VL as
    the fallback. The answering model is Laguna, and the two are deliberately
    different: reading a room and answering in it want different things, and
    the scanner is the one that has to be quick and literal about two
    characters while the answer is the one that has to sound like a
    person. Laguna is not used here - if the scanner and the answer came
    from the same model, the bot would be marking its own homework and
    would inherit the same blind spots in both.

    Groq is not used here either: it is the last resort of the answering
    chain, and a failed scan should not quietly become a different
    provider's opinion.
    """
    try:
        from config import extra_body_for, nvidia_client

        models = scanner_models()
    except Exception as exc:
        print(f"presence classifier unavailable: {exc}")
        return None

    if nvidia_client is None:
        return None

    prompt = _classifier_prompt(
        text,
        author_name=author_name,
        roster_line=roster_line,
        bot_name=bot_name,
        chat_on=chat_on,
        recent=recent,
        reply_to=reply_to,
        bot_spoke_seconds=bot_spoke_seconds,
        final=final,
        name_note=name_note,
        bot_aliases=bot_aliases,
    )

    # Thinking is switched off: the answer is two characters, and a model
    # that reasons first would spend the whole token budget before saying
    # "A1".
    for model in models:
        tokens = _CLASSIFIER_TOKENS
        waited = False
        while True:
            try:
                resp = await nvidia_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=tokens,
                    extra_body=extra_body_for("BOLT", thinking=False),
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                if _rate_limited(exc) and not waited:
                    # Busy, not broken: wait a moment and ask again
                    # before giving up on this model.
                    waited = True
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                print(f"presence classifier failed on {model}: {exc}")
                break
            decision, shape = _parse_verdict(content)
            if decision is not None:
                if decision == "wait":
                    return False, True, shape
                if decision == "answer":
                    return True, False, shape
                return False, False, shape
            if tokens < 16:
                # Nothing usable came back; give it one more try with
                # more room.
                tokens = 16
                continue
            break
    return None


# ---------------------------------------------------------------------------
# Does this message ask for something that should be looked up?
# ---------------------------------------------------------------------------
#
# This is the difference between "what is filtering theory?" - where the
# answer has to come from a source - and "mệt quá đi" - where there is no
# answer to look up. It is asked by the model, not by a list of words,
# because the words are not what tells them apart.
#
# It is only asked when a turn has already answered without calling a
# tool, so it is paid for rarely and only when it can change something.

_LOOKUP_PROMPT = (
    "Does this message ask for information that has to come from a source?\n"
    "Answer Y if it is asking for a fact, a date, a definition, a paper, "
    "code, a calculation, current information, a price, or anything with a "
    "right answer that is not in the message itself.\n"
    "Answer N if the person is talking about themselves, their day, their "
    "feelings, or saying something with no answer in it.\n"
    'Message: "{text}"\n'
    "Answer with one character, Y or N."
)


async def needs_lookup(text: str) -> bool:
    """Does this message ask for something that should be looked up?

    A tiny call, made only when a turn has already answered without
    using a tool. False when the call fails: a bot that cannot tell is
    better quiet than wrong, and a lookup that was not needed costs
    nothing but a second.

    The same scanner model as the presence decision - MiniMax first,
    Nemotron behind it - so the whole "should this have used a tool?"
    side of the bot speaks with one voice.
    """
    if not (text or "").strip():
        return False
    try:
        from config import extra_body_for, nvidia_client

        models = scanner_models()
    except Exception as exc:
        print(f"lookup check unavailable: {exc}")
        return False

    if nvidia_client is None:
        return False

    prompt = _LOOKUP_PROMPT.replace("{text}", (text or "")[:1000])
    for model in models:
        waited = False
        while True:
            try:
                resp = await nvidia_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=4,
                    extra_body=extra_body_for("BOLT", thinking=False),
                )
                content = (resp.choices[0].message.content or "").strip().upper()
            except Exception as exc:
                if _rate_limited(exc) and not waited:
                    waited = True
                    await asyncio.sleep(_RETRY_DELAY)
                    continue
                print(f"lookup check failed on {model}: {exc}")
                break
            if content.startswith("Y"):
                return True
            if content.startswith("N"):
                return False
            # Nothing usable came back; ask the fallback.
            break
    return False


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

async def decide(message, *, is_dm: bool, shared: bool, bot_user_id=None,
                 bot_names=(), bot_message_ids=None, chat_on: bool = False,
                 roster=None, author_name: str = "", roster_line: str = "",
                 bot_name: str = "the bot", recent=(),
                 reply_to: str = "", bot_spoke_seconds: float | None = None,
                 final: bool = False, cache_extra: str = ""
                 ) -> Decision:
    """Should the bot answer this message?

    The structural checks are exact and free. Everything else is decided by
    the model reading the room, and the answer is cached so the second look
    (after the channel lock) costs nothing.

    `cache_extra` distinguishes two looks at the same message: the first
    one, and the second one after the room has had time to change. When the
    room has changed, the classifier is asked again with the new context -
    a message that looked like it was for the bot can look quite different
    once somebody else has answered it.
    """
    author = getattr(message, "author", None)
    if author is not None and getattr(author, "bot", False):
        return Decision(False, "the author is a bot")

    text = getattr(message, "content", "") or ""

    if is_dm:
        return Decision(True, "private message")

    if not shared:
        # One human and the bot. There is nobody else to talk to.
        return Decision(True, "one-to-one channel")

    if mentioned(message, bot_user_id):
        return Decision(True, "mentioned")

    if replies_to_bot(message, bot_message_ids):
        return Decision(True, "replying to the bot")

    # Some of the bot's names may also be somebody else's name in this
    # room - the bot is "lain2" and one of the people is called "Lain".
    # Those names cannot be used as an address on their own, or the bot
    # would answer every time somebody gave their name. The model decides
    # them, because it can see the room.
    collisions = name_collisions(bot_names, roster)

    # The bot's name at the front of the message is an address: "sup lain",
    # "yo lain". The same name in the middle of a sentence is somebody
    # talking about the bot - "I love lain2 more than you" - and that is
    # left to the model, which can see who is talking to whom. A name that
    # belongs to somebody else in the room is only an address when there
    # are other words around it.
    if addresses_bot(text, bot_names, exclude=collisions) and not reply_to:
        return Decision(True, "named in the message")

    # Nothing structural. Let the model read the room.
    hits = names_in(text, bot_names)
    name_note = _name_note(hits, collisions)
    aliases = ", ".join(n for n in (bot_names or ()) if n != bot_name.lower())
    key = str(getattr(message, "id", "") or "")
    if key:
        key = f"{key}:{cache_extra}"
    cached = _DECISIONS.get(key) if key else None
    if cached is None:
        cached = await classify_reply(
            text,
            author_name=author_name or "someone",
            roster_line=roster_line,
            bot_name=bot_name,
            chat_on=chat_on,
            recent=recent,
            reply_to=reply_to,
            bot_spoke_seconds=bot_spoke_seconds,
            final=final,
            name_note=name_note,
            bot_aliases=aliases,
        )
        if key and cached is not None:
            _cache_put(key, cached)
    if cached is None:
        return Decision(False, "could not tell whether it was for the bot",
                        from_classifier=True)
    reply, wait, shape = cached
    if wait:
        if final:
            # The wait is over: the person stopped typing and the message
            # is still unfinished. Answer it as it is rather than leave it
            # unanswered - a bot that goes quiet when somebody repeats
            # themselves is worse than one that answers a half sentence.
            return Decision(True, "the person has stopped typing; answering "
                                   "it as it is",
                            shape=shape, from_classifier=True)
        return Decision(False, "the message is not finished yet", wait=True,
                        shape=shape, from_classifier=True)
    if reply:
        return Decision(True, "the model read the room and said it was for the bot",
                        shape=shape, from_classifier=True)
    return Decision(False, "the model read the room and said it was not for the bot",
                    shape=shape, from_classifier=True)


class ChannelTracker:
    """Who has spoken in each channel, and which channel is shared."""

    def __init__(self, shared_at: int = 2):
        self._people: dict[str, set] = {}
        self._recent: dict[str, list] = {}
        self._shared: set = set()
        self.shared_at = shared_at

    def note(self, channel_id, user_id) -> None:
        """Record that someone spoke in a channel."""
        key = str(channel_id)
        self._people.setdefault(key, set()).add(str(user_id))
        recent = self._recent.setdefault(key, [])
        if user_id not in recent:
            recent.append(user_id)
            if len(recent) > 20:
                recent.pop(0)

    def mark_shared(self, channel_id) -> None:
        """Record that this channel is a room, whatever the tracker knows.

        After a restart the tracker starts empty, so a channel that has
        been shared for weeks looks like a private chat until somebody speaks.
        """
        self._shared.add(str(channel_id))

    def participants(self, channel_id) -> set:
        return set(self._people.get(str(channel_id), set()))

    def is_shared(self, channel_id) -> bool:
        """Are there two or more people in this channel?"""
        key = str(channel_id)
        if key in self._shared:
            return True
        return len(self.participants(key)) >= self.shared_at

    def scope_for(self, channel_id) -> str:
        """The key this channel's shared context is stored under."""
        if self.is_shared(channel_id):
            return channel_scope(channel_id)
        return None
