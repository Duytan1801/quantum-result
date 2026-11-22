"""Deciding when the bot should speak.

This used to be a list of keywords. The bot had a table of "task verbs" and
a table of "subjects it knows about", and it decided from those whether a
message was for it. That is brittle in a way that is easy to see once you
look at two real messages:

    "can you look up the Nvidia earnings date?"
    "did you finish the report?"

Those differ by a couple of words, and no keyword list gets both right at
the same time - every list either misses the first or answers the second.
So the list is gone.

What is left is two layers:

**Structure**, which is free and exact. A real mention, a reply to one of
the bot's own messages, the bot's own name (derived from its Discord user,
not written down). Those are unambiguous, and they are how people actually
ask a bot something.

**Reading the room**, for everything else. A small call to the same model
that does the answering, asked one question - should I reply to this? - and
answered with a single character, 1 or 0. It sees the message, who wrote it,
and who else is in the chat, so it can tell a question aimed at the bot
from a question aimed at the person next to it.
"""

import json
import os
from dataclasses import dataclass

from people import mentioned, replies_to_bot, says_bot_name

# The classifier's answer is cached by message id: the same message is
# looked at twice - once before the channel lock and once after it - and
# the classifier is a network call that must not be paid for twice.
_DECISIONS: dict = {}
_CACHE_MAX = 500

# How many tokens the classifier may use. It answers one character, so this
# is a safety net rather than a budget.
_CLASSIFIER_TOKENS = int(os.getenv("PRESENCE_CLASSIFIER_TOKENS", "4"))


@dataclass
class Decision:
    """Should the bot answer, and why."""

    reply: bool
    reason: str
    from_classifier: bool = False


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
    "be summoned. Reply 1 if any of these is true:\n"
    "- the message is addressed to the bot - a mention, its name, a reply to "
    "one of its own messages, or a direct request ('can you look up X');\n"
    "- the message asks something the bot would clearly be the best person to "
    "answer: a fact, current events, prices, the weather, a paper, code, "
    "maths, or anything where it knows more than the people in the chat;\n"
    "- the message is an open question with a real answer, or a topic the bot "
    "could genuinely add something useful to.\n"
    "Reply 0 when the message is:\n"
    "- small talk with nothing in it to answer ('cai gi vay troi', 'hello', "
    "'yo', 'ok', 'good morning');\n"
    "- banter, or personal chat between the people in the room;\n"
    "- a question aimed at somebody else, or about somebody else by name;\n"
    "- somebody thinking out loud, or talking to themselves."
)

_CLASSIFIER_RULE_QUIET = (
    "Reply 1 ONLY if the message is clearly addressed to the bot: a mention, "
    "its name, a reply to one of its own messages, or a direct request to the "
    "bot ('can you look up X'). Reply 0 for everything else, including "
    "anything that looks like a task but is aimed at somebody else."
)

# Two examples of each answer. The classifier answers one character, so the
# examples are what decide the character - without them it drifts towards
# "0" and the bot goes quiet in a room it should be joining.
_CLASSIFIER_EXAMPLES = (
    "Examples:\n"
    'Message from someone in the chat: "can you look up the Nvidia earnings date?" -> 1\n'
    'Message from someone in the chat: "did you finish the report?" -> 0\n'
    'Message from someone in the chat: "what is the weather in Hanoi right now?" -> 1\n'
    'Message from someone in the chat: "mệt quá đi, hôm nay học nhiều quá" -> 0\n'
    'Message from someone in the chat: "what do you think about the new model?" -> 1\n'
    'Message from someone in the chat: "cai gi vậy trời" -> 0\n'
    'Message from someone in the chat: "hows jade" -> 0\n'
    'Message from someone in the chat: "hello?" -> 0\n'
)


def _classifier_prompt(text: str, *, author_name: str, roster_line: str,
                       bot_name: str, chat_on: bool) -> str:
    rule = _CLASSIFIER_RULE_CHAT if chat_on else _CLASSIFIER_RULE_QUIET
    people = roster_line or author_name
    return (
        f'A bot named "{bot_name}" is in a group chat with: {people}.\n\n'
        f"Message from {author_name}:\n"
        f'"{text[:1500]}"\n\n'
        f"{rule}\n\n"
        + (_CLASSIFIER_EXAMPLES if chat_on else "")
        + "\nAnswer with one character: 1 or 0."
    )


def _parse_verdict(content: str) -> bool | None:
    """Read the classifier's one-character answer.

    True means reply, False means stay quiet, None means the answer was not
    usable - and then the bot stays quiet, because a bot that cannot tell is
    better quiet than wrong.
    """
    for char in (content or "").strip():
        if char in "01":
            return char == "1"
    return None


async def classify_reply(text: str, *, author_name: str, roster_line: str,
                         bot_name: str, chat_on: bool) -> bool | None:
    """Ask the model whether this message is for the bot.

    Returns True (reply), False (stay quiet) or None if the call failed -
    in which case the bot stays quiet, because a bot that cannot tell is
    better quiet than wrong.

    Only NVIDIA's endpoint is used here - Laguna first, then MiniMax, then
    Nemotron. The answer is one character, so there is nothing to be gained
    from another provider, and a failed call should not quietly become a
    different model's opinion.
    """
    try:
        from config import (LAGUNA_MODEL, MINIMAX_MODEL, NVIDIA_MODEL,
                            extra_body_for, nvidia_client)
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
    )

    # Thinking is switched off: the answer is one character, and a model
    # that reasons first would spend the whole token budget before saying
    # "1".
    for model in (LAGUNA_MODEL, MINIMAX_MODEL, NVIDIA_MODEL):
        for tokens in (_CLASSIFIER_TOKENS, 16):
            try:
                resp = await nvidia_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=tokens,
                    extra_body=extra_body_for("Laguna", thinking=False),
                )
                content = (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"presence classifier failed on {model}: {exc}")
                break
            verdict = _parse_verdict(content)
            if verdict is not None:
                return verdict
            # Nothing usable came back; give it one more try with more room.
    return None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

async def decide(message, *, is_dm: bool, shared: bool, bot_user_id=None,
                 bot_names=(), bot_message_ids=None, chat_on: bool = False,
                 roster=None, author_name: str = "", roster_line: str = "",
                 bot_name: str = "the bot") -> Decision:
    """Should the bot answer this message?

    The structural checks are exact and free. Everything else is decided by
    the model reading the room, and the answer is cached so the second look
    (after the channel lock) costs nothing.
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

    if says_bot_name(text, bot_names):
        return Decision(True, "named in the message")

    # Nothing structural. Let the model read the room.
    key = str(getattr(message, "id", "") or "")
    cached = _DECISIONS.get(key) if key else None
    if cached is None:
        cached = await classify_reply(
            text,
            author_name=author_name or "someone",
            roster_line=roster_line,
            bot_name=bot_name,
            chat_on=chat_on,
        )
        if key and cached is not None:
            _cache_put(key, cached)
    if cached is None:
        return Decision(False, "could not tell whether it was for the bot",
                        from_classifier=True)
    if cached:
        return Decision(True, "the model read the room and said it was for the bot",
                        from_classifier=True)
    return Decision(False, "the model read the room and said it was not for the bot",
                    from_classifier=True)


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

        After a restart the tracker starts empty, so a channel that has been
        shared for weeks looks like a private chat until somebody speaks.
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
