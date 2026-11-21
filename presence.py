"""Deciding when the bot should speak, and with which context.

Two problems are solved here.

**The bot answered everything.** Chat mode was on in #model, so the bot
replied to every single message - including the ones plainly meant for
somebody else, and including the other people's casual chat. In a channel
with more than one person it now waits to be spoken to.

**Each person had their own database.** In a channel with more than one
person the conversation and the long-term memory are shared, so the bot
knows what everyone in the room has been talking about. The scope key is
what makes that work: a channel with several people gets one scope, and a
private chat keeps the per-user one.
"""

import re

# The names people use for the bot. A message that uses one of them is
# addressed to the bot even without a real mention.
BOT_NAMES = ("lain", "lain2", "lainbot", "assistant", "bot")

# A message that asks for some work to be done. These verbs mean "do this
# task", which is what separates a request for the bot from two people
# talking to each other ("did you finish the report?" is a question, but it
# is not a task).
TASK_VERBS = (
    "search", "look up", "lookup", "find out", "explain", "summarise",
    "summarize", "translate", "calculate", "compute", "research", "plot",
    "draw", "build", "write", "run", "check", "verify", "analyse", "analyze",
    "parse", "extract", "fetch", "download", "debug", "fix", "code",
)

# Questions. A question in a shared channel is not automatically for the bot -
# people ask each other questions all the time - but a question about one of
# these subjects usually is.
QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which",
                  "is", "are", "do", "does", "can", "could", "should", "will")

# Subjects the bot is the right person for. With chat mode on, a question
# about one of these is worth answering even without a mention.
BOT_TOPICS = (
    "weather", "temperature", "forecast", "price", "stock", "share price",
    "market cap", "exchange rate", "inflation", "cpi", "gdp", "unemployment",
    "interest rate", "news", "headline", "paper", "arxiv", "citation",
    "literature", "code", "error", "traceback", "bug", "exception", "api",
    "python", "javascript", "sql", "regex", "equation", "formula", "theorem",
    "proof", "derivative", "integral", "matrix", "eigenvalue", "probability",
    "kaggle", "dataset", "model", "pytorch", "tensorflow", "hugging face",
    "covid", "crypto", "bitcoin", "gold price", "oil price", "company",
    "revenue", "earnings", "valuation",
    # The channel is bilingual, so the same subjects in Vietnamese.
    "thời tiết", "cổ phiếu", "chứng khoán", "tin tức", "bài báo", "lãi suất",
    "tỷ giá", "lạm phát", "giá vàng", "dự báo", "tăng trưởng",
)

# A request aimed at the listener: "can you look up X", "would you explain Y".
_AIMED_RE = re.compile(
    r"(?<![a-z])(?:can|could|would|will|you\s+please|please)\s+"
    r"(?:you\s+)?(?:help\s+me|" +
    "|".join(re.escape(v) for v in TASK_VERBS) + r")(?![a-z])",
    re.IGNORECASE,
)

# An instruction: "look up X", "explain the proof", "plot the clusters".
_INSTRUCTION_RE = re.compile(
    r"^\s*(?:please\s+|hey\s+|ok\s+|okay\s+)?(?:" +
    "|".join(re.escape(v) for v in TASK_VERBS) + r")(?![a-z])",
    re.IGNORECASE,
)

# An instruction that comes after a clause: "the fourth one, explain it".
_FOLLOWUP_RE = re.compile(
    r"(?:,|\band\b|\bthen\b|\balso\b|\bnow\b|\bpls\b)\s+"
    r"(?:please\s+|can you\s+|could you\s+)?(?:" +
    "|".join(re.escape(v) for v in TASK_VERBS) + r")(?![a-z])",
    re.IGNORECASE,
)

# A question about one of the bot's subjects.
_TOPIC_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in BOT_TOPICS) + r")(?![a-z])",
    re.IGNORECASE,
)


def channel_scope(channel_id) -> str:
    """The key a shared channel's context is stored under."""
    return f"channel:{channel_id}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _mentions_name(text: str, names) -> bool:
    """Does the message use one of the bot's names?"""
    low = _norm(text)
    for name in names:
        # A whole word, so "bot" does not match "robotics" and "lain" does
        # not match "explain".
        if re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])", low):
            return True
    return False


def looks_like_request(text: str) -> bool:
    """Is this a request for some work to be done?

    Deliberately narrow. "Did you finish the report?" is a question but not a
    task, and answering it in a room where two people are talking is exactly
    the noise this is meant to stop.
    """
    low = _norm(text)
    if not low:
        return False
    if _AIMED_RE.search(low):
        return True
    if _INSTRUCTION_RE.match(low):
        return True
    if _FOLLOWUP_RE.search(low):
        return True
    return False


def looks_like_bot_question(text: str) -> bool:
    """Is this a question about something the bot is the right person for?"""
    low = _norm(text)
    if not low or "?" not in low:
        return False
    if not _TOPIC_RE.search(low):
        return False
    stripped = low.rstrip("?!. ")
    first = stripped.split(" ")[0] if stripped else ""
    # "what's" and "how's" are questions too.
    first = re.sub(r"'s$", "", first)
    return first in QUESTION_WORDS or looks_like_request(text)


def addressed_to_bot(text: str, *, names, bot_user_id=None) -> bool:
    """Was this message meant for the bot?"""
    if not text:
        return False
    if bot_user_id:
        if f"<@{bot_user_id}>" in text or f"<@!{bot_user_id}>" in text:
            return True
    if _mentions_name(text, names):
        return True
    return False


def should_answer(message, *, is_dm: bool, participants: set,
                  bot_user_id=None, names=BOT_NAMES,
                  bot_message_ids=None, chat_on: bool = False,
                  shared: bool = None) -> tuple[bool, str]:
    """Should the bot answer this message?

    Returns (answer, reason). The reason is for the log, so it is clear why
    the bot stayed quiet.

    The rule of thumb: in a private conversation the bot always answers. In
    a room with more than one person it answers when it is spoken to, and
    stays out of the way otherwise.

    `shared` overrides the count of participants, for the case where the
    channel is known to be a room but the tracker has only just started.
    """
    text = message.content or ""
    author = message.author

    # The bot never talks to itself, and a message from another bot is not
    # for it.
    if getattr(author, "bot", False):
        return False, "the author is a bot"

    if is_dm:
        return True, "private message"

    if shared is None:
        shared = len(participants or set()) >= 2

    if not shared:
        # One human and the bot. There is nobody else to talk to.
        return True, "one-to-one channel"

    # In a room, a real mention is the clearest sign.
    if bot_user_id and (f"<@{bot_user_id}>" in text or f"<@!{bot_user_id}>" in text):
        return True, "mentioned"

    # A reply to one of the bot's own messages is addressed to the bot.
    ref = getattr(message, "reference", None)
    if ref is not None and getattr(ref, "message_id", None):
        if bot_message_ids and ref.message_id in bot_message_ids:
            return True, "replying to the bot"

    # The bot's name in the text.
    if _mentions_name(text, names):
        return True, "named in the message"

    # A request for work: "can you look up X", "explain the proof", "the
    # fourth one, explain it". This is the bot being useful without being
    # summoned by name.
    if looks_like_request(text):
        return True, "clear request"

    if chat_on:
        # Chat mode is on for this channel, which means the owner wants the
        # bot to join in. It still waits for something that looks like it is
        # addressed to it, but it will speak up about anything it can help
        # with.
        if looks_like_bot_question(text):
            return True, "chat mode, question about a bot subject"

    return False, "not addressed to the bot"


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
