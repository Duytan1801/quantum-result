"""The guard: what the bot says on its way out.

Every rule in the prompt is a request, and a request is not a guarantee.
The model still opens with "Great question!", still closes with "Let me know
if you need anything else!", and - after a long personal conversation -
occasionally recites the register rules it was just given, as if the rules
were the answer.

This module is the last thing the reply passes through. It strips the
scaffolding: the bot-speak, the prompt echoes, the sycophancy, and the
"let me look that up" line that should never have been the answer.

Two rules govern how aggressive this is.

**Be conservative.** A reply that has been cut badly is worse than a reply
that keeps one stiff sentence, so every pattern has to be either a whole
line that is clearly scaffolding or a phrase that cannot mean anything else
in context. Nothing here matches on a word alone.

**Never touch the substance.** The numbers, the code, the citations and the
sentences that carry meaning are left exactly as they were. The guard only
removes the wrapper.

The patterns are deliberately conservative about one thing in particular:
"I cannot do X" is a legitimate answer when the thing really cannot be done,
so only the AI-disclaimer shapes ("I'm just an AI", "As a language model")
are removed - never the refusal itself.

Kaiacord does the same thing in its sanitizer: strip the runtime
scaffolding before the message is persisted, so the scaffolding never gets
remembered either.
"""

import re

# ---------------------------------------------------------------------------
# Whole lines that are the prompt talking back
# ---------------------------------------------------------------------------

# The headings and rule lines from the system prompt. If a reply contains
# one of these as a line of its own, the model has started quoting its own
# instructions instead of answering.
_ECHO_LINES = (
    "two registers",
    "how to choose",
    "talking - the default",
    "working - there is a task",
    "when to answer, and when to stay quiet",
    "your tools, for when there is a task",
    "two ways to run code",
    "for academic work",
    "sourcing rules",
    "verification rules",
    "honesty rules",
    "when to delegate",
    "before answering: look at this message only",
    "choose the register",
    "one technical question in the middle",
    "not for the conversation as a whole",
)

# The same rule blocks, but as a bullet or a heading with decoration.
_ECHO_BULLET = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])?\s*"
    r"(?:TALKING|WORKING|talking|working)\s*[-–—:]\s*"
    r".{0,80}$"
)

# ---------------------------------------------------------------------------
# Phrases that are bot-speak wherever they appear
# ---------------------------------------------------------------------------

# The disclaimer family. These are removed even mid-sentence, because
# "I'm just an AI" has no meaning other than "I am about to refuse you".
#
# The Discord-capability statements are NOT in this list: they are removed
# whole, as clauses, further down. Listing them here as well would take
# the phrase out first and leave the rest of the sentence behind.
_AI_DISCLAIMERS = (
    "as an ai language model",
    "as a large language model",
    "as an ai assistant",
    "as an artificial intelligence",
    "as a language model",
    "i am an ai",
    "i'm an ai",
    "i am just an ai",
    "i'm just an ai",
    "i am merely an ai",
    "i'm merely an ai",
    "i am an artificial intelligence",
    "i'm an artificial intelligence",
    "i do not have personal feelings",
    "i don't have personal feelings",
)

# The "let me know if you need anything else" family. These are closers:
# they are stripped from the end of a reply.
_CLOSERS = (
    "i hope this helps",
    "i hope that helps",
    "hope this helps",
    "hope that helps",
    "i hope you find this helpful",
    "hope you find this helpful",
    "let me know if you need anything else",
    "let me know if you need anything",
    "let me know if there's anything else",
    "let me know if there is anything else",
    "feel free to ask me anything",
    "feel free to reach out",
    "if you have any other questions",
    "if you have any further questions",
    "is there anything else i can help with",
    "anything else i can help with",
    "don't hesitate to ask",
    "do not hesitate to ask",
    "happy to help",
    "glad to help",
)

# The "here is the answer" frame. These are openers.
_FRAMES = (
    "here is a helpful response",
    "here's a helpful response",
    "here is my response",
    "here's my response",
    "here is the answer",
    "here's the answer",
    "here is a detailed",
    "here's a detailed",
    "below is a",
    "response:",
    "answer:",
    "output:",
)

# The praise openers. Sycophancy is not warmth; it is a reflex, and it
# makes a reply sound like a form letter.
_SYCOPHANCY = (
    "great question",
    "that's a great question",
    "what a great question",
    "excellent question",
    "that's an excellent question",
    "wonderful question",
    "that's a wonderful question",
    "good question",
    "that's a good question",
    "you're absolutely right",
    "you are absolutely right",
    "that's a completely valid point",
    "that is a completely valid point",
    "that's a very valid point",
    "that's an interesting question",
    "interesting question",
    "great observation",
    "that's a great observation",
)

# The Vietnamese versions of the same few. The bot replies in whatever
# language the room is using, and the scaffolding comes along with it.
_CLOSERS_VI = (
    "hy vọng điều này giúp ích",
    "hy vọng những điều trên hữu ích",
    "rất vui được hỗ trợ",
    "rất vui được giúp đỡ",
    "hãy cho tôi biết nếu bạn cần thêm",
    "nếu bạn cần thêm",
)

_SYCOPHANCY_VI = (
    "câu hỏi hay",
    "đây là một câu hỏi hay",
    "câu hỏi rất hay",
    "bạn hoàn toàn đúng",
)

_AI_DISCLAIMERS_VI = (
    "tôi là một trợ lý ai",
    "mình là một trợ lý ai",
    "tôi chỉ là một ai",
    "là một trợ lý ai",
)

# The Discord-capability family. These are not refusals - they are false
# statements about what the bot can do, and the bot can do them. The whole
# clause goes, rather than the phrase, so that "I can't access Discord
# from here" does not leave "from here" behind.
_CLAUSE_DISCLAIMERS = (
    "i cannot interact with discord",
    "i can't interact with discord",
    "i cannot access discord",
    "i can't access discord",
    "i don't know who you are referring to",
    "i do not know who you are referring to",
    "i cannot interact with mentions",
    "i can't interact with mentions",
)

# A connector left behind at the start of a reply after a clause was
# removed: "..., so I can't help with that" becomes "so I can't help with
# that", which reads as if something before it was still there.
_LEADING_CONNECTOR = re.compile(
    r"^\s*(?:so|but|and|then|therefore|thus|hence|also)\b\s*[,\.]?\s*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# The "let me do that" line
# ---------------------------------------------------------------------------

# A status line the model writes before it acts. In a turn where it really
# did use a tool, the line is noise; in a turn where it did not, the line
# was a promise instead of an answer.
_LEADING_STATUS = re.compile(
    r"^\s*(?:(?:ok(?:ay)?|alright|sure|right|well|now|here)\s*,?\s*)?"
    r"(?:let me|i(?:'ll| will)|i am going to|i'm going to|going to)\b"
    r"[^.!?\n]*[.!?]?\s*",
    re.IGNORECASE,
)


def _strip_ai_disclaimers(text: str) -> str:
    """Remove the "I am just an AI" family."""
    for phrase in _AI_DISCLAIMERS + _AI_DISCLAIMERS_VI:
        # Match the phrase with the word boundaries around it, and eat the
        # comma or dash that usually follows it.
        pattern = re.compile(
            r"(?<![\w])" + re.escape(phrase) + r"(?![\w])\s*[,\-–—:]*\s*",
            re.IGNORECASE,
        )
        text = pattern.sub("", text)

    # The clause family, which goes whole.
    for phrase in _CLAUSE_DISCLAIMERS:
        pattern = re.compile(
            r"[^.!?,\n]*" + re.escape(phrase) + r"[^.!?,\n]*",
            re.IGNORECASE,
        )
        text = pattern.sub("", text)

    # Whatever is left at the front may start with a connector that used to
    # have something in front of it.
    text = _LEADING_CONNECTOR.sub("", text.lstrip())
    return text


def _strip_line_echoes(text: str) -> str:
    """Drop lines that are the system prompt quoting itself back."""
    kept = []
    for line in text.split("\n"):
        bare = line.strip().lower().lstrip("-*• ").strip()
        bare = re.sub(r"^\d+[.)]\s*", "", bare)
        if bare in _ECHO_LINES:
            continue
        if _ECHO_BULLET.match(line) and len(line.strip()) < 200:
            # A bullet that starts with TALKING or WORKING is the register
            # description, not an answer.
            continue
        kept.append(line)
    return "\n".join(kept)


def _strip_closers(text: str) -> str:
    """Remove the "let me know if you need anything else" family."""
    stripped = text.strip()
    for phrase in _CLOSERS + _CLOSERS_VI:
        pattern = re.compile(
            r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
            r"[^.!?\n]*[.!?]?\s*$",
            re.IGNORECASE,
        )
        new = pattern.sub("", stripped).strip()
        if new != stripped:
            stripped = new
    return stripped


def _strip_frames(text: str) -> str:
    """Remove a "here is the answer:" frame at the very start."""
    stripped = text.lstrip()
    for phrase in _FRAMES:
        pattern = re.compile(
            r"^\s*" + re.escape(phrase) + r"\s*:?\s*",
            re.IGNORECASE,
        )
        new = pattern.sub("", stripped, count=1)
        if new != stripped:
            stripped = new.lstrip()
    return stripped


def _strip_sycophancy(text: str) -> str:
    """Remove a praise opener."""
    stripped = text.lstrip()
    for phrase in _SYCOPHANCY + _SYCOPHANCY_VI:
        pattern = re.compile(
            r"^\s*" + re.escape(phrase) + r"\s*[!,.]*\s*",
            re.IGNORECASE,
        )
        new = pattern.sub("", stripped, count=1)
        if new != stripped:
            stripped = new.lstrip()
            break
    return stripped


def _strip_leading_status(text: str) -> str:
    """Remove a "let me look that up" line when there is a real answer after it.

    Only the first line is considered, and only when what follows is
    substantial. If the whole reply is "Let me check that", it is kept: in
    that case it is all the model has to say.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    first = lines[0]
    rest = "\n".join(lines[1:]).strip()
    if len(rest) < 40:
        return text
    match = _LEADING_STATUS.match(first)
    if not match:
        return text
    # The line has to be nothing but the status, not a sentence that happens
    # to start with "let me".
    leftover = first[match.end():].strip()
    if leftover and not leftover.startswith(("and", "but", ",", "-")):
        return text
    return rest


def sanitize_reply(text: str) -> str:
    """Strip the scaffolding out of a reply.

    Returns the cleaned text. A reply that was *entirely* scaffolding comes
    back empty, because there was no answer in it to begin with - and the
    bot would rather say nothing than post the rules back at somebody. The
    caller decides what an empty reply means; the Discord layer does not
    send one.
    """
    if not text:
        return text

    cleaned = text

    cleaned = _strip_ai_disclaimers(cleaned)
    cleaned = _strip_line_echoes(cleaned)
    cleaned = _strip_frames(cleaned)
    cleaned = _strip_sycophancy(cleaned)
    cleaned = _strip_leading_status(cleaned)
    cleaned = _strip_closers(cleaned)

    # Tidy up the seams: blank lines left behind by a removed block, and
    # trailing spaces.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()

    # A leftover piece of punctuation on its own is not a reply either.
    if not cleaned.strip(" .,!?:;-_*`~"):
        return ""
    return cleaned
