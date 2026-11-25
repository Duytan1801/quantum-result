"""The news in #general, and what Lain could do with it.

#general is fed by a bot that posts a news digest: a role ping, an
emoji, a bold headline, an italic detail, a sentence or two of
explanation, sometimes bullets, and a "-# Sources:" line with links.
It is the room's feed of what happened in the world.

That is a very good thing for a proactive bot to have, and a very easy
way to go wrong. Two failure modes, and they are opposite:

  * **Reciting.** The bot posts the headline back. "Did you see that
    Nvidia thing?" is worth nothing - the person who wants the news
    reads the news, and the person who does not want it is annoyed.
  * **The feed as a monologue.** The bot treats #general as its own
    channel and starts a discussion nobody asked for.

What the research says about this:

  - "Towards Human-centered Proactive Conversational Agents" (arXiv
    2404.12670): a proactive system should exercise restraint rather
    than assertiveness, and it should be guided by human needs rather
    than by its own goals. So the news is a *reason* to speak, never a
    *reason to perform*.

  - Jarvis (arXiv 2610.07506) speaks only when there is a specific,
    verifiable gap in what the group already knows, and it stays silent
    97% of the time. So the trigger here is a fact the room has not
    mentioned, and the silence rate has to stay high.

  - The survey on proactive dialogue (arXiv 2305.02750) notes that
    proactivity raises the factuality risk, because the system
    contributes more of its own information. So a remark built on the
    news must carry its source, and must not claim to have verified
    anything.

So the shape of it:

  1. Read the news posts from #general - the digest format, so they are
     easy to recognise and hard to confuse with somebody talking.
  2. Pick at most one, and only one that is actually about something
     the room cares about - the people in #model, or the topics they
     talk about.
  3. Say ONE line about it, in the room's voice, with a question or a
     thought rather than a summary.
  4. Never in #general itself. The news belongs to the channel that
     posted it.

The alternative - and this is the cheaper and better one - is not to
post the news at all, but to *have* it. The knowledge changes what Lain
can say: she can say "à cái đó hôm qua có tin này nè" about something
she has actually read, rather than inventing something to say. A
person in a room knows what happened that day. That is most of what
"having a subconscious" means.
"""

import re

# The digest format, as it is actually written:
#
#   @Nerds 🎮 **Discord** & **Rockstar Games** have begun cracking down
#   on servers about __GTA 6 PC unofficial ports__.
#
#   - the bold parts are who it is about
#   - the italic part is the thing that happened
#   - a "-# Sources:" line with the links comes at the end
#
# So the first line is the whole claim, and it has to be read whole: the
# bold bits are only the subject of the sentence, not the headline.
_BOLD = re.compile(r"\*\*([^*]{2,80})\*\*")
_ITALIC = re.compile(r"__([^_]{2,200})__")
_SOURCES = re.compile(r"-#\s*Sources?:", re.IGNORECASE)
# A source line, so the post can be traced back.
_LINK = re.compile(r"\[([^\]]{1,60})\]\(<?([^)>]{5,300})>?\)")
# The role ping at the start.
_PING = re.compile(r"^@\w+")
# A leading emoji, which the digest bot puts on every post.
_EMOJI_HEAD = re.compile(r"^\s*(@\w+\s*)?"
                         r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF"
                         r"\U0001F1E6-\U0001F1FF]️?\s*")
# The tag that marks a post as AI-related.
_AI_TAG = re.compile(r"\bAI\b", re.IGNORECASE)


def parse_news(text: str) -> dict | None:
    """Read a news digest post. Returns None if it is not one.

    The digest format is specific enough to recognise: bold subjects, an
    italic detail, and a sources line. A person talking in the channel
    does not write like that, and the bot's own replies do not either -
    so this is a cheap and quite reliable filter.
    """
    if not text:
        return None
    if not _SOURCES.search(text):
        return None

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # A quote is somebody talking about the digest, not the digest.
    if lines[0].lstrip().startswith(">"):
        return None

    # The headline is the first line with the role ping and the emoji
    # taken off. It is the whole claim, so it is read whole.
    headline = lines[0].strip()
    headline = _EMOJI_HEAD.sub("", headline)
    headline = headline.replace("**", "").replace("__", "").strip()
    if len(headline) < 10:
        return None

    # Who it is about, and what happened. There is usually one subject
    # and one detail, but not always.
    subjects = [m.strip() for m in _BOLD.findall(text)]
    details = [m.strip() for m in _ITALIC.findall(text)]

    sources = []
    for name, url in _LINK.findall(text):
        if "discord.com" in url or "tiktok.com" in url:
            continue
        sources.append({"name": name.strip(), "url": url.strip()})

    # The explanation, which is the part that actually says what
    # happened, rather than the headline's compressed version.
    body = "\n".join(lines[1:])
    body = re.sub(r"^-#.*$", "", body, flags=re.MULTILINE).strip()

    return {
        "headline": headline[:300],
        "subjects": subjects[:4],
        "detail": " - ".join(details)[:300],
        "body": body[:600],
        "sources": sources,
        "is_ai": bool(_AI_TAG.search(text)),
    }


def is_digest(text: str) -> bool:
    """Is this a news digest post rather than somebody talking?"""
    return parse_news(text) is not None


# A question about the news, in the two languages the room uses. The
# point of this is not to catch every question that mentions the news -
# it is to know when the room's own feed is a better answer than a web
# search, because it is fresher and about the things this room reads.
_NEWS_QUESTION_EN = re.compile(
    r"(news|headlines?|what'?s going on|what'?s happening|"
    r"anything new|anything happening|what is new|what's new)",
    re.IGNORECASE)
_NEWS_QUESTION_VI = re.compile(
    r"(tin tức|tin mới|có gì mới|có gì hot|hôm nay có gì|"
    r"có gì hay|update gì|có tin gì)",
    re.IGNORECASE)


def is_news_question(text: str) -> bool:
    """Is this a question about what is going on?"""
    t = (text or "").strip()
    if not t:
        return False
    return bool(_NEWS_QUESTION_EN.search(t) or _NEWS_QUESTION_VI.search(t))


def brief(item: dict, *, body_chars: int = 200) -> str:
    """One digest post, as a short piece of text for a prompt.

    The headline is the compressed claim; the body is the explanation
    that says what actually happened. Both go in, because a headline
    on its own is thin - "Nvidia has reinstated Jensen Huang as CEO" is
    something to have an opinion about, but it is not enough to build
    a remark on without knowing what happened.
    """
    headline = (item.get("headline") or "").strip()
    detail = (item.get("detail") or "").strip()
    body = re.sub(r"\s+", " ", (item.get("body") or "").strip())
    # The digest is written with markdown, and a prompt full of asterisks
    # is a prompt full of noise.
    body = body.replace("**", "").replace("__", "").replace("#", "")
    body = re.sub(r"^-\s*", "", body).strip()

    parts = [headline]
    # The detail is usually a piece of the headline already, so it only
    # adds anything when it is new information.
    if detail and detail.lower() not in headline.lower():
        parts.append(detail)
    if body:
        parts.append(body[:body_chars])
    return " | ".join(p for p in parts if p)
