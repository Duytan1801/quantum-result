"""The news parser, and the wiring that puts the news in a prompt.

The digest bot posts in a format: a role ping, an emoji, bold subjects,
an italic detail, an explanation, and a "-# Sources:" line. The parser
has to read that format and nothing else, because the news channel is a
normal channel with people in it.

    python tests/test_newsfeed.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

from newsfeed import brief, is_digest, parse_news  # noqa: E402

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")


def check_true(label, cond):
    check(label, bool(cond), True)


# The real format, as the digest bot posts it.
DIGEST = (
    "@Nerds 🎮 **Discord** & **Rockstar Games** have begun cracking down "
    "on servers about __GTA 6 PC unofficial ports__.\n"
    "\n"
    "Servers with unofficial ports have been taken down one by one.\n"
    "\n"
    "-# Sources: [X](<https://x.com/Dexerto/status/2125235809053749693>)"
)

AI_DIGEST = (
    "@Nerds 🎵 **Tidal** removes __AI-generated__ songs entirely.\n"
    "\n"
    "Previously, they've demonetized and labelled 100% AI songs. Starting "
    "December 9, detected songs will be removed completely from the "
    "platform.\n"
    "\n"
    "-# Sources: [X](<https://x.com/TIDAL/status/2125230421952925956>)"
)

print("=" * 70)
print("THE DIGEST IS READ")
print("=" * 70)

item = parse_news(DIGEST)
check_true("the digest parses", item is not None)
check("the headline is the whole first line",
      item["headline"],
      "Discord & Rockstar Games have begun cracking down on servers about "
      "GTA 6 PC unofficial ports.")
check("the subjects are found", item["subjects"], ["Discord", "Rockstar Games"])
check("the detail is found", item["detail"], "GTA 6 PC unofficial ports")
check("the source is found", len(item["sources"]), 1)
check("the source name", item["sources"][0]["name"], "X")
check("the body is the explanation",
      "Servers with unofficial ports" in item["body"], True)
check("the body does not carry the sources line",
      "Sources:" in item["body"], False)
check("it is not an AI story", item["is_ai"], False)

ai_item = parse_news(AI_DIGEST)
check_true("the AI digest parses", ai_item is not None)
check("the AI digest is flagged", ai_item["is_ai"], True)
check("the AI headline", ai_item["headline"],
      "Tidal removes AI-generated songs entirely.")

print()
print("=" * 70)
print("THE OTHER THINGS IN THE CHANNEL ARE NOT NEWS")
print("=" * 70)

check("an empty message is not news", parse_news(""), None)
check("a person talking is not news",
      parse_news("e còn nguyên MC trên đt"), None)
check("a link on its own is not news",
      parse_news("https://x.com/tetsu0722/status/2125109943284138313"), None)
check("a quote of a digest is not a digest",
      parse_news("> **Tidal** removes songs\n-# Sources: [X](<https://x.com/a>)"),
      None)
check_true("is_digest agrees with parse_news", is_digest(DIGEST))
check_true("is_digest says no to a person", not is_digest("hello"))

print()
print("=" * 70)
print("THE BRIEF")
print("=" * 70)

text = brief(item)
check_true("the brief has the headline",
           text.startswith("Discord & Rockstar Games"))
check_true("the brief has no markdown", "**" not in text)
check_true("the brief has no underscore italics", "__" not in text)
check_true("the brief is not empty", len(text) > 30)

print()
print("=" * 70)
print("THE WIRING")
print("=" * 70)

from config import NEWS_CHANNEL_ID, NEWS_MAX_AGE_HOURS  # noqa: E402

check("the news is read from #general", NEWS_CHANNEL_ID, "1568478571872133140")
check_true("the news has an age limit", NEWS_MAX_AGE_HOURS > 0)

import prompts  # noqa: E402
import proactive  # noqa: E402

prompt = prompts.build_proactive_prompt(
    {}, shared=True, roster=None, bot_name="Lain",
    bot_aliases=("lain",), voice_lines=(), kind="join",
    target="a thought about the Tidal news", news=[ai_item])
check_true("the prompt says the news is not the point",
           "not here to report anything" in prompt)
check_true("the prompt says not to state facts",
           "as a fact you verified" in prompt)

# The news itself is knowledge, and in Neuro-sama's shape the knowledge
# goes in the turn - the things she knows, then the room, then her name.
turn = prompts.build_proactive_turn(bot_name="Lain", news=[ai_item])
check_true("the news is in the proactive turn", "Tidal removes" in turn)
check_true("the turn says she knows it", "knows these things" in turn)

context = proactive._context_text(channel_name="model", people="Jade",
                                  lines=[("Jade", "e chơi MC đi")],
                                  now=0, news=[ai_item])
check_true("the news is in the moment context", "Tidal removes" in context)

# The reply path: the news is knowledge for an ordinary answer too.
reply_prompt = prompts.build_system_prompt({}, "", shared=False,
                                           bot_name="Lain", news=[ai_item])
check_true("the news is in the reply prompt", "Tidal removes" in reply_prompt)
check_true("the reply prompt says not to bring it up",
           "Do not bring the news up on your own" in reply_prompt)
check_true("the reply prompt says the feed beats a search",
           "answer from this" in reply_prompt)

from newsfeed import is_news_question  # noqa: E402

check_true("a news question is recognised", is_news_question("Hows the news"))
check_true("a Vietnamese news question is recognised",
           is_news_question("có tin gì mới hôm nay"))
check_true("a normal question is not a news question",
           not is_news_question("what time is it"))

# A question the feed covers must not be forced through a web search.
import inspect  # noqa: E402
from agent import get_nvidia_response_full  # noqa: E402

sig = inspect.signature(get_nvidia_response_full)
check_true("the reply takes the news", "news" in sig.parameters)

src = inspect.getsource(get_nvidia_response_full)
check_true("the freshness check lets the feed answer",
           "and not digest_covers" in src)

print()
print("=" * 70)
print(f"{PASSED} passed, {len(FAILED)} failed")
print("=" * 70)
if FAILED:
    for name in FAILED:
        print(f"  FAILED: {name}")
    sys.exit(1)
