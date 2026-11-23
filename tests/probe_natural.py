"""Does the bot sound like the room it is in?

This drives the real answer path - the same prompt, the same memory, the
same model chain - with the messages Jade actually sent, and checks the
things that made the old replies read like a helpdesk:

  * "Bạn" and "Tôi" in a room where nobody says either
  * "anh" or "chị" used for Jade, who is "em"
  * a reply that runs on for paragraphs
  * an opener like "Absolutely" / "Chắc chắn rồi"
  * the bot's own name missing when somebody asks what it is called

The voice anchor is fed with real lines from the room, because that is
what the anchor is for.
"""

import asyncio
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


import agent  # noqa: E402
from cache import CACHES  # noqa: E402
from storage import db_add_message  # noqa: E402

JADE = {"id": "909453181916938301", "name": "\U0001d4a7\U0001d4a0\U0001d4a7\U0001d4e8_\U0001d4ec", "plain": "jade_m8"}
DAMIAN = {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}
# A fresh scope every run, so the room's history is the three seed lines
# and not the leftovers of the last run.
SCOPE = f"channel:natural-test-{int(time.time())}"

# Real lines from #model, so the style anchor has the room's voice in it.
ROOM = [
    "Kiểu mình nhường tí í",
    "Mẹ a đập nhiều thôi chứ học hành là trên hết nên gặp gđ bucky của e a thua quá",
    "dm a coi cái văn",
]

CASES = [
    "mệt quá đi, hôm nay học nhiều quá",
    "i mean văn xin lỗi ba má",
    "xin loi sao cho toi duoc di hoc va sua duoc dien thoai i",
    "what's your name?",
]

# Words that mean the reply came out in the wrong voice.
FORMAL_VI = ("bạn ", "tôi ", "chắc chắn rồi", "được rồi", "rất vui được")
WRONG_KINSHIP = ("anh ", "chị ")
RIGHT_KINSHIP = ("em ", "anh ")


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
    return ok


def fold(text):
    """The message without its accents, so the comparison is fair."""
    stripped = unicodedata.normalize("NFD", text)
    return "".join(c for c in stripped if not unicodedata.combining(c)).lower()


async def main():
    passed = 0
    total = 0
    for index, message in enumerate(CASES):
        # A fresh conversation for each message, so one reply does not
        # become the context for the next one.
        scope = f"{SCOPE}-{index}"
        for line in ROOM:
            await db_add_message(scope, "user", line)

        CACHES.clear("answer")
        result = await agent.get_nvidia_response_full(
            JADE["id"], message,
            scopes=[JADE["id"], scope],
            write_scope=scope, shared=True,
            speaker=JADE, roster=[DAMIAN, JADE],
            shape="question",
            bot_name="Lain", bot_aliases=("lain2", "lainy"),
        )
        answer = result["answer"]
        low = answer.lower()
        print(f"\n  > {message!r}   [model: {result.get('model')}]")
        print(f"    {answer}")
        print()

        if message == "what's your name?":
            total += 1
            passed += check("it says its name is Lain",
                            "lain" in low, answer[:80])
            continue

        total += 1
        passed += check("no formal 'Bạn'/'Tôi'",
                        not any(w in low for w in FORMAL_VI), answer[:80])
        total += 1
        passed += check("it does not repeat the message back",
                        not fold(answer).startswith(fold(message)[:18]),
                        answer[:80])
        total += 1
        passed += check("no confirmation-word opener",
                        not low.startswith(("absolutely", "certainly", "sure",
                                            "yes,", "okay,", "of course",
                                            "chắc chắn", "được rồi")))
        total += 1
        bad_term = [w for w in WRONG_KINSHIP if w in low]
        passed += check("no wrong kinship term for Jade",
                        not bad_term,
                        f"found {bad_term} in {answer[:90]!r}")
        total += 1
        passed += check("it is a chat reply, not an essay",
                        len(answer) < 400, f"{len(answer)} chars")
        total += 1
        passed += check("no bot-speak",
                        "let me know if you need" not in low
                        and "i hope this helps" not in low)

    print(f"\n{passed}/{total} checks passed")


asyncio.run(main())
