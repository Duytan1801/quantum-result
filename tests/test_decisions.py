"""The three decisions, on real messages from the log and on unfinished ones.

A = answer, W = wait, Q = quiet.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


from presence import _name_note, classify_reply  # noqa: E402
from people import name_collisions, names_in  # noqa: E402

BOT_NAMES = ("lain2", "lain", "lainy")
ROSTER = [
    {"id": "1", "name": "Lain", "plain": "Lain"},
    {"id": "2", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"},
]
COLLISIONS = name_collisions(BOT_NAMES, ROSTER)

# (text, expected, reply_to, spoken_seconds_ago)
CASES = [
    # unfinished - the bot should wait for the rest
    ("can you look up the Nvidia", "W", "", None),
    ("i was thinking, like,", "W", "", None),
    ("the thing we talked about yesterday, the one where", "W", "", None),
    # finished and for the bot
    ("can you look up the Nvidia earnings date?", "A", "", None),
    ("Can you check on the plots?", "A", "", None),
    # "Lainy lainy odyssey, whos the wisest of all prey" is not here: it is
    # caught by the structural check, because the bot's name is at the
    # front of it. This test only exercises the model, without the
    # checks around it. test_addresses.py covers the structural path.
    ("U hear me, lain?", "A", "", None),
    ("mệt quá đi, hôm nay học nhiều quá", "A", "", None),
    ("Is there any research shows that a female doctor while on the way "
     "gettin diploma would give birth to a child that carries a risk?",
     "A", "", None),
    # finished and for somebody else
    ("Mốt chắc lain2 cướp luôn e từ lain", "Q", "", None),
    ("A tên j", "Q", "Damian", None),
    ("Em của t", "Q", "Damian", None),
    ("Mà thấy sao nó ko dc tự nhiên cho lắm", "Q", "Jade", None),
    ("Tại minimax có điểm intelligence cao hơn nên a chuyển sang nó", "Q", "Jade", None),
    ("Lain", "Q", "", None),
    # the room belongs to them for now
    ("mệt quá đi", "Q", "", 30),
]


async def main():
    passed = 0
    for text, want, reply_to, spoke in CASES:
        await asyncio.sleep(1.0)
        hits = names_in(text, BOT_NAMES)
        verdict = await classify_reply(
            text,
            author_name="Jade" if reply_to != "Damian" else "Damian",
            roster_line="Lain, Jade",
            bot_name="Lain",
            chat_on=True,
            recent=[("Damian", "A tên j"), ("Jade", "Tot")],
            reply_to=reply_to,
            bot_spoke_seconds=spoke,
            name_note=_name_note(hits, COLLISIONS),
            bot_aliases="lain2, lain, lainy",
        )
        if verdict is None:
            got = "?"
        elif verdict[1]:
            got = "W"
        elif verdict[0]:
            got = "A"
        else:
            got = "Q"
        ok = got == want
        passed += ok
        shape = verdict[2] if verdict else ""
        print(f"  {'ok  ' if ok else 'FAIL'} got={got} want={want} shape={shape:8s} "
              f"{text[:58]!r}")

    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


asyncio.run(main())
