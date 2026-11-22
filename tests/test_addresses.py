"""Does the new name-addressing test get the real cases right?

These are the messages from the log that contain the bot's name, and what
each of them actually is.

The check the bot really makes excludes the names that belong to somebody
else in the room - "lain" is the bot's short name and also Damian's name
- so that is modelled here too.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

from people import (addresses_bot, name_collisions, names_in,  # noqa: E402
                    says_bot_name)

NAMES = ("lain2", "lain", "lainy")
ROSTER = [
    {"id": "1", "name": "Lain", "plain": "Lain"},
    {"id": "2", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"},
]

COLLISIONS = name_collisions(NAMES, ROSTER)
STRICT = tuple(n for n in NAMES if n not in COLLISIONS)

CASES = [
    # (message, is it addressed to the bot?)
    ("Sup lain", True),
    ("Yo lain she went to sleep", True),
    ("U hear me, lain?", False),   # not in the first two words
    ("Lain", False),                # Damian telling Jade his name
    ("E yeu lain2 hon ca toi", False),
    ("E có yêu lain2 hơn cả tỏi đâu", False),
    ("Th e thich lain nay duoc chx", False),
    ("Mốt chắc lain2 cướp luôn e từ lain", False),
    ("Mà thấy sao nó ko dc tự nhiên cho lắm", False),
    ("=))) ulatr tổng tài qus", False),
    ("Lainy lainy odyssey, whos the wisest of all prey", True),
    ("ULATR LAIN ĐÂU R", True),
]

passed = 0
for text, want in CASES:
    got = addresses_bot(text, NAMES, exclude=COLLISIONS)
    ok = got == want
    passed += ok
    print(f"  {'ok  ' if ok else 'FAIL'} addressed={got!s:5s} want={want!s:5s} "
          f"mentions_name={says_bot_name(text, NAMES)!s:5s} {text!r}")

print(f"\ncollisions: {COLLISIONS}")
print(f"strict names: {STRICT}")

# The collision has to be found, and only for the names that really clash.
assert COLLISIONS == {"lain": "Lain"}, COLLISIONS
assert STRICT == ("lain2", "lainy"), STRICT

# The names that are used, not just whether one of them is used.
assert names_in("Lainy lainy odyssey", NAMES) == ("lainy",)
assert names_in("nothing here", NAMES) == ()

print(f"\n{passed}/{len(CASES)} passed")
sys.exit(0 if passed == len(CASES) else 1)
