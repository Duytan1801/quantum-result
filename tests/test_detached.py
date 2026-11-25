"""Does the secrets room know the whole, and share nothing?

The requirement has two halves, and they pull in opposite directions:

  * the secrets room reads everything - every channel, every person - so
    it can work with the whole picture;
  * nothing it learns travels. No other room's reply ever loads what was
    said or learned in there.

The second half is the one that is easy to get wrong, because the bot
writes a learned fact to every scope it read. So the test checks the
write list, not just the read list.

It also checks the case that leaks by accident: a private room with only
one person in it. The tracker does not call that a room, so the turn
would be written to the person's own scope - and the person's own scope
is read in every other room. The detached room has to write to its own
scope whether or not anybody else is in it.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


from config import is_detached  # noqa: E402
from presence import ChannelTracker, channel_scope  # noqa: E402
from scopes import scope_plan  # noqa: E402

SECRETS = "1573589377744117881"
MODEL = "1569657875368902666"
DAMIAN = "1499211380052721684"
JADE = "909453181916938301"

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got}\n         want: {want}")


def check_true(label, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label} {detail}")


async def main():
    print("=" * 70)
    print("THE DETACHED ROOM")
    print("=" * 70)

    check_true("#secrets is detached", is_detached(SECRETS))
    check_true("#model is not detached", not is_detached(MODEL))

    # A room with two people in it.
    tracker = ChannelTracker()
    tracker.note(MODEL, DAMIAN)
    tracker.note(MODEL, JADE)

    plan = await scope_plan(SECRETS, DAMIAN, tracker=tracker)
    check_true("the secrets room is a room", plan.shared)
    check_true("the secrets room is detached", plan.detached)
    check_true("the secrets room reads its own scope",
               channel_scope(SECRETS) in plan.read)
    check_true("the secrets room reads the other room's scope",
               channel_scope(MODEL) in plan.read)
    check_true("the secrets room reads the person's own scope",
               DAMIAN in plan.read)
    check_true("the secrets room reads more than one scope",
               len(plan.read) > 2, f"read {plan.read}")

    # The half that matters: what it writes.
    check("the secrets room writes only to itself",
          plan.write, [channel_scope(SECRETS)])
    check_true("the secrets room does not write to the person",
               DAMIAN not in plan.write)
    check_true("the secrets room does not write to the other room",
               channel_scope(MODEL) not in plan.write)
    check("the conversation goes to the secrets scope",
          plan.primary, channel_scope(SECRETS))

    # A private secrets room: one person, so the tracker does not call it
    # a room. The turn still must not go to the person's own scope.
    alone = ChannelTracker()
    alone.note(SECRETS, DAMIAN)
    plan_alone = await scope_plan(SECRETS, DAMIAN, tracker=alone)
    check("a one-person secrets room still writes only to itself",
          plan_alone.write, [channel_scope(SECRETS)])
    check("a one-person secrets room still reads everything",
          plan_alone.shared, True)

    print()
    print("=" * 70)
    print("THE OTHER ROOMS")
    print("=" * 70)

    model = await scope_plan(MODEL, DAMIAN, tracker=tracker)
    check_true("a normal room is not detached", not model.detached)
    check_true("a normal room reads the other room's scope",
               channel_scope(MODEL) in model.read)
    check_true("a normal room does not read the secrets scope",
               channel_scope(SECRETS) not in model.read,
               f"read {model.read}")
    check_true("a normal room does not write the secrets scope",
               channel_scope(SECRETS) not in model.write,
               f"write {model.write}")

    # And the same for every other room the bot knows about.
    for channel in ("1565378867567534100", "1566224438439968818",
                    "1568478571872133140", "1572434981081452595",
                    "1574015015969624204"):
        other = await scope_plan(channel, DAMIAN, tracker=tracker)
        check_true(f"room {channel} does not read the secrets scope",
                   channel_scope(SECRETS) not in other.read)
        check_true(f"room {channel} does not write the secrets scope",
                   channel_scope(SECRETS) not in other.write)

    # A private chat: the person's own scope only.
    private = await scope_plan(MODEL, DAMIAN, tracker=ChannelTracker())
    check("a private chat reads only the person", private.read, [DAMIAN])
    check("a private chat writes only the person", private.write, [DAMIAN])

    print()
    print("=" * 70)
    print("THE STORE")
    print("=" * 70)

    # Every scope the bot has, so the detached room really does read the
    # whole thing and not just the scopes the test made up.
    from storage import db_all_scopes

    everything = await db_all_scopes()
    check_true("the store can be enumerated", len(everything) > 1,
               f"{len(everything)} scopes")
    check_true("the model's scope is in the store",
               channel_scope(MODEL) in everything)
    check_true("the people's scopes are in the store",
               DAMIAN in everything and JADE in everything)

    plan_all = await scope_plan(SECRETS, DAMIAN, tracker=tracker)
    missing = [s for s in everything if s not in plan_all.read]
    check("the secrets room reads every scope there is", missing, [])

    print()
    print("=" * 70)
    print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    print("=" * 70)
    sys.exit(1 if FAILED else 0)


asyncio.run(main())
