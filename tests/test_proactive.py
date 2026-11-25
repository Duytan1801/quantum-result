"""Speaking first, and the decision to answer.

Two things are checked here:

  * the gate. There is no cap on how much the bot talks and no quiet
    window any more - Neuro-sama's number is a minute of silence, and
    after that the small model decides.
  * the answer decision. A message the bot has decided to answer stays
    decided: the second look, made after the person has stopped typing,
    is allowed to turn "wait" into "answer" but not to turn "answer" into
    silence.

    python tests/test_proactive.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import presence  # noqa: E402
import proactive  # noqa: E402
from config import (PROACTIVE_MAX_PER_DAY, PROACTIVE_MIN_GAP_MINUTES,  # noqa: E402
                    PROACTIVE_MIN_IDLE_SECONDS, PROACTIVE_QUIET_HOURS,
                    PROACTIVE_TICK_SECONDS)

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


class FakeAuthor:
    id = 1499211380052721684
    bot = False
    name = "Lain"
    display_name = "Lain"


class FakeMessage:
    def __init__(self, content, message_id="12345"):
        self.content = content
        self.id = message_id
        self.author = FakeAuthor()


async def main():
    print("=" * 70)
    print("THE GATE")
    print("=" * 70)

    check("there is no cap on how much it talks", PROACTIVE_MAX_PER_DAY, 0)
    check("there are no quiet hours", PROACTIVE_QUIET_HOURS, "")
    check("it checks every minute", PROACTIVE_TICK_SECONDS, 60)
    check("a minute of silence is enough", PROACTIVE_MIN_IDLE_SECONDS, 60)
    check("the gap is short", PROACTIVE_MIN_GAP_MINUTES, 2)

    channel = "1569657875368902666"
    now = 1_700_000_000.0

    state = proactive.state_for(channel)
    state.last_human = now - 120          # two minutes ago
    state.last_proactive = 0.0
    state.last_bot = 0.0
    state.ignored = 0
    state.today = 0
    state.day = ""

    ok, why = proactive.gate(channel, now=now)
    check_true(f"a quiet room is a moment to speak ({why})", ok)

    # A room that is busy is not.
    state.last_human = now - 10
    ok, why = proactive.gate(channel, now=now)
    check_true("a busy room is not a moment", not ok)

    # A message that was ignored makes the next one wait longer, but the
    # bot is never capped for good.
    state.last_human = now - 3600
    state.ignored = 3
    ok, why = proactive.gate(channel, now=now)
    check_true(f"the backoff still holds ({why})", ok)

    state.ignored = 0
    state.today = 99
    ok, why = proactive.gate(channel, now=now)
    check_true(f"there is no daily cap ({why})", ok)

    print()
    print("=" * 70)
    print("THE MOMENT DECISION")
    print("=" * 70)

    check("the kind and the thing to say",
          proactive._parse_moment("SPEAK join | a thought about the news")[:2],
          ("join", "a thought about the news"))
    check("it asks what to look up",
          proactive._parse_moment(
              "SPEAK join | a thought about the news | nvidia earnings"),
          ("join", "a thought about the news", "nvidia earnings"))
    check("a dash means nothing to look up",
          proactive._parse_moment("SPEAK join | a thought | -"),
          ("join", "a thought", ""))
    check("quiet is quiet",
          proactive._parse_moment("QUIET the room is busy"), ("", "", ""))

    print()
    print("=" * 70)
    print("THE ANSWER DECISION")
    print("=" * 70)

    # The classifier says "answer" the first time and "quiet" the second
    # time. The second look does not get to take the answer back.
    calls = []

    async def flipping(text, **kwargs):
        calls.append(kwargs.get("final"))
        if kwargs.get("final"):
            return (False, False, "question")
        return (True, False, "question")

    old = presence.classify_reply
    presence.classify_reply = flipping
    presence._DECISIONS.clear()
    try:
        message = FakeMessage("What u thinking bout this lain", "msg-1")
        first = await presence.decide(
            message, is_dm=False, shared=True, bot_user_id="1",
            bot_names=("lain2", "lain"), chat_on=True,
            roster=[{"id": "2", "name": "Lain"}], roster_line="Lain")
        second = await presence.decide(
            message, is_dm=False, shared=True, bot_user_id="1",
            bot_names=("lain2", "lain"), chat_on=True,
            roster=[{"id": "2", "name": "Lain"}], roster_line="Lain",
            final=True)
    finally:
        presence.classify_reply = old
        presence._DECISIONS.clear()

    check_true("the first look says answer", first.reply)
    check_true("the second look does not take it back", second.reply)

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(main())
