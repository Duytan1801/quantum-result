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
import time

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

    # The same remark twice is not a remark.
    said = [("the bot", "cơ chế salt ổn thật."), ("Damian", "ok")]
    check_true("the same line is a repeat",
               proactive.repeats("cơ chế salt ổn thật.", said))
    check_true("the same thought in other words is a repeat",
               proactive.repeats("cơ chế salt ổn thật đó", said))
    check_true("a different line is not a repeat",
               not proactive.repeats("tidal xóa hết bài ai rồi", said))

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
    print("THE TICK")
    print("=" * 70)

    # The bug: after a deploy, the in-memory map of rooms is empty until
    # somebody says something, and the tick skipped every channel that was
    # not in it. So a room where nobody had spoken since the deploy - the
    # room the bot exists for - was never even looked at.
    import storage

    spoke = {}

    class FakeChannel:
        id = 1569657875368902666
        name = "model"

    class FakeBot:
        def get_channel(self, cid):
            return FakeChannel()

        class user:
            id = "1"
            name = "lain2"

    async def fake_chat_channels():
        return ["1569657875368902666"]

    async def fake_is_shared(cid, channel=None):
        return True

    async def fake_last_human(channel):
        return time.time() - 600

    async def fake_news(limit=8, force=False):
        return []

    async def fake_recent(channel, limit=12, exclude_id=None):
        return [("Damian", "hello")]

    async def fake_roster(cid, channel=None, force=False):
        return []

    async def fake_speak_once(**kwargs):
        spoke["line"] = kwargs.get("channel_id")
        return "a line"

    import main as main_module

    old = (main_module.bot, main_module._channel_is_shared,
           main_module._last_human_at, main_module._read_news,
           main_module._recent_messages, main_module._channel_roster,
           main_module.bot_canonical_name, main_module.bot_name_variants,
           storage.db_list_chat_channels, proactive.speak_once)
    main_module.bot = FakeBot()
    main_module._channel_is_shared = fake_is_shared
    main_module._last_human_at = fake_last_human
    main_module._read_news = fake_news
    main_module._recent_messages = fake_recent
    main_module._channel_roster = fake_roster
    main_module.bot_canonical_name = lambda user: "Lain"
    main_module.bot_name_variants = lambda user: ("lain", "lain2")
    storage.db_list_chat_channels = fake_chat_channels
    proactive.speak_once = fake_speak_once
    main_module.SHARED_CHANNELS.clear()
    try:
        await main_module._tick_once()
    finally:
        (main_module.bot, main_module._channel_is_shared,
         main_module._last_human_at, main_module._read_news,
         main_module._recent_messages, main_module._channel_roster,
         main_module.bot_canonical_name, main_module.bot_name_variants,
         storage.db_list_chat_channels, proactive.speak_once) = old

    check_true("the tick speaks in a room it has not seen since the deploy",
               spoke.get("line") == "1569657875368902666")

    # And a room that is not a room - one person, no history - is still
    # left alone.
    spoke.clear()
    main_module.SHARED_CHANNELS.clear()

    async def not_shared(cid, channel=None):
        return False

    old_shared = main_module._channel_is_shared
    old_speak = proactive.speak_once
    main_module._channel_is_shared = not_shared
    proactive.speak_once = fake_speak_once
    try:
        await main_module._tick_once()
    finally:
        main_module._channel_is_shared = old_shared
        proactive.speak_once = old_speak

    check_true("a private chat is left alone", not spoke)

    # The switch: a room where proactive talking has been turned off is
    # left alone, whatever the rest of the state says.
    spoke.clear()

    async def switched_off(cid, default=True):
        return False

    old_enabled = main_module.db_is_proactive_enabled
    main_module.db_is_proactive_enabled = switched_off
    try:
        await main_module._tick_once()
    finally:
        main_module.db_is_proactive_enabled = old_enabled

    check_true("a room with proactive talking off is left alone", not spoke)

    # The tick has to actually start. It used to be started by a line
    # inside on_ready that imported its settings from the wrong module,
    # and the ImportError killed on_ready at the end - so the bot logged
    # in, answered messages, and never once looked for a moment to speak.
    import main as main_module

    old_task = main_module.TICK_TASK
    main_module.TICK_TASK = None
    try:
        main_module._start_tick()
        started = main_module.TICK_TASK
        check_true("the tick starts", started is not None and not started.done())
        if started is not None:
            started.cancel()
    finally:
        main_module.TICK_TASK = old_task

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(main())
