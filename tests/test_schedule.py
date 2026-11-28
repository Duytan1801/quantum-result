"""Scheduling a message.

The schedule is small - a time, some days, a person, a message - but
the parts around it are where the mistakes live:

  - the times are Vietnam's, and they stay Vietnam's whatever the
    machine's timezone is;
  - the message goes once, not once every twenty seconds for the five
    minutes of the grace window;
  - a name is not a ping: the message is sent to the person as it was
    written, and only that person may be pinged by it;
  - a schedule survives a restart, because it lives in the database.

Nothing here touches the network: the storage is stubbed out, and the
Discord objects are fakes. The mentions - what the bot may ping and
what it may never ping - are in test_mentions.py.

    python tests/test_schedule.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import main as main_module  # noqa: E402
import people  # noqa: E402
import schedule as schedule_module  # noqa: E402
import storage  # noqa: E402

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


CHANNEL_ID = "1569657875368902666"
JADE_ID = "909453181916938301"
DAMIAN_ID = "1499211380052721684"

JADE = {"id": JADE_ID, "name": "\U0001d4a7\U0001d4a0\U0001d4a7\U0001d4e8_\U0001d4ec",
        "plain": "jade_m8"}
DAMIAN = {"id": DAMIAN_ID, "name": "Lain", "plain": "Lain"}


class FakeChannel:
    def __init__(self, channel_id=CHANNEL_ID, name="model"):
        self.id = int(channel_id)
        self.name = name
        self.sent = []

    async def send(self, text, **kwargs):
        self.sent.append((text, kwargs))
        return text


class FakeGuild:
    def __init__(self, channels):
        self.text_channels = channels


class FakeCtx:
    def __init__(self, channels=None):
        self.channel = FakeChannel()
        self.author = FakeUser(DAMIAN_ID, "Damian")
        self.guild = FakeGuild(channels or [FakeChannel()])
        self.sent = []

    async def send(self, text, **kwargs):
        self.sent.append(text)


class FakeUser:
    def __init__(self, user_id, name):
        self.id = int(user_id)
        self.name = name
        self.bot = False


# ---------------------------------------------------------------------------
# The times
# ---------------------------------------------------------------------------


def test_times():
    print()
    print("=" * 70)
    print("THE TIME")
    print("=" * 70)

    check("8:00 is eight in the morning", schedule_module.parse_time("8:00"), (8, 0))
    check("08h00 is eight in the morning", schedule_module.parse_time("08h00"), (8, 0))
    check("a bare hour is a 24-hour one", schedule_module.parse_time("20"), (20, 0))
    check("20:30 is half past eight", schedule_module.parse_time("20:30"), (20, 30))
    check("8pm is eight in the evening", schedule_module.parse_time("8pm"), (20, 0))
    check("8am is eight in the morning", schedule_module.parse_time("8am"), (8, 0))
    check("12am is midnight", schedule_module.parse_time("12am"), (0, 0))
    check("12pm is noon", schedule_module.parse_time("12pm"), (12, 0))
    check("at 7:15 is read", schedule_module.parse_time("at 7:15"), (7, 15))

    for bad in ("25:00", "8:75", "13pm", "half past eight", ""):
        try:
            schedule_module.parse_time(bad)
            check_true(f"'{bad}' is refused", False)
        except schedule_module.ScheduleError:
            check_true(f"'{bad}' is refused", True)


def test_days():
    print()
    print("=" * 70)
    print("THE DAYS")
    print("=" * 70)

    every = (0, 1, 2, 3, 4, 5, 6)
    weekdays = (0, 1, 2, 3, 4)
    weekends = (5, 6)

    check("daily is every day", schedule_module.parse_days("daily"), every)
    check("every day is every day", schedule_module.parse_days("every day"), every)
    check("weekdays are the working days", schedule_module.parse_days("weekdays"),
          weekdays)
    check("weekends are the other two", schedule_module.parse_days("weekends"),
          weekends)
    check("monday is day zero", schedule_module.parse_days("monday"), (0,))
    check("a list is read", schedule_module.parse_days("mon,wed,fri"), (0, 2, 4))
    check("a list with spaces is read",
          schedule_module.parse_days("mon wed fri"), (0, 2, 4))
    check("mondays is monday", schedule_module.parse_days("mondays"), (0,))
    check("the days come out in order",
          schedule_module.parse_days("fri,mon"), (0, 4))
    check("a day is not repeated",
          schedule_module.parse_days("mon,mon,tue"), (0, 1))

    for bad in ("funday", "", "someday"):
        try:
            schedule_module.parse_days(bad)
            check_true(f"'{bad}' is refused", False)
        except schedule_module.ScheduleError:
            check_true(f"'{bad}' is refused", True)


def test_parse():
    print()
    print("=" * 70)
    print("THE COMMAND")
    print("=" * 70)

    parsed = schedule_module.parse("8:00 daily to jade: good morning")
    check("the hour is read", parsed["hour"], 8)
    check("the minute is read", parsed["minute"], 0)
    check("the days are read", parsed["days"], list(range(7)))
    check("the person is read", parsed["to"], "jade")
    check("the message is read", parsed["message"], "good morning")
    check("it goes privately by default", parsed["where"], "dm")

    parsed = schedule_module.parse("20:00 weekdays to 909453181916938301: standup")
    check("an id is a person", parsed["to"], "909453181916938301")
    check("the message keeps its colons",
          schedule_module.parse("8:00 daily to jade: note: bring the thing")["message"],
          "note: bring the thing")

    parsed = schedule_module.parse("9:30 mon,wed,fri to jade: standup in #model")
    check("a room is read", parsed["where"], "channel")
    check("the room's name is read", parsed["channel"], "model")
    check("the room is taken out of the message", parsed["message"], "standup")

    parsed = schedule_module.parse("9:30 mon to jade: check in 5 minutes")
    check("a number is not a room", parsed["where"], "dm")
    check("the message is whole", parsed["message"], "check in 5 minutes")

    for bad in ("", "8:00 daily jade hello", "8:00 daily to jade",
                "8:00 daily to jade:", "daily to jade: hello"):
        try:
            schedule_module.parse(bad)
            check_true(f"'{bad}' is refused", False)
        except schedule_module.ScheduleError:
            check_true(f"'{bad}' is refused", True)


def test_due():
    print()
    print("=" * 70)
    print("WHEN IT GOES OUT")
    print("=" * 70)

    schedule = {"hour": 8, "minute": 0, "days": list(range(7))}

    # A Monday.
    at_eight = datetime(2026, 11, 30, 8, 0, tzinfo=schedule_module.VIETNAM)
    check("it is due at the time", schedule_module.is_due(schedule, at_eight), True)
    check("it is due a minute later", schedule_module.is_due(
        schedule, at_eight + timedelta(minutes=1)), True)
    check("it is due inside the grace window", schedule_module.is_due(
        schedule, at_eight + timedelta(minutes=5)), True)
    check("it is not due after the grace window", schedule_module.is_due(
        schedule, at_eight + timedelta(minutes=6)), False)
    check("it is not due before the time", schedule_module.is_due(
        schedule, at_eight - timedelta(minutes=1)), False)

    # The mark that says it has already gone out. It has to be the same
    # every time within the day, or the grace window sends it again.
    stamp = schedule_module.day_stamp(
        schedule_module.target_time(schedule, at_eight + timedelta(minutes=3)))
    check("the mark is the same across the grace window", stamp, "2026-11-30 08:00")
    check("the mark changes with the day", schedule_module.day_stamp(
        schedule_module.target_time(
            schedule, at_eight + timedelta(days=1))), "2026-12-01 08:00")

    # The days it is for.
    only_monday = {"hour": 8, "minute": 0, "days": [0]}
    check("it is due on the day it is for",
          schedule_module.is_due(only_monday, at_eight), True)
    check("it is not due on another day", schedule_module.is_due(
        only_monday, at_eight + timedelta(days=1)), False)

    # Vietnam is UTC+7, whatever the machine thinks.
    check("Vietnam is seven hours ahead of UTC",
          schedule_module.VIETNAM.utcoffset(None), timedelta(hours=7))


def test_describe():
    print()
    print("=" * 70)
    print("HOW IT IS SHOWN")
    print("=" * 70)

    check("the time is shown as it is written",
          schedule_module.format_time({"hour": 8, "minute": 5}), "08:05")
    check("every day is shown as every day",
          schedule_module.format_days({"days": list(range(7))}), "every day")
    check("the working days are shown as weekdays",
          schedule_module.format_days({"days": [0, 1, 2, 3, 4]}), "weekdays")
    check("a list of days is shown by name",
          schedule_module.format_days({"days": [0, 2, 4]}), "Mon, Wed, Fri")

    line = schedule_module.describe({
        "hour": 8, "minute": 0, "days": list(range(7)),
        "to_name": "jade_m8", "message": "good morning", "where": "dm"})
    check_true("the line has the time", "08:00" in line)
    check_true("the line has the person", "jade_m8" in line)
    check_true("the line has the message", "good morning" in line)
    check_true("the line says it is private", "privately" in line)



async def test_command():
    print()
    print("=" * 70)
    print("THE COMMAND, END TO END")
    print("=" * 70)

    stored = {}
    counter = [0]

    async def fake_set(schedule_id, fields):
        stored.setdefault(schedule_id, {}).update(
            {k: ("" if v is None else str(v)) for k, v in fields.items()})

    async def fake_get(schedule_id):
        return dict(stored.get(schedule_id, {}))

    async def fake_list():
        return [{"id": key, **value} for key, value in stored.items()]

    async def fake_delete(schedule_id):
        stored.pop(schedule_id, None)

    def fake_id():
        counter[0] += 1
        return f"s{counter[0]:03d}"

    old = (storage.db_set_schedule, storage.db_get_schedule,
           storage.db_list_schedules, storage.db_delete_schedule,
           storage._schedule_id)
    storage.db_set_schedule = fake_set
    storage.db_get_schedule = fake_get
    storage.db_list_schedules = fake_list
    storage.db_delete_schedule = fake_delete
    storage._schedule_id = fake_id

    people.SEEN.clear()
    people.remember(JADE)

    try:
        callback = main_module.schedule_cmd.callback


        # With nothing to say it explains itself.
        ctx = FakeCtx()
        await callback(ctx, argument="")
        check_true("no argument shows the help", "schedule" in ctx.sent[0])
        check_true("the help shows an example", "!schedule 8:00 daily" in ctx.sent[0])

        # Setting one stores it, for the right person.
        ctx = FakeCtx()
        await callback(ctx, argument="8:00 daily to jade: good morning")
        check("the schedule is stored", len(stored), 1)
        row = list(stored.values())[0]
        check("the hour is stored", row["hour"], "8")
        check("the days are stored", row["days"], "0,1,2,3,4,5,6")
        check("the person is stored", row["to"], JADE_ID)
        check("the message is stored", row["message"], "good morning")
        check("it goes privately", row["where"], "dm")
        check("the owner is stored", row["owner"], DAMIAN_ID)
        check_true("it says it is set", "good morning" not in ctx.sent[0]
                   and "Set" in ctx.sent[0])

        # The listing shows it.
        ctx = FakeCtx()
        await callback(ctx, argument="list")
        check_true("the listing shows the schedule",
                   "08:00" in ctx.sent[0] and "jade_m8" in ctx.sent[0])

        # A second one is a second one.
        ctx = FakeCtx()
        await callback(ctx, argument="20:00 weekdays to jade: standup")
        check("there are two now", len(stored), 2)

        # Somebody the bot has never heard of is refused, politely.
        ctx = FakeCtx()
        await callback(ctx, argument="8:00 daily to nobody: hello")
        check_true("an unknown person is refused",
                   "do not know who" in ctx.sent[0])
        check("an unknown person is not stored", len(stored), 2)

        # A message that cannot be read is refused with the help.
        ctx = FakeCtx()
        await callback(ctx, argument="half past eight to jade: hello")
        check_true("an unreadable command is refused",
                   "schedule" in ctx.sent[0].lower())

        # Clearing one removes it.
        first_id = list(stored.keys())[0]
        ctx = FakeCtx()
        await callback(ctx, argument=f"clear {first_id}")
        check("the schedule is gone", len(stored), 1)
        check_true("it says what it removed", first_id in ctx.sent[0])

        # The tick: a schedule that is due goes out, and only once.
        sent = []

        async def fake_send(schedule):
            sent.append(schedule)
            return "sent privately"

        old_send = main_module._send_scheduled
        main_module._send_scheduled = fake_send

        try:
            due = {"id": "sdue", "hour": 8, "minute": 0,
                   "days": "0,1,2,3,4,5,6", "to": JADE_ID,
                   "to_name": "jade_m8", "message": "good morning",
                   "where": "dm", "channel": "", "owner": DAMIAN_ID,
                   "set_in": CHANNEL_ID, "fired": ""}

            # Now is the time it is for.
            real_now = schedule_module.now_vietnam
            when = datetime.now(schedule_module.VIETNAM).replace(
                hour=8, minute=0, second=0, microsecond=0)
            schedule_module.now_vietnam = lambda: when
            try:
                stored["sdue"] = dict(due)
                count = await main_module._schedules_once()
                check("a due schedule goes out", count, 1)
                check("it was the right one", len(sent), 1)

                # The tick runs every twenty seconds, and the grace
                # window is five minutes wide. The mark is what stops it
                # going out again.
                count = await main_module._schedules_once()
                check("it does not go out twice", count, 0)
                check("and it is not sent twice", len(sent), 1)

                # A schedule for another time is left alone.
                stored["sdue"]["hour"] = "9"
                stored["sdue"]["fired"] = ""
                count = await main_module._schedules_once()
                check("a schedule that is not due is left alone", count, 0)

                # And one that is not for today is left alone too.
                stored["sdue"]["hour"] = "8"
                stored["sdue"]["days"] = "6"
                stored["sdue"]["fired"] = ""
                count = await main_module._schedules_once()
                check("a schedule for another day is left alone", count, 0)
            finally:
                schedule_module.now_vietnam = real_now
        finally:
            main_module._send_scheduled = old_send

    finally:
        (storage.db_set_schedule, storage.db_get_schedule,
         storage.db_list_schedules, storage.db_delete_schedule,
         storage._schedule_id) = old


async def run_tests():
    test_times()
    test_days()
    test_parse()
    test_due()
    test_describe()
    await test_command()

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(run_tests())
