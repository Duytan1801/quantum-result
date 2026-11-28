"""Messages the bot sends at a time Damian chooses.

Damian tells the bot what to send and when to send it, and the bot
sends it. That is the whole feature, and it is smaller than it sounds:
a time, a day or days, a person, and a message.

    !schedule 8:00 daily to jade: good morning, don't be late

The times are Vietnam's. The bot runs on Asia/Ho_Chi_Minh, but the
schedule does not depend on that being set on the machine: the offset
is written out here, because a schedule that shifts by an hour when a
container moves is not a schedule.

The message is sent to the person as it was written. The bot does not
rewrite it, does not add a header, and does not say who it is from -
Damian wrote it, and it is his.

A schedule lives in the facts table under a scope of its own, in the
settings namespace. It is not a memory: it is never read into a prompt,
so a private message does not become something the model knows and can
repeat in another room.
"""

import re
import time
from datetime import datetime, timedelta, timezone

# Vietnam. UTC+7, no daylight saving, ever.
VIETNAM = timezone(timedelta(hours=7))


def now_vietnam() -> datetime:
    """The time it is in Vietnam."""
    return datetime.now(VIETNAM)


# The days of the week, as people write them.
DAYS: dict[str, int] = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}

WEEKDAYS = (0, 1, 2, 3, 4)
WEEKEND = (5, 6)
EVERY_DAY = (0, 1, 2, 3, 4, 5, 6)

# The words for "every day", "every weekday" and so on.
_ALIASES = {
    "daily": EVERY_DAY,
    "everyday": EVERY_DAY,
    "every day": EVERY_DAY,
    "every single day": EVERY_DAY,
    "every day of the week": EVERY_DAY,
    "all week": EVERY_DAY,
    "weekday": WEEKDAYS,
    "weekdays": WEEKDAYS,
    "every weekday": WEEKDAYS,
    "school days": WEEKDAYS,
    "mon-fri": WEEKDAYS,
    "weekend": WEEKEND,
    "weekends": WEEKEND,
    "every weekend": WEEKEND,
}

# The time, at the start of the left-hand side.
_TIME = re.compile(
    r"^(?:at\s+|from\s+)?(?P<hour>\d{1,2})"
    r"(?:[:h](?P<minute>\d{2}))?"
    r"\s*(?P<ampm>am|pm)?\b",
    re.IGNORECASE)


class ScheduleError(Exception):
    """The command could not be read."""


def parse_time(text: str) -> tuple[int, int]:
    """The time in a schedule command, as (hour, minute).

    "8:00", "08h00", "8", "8pm", "20:30" all work. A bare hour is a
    24-hour one - "20" is eight in the evening, the way it is written
    in Vietnam - and "am"/"pm" overrides it.
    """
    match = _TIME.match((text or "").strip())
    if not match:
        raise ScheduleError("I could not read the time.")

    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").lower()

    if ampm:
        if hour < 1 or hour > 12:
            raise ScheduleError("the hour has to be between 1 and 12 with am or pm.")
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    elif hour > 23:
        raise ScheduleError("the hour has to be between 0 and 23.")

    if minute > 59:
        raise ScheduleError("the minutes have to be between 00 and 59.")
    return hour, minute


def parse_days(text: str) -> tuple[int, ...]:
    """The days a schedule is for, as weekday numbers (Monday is 0)."""
    low = " ".join((text or "").strip().lower().split())
    if not low:
        raise ScheduleError("I could not read which days.")

    if low in _ALIASES:
        return _ALIASES[low]

    days: list[int] = []
    for token in re.split(r"[,/\s]+", low):
        token = token.strip(" .")
        if not token or token in ("every", "on", "and", "the", "each"):
            continue
        if token in DAYS:
            days.append(DAYS[token])
            continue
        # "mondays" and "monday," are the same day.
        trimmed = token.rstrip("s")
        if trimmed in DAYS:
            days.append(DAYS[trimmed])
            continue
        raise ScheduleError(f"'{token}' is not a day I know.")

    if not days:
        raise ScheduleError("I could not read which days.")
    return tuple(sorted(set(days)))


def parse(text: str) -> dict:
    """Read a schedule out of a command.

    The shape is:

        <time> <days> to <person>: <message>

    with "at" in front of the time optional, and "in <room>" after the
    message optional. The message is everything after the colon, so it
    can contain colons, commas and anything else.
    """
    raw = (text or "").strip()
    if not raw:
        raise ScheduleError("there was nothing in the command.")

    # The person and the message are on the right of "to".
    parts = re.split(r"\s+to\s+", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        raise ScheduleError(
            "it has to look like: !schedule 8:00 daily to jade: the message")
    left, right = parts[0].strip(), parts[1].strip()

    # The message starts after the first colon in the right-hand side.
    if ":" not in right:
        raise ScheduleError(
            "the message goes after a colon: !schedule 8:00 daily to jade: hello")
    to_part, message = right.split(":", 1)
    to_part = to_part.strip()
    message = message.strip()
    if not to_part:
        raise ScheduleError("I could not tell who the message is for.")
    if not message:
        raise ScheduleError("there was no message to send.")

    # The time and the days are on the left.
    match = _TIME.match(left)
    if not match:
        raise ScheduleError(
            "it has to look like: !schedule 8:00 daily to jade: the message")
    hour, minute = parse_time(left)
    rest = left[match.end():].strip()
    if not rest:
        raise ScheduleError("I could not read which days.")
    days = parse_days(rest)

    # An optional room, for a message that is posted rather than sent
    # privately. The room is either "#name" or a bare word - a bare
    # number is not a room, so "check in 5 minutes" stays a message.
    where = "dm"
    channel = ""
    room = re.search(
        r"\s+in\s+(?:the\s+)?(?:#([\w\-]+)|([a-zA-Z][\w\-]*))\s*$",
        message, re.IGNORECASE)
    if room:
        where = "channel"
        channel = room.group(1) or room.group(2)
        message = message[:room.start()].strip()

    return {
        "hour": hour,
        "minute": minute,
        "days": list(days),
        "to": to_part,
        "message": message,
        "where": where,
        "channel": channel,
    }


def format_time(schedule: dict) -> str:
    """The time of a schedule, as it is shown to people."""
    hour = int(schedule.get("hour", 0))
    minute = int(schedule.get("minute", 0))
    return f"{hour:02d}:{minute:02d}"


def format_days(schedule: dict) -> str:
    """The days of a schedule, as they are shown to people."""
    days = tuple(schedule.get("days") or EVERY_DAY)
    if tuple(days) == EVERY_DAY:
        return "every day"
    if tuple(days) == WEEKDAYS:
        return "weekdays"
    if tuple(days) == WEEKEND:
        return "weekends"
    names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    return ", ".join(names[d] for d in days if 0 <= d <= 6)


def day_stamp(when: datetime) -> str:
    """The mark that says a schedule has already gone out today."""
    return when.strftime("%Y-%m-%d %H:%M")


def target_time(schedule: dict, when: datetime | None = None) -> datetime:
    """The moment a schedule is for, on the day it is checked.

    It is the same moment every time it is looked at within the day, so
    it can be used as a mark that says "this one has already gone out
    today" - a mark taken from the time of the attempt would change
    every minute and the message would go out again and again.
    """
    when = when or now_vietnam()
    hour = int(schedule.get("hour", 0))
    minute = int(schedule.get("minute", 0))
    return when.replace(hour=hour, minute=minute, second=0, microsecond=0)


def is_due(schedule: dict, when: datetime | None = None, grace: int = 5) -> bool:
    """Should this message go out in this minute?

    `grace` is the number of minutes after the time that still counts.
    It is there for the deploys: the bot is restarted every time it is
    deployed, and a restart that lands three minutes after the time
    should not lose the message.
    """
    when = when or now_vietnam()
    days = tuple(schedule.get("days") or EVERY_DAY)
    if when.weekday() not in days:
        return False

    target = target_time(schedule, when)
    delta = (when - target).total_seconds() / 60.0
    return 0 <= delta <= grace


def describe(schedule: dict) -> str:
    """One line about a schedule, for the list."""
    who = schedule.get("to_name") or schedule.get("to") or "somebody"
    where = "privately" if schedule.get("where") == "dm" else \
        f"in #{schedule.get('channel')}"
    message = (schedule.get("message") or "").replace("\n", " ")
    if len(message) > 60:
        message = message[:57] + "..."
    return (f"{format_time(schedule)} {format_days(schedule)} → {who} "
            f"({where}): {message}")
