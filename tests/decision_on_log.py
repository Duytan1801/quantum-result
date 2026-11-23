"""Run the whole presence decision over the real channel log.

This time the real `decide()` is used, with real Discord messages turned
into objects, so the structural checks, the name collisions and the
classifier all run exactly as they do in the bot.

Each human message is put to the decision with the room as it was at that
moment: the messages before it, who it was replying to, and how long since
the bot last spoke. The bot's own behaviour is printed next to it.
"""

import asyncio
import json
import os
import sys
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY", "DISCORD_TOKEN")


CHANNEL_ID = os.environ.get("CHANNEL_ID", "1569657875368902666")
TOKEN = os.environ["DISCORD_TOKEN"]
BOT_ID = "1569644550899630112"
LOG = os.environ.get("CHANNEL_LOG", "/tmp/opencode/channel_log.json")
PAUSE = float(os.getenv("PAUSE", "1.0"))

NAMES = {
    "909453181916938301": "Jade",
    "1499211380052721684": "Damian",
    BOT_ID: "the bot",
}


class FakeUser:
    def __init__(self, uid, name):
        self.id = uid
        self.display_name = name
        self.global_name = name
        self.name = name
        self.bot = uid == BOT_ID


class FakeRef:
    def __init__(self, message_id):
        self.message_id = message_id


class FakeMessage:
    def __init__(self, raw):
        self.content = raw.get("content") or ""
        self.id = raw["id"]
        author = raw["author"]
        self.author = FakeUser(author["id"],
                               NAMES.get(author["id"],
                                         author.get("global_name") or "?"))
        self.author.bot = author.get("bot", False)
        ref = raw.get("message_reference") or {}
        mid = ref.get("message_id")
        if not mid and raw.get("referenced_message"):
            mid = raw["referenced_message"].get("id")
        self.reference = FakeRef(mid) if mid else None
        self.referenced_message = raw.get("referenced_message")


def fetch():
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            return json.load(f)
    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages?limit=100"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bot {TOKEN}",
        "User-Agent": "DiscordBot (https://github.com/Duytan1801/quantum-result, 1.0)",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


async def main():
    from presence import decide
    from people import bot_name_variants, plain_name, roster_from_messages

    raw_messages = list(reversed(fetch()))
    messages = [FakeMessage(m) for m in raw_messages]

    bot_user = FakeUser(BOT_ID, "lain2")
    bot_names = bot_name_variants(bot_user)

    # The roster, read from the messages the way the bot reads it.
    roster = roster_from_messages(messages)
    roster_line = ", ".join(r["name"] for r in roster)

    print(f"{len(messages)} messages in the log")
    print(f"bot names: {bot_names}")
    print(f"roster: {roster_line}\n")

    recent = []
    last_bot_reply = None
    bot_message_ids = set()
    rows = []

    for message in messages:
        when = ts(next(m["timestamp"] for m in raw_messages
                       if m["id"] == message.id))
        content = message.content

        if message.author.id != BOT_ID:
            reply_to = ""
            ref = message.referenced_message
            if ref:
                ref_who = NAMES.get(ref["author"]["id"])
                if ref_who and ref_who != "the bot":
                    reply_to = ref_who

            spoke = None
            if last_bot_reply is not None:
                spoke = (when - last_bot_reply).total_seconds()

            await asyncio.sleep(PAUSE)
            d = await decide(
                message,
                is_dm=False,
                shared=True,
                bot_user_id=BOT_ID,
                bot_names=bot_names,
                bot_message_ids=bot_message_ids,
                chat_on=True,
                roster=roster,
                author_name=message.author.display_name,
                roster_line=roster_line,
                bot_name="Lain",
                recent=recent[-6:],
                reply_to=reply_to,
                bot_spoke_seconds=spoke,
                cache_extra="log",
            )
            rows.append({
                "when": when, "who": message.author.display_name,
                "text": content, "reply_to": reply_to, "decision": d,
                "spoke": spoke,
            })
        else:
            last_bot_reply = when
            bot_message_ids.add(message.id)

        who = message.author.display_name
        recent.append((who, content[:200]))

    for row in rows:
        after = [m for m in raw_messages if ts(m["timestamp"]) > row["when"]]
        row["actual"] = bool(after) and after[0]["author"]["id"] == BOT_ID

    print("=" * 112)
    print(f"{'when':14s} {'who':7s} {'bot did':9s} {'new':7s} {'shape':8s} "
          f"{'since bot spoke':16s} why / message")
    print("=" * 112)
    for row in rows:
        d = row["decision"]
        if d.wait:
            new = "WAIT"
        elif d.reply:
            new = "ANSWER"
        else:
            new = "quiet"
        actual = "answered" if row["actual"] else "-"
        spoke = (f"{row['spoke']:.0f}s" if row["spoke"] is not None else "-")
        text = row["text"].replace("\n", " ")[:46]
        changed = (row["actual"]) != (new == "ANSWER")
        print(f"{row['when']:%m-%d %H:%M:%S} {row['who']:7s} {actual:9s} "
              f"{new:7s} {d.shape:8s} {spoke:16s} {text!r}")
        if row["reply_to"] or changed:
            print(f"{'':14s} (reply to {row['reply_to'] or '-'}, by {d.reason})")

    answered_now = sum(1 for r in rows if r["decision"].reply)
    waited = sum(1 for r in rows if r["decision"].wait)
    answered_then = sum(1 for r in rows if r["actual"])
    same = sum(1 for r in rows if r["actual"] == r["decision"].reply)
    print()
    print(f"the bot answered {answered_then} of {len(rows)} human messages")
    print(f"the new decision answers {answered_now} and waits on {waited}")
    print(f"the two agree on {same} of {len(rows)}")


asyncio.run(main())
