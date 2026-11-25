"""Every candidate scanner over the real channel log.

One scanner: Bolt, now as the presence scanner, with Laguna doing the
answering. Every human message in the last hundred messages of #model,
put to the classifier with the room as it was at that moment.

The bot's own behaviour on each message is printed next to them, so
the disagreements can be looked at rather than just counted.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from openai import AsyncOpenAI

sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot/tests")
sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot")
os.chdir("/home/bandoc/Downloads/discord/groq-discord-bot")
from dotenv import load_dotenv  # noqa: E402

load_dotenv("/home/bandoc/Downloads/discord/groq-discord-bot/.env")

CHANNEL_ID = os.environ.get("CHANNEL_ID", "1569657875368902666")
TOKEN = os.environ["DISCORD_TOKEN"]
BOT_ID = "1569644550899630112"
LOG = "/tmp/opencode/channel_log.json"

NAMES = {
    "909453181916938301": "Jade",
    "1499211380052721684": "Damian",
    BOT_ID: "the bot",
}

BOLT = "nvidia/nemotron-3.5-bolt-e4b"
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Accept": "application/json",
}


def fetch():
    import urllib.request
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


async def classify_bolt(prompt):
    """Ask Bolt the presence question."""
    t0 = time.time()
    try:
        r = await AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                              api_key=os.environ["NVIDIA_API_KEY"]).chat.completions.create(
            model=BOLT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_completion_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return (r.choices[0].message.content or "").strip(), time.time() - t0
    except Exception as exc:
        return None, time.time() - t0


def rerank(query):
    """Ask the reranker if we really need it."""
    EXAMPLES = {
        "answer": (
            '"can you look up the Nvidia earnings date?" / '
            '"U hear me, lain?" / "what do you think about the new model?"'
        ),
        "quiet": (
            '"Mốt chắc lain2" / "A tên j" / "Mà thấy sao nó ko dc"'
        ),
        "wait": (
            '"can you look up the Nvidia" / "i was thinking, like," / '
            '"the thing we talked about yesterday, the one where"'
        ),
    }
    payload = {
        "model": "nvidia/nemotron-3.5-rerank-8b",
        "query": {"text": query},
        "passages": [{"text": EXAMPLES[c]} for c in ("answer", "quiet", "wait")],
    }
    try:
        import requests
        r = requests.post("https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-rerank-8b/reranking",
                          headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            return None
        rows = r.json().get("rankings", [])
        if not rows:
            return None
        return rows[0]["index"]  # 0=answer, 1=quiet, 2=wait
    except Exception:
        return None


async def main():
    from presence import _classifier_prompt, _name_note, _parse_verdict
    from people import bot_name_variants, name_collisions, names_in

    raw = list(reversed(fetch()))
    bot_names = ("lain2", "lain", "lainy")
    roster = [{"id": "909453181916938301", "name": "Jade", "plain": "Jade"},
              {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}]
    collisions = name_collisions(bot_names, roster)
    roster_line = "Lain, Jade"

    recent = []
    last_bot = None
    bot_ids = set()
    rows = []

    for message in raw:
        when = ts(message["timestamp"])
        content = message.get("content") or ""
        if message["author"]["id"] != BOT_ID:
            reply_to = ""
            ref = message.get("referenced_message")
            if ref:
                who = NAMES.get(ref["author"]["id"])
                if who and who != "the bot":
                    reply_to = who
            spoke = None
            if last_bot is not None:
                spoke = (when - last_bot).total_seconds()
            rows.append({"when": when, "who": NAMES.get(
                message["author"]["id"], "?"), "text": content,
                "reply_to": reply_to, "spoke": spoke})
        else:
            last_bot = when
            bot_ids.add(message["id"])
        recent.append((NAMES.get(message["author"]["id"], "?"), content[:200]))

    for row in rows:
        after = [m for m in raw if ts(m["timestamp"]) > row["when"]]
        row["actual"] = bool(after) and after[0]["author"]["id"] == BOT_ID

    print(f"{len(rows)} human messages\n")

    answers = waits = quiet = 0
    same = 0
    print("=" * 100)
    print(f"{'when':14s} {'who':7s} {'bot did':9s} {'bolt':6s} "
          f"{'since bot spoke':16s} why / message")
    print("=" * 100)

    for i, row in enumerate(rows):
        text = row["text"]
        author = row["who"]
        reply_to = row["reply_to"]
        spoke = row["spoke"]
        hits = names_in(text, bot_names)
        prompt = _classifier_prompt(
            text, author_name=author, roster_line=roster_line,
            bot_name="Lain", chat_on=True, recent=recent[-6:],
            reply_to=reply_to, bot_spoke_seconds=spoke,
            name_note=_name_note(hits, collisions),
            bot_aliases="lain2, lain, lainy")

        content, t = await classify_bolt(prompt)
        if content:
            d, s = _parse_verdict(content)
            if d == "wait":
                pick = "WAIT"
                waits += 1
            elif d == "answer":
                pick = "ANSWER"
                answers += 1
            elif d == "quiet":
                pick = "quiet"
                quiet += 1
            else:
                pick = "?"
        else:
            pick = "?????"
        if row["actual"] == (pick == "ANSWER"):
            same += 1

        actual = "answered" if row["actual"] else "-"
        spoke_str = f"{int(spoke)}s" if spoke else "-"
        text_short = text.replace("\n", " ")[:46]
        if row["reply_to"] or (row["actual"] != (pick == "ANSWER")):
            print(f"{row['when']:%m-%d %H:%M:%S} {row['who']:7s} {actual:9s} "
                  f"{pick:6s} {spoke_str:16s} {text_short!r}")

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(rows)} done", flush=True)
        await asyncio.sleep(0.5)

    print(f"\nthe new scanner answers {answers + waits + quiet} of {len(rows)}")
    print(f"  answers={answers} waits={waits} quiet={quiet}")
    print(f"agrees with the bot's actual behaviour: {same}/{len(rows)}")


asyncio.run(main())