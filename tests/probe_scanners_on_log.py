"""Every candidate scanner over the real channel log.

The 15 test cases are a test set, and they are small. This is the real
thing: every human message in the last hundred messages of #model, put
to each scanner with the room as it was at that moment.

Three scanners, all reading the room, all answering the same question:

  minimax   - the generative classifier, MiniMax first (the user's
              choice: MiniMax scans, Laguna answers)
  bolt      - the same prompt, but on Nemotron 3.5 Bolt e4b, which is
              a much smaller and much faster model
  rerank    - the reranker, with example messages as the passages

The bot's own behaviour on each message is printed next to them, so
the disagreements can be looked at rather than just counted.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

import requests

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

BOLD = "nvidia/nemotron-3.5-bolt-e4b"
RERANK_MODEL = "nvidia/nemotron-3.5-rerank-8b"
RERANK_URL = (f"https://ai.api.nvidia.com/v1/retrieval/{RERANK_MODEL}"
              f"/reranking")
RERANK_HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Accept": "application/json",
}

EXAMPLES = {
    "answer": (
        "can you look up the Nvidia earnings date? / "
        "U hear me, lain? / what do you think about the new model? / "
        "mệt quá đi, hôm nay học nhiều quá / can you check on the plots? / "
        "yo lain whos jade / is there any research shows that a female "
        "doctor while on the way gettin diploma would give birth to a child "
        "that carries a risk? / Lainy, what do you think? / hello bot"
    ),
    "quiet": (
        "A tên j / Em của t / Mốt chắc lain2 cướp luôn e từ lain / "
        "Mà thấy sao nó ko dc tự nhiên cho lắm / cai gi vậy trời / "
        "Tại minimax có điểm intelligence cao hơn nên a chuyển sang nó / "
        "I love lain2 more than you / =))) ulatr tổng tài qus / "
        "the new model is slower but smarter"
    ),
    "wait": (
        "can you look up the Nvidia / i was thinking, like, / "
        "the thing we talked about yesterday, the one where / "
        "so i was saying that the / and then she"
    ),
}
CLASSES = ["answer", "quiet", "wait"]


def fetch():
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            return json.load(f)
    url = (f"https://discord.com/api/v10/channels/{CHANNEL_ID}"
           f"/messages?limit=100")
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


def rerank(query):
    payload = {
        "model": RERANK_MODEL,
        "query": {"text": query},
        "passages": [{"text": EXAMPLES[c]} for c in CLASSES],
    }
    try:
        r = requests.post(RERANK_URL, headers=RERANK_HEADERS, json=payload,
                           timeout=60)
        if r.status_code != 200:
            return None
        rows = r.json().get("rankings", [])
        if not rows:
            return None
        best = rows[0]
        margin = (best["logit"] - rows[1]["logit"]) if len(rows) > 1 else 0.0
        return CLASSES[best["index"]], margin
    except Exception:
        return None


async def main():
    from presence import (_classifier_prompt, _name_note, _parse_verdict,
                          classify_reply)
    from people import bot_name_variants, name_collisions, names_in

    raw = list(reversed(fetch()))
    bot_names = bot_name_variants(type("U", (), {
        "id": BOT_ID, "display_name": "lain2", "global_name": "lain2",
        "name": "lain2", "bot": True})())
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

    results = {"minimax": [], "bolt": [], "rerank": []}
    timings = {"minimax": 0.0, "bolt": 0.0, "rerank": 0.0}

    for i, row in enumerate(rows):
        text = row["text"]
        author = row["who"]
        reply_to = row["reply_to"]
        spoke = row["spoke"]
        hits = names_in(text, bot_names)
        name_note = _name_note(hits, collisions)
        prompt = _classifier_prompt(
            text, author_name=author, roster_line=roster_line,
            bot_name="Lain", chat_on=True, recent=recent[-6:],
            reply_to=reply_to, bot_spoke_seconds=spoke,
            name_note=name_note, bot_aliases="lain2, lain, lainy")

        # 1. MiniMax (the generative classifier, MiniMax first).
        t0 = time.time()
        verdict = await classify_reply(
            text, author_name=author, roster_line=roster_line,
            bot_name="Lain", chat_on=True, recent=recent[-6:],
            reply_to=reply_to, bot_spoke_seconds=spoke,
            name_note=name_note, bot_aliases="lain2, lain, lainy")
        timings["minimax"] += time.time() - t0
        if verdict is None:
            results["minimax"].append(("?", 0.0))
        elif verdict[1]:
            results["minimax"].append(("wait", 0.0))
        elif verdict[0]:
            results["minimax"].append(("answer", 0.0))
        else:
            results["minimax"].append(("quiet", 0.0))

        # 2. Bolt e4b on the same prompt.
        t0 = time.time()
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                                 api_key=os.environ["NVIDIA_API_KEY"])
            r = await client.chat.completions.create(
                model=BOLD,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_completion_tokens=8,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            content = (r.choices[0].message.content or "").strip()
            decision, _shape = _parse_verdict(content)
            results["bolt"].append((decision or "?", 0.0))
        except Exception as exc:
            results["bolt"].append((f"err:{str(exc)[:20]}", 0.0))
        timings["bolt"] += time.time() - t0

        # 3. The reranker.
        t0 = time.time()
        q = ["Chat: " + roster_line, "The last messages:"]
        for who, said in recent[-6:]:
            q.append(f"  {who}: {said}")
        if reply_to:
            q.append(f"The new message is a reply to {reply_to}.")
        if spoke is not None and spoke < 180:
            q.append(f"The bot spoke {int(spoke)} seconds ago, so the room "
                     f"is not its own right now.")
        q.append(f"New message from {author}: {text}")
        out = rerank("\n".join(q))
        timings["rerank"] += time.time() - t0
        results["rerank"].append(out if out else ("?", 0.0))

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(rows)} done", flush=True)
        await asyncio.sleep(0.7)

    print()
    print("=" * 100)
    print(f"{'when':14s} {'who':7s} {'bot did':9s} "
          + " ".join(f"{k:8s}" for k in results))
    print("=" * 100)
    for i, row in enumerate(rows):
        actual = "answered" if row["actual"] else "-"
        cells = []
        for k in results:
            pick, margin = results[k][i]
            short = {"answer": "ANSWER", "quiet": "quiet", "wait": "WAIT"}.get(
                pick, pick[:7])
            cells.append(f"{short:8s}")
        print(f"{row['when']:%m-%d %H:%M:%S} {row['who']:7s} {actual:9s} "
              + " ".join(cells) + f" {row['text'][:34]!r}")

    print()
    for k in results:
        picks = [p[0] for p in results[k]]
        answers = picks.count("answer")
        waits = picks.count("wait")
        quiet = picks.count("quiet")
        errors = sum(1 for p in picks if p.startswith(("?", "err")))
        same = sum(1 for i, row in enumerate(rows)
                   if row["actual"] == (results[k][i][0] == "answer"))
        print(f"{k:9s} answers={answers:2d} waits={waits:2d} quiet={quiet:2d} "
              f"errors={errors:2d}  agrees with the bot on {same}/{len(rows)}  "
              f"{timings[k] / len(rows):.2f}s/msg")

    with open("/tmp/opencode/scanner_log.json", "w") as f:
        json.dump({"rows": [{k: str(v) if k == "when" else v
                             for k, v in r.items()} for r in rows],
                   "results": {k: v for k, v in results.items()}}, f,
                  indent=1, ensure_ascii=False)


asyncio.run(main())
