"""Can the reranker decide whether the bot should answer?

The MiniMax classifier is the thing in the bot today. The reranker is
the idea: put the message in the query, put the possible intents in the
passages, and let the reranker say which one the message is closest to.

This runs both over the same cases and prints them side by side, with
the margin between the top two scores - because a reranker that is right
but only by a hair is not actually right, it is a coin toss with extra
steps.
"""

import asyncio
import json
import os
import sys
import time

import requests

sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot/tests")
sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot")
os.chdir("/home/bandoc/Downloads/discord/groq-discord-bot")
from dotenv import load_dotenv  # noqa: E402

load_dotenv("/home/bandoc/Downloads/discord/groq-discord-bot/.env")

MODEL = "nvidia/nemotron-3.5-rerank-8b"
URL = f"https://ai.api.nvidia.com/v1/retrieval/{MODEL}/reranking"
HEADERS = {
    "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
    "Accept": "application/json",
}

# The three intents, written as passages.
PASSAGES = [
    ("answer",
     "Someone is talking to the bot: they asked it a question, asked it to "
     "do something, mentioned it by name, or are telling it how they are. "
     "The bot should reply."),
    ("quiet",
     "Someone is talking to another person in the chat: a joke, a remark "
     "in passing, a reply to the person next to them, or the bot being "
     "mentioned in the third person. The bot should stay quiet."),
    ("wait",
     "Someone is typing a message that is not finished yet - the sentence "
     "trails off or stops in the middle. More of it is coming, so the bot "
     "should wait and not answer yet."),
]

TEXT = [p[1] for p in PASSAGES]
NAMES = [p[0] for p in PASSAGES]


def rerank(query: str):
    payload = {
        "model": MODEL,
        "query": {"text": query},
        "passages": [{"text": t} for t in TEXT],
    }
    t0 = time.time()
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
    elapsed = time.time() - t0
    if r.status_code != 200:
        return None, elapsed, r.text[:200]
    data = r.json()
    scores = {}
    for row in data.get("rankings", []):
        scores[NAMES[row["index"]]] = row["logit"]
    return scores, elapsed, data.get("usage", {})


def build_query(text, author, roster, recent, reply_to):
    lines = [f"Chat: {roster}"]
    if recent:
        lines.append("The last messages:")
        for who, said in recent:
            lines.append(f"  {who}: {said}")
    if reply_to:
        lines.append(f"The new message is a reply to {reply_to}.")
    lines.append(f"New message from {author}: {text}")
    return "\n".join(lines)


CASES = [
    ("can you look up the Nvidia", "W", "", None),
    ("i was thinking, like,", "W", "", None),
    ("the thing we talked about yesterday, the one where", "W", "", None),
    ("can you look up the Nvidia earnings date?", "A", "", None),
    ("Can you check on the plots?", "A", "", None),
    ("U hear me, lain?", "A", "", None),
    ("mệt quá đi, hôm nay học nhiều quá", "A", "", None),
    ("Is there any research shows that a female doctor while on the way "
     "gettin diploma would give birth to a child that carries a risk?",
     "A", "", None),
    ("Mốt chắc lain2 cướp luôn e từ lain", "Q", "", None),
    ("A tên j", "Q", "Damian", None),
    ("Em của t", "Q", "Damian", None),
    ("Mà thấy sao nó ko dc tự nhiên cho lắm", "Q", "Jade", None),
    ("Tại minimax có điểm intelligence cao hơn nên a chuyển sang nó", "Q", "Jade", None),
    ("Lain", "Q", "", None),
    ("mệt quá đi", "Q", "", 30),
]

RECENT = [("Damian", "A tên j"), ("Jade", "Tot")]


async def main():
    from presence import classify_reply, _name_note
    from people import name_collisions, names_in

    BOT_NAMES = ("lain2", "lain", "lainy")
    ROSTER = [{"id": "1", "name": "Lain", "plain": "Lain"},
              {"id": "2", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}]
    COLLISIONS = name_collisions(BOT_NAMES, ROSTER)

    same = 0
    total = 0
    margins = []

    for text, want, reply_to, spoke in CASES:
        author = "Jade" if reply_to != "Damian" else "Damian"
        hits = names_in(text, BOT_NAMES)
        query = build_query(text, author, "Lain, Jade", RECENT, reply_to)
        scores, elapsed, usage = rerank(query)

        if scores:
            order = sorted(scores.items(), key=lambda kv: -kv[1])
            got = order[0][0]
            margin = order[0][1] - order[1][1]
            margins.append(margin)
            detail = (f"{got:6s} margin={margin:5.2f}  "
                      f"a={scores['answer']:6.2f} q={scores['quiet']:6.2f} "
                      f"w={scores['wait']:6.2f}")
        else:
            got = "?"
            detail = f"FAILED {usage}"

        # The MiniMax classifier, for the same case.
        await asyncio.sleep(1.0)
        verdict = await classify_reply(
            text,
            author_name=author,
            roster_line="Lain, Jade",
            bot_name="Lain",
            chat_on=True,
            recent=RECENT,
            reply_to=reply_to,
            bot_spoke_seconds=spoke,
            name_note=_name_note(hits, COLLISIONS),
            bot_aliases="lain2, lain, lainy",
        )
        if verdict is None:
            mm = "?"
        elif verdict[1]:
            mm = "wait"
        elif verdict[0]:
            mm = "answer"
        else:
            mm = "quiet"

        agree = (got == mm)
        same += agree
        total += 1
        print(f"want={want}  rerank={got:6s} minimax={mm:6s} "
              f"{'agree' if agree else 'DIFFER':6s} {detail}")
        print(f"    {text[:70]!r}")

    print()
    print(f"the reranker and MiniMax agree on {same} of {total}")
    if margins:
        margins.sort()
        print(f"margin between the top two: min={margins[0]:.2f} "
              f"median={margins[len(margins) // 2]:.2f} "
              f"max={margins[-1]:.2f}")


asyncio.run(main())
