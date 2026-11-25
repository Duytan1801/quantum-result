"""Several ways to ask the reranker the same question.

The first attempt - three intent descriptions as passages, the room
context in the query - was right 12 times out of 15 but with a margin
of 0.04 between the top two scores, which is no margin at all.

So this tries the other ways the same model could be asked:

  A  three intents, room context in the query
  B  two intents (answer vs quiet), room context in the query
  C  three intents, the bare message as the query
  D  two intents, the bare message as the query
  E  example messages as the passages, the bare message as the query
  F  three intents, the bare message as the query, no room at all

Every case is scored against the same 15 expected answers, and the
margin is recorded - because the margin is the part that decides
whether this can be used at all.
"""

import asyncio
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

INTENT = {
    "answer":
        "Someone is talking to the bot: they asked it a question, asked it "
        "to do something, mentioned it by name, or are telling it how they "
        "are. The bot should reply.",
    "quiet":
        "Someone is talking to another person in the chat: a joke, a remark "
        "in passing, a reply to the person next to them, or the bot being "
        "mentioned in the third person. The bot should stay quiet.",
    "wait":
        "Someone is typing a message that is not finished yet - the "
        "sentence trails off or stops in the middle. More of it is "
        "coming, so the bot should wait and not answer yet.",
}

EXAMPLES = {
    "answer": (
        "can you look up the Nvidia earnings date? / "
        "U hear me, lain? / what do you think about the new model? / "
        "mệt quá đi, hôm nay học nhiều quá / can you check on the plots?"
    ),
    "quiet": (
        "A tên j / Em của t / Mốt chắc lain2 cướp luôn e từ lain / "
        "Mà thấy sao nó ko dc tự nhiên cho lắm / cai gi vậy trời"
    ),
    "wait": (
        "can you look up the Nvidia / i was thinking, like, / "
        "the thing we talked about yesterday, the one where"
    ),
}

VARIANTS = {
    "A 3 intents + room": (["answer", "quiet", "wait"], INTENT, "room"),
    "B 2 intents + room": (["answer", "quiet"], INTENT, "room"),
    "C 3 intents + bare": (["answer", "quiet", "wait"], INTENT, "bare"),
    "D 2 intents + bare": (["answer", "quiet"], INTENT, "bare"),
    "E examples + bare": (["answer", "quiet", "wait"], EXAMPLES, "bare"),
    "F 3 intents + room+name": (["answer", "quiet", "wait"], INTENT, "full"),
}

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
WANT = {"A": "answer", "W": "wait", "Q": "quiet"}


def room_query(text, author, reply_to):
    lines = ["Chat: Lain, Jade", "The last messages:"]
    for who, said in RECENT:
        lines.append(f"  {who}: {said}")
    if reply_to:
        lines.append(f"The new message is a reply to {reply_to}.")
    lines.append(f"New message from {author}: {text}")
    return "\n".join(lines)


def full_query(text, author, reply_to):
    lines = [
        'A bot named "Lain" is in a group chat with: Lain, Jade.',
        "Chat mode is ON: the bot is here to join in.",
        "The last messages:",
    ]
    for who, said in RECENT:
        lines.append(f"  {who}: {said}")
    if reply_to:
        lines.append(f"The new message is a reply to {reply_to}.")
    lines.append(f"New message from {author}: {text}")
    return "\n".join(lines)


def rerank(query, passages):
    payload = {
        "model": MODEL,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
    }
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
    if r.status_code != 200:
        return None
    out = {}
    for row in r.json().get("rankings", []):
        out[row["index"]] = row["logit"]
    return out


async def main():
    results = {}
    timings = {}

    for label, (classes, source, qmode) in VARIANTS.items():
        t0 = time.time()
        right = 0
        margins = []
        picks = []
        for text, want, reply_to, spoke in CASES:
            author = "Jade" if reply_to != "Damian" else "Damian"
            if qmode == "bare":
                query = text
            elif qmode == "full":
                query = full_query(text, author, reply_to)
            else:
                query = room_query(text, author, reply_to)

            scores = rerank(query, [source[c] for c in classes])
            if not scores:
                picks.append("?")
                margins.append(0.0)
                continue
            order = sorted(scores.items(), key=lambda kv: -kv[1])
            pick = classes[order[0][0]]
            picks.append(pick)
            margins.append(order[0][1] - order[1][1])
            if pick == WANT[want]:
                right += 1
        elapsed = time.time() - t0
        results[label] = (right, margins, picks)
        timings[label] = elapsed / len(CASES)
        margins_sorted = sorted(margins)
        print(f"{label:24s} right={right:2d}/15  "
              f"margin min={margins_sorted[0]:5.2f} "
              f"med={margins_sorted[len(margins_sorted) // 2]:5.2f} "
              f"max={margins_sorted[-1]:5.2f}  "
              f"{timings[label]:.2f}s/case")

    print()
    print("per-case picks (want / A=answer Q=quiet W=wait)")
    for i, (text, want, _, _) in enumerate(CASES):
        line = " ".join(f"{label.split()[0]}={results[label][2][i][:5]:5s}"
                        for label in VARIANTS)
        print(f"  want={want}  {line}  {text[:44]!r}")


asyncio.run(main())
