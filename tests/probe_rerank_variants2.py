"""The reranker, with the room context put back in.

The first pass put intent descriptions in the passages and they all
scored the same: a margin of 0.04, which is nothing. The second pass
put example messages in the passages instead - that is the same thing a
reranker was trained to do, compare a query against a document - and the
margin went to 0.84 with 13 of 15 right.

This tries that again with the room context in the query, with more
examples, and with the examples written the way the real messages are
written - so the reranker is comparing like with like.
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

SMALL = {
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

# The same three, with the real shape of the room's messages added.
BIG = {
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

VARIANTS = {
    "E  small + bare": (SMALL, "bare"),
    "G  small + room": (SMALL, "room"),
    "H  big + bare": (BIG, "bare"),
    "I  big + room": (BIG, "room"),
    "J  big + room+name": (BIG, "full"),
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
CLASSES = ["answer", "quiet", "wait"]


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

    for label, (source, qmode) in VARIANTS.items():
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
            scores = rerank(query, [source[c] for c in CLASSES])
            if not scores:
                picks.append("?")
                margins.append(0.0)
                continue
            order = sorted(scores.items(), key=lambda kv: -kv[1])
            pick = CLASSES[order[0][0]]
            picks.append(pick)
            margins.append(order[0][1] - order[1][1])
            if pick == WANT[want]:
                right += 1
        elapsed = (time.time() - t0) / len(CASES)
        results[label] = (right, margins, picks)
        ms = sorted(margins)
        print(f"{label:20s} right={right:2d}/15  "
              f"margin min={ms[0]:5.2f} med={ms[len(ms)//2]:5.2f} "
              f"max={ms[-1]:5.2f}  {elapsed:.2f}s/case")

    print()
    print("per-case picks")
    for i, (text, want, _, _) in enumerate(CASES):
        line = " ".join(f"{lab.split()[0]}={results[lab][2][i][:5]:5s}"
                        for lab in VARIANTS)
        print(f"  want={want}  {line}  {text[:44]!r}")


asyncio.run(main())
