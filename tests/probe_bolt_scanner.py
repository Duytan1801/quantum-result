"""How good would Nemotron 3.5 Bolt e4b be as the scanner?

It answered the presence question in 0.58s and got it right. This runs
it over the same 15 cases the MiniMax classifier passes, and then
measures whether it survives the same pace the test suite throws at it -
the MiniMax model 429s about half the time at that pace, and a scanner
that cannot be called is not a scanner.
"""

import asyncio
import os
import sys
import time

from openai import AsyncOpenAI

sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot/tests")
sys.path.insert(0, "/home/bandoc/Downloads/discord/groq-discord-bot")
os.chdir("/home/bandoc/Downloads/discord/groq-discord-bot")
from dotenv import load_dotenv  # noqa: E402

load_dotenv("/home/bandoc/Downloads/discord/groq-discord-bot/.env")

BOLT = "nvidia/nemotron-3.5-bolt-e4b"

client = AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1",
                     api_key=os.environ["NVIDIA_API_KEY"])

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


async def bolt_verdict(prompt, tokens=8):
    t0 = time.time()
    try:
        r = await client.chat.completions.create(
            model=BOLT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (r.choices[0].message.content or "").strip(), time.time() - t0, None
    except Exception as exc:
        return None, time.time() - t0, str(exc)[:90]


async def main():
    from presence import _classifier_prompt, _name_note, _parse_verdict
    from people import name_collisions, names_in

    BOT_NAMES = ("lain2", "lain", "lainy")
    ROSTER = [{"id": "1", "name": "Lain", "plain": "Lain"},
              {"id": "2", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}]
    COLLISIONS = name_collisions(BOT_NAMES, ROSTER)

    right = 0
    times = []
    print("=" * 76)
    print("BOLT e4b AS THE SCANNER")
    print("=" * 76)
    for text, want, reply_to, spoke in CASES:
        author = "Jade" if reply_to != "Damian" else "Damian"
        hits = names_in(text, BOT_NAMES)
        prompt = _classifier_prompt(
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
        content, elapsed, err = await bolt_verdict(prompt)
        times.append(elapsed)
        if err:
            print(f"  ERR  {err}  {text[:40]!r}")
            continue
        decision, shape = _parse_verdict(content)
        got = ("?" if decision is None
               else "W" if decision == "wait"
               else "A" if decision == "answer" else "Q")
        ok = got == want
        right += ok
        print(f"  {'ok  ' if ok else 'FAIL'} got={got} want={want} "
              f"shape={shape:8s} {elapsed:5.2f}s raw={content!r:10s} "
              f"{text[:44]!r}")
        await asyncio.sleep(1.0)

    print(f"\n{right}/{len(CASES)} passed")
    times.sort()
    print(f"call time: min={times[0]:.2f}s median={times[len(times)//2]:.2f}s "
          f"max={times[-1]:.2f}s")

    print()
    print("=" * 76)
    print("RATE LIMIT: 20 calls at 1 second apart")
    print("=" * 76)
    ok = err = 0
    for i in range(20):
        content, elapsed, e = await bolt_verdict("Answer with one character: Y")
        if e:
            err += 1
            if err == 1:
                print(f"  first error: {e}")
        else:
            ok += 1
        await asyncio.sleep(1.0)
    print(f"  ok={ok} err={err} of 20 at 1s apart")


asyncio.run(main())
