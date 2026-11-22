"""A live run with Laguna first in the chain.

The same three checks as before - a personal message, a technical
question, and the cache - but now the first model in the chain is Laguna,
so this is what the bot will actually sound like.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY", "KAGGLE_API_TOKEN")


import agent  # noqa: E402
from cache import CACHES  # noqa: E402

JADE = {"id": "909453181916938301", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}
DAMIAN = {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}
SCOPES = ["1499211380052721684", "channel:1569657875368902666"]

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


BOT_SPEAK = (
    "as an ai", "i'm just an ai", "great question", "i hope this helps",
    "let me know if you need", "happy to help", "talking - the default",
    "working - there is a task",
)

# The advice-giving opener that the prompt forbids and that the code
# model is inclined to produce.
ADVICE = ("i'd suggest", "i would suggest", "you could try", "here are some",
          "some ideas", "maybe you could", "consider ")


async def ask(message, speaker, shape=""):
    t0 = time.time()
    result = await agent.get_nvidia_response_full(
        speaker["id"], message, scopes=SCOPES, write_scope="channel:live-test",
        shared=True, speaker=speaker, roster=[JADE, DAMIAN], shape=shape,
    )
    return result, round(time.time() - t0, 1)


async def main():
    print("=" * 70)
    print("1. A PERSONAL MESSAGE FROM JADE (Vietnamese)")
    print("=" * 70)

    result, elapsed = await ask("Hôm nay mệt quá, mình muốn nghỉ thôi", JADE)
    answer = result["answer"]
    print(f"  ({elapsed}s, model: {result.get('model')}, "
          f"tools: {result.get('tools_used')})")
    print(f"  reply: {answer[:500]}")
    print()

    check_true("it answered", bool(answer.strip()))
    check("it used no tools", result.get("tools_used"), [])
    for phrase in BOT_SPEAK:
        check_true(f"no bot-speak: {phrase!r}", phrase not in answer.lower())
    check_true("it is in Vietnamese",
               any(" " <= ch <= "ỹ" for ch in answer.lower()))
    check_true("no unsolicited advice",
               not any(p in answer.lower() for p in ADVICE))

    print()
    print("=" * 70)
    print("2. A TECHNICAL QUESTION")
    print("=" * 70)

    CACHES.clear("answer")
    result2, elapsed2 = await ask("What is filtering theory?", JADE, shape="deep")
    answer2 = result2["answer"]
    print(f"  ({elapsed2}s, model: {result2.get('model')}, "
          f"tools: {result2.get('tools_used')})")
    print(f"  reply: {answer2[:400]}")
    print()

    check_true("it answered", bool(answer2.strip()))
    check_true("it used a tool", bool(result2.get("tools_used")))
    for phrase in BOT_SPEAK:
        check_true(f"no bot-speak: {phrase!r}", phrase not in answer2.lower())

    print()
    print("=" * 70)
    print("3. THE CACHE")
    print("=" * 70)

    result3, elapsed3 = await ask("What is filtering theory?", JADE, shape="deep")
    print(f"  ({elapsed3}s)")
    check("the cached answer is the same", result3["answer"], answer2)
    check("the model was not called again", result3.get("steps"), 0)

    print()
    print("=" * 70)
    print("4. A PASSING REMARK GETS A SHORT REPLY")
    print("=" * 70)

    CACHES.clear("answer")
    result4, elapsed4 = await ask("mệt quá đi, hôm nay học nhiều quá", JADE,
                                   shape="aside")
    answer4 = result4["answer"]
    words = len(answer4.split())
    print(f"  ({elapsed4}s, {words} words, "
          f"tools: {result4.get('tools_used')})")
    print(f"  reply: {answer4[:300]}")
    print()

    check_true("it answered", bool(answer4.strip()))
    check_true("the reply is short", words <= 60)
    check("it used no tools", result4.get("tools_used"), [])
    for phrase in BOT_SPEAK:
        check_true(f"no bot-speak: {phrase!r}", phrase not in answer4.lower())

    print()
    print("=" * 70)
    print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
    if FAILED:
        for f in FAILED:
            print(f"  FAILED: {f}")
        sys.exit(1)
    print("ALL GREEN")


asyncio.run(main())
