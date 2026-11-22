"""Does the tool retry actually hold?

The model sometimes answers "what is X" straight out of its training data
instead of looking it up. The turn is then asked once more. This runs the
same question several times, with the cache cleared each time, and checks
that every reply looked something up.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


import agent  # noqa: E402
from cache import CACHES  # noqa: E402

JADE = {"id": "909453181916938301", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}
DAMIAN = {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}

QUESTIONS = [
    "What is filtering theory?",
    "What is a Kalman filter?",
    "Who won the Turing award in 2018?",
    "What is the difference between TCP and UDP?",
]

ROUNDS = int(os.getenv("ROUNDS", "3"))


async def main():
    used_every_time = 0
    total = 0
    for round_no in range(ROUNDS):
        for question in QUESTIONS:
            CACHES.clear("answer")
            result = await agent.get_nvidia_response_full(
                JADE["id"], question,
                scopes=["1499211380052721684", "channel:live-test"],
                write_scope="channel:live-test", shared=True,
                speaker=JADE, roster=[JADE, DAMIAN], shape="deep",
            )
            tools = result.get("tools_used") or []
            total += 1
            ok = bool(tools)
            used_every_time += ok
            print(f"  {'ok  ' if ok else 'FAIL'} round {round_no + 1} "
                  f"steps={result.get('steps')} tools={tools} - {question!r}")

    print()
    print(f"{used_every_time}/{total} answered with a lookup")
    sys.exit(0 if used_every_time == total else 1)


asyncio.run(main())
