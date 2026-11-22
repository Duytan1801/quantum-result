"""What happens when the classifier calls a personal message a real job?

The tool retry is fired by the presence classifier's "deep" shape. If the
classifier is wrong and a personal message is called a real job, the
model is asked to look something up. This is what that looks like: the
answer must still be warm and personal, and it must not start quoting
sources at somebody who was talking about their day.
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

PERSONAL = [
    "Hôm nay mệt quá, mình muốn nghỉ thôi",
    "I said that i miss jade",
    "mệt quá đi, hôm nay học nhiều quá",
]

SOURCE_TALK = ("according to", "source", "reference", "accordingly",
               "the article", "the paper", "the website", "searched",
               "per the", "as stated")


async def main():
    searched = 0
    for message in PERSONAL:
        CACHES.clear("answer")
        result = await agent.get_nvidia_response_full(
            JADE["id"], message,
            scopes=["1499211380052721684", "channel:live-test"],
            write_scope="channel:live-test", shared=True,
            speaker=JADE, roster=[JADE, DAMIAN], shape="deep",
        )
        answer = result["answer"]
        tools = result.get("tools_used") or []
        searched += bool(tools)
        says_sources = any(p in answer.lower() for p in SOURCE_TALK)
        print(f"  tools={tools} sources={says_sources} - {message!r}")
        print(f"    reply: {answer[:220]}")
        print()

    print(f"{searched}/{len(PERSONAL)} personal messages dragged into a search")
    print("The answer should still be warm, and still not talk about sources.")


asyncio.run(main())
