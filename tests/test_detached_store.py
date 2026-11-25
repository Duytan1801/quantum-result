"""The end-to-end proof: what is stored in the secrets room stays there.

The scope plan says the secrets room writes only to itself and that other
rooms do not read its scope. This checks the thing that actually matters:
a memory written through the real store, and then looked for the way each
room looks for it.

The store is a vector search, so this uses the real embedding model and
the real search function. The text is made unique so it cannot be confused
with anything else in the store.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


from config import is_detached  # noqa: E402
from memory import retrieve_memories, store_memories  # noqa: E402
from presence import ChannelTracker, channel_scope  # noqa: E402
from scopes import scope_plan  # noqa: E402
from storage import db_add_message  # noqa: E402

SECRETS = "1573589377744117881"
MODEL = "1569657875368902666"
DAMIAN = "1499211380052721684"

# Something nobody in this bot has ever written down.
STAMP = f"zzsecret{int(time.time())}"
MEMORY_TEXT = (f"{STAMP}: the test of a memory that lives in the secrets "
               f"room and nowhere else")

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got}\n         want: {want}")


def check_true(label, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label} {detail}")


async def main():
    tracker = ChannelTracker()
    tracker.note(MODEL, DAMIAN)
    tracker.note(MODEL, "909453181916938301")

    secrets_plan = await scope_plan(SECRETS, DAMIAN, tracker=tracker)
    model_plan = await scope_plan(MODEL, DAMIAN, tracker=tracker)

    print("=" * 70)
    print("STORING A MEMORY THE WAY THE SECRETS ROOM WOULD")
    print("=" * 70)

    written = await store_memories(
        DAMIAN,
        [{"content": MEMORY_TEXT, "kind": "fact"}],
        source="conversation",
        scopes=secrets_plan.write,
    )
    check("the memory was stored", written, 1)

    # Where did it actually land?
    from storage import supabase

    resp = await asyncio.to_thread(
        lambda: supabase.table("memories")
        .select("user_id,content")
        .like("content", f"%{STAMP}%")
        .execute()
    )
    rows = resp.data or []
    landed = sorted({str(r.get("user_id")) for r in rows})
    check("it landed in the secrets scope and nowhere else",
          landed, [channel_scope(SECRETS)])

    print()
    print("=" * 70)
    print("LOOKING FOR IT THE WAY EACH ROOM LOOKS")
    print("=" * 70)

    # The secrets room: it must find it.
    found = await retrieve_memories(DAMIAN, MEMORY_TEXT, top_k=10,
                                    scopes=secrets_plan.read)
    check_true("the secrets room finds it",
               any(STAMP in (f.get("content") or "") for f in found),
               f"found {len(found)}")

    # The other room: it must not.
    found_model = await retrieve_memories(DAMIAN, MEMORY_TEXT, top_k=10,
                                          scopes=model_plan.read)
    check_true("the other room does not find it",
               not any(STAMP in (f.get("content") or "")
                       for f in found_model),
               f"found {len(found_model)}")

    # And the person's own scope must not have it either - that scope is
    # read in every room.
    check_true("the person's own scope does not have it",
               DAMIAN not in landed)

    print()
    print("=" * 70)
    print("THE CONVERSATION GOES THE SAME WAY")
    print("=" * 70)

    await db_add_message(secrets_plan.primary, "user",
                         f"{STAMP}: something said in the secrets room")
    resp = await asyncio.to_thread(
        lambda: supabase.table("conversation_history")
        .select("user_id,content")
        .like("content", f"%{STAMP}%")
        .execute()
    )
    landed = sorted({str(r.get("user_id")) for r in (resp.data or [])})
    check("the conversation landed in the secrets scope and nowhere else",
          landed, [channel_scope(SECRETS)])

    # Clean up, so the test leaves nothing behind.
    await asyncio.to_thread(
        lambda: supabase.table("memories")
        .delete().like("content", f"%{STAMP}%").execute()
    )
    await asyncio.to_thread(
        lambda: supabase.table("conversation_history")
        .delete().like("content", f"%{STAMP}%").execute()
    )

    print()
    print("=" * 70)
    print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    print("=" * 70)
    sys.exit(1 if FAILED else 0)


asyncio.run(main())
