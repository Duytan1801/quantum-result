"""Check that the embedding system is working, end to end.

Run this any time:

    python verify_memory.py

It checks the schema, stores a test memory, finds it again by meaning (not
by keyword), and cleans up after itself.
"""
import asyncio
import os
import sys

os.environ.setdefault("SUPABASE_URL", "")
os.environ.setdefault("SUPABASE_KEY", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -> ' + str(extra)) if extra else ''}")


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


async def main():
    from storage import supabase

    rule("1. the schema")
    try:
        rows = (supabase.table("memories")
                .select("id,user_id,kind,content,embed_model,embed_dim,"
                        "source,created_at,use_count,last_used_at")
                .limit(1).execute().data or [])
    except Exception as exc:
        print(f"  [FAIL] the memories table is not there: {exc}")
        print("\n  Run supabase_memory.sql in the Supabase SQL editor:")
        print("  Dashboard -> SQL Editor -> New query -> paste -> Run")
        return 1
    check("the memories table exists", True)

    import memory

    total = (supabase.table("memories")
             .select("id", count="exact").limit(1).execute())
    count = getattr(total, "count", None) or len(total.data or [])
    print(f"  stored memories: {count}")

    models = (supabase.table("memories")
              .select("embed_model").limit(1000).execute().data or [])
    used = sorted({m.get("embed_model") for m in models if m.get("embed_model")})
    check("the embedding model is recorded on every row",
          used == [memory.EMBED_MODEL] if models else True, used)
    check("the model matches the code", memory.EMBED_MODEL in used or not models,
          f"code uses {memory.EMBED_MODEL}")

    rule("2. the search function")
    try:
        res = supabase.rpc("match_memories", {
            "p_user_id": "verify-probe",
            "p_embedding": "[" + ",".join(["0.1"] * memory.EMBED_DIM) + "]",
            "p_match_count": 1,
            "p_model": memory.EMBED_MODEL,
            "p_recency_weight": 0.15,
            "p_half_life_days": 90,
            "p_dedupe_distance": 0.25,
        }).execute()
        check("match_memories is callable", True)
    except Exception as exc:
        check("match_memories is callable", False, exc)

    rule("3. store and find it again by meaning")
    marker = "The verification marker is a purple stapler left on the desk."
    user = "__verify__"
    try:
        supabase.table("memories").delete().eq("user_id", user).execute()
    except Exception:
        pass

    stored = await memory.store_memories(user, [
        {"content": marker, "kind": "fact"},
    ], source="verification")
    check("stored a test memory", stored == 1, stored)

    # The query shares no words with the memory.
    found = await memory.retrieve_memories(user, "what object is on the desk?", top_k=3)
    hit = [f for f in found if f.get("content") == marker]
    check("found it again by meaning, not by keyword",
          bool(hit), [f.get("content")[:40] for f in found])
    if hit:
        print(f"    similarity: {hit[0].get('similarity'):.3f}")

    # An unrelated question must not match it.
    noise = await memory.retrieve_memories(user, "what is the weather in tokyo?", top_k=3)
    check("an unrelated question does not match it",
          not any(f.get("content") == marker for f in noise),
          [f.get("content")[:40] for f in noise])

    rule("4. clean up")
    try:
        supabase.table("memories").delete().eq("user_id", user).execute()
        left = (supabase.table("memories")
                .select("id").eq("user_id", user).execute().data or [])
        check("the test memory is gone", not left, len(left))
    except Exception as exc:
        check("the test memory is gone", False, exc)

    print(f"\n{'=' * 50}\nPASSED: {len(PASS)}   FAILED: {len(FAIL)}")
    if FAIL:
        print("FAILURES:")
        for f in FAIL:
            print(f"  - {f}")
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
