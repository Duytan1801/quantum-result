"""Semantic long-term memory.

SimpleMem's core idea - store atomic memories and retrieve the ones relevant to
the current question, instead of summarising the whole conversation into one
lossy blob - is implemented here against the stack the bot already has:
Supabase for storage and the NVIDIA embeddings API for vectors.

No heavy dependencies: no torch, no LanceDB, no local vector store.
"""

import asyncio
import json
import os
import re
import time

import httpx

from config import NVIDIA_API_KEY, NVIDIA_API_URL, provider_chain

# Benchmarked against the other NVIDIA embedding models on real memories:
# nemotron-3-embed-1b is the fastest, but nemotron-3.5-embed-1b is the newer
# model line and matched it on retrieval quality (R@1 0.933, MRR 0.943) in
# testing. Both are 2048 dimensions, so the vector index size is the same.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
EMBED_DIM = 2048

MEMORY_TABLE = "memories"
MATCH_FN = "match_memories"

# Where the memories live is optional - if the table is missing the bot just
# runs without long-term memory.
_available = None
_available_checked_at = 0.0

# Re-probe this often if the table is missing, so the bot picks it up after
# the SQL has been run without needing a restart.
REPROBE_SECONDS = 300


async def is_available() -> bool:
    """Is the memories table set up? Cached, but re-probed periodically."""
    global _available, _available_checked_at
    now = time.monotonic()
    if _available is None or (not _available and now - _available_checked_at > REPROBE_SECONDS):
        _available_checked_at = now
        try:
            await retrieve_memories("__probe__", "probe", top_k=1)
            _available = True
        except Exception:
            _available = False
    return _available


async def embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embed texts with the NVIDIA embeddings API."""
    if not texts:
        return []
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{NVIDIA_API_URL}/embeddings",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": EMBED_MODEL,
                "input": texts,
                "encoding_format": "float",
                "input_type": input_type,
                "truncate": "END",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    # The API may reorder items; sort by index to be safe.
    items = sorted(data["data"], key=lambda d: d.get("index", 0))
    return [item["embedding"] for item in items]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


async def retrieve_memories(user_id: str, query: str, top_k: int = 6) -> list[dict]:
    """Find the stored memories most relevant to a query."""
    vectors = await embed([query], input_type="query")
    if not vectors:
        return []
    from storage import supabase

    result = await asyncio.to_thread(
        _match_memories,
        str(user_id),
        _vector_literal(vectors[0]),
        top_k,
    )
    return result


def _match_memories(user_id: str, embedding: str, top_k: int) -> list[dict]:
    """Call the pgvector search function. Runs in a thread (sync client)."""
    from storage import supabase

    resp = supabase.rpc(
        MATCH_FN,
        {
            "p_user_id": user_id,
            "p_embedding": embedding,
            "p_match_count": top_k,
        },
    ).execute()
    return resp.data or []


async def store_memories(user_id: str, memories: list[dict]) -> int:
    """Store memories. Each is {"content": str, "kind": str}."""
    memories = [m for m in memories if (m.get("content") or "").strip()]
    if not memories:
        return 0
    try:
        vectors = await embed([m["content"] for m in memories])
    except Exception as exc:
        print(f"could not embed memories: {exc}")
        return 0

    rows = [
        {
            "user_id": str(user_id),
            "kind": m.get("kind", "fact"),
            "content": m["content"].strip(),
            "embedding": vectors[i],
        }
        for i, m in enumerate(memories)
    ]

    from storage import supabase

    def _insert():
        supabase.table(MEMORY_TABLE).upsert(rows, on_conflict="user_id,content").execute()

    try:
        await asyncio.to_thread(_insert)
    except Exception as exc:
        print(f"could not store memories: {exc}")
        return 0
    return len(rows)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
_EXTRACT_PROMPT = (
    "Extract the durable facts from this exchange that would be useful in a much "
    "later conversation: specific figures, dates, names, decisions, stated "
    "preferences, and conclusions the user or assistant settled on.\n"
    "Rules:\n"
    "- Each memory is ONE self-contained sentence that makes sense on its own.\n"
    "- Include the actual number, date or name, not a vague reference.\n"
    "- Skip anything that is just this conversation's scaffolding, and skip "
    "speculation that was not settled.\n"
    "- If there is nothing worth remembering, return an empty list.\n"
    "Return JSON only: {\"memories\": [{\"content\": \"...\", \"kind\": \"fact\"}]}\n"
    "where kind is one of fact, decision, finding, preference."
)


async def extract_memories(text: str) -> list[dict]:
    """Use the LLM to pull durable facts out of a piece of text."""
    if not text or len(text.strip()) < 40:
        return []

    request = {
        "messages": [
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": text[:12000]},
        ],
        "temperature": 0.1,
    }

    for _name, client, model in provider_chain():
        try:
            request["model"] = model
            resp = await client.chat.completions.create(
                max_completion_tokens=1200, **request
            )
            content = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                continue
            data = json.loads(match.group(0))
            memories = data.get("memories") or []
            return [
                {"content": m["content"], "kind": m.get("kind", "fact")}
                for m in memories
                if isinstance(m, dict) and m.get("content")
            ]
        except Exception:
            continue
    return []


def format_memories(memories: list[dict]) -> str:
    """Render retrieved memories for injection into the prompt."""
    if not memories:
        return ""
    lines = []
    for m in memories:
        kind = m.get("kind", "fact")
        content = m.get("content", "").strip()
        if content:
            lines.append(f"- ({kind}) {content}")
    if not lines:
        return ""
    return (
        "\n\nRelevant things you remember from earlier conversations with this "
        "user (retrieved automatically - treat as established fact unless "
        "contradicted by something newer):\n" + "\n".join(lines)
    )
