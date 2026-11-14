"""Semantic long-term memory.

SimpleMem's core idea - store atomic memories and retrieve the ones relevant to
the current question, instead of summarising the whole conversation into one
lossy blob - is implemented here against the stack the bot already has:
Supabase for storage and the NVIDIA embeddings API for vectors.

No heavy dependencies: no torch, no LanceDB, no local vector store.

The architecture, end to end:

  extract      An LLM turns each exchange into atomic, self-contained facts.
               Only exchanges that could contain something durable get here -
               a greeting does not cost an LLM call.
  embed        Every memory is embedded with the NVIDIA API. The model and its
               dimension are recorded with the row, so vectors from different
               models are never silently compared.
  store        Memories are upserted on (user_id, content), so the same fact
               learned twice is stored once.
  retrieve     The question is embedded and the nearest memories are found by
               cosine similarity, biased towards recent ones. A relevance
               floor keeps unrelated memories out of the prompt.
  inject       The memories that survive are handed to the model as
               established context for the current turn.

Run `supabase_memory.sql` in the Supabase SQL editor first - until the table
exists, the bot detects that and runs without long-term memory.
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

# Relevance floor for injecting a memory into the prompt.
#
# The embeddings API embeds queries and passages with different prefixes, so
# cosine similarity between them runs lower than a plain "query vs query"
# number. Measured over 15 real questions against real memories, correct
# matches scored 0.33-0.68 (median 0.58) while questions with no matching
# memory topped out at 0.14. A floor of 0.15 separates the two cleanly, so
# irrelevant memories are not pasted into the prompt and mistaken for fact.
MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_SCORE", "0.15"))

# How much a recent memory is favoured over an old one. 0.15 means a memory
# from today scores 0.15 higher than the same memory from a year ago.
RECENCY_WEIGHT = float(os.getenv("MEMORY_RECENCY_WEIGHT", "0.15"))

# After this many days a memory's recency bonus has halved.
RECENCY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "90"))

# Cosine distance below which two memories are treated as revisions of each
# other (the newer one wins). Measured on real memories: a revision pair sits
# at similarity 0.83, two unrelated facts at 0.63.
DEDUPE_DISTANCE = float(os.getenv("MEMORY_DEDUPE_DISTANCE", "0.25"))

# How many memories to keep per user. The store is pruned when it grows
# past this, dropping the least useful rows first.
MEMORY_CAP = int(os.getenv("MEMORY_CAP", "2000"))

# Where the memories live is optional - if the table is missing the bot just
# runs without long-term memory.
_available = None
_available_checked_at = 0.0

# Re-probe this often if the table is missing, so the bot picks it up after
# the SQL has been run without needing a restart.
REPROBE_SECONDS = 300

# Queries are often repeated ("what did I ask you to simulate?"), and each
# embedding costs about half a second. A small cache makes the repeat free.
_QUERY_CACHE: dict[str, list[float]] = {}
_QUERY_CACHE_MAX = 256

# The embeddings API is occasionally slow. A call is retried rather than
# being allowed to fail the turn, but it must not hang for a minute either.
EMBED_TIMEOUT = float(os.getenv("MEMORY_EMBED_TIMEOUT", "20"))
EMBED_ATTEMPTS = int(os.getenv("MEMORY_EMBED_ATTEMPTS", "3"))


def _cache_get(key: str):
    return _QUERY_CACHE.get(key)


def _cache_put(key: str, value: list[float]):
    if len(_QUERY_CACHE) >= _QUERY_CACHE_MAX:
        # Drop the oldest entry; dicts keep insertion order.
        _QUERY_CACHE.pop(next(iter(_QUERY_CACHE)))
    _QUERY_CACHE[key] = value


async def is_available() -> bool:
    """Is the memories table set up? Cached, but re-probed periodically."""
    global _available, _available_checked_at
    now = time.monotonic()
    if _available is None or (not _available and now - _available_checked_at > REPROBE_SECONDS):
        _available_checked_at = now
        _available = await asyncio.to_thread(_table_exists)
    return _available


def _table_exists() -> bool:
    """Cheap check that the memories table is there. No embedding call."""
    from storage import supabase

    try:
        supabase.table(MEMORY_TABLE).select("id", count="exact").limit(1).execute()
        return True
    except Exception:
        return False


async def embed(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embed texts with the NVIDIA embeddings API.

    Queries are cached: the same question asked twice only pays for the
    embedding once. The API is occasionally slow, so a call is retried
    rather than being allowed to fail the turn.
    """
    if not texts:
        return []
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    if input_type == "query":
        key = f"{EMBED_MODEL}\x00{texts[0]}"
        cached = _cache_get(key)
        if cached is not None:
            return [cached]

    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    vectors = []
    for attempt in range(EMBED_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
                resp = await client.post(f"{NVIDIA_API_URL}/embeddings",
                                         headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            # The API may reorder items; sort by index to be safe.
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors = [item["embedding"] for item in items]
            break
        except Exception as exc:
            last_error = exc
            if attempt < EMBED_ATTEMPTS - 1:
                await asyncio.sleep(1.5 * (attempt + 1))

    if not vectors:
        raise RuntimeError(f"embedding failed after {EMBED_ATTEMPTS} tries: {last_error}")

    if len(vectors[0]) != EMBED_DIM:
        raise RuntimeError(
            f"{EMBED_MODEL} returned {len(vectors[0])}-dimension vectors but the "
            f"table is vector({EMBED_DIM}). Change EMBED_MODEL and EMBED_DIM "
            f"together, and re-embed the stored memories."
        )

    if input_type == "query":
        _cache_put(key, vectors[0])
    return vectors


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


async def retrieve_memories(user_id: str, query: str, top_k: int = 6,
                            min_score: float = None) -> list[dict]:
    """Find the stored memories most relevant to a query.

    Results below the relevance floor are dropped: a memory that has nothing
    to do with the question does not belong in the prompt, where the model
    would treat it as established fact.

    Revisions are resolved by the search function, which keeps the newer of
    two memories that are about the same thing but say different things.
    """
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
    if min_score is None:
        min_score = MIN_RELEVANCE

    kept = []
    for row in result:
        similarity = row.get("similarity")
        if similarity is None:
            similarity = row.get("score") or 0.0
        if similarity < min_score:
            continue
        row["similarity"] = similarity
        kept.append(row)

    if kept:
        _touch([r["id"] for r in kept if r.get("id")])
    return kept


def _touch(ids: list) -> None:
    """Record that these memories were used. Never worth failing a turn over."""
    if not ids:
        return
    try:
        from storage import supabase

        supabase.rpc("touch_memories", {"p_ids": ids}).execute()
    except Exception:
        pass


def _match_memories(user_id: str, embedding: str, top_k: int) -> list[dict]:
    """Call the pgvector search function. Runs in a thread (sync client)."""
    from storage import supabase

    resp = supabase.rpc(
        MATCH_FN,
        {
            "p_user_id": user_id,
            "p_embedding": embedding,
            "p_match_count": top_k,
            "p_model": EMBED_MODEL,
            "p_recency_weight": RECENCY_WEIGHT,
            "p_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "p_dedupe_distance": DEDUPE_DISTANCE,
        },
    ).execute()
    return resp.data or []


async def store_memories(user_id: str, memories: list[dict],
                         source: str = None) -> int:
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
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "source": source,
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

    await prune_memories(user_id)
    return len(rows)


async def forget_memories(user_id: str, text: str) -> int:
    """Delete the user's memories that mention `text`.

    This is the way a user gets rid of something the bot remembered about
    them. Matching is on the text, case-insensitively, so "forget tokyo"
    removes the memories that talk about Tokyo.
    """
    text = (text or "").strip()
    if not text:
        return 0
    from storage import supabase

    try:
        rows = (supabase.table(MEMORY_TABLE)
                .select("id,content")
                .eq("user_id", str(user_id))
                .ilike("content", f"%{text}%")
                .execute().data or [])
    except Exception as exc:
        print(f"could not find memories to forget: {exc}")
        return 0

    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    try:
        await asyncio.to_thread(
            lambda: supabase.table(MEMORY_TABLE).delete().in_("id", ids).execute()
        )
    except Exception as exc:
        print(f"could not forget memories: {exc}")
        return 0
    return len(ids)


async def prune_memories(user_id: str, cap: int = None) -> int:
    """Keep the store within its cap, dropping the least useful memories.

    A memory is worth keeping if it has been used before, or if it is recent.
    The rest go, oldest and least used first.
    """
    cap = cap or MEMORY_CAP
    from storage import supabase

    try:
        rows = (supabase.table(MEMORY_TABLE)
                .select("id,use_count,last_used_at,created_at")
                .eq("user_id", str(user_id))
                .execute().data or [])
    except Exception:
        return 0

    if len(rows) <= cap:
        return 0

    def rank(row):
        used = row.get("last_used_at")
        # Memories that have been retrieved before are worth keeping.
        return (1 if used else 0, row.get("use_count") or 0, str(row.get("created_at") or ""))

    rows.sort(key=rank)
    doomed = [r["id"] for r in rows[:len(rows) - cap]]
    if not doomed:
        return 0
    try:
        await asyncio.to_thread(
            lambda: supabase.table(MEMORY_TABLE).delete().in_("id", doomed).execute()
        )
    except Exception as exc:
        print(f"could not prune memories: {exc}")
        return 0
    return len(doomed)


async def reembed_stale_memories(user_id: str = None) -> int:
    """Re-embed memories that were stored with a different embedding model.

    Vectors from different models are not comparable, so a model change
    leaves the old rows useless. This re-embeds them in place.
    """
    from storage import supabase

    try:
        query = (supabase.table(MEMORY_TABLE)
                 .select("id,content")
                 .neq("embed_model", EMBED_MODEL))
        if user_id:
            query = query.eq("user_id", str(user_id))
        rows = query.execute().data or []
    except Exception as exc:
        print(f"could not list stale memories: {exc}")
        return 0
    if not rows:
        return 0

    try:
        vectors = await embed([r["content"] for r in rows])
    except Exception as exc:
        print(f"could not embed stale memories: {exc}")
        return 0

    def _update():
        for row, vector in zip(rows, vectors):
            supabase.table(MEMORY_TABLE).update(
                {"embedding": vector, "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM}
            ).eq("id", row["id"]).execute()

    try:
        await asyncio.to_thread(_update)
    except Exception as exc:
        print(f"could not re-embed memories: {exc}")
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
    "- If the exchange updates something stated earlier, write the NEW value.\n"
    "- Skip anything that is just this conversation's scaffolding, and skip "
    "speculation that was not settled.\n"
    "- If there is nothing worth remembering, return an empty list.\n"
    "Return JSON only: {\"memories\": [{\"content\": \"...\", \"kind\": \"fact\"}]}\n"
    "where kind is one of fact, decision, finding, preference."
)

# Conversations too small to hold a durable fact. Extracting from these costs
# an LLM call and an embedding call for nothing.
_MIN_EXCHANGE_CHARS = 60

# Replies that are pure acknowledgements.
_ACK_ONLY = re.compile(
    r"^\s*(hi|hey|hello|yo|ok|okay|k|kk|sure|thanks|ty|thx|yes|yeah|yep|no|nope|"
    r"good|great|nice|cool|got it|sounds good|np|welcome|bye|lol|haha|hmm+|"
    r"done|perfect|right|true|exactly|agreed|will do|on it)\s*[.!]*\s*$",
    re.IGNORECASE,
)


def worth_extracting(user_message: str, answer: str) -> bool:
    """Is this exchange worth an extraction pass?

    A greeting, an acknowledgement or a one-word answer does not contain a
    durable fact, and running the extractor over it is a wasted LLM call on
    every single message.
    """
    user_message = (user_message or "").strip()
    answer = (answer or "").strip()
    if not user_message and not answer:
        return False
    if len(user_message) + len(answer) < _MIN_EXCHANGE_CHARS:
        return False
    if user_message and _ACK_ONLY.match(user_message):
        return False
    # A long answer to a short question usually contains the fact.
    if len(user_message) < 25 and len(answer) < _MIN_EXCHANGE_CHARS:
        return False
    return True


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
