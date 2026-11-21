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

# The preferred model sometimes hangs at the API's end - nemotron-3.5-embed-1b
# stopped answering altogether on 2026-11-19 - which would leave the bot with
# no memory at all. The older model is the backup: same 2048 dimensions, same
# measured retrieval quality, and it answers in about a second.
EMBED_MODEL_FALLBACK = os.getenv("EMBED_MODEL_FALLBACK", "nvidia/nemotron-3-embed-1b")

# Which model is actually in use. Settled by a probe and then kept for the
# life of the process, so every vector in the table comes from the same
# model and stays comparable.
_active_embed_model = EMBED_MODEL
_active_model_checked = False

# Models that stopped answering, and when they may be tried again.
_dropped_until: dict[str, float] = {}

# How long a model that failed is left alone before it is tried again.
MODEL_RETRY_AFTER = float(os.getenv("MEMORY_MODEL_RETRY_AFTER", "3600"))

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

# How many texts go in one embedding call.
EMBED_BATCH = int(os.getenv("MEMORY_EMBED_BATCH", "64"))

# The probe decides which model every memory in the table is compared with,
# so it is given more time than an ordinary call. The preferred model
# answers only now and then, and a too-hasty probe would give up on it.
PROBE_TIMEOUT = float(os.getenv("MEMORY_PROBE_TIMEOUT", "45"))


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


async def _probe_model(model: str) -> bool:
    """Does this embedding model answer?"""
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            resp = await client.post(
                f"{NVIDIA_API_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": ["probe"],
                    "encoding_format": "float",
                    "input_type": "passage",
                    "truncate": "END",
                },
            )
        if resp.status_code != 200:
            return False
        data = resp.json()
        vectors = data.get("data") or []
        return bool(vectors) and len(vectors[0].get("embedding") or []) == EMBED_DIM
    except Exception:
        return False


async def active_model(force: bool = False) -> str:
    """The embedding model to use, settled by a probe.

    The preferred model is tried first. If it does not answer, the backup is
    used instead. The choice is then kept, so every memory is written and
    searched with the same model.

    A model that has already been dropped is not tried again for a while.
    The preferred model is currently answering only now and then, and
    flipping back and forth would mean re-embedding every memory each time.
    """
    global _active_embed_model, _active_model_checked

    if _active_model_checked and not force:
        return _active_embed_model

    _active_model_checked = True
    now = time.monotonic()

    for candidate in (EMBED_MODEL, EMBED_MODEL_FALLBACK):
        if _dropped_until.get(candidate, 0) > now:
            # It was dropped recently; leave it alone for now.
            continue

        if await _probe_model(candidate):
            if candidate != _active_embed_model:
                print(f"memory: using the {candidate} embedding model")
            _active_embed_model = candidate
            return _active_embed_model

        print(f"memory: {candidate} did not answer")
        _dropped_until[candidate] = now + MODEL_RETRY_AFTER

    # Neither answered. Keep whatever is in use and let the retry logic in
    # embed() deal with it.
    return _active_embed_model


async def _embed_with(model: str, texts: list[str], input_type: str) -> list:
    """One embedding call, with the retries. Returns [] on failure."""
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(EMBED_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
                resp = await client.post(f"{NVIDIA_API_URL}/embeddings",
                                         headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            # The API may reorder items; sort by index to be safe.
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in items]
        except Exception:
            if attempt < EMBED_ATTEMPTS - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
    return []


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

    model = await active_model()

    if input_type == "query":
        key = f"{model}\x00{texts[0]}"
        cached = _cache_get(key)
        if cached is not None:
            return [cached]

    vectors = await _embed_with(model, texts, input_type)
    if not vectors:
        # The model stopped answering. Settle on whichever one works now and
        # try again, rather than losing memory for the whole session.
        other = await active_model(force=True)
        if other != model:
            print(f"memory: switched the embedding model to {other}")
            vectors = await _embed_with(other, texts, input_type)
            if vectors:
                # The stored memories were made with the old model and
                # cannot be compared with these, so they are re-embedded.
                asyncio.create_task(_reembed_in_background(other))

    if not vectors:
        raise RuntimeError(f"embedding failed after {EMBED_ATTEMPTS} tries "
                           f"with {model}")

    if len(vectors[0]) != EMBED_DIM:
        raise RuntimeError(
            f"{model} returned {len(vectors[0])}-dimension vectors but the "
            f"table is vector({EMBED_DIM}). Change EMBED_MODEL and EMBED_DIM "
            f"together, and re-embed the stored memories."
        )

    if input_type == "query":
        _cache_put(key, vectors[0])
    return vectors


async def _reembed_in_background(model: str):
    """Re-embed the stored memories after the model changed."""
    try:
        count = await reembed_stale_memories()
        if count:
            print(f"memory: re-embedded {count} memories for {model}")
    except Exception as exc:
        print(f"memory: re-embed failed ({exc})")


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


def _norm_content(content: str) -> str:
    """Normalise a memory so two copies of it compare equal."""
    return " ".join(re.findall(r"\w+", (content or "").lower()))


async def retrieve_memories(user_id: str, query: str, top_k: int = 6,
                            min_score: float = None,
                            scopes: list[str] = None) -> list[dict]:
    """Find the stored memories most relevant to a query.

    Results below the relevance floor are dropped: a memory that has nothing
    to do with the question does not belong in the prompt, where the model
    would treat it as established fact.

    Revisions are resolved by the search function, which keeps the newer of
    two memories that are about the same thing but say different things.

    `scopes` are the keys to search. A shared channel passes both the
    person's own key and the channel's key, so the bot remembers what
    everyone in the room has been talking about. The same memory stored
    under two keys is returned once.
    """
    scopes = [str(s) for s in (scopes or [user_id]) if s]
    vectors = await embed([query], input_type="query")
    if not vectors:
        return []
    from storage import supabase

    model = await active_model()
    literal = _vector_literal(vectors[0])

    results = []
    for scope in scopes:
        try:
            found = await asyncio.to_thread(
                _match_memories, scope, literal, top_k, model
            )
        except Exception as exc:
            print(f"memory search failed for scope {scope}: {exc}")
            continue
        results.extend(found or [])

    if min_score is None:
        min_score = MIN_RELEVANCE

    kept = []
    seen_content = set()
    for row in results:
        similarity = row.get("similarity")
        if similarity is None:
            similarity = row.get("score") or 0.0
        if similarity < min_score:
            continue
        # The same memory can live under more than one scope (a shared
        # channel stores it under the person's key and the channel's key).
        key = _norm_content(row.get("content"))
        if key in seen_content:
            continue
        seen_content.add(key)
        row["similarity"] = similarity
        kept.append(row)

    kept.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
    kept = kept[:top_k]

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


def _match_memories(user_id: str, embedding: str, top_k: int,
                    model: str = "") -> list[dict]:
    """Call the pgvector search function. Runs in a thread (sync client)."""
    from storage import supabase

    resp = supabase.rpc(
        MATCH_FN,
        {
            "p_user_id": user_id,
            "p_embedding": embedding,
            "p_match_count": top_k,
            "p_model": model or EMBED_MODEL,
            "p_recency_weight": RECENCY_WEIGHT,
            "p_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "p_dedupe_distance": DEDUPE_DISTANCE,
        },
    ).execute()
    return resp.data or []


async def store_memories(user_id: str, memories: list[dict],
                         source: str = None,
                         scopes: list[str] = None) -> int:
    """Store memories. Each is {"content": str, "kind": str}.

    `scopes` are the keys to write to. A shared channel passes both the
    person's own key and the channel's key, so a fact learned in a room is
    still known in that room even when somebody else asks about it later.
    """
    memories = [m for m in memories if (m.get("content") or "").strip()]
    if not memories:
        return 0
    scopes = [str(s) for s in (scopes or [user_id]) if s]
    model = await active_model()
    try:
        vectors = await embed([m["content"] for m in memories])
    except Exception as exc:
        print(f"could not embed memories: {exc}")
        return 0

    rows = []
    for scope in scopes:
        for i, m in enumerate(memories):
            rows.append({
                "user_id": scope,
                "kind": m.get("kind", "fact"),
                "content": m["content"].strip(),
                "embedding": vectors[i],
                "embed_model": model,
                "embed_dim": EMBED_DIM,
                "source": source,
            })

    from storage import supabase

    def _insert():
        supabase.table(MEMORY_TABLE).upsert(rows, on_conflict="user_id,content").execute()

    try:
        await asyncio.to_thread(_insert)
    except Exception as exc:
        print(f"could not store memories: {exc}")
        return 0

    for scope in scopes:
        await prune_memories(scope)
    return len(memories)


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

    model = await active_model()
    try:
        query = (supabase.table(MEMORY_TABLE)
                 .select("id,user_id,kind,content,source")
                 .neq("embed_model", model))
        if user_id:
            query = query.eq("user_id", str(user_id))
        rows = query.execute().data or []
    except Exception as exc:
        print(f"could not list stale memories: {exc}")
        return 0
    if not rows:
        return 0

    try:
        # The API takes a limited number of texts at a time, so the rows go
        # through in batches.
        vectors = []
        for start in range(0, len(rows), EMBED_BATCH):
            chunk = await embed([r["content"] for r in rows[start:start + EMBED_BATCH]])
            if not chunk:
                raise RuntimeError(f"the batch at {start} did not embed")
            vectors.extend(chunk)
    except Exception as exc:
        print(f"could not embed stale memories: {exc}")
        return 0

    def _update():
        # One request per row would be hundreds of round trips, so the rows
        # are written with a single upsert.
        payload = []
        for row, vector in zip(rows, vectors):
            payload.append({
                "id": row["id"],
                "user_id": row.get("user_id"),
                "kind": row.get("kind") or "fact",
                "content": row["content"],
                "embedding": vector,
                "embed_model": model,
                "embed_dim": EMBED_DIM,
                "source": row.get("source"),
            })
        for start in range(0, len(payload), 50):
            supabase.table(MEMORY_TABLE).upsert(
                payload[start:start + 50], on_conflict="id").execute()

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
    "Extract the durable facts about the USER from this exchange that would be "
    "useful in a much later conversation: who they are, what they are working "
    "on, their preferences, their decisions, their plans, the numbers and dates "
    "that concern them, and the people in their life.\n"
    "Rules:\n"
    "- Each memory is ONE self-contained sentence that makes sense on its own.\n"
    "- Include the actual number, date or name, not a vague reference.\n"
    "- If the exchange updates something stated earlier, write the NEW value.\n"
    "- Mark every candidate with \"about_user\": true ONLY if it is a fact about "
    "the person who is talking. Everything else must be false.\n"
    "- Write each memory so that it is obviously about that person: start it "
    "with \"The user\" or use their name. A sentence that names no person at "
    "all will be thrown away.\n"    "- A fact about the world is NOT a fact about the user. General knowledge, "
    "research findings, definitions, homework solutions, history, product facts "
    "and statistics are all about_user: false, even when the user asked for "
    "them and even when the answer was good.\n"
    "- NEVER store facts about whether a tool, service, API, credential, "
    "environment or network is available, working or configured (Kaggle, the "
    "sandbox, internet access, tokens). Those states change between deploys, "
    "so a remembered version goes stale and makes the assistant refuse tools "
    "that actually work. Report live status in the conversation instead.\n"
    "- NEVER store what the assistant said about itself: what it can do, what "
    "it refused, what it offered to do, or how it works.\n"
    "- Do not store what the assistant offered or attempted in a past "
    "conversation unless the user settled on it as a decision.\n"
    "- If there is nothing worth remembering, return an empty list.\n"
    "Return JSON only: {\"memories\": [{\"content\": \"...\", \"kind\": \"fact\", "
    "\"about_user\": true}]}\n"
    "where kind is one of fact, decision, finding, preference."
)

# ---------------------------------------------------------------------------
# What must never become a memory
# ---------------------------------------------------------------------------
#
# The store used to be full of the assistant's own research output: Vietnamese
# inflation figures, marriage-law dates, proof steps. Those are facts about
# the world, not about the user, and they crowded out the handful of real
# facts ("the user is an applied math major") in every retrieval. Three
# patterns are rejected outright.

# The assistant describing itself: what it can do, refused, offered, said.
# Anchored at the start of the sentence, so that "the user's model is X" is
# not mistaken for the assistant talking about itself.
_ASSISTANT_SELF_RE = re.compile(
    r"^\s*(?:the\s+|this\s+|that\s+|an?\s+)?"
    r"(?:assistant|bot|ai\s+assistant|agent|model)\s+"
    r"(?:can|could|will|would|should|does|did|is|was|has|have|refuses?|refused|"
    r"offered|offers?|provides?|provided|said|told|explained|listed|cited|"
    r"answered|decided)\b",
    re.IGNORECASE,
)

# The assistant describing what it can do, in the first person.
_SELF_CAPABILITY_RE = re.compile(
    r"^\s*(?:i|we)\s+(?:can|could|will|would|should|must|offered|refused|"
    r"ran|used|called|checked|explained|suggested|recommended|listed|"
    r"provided)\b",
    re.IGNORECASE,
)

# The state of a tool, service or credential. These go stale between deploys.
_TOOL_STATE_RE = re.compile(
    r"\b(?:kaggle|sandbox|daytona|piston|api|token|credential|network|internet|"
    r"web[_ ]?search|wolfram|supabase|hugging\s?face|hub|code\s+tool|tools?)\b"
    r"[^.]{0,80}?"
    r"\b(?:is|are|was|were|be|been|being|works?|working|blocked|available|"
    r"configured|installed|reachable|usable|functional|broken|down|expired|"
    r"valid|unavailable|cannot|can't|could\s+not|couldn't|does\s+not|doesn't)\b",
    re.IGNORECASE,
)

# A sentence is about the user only when it names them. "The user is an
# applied math major" qualifies; "Pao travels at 5 times the plane's speed"
# and "For i.i.d. random variables..." do not, however useful they are.
_USER_SIGNAL_RE = re.compile(
    r"\b(?:the\s+user|damian|the\s+creator)\b"
    r"|^\s*(?:i|my|me)\b",
    re.IGNORECASE,
)


def _is_about_user(candidate: dict) -> bool:
    """Should this extracted candidate be stored as a memory about the user?"""
    content = (candidate.get("content") or "").strip()
    if not content:
        return False
    if _ASSISTANT_SELF_RE.search(content):
        return False
    if _SELF_CAPABILITY_RE.search(content):
        return False
    if _TOOL_STATE_RE.search(content):
        return False

    flag = candidate.get("about_user")
    if isinstance(flag, bool) and not flag:
        return False
    if isinstance(flag, str) and flag.strip().lower() in ("false", "no", "0"):
        return False

    # Whatever the extractor said, the sentence itself has to be about a
    # person. "The user is an applied math major" qualifies; "Vietnam's CPI
    # was +5.09% in October 2026" does not, however confident the extractor
    # was about it.
    return bool(_USER_SIGNAL_RE.search(content))


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
            kept = []
            for m in memories:
                if not isinstance(m, dict) or not m.get("content"):
                    continue
                if not _is_about_user(m):
                    continue
                kept.append({
                    "content": m["content"].strip(),
                    "kind": m.get("kind", "fact"),
                })
            return kept
        except Exception:
            continue
    return []


def format_memories(memories: list[dict], shared: bool = False) -> str:
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
    who = ("these people" if shared else "this user")
    return (
        "\n\nRelevant things you remember from earlier conversations with "
        f"{who} (retrieved automatically - treat as established fact unless "
        "contradicted by something newer):\n" + "\n".join(lines)
    )
