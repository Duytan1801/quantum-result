-- Long-term semantic memory for the Discord bot.
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- Stores durable facts as atomic, embedded memories so the bot can retrieve
-- the ones relevant to a question instead of relying on one lossy summary.
--
-- The schema is deliberately version-aware: every row records which embedding
-- model produced its vector. If the model is ever changed, the old rows are
-- identified (rather than silently mixed into the same vector space) and can
-- be re-embedded.

create extension if not exists vector;

create table if not exists memories (
    id           bigserial primary key,
    user_id      text        not null,
    kind         text        not null default 'fact',
    content      text        not null,

    -- The vector, and the model that produced it. Both matter: vectors from
    -- different models are not comparable, so retrieval filters on the model.
    embedding    vector(2048),
    embed_model  text        not null default 'nvidia/nemotron-3.5-embed-1b',
    embed_dim    int         not null default 2048,

    -- Where the memory came from, for debugging and for pruning.
    source       text,

    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),

    -- Usage bookkeeping: a memory that keeps getting retrieved is one the
    -- user actually cares about, and one that never gets used is a candidate
    -- for pruning.
    last_used_at timestamptz,
    use_count    int         not null default 0,

    unique (user_id, content)
);

create index if not exists memories_user_idx on memories (user_id);
create index if not exists memories_user_model_idx on memories (user_id, embed_model);
create index if not exists memories_created_idx on memories (user_id, created_at desc);

-- Cosine search over a user's memories, biased towards recent ones.
--
-- The score blends similarity with recency, so a memory from this morning
-- ("I now work at X") outranks a fresher-looking but stale one ("I worked at
-- Y"). The raw similarity is returned too, because the caller decides what
-- counts as relevant - the score itself mixes two different things.
--
-- Memories that are about the same thing but say different things are
-- revisions of each other ("I work at Initech" / "I work at Globex"), and
-- only the newer one is kept. The comparison is between the memories
-- themselves, not their scores against the query: measured on real
-- memories, a revision pair sits at cosine similarity 0.83 while two
-- unrelated facts sit at 0.63, so a distance of 0.25 separates them.
create or replace function match_memories(
    p_user_id        text,
    p_embedding      text,
    p_match_count    int  default 6,
    p_model          text default null,
    p_recency_weight real default 0.15,
    p_half_life_days real default 90,
    p_dedupe_distance real default 0.25
)
returns table (
    id         bigint,
    kind       text,
    content    text,
    score      float,
    similarity float,
    created_at timestamptz,
    age_days   float
)
language sql stable as $$
    with scored as (
        select m.id,
               m.kind,
               m.content,
               m.created_at,
               m.embedding,
               1 - (m.embedding <=> p_embedding::vector) as similarity,
               extract(epoch from (now() - m.created_at)) / 86400.0 as age_days
        from memories m
        where m.user_id = p_user_id
          and m.embedding is not null
          and (p_model is null or m.embed_model = p_model)
    ),
    ranked as (
        select s.id,
               s.kind,
               s.content,
               s.created_at,
               s.embedding,
               s.similarity,
               s.age_days,
               (s.similarity
                + p_recency_weight
                  * exp(-ln(2) * s.age_days / greatest(p_half_life_days, 1)))::float as score,
               row_number() over (
                   order by (s.similarity
                             + p_recency_weight
                               * exp(-ln(2) * s.age_days / greatest(p_half_life_days, 1))) desc
               ) as rn
        from scored s
    )
    select r.id,
           r.kind,
           r.content,
           r.score,
           r.similarity::float,
           r.created_at,
           r.age_days::float
    from ranked r
    where not exists (
        -- A newer memory that is about the same thing supersedes this one.
        select 1
        from ranked newer
        where newer.rn < r.rn
          and newer.content <> r.content
          and (newer.embedding <=> r.embedding) < p_dedupe_distance
    )
    order by r.score desc
    limit p_match_count;
$$;

-- Mark the memories that were just used, so pruning can tell the useful
-- ones from the rest.
create or replace function touch_memories(p_ids bigint[])
returns void
language sql as $$
    update memories
    set last_used_at = now(),
        use_count = use_count + 1
    where id = any(p_ids);
$$;

-- How many memories each model has, so a model change is visible.
create or replace function memory_model_counts()
returns table (embed_model text, rows bigint)
language sql stable as $$
    select m.embed_model, count(*)::bigint
    from memories m
    group by m.embed_model;
$$;

-- Optional: once there are enough rows for it to help, an approximate index
-- speeds up retrieval. It cannot be created on an empty table (ivfflat needs
-- rows to cluster), so run it later by hand if the table grows large:
--
--   create index memories_vec_idx on memories
--     using ivfflat (embedding vector_cosine_ops) with (lists = 100);
--
-- For a single user with a few thousand rows, exact search is fast enough.
