-- Long-term semantic memory for the Discord bot.
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- Stores durable facts as atomic, embedded memories so the bot can retrieve
-- the ones relevant to a question instead of relying on one lossy summary.

create extension if not exists vector;

create table if not exists memories (
    id         bigserial primary key,
    user_id    text        not null,
    kind       text        not null default 'fact',
    content    text        not null,
    embedding  vector(2048),
    created_at timestamptz default now(),
    unique (user_id, content)
);

create index if not exists memories_user_idx on memories (user_id);

-- Cosine-similarity search over a user's memories.
create or replace function match_memories(
    p_user_id     text,
    p_embedding   text,
    p_match_count int default 6
)
returns table (id bigint, kind text, content text, score float)
language sql stable as $$
    select m.id,
           m.kind,
           m.content,
           1 - (m.embedding <=> p_embedding::vector) as score
    from memories m
    where m.user_id = p_user_id
      and m.embedding is not null
    order by m.embedding <=> p_embedding::vector
    limit p_match_count;
$$;
