"""Supabase persistence: conversation history and remembered user facts.

NOTE: user_id is a text column, so a PostgREST filter like
``user_id=like.__*`` treats ``_`` as a single-character wildcard and matches
EVERY user. Always filter with an exact ``eq.`` value, never a LIKE pattern.
"""

import asyncio

from config import HISTORY_ROWS, supabase


async def db_is_chat_mode_enabled(channel_id: str) -> bool:
    """Check if chat mode is enabled for a channel."""
    def _fetch():
        try:
            resp = (
                supabase.table("chat_mode")
                .select("enabled")
                .eq("channel_id", str(channel_id))
                .execute()
            )
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return False
            raise
        return bool(resp.data and resp.data[0].get("enabled"))
    return await asyncio.to_thread(_fetch)


async def db_set_chat_mode(channel_id: str, enabled: bool):
    """Enable or disable chat mode for a channel."""
    def _upsert():
        try:
            supabase.table("chat_mode").upsert({
                "channel_id": str(channel_id),
                "enabled": enabled,
            }, on_conflict="channel_id").execute()
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return
            raise
    await asyncio.to_thread(_upsert)


# The proactive switch is another row in the same table.
#
# chat_mode is already a per-channel settings table, and a second column
# would mean a schema change - and the schema can only be changed by
# hand in the Supabase editor. So the switch is a second row, keyed with
# a prefix that can never be a channel id. The prefix is filtered out
# everywhere the channel list is read, so a row here can never be
# mistaken for a room.
PROACTIVE_PREFIX = "proactive:"


async def db_set_proactive_mode(channel_id: str, enabled: bool):
    """Switch proactive talking on or off for a room."""
    def _upsert():
        try:
            supabase.table("chat_mode").upsert({
                "channel_id": f"{PROACTIVE_PREFIX}{channel_id}",
                "enabled": bool(enabled),
            }, on_conflict="channel_id").execute()
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return
            raise
    await asyncio.to_thread(_upsert)


async def db_is_proactive_enabled(channel_id: str, default: bool = True) -> bool:
    """Is proactive talking switched on for this room?

    On unless somebody has turned it off. A row that cannot be read is
    an error, not an answer: the caller is told, so it can stay quiet
    rather than talk when it was told not to.
    """
    def _fetch():
        resp = (
            supabase.table("chat_mode")
            .select("enabled")
            .eq("channel_id", f"{PROACTIVE_PREFIX}{channel_id}")
            .limit(1)
            .execute()
        )
        return resp.data or []

    try:
        rows = await asyncio.to_thread(_fetch)
    except Exception as e:
        err = str(e).lower()
        if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
            return default
        raise

    if not rows:
        return default
    return bool(rows[0].get("enabled"))


async def db_list_chat_channels() -> list[str]:
    """The channels with chat mode switched on.

    The proactive tick looks at these: a channel with chat mode off is a
    channel where the bot stays quiet, and that is true whether or not
    anybody is talking to it.
    """
    def _fetch():
        try:
            resp = (
                supabase.table("chat_mode")
                .select("channel_id")
                .eq("enabled", True)
                .execute()
            )
        except Exception as e:
            err = str(e).lower()
            if "42703" in err or "column" in err or "404" in err or "table" in err or "pgrst205" in err:
                return []
            raise
        return [str(r.get("channel_id")) for r in (resp.data or [])
                if r.get("channel_id")
                and not str(r.get("channel_id")).startswith(PROACTIVE_PREFIX)]
    return await asyncio.to_thread(_fetch)


async def db_get_history(user_id, limit=HISTORY_ROWS):
    """Fetch the most recent conversation rows for a user, oldest first."""
    def _fetch():
        try:
            resp = (
                supabase.table("conversation_history")
                .select("role, content, tool_calls, tool_call_id, name, created_at")
                .eq("user_id", str(user_id))
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                resp = (
                    supabase.table("conversation_history")
                    .select("role, content, created_at")
                    .eq("user_id", str(user_id))
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
            else:
                raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return list(reversed(rows))


async def db_get_history_multi(user_ids, limit=HISTORY_ROWS):
    """Fetch the most recent conversation rows across several scopes.

    A channel with more than one person uses two scopes: the person's own
    rows and the channel's shared rows. They are merged and sorted by time,
    so the bot reads one continuous thread rather than two half-histories.
    """
    ids = [str(u) for u in (user_ids or []) if u is not None]
    if not ids:
        return []
    if len(ids) == 1:
        return await db_get_history(ids[0], limit)

    def _fetch():
        try:
            resp = (
                supabase.table("conversation_history")
                .select("role, content, tool_calls, tool_call_id, name, created_at")
                .in_("user_id", ids)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                resp = (
                    supabase.table("conversation_history")
                    .select("role, content, created_at")
                    .in_("user_id", ids)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
            else:
                raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return list(reversed(rows))


async def db_get_user_facts(user_id):
    """All remembered facts for a user, as a dict of key -> value."""
    def _fetch():
        try:
            resp = (
                supabase.table("user_facts")
                .select("fact_key, fact_value")
                .eq("user_id", str(user_id))
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower() or "404" in str(e):
                return []
            raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return {r["fact_key"]: r["fact_value"] for r in rows}


async def db_scope_has_rows(scope: str) -> bool:
    """Does this scope already have conversation rows stored under it?

    Used to work out whether a channel has been a room before. The tracker in
    memory only knows who has spoken since the bot started; after a restart
    it would think a busy channel was a private chat until somebody said
    something.
    """
    def _fetch():
        try:
            resp = (
                supabase.table("conversation_history")
                .select("id")
                .eq("user_id", str(scope))
                .limit(1)
                .execute()
            )
        except Exception:
            return False
        return bool(resp.data)
    return await asyncio.to_thread(_fetch)


async def db_all_scopes() -> list[str]:
    """Every scope the bot has ever written to.

    A detached room reads the whole picture, so it needs to know every key
    the memory is stored under: one per person, one per room. The tables
    are read one column at a time and paged through, so this costs a few
    small queries rather than a scan of the conversation itself.
    """
    scopes: set[str] = set()

    for table in ("conversation_history", "memories", "user_facts"):
        def _fetch(table=table):
            found: set[str] = set()
            last = 0
            while True:
                resp = (
                    supabase.table(table)
                    .select("id,user_id")
                    .gt("id", last)
                    .order("id")
                    .limit(1000)
                    .execute()
                )
                rows = resp.data or []
                if not rows:
                    break
                for row in rows:
                    value = row.get("user_id")
                    if value:
                        found.add(str(value))
                last = rows[-1]["id"]
                if len(rows) < 1000:
                    break
            return found

        try:
            scopes |= await asyncio.to_thread(_fetch)
        except Exception:
            # A table without an id column, or one that is not there yet,
            # simply does not contribute scopes.
            continue

    return sorted(scopes)


async def db_get_user_facts_multi(user_ids) -> dict:
    """All remembered facts across several scopes, as key -> value."""
    ids = [str(u) for u in (user_ids or []) if u is not None]
    if not ids:
        return {}
    if len(ids) == 1:
        return await db_get_user_facts(ids[0])

    def _fetch():
        try:
            resp = (
                supabase.table("user_facts")
                .select("fact_key, fact_value")
                .in_("user_id", ids)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower() or "404" in str(e):
                return []
            raise
        return resp.data or []
    rows = await asyncio.to_thread(_fetch)
    return {r["fact_key"]: r["fact_value"] for r in rows}


async def db_get_fact(user_id, key):
    """Return a single stored fact value, or None if it does not exist."""
    facts = await db_get_user_facts(user_id)
    return facts.get(key)


async def db_add_message(user_id, role, content, tool_calls=None, tool_call_id=None, name=None):
    """Append one row to the conversation history."""
    content = content or ""

    def _insert():
        payload = {
            "user_id": str(user_id),
            "role": role,
            "content": content,
        }
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        if tool_call_id is not None:
            payload["tool_call_id"] = tool_call_id
        if name is not None:
            payload["name"] = name
        try:
            supabase.table("conversation_history").insert(payload).execute()
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                fallback_payload = {
                    "user_id": str(user_id),
                    "role": role,
                    "content": content,
                }
                supabase.table("conversation_history").insert(fallback_payload).execute()
            else:
                raise
    await asyncio.to_thread(_insert)


async def db_set_fact(user_id, key, value):
    """Store (or overwrite) one remembered fact."""
    def _upsert():
        supabase.table("user_facts").upsert({
            "user_id": str(user_id),
            "fact_key": key,
            "fact_value": value,
        }, on_conflict="user_id,fact_key").execute()
    await asyncio.to_thread(_upsert)


async def db_delete_fact(user_id, key):
    """Forget one remembered fact."""
    def _delete():
        supabase.table("user_facts").delete().eq("user_id", str(user_id)).eq("fact_key", key).execute()
    await asyncio.to_thread(_delete)
