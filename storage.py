"""Supabase persistence: conversation history and remembered user facts.

NOTE: user_id is a text column, so a PostgREST filter like
``user_id=like.__*`` treats ``_`` as a single-character wildcard and matches
EVERY user. Always filter with an exact ``eq.`` value, never a LIKE pattern.
"""

import asyncio

from config import MAX_HISTORY, supabase


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


async def db_get_history(user_id, limit=MAX_HISTORY * 6):
    """Fetch the most recent conversation rows for a user, oldest first."""
    def _fetch():
        try:
            resp = (
                supabase.table("conversation_history")
                .select("role, content, tool_calls, tool_call_id, name")
                .eq("user_id", str(user_id))
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower():
                resp = (
                    supabase.table("conversation_history")
                    .select("role, content")
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
