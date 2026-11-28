"""Supabase persistence: conversation history and remembered user facts.

NOTE: user_id is a text column, so a PostgREST filter like
``user_id=like.__*`` treats ``_`` as a single-character wildcard and matches
EVERY user. Always filter with an exact ``eq.`` value, never a LIKE pattern.
"""

import asyncio
import uuid

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


async def db_get_facts_by_scope(scopes) -> dict:
    """The facts of several scopes at once, as scope -> {key: value}.

    The flat version of this (db_get_user_facts_multi) throws the scope
    away, which is fine for the memory and useless for the schedules:
    every schedule has the same keys, so without the scope there is no
    way to tell whose "message" is whose.
    """
    ids = [str(s) for s in (scopes or []) if s is not None]
    if not ids:
        return {}

    def _fetch():
        try:
            resp = (
                supabase.table("user_facts")
                .select("user_id, fact_key, fact_value")
                .in_("user_id", ids)
                .execute()
            )
        except Exception as e:
            if "42703" in str(e) or "column" in str(e).lower() or "404" in str(e):
                return []
            raise
        return resp.data or []

    rows = await asyncio.to_thread(_fetch)
    out: dict[str, dict] = {}
    for row in rows or []:
        scope = str(row.get("user_id") or "")
        if not scope:
            continue
        out.setdefault(scope, {})[row.get("fact_key")] = row.get("fact_value")
    return out


async def db_get_fact(user_id, key):
    """Return a single stored fact value, or None if it does not exist."""
    facts = await db_get_user_facts(user_id)
    return facts.get(key)


async def db_clear_room_model(channel_id: str):
    """Forget a room's chosen model, so the default chain is used again."""
    await db_delete_fact(f"{MODEL_SETTING_PREFIX}{channel_id}", "model")


# A room's chosen model lives in the facts table, under a scope that is
# never read into a prompt. The room's own scope is "channel:<id>", and
# this one is "setting:model:<id>", so a model id can never turn up as
# something the bot knows about the people.
MODEL_SETTING_PREFIX = "setting:model:"


async def db_set_room_model(channel_id: str, model: str):
    """Choose which model answers in this room."""
    await db_set_fact(f"{MODEL_SETTING_PREFIX}{channel_id}", "model", model)


async def db_get_room_model(channel_id: str) -> str:
    """The model chosen for this room, or "" for the default chain."""
    try:
        value = await db_get_fact(f"{MODEL_SETTING_PREFIX}{channel_id}", "model")
    except Exception:
        return ""
    return str(value or "")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
#
# A schedule is a message the bot sends at a time Damian chose. It lives
# in the facts table under a scope of its own, in the settings namespace:
#
#     setting:schedule:<id>   at, days, to, message, where, channel, ...
#
# The same trick the room's model uses. It is a setting, not a memory -
# it is never read into a prompt - so a private message does not become
# something the model knows and can repeat somewhere else. And it is
# keyed by its own id, so one schedule can be edited or deleted without
# touching the others.
#
# The list of schedules is itself a row - "setting:schedules" - rather
# than a LIKE over the scopes. This table's user_id is a text column, so
# a pattern like "setting:schedule:%" would be read as a pattern and
# not as the prefix it looks like; the index is read with an exact
# match and the scopes are then fetched with "in", which is exact.

SCHEDULE_SETTING_PREFIX = "setting:schedule:"
SCHEDULE_INDEX_SCOPE = "setting:schedules"
SCHEDULE_INDEX_KEY = "ids"


def _schedule_id() -> str:
    """A new schedule id."""
    return "s" + uuid.uuid4().hex[:10]


async def _schedule_ids() -> list[str]:
    """The ids of every schedule that has been set."""
    try:
        value = await db_get_fact(SCHEDULE_INDEX_SCOPE, SCHEDULE_INDEX_KEY)
    except Exception:
        return []
    return [part for part in str(value or "").split(",") if part]


async def db_set_schedule(schedule_id: str, fields: dict):
    """Store a schedule, or change one."""
    scope = f"{SCHEDULE_SETTING_PREFIX}{schedule_id}"
    for key, value in (fields or {}).items():
        await db_set_fact(scope, key, "" if value is None else str(value))

    ids = await _schedule_ids()
    if schedule_id not in ids:
        ids.append(schedule_id)
        await db_set_fact(SCHEDULE_INDEX_SCOPE, SCHEDULE_INDEX_KEY,
                          ",".join(ids))


async def db_get_schedule(schedule_id: str) -> dict:
    """One schedule, as a dict of its fields."""
    try:
        facts = await db_get_user_facts(f"{SCHEDULE_SETTING_PREFIX}{schedule_id}")
    except Exception:
        return {}
    return {k: str(v) for k, v in (facts or {}).items()}


async def db_list_schedules() -> list[dict]:
    """Every schedule that has been set."""
    ids = await _schedule_ids()
    if not ids:
        return []

    scopes = [f"{SCHEDULE_SETTING_PREFIX}{schedule_id}" for schedule_id in ids]
    try:
        facts = await db_get_facts_by_scope(scopes)
    except Exception:
        return []

    out = []
    for schedule_id in ids:
        fields = facts.get(f"{SCHEDULE_SETTING_PREFIX}{schedule_id}") or {}
        if not fields:
            continue
        out.append({"id": schedule_id,
                    **{k: str(v) for k, v in fields.items()}})
    return out


async def db_delete_schedule(schedule_id: str):
    """Forget a schedule."""
    scope = f"{SCHEDULE_SETTING_PREFIX}{schedule_id}"
    try:
        facts = await db_get_user_facts(scope)
    except Exception:
        facts = {}
    for key in list(facts or {}):
        await db_delete_fact(scope, key)

    ids = await _schedule_ids()
    if schedule_id in ids:
        ids.remove(schedule_id)
        await db_set_fact(SCHEDULE_INDEX_SCOPE, SCHEDULE_INDEX_KEY,
                          ",".join(ids))


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
