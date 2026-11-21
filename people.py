"""The people in the room: who they are, and what to call them.

Nobody is hard-coded here. The roster is read from the channel's own
messages, so somebody new in the channel is picked up automatically and the
bot calls them by the name they actually go by - which matters, because the
name people choose is often styled ("𝐣𝐚𝐝𝐞_𝐦𝟖") and the readable form of it
("jade_m8") is what has to end up in a stored memory, where a model can
still match it.
"""

import re
import unicodedata


def plain_name(text: str) -> str:
    """The readable form of a styled name.

    Discord names are often written in the mathematical bold/sans characters
    ("𝐣𝐚𝐝𝐞_𝐦𝟖"). NFKC folds those back to ASCII, so the name can be used in
    a sentence and still be recognised later.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).strip()


def display_name(user) -> str:
    """The name to call somebody by."""
    for attr in ("display_name", "global_name", "name"):
        value = getattr(user, attr, None)
        if value and str(value).strip():
            return str(value).strip()
    return "someone"


def person(user) -> dict:
    """One person, as the prompt and the memory layer need them."""
    name = display_name(user)
    return {
        "id": str(getattr(user, "id", "") or ""),
        "name": name,
        "plain": plain_name(name),
    }


def roster_from_messages(messages) -> list[dict]:
    """The people in a channel, read from the messages in it.

    The bot's own messages are skipped, and the first time somebody is seen
    is the time they are recorded - so the roster reads in the order the
    conversation actually happened.
    """
    people = []
    seen = set()
    for message in messages or []:
        author = getattr(message, "author", None)
        if author is None or getattr(author, "bot", False):
            continue
        user_id = str(getattr(author, "id", "") or "")
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        people.append(person(author))
    return people


def roster_line(roster) -> str:
    """One line describing who is in the room."""
    if not roster:
        return ""
    names = []
    for entry in roster:
        name = entry.get("name") or ""
        plain = entry.get("plain") or ""
        if plain and plain.lower() != name.lower():
            names.append(f"{name} ({plain})")
        else:
            names.append(name)
    return ", ".join(names)


def bot_name_variants(user) -> tuple[str, ...]:
    """The names people might call the bot by, derived from its Discord user.

    A bot called "lain2" is also "lain"; one called "quantum-bot" is also
    "quantum". This is derived rather than written down, so renaming the bot
    needs no code change.
    """
    names = set()
    for attr in ("display_name", "global_name", "name"):
        value = getattr(user, attr, None)
        if not value:
            continue
        for raw in (str(value), plain_name(str(value))):
            raw = raw.strip().lower()
            if not raw:
                continue
            names.add(raw)
            # "lain2" -> "lain", "quantum_bot" -> "quantum"
            trimmed = raw.rstrip("0123456789").rstrip("_- ")
            if trimmed:
                names.add(trimmed)
            # "quantumbot" -> "quantum"
            if trimmed.endswith("bot") and len(trimmed) > 3:
                names.add(trimmed[:-3].rstrip("_- "))
    return tuple(sorted(n for n in names if len(n) >= 3))


def mentioned(message, bot_user_id) -> bool:
    """Was the bot actually mentioned?"""
    if not bot_user_id:
        return False
    content = getattr(message, "content", "") or ""
    return (f"<@{bot_user_id}>" in content
            or f"<@!{bot_user_id}>" in content)


def replies_to_bot(message, bot_message_ids) -> bool:
    """Is this a reply to one of the bot's own messages?"""
    if not bot_message_ids:
        return False
    reference = getattr(message, "reference", None)
    message_id = getattr(reference, "message_id", None)
    if not message_id:
        return False
    return message_id in bot_message_ids


def says_bot_name(text: str, names) -> bool:
    """Does the message use one of the bot's names?"""
    if not text:
        return False
    low = text.lower()
    for name in names or ():
        if re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])", low):
            return True
    return False
