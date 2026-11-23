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


def bot_canonical_name(user) -> str:
    """The name the bot goes by, with the decorations taken off.

    The bot's Discord name is "lain2" - the 2 is there because somebody else
    in the room is called "Lain". But the bot is Lain too, and when it is
    asked what it is called, "Lain" is the answer it gives.
    """
    raw = display_name(user)
    trimmed = re.sub(r"[\d_]+$", "", raw).strip()
    if not trimmed:
        trimmed = raw
    if trimmed.islower():
        trimmed = trimmed.capitalize()
    return trimmed or raw


def bot_name_variants(user) -> tuple[str, ...]:
    """The names people might call the bot by, derived from its Discord user.

    A bot called "lain2" is also "lain"; one called "quantum-bot" is also
    "quantum". This is derived rather than written down, so renaming the bot
    needs no code change.

    A diminutive is in there too: people who like a name shorten it, and
    "lainy" was how the bot was addressed in the log while its own name was
    "lain2".
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
                # "lain" -> "lainy", "quantum" -> "quantumy"
                if len(trimmed) >= 4:
                    names.add(trimmed + "y")
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


def names_in(text: str, names) -> tuple[str, ...]:
    """Which of these names the message actually uses.

    "lain" and "lain2" are two spellings of the same name, so a message
    that uses both is one mention, not two.
    """
    if not text:
        return ()
    low = text.lower()
    found = []
    for name in names or ():
        needle = name.lower()
        if not needle:
            continue
        if re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", low):
            found.append(name)
    return tuple(found)


def says_bot_name(text: str, names) -> bool:
    """Does the message use one of the bot's names?"""
    return bool(names_in(text, names))


def name_collisions(bot_names, roster) -> dict[str, str]:
    """Bot names that are also somebody else's name in this room.

    The bot is "lain2"; one of the people in the chat is called "Lain".
    A message that is just "Lain" is much more likely to be that person
    giving their name than the bot being addressed - and the bot cannot
    answer to it, or it would answer its own name every time somebody
    introduced themselves.

    This cannot be a list of names, because it depends on who is in the
    room today. It is read from the roster and then handed to the model,
    which can see whether the message is talking to the bot or saying
    somebody's name.
    """
    humans: dict[str, str] = {}
    for entry in roster or ():
        if not isinstance(entry, dict):
            continue
        for value in (entry.get("name"), entry.get("plain")):
            if not value:
                continue
            key = plain_name(str(value)).strip().lower()
            if key and key not in humans:
                humans[key] = str(value)
    out: dict[str, str] = {}
    for name in bot_names or ():
        needle = plain_name(str(name)).strip().lower()
        if needle and needle in humans:
            out[name] = humans[needle]
    return out


def addresses_bot(text: str, names, exclude=()) -> bool:
    """Is the bot's name used to address it, rather than mentioned in passing?

    The difference showed up in the log. These are all the same name:

        "Sup lain"                      - the bot is being talked to
        "Yo lain she went to sleep"     - the bot is being talked to
        "Lain"                          - Damian telling Jade his name
        "E yeu lain2 hon ca toi"        - Damian telling Jade he loves the bot
        "Mốt chắc lain2 cướp luôn e"   - two people talking about the bot

    The first two are addressed to the bot and the last three are not, and
    the difference is where the name sits: a name at the front of the
    message is who you are talking to, a name in the middle of a sentence
    is who you are talking about. The middle ones are left to the model,
    which reads the room properly.

    `exclude` holds names that belong to somebody else in the room as
    well - the bot is "lain2" and one of the people is called "Lain". A
    message that is nothing but that name is left to the model, because
    it may well be that person giving their name. With any other words
    around it, the name is an address like any other.
    """
    if not text:
        return False
    skip = {str(n).lower() for n in exclude or ()}
    stripped = text.strip().strip(".,!?¿¡:;…-–— \t\n")
    words = stripped.split()
    if not words:
        return False
    # Only the first two words are considered: a name after that is
    # somebody talking about the bot.
    front = " ".join(words[:2]).lower()
    bare_name = len(words) == 1
    for name in names or ():
        needle = name.lower()
        if not needle:
            continue
        if needle in skip and bare_name:
            # The message is nothing but a name that belongs to somebody
            # else too. It is left to the model, which can see who is
            # talking to whom.
            continue
        if re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", front):
            return True
    return False
