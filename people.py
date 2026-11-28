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


# Everybody the bot has met, wherever it met them. The roster is per
# channel - a name means the same person in every channel, but a room is
# only the people who speak in it - so this is the wider memory: a
# schedule set in one room can be about somebody who only ever speaks in
# another.
SEEN: dict[str, dict] = {}


def remember(entry: dict) -> dict | None:
    """Remember one person, so a name can be resolved later."""
    if not isinstance(entry, dict):
        return None
    user_id = str(entry.get("id") or "")
    if not user_id:
        return None
    known = {
        "id": user_id,
        "name": entry.get("name") or "",
        "plain": entry.get("plain") or plain_name(entry.get("name") or ""),
    }
    SEEN[user_id] = known
    return known


def _name_tokens(text: str) -> list[str]:
    """The words a name is made of.

    A Discord name is often "jade_m8" or "LainWasHere", and people ask
    for each other by one piece of it - so a name is matched by its
    parts as well as by itself.
    """
    plain = plain_name(str(text or "")).lower()
    return [part for part in re.split(r"[^a-z0-9]+", plain) if part]


def find_person(name: str) -> dict | None:
    """The person with this name, if the bot has met them.

    The name is matched the way people write it: the styled form and the
    plain form are the same name, the comparison ignores case and
    spaces, and a name matches a part of somebody's name - "jade" is
    how she is asked for, even though she goes by "jade_m8". An id is
    accepted too, because that is the one form that is never wrong.
    """
    if not name:
        return None
    want = plain_name(str(name)).strip().lower()
    if not want:
        return None

    for entry in SEEN.values():
        if entry["id"] == want:
            return entry

    for entry in SEEN.values():
        for value in (entry.get("plain"), entry.get("name")):
            if not value:
                continue
            plain = plain_name(str(value)).strip().lower()
            if not plain:
                continue
            if plain == want:
                return entry
            tokens = _name_tokens(value)
            if want in tokens or plain in _name_tokens(want):
                return entry
    return None


def mention_for(entry) -> str:
    """The tag that pings this person.

    It is a Discord user mention - "<@id>" - which is the only thing that
    actually notifies anybody. Writing somebody's name does not.
    """
    if not isinstance(entry, dict):
        return ""
    user_id = str(entry.get("id") or "")
    return f"<@{user_id}>" if user_id else ""


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

    Every person seen is also remembered by id, so a name can be resolved
    later from anywhere - which is what a schedule needs when it is set in
    one room and is about somebody who speaks in another.
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
        entry = person(author)
        remember(entry)
        people.append(entry)
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


# ---------------------------------------------------------------------------
# Mentions
# ---------------------------------------------------------------------------
#
# Writing somebody's name does not notify anybody. The only thing that
# does is a Discord user mention - "<@id>" - and only when the message
# says that person is allowed to be mentioned.
#
# The model is allowed to write a tag when somebody has to see the
# message. It is not allowed to write one because their name came up,
# so there is a rule about where in the message a tag counts as an
# address: a tag at the front is who you are talking to, a tag further
# down is who you are talking about. The second one is written out as
# the person's name, and nobody is pinged.
#
# The tags themselves are safe by construction: Discord's own default
# parses every mention in a message, so a message that quoted somebody's
# "@everyone" would wake the whole server. The bot never allows that -
# the server-wide pings are taken out of its words, and only the people
# it means to ping are allowed.

# A user mention, in either of the two forms Discord writes.
_MENTION = re.compile(r"<@!?(\d{5,25})>")

# How far into the message a tag still counts as an address.
ADDRESS_WINDOW = 120

# The words that can come before a tag and still leave it an address.
# "hey <@Jade>" is addressed to her; "I told <@Jade>" is not.
_GREETINGS = {
    "hey", "hi", "hello", "yo", "oi", "ok", "okay", "well", "so",
    "e", "em", "anh", "a", "everyone", "everybody", "all", "ps", "p.s.",
    "mọi", "mọi người",
}

# The punctuation that can sit around a word.
_EDGE = ".,!?¿¡:;…-–— \t\n"


def _reads_as_address(text: str, start: int, entry) -> bool:
    """Does this tag read as "I am talking to you", or as "about you"?

    The model is told to start a message with somebody's tag when
    they have to see it. So the tag is an address when it is the first
    thing in the message, or when everything before it is a greeting or
    the person's own name:

        <@Jade> did you see this?      an address
        hey <@Jade> did you see this?  an address
        Jade, <@Jade> did you see this? an address
        I told <@Jade> about it         somebody being talked about
    """
    prefix = text[:start].strip().strip(_EDGE).strip()
    if not prefix:
        return True

    words = prefix.split()
    if len(words) > 3:
        return False

    last = words[-1].strip(_EDGE)
    if not last:
        return False
    if last.lower() in _GREETINGS:
        return True

    # The person's own name.
    if entry:
        for value in (entry.get("plain"), entry.get("name")):
            if not value:
                continue
            plain = plain_name(str(value)).strip().lower()
            if not plain:
                continue
            if last.lower() == plain or last.lower() in _name_tokens(value):
                return True
    return False


def mentions_in(text: str) -> list[str]:
    """The ids of everybody tagged in this text."""
    return _MENTION.findall(text or "")


def neutralise_mass_mentions(text: str) -> str:
    """Take the server-wide pings out of the bot's own words.

    "@everyone", "@here" and a role tag wake everybody in the channel.
    The bot has no business doing that - it is not an announcement
    service - so they are written out as plain words.
    """
    if not text:
        return text
    text = re.sub(r"<@&\d{5,25}>", "@a role", text)
    text = re.sub(r"@everyone\b", "everyone", text, flags=re.IGNORECASE)
    text = re.sub(r"@here\b", "here", text, flags=re.IGNORECASE)
    return text


def address_mentions(text: str, people: dict | None = None) -> tuple[str, list[str]]:
    """Work out which tags in a message are an address, and keep only those.

    Returns the text to send and the ids that may be pinged. A tag that
    is not an address is written out as the person's name, so the message
    still reads the way it was meant to - it just does not ping anybody.
    """
    if not text:
        return text or "", []

    lookup = people if people is not None else SEEN
    out = []
    pings = []
    last = 0

    for match in _MENTION.finditer(text):
        user_id = match.group(1)
        entry = lookup.get(user_id)
        name = ""
        if entry:
            name = entry.get("plain") or entry.get("name") or ""

        out.append(text[last:match.start()])
        is_address = (
            entry is not None
            and match.start() <= ADDRESS_WINDOW
            and _reads_as_address(text, match.start(), entry))
        if is_address:
            # It is an address: the tag stays, and the person may be pinged.
            out.append(f"<@{user_id}>")
            if user_id not in pings:
                pings.append(user_id)
        else:
            # It is a mention in passing, or of somebody the bot has never
            # met. The name is written out instead.
            out.append(name or "@someone")
        last = match.end()

    out.append(text[last:])
    return "".join(out), pings
