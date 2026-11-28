"""Being allowed to ping somebody, and not ping anybody else.

Writing somebody's name does not notify anybody. The only thing that
does is a Discord user mention - "<@id>" - and only when the message
says that person may be mentioned. Discord's own default parses every
mention in a message, so a message that quoted somebody's "@everyone"
would wake the whole server.

So there are two halves to this:

  - the bot may tag somebody, when the message is addressed to them;
  - the bot may never tag anybody else, and never wake the server.

Both are checked here, with the Discord objects faked and the storage
stubbed out. Nothing touches the network.

    python tests/test_mentions.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import people  # noqa: E402

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")


def check_true(label, cond):
    check(label, bool(cond), True)


JADE_ID = "909453181916938301"
DAMIAN_ID = "1499211380052721684"

JADE = {"id": JADE_ID, "name": "\U0001d4a7\U0001d4a0\U0001d4a7\U0001d4e8_\U0001d4ec",
        "plain": "jade_m8"}
DAMIAN = {"id": DAMIAN_ID, "name": "Lain", "plain": "Lain"}


class FakeTarget:
    """Something that can be sent to."""

    def __init__(self):
        self.sent = []

    async def send(self, content=None, **kwargs):
        self.sent.append((content, kwargs))
        return content


def test_names():
    print()
    print("=" * 70)
    print("THE NAMES")
    print("=" * 70)

    people.SEEN.clear()
    people.remember(JADE)
    people.remember(DAMIAN)

    check("a name is found", (people.find_person("jade") or {}).get("id"), JADE_ID)
    check("a styled name is found",
          (people.find_person("\U0001d4a7\U0001d4a0\U0001d4a7\U0001d4e8_\U0001d4ec") or {}).get("id"),
          JADE_ID)
    check("a plain name is found",
          (people.find_person("jade_m8") or {}).get("id"), JADE_ID)
    check("an id is found", (people.find_person(JADE_ID) or {}).get("id"), JADE_ID)
    check("somebody nobody has met is not found",
          people.find_person("nobody at all"), None)

    check("the tag is the tag", people.mention_for(JADE), f"<@{JADE_ID}>")
    check("a person with no id has no tag",
          people.mention_for({"id": "", "name": "x"}), "")


def test_addresses():
    print()
    print("=" * 70)
    print("AN ADDRESS, OR SOMEBODY BEING TALKED ABOUT")
    print("=" * 70)

    people.SEEN.clear()
    people.remember(JADE)

    # A tag at the front is an address: it stays, and it pings.
    text, pings = people.address_mentions(f"<@{JADE_ID}> did you see this?")
    check("an address keeps its tag", text, f"<@{JADE_ID}> did you see this?")
    check("an address pings", pings, [JADE_ID])

    # A greeting in front does not change that.
    text, pings = people.address_mentions(f"hey <@{JADE_ID}> did you see this?")
    check("a greeting keeps it an address", pings, [JADE_ID])

    # Their own name in front does not change it either.
    text, pings = people.address_mentions(f"jade, <@{JADE_ID}> did you see this?")
    check("a name keeps it an address", pings, [JADE_ID])

    # A tag further down is somebody being talked about: it is written
    # out as their name, and nobody is pinged.
    text, pings = people.address_mentions(
        "I told <@" + JADE_ID + "> about it earlier")
    check("a mention in passing becomes a name", text,
          "I told jade_m8 about it earlier")
    check("a mention in passing does not ping", pings, [])

    # A tag past the window is not an address either.
    long_text = "x" * (people.ADDRESS_WINDOW + 10) + f" <@{JADE_ID}> hello"
    text, pings = people.address_mentions(long_text)
    check("a tag past the window does not ping", pings, [])
    check("a tag past the window becomes a name", "<@" in text, False)

    # Somebody the bot has never met is not pinged either.
    text, pings = people.address_mentions("<@12345678901> hello")
    check("a stranger is not pinged", pings, [])
    check("a stranger is not left as a raw tag", "<@" in text, False)


def test_mass_mentions():
    print()
    print("=" * 70)
    print("THE SERVER-WIDE PINGS")
    print("=" * 70)

    check("everyone is taken out",
          people.neutralise_mass_mentions("hello @everyone"), "hello everyone")
    check("here is taken out",
          people.neutralise_mass_mentions("hello @here"), "hello here")
    check("a role tag is taken out",
          people.neutralise_mass_mentions("hello <@&123456789012345678>"),
          "hello @a role")
    check("a user tag is left alone",
          people.neutralise_mass_mentions(f"hi <@{JADE_ID}>"),
          f"hi <@{JADE_ID}>")

    check("the ids in a message are found",
          people.mentions_in(f"a <@{JADE_ID}> b <@!{DAMIAN_ID}>"),
          [JADE_ID, DAMIAN_ID])


async def test_send_path():
    print()
    print("=" * 70)
    print("THE SEND PATH")
    print("=" * 70)

    import discord
    from discord_utils import send_long

    people.SEEN.clear()
    people.remember(JADE)

    # An address is sent as a tag, and only that person may be pinged.
    target = FakeTarget()
    await send_long(target, f"<@{JADE_ID}> did you see this?", render_math=False)
    check("the message goes out", len(target.sent), 1)
    text, kwargs = target.sent[0]
    check("the tag is still there", text, f"<@{JADE_ID}> did you see this?")
    allowed = kwargs.get("allowed_mentions")
    check_true("the message says who may be pinged", allowed is not None)
    payload = allowed.to_dict()
    check("only that person may be pinged", payload.get("users"), [int(JADE_ID)])
    check("nobody else may be parsed", payload.get("parse"), [])

    # A mention in passing is written out, and nobody is pinged.
    target = FakeTarget()
    await send_long(target, f"I told <@{JADE_ID}> earlier", render_math=False)
    text, kwargs = target.sent[0]
    check("a mention in passing is a name", text, "I told jade_m8 earlier")
    check("and nobody is pinged",
          kwargs.get("allowed_mentions").to_dict().get("users"), [])

    # The server-wide pings never get through, whatever the model wrote.
    target = FakeTarget()
    await send_long(target, "hello @everyone and <@&123456789012345678>",
                    render_math=False)
    text, kwargs = target.sent[0]
    check_true("everyone is gone", "@everyone" not in text)
    check_true("the role is gone", "<@&" not in text)
    check("and no role may be pinged",
          kwargs.get("allowed_mentions").to_dict().get("parse"), [])

    # A reply may still ping the person who was replied to - that is
    # Discord's own rule and it is not a broadcast.
    allowed = discord.AllowedMentions(
        users=[discord.Object(id=int(JADE_ID))],
        roles=False, everyone=False, replied_user=True)
    payload = allowed.to_dict()
    check("the reply rule is kept", payload.get("replied_user"), True)
    check("and only the person", payload.get("users"), [int(JADE_ID)])


async def run_tests():
    test_names()
    test_addresses()
    test_mass_mentions()
    await test_send_path()

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(run_tests())
