"""Drive the real on_message handler with fake Discord objects.

Everything below the handler has been tested on its own, but the wiring
between the pieces is what breaks when one of them changes: the decision
reaches the agent as a shape, the wait holds the message and then
answers it, and a message meant for the person next to it never gets to
the model at all.

Nothing here touches the network: the agent is stubbed, and the Discord
objects are fakes.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")


import main  # noqa: E402
import presence  # noqa: E402

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


CHANNEL_ID = "1569657875368902666"
BOT_ID = "bot-id"

ROSTER = [{"id": "1", "name": "Lain", "plain": "Lain"},
          {"id": "2", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}]


class FakeUser:
    def __init__(self, uid, name):
        self.id = uid
        self.display_name = name
        self.global_name = name
        self.name = name
        self.bot = False

    def mentioned_in(self, message):
        return f"<@{BOT_ID}>" in (message.content or "")


class FakeBotUser(FakeUser):
    def __init__(self):
        super().__init__(BOT_ID, "lain2")
        self.bot = True


class FakeRef:
    def __init__(self, message_id):
        self.message_id = message_id


class FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeChannel:
    def __init__(self):
        self.id = CHANNEL_ID
        self.history_items = []
        self.sent = []

    def history(self, limit=50):
        async def gen():
            for m in self.history_items[:limit]:
                yield m
        return gen()

    def typing(self):
        return FakeTyping()

    async def send(self, content):
        self.sent.append(content)
        return [FakeMessage(content, FakeBotUser(),
                            mid=f"bot-{len(self.sent)}")]


class FakeMessage:
    def __init__(self, content, author, mid=None, reference=None):
        if mid is None:
            FakeMessage._seq = getattr(FakeMessage, "_seq", 0) + 1
            mid = f"m{FakeMessage._seq}"
        self.content = content
        self.author = author
        self.id = mid
        self.reference = reference
        self.referenced_message = None
        self.channel = FakeChannel()
        self.attachments = []

    async def reply(self, content, **kwargs):
        return await self.channel.send(content)


JADE = FakeUser("2", "𝐣𝐚𝐝𝐞_𝐦𝟖")
DAMIAN = FakeUser("1", "Lain")

# What the presence classifier said, per message. The handler is driven
# with these, so the wiring is tested without the network. The second
# table is what the classifier says on the final pass - after the person
# has stopped typing.
VERDICTS = {}
VERDICTS_FINAL = {}


async def fake_decide(message, **kwargs):
    text = message.content
    if kwargs.get("final"):
        return VERDICTS_FINAL.get(text,
                                  VERDICTS.get(text,
                                               presence.Decision(False, "quiet")))
    return VERDICTS.get(text, presence.Decision(True, "answer"))


AGENT_CALLS = []


async def fake_agent(user_id, message, channel=None, **kwargs):
    AGENT_CALLS.append({"message": message, **kwargs})
    return {"answer": "an answer", "footer": "-# model", "files": []}


async def main_probe():
    # The pieces around the handler, so nothing reaches the network.
    class FakeBot:
        user = FakeBotUser()
        command_prefix = "!"

        def get_channel(self, cid):
            return None

        async def process_commands(self, message):
            return None

    main.bot = FakeBot()
    main.db_is_chat_mode_enabled = lambda cid: asyncio.sleep(0, result=True)

    async def fake_is_shared(channel_id, channel=None):
        # The real one marks the tracker, which is what the handler reads.
        main.TRACKER.mark_shared(channel_id)
        main.SHARED_CHANNELS[channel_id] = True
        return True

    main._channel_roster = lambda cid, channel=None, force=False: asyncio.sleep(
        0, result=ROSTER)
    main._channel_is_shared = fake_is_shared
    main._recent_messages = lambda channel, limit=8, exclude_id=None: asyncio.sleep(
        0, result=[("Lain", "hello"), ("Jade", "hey")])
    main.decide = fake_decide
    main.get_nvidia_response_full = fake_agent
    main.WAIT_SECONDS = 0.2
    # The model this room chose with !model. It is read from the
    # database, and the database has an opinion of its own about this
    # room - which is not what this test is about.
    main.db_get_room_model = lambda cid: asyncio.sleep(0, result="")
    main.ROOM_MODELS.clear()

    async def fake_send_long(message, content, footer=None):
        return await message.channel.send(content)

    main.send_long = fake_send_long
    main.attach_images = lambda target, files: asyncio.sleep(0)

    print("=" * 70)
    print("A MESSAGE THE MODEL SAYS IS FOR THE BOT")
    print("=" * 70)

    VERDICTS["yo lain what is filtering theory"] = presence.Decision(
        True, "the model read the room", shape="deep", from_classifier=True)
    AGENT_CALLS.clear()
    message = FakeMessage("yo lain what is filtering theory", JADE)
    await main.on_message(message)

    check("the agent was called", len(AGENT_CALLS), 1)
    check("the shape reached the agent", AGENT_CALLS[0].get("shape"), "deep")
    check("the reply was sent", len(message.channel.sent), 1)
    check_true("the shared scope was used",
               AGENT_CALLS[0].get("write_scope") == f"channel:{CHANNEL_ID}")
    check_true("the person's own scope is read as well",
               "2" in AGENT_CALLS[0].get("scopes", []))
    check_true("the room's scope is read as well",
               f"channel:{CHANNEL_ID}" in AGENT_CALLS[0].get("scopes", []))

    print()
    print("=" * 70)
    print("A MESSAGE FOR THE PERSON NEXT TO IT")
    print("=" * 70)

    VERDICTS["mệt quá đi"] = presence.Decision(
        False, "not for the bot", shape="aside", from_classifier=True)
    AGENT_CALLS.clear()
    message = FakeMessage("mệt quá đi", DAMIAN)
    await main.on_message(message)
    check("the agent was not called", len(AGENT_CALLS), 0)
    check("nothing was sent", len(message.channel.sent), 0)

    print()
    print("=" * 70)
    print("A MESSAGE THAT IS NOT FINISHED YET")
    print("=" * 70)

    VERDICTS["can you look up the Nvidia"] = presence.Decision(
        False, "not finished", wait=True, shape="deep", from_classifier=True)
    VERDICTS_FINAL["can you look up the Nvidia"] = presence.Decision(
        True, "the person has stopped typing", shape="deep", from_classifier=True)
    VERDICTS["can you look up the Nvidia earnings date"] = presence.Decision(
        True, "answered", shape="deep", from_classifier=True)
    AGENT_CALLS.clear()
    message = FakeMessage("can you look up the Nvidia", JADE)
    await main.on_message(message)
    check("the agent was not called yet", len(AGENT_CALLS), 0)
    await asyncio.sleep(0.5)
    check("the agent was called after the wait", len(AGENT_CALLS), 1)
    check("the shape reached the agent", AGENT_CALLS[0].get("shape"), "deep")
    check("the reply was sent", len(message.channel.sent), 1)

    print()
    print("=" * 70)
    print("A WAITING MESSAGE OVERTaken BY A NEWER ONE")
    print("=" * 70)

    VERDICTS["the thing we talked about"] = presence.Decision(
        False, "not finished", wait=True, shape="aside", from_classifier=True)
    AGENT_CALLS.clear()
    first = FakeMessage("the thing we talked about", JADE)
    await main.on_message(first)
    # The next message from the same person is the one that gets answered.
    VERDICTS["the thing we talked about yesterday, the one where"] = \
        presence.Decision(True, "answered", shape="aside",
                               from_classifier=True)
    second = FakeMessage("the thing we talked about yesterday, the one where",
                         JADE)
    await main.on_message(second)
    await asyncio.sleep(0.5)
    check("only the newer message was answered", len(AGENT_CALLS), 1)
    check("the newer message was the one answered",
          AGENT_CALLS[0].get("message"),
          "the thing we talked about yesterday, the one where")

    print()
    print("=" * 70)
    print("A PASSING REMARK")
    print("=" * 70)

    VERDICTS["ừ"] = presence.Decision(
        True, "the model read the room", shape="aside", from_classifier=True)
    AGENT_CALLS.clear()
    message = FakeMessage("ừ", JADE)
    await main.on_message(message)
    check("the agent was called", len(AGENT_CALLS), 1)
    check("the aside shape reached the agent", AGENT_CALLS[0].get("shape"), "aside")


asyncio.run(main_probe())

print()
print("=" * 70)
print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("ALL GREEN")
