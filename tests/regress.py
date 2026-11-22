"""A regression pass over everything that was changed.

The bot has grown a lot of moving parts, and the changes here touch the
prompt, the answer path, the tools and the memory layer at once. So this
imports every module, then exercises each of the new pieces.

    python tests/regress.py
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY", "NVIDIA_API_KEY")

# Some of the checks write files: the file tools, and the Kaggle
# carry-over. They do it in a scratch directory, so that a test run
# leaves nothing behind in the repository. The bot's own file tools work
# out where to write from the working directory, so this has to happen
# before the bot is imported.
_SCRATCH = tempfile.mkdtemp(prefix="bot-regress-")
os.chdir(_SCRATCH)

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


print("=" * 70)
print("EVERY MODULE IMPORTS")
print("=" * 70)

MODULES = [
    "cache", "config", "context", "discord_utils", "guard", "huggingface",
    "jobs", "kaggle_backend", "latex_render", "mathrender", "memory",
    "papers", "people", "presence", "prompts", "sandbox", "storage", "tools",
]

for name in MODULES:
    try:
        __import__(name)
        PASSED += 1
        print(f"  ok   {name}")
    except Exception as exc:
        FAILED.append(f"import {name}")
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")

# agent imports last: it pulls in nearly everything.
try:
    import agent  # noqa: E402

    PASSED += 1
    print("  ok   agent")
except Exception as exc:
    FAILED.append("import agent")
    print(f"  FAIL agent: {type(exc).__name__}: {exc}")

print()
print("=" * 70)
print("THE PROVIDER CHAIN")
print("=" * 70)

from config import provider_chain, vision_models  # noqa: E402

check("the chain is Laguna, MiniMax, Nemotron",
      [n for n, _, _ in provider_chain()], ["Laguna", "MiniMax", "NVIDIA"])
check("the vision chain is Nemotron",
      [n for n, _, _ in vision_models()], ["NVIDIA"])

print()
print("=" * 70)
print("NO SECRETS IN THE SOURCE")
print("=" * 70)

import re  # noqa: E402

ROOT_DIR = ROOT
PATTERNS = {
    "kaggle token": r"KGAT_[A-Za-z0-9]{20,}",
    "hf token": r"hf_[A-Za-z0-9]{30,}",
    "nvidia key": r"nvapi-[A-Za-z0-9_-]{30,}",
    "github pat": r"github_pat_[A-Za-z0-9_]{20,}",
    "discord token": r"MTU2OTY0NDU1[A-Za-z0-9._-]{20,}",
    "supabase key": r"sb_secret_[A-Za-z0-9_]{20,}",
}

leaks = []
for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
    dirnames[:] = [d for d in dirnames
                   if d not in {"node_modules", "__pycache__", "bot_files", ".git"}]
    for name in sorted(filenames):
        if name.endswith((".pyc", ".pyo")):
            continue
        path = os.path.join(dirpath, name)
        try:
            body = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            if re.search(pattern, body):
                leaks.append(f"{os.path.relpath(path, ROOT_DIR)}: {label}")

check("no live credentials in the source", leaks, [])

print()
print("=" * 70)
print("THE KAGGLE CARRY-OVER (without the network)")
print("=" * 70)

import kaggle_backend as kb  # noqa: E402

STATE = "kaggle_unit_state"
os.makedirs(STATE, exist_ok=True)
for stale in os.listdir(STATE):
    os.remove(os.path.join(STATE, stale))
with open(os.path.join(STATE, "answer.txt"), "w") as fh:
    fh.write("the answer is 42")

preamble, carried = kb._carry_preamble(STATE)
check("the file is carried", carried, ["answer.txt"])
check_true("the preamble writes the file", "answer.txt" in preamble)
check_true("the preamble targets the working directory", "/kaggle/working" in preamble)

# The preamble must compile and must actually write, with the brackets in
# the right place. Without them Python reads it as "path, or else write".
compile(preamble, "main.py", "exec")
import base64  # noqa: E402
import pathlib  # noqa: E402

ns = {"pathlib": pathlib, "base64": base64}
exec(preamble, ns)
check_true("the preamble wrote the file",
           os.path.exists(os.path.join(os.getcwd(), "answer.txt")))

check("the default state dir is the workspace one",
      kb._state_dir(""), kb.STATE_ROOT)
check_true("the state dir is under the workspace",
           kb.STATE_ROOT.endswith(os.path.join("bot_files", "kaggle")))

print()
print("=" * 70)
print("THE FILE TOOLS")
print("=" * 70)

import tools  # noqa: E402


async def file_probe():
    await tools.write_file("probe/f.txt", "one\ntwo\n")
    read = await tools.read_file("probe/f.txt")
    await tools.edit_file("probe/f.txt", "one", "ONE")
    edited = await tools.read_file("probe/f.txt")
    missing = await tools.read_file("probe/nope.txt")
    escaped = await tools.read_file("../../etc/passwd")
    return read, edited, missing, escaped


read, edited, missing, escaped = asyncio.run(file_probe())
check("a file is written and read", read, "one\ntwo\n")
check("a file is edited", edited, "ONE\ntwo\n")
check_true("a missing file says so", "not found" in missing.lower())
check_true("the workspace cannot be escaped", "access denied" in escaped.lower())

print()
print("=" * 70)
print("THE PROMPT")
print("=" * 70)

from prompts import build_system_prompt, turn_note  # noqa: E402

prompt = build_system_prompt({}, "", shared=False,
                              speaker={"id": "1", "name": "Jade", "plain": "Jade"},
                              roster=[{"id": "1", "name": "Jade", "plain": "Jade"},
                                      {"id": "2", "name": "Lain", "plain": "Lain"}])
check_true("the prompt has the register rules", "Two registers" in prompt)
check_true("the prompt has the quiet rules", "When to answer" in prompt)
check_true("the prompt names the people", "Jade" in prompt and "Lain" in prompt)
check_true("the turn note says how to choose", "WORKING" in turn_note())
check_true("the prompt does not carry a credential",
           "KGAT_" not in prompt and "hf_" not in prompt.split("Hub token:")[-1][:20])

print()
print("=" * 70)
print("THE PRESENCE DECISION")
print("=" * 70)

import presence  # noqa: E402
from people import (addresses_bot, bot_name_variants, name_collisions,  # noqa: E402
                    names_in, says_bot_name)

# The classifier answers two characters. A third one is a decision and a
# shape, and anything else is no answer at all.
check("A3 is an answer with a deep shape",
      presence._parse_verdict("A3"), ("answer", "deep"))
check("W1 is a wait with an aside shape",
      presence._parse_verdict("W1"), ("wait", "aside"))
check("Q2 is quiet with a question shape",
      presence._parse_verdict("Q2"), ("quiet", "question"))
check("nonsense is no decision",
      presence._parse_verdict("hm"), (None, ""))
check("an empty answer is no decision",
      presence._parse_verdict(""), (None, ""))


class FakeUser:
    def __init__(self, uid, name, bot=False):
        self.id = uid
        self.display_name = name
        self.global_name = name
        self.name = name
        self.bot = bot


class FakeRef:
    def __init__(self, message_id):
        self.message_id = message_id


class FakeMessage:
    _seq = 0

    def __init__(self, content, author, mid=None, reference=None):
        if mid is None:
            FakeMessage._seq += 1
            mid = f"m{FakeMessage._seq}"
        self.content = content
        self.author = author
        self.id = mid
        self.reference = reference
        self.referenced_message = None


BOT_ID = "bot-id"
NAMES = ("lain2", "lain", "lainy")
ROSTER = [{"id": "1", "name": "Lain", "plain": "Lain"},
          {"id": "2", "name": "Jade", "plain": "Jade"}]
COLLISIONS = name_collisions(NAMES, ROSTER)
check("the bot's short name clashes with somebody in the room",
      COLLISIONS, {"lain": "Lain"})

# A name at the front of the message is an address. A name in the middle
# of a sentence is somebody talking about the bot.
check("'Sup lain' addresses the bot",
      addresses_bot("Sup lain", NAMES, exclude=COLLISIONS), True)
check("'Lainy lainy odyssey' addresses the bot",
      addresses_bot("Lainy lainy odyssey, whos the wisest", NAMES,
                    exclude=COLLISIONS), True)
check("'I love lain2 more than you' does not address the bot",
      addresses_bot("E yeu lain2 hon ca toi", NAMES, exclude=COLLISIONS),
      False)
check("a bare name that belongs to somebody else is not an address",
      addresses_bot("Lain", NAMES, exclude=COLLISIONS), False)
check("'U hear me, lain?' is not a front-of-message address",
      addresses_bot("U hear me, lain?", NAMES, exclude=COLLISIONS), False)
check_true("the name is noticed in passing",
           says_bot_name("Mốt chắc lain2 cướp luôn e", NAMES))
check("the names that are used are found",
      names_in("Lainy lainy odyssey", NAMES), ("lainy",))

# The whole decision, with the classifier stubbed out so the structural
# checks can be tested without the network.
CALLS = []


async def stub_classifier(text, **kwargs):
    CALLS.append((text, kwargs))
    return STUB[0]


STUB = [(True, False, "question")]
presence.classify_reply = stub_classifier

JADE = FakeUser("2", "Jade")
DAMIAN = FakeUser("1", "Lain")


async def decide_probe(message, **kwargs):
    CALLS.clear()
    base = dict(is_dm=False, shared=True, bot_user_id=BOT_ID,
                bot_names=NAMES, bot_message_ids=set(), chat_on=True,
                roster=ROSTER, author_name="Jade", roster_line="Lain, Jade",
                bot_name="Lain", recent=(), reply_to="",
                bot_spoke_seconds=None, cache_extra="test")
    base.update(kwargs)
    return await presence.decide(message, **base)


async def presence_probe():
    # A mention is exact and free.
    d = await decide_probe(FakeMessage(f"<@{BOT_ID}> hello", JADE))
    check("a mention is answered", d.reply, True)
    check("a mention does not reach the model", len(CALLS), 0)

    # A reply to the bot is exact too.
    bot_ids = {"bot-message-1"}
    d = await decide_probe(
        FakeMessage("hello", JADE, reference=FakeRef("bot-message-1")),
        bot_message_ids=bot_ids)
    check("a reply to the bot is answered", d.reply, True)
    check("a reply to the bot does not reach the model", len(CALLS), 0)

    # The bot's name at the front is an address.
    d = await decide_probe(FakeMessage("Sup lain", JADE))
    check("a name at the front is answered", d.reply, True)
    check("a name at the front does not reach the model", len(CALLS), 0)

    # A name in passing is the model's business.
    d = await decide_probe(FakeMessage("E yeu lain2 hon ca toi", JADE))
    check("a name in passing is decided by the model", len(CALLS), 1)
    check_true("the model is told the name is there",
               "name appears" in CALLS[0][1].get("name_note", ""))

    # A bare name that belongs to somebody else in the room is the
    # model's business too.
    d = await decide_probe(FakeMessage("Lain", DAMIAN))
    check("a bare clashing name is decided by the model", len(CALLS), 1)
    check_true("the model is told about the clash",
               "also the name of somebody else" in CALLS[0][1].get("name_note", ""))

    # Everything else is the model's business.
    d = await decide_probe(FakeMessage("can you look up the Nvidia?", JADE))
    check("an ordinary message reaches the model", len(CALLS), 1)
    check("the model decided to answer", d.reply, True)

    # The three states come back as they should.
    STUB[0] = (False, True, "aside")
    d = await decide_probe(FakeMessage("can you look up the Nvidia", JADE))
    check("an unfinished message waits", d.wait, True)
    check("an unfinished message is not answered", d.reply, False)
    check("the shape is carried through", d.shape, "aside")

    # The final pass never leaves the message unanswered.
    d = await decide_probe(FakeMessage("can you look up the Nvidia", JADE),
                           final=True)
    check("the final pass answers a message that is still unfinished",
          (d.reply, d.wait), (True, False))

    STUB[0] = (False, False, "aside")
    d = await decide_probe(FakeMessage("ok", JADE))
    check("a quiet verdict is quiet", (d.reply, d.wait), (False, False))

    # A failed classifier is quiet rather than wrong.
    async def broken(text, **kwargs):
        return None

    presence.classify_reply = broken
    d = await decide_probe(FakeMessage("something", JADE))
    check("a failed classifier stays quiet", d.reply, False)
    presence.classify_reply = stub_classifier


asyncio.run(presence_probe())

# The reply size follows the turn, and the model is told why.
from config import extra_body_for  # noqa: E402

check("thinking can be switched off for a short reply",
      extra_body_for("Laguna", thinking=False),
      {"chat_template_kwargs": {"enable_thinking": False}})
check("thinking is on by default",
      extra_body_for("Laguna", thinking=True),
      {"chat_template_kwargs": {"enable_thinking": True}})

import inspect  # noqa: E402

check_true("the agent takes the turn shape",
           "shape" in inspect.signature(agent.get_nvidia_response_full).parameters)
check_true("the agent loop takes the thinking switch",
           "thinking" in inspect.signature(agent.run_agent_loop).parameters)
check_true("the agent loop can insist on a tool",
           "require_tool" in inspect.signature(agent.run_agent_loop).parameters)

# The answer-from-memory repair. The note has to put the message back in
# front of the model, because after the note the last thing in the
# conversation is the note.
_agent_source = inspect.getsource(agent)
check_true("the retry puts the message back in front of the model",
           "The message to answer is this one" in _agent_source)
check_true("the retry note keeps the voice",
           "tone of the conversation" in agent.TOOL_RETRY_NOTE)
check_true("the retry note does not leak the machinery",
           "classified" not in agent.TOOL_RETRY_NOTE)

# The lookup check is a model call, and the prompt has to ask the right
# question.
check_true("the lookup check asks about sources",
           "come from a source" in presence._LOOKUP_PROMPT)
check_true("the lookup check tells personal talk apart",
           "about themselves" in presence._LOOKUP_PROMPT)

print()
print("=" * 70)
print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("ALL GREEN")
