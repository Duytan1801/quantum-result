"""A regression pass over everything that was changed.

The bot has grown a lot of moving parts, and the changes here touch the
prompt, the answer path, the tools and the memory layer at once. So this
imports every module, then exercises each of the new pieces.

    python tests/regress.py
"""

import asyncio
import inspect
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

from config import provider_chain, vision_models, BOLT_MODEL  # noqa: E402

check("the chain is Nemotron, MiniMax",
      [n for n, _, _ in provider_chain()], ["NVIDIA", "MiniMax"])
check("the vision chain is Nemotron",
      [n for n, _, _ in vision_models()], ["NVIDIA"])

# The answer and the scan are two different models on purpose: Nemotron
# answers, Bolt reads the room. If they were the same, the bot would
# be marking its own homework.
import presence  # noqa: E402
from config import BOLT_MODEL, MINIMAX_MODEL, NVIDIA_MODEL  # noqa: E402

check("the answer chain starts with Nemotron",
      provider_chain()[0][2], NVIDIA_MODEL)
check("the scanner is Bolt", presence.scanner_models()[0], BOLT_MODEL)
check("the scanner falls back to Nemotron",
      presence.scanner_models(), (BOLT_MODEL, NVIDIA_MODEL))
check_true("the scanner does not scan with the answering model first",
           presence.scanner_models()[0] != provider_chain()[0][2])
check_true("the scanner does not fall back to Groq",
           "groq" not in " ".join(presence.scanner_models()).lower())
check_true("Laguna is gone from the chain",
           "laguna" not in " ".join(m for _, _, m in provider_chain()).lower())

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
        # The local .env is where the keys are *meant* to be: it is in
        # .gitignore and never goes up to GitHub. What this check is
        # about is a key written into a file that does go up.
        if name == ".env" or name.startswith(".env."):
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

from prompts import (build_system_prompt, turn_note, voice_examples,  # noqa: E402
                      voice_note, _kinship_note)

# The roster with the ids the room actually uses: the kinship terms are
# keyed by Discord id, so a room of made-up people gets no terms at all.
PROMPT_ROSTER = [{"id": "909453181916938301", "name": "Jade", "plain": "Jade"},
                 {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}]

prompt = build_system_prompt({}, "", shared=False,
                              speaker={"id": "1", "name": "Jade", "plain": "Jade"},
                              roster=PROMPT_ROSTER,
                              bot_name="Lain",
                              bot_aliases=("lain2", "lainy"))
check_true("the prompt has the register rules", "Two registers" in prompt)
check_true("the prompt has the quiet rules", "When to answer" in prompt)
check_true("the prompt names the people", "Jade" in prompt and "Lain" in prompt)
check_true("the turn note says how to choose", "WORKING" in turn_note())
check_true("the prompt does not carry a credential",
           "KGAT_" not in prompt and "hf_" not in prompt.split("Hub token:")[-1][:20])

# The naturalness rules. The bot said "Bạn" and "Tôi" to people who have
# been talking as friends all evening, and that is the single thing that
# made it read as a helpdesk instead of a friend.
check_true("the prompt says to match the room's register",
           "Match the register of the room" in prompt)
check_true("the prompt names the right pronouns",
           "anh/em/a/e" in prompt and "'Bạn'" in prompt)
check_true("the prompt says facts do not need a formal voice",
           "A fact does" in prompt and "formal voice" in prompt)
check_true("the prompt forbids the confirmation-word openers",
           "Absolutely" in prompt and "Chắc chắn rồi" in prompt)
check_true("the prompt says not to use the name every time",
           "tic, not warmth" in prompt)
check_true("the prompt caps the length",
           "Two or three sentences" in prompt)
check_true("the prompt says to have a position",
           "not a helpdesk" in prompt)
check_true("the prompt follows the language of the message",
           "language of the message you are answering" in prompt)

# The bot's own name. It was asked what it is called and had no answer.
check_true("the prompt gives the bot its name",
           "Your name is Lain" in prompt)
check_true("the prompt gives the other names too",
           "lain2" in prompt and "lainy" in prompt)
check_true("the prompt says what to answer when asked",
           "the answer is Lain" in prompt)
check_true("the prompt warns about the person with the same name",
           "goes by Lain as well" in prompt)

# The style anchor: real lines from the room, not adjectives.
anchor = voice_note(voice_examples([
    {"role": "user", "content": "mệt quá đi, hôm nay học nhiều quá"},
    {"role": "assistant", "content": "Bạn nghỉ ngơi đi nhé"},
    {"role": "user", "content": "E yeu lain2 hon ca toi"},
]))
check_true("the anchor uses the real line",
           "mệt quá đi" in anchor)
check_true("the anchor leaves out the bot's own replies",
           "nghỉ ngơi" not in anchor)
check_true("the anchor says what to take from it",
           "Take the register from these lines" in anchor)
check_true("the anchor says how to talk about yourself",
           "Talk about yourself the way they talk about themselves" in anchor)
check_true("an empty history gives no anchor", voice_note([]) == "")

# The kinship terms come from the room, not from a sentence in the code
# that goes stale when somebody new arrives.
check_true("the prompt says Jade is 'em'",
           "Jade is 'em'" in prompt)
check_true("the prompt says Damian is 'anh'",
           "Lain is 'anh'" in prompt)
check_true("the prompt does not name the forbidden terms",
           "'chị' is nobody" not in prompt)

# A room of people the bot has no terms for gets no terms.
check_true("a room with no known people gets no kinship terms",
           _kinship_note([{"id": "999", "name": "Someone",
                           "plain": "Someone"}]) == "")

# The register check. A reply in the wrong voice is asked for again, and
# the model is told what is wrong rather than the words being rewritten.
from guard import kinship_problem, register_problem  # noqa: E402

check("a formal 'Bạn' is named",
      register_problem("Bạn muốn viết văn à? Mình giúp nhé."), "'Bạn'")
check("a formal 'Tôi' is named",
      register_problem("Xin lỗi. Tôi thực sự rất tiếc."), "'Tôi'")
check("the casual voice is left alone",
      register_problem("mệt quá đi, nghỉ ngơi đi nhé mình"), "")
check("a letter is left alone",
      register_problem("Kính gửi bà mẹ, con xin lỗi bạn. Tôi hứa sẽ cố gắng."),
      "")
check("an English reply is left alone",
      register_problem("That sounds rough. I hope you get some rest."), "")
check("the wrong kinship term for Jade is named",
      kinship_problem("chị ấy cứ ngồi nghỉ đi", "em"), "'Chị'")
check("the right kinship term is left alone",
      kinship_problem("mình giúp em nhé", "em"), "")
check("the wrong kinship term for Damian is named",
      kinship_problem("em ơi, kể mình nghe", "anh"), "'Em'")

# The acknowledgement opener. "Mình hiểu rồi, e mệt quá vì hôm nay
# học nhiều quá" spends its first clause on nothing, and when every reply
# starts that way it is a tic rather than a person.
from guard import acknowledgement_opener, sanitize_reply  # noqa: E402

check("the Vietnamese acknowledgement is named",
      acknowledgement_opener("Mình hiểu rồi, e mệt quá"), "mình hiểu rồi")
check("the English acknowledgement is named",
      acknowledgement_opener("Got it, I'll check."), "got it")
check("a reply that starts with content has no opener",
      acknowledgement_opener("mệt quá à? ngủ sớm đi"), "")
check("the acknowledgement is cut off the front",
      sanitize_reply("Mình hiểu rồi, e mệt quá quá."),
      "e mệt quá quá.")
check("the English acknowledgement is cut off too",
      sanitize_reply("Sure, I can do that."), "I can do that.")
check("a reply that is only an acknowledgement is left alone",
      sanitize_reply("Okay."), "Okay.")
check("a reply with no acknowledgement is untouched",
      sanitize_reply("mệt quá à? ngủ sớm đi nhé mình"),
      "mệt quá à? ngủ sớm đi nhé mình")

# The footer and the status line. Both are things the room reads, and
# both were giving the game away: a "Step 1 -" line and a tool name in
# the footer are the bot talking about its own insides.
check("the footer is just the model",
      agent._footer_for({"model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                         "tools_used": ["web_search"]}),
      "nvidia/nemotron-3.5-super-vl-120b-a12b")
check_true("the status line has no step number",
           "**Step" not in inspect.getsource(agent.run_agent_loop))

# The detached room. It knows the whole picture and shares nothing, and
# the two halves have to stay apart - the write list is what keeps a
# secret in the room it was said in.
from config import is_detached  # noqa: E402
from scopes import scope_plan  # noqa: E402
from presence import ChannelTracker, channel_scope  # noqa: E402

SECRETS_CHANNEL = "1573589377744117881"
MODEL_CHANNEL = "1569657875368902666"

check_true("#secrets is a detached room", is_detached(SECRETS_CHANNEL))
check_true("#model is not", not is_detached(MODEL_CHANNEL))

_tracker = ChannelTracker()
_tracker.note(MODEL_CHANNEL, "1499211380052721684")
_tracker.note(MODEL_CHANNEL, "909453181916938301")

_detached = asyncio.run(scope_plan(SECRETS_CHANNEL, "1499211380052721684",
                                    tracker=_tracker))
check_true("the detached room reads the whole picture",
           channel_scope(MODEL_CHANNEL) in _detached.read
           and "1499211380052721684" in _detached.read)
check("the detached room writes only to itself",
      _detached.write, [channel_scope(SECRETS_CHANNEL)])
check("the detached room's conversation stays there",
      _detached.primary, channel_scope(SECRETS_CHANNEL))

_ordinary = asyncio.run(scope_plan(MODEL_CHANNEL, "1499211380052721684",
                                    tracker=_tracker))
check_true("an ordinary room does not read the detached scope",
           channel_scope(SECRETS_CHANNEL) not in _ordinary.read)
check_true("an ordinary room does not write the detached scope",
           channel_scope(SECRETS_CHANNEL) not in _ordinary.write)
check("an ordinary room reads the person and the room",
      sorted(_ordinary.read),
      sorted(["1499211380052721684", channel_scope(MODEL_CHANNEL)]))

# The write list is a separate argument, because a memory learned in a
# detached room must not go to every scope the reply could read.
check_true("the agent takes the write scopes",
           "write_scopes" in inspect.signature(agent.get_nvidia_response_full).parameters)
check_true("the memory update uses the write scopes",
           "scopes=write_scopes" in inspect.getsource(agent.get_nvidia_response_full))

# The salt. A turn that uses tools several times in a row is either a
# loop or a model that has stopped reading what came back, and the tool
# output it is collecting ends up in the history as though it were fact.
from config import TOOL_SALT_AFTER  # noqa: E402

check("the salt goes in after a few tool rounds", TOOL_SALT_AFTER, 3)
check_true("the salt asks whether the model is repeating itself",
           "Are you repeating yourself?" in agent.TOOL_SALT)
check_true("the salt asks what the model is about to believe",
           "What are you about to believe?" in agent.TOOL_SALT)
check_true("the salt says a tool's output is not fact",
           "not what is true" in agent.TOOL_SALT)
check_true("the salt tells the model to answer",
           "answer now" in agent.TOOL_SALT)
check_true("the salt is not written to the history",
           "db_add_message" not in agent.TOOL_SALT)
check_true("the loop counts the tool rounds",
           "tool_rounds" in inspect.getsource(agent.run_agent_loop))

# The salt has to go in after the tool results, not before. The API needs
# the assistant turn that asked for the tools to be followed by their
# results, and a message in between is a rejected request.
_loop_source = inspect.getsource(agent.run_agent_loop)
_tool_result_at = _loop_source.find('"role": "tool"')
_salt_at = _loop_source.find("TOOL_SALT.format(count=tool_rounds)")
check_true("the salt goes in after the tool results",
           _tool_result_at != -1 and _salt_at != -1
           and _salt_at > _tool_result_at)

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
      extra_body_for("NVIDIA", thinking=False),
      {"chat_template_kwargs": {"enable_thinking": False}})
check("thinking is on by default",
      extra_body_for("NVIDIA", thinking=True),
      {"chat_template_kwargs": {"enable_thinking": True}})
check("MiniMax takes the same switch",
      extra_body_for("MiniMax", thinking=False),
      {"chat_template_kwargs": {"enable_thinking": False}})

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
