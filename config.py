"""Environment configuration and shared clients."""

import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client, create_client

load_dotenv()


# The room is in Vietnam, and every hour the bot reasons about - quiet
# hours, the day it is on, what time it is - has to be the room's hours.
#
# This is set here rather than left to the platform. The container ran in
# UTC with TZ unset, even with the variable written out in render.yaml:
# the env vars in the blueprint never reached it. A bot whose clock is
# seven hours out is a bot whose quiet hours are the wrong hours.
os.environ["TZ"] = os.getenv("ROOM_TIMEZONE", "Asia/Ho_Chi_Minh")
time.tzset()


def _require(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )


# --- LLM providers --------------------------------------------------------
#
# Three models, tried in this order. The first one that answers does the
# work; the others are there for when one is down, rate-limited, or simply
# cannot do what was asked.
#
# 1. Nemotron 3.5 Super VL - the strong one, and it goes first. It is the
#    model that holds the whole picture of a turn: the long tool-calling
#    rounds that need several steps, the reading of a table, the register
#    of a sentence. It honours tool_choice="required" (checked against the
#    live endpoint), and it is the one that can look at a picture.
#
#    The cost is time. On the public endpoint it takes roughly 17s to the
#    first token with thinking off and 26s with it on, so a reply starts
#    late. That is the trade being made knowingly: a better answer that
#    arrives later, rather than a quick one that says the wrong thing.
#
# 2. MiniMax M3.1 Flash - the fallback. It is quick and it is steady, but
#    the endpoint rate-limits it to about four to six requests a minute,
#    so it cannot carry the ordinary traffic. It is here for when
#    Nemotron is down or the endpoint is busy.
#
# 3. Groq - the last resort. A different provider with its own key, so it
#    only runs when NVIDIA's endpoint is out for both of the above.
#
# Laguna XS 2.1 used to be first. It was quick, but it was a code model
# answering a conversation, and it was reading the room worse than the
# general models did, so it is gone from the chain entirely.

MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "minimaxai/minimax-m3.1-flash")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
BOLT_MODEL = os.getenv("BOLT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# What the endpoint will accept, measured against the live API rather than
# taken from a model card:
#
# - Thinking: yes, on both. `chat_template_kwargs.enable_thinking` turns
#   it on and the reasoning comes back in `reasoning_content`, separate
#   from the answer.
# - Output: up to 1,048,576 tokens. Asking for more is refused outright
#   ("Max tokens must not exceed 1048576").
# - The reasoning and the answer come out of the same allowance. A step
#   with a 2048-token budget and a hard question can spend all of it
#   thinking and return an empty answer, so the per-step budget has to
#   leave room for both.
MAX_OUTPUT_TOKENS = 1048576

# Groq is the last fallback in the chain. It has its own key, and it is
# only reached when NVIDIA's endpoint is out for all three of the models
# above - so the bot still answers if one provider dies.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Third-party services -------------------------------------------------
# Daytona provides the sandboxed code execution backend (replaces Piston,
# whose public API became whitelist-only in February 2026).
DAYTONA_API_KEY = os.getenv("DAYTONA_API_KEY")
DAYTONA_API_URL = os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")

LANGSEARCH_API_URL = os.getenv("LANGSEARCH_API_URL", "https://api.langsearch.com/v1/web-search")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")

WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID")
WOLFRAM_API_URL = "https://api.wolframalpha.com/v2/query"

# Kaggle. Either the new API token (KGAT_...) or the legacy
# ~/.kaggle/kaggle.json file. Kaggle provides free CPU, GPU and TPU compute
# plus datasets and trained models; see kaggle.py for what it can and cannot
# do.
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Hugging Face Hub. The token lets the bot see the private and gated
# repositories the account can reach, and gives it the account's rate limit
# rather than the anonymous one.
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

# Which backend the plain `code` tool uses.
#
#   daytona  (default) a fast sandbox: ~0.3s per call, but a bare environment.
#   kaggle   free cloud compute with numpy/pandas/torch and a GPU option, but
#            ~21s per call because every run is a fresh Kaggle session.
#
# `both` leaves the choice to the model, which is usually what you want: the
# sandbox for snippets, Kaggle for heavy work.
CODE_BACKEND = os.getenv("CODE_BACKEND", "both").strip().lower()

# --- Database -------------------------------------------------------------
_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")
_require("SUPABASE_URL", "SUPABASE_KEY")
supabase: Client = create_client(_supabase_url, _supabase_key)

# --- Limits ---------------------------------------------------------------
MAX_AGENT_STEPS = 100        # tool-calling rounds for an interactive reply
JOB_MAX_STEPS = 100          # tool-calling rounds for a background research job

# How many tool-calling rounds in a row are fine before the model is asked
# to check itself.
#
# A turn that uses one or two tools is normal work. A turn that uses five
# in a row is usually one of two things: a loop - the same search, or the
# same question asked a different way - or a model that has stopped
# reading what came back and is gathering instead of thinking.
#
# Both are worth interrupting. The tool results are stored in the
# conversation, so a turn that keeps calling tools also keeps filling the
# history with output that a later turn will read as though it were
# established fact. That is how a bad page becomes a remembered fact.
#
# So after this many consecutive tool rounds, a paragraph goes in front of
# the model: stop, check whether you are repeating yourself, and be careful
# about what you are about to believe.
TOOL_SALT_AFTER = int(os.getenv("TOOL_SALT_AFTER", "3"))

# --- Trying again ----------------------------------------------------------
#
# A turn that produces nothing is not an answer. The provider might be
# rate limited, the network might drop the stream half way through, or
# the model might spend the whole allowance thinking and say nothing.
#
# None of those are worth showing the room: the person who asked is still
# waiting, and "LLM API error" in a chat room is not a reply. So the turn
# is tried again, five seconds later.
#
# There are two kinds of failure, and they are treated differently.
#
# The first is the provider saying "not now": a 429 because too many
# people are asking at once, a 503 because the service is full, a dropped
# connection. Those clear on their own - a rate limit is a matter of
# seconds, and a busy service is a matter of a minute - so the turn keeps
# trying, five seconds apart, for as long as it takes. A bad key or a
# model that does not exist is not one of these, and waiting does not help
# with that, so it is given up on at once.
#
# The second is a turn that got no answer out of the model itself: it
# said nothing, or it used the whole budget thinking. That is tried again
# a few times, because a model that says nothing usually says something
# the next time, but it is not tried for ever.
ANSWER_RETRY_DELAY = float(os.getenv("ANSWER_RETRY_DELAY", "5"))
ANSWER_RETRY_ATTEMPTS = int(os.getenv("ANSWER_RETRY_ATTEMPTS", "3"))

# How long the turn keeps waiting out a provider that says "not now".
# Zero means for as long as it takes, which is what it does by default.
ANSWER_RETRY_MAX_WAIT = float(os.getenv("ANSWER_RETRY_MAX_WAIT", "0"))

# --- Proactive talking ----------------------------------------------------
#
# The bot answers when it is spoken to. This is the other half: saying
# something when nobody asked - the way a person in a room does.
#
# The model for this is Neuro-sama, and the number that matters is hers:
# PATIENCE, sixty seconds of silence, and then the prompt asks the model
# to say something. That is the whole trick. Not a schedule, not a daily
# allowance - a short attention span and something to say.
#
# So there is no cap on how much the bot talks and no quiet window. It
# checks every minute, and the small model decides whether there is
# anything worth saying; most of the time there is not, and the bot is
# quiet. When there is, it says one line.
#
# The one thing that still holds is the backoff: a message nobody
# answered makes the next one wait longer, because a bot that keeps
# talking into a silence is the worst kind of bot.

PROACTIVE_ENABLED = (os.getenv("PROACTIVE_ENABLED", "1").strip().lower()
                     not in ("0", "false", "no"))

# How often the bot looks for a moment to speak. This is a check, not a
# schedule: most of these checks end in silence.
PROACTIVE_TICK_SECONDS = int(os.getenv("PROACTIVE_TICK_SECONDS", "60"))

# The shortest gap between two unprompted messages in one channel. It is
# short - two ticks cannot land on top of each other - and it grows when
# the messages go unanswered.
PROACTIVE_MIN_GAP_MINUTES = int(os.getenv("PROACTIVE_MIN_GAP_MINUTES", "2"))

# The most unprompted messages in one channel in one day. Zero means no
# limit at all, which is what it is: the model decides whether to speak,
# not a counter.
PROACTIVE_MAX_PER_DAY = int(os.getenv("PROACTIVE_MAX_PER_DAY", "0"))

# The room has to have been quiet for at least this long before the bot
# considers joining in - otherwise it is talking over a live conversation.
# A minute of silence is the number Neuro-sama uses.
PROACTIVE_MIN_IDLE_SECONDS = int(os.getenv("PROACTIVE_MIN_IDLE_SECONDS", "60"))

# ...and not for longer than this, or the room is simply dead and the
# bot would be talking to nobody. Zero means no limit at all: the model
# decides whether there is something to say, and a room that has been
# quiet for a day is still a room. It used to be 1440 - a day - and that
# meant the bot went permanently quiet in any room that nobody happened
# to be in that evening.
PROACTIVE_MAX_IDLE_MINUTES = int(os.getenv("PROACTIVE_MAX_IDLE_MINUTES", "0"))

# The hours of the day when the bot does not start anything. "0-7" is
# from midnight to seven; "22-3" wraps around midnight. Empty means no
# quiet hours at all: the bot talks whenever it has something to say.
PROACTIVE_QUIET_HOURS = os.getenv("PROACTIVE_QUIET_HOURS", "")

# How long after an unprompted message the bot still treats the next
# human message as a reply to it.
PROACTIVE_REPLY_WINDOW_MINUTES = int(os.getenv("PROACTIVE_REPLY_WINDOW_MINUTES", "45"))

# The longest a proactive message may be. It is one line, and it has to
# be easy to ignore.
PROACTIVE_MAX_LINE_CHARS = int(os.getenv("PROACTIVE_MAX_LINE_CHARS", "160"))

# The channel the server's news digest is posted in. Lain reads it so
# she knows what happened today - the knowledge changes what she can
# say. It is not a channel she posts in: the news belongs to the room
# that posted it.
#
# This is #general, where the "Face's attic" bot posts its digest. The
# channel called #news is a tweet feed with empty message bodies, so
# there is nothing there to read.
NEWS_CHANNEL_ID = os.getenv("NEWS_CHANNEL_ID", "1568478571872133140")

# The bot that posts the digest, so the news can be told apart from
# somebody talking in the same channel. The format is checked as well,
# so an unfamiliar author is still usable.
NEWS_DIGEST_AUTHOR_ID = os.getenv("NEWS_DIGEST_AUTHOR_ID", "1570220594858041365")

# How old a news post may be and still count as something Lain knows
# about today.
NEWS_MAX_AGE_HOURS = float(os.getenv("NEWS_MAX_AGE_HOURS", "36"))

# How many conversation rows to pull from the database. The whole conversation
# is kept and compacted when it gets large, rather than being truncated.
HISTORY_ROWS = 300

# Character budget for the messages sent to the model. The Nemotron 3.5 Super
# VL window is 262k tokens; staying well under it leaves room for the reply.
# 300k characters is roughly 75k tokens, which leaves plenty of headroom and
# means the summary step only runs once the history genuinely outgrows it
# (around 550 rows) instead of on every single turn.
CONTEXT_CHAR_BUDGET = 300_000

# When compacting, keep this much of the most recent conversation verbatim and
# summarise everything before it.
KEEP_RECENT_CHARS = 30_000
JOB_TICK_SECONDS = 100       # how often a running job reports progress

# --- Clients --------------------------------------------------------------
if not NVIDIA_API_KEY:
    raise SystemExit(
        "NVIDIA_API_KEY must be set - both MiniMax and Nemotron are served "
        "from NVIDIA's endpoint. Copy .env.example to .env and fill in the "
        "values."
    )

nvidia_client = AsyncOpenAI(base_url=NVIDIA_API_URL, api_key=NVIDIA_API_KEY or "not-used")
groq_client = (
    AsyncOpenAI(base_url=GROQ_API_URL, api_key=GROQ_API_KEY or "not-used")
    if GROQ_API_KEY
    else None
)


def provider_chain() -> list[tuple[str, AsyncOpenAI, str]]:
    """Providers to try in order: Nemotron, MiniMax, Groq.

    Each entry is (name, client, model). The name is what
    `extra_body_for` keys off, so "NVIDIA", "MiniMax" and "Groq" must
    all be recognised there.

    The order is deliberate. Nemotron 3.5 Super VL goes first because it
    is the one that holds the whole picture of a turn: the long
    tool-calling rounds, the reading of a table, the register of a
    sentence. It is also the only one in the chain that can read a
    picture, so a turn with an image goes to it directly (see
    `vision_models`). MiniMax M3.1 Flash is next: quick and steady, but
    the endpoint rate-limits it to four to six requests a minute, so it
    is a fallback rather than the thing carrying the traffic. Groq is the
    last resort - a different provider, with its own key, so it only runs
    when NVIDIA's endpoint is out for both of the models above.

    Groq is only in the chain when its key is configured; without one
    there is nothing to fall back to.
    """
    chain = [("NVIDIA", nvidia_client, NVIDIA_MODEL),
             ("MiniMax", nvidia_client, MINIMAX_MODEL)]
    if groq_client is not None:
        chain.append(("Groq", groq_client, GROQ_MODEL))
    return chain


def vision_models() -> list[tuple[str, AsyncOpenAI, str]]:
    """The chain for a turn that has a picture in it.

    MiniMax cannot look at images, so a turn with an attachment goes
    straight to Nemotron, which can. Groq's llama-3.3 is text only, so
    it is no use here either.
    """
    return [("NVIDIA", nvidia_client, NVIDIA_MODEL)]

def extra_body_for(provider_name: str, thinking: bool = True) -> dict:
    """Provider-specific request options.

    NVIDIA's Nemotron models can be asked to reason before they answer.
    The switch is `chat_template_kwargs.enable_thinking` - a top-level
    `enable_thinking` is rejected with "Unsupported parameter(s)". The
    reasoning comes back in its own field, so it does not end up in the
    reply.

    `thinking=False` is for the small calls that need a direct answer - the
    presence classifier, for instance, which has one character to say.

    MiniMax and Bolt are also served from NVIDIA's endpoint and take the
    same flag, so they are treated like NVIDIA here. Both were checked
    against the live endpoint: with thinking on, they reason in
    `reasoning_content` and answer in `content`, exactly like Nemotron
    does.
    """
    if provider_name in ("NVIDIA", "MiniMax", "BOLT"):
        return {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    return {}


# --- The bot's owner -------------------------------------------------------
# The owner is the one person the bot was built for, and the one whose
# context it keeps. Everything else about the people in the chat is worked
# out at runtime from the channel itself - see people.py.
OWNER_DISCORD_ID = os.getenv("OWNER_DISCORD_ID", "")
OWNER_TAG = os.getenv("OWNER_TAG", "damianlewis479")
OWNER_NAME = os.getenv("OWNER_NAME", "Damian")


# --- The room's kinship terms ----------------------------------------------
#
# The room talks in Vietnamese kinship terms. Damian calls himself "a" and
# Jade calls herself "e", so Damian is "anh" and Jade is "em" - and the bot
# calls them the same words.
#
# Getting one wrong is worse than not using one: calling Jade "anh" or
# "chị" is the kind of mistake a person does not make about somebody they
# know. So the prompt says which term belongs to whom, and the guard
# checks the reply for one that does not.
#
# The terms are keyed by Discord id and the ids are configuration, so
# nobody has to be edited into the code, and somebody new in the room has
# no entry and simply gets no kinship term at all.
DAMIAN_DISCORD_ID = (os.getenv("DAMIAN_DISCORD_ID")
                     or os.getenv("OWNER_DISCORD_ID")
                     or "1499211380052721684")
JADE_DISCORD_ID = os.getenv("JADE_DISCORD_ID") or "909453181916938301"

# person id -> (the term the bot uses for them, the shorter spellings)
KINSHIP_TERMS = {
    DAMIAN_DISCORD_ID: ("anh", ("a",)),
    JADE_DISCORD_ID: ("em", ("e",)),
}

# Every kinship term the room uses. The guard uses this list to spot a
# term that does not belong in a reply.
KINSHIP_WORDS = ("anh", "em", "chị")


def kinship_for(person_id) -> tuple[str, tuple[str, ...]] | None:
    """The kinship term for somebody, if the room has one for them."""
    if not person_id:
        return None
    entry = KINSHIP_TERMS.get(str(person_id))
    if not entry:
        return None
    return entry[0], tuple(entry[1])


# --- Detached channels ----------------------------------------------------
#
# A detached channel knows everything and shares nothing.
#
# The bot's memory is a set of scopes: one per person, one per room. A
# reply reads the scopes of the room it is in, so a room only ever knows
# what was said in it.
#
# Some rooms need to know more than their own. A private room may need
# the whole picture - every channel, every person - to do its job. But
# what is learned there must not travel: the memories that come out of
# a detached channel stay in it, and no other room's reply ever loads
# them.
#
# This is the list of channels that get the whole picture. It is
# configuration, so a channel can be detached without touching the
# code, and nothing else is detached by accident.
def _id_list(raw: str) -> set[str]:
    """A comma-separated list of Discord ids."""
    return {part.strip() for part in (raw or "").split(",") if part.strip()}


DETACHED_CHANNELS = _id_list(os.getenv("DETACHED_CHANNEL_IDS", ""))
DETACHED_CHANNELS |= _id_list(os.getenv("SECRETS_CHANNEL_ID", ""))
# #secrets - the private room. It knows the whole picture; the rest of
# the bot knows nothing about what happens in it.
DETACHED_CHANNELS.add("1573589377744117881")


def is_detached(channel_id) -> bool:
    """Does this room know the whole picture and share nothing?"""
    if not channel_id:
        return False
    return str(channel_id) in DETACHED_CHANNELS
