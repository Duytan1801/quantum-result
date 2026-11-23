"""Environment configuration and shared clients."""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import Client, create_client

load_dotenv()


def _require(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )


# --- LLM providers --------------------------------------------------------
#
# Four models, tried in this order. The first one that answers does the
# work; the others are there for when one is down, rate-limited, or simply
# cannot do what was asked.
#
# 1. Laguna XS 2.1 (poolside) - a code model, and it goes first. It
#    answers in about a second, it handles tool calls, thinking and
#    one-character answers, and it is nearly as good as Nemotron on the
#    work. Speed is the reason it is first.
# 2. MiniMax M3.1 Flash - the general model. This is the one that does
#    the tasks and the research: it is the steadier of the two at the
#    conversational register, where a reply has to sound like a person.
# 3. Nemotron 3.5 Super VL - the strong one. Slower, but better at the
#    long tool-calling turns that need several steps, and it is the only
#    one that can look at a picture.
# 4. Groq - the last resort. A different provider with its own key, so it
#    only runs when NVIDIA's endpoint is out for all three of the above.
#
# The first three are served from NVIDIA's endpoint, so they share one
# client and one key. Only the model name changes between them.

LAGUNA_MODEL = os.getenv("LAGUNA_MODEL", "poolside/laguna-xs-2.1")
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
    """Providers to try in order: Laguna, MiniMax, Nemotron, Groq.

    Each entry is (name, client, model). The name is what
    `extra_body_for` keys off, so "Laguna", "MiniMax", "NVIDIA" and
    "Groq" must all be recognised there.

    The order is deliberate. Laguna XS 2.1 goes first because it is
    quick - it answers in about a second - and it can call tools, think,
    and handle everything the bot does, so it does the ordinary work.
    MiniMax M3.1 Flash is next: the general model, and the steadier of
    the two at the conversational register, where a reply has to sound
    like a person. Nemotron 3.5 Super VL is third: slower and stronger,
    it covers the long multi-step turns, and because it is the one that
    can read a picture, a turn with an image is sent to it directly (see
    `vision_models`). Groq is the last resort - a different provider,
    with its own key, so it only runs when NVIDIA's endpoint is out for
    all three of the models above.

    Groq is only in the chain when its key is configured; without one
    there is nothing to fall back to.
    """
    chain = [("Laguna", nvidia_client, LAGUNA_MODEL),
             ("MiniMax", nvidia_client, MINIMAX_MODEL),
             ("NVIDIA", nvidia_client, NVIDIA_MODEL)]
    if groq_client is not None:
        chain.append(("Groq", groq_client, GROQ_MODEL))
    return chain


def vision_models() -> list[tuple[str, AsyncOpenAI, str]]:
    """The chain for a turn that has a picture in it.

    Neither Laguna nor MiniMax can look at images, so a turn with an
    attachment goes straight to Nemotron, which can.
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

    Laguna, MiniMax and Bolt are all served from NVIDIA's endpoint and take the
    same flag, so they are treated like NVIDIA here. Both were checked
    against the live endpoint: with thinking on, they reason in
    `reasoning_content` and answer in `content`, exactly like Nemotron
    does.
    """
    if provider_name in ("NVIDIA", "MiniMax", "Laguna", "BOLT"):
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
