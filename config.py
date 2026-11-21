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
# Three models, tried in this order. The first one that answers does the
# work; the others are there for when one is down, rate-limited, or simply
# cannot do what was asked.
#
# 1. MiniMax M3.1 Flash - the fast one. It answers in about a second and
#    handles tool calls, thinking and one-character answers, so it is the
#    default for everything.
# 2. Nemotron 3.5 Super VL - the strong one. Slower, but better at the long
#    tool-calling turns that need several steps, and it is the only one that
#    can look at a picture.
# 3. Groq - the last resort, in case NVIDIA as a whole is unreachable.
#
# MiniMax and Nemotron are both served from NVIDIA's endpoint, so they share
# one client and one key. Only the model name changes between them.

MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "minimaxai/minimax-m3.1-flash")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
NVIDIA_API_URL = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Groq is an optional fallback provider (used when NVIDIA is unavailable).
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
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
if not NVIDIA_API_KEY and not GROQ_API_KEY:
    raise SystemExit(
        "NVIDIA_API_KEY (or GROQ_API_KEY) must be set. Copy .env.example to .env "
        "and fill in the values."
    )

nvidia_client = AsyncOpenAI(base_url=NVIDIA_API_URL, api_key=NVIDIA_API_KEY or "not-used")
groq_client = (
    AsyncOpenAI(base_url=GROQ_API_URL, api_key=GROQ_API_KEY or "not-used")
    if GROQ_API_KEY
    else None
)


def provider_chain() -> list[tuple[str, AsyncOpenAI, str]]:
    """Providers to try in order: MiniMax first, then Nemotron, then Groq.

    Each entry is (name, client, model). The name is what
    `extra_body_for` keys off, so "MiniMax" and "NVIDIA" must both be
    recognised there.

    The order is deliberate. MiniMax M3.1 Flash is the quickest of the
    three and handles tool calls and thinking perfectly well, so it does
    the ordinary work. Nemotron is the slower, stronger model and stays in
    the chain for the long multi-step turns - and because it is the one
    that can read a picture, a turn with an image is sent to it directly
    (see `vision_models`).
    """
    chain = [("MiniMax", nvidia_client, MINIMAX_MODEL),
             ("NVIDIA", nvidia_client, NVIDIA_MODEL)]
    if groq_client is not None:
        chain.append(("Groq", groq_client, GROQ_MODEL))
    return chain


def vision_models() -> list[tuple[str, AsyncOpenAI, str]]:
    """The chain for a turn that has a picture in it.

    MiniMax M3.1 Flash cannot look at images, so a turn with an attachment
    goes straight to Nemotron, which can. If Nemotron is down, Groq gets a
    turn with the picture described as text rather than no answer at all.
    """
    chain = [("NVIDIA", nvidia_client, NVIDIA_MODEL)]
    if groq_client is not None:
        chain.append(("Groq", groq_client, GROQ_MODEL))
    return chain


def extra_body_for(provider_name: str, thinking: bool = True) -> dict:
    """Provider-specific request options.

    NVIDIA's Nemotron models can be asked to reason before they answer.
    The switch is `chat_template_kwargs.enable_thinking` - a top-level
    `enable_thinking` is rejected with "Unsupported parameter(s)". The
    reasoning comes back in its own field, so it does not end up in the
    reply. Groq does not know the flag, so it is only ever sent to NVIDIA.

    `thinking=False` is for the small calls that need a direct answer - the
    presence classifier, for instance, which has one character to say.

    MiniMax is served from NVIDIA's endpoint and takes the same flag, so it
    is treated like NVIDIA here. It was checked against the live endpoint:
    with thinking on, it reasons in `reasoning_content` and answers in
    `content`, exactly like Nemotron does.
    """
    if provider_name in ("NVIDIA", "MiniMax"):
        return {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    return {}


# --- The bot's owner -------------------------------------------------------
# The owner is the one person the bot was built for, and the one whose
# context it keeps. Everything else about the people in the chat is worked
# out at runtime from the channel itself - see people.py.
OWNER_DISCORD_ID = os.getenv("OWNER_DISCORD_ID", "")
OWNER_TAG = os.getenv("OWNER_TAG", "damianlewis479")
OWNER_NAME = os.getenv("OWNER_NAME", "Damian")
