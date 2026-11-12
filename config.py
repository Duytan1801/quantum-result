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

# --- Database -------------------------------------------------------------
_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")
_require("SUPABASE_URL", "SUPABASE_KEY")
supabase: Client = create_client(_supabase_url, _supabase_key)

# --- Limits ---------------------------------------------------------------
MAX_AGENT_STEPS = 50         # tool-calling rounds for an interactive reply
JOB_MAX_STEPS = 50           # tool-calling rounds for a background research job

# How many conversation rows to pull from the database. The whole conversation
# is kept and compacted when it gets large, rather than being truncated.
HISTORY_ROWS = 200

# Character budget for the messages sent to the model. The Nemotron 3.5 Super
# VL window is 262k tokens; staying well under it leaves room for the reply.
CONTEXT_CHAR_BUDGET = 120_000

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
    """Providers to try in order: NVIDIA first, then Groq as a fallback."""
    chain = [("NVIDIA", nvidia_client, NVIDIA_MODEL)]
    if groq_client is not None:
        chain.append(("Groq", groq_client, GROQ_MODEL))
    return chain
