"""The environment the tests run in.

The tests need real keys - the bot talks to a real model and a real
database - but a key does not belong in a file that goes up to GitHub.
So they are read from the environment, and the .env file is loaded if
there is one, the same way the bot loads it.

    export NVIDIA_API_KEY=... SUPABASE_KEY=... KAGGLE_API_TOKEN=...
    python tests/regress.py

A test that needs a key and cannot find it says so and stops, rather
than failing in a way that looks like a bug in the bot.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_dotenv() -> None:
    """Read the .env file, if there is one. Nothing is overwritten."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def need(*names: str) -> None:
    """Stop with a clear message if a key the test needs is missing."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        sys.exit(
            "these environment variables are not set: "
            + ", ".join(missing)
            + "\nSet them, or put them in a .env file next to the bot, and "
            "run the test again."
        )


load_dotenv()
