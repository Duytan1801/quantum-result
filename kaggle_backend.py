"""Kaggle as a compute backend.

Kaggle runs code as a *notebook kernel*: you push a script, Kaggle runs it as
a batch session on its own workers, and you collect the logs and the files it
wrote. That is a very different shape from a sandbox you call synchronously,
and it has one big consequence: there is no warm session to reuse, so every
call costs about twenty seconds.

Measured on the same trivial script (Nov 2026):

    Daytona   2.9s cold, 0.3s warm
    Kaggle   21.0s every time

What that buys you, and what it costs:

  * free CPU (unlimited), free GPU (~30h/week) and TPU (20h/week);
  * numpy, pandas, scikit-learn, torch and friends already installed;
  * free access to Kaggle's datasets and trained models;
  * but ~21s per run, the code has to be a whole script rather than a
    snippet, and two runs cannot share a kernel.

So Kaggle is not a drop-in replacement for a fast sandbox - it is a place to
do the heavy work: train a model on a real dataset, or run something that
needs a GPU, and it fits the bot's background job model perfectly.
"""

import asyncio
import base64
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time

from config import KAGGLE_API_TOKEN

# The Kaggle SDK is synchronous and pulls in a lot; it is only imported when
# Kaggle is actually used, so the bot still starts if it is not installed.
#
# The module is called kaggle_backend rather than kaggle because a file called
# kaggle.py in the working directory would shadow the `kaggle` package that
# the SDK imports.
_api = None
_api_failed = False
_api_error = ""

# The Kaggle credentials are read from the environment (KAGGLE_API_TOKEN),
# from the legacy username/key pair, or from ~/.kaggle/kaggle.json.
# They are deliberately not in this file: a token that lives in the source
# is a token that ends up in the repository, in every copy of it, and in
# every screenshot of it.


def _find_token() -> tuple[str, str]:
    """Look for a Kaggle API token. Returns (token, where it came from)."""
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_TOKEN", "KAGGLE_ACCESS_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name
    return "", ""


def _has_legacy_key() -> bool:
    """The old username + key pair, which the SDK reads on its own."""
    user = os.getenv("KAGGLE_USERNAME") or os.getenv("KAGGLE_USER")
    key = os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_KEY")
    return bool(user and key)


def _has_config_file() -> bool:
    return os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))


def credentials_source() -> str:
    """Where the Kaggle credentials come from, for the startup banner."""
    token, name = _find_token()
    if token:
        return name
    if _has_legacy_key():
        return "KAGGLE_USERNAME + KAGGLE_KEY"
    if _has_config_file():
        return "~/.kaggle/kaggle.json"
    return ""


# The system prompt carries a one-line statement about whether Kaggle works
# right now. Checking auth live costs a round trip to Kaggle and a slow SDK
# import, so the hint is cached: a fresh value is returned directly, a stale
# one is returned immediately and refreshed in a background thread, and only
# the very first call (before any value exists) waits for the network.
_HINT_TTL = 600.0
_hint_lock = threading.Lock()
_hint_cache = {"ts": 0.0, "text": None}


def force_refresh() -> None:
    """Clear any cached failure state so the next call retries from scratch.

    Call this after the user has set KAGGLE_API_TOKEN (or restarted Render)
    so a previous failed attempt doesn't keep returning None.
    """
    global _api, _api_failed, _api_error
    _api = None
    _api_failed = False
    _api_error = ""
    _clear_failed_kaggle_imports()
    with _hint_lock:
        _hint_cache["ts"] = 0.0
        _hint_cache["text"] = None


def kaggle_status() -> dict:
    """A plain-English report of why Kaggle is or isn't available.

    Use this to debug: it checks the env var, the legacy key pair, the
    config file, and a live authentication attempt.

    Two different questions are answered separately, because Kaggle
    answers them separately: is the credential accepted at all, and is
    it allowed to run kernels. A token that can read datasets but is
    refused for kernels is a token that has to be re-issued.
    """
    token, name = _find_token()
    env_set = bool(os.getenv("KAGGLE_API_TOKEN"))
    report = {
        "env_KAGGLE_API_TOKEN_set": env_set,
        "credentials_source": credentials_source() or "none",
        "legacy_key_present": _has_legacy_key(),
        "config_file_present": _has_config_file(),
        "authenticated": False,
        "kernels_allowed": False,
        "error": None,
    }
    if not token and not _has_legacy_key() and not _has_config_file():
        report["error"] = "No Kaggle credentials found anywhere."
        return report
    try:
        force_refresh()
        client = _kaggle_api()
        if client is None:
            report["error"] = _api_error or "authentication returned no client"
            return report

        # Is the credential itself accepted? Reading the datasets is the
        # cheapest call that does not touch the kernels service.
        try:
            client.dataset_list(search="titanic", max_size=1)
            report["authenticated"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            return report

        # And is it allowed to run kernels? That is a different service
        # with a different answer.
        try:
            client.quota_view()
            report["kernels_allowed"] = True
        except Exception as exc:
            if _is_auth_refusal(exc):
                report["error"] = AUTH_REFUSED
            else:
                report["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _hint_from_status(report: dict) -> str:
    """Turn a kaggle_status() report into one prompt-sized line."""
    if report.get("kernels_allowed"):
        source = report.get("credentials_source") or "unknown source"
        return (
            f"Kaggle is configured and working - live authentication succeeded "
            f"(credentials from {source}), so kaggle_run and kaggle_datasets "
            f"are available for heavy compute."
        )
    if report.get("authenticated"):
        # The credential is fine, but the kernels service refuses it.
        return (
            "Kaggle credentials are accepted but are NOT allowed to run "
            "kernels - the token has to be re-issued from the Kaggle "
            "account settings with permission to run code. kaggle_run will "
            "fail until it is; kaggle_datasets still works."
        )
    error = report.get("error") or "unknown error"
    return (
        f"Kaggle is currently NOT working ({error}). If kaggle_run is needed, "
        f"try it once and report the actual error, and fall back to the code "
        f"tool for anything that does not need the network."
    )


def status_hint(force: bool = False, report: dict = None) -> str:
    """One line for the system prompt: can Kaggle be used right now?

    A stale value is still returned (and refreshed in the background) so the
    prompt is never delayed by a network round trip. A pre-computed status
    report can be passed in to avoid a second live check.
    """
    with _hint_lock:
        age = time.monotonic() - _hint_cache["ts"]
        cached = _hint_cache["text"]

    if cached is not None and (age < _HINT_TTL or not force):
        if age >= _HINT_TTL:
            threading.Thread(target=status_hint, kwargs={"force": True},
                             daemon=True).start()
        return cached

    report = report or kaggle_status()
    hint = _hint_from_status(report)
    with _hint_lock:
        _hint_cache["ts"] = time.monotonic()
        _hint_cache["text"] = hint
    return hint



NOT_CONFIGURED = (
    "Kaggle is not configured, so kaggle_run cannot run anything. Set the "
    "KAGGLE_API_TOKEN environment variable to your Kaggle API token (the "
    "KGAT_... one from your account's API tokens page) and redeploy. Until "
    "then, use the code tool for anything that does not need the network."
)


def _clear_failed_kaggle_imports():
    """Remove half-imported kaggle modules from sys.modules.

    The kaggle package has an import cycle: kaggle/__init__.py does
    `from kaggle.api.kaggle_api_extended import KaggleApi`, and that module
    imports `six`, which kaggle does not declare as a dependency. When the
    import fails half way through, Python leaves the partly initialised
    module in sys.modules, and every later attempt then fails with

        cannot import name 'KaggleApi' from partially initialized module
        'kaggle.api.kaggle_api_extended' (most likely due to a circular
        import)

    which hides the real error. Clearing them makes each attempt start
    clean, so the actual cause is reported instead of a confusing one.
    """
    import sys

    for name in [n for n in sys.modules
                 if n == "kaggle" or n.startswith("kaggle.")]:
        sys.modules.pop(name, None)


def _kaggle_api():
    """Get an authenticated Kaggle client, or None if it is not set up."""
    global _api, _api_failed, _api_error
    if _api is not None:
        return _api

    token, name = _find_token()
    if token:
        # The SDK only looks at KAGGLE_API_TOKEN, so an alias the user set
        # (KAGGLE_TOKEN, KAGGLE_ACCESS_TOKEN) is normalised into it here.
        os.environ["KAGGLE_API_TOKEN"] = token
        # If a previous attempt failed - e.g. the six module was missing on
        # Render's Python 3.14, or a transient network error - clear it and
        # retry now that credentials are present. We only cache failure
        # permanently when there are genuinely no credentials at all.
        if _api_failed:
            _api_failed = False
            _api_error = ""
            _clear_failed_kaggle_imports()
    elif not (_has_legacy_key() or _has_config_file()):
        _api_error = NOT_CONFIGURED
        _api_failed = True
        return None

    try:
        # Start from a clean slate: a previous failed import can leave a
        # half-built module behind that poisons every later attempt.
        _clear_failed_kaggle_imports()

        from kaggle.api.kaggle_api_extended import KaggleApi

        client = KaggleApi()
        # The SDK calls exit(1) when it cannot authenticate, which would kill
        # the whole bot. Turn that into a normal error instead.
        try:
            client.authenticate()
        except SystemExit:
            raise RuntimeError("the Kaggle API rejected the credentials")
        _api = client
        _api_failed = False
        _api_error = ""
    except Exception as exc:
        _clear_failed_kaggle_imports()
        _api_error = f"could not set up the Kaggle client: {exc}"
        print(_api_error)
        # Not cached as a permanent failure: the next call tries again, so a
        # transient network problem does not disable Kaggle for the session.
        return None
    return _api


def available() -> bool:
    """Is Kaggle configured and reachable?"""
    client = _kaggle_api()
    if client is None:
        return False
    try:
        client.competitions_list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Kernels: the slug pool
# ---------------------------------------------------------------------------
# A kernel can only run one session at a time - pushing a second version
# while the first is running cancels it. So runs are spread over a small
# pool of kernels, and a kernel is only used when it is idle.
POOL_SIZE = 3
POOL_PREFIX = "lain2-bot"

# The pool is guarded by a threading lock, not an asyncio one: the runs
# happen in worker threads, and a check-then-add that is not atomic lets two
# runs pick the same kernel - which cancels the first one, so it never
# finishes and the second waits forever.
_pool_lock = threading.Lock()
_pool_busy: set[str] = set()


def _pool_slug() -> str:
    """A free kernel slug from the pool."""
    with _pool_lock:
        for i in range(POOL_SIZE):
            slug = f"{POOL_PREFIX}-{i}"
            if slug not in _pool_busy:
                _pool_busy.add(slug)
                return slug
    # Everything is busy. Rather than push to a kernel that is already
    # running - which would cancel that run - wait for one to free up.
    raise RuntimeError(
        f"all {POOL_SIZE} Kaggle kernels are busy; try again in a moment"
    )


def _release_slug(slug: str):
    with _pool_lock:
        _pool_busy.discard(slug)


def _owner() -> str:
    client = _kaggle_api()
    if client is None:
        return ""
    try:
        return client.get_config_value("username") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
#
# A dataset is mounted by its full ref - "owner/slug" - and nothing else
# works. A model asked for a dataset usually says "titanic" or
# "imdb-movies", which is a name and not a ref, and a kernel pushed with
# a name that does not exist comes up without the data it was supposed
# to have.
#
# So a bare name is looked up and turned into a real ref. If there is no
# dataset by that name, the run says so instead of coming up empty.

def _find_dataset(name: str) -> str:
    """The full ref of the dataset that matches this name."""
    client = _kaggle_api()
    if client is None:
        return ""
    want = (name or "").strip().lower()
    if not want:
        return ""
    try:
        response = _quiet(client.dataset_list, search=want)
    except Exception as exc:
        print(f"could not look the dataset up: {exc}")
        return ""

    best = ""
    for d in _items(response, "datasets", "items"):
        ref = getattr(d, "ref", "") or ""
        slug = ref.split("/")[-1].lower()
        title = (getattr(d, "title", "") or "").lower()
        if slug == want or title == want:
            return ref
        if not best and (want in slug or want in title):
            best = ref
    return best


def _resolve_datasets(refs: list) -> tuple[list, list]:
    """Turn the dataset names into refs they can actually be mounted by.

    Returns (refs, notes). The notes say what was changed and what could
    not be found, so the model is told rather than left guessing.
    """
    resolved, notes = [], []
    for ref in refs or []:
        ref = str(ref).strip()
        if not ref:
            continue
        if "/" in ref:
            resolved.append(ref)
            continue
        found = _find_dataset(ref)
        if found:
            resolved.append(found)
            notes.append(f"dataset '{ref}' is {found}")
        else:
            notes.append(f"no dataset called '{ref}' was found, so it was not mounted")
    return resolved, notes


# ---------------------------------------------------------------------------
# The weekly quota
# ---------------------------------------------------------------------------
# A GPU run that has no quota left fails inside the kernel, with a
# message about the quota rather than about the code. Better to say so
# before the twenty seconds are spent.
#
# The same call answers another question: whether the credential is
# allowed to run kernels at all. Kaggle's kernels service refuses a
# token it does not recognise with a 401, while the datasets service
# keeps working - so a token that can read datasets and cannot run a
# kernel is a token that has to be re-issued, and the run is going to
# fail however many times it is tried.

AUTH_REFUSED = (
    "Kaggle refused the credentials for kernel runs (401). The token can "
    "still read datasets, but it is not accepted for running code - it has "
    "to be re-issued from the Kaggle account settings with the permission "
    "to run kernels, and KAGGLE_API_TOKEN updated. Nothing is wrong with "
    "the script."
)


def _is_auth_refusal(exc: Exception) -> bool:
    """Did Kaggle refuse the credential?"""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 401:
        return True
    text = str(exc).lower()
    return "401" in text or "unauthenticated" in text or "unauthorized" in text


def _seconds(text) -> int:
    """Kaggle's durations - "3600s", "1h 2m 3s" - as seconds."""
    if text is None:
        return 0
    if isinstance(text, (int, float)):
        return int(text)
    text = str(text).strip()
    if not text:
        return 0
    total = 0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms]?)", text):
        value = float(value)
        if unit == "h":
            total += value * 3600
        elif unit == "m":
            total += value * 60
        else:
            total += value
    return int(total)


def _quota_data() -> tuple[dict, str]:
    """The weekly quota, or a note about why there isn't any.

    The quota comes from the kernels service, so a refusal here is the
    credential being refused, not a quota problem.
    """
    client = _kaggle_api()
    if client is None:
        return {}, ""
    try:
        quota = _quiet(client.quota_view)
        data = quota.to_dict() if hasattr(quota, "to_dict") else {}
        return data, ""
    except Exception as exc:
        if _is_auth_refusal(exc):
            return {}, AUTH_REFUSED
        # Not knowing is not the same as knowing there is none left.
        print(f"could not read the quota: {exc}")
        return {}, ""


def _quota_problem(accelerator: str) -> str:
    """Why this accelerator cannot be used right now, or "" if it can."""
    accelerator = (accelerator or "").strip().upper()
    data, problem = _quota_data()
    if problem:
        return problem
    if not accelerator:
        return ""

    if accelerator.startswith("NVIDIA"):
        gpu = data.get("gpuQuota") or {}
        used = _seconds(gpu.get("timeUsed"))
        total = _seconds(gpu.get("totalTimeAllowed"))
    elif accelerator.startswith("TPU"):
        tpu = data.get("tpuQuota") or {}
        used = _seconds(tpu.get("timeUsed"))
        total = _seconds(tpu.get("totalTimeAllowed"))
    else:
        return ""

    if total and used >= total:
        kind = "GPU" if accelerator.startswith("NVIDIA") else "TPU"
        return (f"the weekly {kind} quota is used up "
                f"({used // 3600}h {used % 3600 // 60}m of "
                f"{total // 3600}h {total % 3600 // 60}m), so a {kind} run "
                f"would not start. Run it on the CPU, or wait for the "
                f"quota to refresh.")
    return ""


# ---------------------------------------------------------------------------
# Running code
# ---------------------------------------------------------------------------
def _metadata(slug: str, owner: str, accelerator: str, datasets: list,
              title: str) -> dict:
    """The kernel-metadata.json that gets pushed.

    `enable_internet` is always on: Kaggle kernels have no network access by
    default, and every script that pulls live data needs it.
    """
    accelerator = (accelerator or "").strip()
    meta = {
        "id": f"{owner}/{slug}" if owner else slug,
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": accelerator.upper().startswith("NVIDIA"),
        "enable_tpu": accelerator.upper().startswith("TPU"),
        "enable_internet": True,
        "dataset_sources": datasets or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if accelerator:
        meta["machine_shape"] = accelerator
    return meta


def _internet_is_on(client, ref: str) -> bool:
    """Check the pushed kernel's own settings for internet access.

    Reading the kernel back is the only way to be sure the flag was applied
    - the push response does not mention it.
    """
    out = tempfile.mkdtemp(prefix="kaggle_meta_")
    try:
        client.kernels_pull(ref, path=out, metadata=True)
        meta_path = os.path.join(out, "kernel-metadata.json")
        with open(meta_path, encoding="utf-8") as fh:
            return bool(json.load(fh).get("enable_internet"))
    except Exception as exc:
        # Not knowing is not the same as knowing it is off; do not stop a
        # working run over a failed check.
        print(f"could not confirm the kernel's internet setting: {exc}")
        return True
    finally:
        shutil.rmtree(out, ignore_errors=True)


# Slugs whose internet setting has already been checked.
_internet_checked: set[str] = set()

# Where the files a Kaggle run produces are kept.
#
# A Kaggle kernel is a fresh machine every time: whatever the script wrote is
# gone when it finishes. Without somewhere to put those files, the only way
# to use a result is to rebuild it from scratch in the next run - which is
# what the model kept trying to do, over and over, for nearly two hours.
#
# The files are kept here and pushed back into the next kernel, so a pipeline
# can be built up over several runs.
STATE_ROOT = os.getenv("KAGGLE_STATE_DIR") or os.path.join(
    os.getcwd(), "bot_files", "kaggle")

# How much is carried into the next run. Kaggle only uploads the script
# itself, so the files travel inside it.
MAX_CARRY_FILE = 2 * 1024 * 1024
MAX_CARRY_TOTAL = 8 * 1024 * 1024


def _state_dir(state_dir: str = "") -> str:
    return state_dir or STATE_ROOT


def _carry_preamble(state_dir: str) -> tuple[str, list[str]]:
    """Build the preamble that recreates the last run's files.

    Kaggle uploads only the script, so the files from the previous run are
    written back out by a short preamble at the top of it. The kernel's
    working directory is /kaggle/working and the script runs there, so a
    file called "data.csv" is simply there for the code to open.
    """
    if not state_dir or not os.path.isdir(state_dir):
        return "", []

    try:
        names = [n for n in os.listdir(state_dir)
                 if os.path.isfile(os.path.join(state_dir, n))]
    except OSError:
        return "", []

    # Newest first: the recent outputs matter more than the old ones.
    names.sort(key=lambda n: os.path.getmtime(os.path.join(state_dir, n)),
               reverse=True)

    lines = ["import base64, os, pathlib"]
    carried = []
    total = 0
    # The kernel's working directory. Kaggle runs a script from
    # /kaggle/working, but the path is spelled out rather than left to the
    # current directory, so the file is where the code expects it even if
    # the kernel starts the script from somewhere else.
    #
    # The brackets around the choice matter: without them Python reads it
    # as "path, or else write the file", so whenever the directory exists
    # - which is always - nothing is written at all.
    target = ("(pathlib.Path('/kaggle/working') "
              "if pathlib.Path('/kaggle/working').is_dir() else pathlib.Path('.'))")
    for name in names:
        path = os.path.join(state_dir, name)
        try:
            size = os.path.getsize(path)
            if size > MAX_CARRY_FILE or total + size > MAX_CARRY_TOTAL:
                continue
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue

        blob = base64.b64encode(data).decode("ascii")
        lines.append(
            f"{target}.joinpath({name!r}).write_bytes(base64.b64decode({blob!r}))"
        )
        carried.append(name)
        total += size

    if not carried:
        return "", []
    return "\n".join(lines) + "\n", carried


def _save_outputs(files: list, state_dir: str) -> list[str]:
    """Keep a run's output files for the next run, and for reading."""
    if not state_dir:
        return []
    try:
        os.makedirs(state_dir, exist_ok=True)
    except OSError:
        return []

    saved = []
    for f in files or []:
        path = f.get("path")
        name = f.get("name") or (os.path.basename(path) if path else "")
        if not path or not name or name.endswith(".log"):
            continue
        if not os.path.isfile(path):
            continue
        try:
            kept = os.path.join(state_dir, name)
            shutil.copy2(path, kept)
            # The copy is the path worth holding on to: the temporary
            # directory the download landed in is not somewhere the bot
            # should be sending files from.
            f["saved_path"] = kept
            saved.append(name)
        except Exception:
            pass
    return saved


# ---------------------------------------------------------------------------
# pip install
# ---------------------------------------------------------------------------
#
# A script that installs a package and then imports it works. A script
# that imports it at the top and installs it ten lines down does not -
# the import runs first, and the script is over before the install
# finishes. Models write it that way all the time, because it reads
# naturally:
#
#     import cowsay
#     ...
#     subprocess.run([sys.executable, "-m", "pip", "install", "cowsay"])
#
# So the packages the script asks for are installed before the script
# runs, and the script's own install line is left alone - running it
# again is a few seconds, and it is the script's business.
#
# The install is done with the same interpreter the script runs under,
# because a kernel can have more than one Python on it and installing
# into the wrong one is the kind of thing that works on your machine
# and nowhere else.

_PIP_LINE = re.compile(
    r"(?:^|\n)[ \t]*(?:![ \t]*)?"
    r"(?:(?:python3?|sys\.executable)[ \t]+-m[ \t]+)?"
    r"pip3?[ \t]+install[ \t]+(?P<args>[^\n]*)",
    re.IGNORECASE)

# The same thing written as a subprocess call:
#     subprocess.run([sys.executable, "-m", "pip", "install", "cowsay"])
_PIP_LIST = re.compile(
    r"[\"']pip3?[\"'][ \t]*,[ \t]*[\"']install[\"'][ \t]*(?P<args>[^\]]*)",
    re.IGNORECASE)

# The flags that are not package names.
_PIP_FLAGS = {"-q", "--quiet", "--no-cache-dir", "--upgrade", "-U",
              "--user", "--disable-pip-version-check", "--no-input",
              "--force-reinstall", "--pre"}


def _pip_args(text: str) -> list[str]:
    """The arguments of an install command, as a list."""
    parts = [p.strip().strip("\"'") for p in (text or "").split()]
    return [p for p in parts if p]


def _pip_packages(code: str) -> list[list[str]]:
    """Every package the script asks to install, as argument lists."""
    found = []
    for match in _PIP_LINE.finditer(code or ""):
        args = _pip_args(match.group("args"))
        if args:
            found.append(args)
    for match in _PIP_LIST.finditer(code or ""):
        args = [a.strip().strip("\"'")
                for a in re.findall(r"[\"']([^\"']*)[\"']", match.group("args"))]
        if args:
            found.append(args)
    # The same package twice is one install.
    unique = []
    for args in found:
        names = [a for a in args if a not in _PIP_FLAGS]
        if not names:
            continue
        key = " ".join(sorted(names)).lower()
        if any(" ".join(sorted(n for n in a if n not in _PIP_FLAGS)).lower() == key
               for a in unique):
            continue
        unique.append(args)
    return unique


# ---------------------------------------------------------------------------
# Missing packages
# ---------------------------------------------------------------------------
#
# A script that imports a package and never installs it dies with:
#
#     ModuleNotFoundError: No module named 'cowsay'
#
# which is the other half of the pip problem: the preamble handles the
# packages the script installs, and this handles the ones it forgets to.
# The name in the error is the import name, which is not always the
# name on PyPI - "sklearn" is "scikit-learn", "cv2" is "opencv-python",
# "PIL" is "pillow". The ones worth knowing are listed; the rest are
# tried under their own name, which is right most of the time.

_MODULE_TO_PACKAGE = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "serial": "pyserial",
    "jwt": "pyjwt",
    "Crypto": "pycryptodome",
    "fitz": "pymupdf",
    "psycopg2": "psycopg2-binary",
    "pkg_resources": "setuptools",
    "attr": "attrs",
    "OpenSSL": "pyopenssl",
    "usb": "pyusb",
    "zmq": "pyzmq",
    "discord": "discord.py",
    "google": "protobuf",
    "mysql": "mysql-connector-python",
    "pymysql": "pymysql",
    "magic": "python-magic",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "Xlib": "python-xlib",
    "win32api": "pywin32",
    "vlc": "python-vlc",
    "git": "GitPython",
    "toml": "toml",
    "yaml": "pyyaml",
}

# The modules that come with Python itself. Installing them is a waste
# of twenty seconds and always fails.
_STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {
    "os", "sys", "re", "json", "math", "time", "datetime", "random",
    "collections", "itertools", "functools", "typing", "pathlib", "csv",
    "subprocess", "asyncio", "base64", "hashlib", "urllib", "sqlite3",
    "statistics", "string", "textwrap", "shutil", "tempfile", "logging",
    "unittest", "argparse", "inspect", "copy", "enum", "dataclasses",
    "concurrent", "threading", "multiprocessing", "pickle", "struct",
    "decimal", "fractions", "uuid", "platform", "socket", "ssl", "email",
    "http", "urllib3", "zlib", "gzip", "tarfile", "zipfile", "io", "abc",
    "contextlib", "warnings", "traceback", "builtins", "__future__",
}


def _missing_modules(stderr: str) -> list[str]:
    """The modules the script could not find."""
    found = []
    for match in re.finditer(
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            stderr or ""):
        name = match.group(1).strip()
        if not name or name in found:
            continue
        found.append(name)
    return found


def _packages_for(modules: list) -> list[str]:
    """The pip names for the modules the script could not find."""
    packages = []
    for name in modules or []:
        # "google.protobuf" is the package "protobuf", and "a.b.c" is
        # the package "a" - the import name is the first part.
        top = name.split(".")[0]
        if not top or top.startswith("_") or top in _STDLIB:
            continue
        package = _MODULE_TO_PACKAGE.get(top, top)
        if package not in packages:
            packages.append(package)
    return packages


def _pip_preamble(code: str, extra: list = None) -> tuple[str, list[list[str]]]:
    """The preamble that installs what the script is going to import."""
    specs = _pip_packages(code)
    for args in extra or []:
        # The extra packages can come as a list of names or as a list of
        # argument lists - one "cowsay" and not five letters.
        if isinstance(args, str):
            args = [args]
        args = [str(a).strip() for a in args if str(a).strip()]
        if not args:
            continue
        if not any(
                " ".join(sorted(a for a in args if a not in _PIP_FLAGS)).lower()
                == " ".join(sorted(a for a in s if a not in _PIP_FLAGS)).lower()
                for s in specs):
            specs.append(list(args))
    if not specs:
        return "", []

    payload = json.dumps(specs)
    preamble = (
        "# Installed before the script runs, because the script imports\n"
        "# these at the top and installs them further down.\n"
        "import subprocess, sys\n"
        f"_bot_pip_specs = {payload}\n"
        "for _bot_pip_args in _bot_pip_specs:\n"
        "    print('[bot] pip install ' + ' '.join(_bot_pip_args), flush=True)\n"
        "    _bot_pip = subprocess.run(\n"
        "        [sys.executable, '-m', 'pip', 'install',\n"
        "         '--disable-pip-version-check'] + _bot_pip_args,\n"
        "        capture_output=True, text=True)\n"
        "    print('[bot] pip exit ' + str(_bot_pip.returncode), flush=True)\n"
        "    if _bot_pip.returncode:\n"
        "        print('[bot] pip said: ' + ((_bot_pip.stderr or\n"
        "              _bot_pip.stdout or '')[-800:]), flush=True)\n"
    )
    return preamble, specs


def _run_sync(code: str, timeout: int, accelerator: str, datasets: list,
              label: str, state_dir: str = "", extra_installs: list = None) -> dict:
    """Push a script to Kaggle, wait for it, and collect the output."""
    client = _kaggle_api()
    if client is None:
        return {"error": _api_error or NOT_CONFIGURED}

    # A GPU or TPU run that has no quota left fails inside the kernel.
    # Better to say so here, before the twenty seconds are spent.
    problem = _quota_problem(accelerator)
    if problem:
        return {"error": problem}

    # A dataset is mounted by its full ref. A bare name is looked up, so
    # "titanic" becomes the ref it stands for.
    datasets, dataset_notes = _resolve_datasets(datasets)

    owner = _owner()
    slug = None
    work = tempfile.mkdtemp(prefix="kaggle_")
    out_dir = tempfile.mkdtemp(prefix="kaggle_out_")
    try:
        # Claim a kernel from the pool. This can fail when every kernel is
        # busy, so it happens inside the try - the finally block below has to
        # cope with there being no slug.
        slug = _pool_slug()
        meta = _metadata(slug, owner, accelerator, datasets, label or slug)
        with open(os.path.join(work, "kernel-metadata.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        # The last run's files go along, so the script can build on them
        # instead of starting from nothing.
        preamble, carried = _carry_preamble(state_dir)

        # What the script wants to install goes in before the script, for
        # the reason in _pip_preamble: the script imports it at the top.
        install, installs = _pip_preamble(code, extra_installs)

        with open(os.path.join(work, "main.py"), "w") as fh:
            fh.write(preamble + install + code)

        ref = f"{owner}/{slug}" if owner else slug

        t0 = time.time()
        try:
            # Kaggle's own session limit, so a long script is not cut off
            # before the timeout we were asked to wait for. `acc` is how the
            # SDK wants the accelerator passed; the metadata is the fallback.
            response = client.kernels_push(
                work,
                timeout=str(min(int(timeout), 43200)),
                acc=accelerator or None,
            )
        except Exception as exc:
            # A refused credential is not worth retrying: the token has
            # to be re-issued, and saying so is more use than the URL
            # that came back.
            if _is_auth_refusal(exc):
                return {"error": AUTH_REFUSED}
            # A 409 Conflict means the kernel slug already exists (stale kernel
            # from a previous run that wasn't cleaned up properly).
            # Delete the old one and retry once.
            if "409" in str(exc):
                try:
                    client.kernels_delete(ref)
                except Exception:
                    pass  # If we can't delete, just fail
                try:
                    response = client.kernels_push(
                        work,
                        timeout=str(min(int(timeout), 43200)),
                        acc=accelerator or None,
                    )
                except Exception as exc2:
                    if _is_auth_refusal(exc2):
                        return {"error": AUTH_REFUSED}
                    return {"error": f"could not push to Kaggle even after cleanup: {exc2}"}
            else:
                return {"error": f"could not push to Kaggle: {exc}"}
        pushed = time.time() - t0

        detail = response.to_dict() if hasattr(response, "to_dict") else {}
        if detail.get("error"):
            return {"error": f"Kaggle rejected the kernel: {detail['error']}"}

        # Make sure the kernel really came up with internet access. If it did
        # not, push it again with the flag forced on - a kernel that cannot
        # reach the network is useless for anything live.
        if slug not in _internet_checked:
            if not _internet_is_on(client, ref):
                print(f"the kernel {ref} came up without internet; pushing it again")
                meta["enable_internet"] = True
                with open(os.path.join(work, "kernel-metadata.json"), "w") as fh:
                    json.dump(meta, fh, indent=2)
                try:
                    client.kernels_push(
                        work,
                        timeout=str(min(int(timeout), 43200)),
                        acc=accelerator or None,
                    )
                except Exception as exc:
                    print(f"could not re-push {ref} with internet enabled: {exc}")
            _internet_checked.add(slug)

        # Wait for it to finish.
        deadline = time.time() + timeout
        state = "UNKNOWN"
        failure = None
        while time.time() < deadline:
            try:
                status = client.kernels_status(ref)
                state = getattr(getattr(status, "status", None), "name", "UNKNOWN")
                failure = getattr(status, "failureMessage", None)
            except Exception as exc:
                state = "UNKNOWN"
                failure = str(exc)
            if state in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED"):
                break
            time.sleep(3)

        # Still running when the wait ran out. A kernel left running
        # holds its slot in the pool - and the next run pushed to the same
        # slug cancels it - so it is deleted rather than left behind.
        timed_out = state not in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED")
        if timed_out:
            try:
                client.kernels_delete(ref)
            except Exception as exc:
                print(f"could not stop the kernel {ref}: {exc}")
            state = "TIMEOUT"

        elapsed = time.time() - t0

        # The logs come back as a list of {stream_name, time, data}.
        stdout, stderr = [], []
        try:
            raw = client.kernels_logs(ref)
            for entry in json.loads(raw or "[]"):
                data = entry.get("data", "")
                if entry.get("stream_name") == "stdout":
                    stdout.append(data)
                else:
                    stderr.append(data)
        except Exception as exc:
            stderr.append(f"(could not fetch the logs: {exc})")

        files = []
        try:
            names, _ = client.kernels_output(ref, path=out_dir)
            for name in names or []:
                path = os.path.join(out_dir, os.path.basename(name))
                if os.path.isfile(path):
                    files.append({
                        "name": os.path.basename(name),
                        "path": path,
                        "size": os.path.getsize(path),
                    })
        except Exception:
            # No output files is normal for a script that only prints.
            pass

        # Keep the files so the next run has them and the user can read them.
        saved = _save_outputs(files, state_dir)

        # What went wrong, when the kernel could not reach the network.
        # A script that downloads something and cannot is a different
        # problem from a script with a bug in it, and the difference is
        # worth stating.
        network_note = ""
        stderr_text = "".join(stderr)
        if state != "COMPLETE" and _looks_like_no_network(stderr_text):
            network_note = (
                "the kernel could not reach the network, so anything that "
                "downloads or installs failed. Kaggle kernels need internet "
                "access turned on; this one did not have it."
            )

        return {
            "stdout": "".join(stdout),
            "stderr": stderr_text,
            "files": files,
            "saved": saved,
            "carried": carried,
            "installed": [" ".join(spec) for spec in installs],
            "dataset_notes": dataset_notes,
            "network_note": network_note,
            "state": state,
            "status": state,
            "success": state == "COMPLETE" and not failure,
            "failure": failure,
            "timed_out": timed_out,
            "elapsed": round(elapsed, 1),
            "push_seconds": round(pushed, 2),
            "ref": ref,
            "state_dir": state_dir,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if slug:
            _release_slug(slug)


async def run_code(code: str, timeout: int = 600, accelerator: str = "",
                   datasets: list = None, label: str = "",
                   state_dir: str = "") -> dict:
    """Run a Python script on Kaggle and return its output.

    `accelerator` is empty for CPU, or a machine shape such as
    "NvidiaTeslaT4" / "NvidiaH100" / "TpuV5E8". `datasets` is a list of
    Kaggle dataset refs to mount at /kaggle/input/<ref>.

    `state_dir` is where the run's files are kept. They are pushed into the
    next run, so consecutive runs can build on each other. When it is not
    given, the shared state directory under the workspace is used - without
    this the files of a run are thrown away and the next run starts from
    nothing, which is what made the bot rebuild the same pipeline over
    and over.
    """
    if not (code or "").strip():
        return {"error": "no code given"}

    # The shared state directory, unless the caller wants its own.
    state_dir = _state_dir(state_dir)

    # Check circuit breaker for known failures
    if _is_circuit_open("kaggle"):
        return {"error": "Kaggle service is temporarily unavailable (circuit breaker open)"}

    # If every kernel in the pool is busy, wait for one rather than failing.
    for attempt in range(12):
        try:
            result = await asyncio.to_thread(
                _run_sync, code, timeout, accelerator or "", datasets or [],
                label, state_dir
            )
            # Success: reset circuit breaker
            _reset_circuit("kaggle")

            # The script died because a package was missing and it never
            # asked for it. Install it and run the script again - once,
            # so a script that is simply broken does not loop.
            if (result.get("state") == "ERROR"
                    and not result.get("timed_out")
                    and not result.get("extra_install_note")):
                missing = _missing_modules(result.get("stderr") or "")
                packages = _packages_for(missing)
                if packages:
                    note = (f"the script was missing {', '.join(packages)}, so "
                            "it was installed and the script was run again")
                    print(f"[kaggle] {note}")
                    try:
                        again = await asyncio.to_thread(
                            _run_sync, code, timeout, accelerator or "",
                            datasets or [], label, state_dir, packages,
                        )
                    except Exception as exc:
                        print(f"[kaggle] the retry failed: {exc}")
                        again = {}
                    if again:
                        again["extra_install_note"] = note
                        again["installed"] = list(again.get("installed") or []) + packages
                        again["extra_installed"] = packages
                        # The first failure is worth keeping: if the retry
                        # fails for a different reason, the two together
                        # say more than either alone.
                        again["first_error"] = {
                            "stderr": (result.get("stderr") or "")[-2000:],
                            "state": result.get("state"),
                        }
                        result = again
            return result
        except RuntimeError as exc:
            if "busy" not in str(exc):
                return {"error": str(exc)}
            if attempt == 11:
                return {"error": str(exc)}
            await asyncio.sleep(5)
        except Exception as exc:
            # Record failure for circuit breaker
            _record_failure("kaggle", str(exc))
            if attempt == 11:
                return {"error": str(exc)}
            await asyncio.sleep(2 ** min(attempt, 4))  # Exponential backoff
    
    return {"error": "could not get a Kaggle kernel"}


# Kaggle's own machinery (nbconvert, mistune) writes warnings to stderr on
# every run. They are noise, not the user's program complaining.
_NOISE = (
    "NbConvertApp",
    "SyntaxWarning: invalid escape sequence",
    "re.sub(",
    "cells[i][c]",
    "text = re.sub",
)

# What a kernel that cannot reach the network says.
_NO_NETWORK = (
    "temporary failure in name resolution",
    "name or service not known",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "could not resolve host",
    "max retries exceeded",
    "ssl: wrong_version_number",
    "proxyerror",
    "network is unreachable",
)


def _looks_like_no_network(text: str) -> bool:
    """Did this fail because the kernel had no internet?"""
    low = (text or "").lower()
    return any(marker in low for marker in _NO_NETWORK)


"""Circuit breaker pattern for external service resilience."""
_CIRCUIT_BREAKERS = {}
_CIRCUIT_FAILURES = {}
_CIRCUIT_FAILURE_COUNTS = {}


def _is_circuit_open(service: str) -> bool:
    """Check if circuit breaker is open for a service."""
    import time
    breaker = _CIRCUIT_BREAKERS.get(service, {})
    if not breaker:
        return False
    if breaker.get("failures", 0) >= breaker.get("threshold", 5):
        if time.time() - breaker.get("last_failure", 0) < breaker.get("cooldown", 60):
            return True
        # Reset after cooldown
        _reset_circuit(service)
    return False


def _record_failure(service: str, error: str):
    """Record a failure for circuit breaker."""
    import time
    failures = _CIRCUIT_FAILURE_COUNTS.get(service, 0) + 1
    _CIRCUIT_FAILURE_COUNTS[service] = failures
    _CIRCUIT_BREAKERS[service] = {
        "failures": failures,
        "last_failure": time.time(),
        "threshold": 5,
        "cooldown": 60
    }
    print(f"circuit breaker: {service} failure #{failures} - {error[:100]}")


def _reset_circuit(service: str):
    """Reset circuit breaker after success or cooldown."""
    _CIRCUIT_BREAKERS.pop(service, None)
    _CIRCUIT_FAILURE_COUNTS.pop(service, None)


def _clean_stderr(text: str) -> str:
    """Drop Kaggle's internal warnings from stderr."""
    if not text:
        return ""
    kept = []
    for line in text.splitlines():
        if any(marker in line for marker in _NOISE):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def format_output(result: dict, max_chars: int = 8000, state_rel: str = "") -> str:
    """Render a run result as text for the model.

    `state_rel` is where the run's files live, relative to the workspace, so
    the model can read them with read_file.
    """
    if result.get("error"):
        return f"Error: {result['error']}"

    parts = []
    if result.get("installed"):
        parts.append("[installed before the run] " + ", ".join(result["installed"]))
    if result.get("extra_install_note"):
        parts.append(f"[what happened] {result['extra_install_note']}")
    if result.get("dataset_notes"):
        parts.append("[datasets] " + "; ".join(result["dataset_notes"]))
    if result.get("stdout"):
        parts.append(result["stdout"].strip())
    stderr = _clean_stderr(result.get("stderr") or "")
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if result.get("network_note"):
        parts.append(f"[the problem] {result['network_note']}")
    saved = set(result.get("saved") or [])
    for f in result.get("files") or []:
        if f["name"].endswith(".log"):
            continue
        where = ""
        if f["name"] in saved:
            where = " (kept for the next run)"
            if state_rel:
                where += f" - read it with read_file(\"{state_rel}/{f['name']}\")"
        parts.append(f"[file] {f['name']} ({f['size']} bytes){where}")
    if result.get("carried"):
        parts.append(
            "[carried into this run, in the working directory] "
            + ", ".join(result["carried"][:40])
        )
    if result.get("failure"):
        parts.append(f"[failed] {result['failure']}")
    elif result.get("timed_out"):
        parts.append(
            f"[the script was still running after {result.get('elapsed')} "
            "seconds and was stopped]")
    elif result.get("state") and result["state"] not in ("COMPLETE",):
        # The kernel stopped without saying why - say that much rather than
        # let it look like the script simply printed nothing.
        parts.append(f"[the kernel ended with status {result['state']}]")
    first = result.get("first_error")
    if first and first.get("stderr"):
        parts.append(
            "[the first attempt, before the missing package was installed]\n"
            + first["stderr"][-1200:].strip())
    if result.get("success") is False and not result.get("error"):
        parts.append("[the run did not finish successfully]")

    text = "\n".join(parts).strip() or "(no output)"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated, {len(text)} chars)"
    return text


# ---------------------------------------------------------------------------
# Datasets and models
# ---------------------------------------------------------------------------
def _quiet(func, *args, **kwargs):
    """Call a Kaggle SDK function without its console noise.

    The SDK prints things like "Next Page Token = ..." straight to stdout,
    which would end up interleaved with the bot's own logs.
    """
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return func(*args, **kwargs)


def _items(response, *names):
    """Pull the list out of a Kaggle response.

    The SDK is inconsistent: some calls return a plain list, others a
    response object with the list under a named attribute.
    """
    if response is None:
        return []
    if isinstance(response, list):
        return response
    for name in names:
        value = getattr(response, name, None)
        if value:
            return value
    return []


def _datasets_sync(search: str, limit: int) -> list[dict]:
    client = _kaggle_api()
    if client is None:
        return []
    try:
        response = _quiet(client.dataset_list, search=search or None)
    except Exception as exc:
        print(f"could not list datasets: {exc}")
        return []
    out = []
    for d in _items(response, "datasets", "items")[:limit]:
        out.append({
            "ref": getattr(d, "ref", ""),
            "title": getattr(d, "title", ""),
            "size": getattr(d, "total_bytes", 0),
            "files": len(getattr(d, "files", None) or []),
            "subtitle": (getattr(d, "subtitle", "") or "")[:120],
            "votes": getattr(d, "vote_count", None),
            "updated": str(getattr(d, "last_updated", "") or "")[:10],
        })
    return out


async def list_datasets(search: str = "", limit: int = 15) -> list[dict]:
    """Search Kaggle's public datasets."""
    return await asyncio.to_thread(_datasets_sync, search, limit)


def _download_dataset_sync(ref: str, dest: str) -> list[str]:
    client = _kaggle_api()
    if client is None:
        return []
    os.makedirs(dest, exist_ok=True)
    client.dataset_download_files(ref, path=dest, unzip=True, force=True)
    files = []
    for root, _dirs, names in os.walk(dest):
        for name in names:
            files.append(os.path.join(root, name))
    return files


async def download_dataset(ref: str, dest: str) -> list[str]:
    """Download a Kaggle dataset into a local directory."""
    return await asyncio.to_thread(_download_dataset_sync, ref, dest)


def _models_sync(search: str, limit: int) -> list[dict]:
    client = _kaggle_api()
    if client is None:
        return []
    try:
        response = _quiet(client.model_list, search=search or None)
    except Exception as exc:
        print(f"could not list models: {exc}")
        return []
    out = []
    for m in _items(response, "models", "modelInstances", "items")[:limit]:
        out.append({
            "ref": getattr(m, "ref", ""),
            "title": getattr(m, "title", ""),
            "subtitle": (getattr(m, "subtitle", "") or "")[:120],
        })
    return out


async def list_models(search: str = "", limit: int = 15) -> list[dict]:
    """Search Kaggle's trained models."""
    return await asyncio.to_thread(_models_sync, search, limit)


def _quota_sync() -> dict:
    client = _kaggle_api()
    if client is None:
        return {}
    try:
        quota = _quiet(client.quota_view)
        data = quota.to_dict() if hasattr(quota, "to_dict") else {}
    except Exception as exc:
        if _is_auth_refusal(exc):
            return {"error": AUTH_REFUSED}
        return {"error": str(exc)}
    gpu = data.get("gpuQuota") or {}
    tpu = data.get("tpuQuota") or {}
    return {
        "gpu_used": gpu.get("timeUsed"),
        "gpu_total": gpu.get("totalTimeAllowed"),
        "tpu_used": tpu.get("timeUsed"),
        "tpu_total": tpu.get("totalTimeAllowed"),
        "refreshes": data.get("quotaRefreshTime"),
    }


async def quota() -> dict:
    """How much GPU and TPU time is left this week."""
    return await asyncio.to_thread(_quota_sync)
