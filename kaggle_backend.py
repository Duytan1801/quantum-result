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
import json
import os
import shutil
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

# Fallback credentials, used when no environment variable reaches the process.
# The environment variable still wins, so this can be deleted once the token is
# reliably set in the deployment.
HARDCODED_TOKEN = "KGAT_438c1c9ee21b06baddc33a0b15336985"


def _find_token() -> tuple[str, str]:
    """Look for a Kaggle API token. Returns (token, where it came from)."""
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_TOKEN", "KAGGLE_ACCESS_TOKEN"):
        value = (os.getenv(name) or "").strip()
        if value:
            return value, name
    if HARDCODED_TOKEN:
        return HARDCODED_TOKEN, "built-in fallback"
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


def kaggle_status() -> dict:
    """A plain-English report of why Kaggle is or isn't available.

    Use this to debug: it checks the env var, the hardcoded fallback, the
    config file, and a live authentication attempt.
    """
    token, name = _find_token()
    env_set = bool(os.getenv("KAGGLE_API_TOKEN"))
    report = {
        "env_KAGGLE_API_TOKEN_set": env_set,
        "credentials_source": name or "none",
        "hardcoded_fallback_present": bool(HARDCODED_TOKEN),
        "legacy_key_present": _has_legacy_key(),
        "config_file_present": _has_config_file(),
        "authenticated": False,
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
        client.competitions_list()
        report["authenticated"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report



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
    if _api_failed:
        return None

    token, name = _find_token()
    if token:
        # The SDK only looks at KAGGLE_API_TOKEN, so an alias the user set
        # (KAGGLE_TOKEN, KAGGLE_ACCESS_TOKEN) is normalised into it here.
        os.environ["KAGGLE_API_TOKEN"] = token
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


def _run_sync(code: str, timeout: int, accelerator: str, datasets: list,
              label: str) -> dict:
    """Push a script to Kaggle, wait for it, and collect the output."""
    client = _kaggle_api()
    if client is None:
        return {"error": _api_error or NOT_CONFIGURED}

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
        with open(os.path.join(work, "main.py"), "w") as fh:
            fh.write(code)

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

        return {
            "stdout": "".join(stdout),
            "stderr": "".join(stderr),
            "files": files,
            "state": state,
            "failure": failure,
            "elapsed": round(elapsed, 1),
            "push_seconds": round(pushed, 2),
            "ref": ref,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if slug:
            _release_slug(slug)


async def run_code(code: str, timeout: int = 600, accelerator: str = "",
                   datasets: list = None, label: str = "") -> dict:
    """Run a Python script on Kaggle and return its output.

    `accelerator` is empty for CPU, or a machine shape such as
    "NvidiaTeslaT4" / "NvidiaH100" / "TpuV5E8". `datasets` is a list of
    Kaggle dataset refs to mount at /kaggle/input/<ref>.
    """
    if not (code or "").strip():
        return {"error": "no code given"}

    # If every kernel in the pool is busy, wait for one rather than failing.
    for attempt in range(12):
        try:
            return await asyncio.to_thread(
                _run_sync, code, timeout, accelerator or "", datasets or [], label
            )
        except RuntimeError as exc:
            if "busy" not in str(exc):
                return {"error": str(exc)}
            if attempt == 11:
                return {"error": str(exc)}
            await asyncio.sleep(5)
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


def format_output(result: dict, max_chars: int = 8000) -> str:
    """Render a run result as text for the model."""
    if result.get("error"):
        return f"Error: {result['error']}"

    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"].strip())
    stderr = _clean_stderr(result.get("stderr") or "")
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    for f in result.get("files") or []:
        if f["name"].endswith(".log"):
            continue
        parts.append(f"[file] {f['name']} ({f['size']} bytes)")
    if result.get("failure"):
        parts.append(f"[failed] {result['failure']}")

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
