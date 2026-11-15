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


def _kaggle_api():
    """Get an authenticated Kaggle client, or None if it is not set up."""
    global _api, _api_failed
    if _api is not None:
        return _api
    if _api_failed:
        return None
    if not (KAGGLE_API_TOKEN or os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))):
        _api_failed = True
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        client = KaggleApi()
        client.authenticate()
        _api = client
    except Exception as exc:
        print(f"could not set up the Kaggle client: {exc}")
        _api_failed = True
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
    return {
        "id": f"{owner}/{slug}" if owner else slug,
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": str(accelerator or "").upper().startswith("NVIDIA"),
        "enable_tpu": str(accelerator or "").upper().startswith("TPU"),
        "enable_internet": True,
        "machine_shape": accelerator or "",
        "dataset_sources": datasets or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _run_sync(code: str, timeout: int, accelerator: str, datasets: list,
              label: str) -> dict:
    """Push a script to Kaggle, wait for it, and collect the output."""
    client = _kaggle_api()
    if client is None:
        return {"error": "Kaggle is not configured. Set KAGGLE_API_TOKEN."}

    owner = _owner()
    slug = None
    work = tempfile.mkdtemp(prefix="kaggle_")
    out_dir = tempfile.mkdtemp(prefix="kaggle_out_")
    try:
        # Claim a kernel from the pool. This can fail when every kernel is
        # busy, so it happens inside the try - the finally block below has to
        # cope with there being no slug.
        slug = _pool_slug()
        with open(os.path.join(work, "kernel-metadata.json"), "w") as fh:
            json.dump(_metadata(slug, owner, accelerator, datasets,
                                label or slug), fh, indent=2)
        with open(os.path.join(work, "main.py"), "w") as fh:
            fh.write(code)

        t0 = time.time()
        try:
            response = client.kernels_push(work)
        except Exception as exc:
            return {"error": f"could not push to Kaggle: {exc}"}
        pushed = time.time() - t0

        detail = response.to_dict() if hasattr(response, "to_dict") else {}
        if detail.get("error"):
            return {"error": f"Kaggle rejected the kernel: {detail['error']}"}

        ref = f"{owner}/{slug}" if owner else slug

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
