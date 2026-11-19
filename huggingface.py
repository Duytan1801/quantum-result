"""Hugging Face Hub: finding models and datasets.

The Hub's public API is used directly with the access token, so the bot can
see the private and gated repositories the account reaches as well as the
public ones, and it gets the account's rate limit rather than the anonymous
one.

Three things are on offer: searching the Hub for models and datasets, reading
what a particular repository is (its card, its tags, its files), and reading
a file out of a repository - the model card, a config, a dataset script.
"""

import asyncio

import httpx

from config import HUGGINGFACE_TOKEN

# Used when no environment variable reaches the process. The environment
# variable still wins, so this can be deleted once the token is reliably set
# in the deployment.
HARDCODED_TOKEN = "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"

HUB_URL = "https://huggingface.co"
TIMEOUT = 30.0
MAX_CARD_CHARS = 4000

# The largest download the tool will make without being told a limit.
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


def token_prompt() -> str:
    """The prompt lines that give the model the token and how to use it.

    The model needs the token to download a gated repository, and to pull a
    model or dataset down inside the sandbox where the code runs.
    """
    key = token()
    if not key:
        return ""
    return (
        "- Hub token: " + key + "\n"
        "- To pull a model or dataset down in code, install the client first "
        "(pip install huggingface_hub), then:\n"
        "    from huggingface_hub import snapshot_download, hf_hub_download\n"
        "    snapshot_download('org/repo', token='" + key + "')\n"
        "    hf_hub_download('org/repo', 'data.csv', repo_type='dataset', "
        "token='" + key + "')\n"
        "- The hf_download tool does the same thing and leaves the files in "
        "the workspace, where you can read them.\n"
    )


def token() -> str:
    return HUGGINGFACE_TOKEN or HARDCODED_TOKEN


def _headers(accept: str = "application/json") -> dict:
    headers = {"Accept": accept}
    key = token()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


async def _get_json(url: str, params: dict = None) -> list | dict:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def _get_text(url: str, accept: str = "text/plain") -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=_headers(accept))
        resp.raise_for_status()
        return resp.text


def _repo_line(item: dict, index: int) -> str:
    repo_id = item.get("id") or item.get("modelId") or "?"
    pipeline = item.get("pipeline_tag") or ""
    downloads = item.get("downloads")
    likes = item.get("likes")
    lines = [
        f"{index}. {repo_id}",
        f"   downloads: {downloads if downloads is not None else '?'}"
        f"   likes: {likes if likes is not None else '?'}"
        + (f"   task: {pipeline}" if pipeline else ""),
    ]
    author = item.get("author")
    if author:
        lines.append(f"   author: {author}")
    tags = item.get("tags") or []
    if tags:
        lines.append(f"   tags: {', '.join(str(t) for t in tags[:10])}")
    if item.get("lastModified"):
        lines.append(f"   updated: {str(item['lastModified'])[:10]}")
    lines.append(f"   link: {HUB_URL}/{repo_id}")
    return "\n".join(lines)


async def search_models(query: str, limit: int = 10, sort: str = "downloads") -> str:
    """Search the Hub for models."""
    query = (query or "").strip()
    if not query:
        return "Error: no search query given."
    limit = max(1, min(int(limit or 10), 50))

    try:
        items = await _get_json(f"{HUB_URL}/api/models", {
            "search": query,
            "limit": limit,
            "sort": sort or "downloads",
            "direction": -1,
            "full": "false",
        })
    except httpx.HTTPStatusError as exc:
        return f"Hugging Face API error ({exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"Hugging Face error: {exc}"

    if not items:
        return f"No models found for '{query}'."

    body = "\n\n".join(_repo_line(m, i + 1) for i, m in enumerate(items))
    return f"{len(items)} model(s) for '{query}' (source: Hugging Face Hub):\n\n{body}"


async def search_datasets(query: str, limit: int = 10, sort: str = "downloads") -> str:
    """Search the Hub for datasets."""
    query = (query or "").strip()
    if not query:
        return "Error: no search query given."
    limit = max(1, min(int(limit or 10), 50))

    try:
        items = await _get_json(f"{HUB_URL}/api/datasets", {
            "search": query,
            "limit": limit,
            "sort": sort or "downloads",
            "direction": -1,
            "full": "false",
        })
    except httpx.HTTPStatusError as exc:
        return f"Hugging Face API error ({exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"Hugging Face error: {exc}"

    if not items:
        return f"No datasets found for '{query}'."

    body = "\n\n".join(_repo_line(d, i + 1) for i, d in enumerate(items))
    return f"{len(items)} dataset(s) for '{query}' (source: Hugging Face Hub):\n\n{body}"


def _info_lines(info: dict, kind: str) -> list[str]:
    lines = [
        f"repository: {info.get('id') or info.get('modelId') or '?'} ({kind})",
        f"downloads: {info.get('downloads', '?')}   likes: {info.get('likes', '?')}",
    ]
    if info.get("pipeline_tag"):
        lines.append(f"task: {info['pipeline_tag']}")
    if info.get("library_name"):
        lines.append(f"library: {info['library_name']}")
    card = info.get("cardData") or {}
    if card.get("license"):
        lines.append(f"license: {card['license']}")
    if card.get("base_model"):
        lines.append(f"base model: {card['base_model']}")
    if card.get("datasets"):
        lines.append(f"trained on: {', '.join(card['datasets'][:8])}")
    if info.get("author"):
        lines.append(f"author: {info['author']}")
    tags = info.get("tags") or []
    if tags:
        lines.append(f"tags: {', '.join(str(t) for t in tags[:12])}")
    if info.get("lastModified"):
        lines.append(f"updated: {str(info['lastModified'])[:10]}")
    lines.append(f"link: {HUB_URL}/{info.get('id') or ''}")
    return lines


def _file_lines(files: list) -> str:
    if not files:
        return ""
    lines = ["files:"]
    for f in files[:40]:
        name = f.get("rfilename") or f.get("filename") or "?"
        size = f.get("size")
        lines.append(f"   {name}" + (f"  ({size} bytes)" if size else ""))
    if len(files) > 40:
        lines.append(f"   ... and {len(files) - 40} more")
    return "\n".join(lines)


async def _card(repo_id: str, repo_type: str = "model") -> tuple[str, str]:
    """Read a repository's card. Returns (text, error note)."""
    prefix = "datasets/" if repo_type == "dataset" else ""
    try:
        text = await _get_text(f"{HUB_URL}/{prefix}{repo_id}/raw/main/README.md")
        return (text or "").strip(), ""
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            return "", ("the card could not be read: this repository is gated "
                        "and the account is not authorized for it "
                        f"(HTTP {exc.response.status_code})")
        return "", f"the card could not be read (HTTP {exc.response.status_code})"
    except Exception as exc:
        return "", f"the card could not be read ({exc})"


async def model_info(model_id: str) -> str:
    """Everything the Hub knows about one model."""
    model_id = (model_id or "").strip()
    if not model_id:
        return "Error: no model id given."

    try:
        info = await _get_json(f"{HUB_URL}/api/models/{model_id}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"No such model: {model_id}"
        return f"Hugging Face API error ({exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"Hugging Face error: {exc}"

    out = "\n".join(_info_lines(info, "model"))
    files = info.get("siblings") or []
    if files:
        out += "\n\n" + _file_lines(files)

    card, card_error = await _card(model_id, "model")
    if card:
        out += f"\n\nmodel card (first {MAX_CARD_CHARS} characters):\n{card[:MAX_CARD_CHARS]}"
    elif card_error:
        out += f"\n\nmodel card: {card_error}"

    return out


async def dataset_info(dataset_id: str) -> str:
    """Everything the Hub knows about one dataset."""
    dataset_id = (dataset_id or "").strip()
    if not dataset_id:
        return "Error: no dataset id given."

    try:
        info = await _get_json(f"{HUB_URL}/api/datasets/{dataset_id}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"No such dataset: {dataset_id}"
        return f"Hugging Face API error ({exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"Hugging Face error: {exc}"

    out = "\n".join(_info_lines(info, "dataset"))
    files = info.get("siblings") or []
    if files:
        out += "\n\n" + _file_lines(files)

    card, card_error = await _card(dataset_id, "dataset")
    if card:
        out += f"\n\ndataset card (first {MAX_CARD_CHARS} characters):\n{card[:MAX_CARD_CHARS]}"
    elif card_error:
        out += f"\n\ndataset card: {card_error}"

    return out


async def read_repo_file(repo_id: str, path: str = "README.md",
                          repo_type: str = "model") -> str:
    """Read one file out of a Hub repository."""
    repo_id = (repo_id or "").strip()
    path = (path or "README.md").strip().lstrip("/")
    if not repo_id:
        return "Error: no repository id given."
    if not path:
        return "Error: no file path given."

    if repo_type not in ("model", "dataset"):
        repo_type = "model"
    prefix = "datasets/" if repo_type == "dataset" else ""

    try:
        text = await _get_text(f"{HUB_URL}/{prefix}{repo_id}/raw/main/{path}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"No such file: {path} in {repo_id}"
        return f"Hugging Face API error ({exc.response.status_code}): {exc.response.text[:300]}"
    except Exception as exc:
        return f"Hugging Face error: {exc}"

    if not text:
        return f"{path} in {repo_id} is empty."

    return (f"{path} from {repo_id} ({len(text)} characters):\n\n"
            + text[:MAX_CARD_CHARS * 4])


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def _repo_size(repo_id: str, repo_type: str) -> tuple[list, int]:
    """The files in a repository and their total size."""
    from huggingface_hub import HfApi

    api = HfApi(token=token() or None)
    info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    files = []
    total = 0
    for sibling in getattr(info, "siblings", None) or []:
        size = getattr(sibling, "size", None) or 0
        files.append((sibling.rfilename, size))
        total += size
    return files, total


def _download_sync(repo_id: str, repo_type: str, filename: str,
                   dest: str, max_bytes: int) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    key = token() or None
    files, total = _repo_size(repo_id, repo_type)

    if filename:
        wanted = [f for f, _ in files if f == filename]
        if not wanted:
            sample = ", ".join(f for f, _ in files[:15])
            return {"error": f"no file called '{filename}' in {repo_id}. "
                             f"Files include: {sample}"}
        size = next((s for f, s in files if f == filename), 0)
        if size > max_bytes:
            return {"error": f"'{filename}' is {size // 1024 // 1024} MB, over "
                             f"the limit of {max_bytes // 1024 // 1024} MB"}
        path = hf_hub_download(repo_id=repo_id, filename=filename,
                               repo_type=repo_type, token=key,
                               local_dir=dest)
        return {"path": path, "files": [(filename, size)], "bytes": size}

    if total > max_bytes:
        biggest = sorted(files, key=lambda f: f[1], reverse=True)[:5]
        detail = ", ".join(f"{n} ({s // 1024 // 1024} MB)" for n, s in biggest if s)
        return {"error": f"{repo_id} is {total // 1024 // 1024} MB, over the "
                         f"limit of {max_bytes // 1024 // 1024} MB. Download a "
                         f"single file instead. Biggest: {detail}"}

    path = snapshot_download(repo_id=repo_id, repo_type=repo_type, token=key,
                             local_dir=dest)
    return {"path": path, "files": files, "bytes": total}


async def download_repo(repo_id: str, repo_type: str = "model", filename: str = "",
                        dest: str = "", max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES) -> str:
    """Download a model or dataset from the Hub into the workspace."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return "Error: no repository id given."
    if repo_type not in ("model", "dataset"):
        repo_type = "model"
    if not dest:
        return "Error: no destination directory given."

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_download_sync, repo_id, repo_type,
                              (filename or "").strip(), dest, int(max_bytes)),
            timeout=1800,
        )
    except asyncio.TimeoutError:
        return "Error: the download timed out after 30 minutes."
    except Exception as exc:
        return f"Error downloading {repo_id}: {type(exc).__name__}: {exc}"

    if result.get("error"):
        return "Error: " + result["error"]

    files = result.get("files") or []
    listed = "\n".join(f"   {n}" + (f"  ({s // 1024 // 1024} MB)" if s else "")
                       for n, s in files[:60])
    more = f"\n   ... and {len(files) - 60} more" if len(files) > 60 else ""
    return (f"Downloaded {repo_id} ({repo_type}) to {result['path']} "
            f"({result.get('bytes', 0) // 1024 // 1024} MB, {len(files)} files).\n"
            f"files:\n{listed}{more}")
