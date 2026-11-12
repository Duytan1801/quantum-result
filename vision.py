"""Image understanding.

The bot runs a vision-language model, so it can look at pictures people post.
Discord attachments are downloaded and passed to the model as base64 data
URLs, which is what the OpenAI-compatible endpoint expects.
"""

import base64

import httpx

# Image formats the endpoint accepts.
SUPPORTED_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

MAX_IMAGES = 4                 # images per message
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # per image
MAX_TOTAL_BYTES = 12 * 1024 * 1024  # across all images in one message
DOWNLOAD_TIMEOUT = 30.0


def _is_image(attachment) -> bool:
    content_type = (getattr(attachment, "content_type", "") or "").lower()
    if content_type in SUPPORTED_TYPES:
        return True
    name = (getattr(attachment, "filename", "") or "").lower()
    return any(name.endswith(f".{ext}") for ext in ("png", "jpg", "jpeg", "webp", "gif"))


async def _download(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").split(";")[0].lower()
        return resp.content, content_type


async def build_image_content(attachments, text: str):
    """Build the multimodal content for a user message.

    Returns (content, note) where content is either the plain text (no images)
    or a list of content parts for the vision endpoint, and note describes what
    was attached - used for the stored conversation row.
    """
    images = [a for a in (attachments or []) if _is_image(a)]
    if not images:
        return text, ""

    parts = []
    notes = []
    total = 0

    for attachment in images[:MAX_IMAGES]:
        if total >= MAX_TOTAL_BYTES:
            break
        size = getattr(attachment, "size", 0) or 0
        if size > MAX_IMAGE_BYTES:
            notes.append(f"{attachment.filename} (too large, skipped)")
            continue
        try:
            data, content_type = await _download(attachment.url)
        except Exception as exc:
            notes.append(f"{attachment.filename} (could not download: {exc})")
            continue

        if content_type not in SUPPORTED_TYPES:
            # Trust the bytes over the header.
            content_type = getattr(attachment, "content_type", "") or content_type
        if content_type not in SUPPORTED_TYPES:
            notes.append(f"{attachment.filename} (unsupported type, skipped)")
            continue

        total += len(data)
        encoded = base64.b64encode(data).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded}"},
        })
        notes.append(getattr(attachment, "filename", "image"))

    if not parts:
        note = (" [images could not be loaded: " + "; ".join(notes) + "]") if notes else ""
        return text + note, ""

    content = [{"type": "text", "text": text}]
    content.extend(parts)

    note = " [attached: " + ", ".join(notes) + "]"
    return content, note
