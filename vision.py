"""Image understanding.

The bot runs a vision-language model, so it can look at pictures people post.
Discord attachments are downloaded and passed to the model as base64 data
URLs, which is what the OpenAI-compatible endpoint expects.

When a picture has text in it, the text is read by NVIDIA's Nemotron-3.5-Parse
endpoint instead. A vision model describes what it sees but guesses at the
words, so a screenshot of a table comes back as a plausible-looking table
with the wrong numbers in it; the parser actually reads it. A picture whose
parsed text is substantial is therefore handed to the model as that text, and
the picture itself is only passed on when the picture is mostly visual.
"""

import asyncio
import base64

import httpx

from doc_parse import parse_document

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

# A picture whose parsed text is at least this long is treated as a text
# picture: the parser's reading is used in place of the vision model.
PARSE_TEXT_MIN = 40


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


async def _prepare(attachment) -> dict:
    """Download one attachment and read any text in it.

    Failures are reported in the result rather than raised, so one bad
    attachment does not lose the rest.
    """
    name = getattr(attachment, "filename", "image")
    out = {"data": None, "content_type": "", "parsed": "", "note": ""}
    try:
        data, content_type = await _download(attachment.url)
    except Exception as exc:
        out["note"] = f"{name} (could not download: {exc})"
        return out

    if content_type not in SUPPORTED_TYPES:
        # Trust the bytes over the header.
        content_type = getattr(attachment, "content_type", "") or content_type
    if content_type not in SUPPORTED_TYPES:
        out["note"] = f"{name} (unsupported type, skipped)"
        return out

    out["data"] = data
    out["content_type"] = content_type
    try:
        out["parsed"] = await parse_document(data, content_type, name)
    except Exception as exc:
        print(f"could not read text from image: {exc}")
    return out


async def build_image_content(attachments, text: str):
    """Build the multimodal content for a user message.

    Returns (content, note) where content is either the plain text (no images)
    or a list of content parts for the vision endpoint, and note describes what
    was attached - used for the stored conversation row.

    Text found in a picture is added to the message as its own block, marked
    as extracted text, so the model reads the words rather than interpreting a
    picture of them.
    """
    images = [a for a in (attachments or []) if _is_image(a)]
    if not images:
        return text, ""

    usable = []
    skipped = []
    total = 0
    for attachment in images[:MAX_IMAGES]:
        size = getattr(attachment, "size", 0) or 0
        if size > MAX_IMAGE_BYTES:
            skipped.append(f"{getattr(attachment, 'filename', 'image')} (too large, skipped)")
            continue
        if total + size > MAX_TOTAL_BYTES:
            skipped.append(f"{getattr(attachment, 'filename', 'image')} (over the size limit, skipped)")
            continue
        total += size
        usable.append(attachment)

    if not usable:
        note = " [images could not be loaded: " + "; ".join(skipped) + "]"
        return text + note, ""

    results = await asyncio.gather(*(_prepare(a) for a in usable))

    parts = []
    blocks = []
    notes = list(skipped)

    for attachment, res in zip(usable, results):
        name = getattr(attachment, "filename", "image")
        if res["note"]:
            notes.append(res["note"])
            continue

        parsed = res["parsed"]
        if parsed:
            blocks.append(
                f"\n\n[Text read from the image {name} with NVIDIA's "
                f"Nemotron-3.5-Parse model - this is the literal text in the "
                f"picture, so use it as written]\n{parsed}"
            )

        if len(parsed) >= PARSE_TEXT_MIN:
            # A text picture: the parsed text is the content, so there is no
            # reason to have the vision model look at it as well.
            notes.append(f"{name} (text parsed)")
            continue

        encoded = base64.b64encode(res["data"]).decode("ascii")
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{res['content_type']};base64,{encoded}"},
        })
        notes.append(name)

    full_text = text + "".join(blocks)

    if not parts:
        if blocks:
            # The text was read out of the pictures, so there is no image to
            # pass on - but the conversation did have attachments.
            return full_text, " [attached: " + ", ".join(notes) + "]"
        note = (" [images could not be loaded: " + "; ".join(notes) + "]") if notes else ""
        return text + note, ""

    content = [{"type": "text", "text": full_text}]
    content.extend(parts)

    note = " [attached: " + ", ".join(notes) + "]"
    return content, note
