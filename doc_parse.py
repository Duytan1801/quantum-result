"""Text extraction from pictures.

NVIDIA's Nemotron-3.5-Parse endpoint reads a rendered page (PNG or JPEG) and
returns the text it contains, annotated with the position and kind of every
block it found.

It exists here because a vision-language model is good at describing what is
in a picture and bad at transcribing it: ask one to read a table off a
screenshot and it will produce something close, sometimes plausible, and
wrong. The parser reads the words instead, so for a picture that has text in
it the parser is the better reader and the bot uses it.

Only images are accepted: this endpoint rejects PDFs outright (it wants a
rendered page), so PDFs are left to the other tooling.
"""

import json
import re

import httpx

from config import NVIDIA_API_KEY

PARSE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse"
PARSE_TIMEOUT = 90.0
MAX_OUTPUT_TOKENS = 4096

# The parser returns its layout annotations inline: a coordinate pair for
# every block it found, and a class tag naming the kind of block it is.
# They come in opening and closing forms, and the closing form of a
# coordinate pair can be split across the text, so both are removed.
_COORD_RE = re.compile(r"</?x_[^>]*>|</?y_[^>]*>")
_CLASS_RE = re.compile(r"</?class_[^>]*>")
# The row and cell tags inside a table.
_ROW_RE = re.compile(r"</?row[^>]*>")
_CELL_RE = re.compile(r"</?cell[^>]*>")
_BLANKS_RE = re.compile(r"\n{3,}")


def clean_markup(text: str) -> str:
    """Strip the layout annotations, leaving the text that was on the page."""
    if not text:
        return ""
    text = _COORD_RE.sub("", text)
    text = _CLASS_RE.sub("", text)
    text = _ROW_RE.sub(" ", text)
    text = _CELL_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    # A table row is now just its cells with spaces between them; tidy the
    # runs of spaces that leaves behind.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


async def parse_document(data: bytes, content_type: str = "image/png",
                         filename: str = "image.png") -> str:
    """Read the text out of an image. Returns "" if there is no text, or if
    the call failed - the caller falls back to the vision model then."""
    if not data or not NVIDIA_API_KEY:
        return ""

    files = {
        "document": (filename, data, content_type or "image/png"),
        "options": (None, json.dumps({
            "decoding": {"max_output_tokens": MAX_OUTPUT_TOKENS},
        }), "application/json"),
    }
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=PARSE_TIMEOUT) as client:
            resp = await client.post(PARSE_URL, headers=headers, files=files)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        print(f"document parse failed: {exc}")
        return ""

    content = payload.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    return clean_markup(content)
