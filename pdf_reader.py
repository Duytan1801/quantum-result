"""Reading PDFs.

PyMuPDF (fitz) extracts the text out of a PDF - papers, reports, scanned
documents - so the bot can actually read a file it was given instead of
guessing from the title.

The text is returned page by page, with the page numbers kept, so the model
can point at the page it is quoting.
"""

import re

import httpx

PDF_TIMEOUT = 60.0
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300

_PAGE_RE = re.compile(r"(\d+)\s*(?:-\s*(\d+))?")


def _parse_pages(spec: str, total: int) -> list[int]:
    """Turn '1-3,7' into a list of zero-based page indexes.

    An empty spec means the whole document. Anything out of range is
    dropped rather than raising.
    """
    if not spec:
        return list(range(total))

    wanted = set()
    for match in _PAGE_RE.finditer(spec):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > end:
            start, end = end, start
        for page in range(start, end + 1):
            index = page - 1
            if 0 <= index < total:
                wanted.add(index)

    if not wanted:
        return list(range(total))
    return sorted(wanted)


def _open(data: bytes):
    """Open a PDF from bytes, whichever name PyMuPDF is installed under."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    return pymupdf.open(stream=data, filetype="pdf")


def extract_pdf_text(data: bytes, pages: str = "", max_chars: int = 60000) -> str:
    """Extract the text of a PDF. Returns a readable string."""
    if not data:
        return "Error: the PDF is empty."

    try:
        doc = _open(data)
    except Exception as exc:
        return f"Error: could not open the PDF ({exc})."

    try:
        if getattr(doc, "needs_pass", False):
            return "Error: the PDF is password protected."

        total = doc.page_count
        if total == 0:
            return "The PDF has no pages."

        indexes = _parse_pages(pages, total)
        if len(indexes) > MAX_PAGES:
            indexes = indexes[:MAX_PAGES]

        meta = doc.metadata or {}
        header = [
            f"PDF: {meta.get('title') or 'untitled'}",
            f"pages: {total}"
            + (f"   reading pages: {', '.join(str(i + 1) for i in indexes[:20])}"
               if len(indexes) != total else ""),
        ]
        if meta.get("author"):
            header.append(f"author: {meta['author']}")

        chunks = []
        used = 0
        for index in indexes:
            try:
                page = doc[index]
                text = page.get_text("text") or ""
            except Exception:
                continue
            text = text.strip()
            if not text:
                continue
            block = f"--- page {index + 1} ---\n{text}\n"
            if used + len(block) > max_chars:
                block = block[: max(0, max_chars - used)]
                if block:
                    chunks.append(block)
                chunks.append("\n[truncated: the character limit was reached]")
                break
            chunks.append(block)
            used += len(block)

        body = "\n".join(chunks).strip()
        if not body:
            return ("\n".join(header)
                    + "\n\nNo text could be extracted - the PDF is probably "
                      "a scan of images with no text layer.")

        return "\n".join(header) + "\n\n" + body
    finally:
        try:
            doc.close()
        except Exception:
            pass


async def fetch_pdf(url: str) -> tuple[bytes, str]:
    """Download a PDF. Returns (data, error)."""
    try:
        async with httpx.AsyncClient(timeout=PDF_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        return b"", str(exc)

    if len(data) > MAX_PDF_BYTES:
        return b"", f"the file is {len(data) // 1024 // 1024} MB, over the limit"
    if not data:
        return b"", "the download was empty"
    return data, ""
