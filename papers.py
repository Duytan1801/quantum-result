"""Academic paper search.

Uses scholarly, which queries Google Scholar, so it reaches the literature a
general web search mostly misses: papers by title, papers behind a paywall,
and the citation counts that say which of them the field actually uses.

Google Scholar blocks most cloud hosts - Render among them - and answers a
blocked request with an empty page rather than an error, so a search that
finds nothing falls through to OpenAlex, then arXiv, then Crossref. The
source that answered is named in the output, so the model knows what it has.
"""

import asyncio
import re
import xml.etree.ElementTree as ET

import httpx

SEARCH_TIMEOUT = 30
DETAILS_TIMEOUT = 30
HTTP_TIMEOUT = 25.0
USER_AGENT = "lain2-discord-bot/1.0 (academic paper search)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")


def _paper(**kwargs) -> dict:
    """The shape every source is normalised into."""
    return {
        "title": kwargs.get("title") or "",
        "authors": kwargs.get("authors") or [],
        "year": kwargs.get("year") or "",
        "venue": kwargs.get("venue") or "",
        "abstract": kwargs.get("abstract") or "",
        "citation": kwargs.get("citation"),
        "url": kwargs.get("url") or "",
        "doi": kwargs.get("doi") or "",
        "arxiv": kwargs.get("arxiv") or "",
        "type": kwargs.get("type") or "",
    }


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def _scholar_search(query: str, limit: int, year: int) -> list[dict]:
    from scholarly import scholarly

    results = []
    search = scholarly.search_pubs(query)
    scanned = 0
    # Scan a few more than asked for, so a year filter can still fill the list.
    while len(results) < limit and scanned < limit * 4:
        scanned += 1
        try:
            pub = next(search)
        except StopIteration:
            break
        try:
            raw = pub.to_dict()
        except Exception:
            raw = {k: getattr(pub, k, None) for k in
                   ("title", "authors", "year", "venue", "abstract",
                    "citation", "url", "externalIds", "publication_types")}
        authors = raw.get("authors") or []
        if isinstance(authors, dict):
            authors = authors.get("author") or []
        names = [a.get("name") if isinstance(a, dict) else str(a)
                 for a in authors]
        ext = raw.get("externalIds") or {}
        if not isinstance(ext, dict):
            ext = {}
        paper = _paper(
            title=raw.get("title"),
            authors=names,
            year=raw.get("year"),
            venue=raw.get("venue"),
            abstract=raw.get("abstract"),
            citation=raw.get("citation"),
            url=raw.get("url"),
            doi=ext.get("DOI") or "",
            arxiv=ext.get("ArXiv") or "",
            type=", ".join(raw.get("publication_types") or []),
        )
        if year and str(paper["year"] or "") != str(year):
            continue
        results.append(paper)
    return results


def _openalex_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}. Put them back."""
    if not inverted:
        return ""
    words = []
    for word, positions in inverted.items():
        for position in positions or []:
            words.append((position, word))
    words.sort()
    return " ".join(w for _, w in words)


def _openalex_search(query: str, limit: int, year: int) -> list[dict]:
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        resp = c.get("https://api.openalex.org/works", params={
            "search": query,
            "per-page": min(max(limit * 2, 10), 50),
        })
        resp.raise_for_status()
        works = resp.json().get("results") or []

    results = []
    for w in works:
        source = ((w.get("primary_location") or {}).get("source") or {})
        authors = [
            (a.get("author") or {}).get("display_name")
            for a in (w.get("authorships") or [])
        ]
        ids = w.get("ids") or {}
        doi = (ids.get("doi") or w.get("doi") or "")
        doi = doi.replace("https://doi.org/", "") if doi else ""
        paper = _paper(
            title=w.get("title") or w.get("display_name"),
            authors=[a for a in authors if a],
            year=w.get("publication_year"),
            venue=source.get("display_name") or "",
            abstract=_openalex_abstract(w.get("abstract_inverted_index")),
            citation=w.get("cited_by_count"),
            url=ids.get("openalex") or w.get("id") or "",
            doi=doi,
            type=w.get("type") or "",
        )
        if year and str(paper["year"] or "") != str(year):
            continue
        results.append(paper)
        if len(results) >= limit:
            break
    return results


def _arxiv_search(query: str, limit: int, year: int, title_only: bool = False) -> list[dict]:
    field = "ti" if title_only else "all"
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        resp = c.get("http://export.arxiv.org/api/query", params={
            "search_query": f'{field}:"{query}"' if title_only else f"{field}:{query}",
            "max_results": min(max(limit * 2, 10), 50),
        })
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

    results = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        published = (entry.findtext(f"{_ATOM}published") or "")[:4]
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href") or ""
        results.append(_paper(
            title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()),
            authors=[a.findtext(f"{_ATOM}name") or ""
                     for a in entry.findall(f"{_ATOM}author")],
            year=published,
            venue="arXiv",
            abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
            citation=None,
            url=raw_id,
            arxiv=arxiv_id,
            pdf_url=pdf_url,
            type="preprint",
        ))
    if year:
        results = [p for p in results if str(p["year"]) == str(year)]
    return results[:limit]


def _crossref_search(query: str, limit: int, year: int) -> list[dict]:
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as c:
        resp = c.get("https://api.crossref.org/works", params={
            "query.bibliographic": query,
            "rows": min(max(limit * 2, 10), 50),
        })
        resp.raise_for_status()
        items = (resp.json().get("message") or {}).get("items") or []

    results = []
    for w in items:
        authors = []
        for a in w.get("author") or []:
            name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
            if name:
                authors.append(name)
        parts = (w.get("issued") or {}).get("date-parts") or []
        year_value = parts[0][0] if parts and parts[0] else ""
        abstract = _TAG_RE.sub(" ", w.get("abstract") or "")
        abstract = re.sub(r"\s+", " ", abstract).strip()
        results.append(_paper(
            title=(w.get("title") or [""])[0],
            authors=authors,
            year=year_value,
            venue=(w.get("container-title") or [""])[0],
            abstract=abstract,
            citation=w.get("is-referenced-by-count"),
            url=w.get("URL") or "",
            doi=w.get("DOI") or "",
            type=w.get("type") or "",
        ))
    if year:
        results = [p for p in results if str(p["year"]) == str(year)]
    return results[:limit]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_paper(d: dict, index: int) -> str:
    authors = ", ".join(d.get("authors") or []) or "unknown authors"
    if len(authors) > 220:
        authors = authors[:217] + "..."
    citations = d.get("citation")
    lines = [
        f"{index}. {d.get('title') or 'Untitled'}",
        f"   authors: {authors}",
        f"   year: {d.get('year') or '?'}   venue: {d.get('venue') or '?'}"
        f"   citations: {citations if citations is not None else '?'}",
    ]
    if d.get("doi"):
        lines.append(f"   doi: {d['doi']}")
    if d.get("arxiv"):
        lines.append(f"   arxiv: {d['arxiv']}")
    if d.get("url"):
        lines.append(f"   link: {d['url']}")
    abstract = (d.get("abstract") or "").strip()
    if abstract:
        lines.append(f"   abstract: {abstract}")
    return "\n".join(lines)


def _format_bibtex(title: str, doi: str, arxiv_id: str) -> str:
    """Ask Crossref for a BibTeX entry when there is a DOI to ask about."""
    if not doi:
        return ""
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT,
                          headers={"User-Agent": USER_AGENT}) as c:
            resp = c.get(f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex")
            if resp.status_code == 200:
                return resp.text.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------
_SOURCES = (
    ("Google Scholar", _scholar_search),
    ("OpenAlex", _openalex_search),
    ("arXiv", _arxiv_search),
    ("Crossref", _crossref_search),
)


async def search_scholar(query: str, limit: int = 5, year: int = 0) -> str:
    """Search academic literature and format the hits."""
    query = (query or "").strip()
    if not query:
        return "Error: no search query given."
    limit = max(1, min(int(limit or 5), 15))

    problems = []
    for name, finder in _SOURCES:
        try:
            papers = await asyncio.wait_for(
                asyncio.to_thread(finder, query, limit, int(year or 0)),
                timeout=SEARCH_TIMEOUT,
            )
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        if not papers:
            problems.append(f"{name}: no results")
            continue

        header = f"{len(papers)} paper(s) for '{query}' (source: {name})"
        if year:
            header += f" [year {year}]"
        body = "\n\n".join(_format_paper(d, i + 1) for i, d in enumerate(papers))
        return header + ":\n\n" + body

    detail = ("; ".join(problems)) if problems else "no source answered"
    return (f"No papers found for '{query}'. "
            f"Searched: {detail}. "
            f"If Google Scholar is the problem, it blocks cloud hosts - the "
            f"other sources are tried automatically.")


async def _details_scholar(title: str) -> dict:
    from scholarly import scholarly

    pub = scholarly.fill(title)
    try:
        raw = pub.to_dict()
    except Exception:
        raw = {k: getattr(pub, k, None) for k in
               ("title", "authors", "year", "venue", "abstract", "citation",
                "url", "externalIds", "publication_types")}
    authors = raw.get("authors") or []
    if isinstance(authors, dict):
        authors = authors.get("author") or []
    ext = raw.get("externalIds") or {}
    if not isinstance(ext, dict):
        ext = {}
    return _paper(
        title=raw.get("title"),
        authors=[a.get("name") if isinstance(a, dict) else str(a) for a in authors],
        year=raw.get("year"),
        venue=raw.get("venue"),
        abstract=raw.get("abstract"),
        citation=raw.get("citation"),
        url=raw.get("url"),
        doi=ext.get("DOI") or "",
        arxiv=ext.get("ArXiv") or "",
    )


async def _details_openalex(title: str) -> dict:
    papers = await asyncio.to_thread(_openalex_search, title, 1, 0)
    if not papers:
        raise RuntimeError("no matching work")
    return papers[0]


async def _details_arxiv(title: str) -> dict:
    papers = await asyncio.to_thread(_arxiv_search, title, 1, 0, True)
    if not papers:
        raise RuntimeError("no matching preprint")
    return papers[0]


async def _details_crossref(title: str) -> dict:
    papers = await asyncio.to_thread(_crossref_search, title, 1, 0)
    if not papers:
        raise RuntimeError("no matching work")
    return papers[0]


_DETAIL_SOURCES = (
    ("Google Scholar", _details_scholar),
    ("OpenAlex", _details_openalex),
    ("arXiv", _details_arxiv),
    ("Crossref", _details_crossref),
)


async def paper_details(title: str) -> str:
    """Look one paper up by title and return its full record."""
    title = (title or "").strip()
    if not title:
        return "Error: no paper title given."

    problems = []
    for name, finder in _DETAIL_SOURCES:
        try:
            paper = await asyncio.wait_for(finder(title), timeout=DETAILS_TIMEOUT)
        except Exception as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not paper or not paper.get("title"):
            problems.append(f"{name}: no match")
            continue

        out = _format_paper(paper, 1) + f"\n   source: {name}"
        bibtex = await asyncio.to_thread(
            _format_bibtex, title, paper.get("doi") or "", paper.get("arxiv") or "")
        if bibtex:
            out += "\n\nBibTeX:\n" + bibtex
        return out

    detail = ("; ".join(problems)) if problems else "no source answered"
    return f"Could not find a paper titled '{title}'. Tried: {detail}."
