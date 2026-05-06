"""
Paper fetcher: Semantic Scholar API client + PDF downloader.

Uses the Semantic Scholar public Graph API (no key required for basic use;
set SEMANTIC_SCHOLAR_API_KEY for higher rate limits).
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SS_BASE = "https://api.semanticscholar.org/graph/v1"
SS_PAPER_FIELDS = (
    "paperId,title,authors,abstract,year,venue,"
    "externalIds,openAccessPdf,citationCount,referenceCount"
)
SS_REF_FIELDS = (
    "title,authors,abstract,year,venue,externalIds,openAccessPdf"
)


def _parse_arxiv_id(text: str) -> Optional[str]:
    """Extract bare arxiv ID (YYMM.NNNNN) from a URL or raw string."""
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
    return m.group(1) if m else None


def _parse_doi(text: str) -> Optional[str]:
    """Extract DOI from a URL or raw string."""
    m = re.search(r"10\.\d{4,9}/[^\s]+", text)
    return m.group(0).rstrip(".,;") if m else None


class SemanticScholarClient:
    """Thin async wrapper around the Semantic Scholar Graph API."""

    def __init__(self, api_key: Optional[str] = None):
        headers = {"User-Agent": "LitReviewBot/1.0"}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        # Conservative rate limit: 1 req/s without key, 5 req/s with key
        self._interval = 0.25 if api_key else 1.1
        self._last_req: float = 0.0

    async def _get(self, url: str, params: Optional[dict] = None) -> dict:
        now = asyncio.get_event_loop().time()
        wait = self._interval - (now - self._last_req)
        if wait > 0:
            await asyncio.sleep(wait)
        for attempt in range(3):
            try:
                resp = await self._client.get(url, params=params)
                self._last_req = asyncio.get_event_loop().time()
                if resp.status_code == 429:
                    await asyncio.sleep(15 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (404, 400):
                    return {}
                if attempt == 2:
                    raise
                await asyncio.sleep(3)
        return {}

    async def find_paper(
        self,
        *,
        arxiv_id: Optional[str] = None,
        doi: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Locate a paper via arxiv ID, DOI, or title search.
        Returns a Semantic Scholar paper dict or None.
        """
        if arxiv_id:
            try:
                data = await self._get(
                    f"{SS_BASE}/paper/arXiv:{arxiv_id}",
                    params={"fields": SS_PAPER_FIELDS},
                )
                if data.get("paperId"):
                    return data
            except Exception as e:
                logger.debug(f"SS arxiv lookup failed ({arxiv_id}): {e}")

        if doi:
            try:
                data = await self._get(
                    f"{SS_BASE}/paper/DOI:{doi}",
                    params={"fields": SS_PAPER_FIELDS},
                )
                if data.get("paperId"):
                    return data
            except Exception as e:
                logger.debug(f"SS DOI lookup failed ({doi}): {e}")

        if title:
            try:
                data = await self._get(
                    f"{SS_BASE}/paper/search",
                    params={"query": title, "fields": SS_PAPER_FIELDS, "limit": 1},
                )
                results = data.get("data", [])
                if results:
                    return results[0]
            except Exception as e:
                logger.debug(f"SS title search failed ('{title[:60]}'): {e}")

        return None

    async def get_references(
        self, paper_id: str, max_refs: int = 200
    ) -> list[dict]:
        """
        Retrieve references (cited papers) for a given Semantic Scholar paper ID.
        Returns a list of paper dicts (the 'citedPaper' side of each reference edge).
        """
        all_refs: list[dict] = []
        offset = 0
        page_size = 100

        while len(all_refs) < max_refs:
            to_fetch = min(page_size, max_refs - len(all_refs))
            try:
                data = await self._get(
                    f"{SS_BASE}/paper/{paper_id}/references",
                    params={
                        "fields": SS_REF_FIELDS,
                        "limit": to_fetch,
                        "offset": offset,
                    },
                )
            except Exception as e:
                logger.warning(f"get_references failed for {paper_id}: {e}")
                break

            batch = [
                edge["citedPaper"]
                for edge in data.get("data", [])
                if edge.get("citedPaper") and edge["citedPaper"].get("paperId")
            ]
            all_refs.extend(batch)

            if len(batch) < to_fetch:
                break  # No more pages
            offset += page_size

        return all_refs

    async def close(self):
        await self._client.aclose()


async def resolve_initial_paper(query: str, ss_client: SemanticScholarClient) -> Optional[dict]:
    """
    Given an arxiv URL/ID, DOI, or free-text title, resolve to a Semantic Scholar record.
    """
    arxiv_id = _parse_arxiv_id(query)
    doi = _parse_doi(query)

    paper = await ss_client.find_paper(arxiv_id=arxiv_id, doi=doi)
    if paper:
        return paper

    # Fall back to title search if the query looks like a title
    if len(query) > 20 and not query.startswith("http"):
        paper = await ss_client.find_paper(title=query)

    return paper


async def download_pdf(paper_info: dict, save_dir: Path) -> Optional[str]:
    """
    Download PDF for a paper.  Tries (in order):
      1. Semantic Scholar openAccessPdf URL
      2. arxiv PDF URL derived from externalIds
    Returns the local file path string, or None if unavailable.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine candidate PDF URL
    pdf_url: Optional[str] = None

    oa = paper_info.get("openAccessPdf")
    if oa and oa.get("url"):
        pdf_url = oa["url"]

    if not pdf_url:
        ext_ids = paper_info.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    if not pdf_url:
        return None

    # Build a safe filename from the paperId
    paper_id = paper_info.get("paperId", "unknown")
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", paper_id)[:40]
    filepath = save_dir / f"{safe_id}.pdf"

    if filepath.exists() and filepath.stat().st_size > 1000:
        return str(filepath)

    try:
        async with httpx.AsyncClient(
            timeout=90.0, follow_redirects=True
        ) as client:
            resp = await client.get(pdf_url)
            if resp.status_code == 200 and len(resp.content) > 5000:
                filepath.write_bytes(resp.content)
                logger.info(f"Downloaded PDF → {filepath.name}")
                return str(filepath)
            else:
                logger.debug(
                    f"PDF download returned {resp.status_code} / "
                    f"{len(resp.content)} bytes for {paper_info.get('title', '?')[:50]}"
                )
    except Exception as e:
        logger.warning(
            f"PDF download failed for '{paper_info.get('title', '?')[:50]}': {e}"
        )

    return None


def extract_metadata(ss_paper: dict) -> dict:
    """
    Flatten a Semantic Scholar paper dict into a simple metadata dict
    suitable for storing in LitReviewPaper.
    """
    ext_ids = ss_paper.get("externalIds") or {}
    authors = [a.get("name", "") for a in (ss_paper.get("authors") or [])]
    oa = ss_paper.get("openAccessPdf") or {}

    arxiv_id = ext_ids.get("ArXiv")
    doi = ext_ids.get("DOI")

    url = None
    if arxiv_id:
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif doi:
        url = f"https://doi.org/{doi}"

    return {
        "semantic_scholar_id": ss_paper.get("paperId"),
        "title": (ss_paper.get("title") or "").strip(),
        "authors": authors,
        "abstract": (ss_paper.get("abstract") or "").strip(),
        "year": ss_paper.get("year"),
        "venue": ss_paper.get("venue") or None,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "url": url,
        "pdf_url": oa.get("url"),
    }
