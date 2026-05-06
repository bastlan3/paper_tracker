"""
Semantic Scholar retrieval channel.
- Keyword search (lexical family)
- Citation neighborhood (citation family) with reference/citation lists
- Rate: 1 req/s without API key, 100/s with S2_API_KEY env var
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = os.path.join(os.path.dirname(__file__), "../../config/sources.yaml")


def _cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)["semantic_scholar"]


def _client() -> httpx.AsyncClient:
    cfg = _cfg()
    api_key = os.environ.get("S2_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    return httpx.AsyncClient(
        base_url=cfg["base_url"],
        headers=headers,
        timeout=cfg["timeout"],
        follow_redirects=True,
    )


_FIELDS = "paperId,externalIds,title,authors,year,venue,abstract,isOpenAccess,openAccessPdf,references,citations,publicationTypes"


def _parse_paper(paper: dict) -> dict:
    ext = paper.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower() or None
    arxiv_id = ext.get("ArXiv")
    pmid = ext.get("PubMed")
    s2_id = paper.get("paperId")

    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]

    oa_url: str | None = None
    oa_pdf = paper.get("openAccessPdf")
    if oa_pdf and oa_pdf.get("url"):
        oa_url = oa_pdf["url"]

    pub_types = paper.get("publicationTypes") or []
    is_preprint = "Preprint" in pub_types

    return {
        "s2_id": s2_id,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pmid": pmid,
        "title": paper.get("title") or "",
        "authors": authors,
        "year": paper.get("year"),
        "venue": paper.get("venue") or "",
        "abstract": paper.get("abstract") or "",
        "oa_url": oa_url,
        "is_preprint": is_preprint,
        "metadata_source": "semantic_scholar",
        "_references": [r.get("paperId") for r in (paper.get("references") or []) if r.get("paperId")],
        "_cited_by": [c.get("paperId") for c in (paper.get("citations") or []) if c.get("paperId")],
    }


async def _sleep(rate: float) -> None:
    await asyncio.sleep(1.0 / rate)


# ── Public retrieval functions ────────────────────────────────────────────────

async def search_keyword(query: str, max_results: int = 100) -> list[dict]:
    cfg = _cfg()
    rate = cfg["rate_limit"]
    results: list[dict] = []
    offset = 0
    limit = min(100, max_results)

    async with _client() as client:
        while len(results) < max_results:
            try:
                resp = await client.get(
                    "/paper/search",
                    params={"query": query, "fields": _FIELDS, "limit": limit, "offset": offset},
                )
                if resp.status_code == 429:
                    logger.warning("S2 rate limited; sleeping 10s")
                    await asyncio.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("S2 search failed: %s", exc)
                break

            batch = data.get("data", [])
            if not batch:
                break
            results.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await _sleep(rate)

    return [_parse_paper(p) for p in results[:max_results]]


async def fetch_paper_by_id(s2_id: str) -> dict | None:
    """Fetch a single paper by its Semantic Scholar ID. Used by Stage 0."""
    cfg = _cfg()
    async with _client() as client:
        try:
            resp = await client.get(
                f"/paper/{s2_id}",
                params={"fields": _FIELDS},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _parse_paper(resp.json())
        except Exception as exc:
            logger.warning("S2 paper lookup failed for %s: %s", s2_id, exc)
    return None


async def fetch_paper_by_doi(doi: str) -> dict | None:
    return await fetch_paper_by_id(f"DOI:{doi}")


async def get_citation_neighborhood(
    s2_ids: list[str],
    direction: str = "both",
    max_per_paper: int = 100,
) -> list[dict]:
    """
    Expand the citation graph 1 hop from the given S2 paper IDs.
    Returns all discovered papers (with their own reference/citation lists).
    """
    cfg = _cfg()
    rate = cfg["rate_limit"]
    seen: set[str] = set(s2_ids)
    results: list[dict] = []

    async with _client() as client:
        for paper_id in s2_ids:
            if direction in ("out", "both"):
                try:
                    resp = await client.get(
                        f"/paper/{paper_id}/references",
                        params={"fields": "paperId,title,authors,year,venue,abstract,externalIds,isOpenAccess,openAccessPdf", "limit": max_per_paper},
                    )
                    resp.raise_for_status()
                    for item in resp.json().get("data", []):
                        cited = item.get("citedPaper", {})
                        pid = cited.get("paperId")
                        if pid and pid not in seen:
                            seen.add(pid)
                            results.append(_parse_paper(cited))
                except Exception as exc:
                    logger.warning("S2 references failed for %s: %s", paper_id, exc)
                await _sleep(rate)

            if direction in ("in", "both"):
                try:
                    resp = await client.get(
                        f"/paper/{paper_id}/citations",
                        params={"fields": "paperId,title,authors,year,venue,abstract,externalIds,isOpenAccess,openAccessPdf", "limit": max_per_paper},
                    )
                    resp.raise_for_status()
                    for item in resp.json().get("data", []):
                        citing = item.get("citingPaper", {})
                        pid = citing.get("paperId")
                        if pid and pid not in seen:
                            seen.add(pid)
                            results.append(_parse_paper(citing))
                except Exception as exc:
                    logger.warning("S2 citations failed for %s: %s", paper_id, exc)
                await _sleep(rate)

    return results
