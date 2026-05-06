"""
OpenAlex retrieval channel.
- Keyword / boolean search (lexical family)
- Embedding similarity search via OpenAlex's native concept matching (semantic family)
- Citation neighborhood expansion (citation family)
- Rate: 10 req/s with polite-pool email header
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

_CFG_PATH = os.path.join(os.path.dirname(__file__), "../../config/sources.yaml")


def _cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)["openalex"]


def _client() -> httpx.AsyncClient:
    cfg = _cfg()
    email = os.environ.get("PAPER_DISCOVER_EMAIL", cfg.get("polite_email", ""))
    headers = {"User-Agent": f"paper-discover/0.1 (mailto:{email})"} if email else {}
    return httpx.AsyncClient(
        base_url=cfg["base_url"],
        headers=headers,
        timeout=cfg["timeout"],
        follow_redirects=True,
    )


_FIELDS = (
    "id,doi,title,authorships,publication_year,primary_location,"
    "abstract_inverted_index,is_oa,open_access,referenced_works,"
    "cited_by_api_url,type,best_oa_location"
)


def _parse_work(work: dict) -> dict:
    """Convert an OpenAlex Work object into our normalised paper dict."""
    oa_id = work.get("id", "")
    doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None

    # Reconstruct abstract from inverted index
    abstract = _invert_abstract(work.get("abstract_inverted_index"))

    authors = []
    for auth in work.get("authorships", []):
        author = auth.get("author", {})
        name = author.get("display_name", "")
        if name:
            authors.append(name)

    venue = (
        (work.get("primary_location") or {})
        .get("source", {})
        .get("display_name", "")
    ) or ""

    oa_url: str | None = None
    best_oa = work.get("best_oa_location") or {}
    if best_oa.get("pdf_url"):
        oa_url = best_oa["pdf_url"]
    elif best_oa.get("landing_page_url"):
        oa_url = best_oa["landing_page_url"]

    return {
        "openalex_id": oa_id.split("/")[-1] if oa_id else None,
        "doi": doi,
        "title": work.get("title") or "",
        "authors": authors,
        "year": work.get("publication_year"),
        "venue": venue,
        "abstract": abstract,
        "oa_url": oa_url,
        "is_preprint": work.get("type") in ("preprint",),
        "metadata_source": "openalex",
    }


def _invert_abstract(index: dict | None) -> str | None:
    if not index:
        return None
    positions: list[tuple[int, str]] = []
    for word, pos_list in index.items():
        for pos in pos_list:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


async def _get_works(params: dict, max_results: int) -> list[dict]:
    cfg = _cfg()
    rate = cfg["rate_limit"]
    results: list[dict] = []
    cursor = "*"
    per_page = min(200, max_results)

    async with _client() as client:
        while len(results) < max_results:
            query = {**params, "per-page": per_page, "cursor": cursor, "select": _FIELDS}
            try:
                resp = await client.get("/works", params=query)
                if resp.status_code == 429:
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("OpenAlex request failed: %s", exc)
                break

            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            await asyncio.sleep(1.0 / rate)

    return results[:max_results]


# ── Public retrieval functions ────────────────────────────────────────────────

async def search_keyword(
    query: str, max_results: int = 100
) -> list[dict]:
    """Free-text search over OpenAlex works."""
    raw = await _get_works({"search": query}, max_results)
    return [_parse_work(w) for w in raw]


async def search_by_concept(
    query: str, max_results: int = 100
) -> list[dict]:
    """Use OpenAlex title+abstract full-text search as a semantic proxy."""
    raw = await _get_works({"search": query, "sort": "relevance_score:desc"}, max_results)
    return [_parse_work(w) for w in raw]


async def get_citation_neighborhood(
    openalex_ids: list[str],
    direction: str = "both",
    max_results: int = 200,
) -> list[dict]:
    """Fetch works that cite or are cited by the given OpenAlex IDs."""
    if not openalex_ids:
        return []

    all_results: list[dict] = []
    pipe_ids = "|".join(openalex_ids)
    cfg = _cfg()
    rate = cfg["rate_limit"]

    async with _client() as client:
        if direction in ("in", "both"):
            params = {
                "filter": f"cites:{pipe_ids}",
                "per-page": min(200, max_results),
                "select": _FIELDS,
            }
            try:
                resp = await client.get("/works", params=params)
                resp.raise_for_status()
                all_results.extend(resp.json().get("results", []))
            except Exception as exc:
                logger.warning("OpenAlex cited-by failed: %s", exc)
            await asyncio.sleep(1.0 / rate)

        if direction in ("out", "both"):
            params = {
                "filter": f"cited_by:{pipe_ids}",
                "per-page": min(200, max_results),
                "select": _FIELDS,
            }
            try:
                resp = await client.get("/works", params=params)
                resp.raise_for_status()
                all_results.extend(resp.json().get("results", []))
            except Exception as exc:
                logger.warning("OpenAlex references failed: %s", exc)

    return [_parse_work(w) for w in all_results[:max_results]]


async def fetch_paper_by_doi(doi: str) -> dict | None:
    """Fetch a single paper by DOI. Used by Stage 0 anchor resolution."""
    async with _client() as client:
        try:
            resp = await client.get(
                "/works",
                params={"filter": f"doi:{doi}", "select": _FIELDS},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                return _parse_work(results[0])
        except Exception as exc:
            logger.warning("OpenAlex DOI lookup failed for %s: %s", doi, exc)
    return None


async def fetch_paper_by_openalex_id(oa_id: str) -> dict | None:
    """Fetch a single paper by its OpenAlex W-id."""
    wid = oa_id if oa_id.startswith("W") else f"W{oa_id}"
    async with _client() as client:
        try:
            resp = await client.get(f"/works/{wid}", params={"select": _FIELDS})
            resp.raise_for_status()
            return _parse_work(resp.json())
        except Exception as exc:
            logger.warning("OpenAlex ID lookup failed for %s: %s", oa_id, exc)
    return None
