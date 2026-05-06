"""
Single async writer task.
All retrieval workers push paper dicts onto a queue; one writer serialises
them to SQLite to avoid write-lock contention on WAL-mode SQLite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import unicodedata
from datetime import datetime, timezone

from ...candidates.db import (
    DBWriter,
    link_candidate_source,
    log_query,
    upsert_candidate,
    upsert_citation_edge,
    upsert_paper,
)

logger = logging.getLogger(__name__)


def canonical_id(paper: dict) -> str:
    """
    Derive a canonical paper_id in precedence order:
      DOI > OpenAlex W-id > Semantic Scholar ID > arXiv ID > PMID > title-hash

    The title-hash is (normalised-title + first-author-lastname + year) to
    reduce collisions on short generic titles (FLAG F3).
    """
    if doi := (paper.get("doi") or "").strip().lower():
        return f"doi:{doi}"
    if oa := (paper.get("openalex_id") or "").strip():
        return oa if oa.startswith("W") else f"W{oa}"
    if s2 := (paper.get("s2_id") or "").strip():
        return f"s2:{s2}"
    if ax := (paper.get("arxiv_id") or "").strip():
        return f"arxiv:{ax}"
    if pm := (paper.get("pmid") or "").strip():
        return f"pmid:{pm}"

    title = _norm_title(paper.get("title") or "")
    authors = paper.get("authors") or []
    first_auth = (
        authors[0].get("name", authors[0]) if isinstance(authors[0], dict) else authors[0]
        if authors
        else ""
    )
    last_name = first_auth.split()[-1].lower() if first_auth else ""
    year = str(paper.get("year") or "")
    digest = hashlib.sha1(f"{title}|{last_name}|{year}".encode()).hexdigest()[:12]
    return f"hash:{digest}"


def _norm_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title.lower())
    return "".join(c for c in t if c.isalnum() or c.isspace()).strip()


def normalise_authors(authors: list) -> list[str]:
    result = []
    for a in authors:
        if isinstance(a, dict):
            name = a.get("name") or f"{a.get('given','')} {a.get('family','')}".strip()
        else:
            name = str(a)
        if name.strip():
            result.append(name.strip())
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetrievalQueue:
    """
    Async queue through which retrieval workers submit discovered papers.
    Call start() before use, stop() when all workers are done.
    """

    _SENTINEL = None

    def __init__(self, writer: DBWriter, run_id: str) -> None:
        self._writer = writer
        self._run_id = run_id
        self._q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._task: asyncio.Task | None = None
        self._stats = {"total": 0, "new": 0, "updated": 0, "errors": 0}

    async def start(self) -> None:
        self._task = asyncio.create_task(self._drain(), name="retrieval-queue-drain")

    async def stop(self) -> None:
        await self._q.put(self._SENTINEL)
        if self._task:
            await self._task
        logger.info(
            "Retrieval queue done. total=%d new=%d updated=%d errors=%d",
            **self._stats,
        )

    async def submit(
        self,
        paper: dict,
        channel: str,
        query_id: str,
        rank: int | None = None,
        hop_distance: int | None = None,
        references: list[str] | None = None,
        cited_by: list[str] | None = None,
    ) -> None:
        await self._q.put({
            "paper": paper,
            "channel": channel,
            "query_id": query_id,
            "rank": rank,
            "hop_distance": hop_distance,
            "references": references or [],
            "cited_by": cited_by or [],
        })

    async def _drain(self) -> None:
        while True:
            item = await self._q.get()
            if item is None:
                self._q.task_done()
                break
            try:
                await self._process(item)
                self._stats["total"] += 1
            except Exception as exc:
                logger.error("Error processing paper in writer: %s", exc)
                self._stats["errors"] += 1
            finally:
                self._q.task_done()

    async def _process(self, item: dict) -> None:
        paper = item["paper"]
        channel = item["channel"]
        query_id = item["query_id"]
        rank = item.get("rank")
        hop_distance = item.get("hop_distance")

        # Normalise
        authors = normalise_authors(paper.get("authors") or [])
        paper_id = canonical_id(paper)
        title = (paper.get("title") or "").strip()
        if not title:
            return  # skip papers with no title

        normalised = {
            **paper,
            "paper_id": paper_id,
            "title": title,
            "title_norm": _norm_title(title),
            "authors": authors,
            "authors_json": json.dumps(authors),
            "first_author": authors[0] if authors else None,
            "fetched_at": _now(),
        }

        await upsert_paper(self._writer, normalised)
        await upsert_candidate(self._writer, self._run_id, paper_id, channel, hop_distance)
        await link_candidate_source(self._writer, self._run_id, paper_id, query_id, rank)

        # Persist citation edges (used by Stage 4 saturation)
        for ref_id in item.get("references", []):
            ref_canonical = ref_id if ref_id.startswith(("doi:", "W", "s2:")) else f"s2:{ref_id}"
            await upsert_citation_edge(self._writer, paper_id, ref_canonical)
        for citer_id in item.get("cited_by", []):
            cit_canonical = citer_id if citer_id.startswith(("doi:", "W", "s2:")) else f"s2:{citer_id}"
            await upsert_citation_edge(self._writer, cit_canonical, paper_id)
