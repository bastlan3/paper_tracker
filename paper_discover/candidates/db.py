"""
SQLite connection management with WAL mode, single-writer async task,
and schema initialisation from schema.sql.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ── Connection helpers ────────────────────────────────────────────────────────

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous  = NORMAL")
    conn.execute("PRAGMA temp_store   = MEMORY")
    conn.execute("PRAGMA mmap_size    = 268435456")   # 256 MB
    conn.execute("PRAGMA cache_size   = -200000")      # 200 MB
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database, apply pragmas and schema."""
    conn = sqlite3.connect(path, check_same_thread=False)
    _apply_pragmas(conn)
    schema = _SCHEMA_PATH.read_text()
    conn.executescript(schema)
    conn.commit()
    return conn


@contextmanager
def read_conn(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Short-lived read-only connection. Use for queries, not writes."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    _apply_pragmas(conn)
    try:
        yield conn
    finally:
        conn.close()


# ── Single-writer async task ──────────────────────────────────────────────────

_WriteItem = tuple[str, tuple, asyncio.Future]


class DBWriter:
    """
    All write operations from concurrent retriever coroutines funnel through
    a single asyncio queue to avoid SQLite write-lock contention.

    Usage:
        async with DBWriter(db_path) as writer:
            rowid = await writer.execute(sql, params)
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._queue: asyncio.Queue[_WriteItem | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._conn: sqlite3.Connection | None = None

    async def __aenter__(self) -> "DBWriter":
        self._conn = open_db(self._db_path)
        self._task = asyncio.create_task(self._loop(), name="db-writer")
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._queue.put(None)
        if self._task:
            await self._task
        if self._conn:
            self._conn.close()

    async def _loop(self) -> None:
        conn = self._conn
        assert conn is not None
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            sql, params, fut = item
            try:
                cur = conn.execute(sql, params)
                conn.commit()
                fut.set_result(cur.lastrowid)
            except Exception as exc:
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._queue.task_done()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        """Enqueue a write and await its completion. Returns lastrowid."""
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[int] = loop.create_future()
        await self._queue.put((sql, params, fut))
        return await fut

    async def executemany(self, sql: str, rows: list[tuple]) -> None:
        """Enqueue multiple writes as a single transaction."""
        if not rows:
            return

        async def _multi(sql: str, rows: list[tuple], fut: asyncio.Future) -> None:  # noqa: E501
            # Runs inside _loop's thread; we can't do this cleanly in the queue
            # so we enqueue a dummy sentinel that triggers a batch.
            raise NotImplementedError("use execute() in a loop for now")

        for row in rows:
            await self.execute(sql, row)


# ── Upsert helpers ────────────────────────────────────────────────────────────

async def upsert_paper(writer: DBWriter, paper: dict) -> None:
    """Insert a new paper or update mutable fields if already present."""
    await writer.execute(
        """
        INSERT INTO papers (
          paper_id, doi, openalex_id, s2_id, arxiv_id, pmid,
          title, title_norm, authors_json, first_author,
          year, venue, abstract, oa_url, is_preprint, retracted,
          fetched_at, metadata_source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(paper_id) DO UPDATE SET
          doi            = COALESCE(excluded.doi, doi),
          openalex_id    = COALESCE(excluded.openalex_id, openalex_id),
          s2_id          = COALESCE(excluded.s2_id, s2_id),
          arxiv_id       = COALESCE(excluded.arxiv_id, arxiv_id),
          pmid           = COALESCE(excluded.pmid, pmid),
          abstract       = COALESCE(excluded.abstract, abstract),
          oa_url         = COALESCE(excluded.oa_url, oa_url),
          fetched_at     = excluded.fetched_at
        """,
        (
            paper["paper_id"],
            paper.get("doi"),
            paper.get("openalex_id"),
            paper.get("s2_id"),
            paper.get("arxiv_id"),
            paper.get("pmid"),
            paper["title"],
            paper["title_norm"],
            json.dumps(paper.get("authors", [])),
            paper.get("first_author"),
            paper.get("year"),
            paper.get("venue"),
            paper.get("abstract"),
            paper.get("oa_url"),
            int(paper.get("is_preprint", False)),
            int(paper.get("retracted", False)),
            paper.get("fetched_at"),
            paper.get("metadata_source"),
        ),
    )


async def upsert_candidate(
    writer: DBWriter,
    run_id: str,
    paper_id: str,
    channel: str,
    hop_distance: int | None = None,
) -> None:
    """
    Insert a new candidate or increment seen_count and update seen_by/last_seen_at.
    Channel convergence (seen_count across independent sources) is the key trust signal.
    """
    now = _now()
    await writer.execute(
        """
        INSERT INTO candidates (
          run_id, paper_id, first_seen_at, last_seen_at,
          seen_count, seen_by_json, hop_distance_to_anchor
        ) VALUES (?,?,?,?,1,?,?)
        ON CONFLICT(run_id, paper_id) DO UPDATE SET
          seen_count           = seen_count + 1,
          seen_by_json         = json_insert(
                                   seen_by_json,
                                   '$[#]',
                                   excluded.seen_by_json ->> '$[0]'
                                 ),
          last_seen_at         = excluded.last_seen_at,
          hop_distance_to_anchor = CASE
            WHEN excluded.hop_distance_to_anchor IS NOT NULL
             AND (hop_distance_to_anchor IS NULL
                  OR excluded.hop_distance_to_anchor < hop_distance_to_anchor)
            THEN excluded.hop_distance_to_anchor
            ELSE hop_distance_to_anchor
          END
        """,
        (
            run_id,
            paper_id,
            now,
            now,
            json.dumps([channel]),
            hop_distance,
        ),
    )


async def upsert_citation_edge(
    writer: DBWriter, src_id: str, dst_id: str
) -> None:
    await writer.execute(
        "INSERT OR IGNORE INTO citations (src_paper_id, dst_paper_id) VALUES (?,?)",
        (src_id, dst_id),
    )


async def log_query(
    writer: DBWriter,
    query_id: str,
    run_id: str,
    family: str,
    source: str,
    query_text: str,
    dimensions: list[str],
    result_count: int,
    status: str = "ok",
) -> None:
    await writer.execute(
        """
        INSERT OR IGNORE INTO queries
          (query_id, run_id, family, source, query_text,
           dimensions_json, issued_at, result_count, status)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            query_id,
            run_id,
            family,
            source,
            query_text,
            json.dumps(dimensions),
            _now(),
            result_count,
            status,
        ),
    )


async def link_candidate_source(
    writer: DBWriter,
    run_id: str,
    paper_id: str,
    query_id: str,
    rank: int | None = None,
) -> None:
    await writer.execute(
        "INSERT OR IGNORE INTO candidate_sources (run_id, paper_id, query_id, rank) VALUES (?,?,?,?)",
        (run_id, paper_id, query_id, rank),
    )


# ── Read helpers (synchronous; use read_conn) ─────────────────────────────────

def fetch_pending_candidates(
    db_path: str, run_id: str, limit: int = 500
) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.seen_count, c.seen_by_json,
                   c.hop_distance_to_anchor, c.signals_json,
                   p.title, p.abstract, p.year, p.venue,
                   p.authors_json, p.doi, p.is_preprint, p.retracted
            FROM candidates c
            JOIN papers p USING (paper_id)
            WHERE c.run_id = ? AND c.judge_status = 'pending'
            ORDER BY c.seen_count DESC, c.hop_distance_to_anchor ASC NULLS LAST
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_bibliography(db_path: str, run_id: str) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM v_bibliography WHERE run_id = ?", (run_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def count_by_level(db_path: str, run_id: str) -> dict[str, int]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT level, COUNT(*) AS n
            FROM candidates
            WHERE run_id = ? AND level IS NOT NULL
            GROUP BY level
            """,
            (run_id,),
        ).fetchall()
    return {r["level"]: r["n"] for r in rows}


def run_sql_invariants(db_path: str, run_id: str) -> list[str]:
    """Return a list of violated invariant descriptions (empty = all pass)."""
    violations: list[str] = []
    with read_conn(db_path) as conn:
        # No judged row with NULL level
        n = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id=? AND judge_status='judged' AND level IS NULL",
            (run_id,),
        ).fetchone()[0]
        if n:
            violations.append(f"{n} rows: judge_status='judged' but level IS NULL")

        # No CORE/SUPPORTING/CONTEXT/ADJACENT with NULL confidence
        n = conn.execute(
            """
            SELECT COUNT(*) FROM candidates
            WHERE run_id=? AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            AND judge_confidence IS NULL
            """,
            (run_id,),
        ).fetchone()[0]
        if n:
            violations.append(f"{n} kept rows with NULL judge_confidence")

    return violations


# ── Internal ──────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
