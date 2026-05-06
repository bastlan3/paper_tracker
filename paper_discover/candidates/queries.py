"""
Parameterised SQL read queries used by pipeline stages.
All queries are synchronous (use read_conn from db.py).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import read_conn


def get_run(db_path: str, run_id: str) -> dict | None:
    with read_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def get_anchors(db_path: str, run_id: str) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.*
            FROM run_anchors a JOIN papers p USING (paper_id)
            WHERE a.run_id = ?
            """,
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def search_fts(
    db_path: str, query: str, limit: int = 200
) -> list[dict]:
    """Full-text search over papers_fts. Returns matching papers."""
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.paper_id, p.title, p.abstract, p.year, p.venue,
                   p.authors_json, p.doi, p.openalex_id, p.s2_id, p.is_preprint
            FROM papers_fts f
            JOIN papers p ON p.rowid = f.rowid
            WHERE papers_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_paper_by_doi(db_path: str, doi: str) -> dict | None:
    with read_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE doi = ?", (doi,)
        ).fetchone()
    return dict(row) if row else None


def get_paper_by_id(db_path: str, paper_id: str) -> dict | None:
    with read_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
    return dict(row) if row else None


def get_candidate(db_path: str, run_id: str, paper_id: str) -> dict | None:
    with read_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE run_id = ? AND paper_id = ?",
            (run_id, paper_id),
        ).fetchone()
    return dict(row) if row else None


def list_runs(db_path: str) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def citation_neighborhood(
    db_path: str, paper_ids: list[str], direction: str = "both"
) -> set[str]:
    """Return paper_ids 1 hop out from the given seeds via citation edges."""
    if not paper_ids:
        return set()
    placeholders = ",".join("?" * len(paper_ids))
    result: set[str] = set()
    with read_conn(db_path) as conn:
        if direction in ("out", "both"):
            rows = conn.execute(
                f"SELECT dst_paper_id FROM citations WHERE src_paper_id IN ({placeholders})",
                paper_ids,
            ).fetchall()
            result.update(r[0] for r in rows)
        if direction in ("in", "both"):
            rows = conn.execute(
                f"SELECT src_paper_id FROM citations WHERE dst_paper_id IN ({placeholders})",
                paper_ids,
            ).fetchall()
            result.update(r[0] for r in rows)
    return result - set(paper_ids)


def zotero_paper_ids(db_path: str, run_id: str) -> set[str]:
    """Paper IDs already in the user's Zotero library for this run."""
    with read_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT paper_id FROM zotero_items WHERE run_id = ? AND paper_id IS NOT NULL",
            (run_id,),
        ).fetchall()
    return {r[0] for r in rows}
