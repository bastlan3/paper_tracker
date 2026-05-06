"""
Tests for the candidates DB layer: schema init, upserts, dedup, WAL mode.
"""

import asyncio
import json
import os
import tempfile

import pytest

from paper_discover.candidates.db import (
    DBWriter,
    fetch_bibliography,
    fetch_pending_candidates,
    open_db,
    run_sql_invariants,
    upsert_candidate,
    upsert_paper,
)
from paper_discover.candidates.queries import citation_neighborhood, zotero_paper_ids


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return db_path


def _sample_paper(idx: int = 0) -> dict:
    return {
        "paper_id": f"doi:10.1234/test{idx}",
        "doi": f"10.1234/test{idx}",
        "openalex_id": None,
        "s2_id": None,
        "arxiv_id": None,
        "pmid": None,
        "title": f"Test Paper {idx}",
        "title_norm": f"test paper {idx}",
        "authors": [f"Author {idx}"],
        "authors_json": json.dumps([f"Author {idx}"]),
        "first_author": f"Author {idx}",
        "year": 2024,
        "venue": "Test Journal",
        "abstract": "This is a test abstract with enough content for triage." * 3,
        "oa_url": None,
        "is_preprint": False,
        "retracted": False,
        "fetched_at": "2024-01-01T00:00:00+00:00",
        "metadata_source": "test",
    }


def _sample_run(conn, run_id: str = "01TEST") -> None:
    conn.execute(
        "INSERT INTO runs (run_id, started_at, mode, plan_json, status) "
        "VALUES (?,?,?,?,?)",
        (run_id, "2024-01-01T00:00:00+00:00", "deep", '{"intent":"test"}', "running"),
    )
    conn.commit()


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_schema_init_creates_tables(tmp_db):
    conn = open_db(tmp_db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    expected = {"papers", "candidates", "runs", "queries", "citations",
                "run_anchors", "candidate_sources", "embeddings",
                "skeptic_flags", "anchor_probes", "saved_searches", "zotero_items"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"
    conn.close()


def test_schema_wal_mode(tmp_db):
    conn = open_db(tmp_db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


def test_schema_fts_table_exists(tmp_db):
    conn = open_db(tmp_db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "papers_fts" in tables
    conn.close()


def test_schema_views_exist(tmp_db):
    conn = open_db(tmp_db)
    views = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()}
    assert "v_bibliography" in views
    assert "v_discovery_curve" in views
    conn.close()


# ── Upsert tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_paper(tmp_db):
    async with DBWriter(tmp_db) as writer:
        await upsert_paper(writer, _sample_paper(0))

    conn = open_db(tmp_db)
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?",
                       ("doi:10.1234/test0",)).fetchone()
    assert row is not None
    assert row["title"] == "Test Paper 0"
    conn.close()


@pytest.mark.asyncio
async def test_upsert_paper_idempotent(tmp_db):
    paper = _sample_paper(0)
    async with DBWriter(tmp_db) as writer:
        await upsert_paper(writer, paper)
        await upsert_paper(writer, paper)  # second upsert

    conn = open_db(tmp_db)
    count = conn.execute("SELECT COUNT(*) FROM papers WHERE paper_id = ?",
                         ("doi:10.1234/test0",)).fetchone()[0]
    assert count == 1
    conn.close()


@pytest.mark.asyncio
async def test_upsert_candidate_seen_count(tmp_db):
    conn = open_db(tmp_db)
    _sample_run(conn, "R1")

    async with DBWriter(tmp_db) as writer:
        await upsert_paper(writer, _sample_paper(0))
        await upsert_candidate(writer, "R1", "doi:10.1234/test0", "openalex:lexical")
        await upsert_candidate(writer, "R1", "doi:10.1234/test0", "s2:semantic")   # second channel

    conn = open_db(tmp_db)
    row = conn.execute(
        "SELECT seen_count, seen_by_json FROM candidates WHERE run_id='R1' AND paper_id=?",
        ("doi:10.1234/test0",),
    ).fetchone()
    assert row["seen_count"] == 2
    seen_by = json.loads(row["seen_by_json"])
    assert "openalex:lexical" in seen_by
    assert "s2:semantic" in seen_by
    conn.close()


@pytest.mark.asyncio
async def test_fetch_pending_candidates(tmp_db):
    conn = open_db(tmp_db)
    _sample_run(conn, "R2")

    async with DBWriter(tmp_db) as writer:
        for i in range(3):
            await upsert_paper(writer, _sample_paper(i))
            await upsert_candidate(writer, "R2", f"doi:10.1234/test{i}", "test")

    pending = fetch_pending_candidates(tmp_db, "R2")
    assert len(pending) == 3


# ── Invariant tests ───────────────────────────────────────────────────────────

def test_invariants_pass_on_empty_run(tmp_db):
    conn = open_db(tmp_db)
    _sample_run(conn, "R3")
    conn.close()
    violations = run_sql_invariants(tmp_db, "R3")
    assert violations == []


@pytest.mark.asyncio
async def test_invariant_judged_with_null_level_is_caught(tmp_db):
    """Simulate a bug where judge_status=judged but level is NULL — invariant should catch it."""
    conn = open_db(tmp_db)
    _sample_run(conn, "R4")
    async with DBWriter(tmp_db) as writer:
        await upsert_paper(writer, _sample_paper(0))
        await upsert_candidate(writer, "R4", "doi:10.1234/test0", "test")

    # Manually corrupt the row
    conn = open_db(tmp_db)
    conn.execute(
        "UPDATE candidates SET judge_status='judged', level=NULL WHERE run_id='R4'"
    )
    conn.commit()
    conn.close()

    violations = run_sql_invariants(tmp_db, "R4")
    assert len(violations) == 1
    assert "judge_status='judged'" in violations[0]


# ── Citation graph tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_citation_neighborhood(tmp_db):
    open_db(tmp_db)
    async with DBWriter(tmp_db) as writer:
        for i in range(3):
            await upsert_paper(writer, _sample_paper(i))
        from paper_discover.candidates.db import upsert_citation_edge
        # paper 0 cites paper 1 and paper 2
        await upsert_citation_edge(writer, "doi:10.1234/test0", "doi:10.1234/test1")
        await upsert_citation_edge(writer, "doi:10.1234/test0", "doi:10.1234/test2")

    neighbors = citation_neighborhood(tmp_db, ["doi:10.1234/test0"], direction="out")
    assert neighbors == {"doi:10.1234/test1", "doi:10.1234/test2"}
