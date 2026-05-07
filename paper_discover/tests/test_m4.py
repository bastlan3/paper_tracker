"""
M4 unit tests — Stage 6 hygiene (retraction parsing, errata, OA resolution).

The HTTP layer is exercised via a tiny stub so no real network is needed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from paper_discover.candidates.db import open_db
from paper_discover.pipeline.stage6_hygiene import (
    parse_errata,
    parse_europepmc_oa,
    parse_retraction,
    parse_unpaywall_oa,
    run_hygiene,
    _persist,
)
from paper_discover.pipeline.stage8_report.bibliography import generate_summary_md


# ── parse_retraction ─────────────────────────────────────────────────────────

def test_retraction_via_relation_is_retraction_of():
    work = {"message": {"relation": {
        "is-retraction-of": [{"id-type": "doi", "id": "10.x/orig"}]
    }}}
    out = parse_retraction(work)
    assert out is not None
    assert out["type"] == "is-retraction-of"
    assert out["details"][0]["id"] == "10.x/orig"


def test_retraction_via_update_to():
    work = {"message": {"update-to": [
        {"type": "retraction", "DOI": "10.x/orig", "label": "Retraction"}
    ]}}
    out = parse_retraction(work)
    assert out is not None
    assert out["type"] == "retraction"


def test_no_retraction_returns_none():
    work = {"message": {"DOI": "10.x/y", "type": "journal-article"}}
    assert parse_retraction(work) is None


def test_retraction_handles_message_at_top_level():
    """Some clients pass the unwrapped 'message' dict."""
    work = {"relation": {"is-retraction-of": [{"id": "10.x/orig"}]}}
    assert parse_retraction(work) is not None


# ── parse_errata ─────────────────────────────────────────────────────────────

def test_errata_via_relation_correction():
    work = {"message": {"relation": {
        "is-corrected-by": [{"id-type": "doi", "id": "10.x/correction"}]
    }}}
    errata = parse_errata(work)
    assert len(errata) == 1
    assert errata[0]["type"] == "is-corrected-by"


def test_errata_via_update_to_erratum():
    work = {"message": {"update-to": [{"type": "erratum", "DOI": "10.x/e"}]}}
    errata = parse_errata(work)
    assert len(errata) == 1
    assert errata[0]["type"] == "erratum"


def test_no_errata_returns_empty_list():
    assert parse_errata({"message": {}}) == []


# ── parse_unpaywall_oa ───────────────────────────────────────────────────────

def test_unpaywall_returns_pdf_url_when_available():
    resp = {"best_oa_location": {
        "url": "https://example.com/abstract",
        "url_for_pdf": "https://example.com/paper.pdf",
    }}
    assert parse_unpaywall_oa(resp) == "https://example.com/paper.pdf"


def test_unpaywall_falls_back_to_url():
    resp = {"best_oa_location": {"url": "https://example.com/abstract"}}
    assert parse_unpaywall_oa(resp) == "https://example.com/abstract"


def test_unpaywall_falls_back_to_other_locations():
    resp = {
        "best_oa_location": {},
        "oa_locations": [{"url_for_pdf": "https://mirror.example/paper.pdf"}],
    }
    assert parse_unpaywall_oa(resp) == "https://mirror.example/paper.pdf"


def test_unpaywall_no_oa_returns_none():
    assert parse_unpaywall_oa({"best_oa_location": None}) is None


# ── parse_europepmc_oa ───────────────────────────────────────────────────────

def test_europepmc_returns_open_access_url():
    resp = {"resultList": {"result": [{
        "isOpenAccess": "Y",
        "fullTextUrlList": {"fullTextUrl": [
            {"availability": "Open access", "url": "https://europepmc.org/articles/PMC123"},
        ]},
    }]}}
    assert parse_europepmc_oa(resp) == "https://europepmc.org/articles/PMC123"


def test_europepmc_falls_back_to_pmcid():
    resp = {"resultList": {"result": [{
        "isOpenAccess": "Y",
        "pmcid": "PMC456",
    }]}}
    assert parse_europepmc_oa(resp) == "https://europepmc.org/article/PMC/456"


def test_europepmc_not_open_access_returns_none():
    resp = {"resultList": {"result": [{"isOpenAccess": "N"}]}}
    assert parse_europepmc_oa(resp) is None


def test_europepmc_empty_result_returns_none():
    assert parse_europepmc_oa({"resultList": {"result": []}}) is None


# ── _persist (DB-level idempotency + flag setting) ───────────────────────────

def _seed_paper(conn: sqlite3.Connection, run_id: str, paper_id: str, doi: str):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, started_at, mode, plan_json, status) VALUES (?,?,?,?,?)",
        (run_id, now, "deep", "{}", "running"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO papers (paper_id, doi, title, title_norm) VALUES (?,?,?,?)",
        (paper_id, doi, "T", "t"),
    )
    conn.execute(
        """INSERT INTO candidates (
            run_id, paper_id, first_seen_at, last_seen_at,
            seen_count, seen_by_json, level, judge_status, judge_confidence
           ) VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, paper_id, now, now, 1, '["s"]', "CORE", "judged", 0.9),
    )
    conn.commit()


def test_persist_marks_retracted_and_adds_flag():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_paper(conn, "r1", "p1", "10.x/y")
        row = {"paper_id": "p1", "flags_json": "[]", "doi": "10.x/y", "oa_url": None, "retracted": 0}
        update = {"retracted": True, "retraction_notice": {"src": "test"}}
        changed = _persist(conn, "r1", row, update)
        conn.commit()
        assert changed.get("retracted") is True
        flags = conn.execute(
            "SELECT flags_json FROM candidates WHERE run_id=? AND paper_id=?", ("r1", "p1")
        ).fetchone()[0]
        assert "retracted" in json.loads(flags)
        retracted = conn.execute(
            "SELECT retracted FROM papers WHERE paper_id=?", ("p1",)
        ).fetchone()[0]
        assert retracted == 1
        conn.close()
    finally:
        os.unlink(path)


def test_persist_does_not_overwrite_existing_oa_url():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_paper(conn, "r1", "p1", "10.x/y")
        conn.execute("UPDATE papers SET oa_url='https://existing.example' WHERE paper_id='p1'")
        conn.commit()
        row = {"paper_id": "p1", "flags_json": "[]", "doi": "10.x/y",
               "oa_url": "https://existing.example", "retracted": 0}
        update = {"oa_url": "https://other.example"}
        changed = _persist(conn, "r1", row, update)
        assert changed.get("oa_url") is None
        url = conn.execute("SELECT oa_url FROM papers WHERE paper_id='p1'").fetchone()[0]
        assert url == "https://existing.example"
        conn.close()
    finally:
        os.unlink(path)


def test_persist_empty_update_is_noop():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_paper(conn, "r1", "p1", "10.x/y")
        row = {"paper_id": "p1", "flags_json": "[]", "doi": "10.x/y", "oa_url": None, "retracted": 0}
        assert _persist(conn, "r1", row, {}) == {}
        conn.close()
    finally:
        os.unlink(path)


# ── run_hygiene end-to-end with stubbed HTTP ────────────────────────────────

def test_run_hygiene_no_dois_returns_zero(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        # Seed run + a kept paper without a DOI
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO runs (run_id, started_at, mode, plan_json, status) VALUES (?,?,?,?,?)",
            ("r1", now, "deep", "{}", "running"),
        )
        conn.execute(
            "INSERT INTO papers (paper_id, title, title_norm) VALUES (?,?,?)",
            ("p1", "T", "t"),
        )
        conn.execute(
            """INSERT INTO candidates (run_id, paper_id, first_seen_at, last_seen_at,
                  seen_count, seen_by_json, level, judge_status, judge_confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("r1", "p1", now, now, 1, '["s"]', "CORE", "judged", 0.9),
        )
        conn.commit()
        conn.close()
        out = asyncio.run(run_hygiene(path, "r1"))
        assert out == {"checked": 0, "retracted": 0, "errata": 0, "oa_resolved": 0, "errors": 0}
    finally:
        os.unlink(path)


def test_run_hygiene_marks_retraction_via_stub(monkeypatch):
    """End-to-end: stub Crossref to report retraction; verify DB state."""
    import paper_discover.pipeline.stage6_hygiene as hyg

    async def fake_fetch_crossref(client, doi):
        return {"message": {"update-to": [{"type": "retraction", "DOI": "10.x/orig"}]}}

    async def fake_fetch_unpaywall(client, doi):
        return None

    async def fake_fetch_europepmc(client, doi):
        return None

    monkeypatch.setattr(hyg, "_fetch_crossref", fake_fetch_crossref)
    monkeypatch.setattr(hyg, "_fetch_unpaywall", fake_fetch_unpaywall)
    monkeypatch.setattr(hyg, "_fetch_europepmc", fake_fetch_europepmc)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_paper(conn, "r1", "p1", "10.x/y")
        conn.close()

        out = asyncio.run(run_hygiene(path, "r1"))
        assert out["checked"] == 1
        assert out["retracted"] == 1

        conn = sqlite3.connect(path)
        retracted = conn.execute("SELECT retracted FROM papers WHERE paper_id='p1'").fetchone()[0]
        flags = conn.execute(
            "SELECT flags_json FROM candidates WHERE run_id='r1' AND paper_id='p1'"
        ).fetchone()[0]
        conn.close()
        assert retracted == 1
        assert "retracted" in json.loads(flags)
    finally:
        os.unlink(path)


def test_run_hygiene_resolves_oa_url_via_stub(monkeypatch):
    import paper_discover.pipeline.stage6_hygiene as hyg

    async def fake_fetch_crossref(client, doi):
        return {"message": {}}

    async def fake_fetch_unpaywall(client, doi):
        return {"best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}}

    async def fake_fetch_europepmc(client, doi):
        return None

    monkeypatch.setattr(hyg, "_fetch_crossref", fake_fetch_crossref)
    monkeypatch.setattr(hyg, "_fetch_unpaywall", fake_fetch_unpaywall)
    monkeypatch.setattr(hyg, "_fetch_europepmc", fake_fetch_europepmc)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_paper(conn, "r1", "p1", "10.x/y")
        conn.close()

        out = asyncio.run(run_hygiene(path, "r1"))
        assert out["oa_resolved"] == 1

        conn = sqlite3.connect(path)
        url = conn.execute("SELECT oa_url FROM papers WHERE paper_id='p1'").fetchone()[0]
        conn.close()
        assert url == "https://example.com/paper.pdf"
    finally:
        os.unlink(path)


# ── Summary rendering — FLAG F8 attribution ──────────────────────────────────

def test_summary_md_includes_retraction_attribution_when_hygiene_provided():
    out = generate_summary_md(
        stats={"CORE": 1, "SUPPORTING": 0, "CONTEXT": 0, "ADJACENT": 0, "CUT": 0},
        plan={"intent": "test"},
        run_id="r1",
        coverage=None,
        hygiene={"checked": 1, "retracted": 0, "errata": 0, "oa_resolved": 0, "errors": 0},
    )
    assert "Retraction Watch" in out
    assert "CC-BY 4.0" in out


def test_summary_md_omits_attribution_when_no_hygiene():
    out = generate_summary_md(
        stats={"CORE": 1, "SUPPORTING": 0, "CONTEXT": 0, "ADJACENT": 0, "CUT": 0},
        plan={"intent": "test"},
        run_id="r1",
        coverage=None,
        hygiene=None,
    )
    assert "Retraction Watch" not in out
