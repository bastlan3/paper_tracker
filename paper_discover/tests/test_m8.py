"""
M8 unit tests — FastAPI app endpoints.

Skipped automatically if FastAPI / httpx aren't installed (the [web]
extra is optional).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from paper_discover.candidates.db import open_db
from paper_discover.web.app import make_app


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def seeded_db(tmp_db):
    """A DB with one run + one CORE paper + a citation edge."""
    conn = open_db(tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (run_id, started_at, mode, plan_json, status, coverage_p, coverage_ci_lo, coverage_ci_hi) VALUES (?,?,?,?,?,?,?,?)",
        ("R1", now, "deep", '{"intent":"x"}', "done", 0.85, 0.7, 0.9),
    )
    conn.execute(
        "INSERT INTO papers (paper_id, title, title_norm, year, doi) VALUES (?,?,?,?,?)",
        ("P1", "Anchor", "anchor", 2020, "10.x/y"),
    )
    conn.execute(
        """INSERT INTO candidates
             (run_id, paper_id, first_seen_at, last_seen_at, seen_count,
              seen_by_json, level, judge_status, judge_tier, judge_confidence)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("R1", "P1", now, now, 1, '["s"]', "CORE", "judged", "T4", 0.9),
    )
    conn.commit()
    conn.close()
    return tmp_db


@pytest.fixture
def client(seeded_db):
    app = make_app(db_path=seeded_db)
    return TestClient(app)


# ── Endpoint tests ───────────────────────────────────────────────────────────

def test_dashboard_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "paper-discover" in r.text
    assert "R1"[:12] in r.text  # run id appears


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_list_runs(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["run_id"] == "R1"


def test_get_run_includes_coverage_signals(client, seeded_db):
    # Add a coverage_signals row
    conn = sqlite3.connect(seeded_db)
    conn.execute(
        """INSERT INTO coverage_signals (
             run_id, saturation_signal, skeptic_signal, channel_jaccard,
             anchor_accuracy, coverage_p, coverage_ci_lo, coverage_ci_hi,
             methodology_json, computed_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        ("R1", 0.9, 0.95, 0.4, 0.8, 0.85, 0.7, 0.9, "{}",
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    r = client.get("/api/runs/R1")
    assert r.status_code == 200
    sig = r.json().get("coverage_signals")
    assert sig is not None
    assert sig["saturation_signal"] == 0.9


def test_get_run_404_for_unknown(client):
    assert client.get("/api/runs/nonexistent").status_code == 404


def test_get_papers_returns_kept(client):
    r = client.get("/api/runs/R1/papers")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "R1"
    assert len(body["papers"]) == 1
    assert body["papers"][0]["paper_id"] == "P1"


def test_get_papers_level_filter(client):
    r = client.get("/api/runs/R1/papers", params={"level": "ADJACENT"})
    assert r.status_code == 200
    assert r.json()["papers"] == []


def test_concept_map_endpoint(client):
    r = client.get("/api/runs/R1/concept_map")
    assert r.status_code == 200
    body = r.json()
    assert {n["id"] for n in body["nodes"]} == {"P1"}
    assert body["edges"] == []


def test_prisma_endpoint(client):
    r = client.get("/api/runs/R1/prisma")
    assert r.status_code == 200
    body = r.json()
    assert body["kept_total"] == 1


# ── Saved-search endpoints ───────────────────────────────────────────────────

def test_saved_searches_crud(client):
    # Empty initially
    assert client.get("/api/saved_searches").json() == {"searches": []}

    # Create
    r = client.post(
        "/api/saved_searches",
        json={"name": "GLP-1 daily", "plan": {"intent": "x"}, "cadence": "daily"},
    )
    assert r.status_code == 201
    sid = r.json()["search_id"]

    # List
    listed = client.get("/api/saved_searches").json()
    assert len(listed["searches"]) == 1
    assert listed["searches"][0]["name"] == "GLP-1 daily"

    # Delete
    d = client.delete(f"/api/saved_searches/{sid}")
    assert d.status_code == 200
    assert d.json()["deleted"] == sid
    assert client.get("/api/saved_searches").json() == {"searches": []}


def test_saved_search_invalid_cadence_returns_400(client):
    r = client.post(
        "/api/saved_searches",
        json={"name": "X", "plan": {}, "cadence": "hourly"},
    )
    assert r.status_code == 400


def test_delete_unknown_saved_search_returns_404(client):
    assert client.delete("/api/saved_searches/nonexistent").status_code == 404
