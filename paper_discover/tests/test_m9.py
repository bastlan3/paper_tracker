"""
M9 unit tests — MCP tool handlers (the runtime wrapper is exercised
manually since the `mcp` package is an optional dep).
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from paper_discover.candidates.db import open_db
from paper_discover.mcp_server import (
    TOOLS,
    call_tool,
    tool_get_bibliography,
    tool_get_concept_map,
    tool_get_prisma,
    tool_get_run,
    tool_list_runs,
    tool_list_saved_searches,
    tool_plan_from_pico,
    tool_save_plan,
)


@pytest.fixture
def tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def seeded(tmp_db):
    conn = open_db(tmp_db)
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO runs (run_id, started_at, mode, plan_json, status, coverage_p) VALUES (?,?,?,?,?,?)",
        ("R1", now, "deep", '{"intent":"x"}', "done", 0.85),
    )
    conn.execute(
        "INSERT INTO papers (paper_id, title, title_norm, year) VALUES (?,?,?,?)",
        ("P1", "Anchor", "anchor", 2020),
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


# ── Registry sanity ──────────────────────────────────────────────────────────

def test_registry_has_expected_tools():
    expected = {
        "list_runs", "get_run", "get_bibliography", "get_concept_map",
        "get_prisma", "list_saved_searches", "save_plan", "plan_from_pico",
    }
    assert set(TOOLS) == expected


def test_every_tool_has_handler_description_and_schema():
    for name, meta in TOOLS.items():
        assert callable(meta["handler"]), name
        assert isinstance(meta["description"], str) and meta["description"]
        assert meta["schema"]["type"] == "object"
        assert "required" in meta["schema"]


# ── Handler unit tests ───────────────────────────────────────────────────────

def test_tool_list_runs(seeded):
    out = tool_list_runs(seeded)
    assert len(out["runs"]) == 1
    assert out["runs"][0]["run_id"] == "R1"


def test_tool_get_run_returns_run(seeded):
    out = tool_get_run(seeded, "R1")
    assert out["run_id"] == "R1"
    assert out["coverage_signals"] is None


def test_tool_get_run_404(seeded):
    out = tool_get_run(seeded, "nope")
    assert "error" in out


def test_tool_get_bibliography(seeded):
    out = tool_get_bibliography(seeded, "R1")
    assert out["run_id"] == "R1"
    assert len(out["papers"]) == 1
    assert out["papers"][0]["paper_id"] == "P1"


def test_tool_get_bibliography_filtered(seeded):
    out = tool_get_bibliography(seeded, "R1", level="ADJACENT")
    assert out["papers"] == []


def test_tool_get_bibliography_invalid_level(seeded):
    out = tool_get_bibliography(seeded, "R1", level="MAGIC")
    assert "error" in out


def test_tool_get_concept_map(seeded):
    out = tool_get_concept_map(seeded, "R1")
    assert {n["id"] for n in out["nodes"]} == {"P1"}


def test_tool_get_prisma(seeded):
    out = tool_get_prisma(seeded, "R1")
    assert out["kept_total"] == 1


def test_tool_save_plan_round_trip(seeded):
    saved = tool_save_plan(seeded, name="S", plan={"intent": "x"}, cadence="daily")
    assert "search_id" in saved
    listed = tool_list_saved_searches(seeded)
    assert any(s["search_id"] == saved["search_id"] for s in listed["searches"])


def test_tool_save_plan_invalid_cadence(seeded):
    out = tool_save_plan(seeded, name="S", plan={}, cadence="hourly")
    assert "error" in out


def test_tool_plan_from_pico():
    out = tool_plan_from_pico({
        "format": "pico",
        "population": "CKD",
        "intervention": "GLP-1",
        "outcome": "CV mortality",
    })
    assert out["intent"].startswith("In CKD")
    assert any(d["name"] == "intervention" for d in out["dimensions"])


def test_tool_plan_from_pico_invalid():
    out = tool_plan_from_pico({"format": "pico", "population": "X"})
    assert "error" in out


# ── call_tool dispatch ───────────────────────────────────────────────────────

def test_call_tool_dispatches_correctly(seeded):
    out = call_tool("list_runs", {"db_path": seeded})
    assert "runs" in out


def test_call_tool_unknown():
    out = call_tool("magic_tool", {})
    assert "error" in out and "unknown tool" in out["error"]


def test_call_tool_bad_arguments(seeded):
    out = call_tool("get_run", {"db_path": seeded})  # missing run_id
    assert "error" in out
    assert "bad arguments" in out["error"]
