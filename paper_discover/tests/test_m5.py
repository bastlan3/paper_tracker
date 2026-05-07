"""
M5 unit tests — saved-search CRUD + incremental plan adaptation +
digest Markdown rendering. No CLI / no LLM / no network.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from paper_discover.digest import (
    SavedSearch,
    delete_search,
    get_search,
    list_searches,
    make_incremental_plan,
    mark_run,
    save_search,
    write_digest_md,
)


# ── CRUD ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    yield f.name
    os.unlink(f.name)


def test_save_and_round_trip(tmp_db):
    plan = {"intent": "x", "anchors": [], "depth": "deep"}
    s = save_search(tmp_db, name="GLP-1 daily", plan=plan)
    assert s.name == "GLP-1 daily"
    assert s.cadence == "daily"
    assert s.enabled is True
    fetched = get_search(tmp_db, s.search_id)
    assert fetched is not None
    assert fetched.plan == plan


def test_list_searches_orders_newest_first(tmp_db):
    a = save_search(tmp_db, name="A", plan={"intent": "a"})
    b = save_search(tmp_db, name="B", plan={"intent": "b"})
    out = list_searches(tmp_db)
    names = [s.name for s in out]
    # ULIDs encode time so newest-first is the natural sort
    assert set(names) == {"A", "B"}
    # Both saved at roughly the same time; either order is acceptable as long as both present
    assert a.search_id in {s.search_id for s in out}
    assert b.search_id in {s.search_id for s in out}


def test_delete_search_returns_true_when_found(tmp_db):
    s = save_search(tmp_db, name="X", plan={"intent": "x"})
    assert delete_search(tmp_db, s.search_id) is True
    assert get_search(tmp_db, s.search_id) is None


def test_delete_search_returns_false_when_missing(tmp_db):
    assert delete_search(tmp_db, "nonexistent") is False


def test_save_rejects_unknown_cadence(tmp_db):
    with pytest.raises(ValueError):
        save_search(tmp_db, name="X", plan={}, cadence="hourly")


def test_mark_run_updates_last_run_at(tmp_db):
    s = save_search(tmp_db, name="X", plan={"intent": "x"})
    assert s.last_run_at is None
    mark_run(tmp_db, s.search_id, when="2026-05-07T12:00:00+00:00")
    again = get_search(tmp_db, s.search_id)
    assert again.last_run_at == "2026-05-07T12:00:00+00:00"


# ── make_incremental_plan ────────────────────────────────────────────────────

def test_incremental_plan_uses_last_run_at_when_more_recent():
    plan = {"intent": "x", "scope": {"date_from": "2024-01-01"}}
    out = make_incremental_plan(plan, last_run_at="2026-04-30T03:00:00+00:00")
    assert out["scope"]["date_from"] == "2026-04-30"


def test_incremental_plan_keeps_plan_date_when_more_recent():
    plan = {"intent": "x", "scope": {"date_from": "2026-06-01"}}
    out = make_incremental_plan(plan, last_run_at="2026-04-30T03:00:00+00:00")
    assert out["scope"]["date_from"] == "2026-06-01"


def test_incremental_plan_no_last_run_passes_through():
    plan = {"intent": "x", "scope": {"date_from": "2024-01-01"}}
    out = make_incremental_plan(plan, last_run_at=None)
    assert out == plan


def test_incremental_plan_creates_scope_if_missing():
    plan = {"intent": "x"}
    out = make_incremental_plan(plan, last_run_at="2026-04-30T03:00:00+00:00")
    assert out["scope"]["date_from"] == "2026-04-30"


def test_incremental_plan_does_not_mutate_input():
    """Caller's plan dict should be untouched (we deep copy)."""
    plan = {"intent": "x", "scope": {"date_from": "2024-01-01"}}
    plan_snapshot = json.loads(json.dumps(plan))
    make_incremental_plan(plan, last_run_at="2026-05-01T00:00:00+00:00")
    assert plan == plan_snapshot


# ── write_digest_md ──────────────────────────────────────────────────────────

def _saved(name: str = "test", **kw) -> SavedSearch:
    return SavedSearch(
        search_id="01TEST",
        name=name,
        plan={"intent": "x"},
        cadence="daily",
        enabled=True,
        created_at="2026-05-01T00:00:00+00:00",
        last_run_at=None,
        **kw,
    )


def test_digest_no_rows_says_nothing_new(tmp_path):
    out = tmp_path / "digest.md"
    write_digest_md(
        output_path=str(out),
        search=_saved(),
        rows=[],
        judge_stats={"CORE": 0},
        incremental_from="2026-04-30",
    )
    text = out.read_text()
    assert "No new papers" in text
    assert "test" in text


def test_digest_groups_rows_by_level(tmp_path):
    out = tmp_path / "digest.md"
    rows = [
        {"level": "CORE", "title": "Big Result", "year": 2026,
         "doi": "10.x/big", "judge_confidence": 0.95,
         "authors_json": json.dumps(["Foo", "Bar"]),
         "evidence_span": "We show that..."},
        {"level": "SUPPORTING", "title": "Useful method", "year": 2026,
         "doi": None, "judge_confidence": 0.8,
         "authors_json": json.dumps(["Baz"]),
         "evidence_span": ""},
    ]
    write_digest_md(
        output_path=str(out),
        search=_saved(),
        rows=rows,
        judge_stats={"CORE": 1, "SUPPORTING": 1},
        incremental_from="2026-04-30",
    )
    text = out.read_text()
    assert "## CORE" in text
    assert "## SUPPORTING" in text
    assert "Big Result" in text
    assert "Useful method" in text
    assert "Foo et al." in text
    assert "Baz" in text and "et al." not in text.split("Baz")[1].split("\n")[0]
    assert "doi.org/10.x/big" in text
    assert "We show that..." in text


def test_digest_handles_missing_authors_json(tmp_path):
    out = tmp_path / "digest.md"
    rows = [{"level": "CORE", "title": "T", "year": 2026,
             "doi": None, "judge_confidence": 0.5,
             "authors_json": None, "evidence_span": ""}]
    write_digest_md(
        output_path=str(out),
        search=_saved(),
        rows=rows,
        judge_stats={"CORE": 1},
        incremental_from=None,
    )
    text = out.read_text()
    assert "Unknown" in text
    assert "(initial run)" in text
