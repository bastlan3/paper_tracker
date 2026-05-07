"""
M2 unit tests — saturation, skeptic, coverage.

These tests do NOT call any LLM or external API. They exercise the pure
helpers (Wilson CI, channel Jaccard, stratified sampling, level mapping)
and the stage-7 weighted aggregation logic.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from paper_discover.candidates.db import open_db
from paper_discover.pipeline.stage5_skeptic import _stratified_sample
from paper_discover.pipeline.stage7_coverage import (
    _compute_channel_jaccard,
    _wilson_ci,
    _WEIGHTS,
)


# ── Migration application ────────────────────────────────────────────────────

def test_m2_migration_creates_saturation_log():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("saturation_log",),
        ).fetchall()
        assert rows, "saturation_log table missing after open_db()"
        conn.close()
    finally:
        os.unlink(path)


def test_m2_migration_creates_coverage_signals():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("coverage_signals",),
        ).fetchall()
        assert rows
        conn.close()
    finally:
        os.unlink(path)


def test_m2_migration_idempotent():
    """Re-opening the DB applies the migration again without error."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        for _ in range(3):
            conn = open_db(path)
            conn.close()
    finally:
        os.unlink(path)


# ── Wilson confidence interval ───────────────────────────────────────────────

def test_wilson_ci_zero_n():
    lo, hi = _wilson_ci(0.5, 0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_ci_extremes():
    lo, hi = _wilson_ci(0.0, 100)
    assert lo == 0.0
    assert hi < 0.1
    lo, hi = _wilson_ci(1.0, 100)
    assert hi == pytest.approx(1.0)
    assert lo > 0.9


def test_wilson_ci_centred():
    lo, hi = _wilson_ci(0.5, 100)
    # Symmetric-ish around 0.5; tight enough at n=100
    assert 0.39 < lo < 0.42
    assert 0.58 < hi < 0.61


def test_wilson_ci_widens_with_small_n():
    lo_big, hi_big = _wilson_ci(0.5, 1000)
    lo_small, hi_small = _wilson_ci(0.5, 10)
    assert (hi_small - lo_small) > (hi_big - lo_big)


# ── Coverage weights consistency ─────────────────────────────────────────────

def test_coverage_weights_sum_to_one():
    assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9


def test_coverage_weights_signals_complete():
    assert set(_WEIGHTS) == {"S1", "S2", "S3", "S4"}


# ── Channel Jaccard ──────────────────────────────────────────────────────────

def _seed_candidates(conn: sqlite3.Connection, run_id: str, items: list[tuple[str, list[str]]]):
    """items = [(paper_id, [channel,...]), ...]"""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO runs (run_id, started_at, mode, plan_json, status) VALUES (?,?,?,?,?)",
        (run_id, now, "deep", "{}", "running"),
    )
    for pid, channels in items:
        conn.execute(
            "INSERT OR IGNORE INTO papers (paper_id, title, title_norm) VALUES (?,?,?)",
            (pid, pid, pid),
        )
        conn.execute(
            """INSERT INTO candidates
                 (run_id, paper_id, first_seen_at, last_seen_at,
                  seen_count, seen_by_json, level, judge_status, judge_confidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (run_id, pid, now, now, len(channels), json.dumps(channels),
             "CORE", "judged", 0.9),
        )
    conn.commit()


def test_channel_jaccard_too_few_channels():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        # Only one channel meets the ≥3-paper threshold
        _seed_candidates(conn, "r1", [
            ("p1", ["openalex"]),
            ("p2", ["openalex"]),
            ("p3", ["openalex"]),
            ("p4", ["s2"]),  # only 1 paper for s2
        ])
        conn.close()
        assert _compute_channel_jaccard(path, "r1") is None
    finally:
        os.unlink(path)


def test_channel_jaccard_full_overlap_is_one():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_candidates(conn, "r1", [
            ("p1", ["openalex", "s2"]),
            ("p2", ["openalex", "s2"]),
            ("p3", ["openalex", "s2"]),
        ])
        conn.close()
        j = _compute_channel_jaccard(path, "r1")
        assert j is not None
        assert abs(j - 1.0) < 1e-9
    finally:
        os.unlink(path)


def test_channel_jaccard_disjoint_is_zero():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        _seed_candidates(conn, "r1", [
            ("p1", ["openalex"]),
            ("p2", ["openalex"]),
            ("p3", ["openalex"]),
            ("p4", ["s2"]),
            ("p5", ["s2"]),
            ("p6", ["s2"]),
        ])
        conn.close()
        j = _compute_channel_jaccard(path, "r1")
        assert j == 0.0
    finally:
        os.unlink(path)


def test_channel_jaccard_partial_overlap():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        conn = open_db(path)
        # openalex sees p1,p2,p3,p4; s2 sees p3,p4,p5,p6 → |∩|=2, |∪|=6 → 2/6
        _seed_candidates(conn, "r1", [
            ("p1", ["openalex"]),
            ("p2", ["openalex"]),
            ("p3", ["openalex", "s2"]),
            ("p4", ["openalex", "s2"]),
            ("p5", ["s2"]),
            ("p6", ["s2"]),
        ])
        conn.close()
        j = _compute_channel_jaccard(path, "r1")
        assert j is not None
        assert abs(j - (2/6)) < 1e-6
    finally:
        os.unlink(path)


# ── Stratified sampling ──────────────────────────────────────────────────────

def _mk_cut_paper(pid: str, score: float) -> dict:
    return {
        "paper_id": pid,
        "signals_json": json.dumps({"reranker_score": score}),
        "title": pid, "abstract": "abs", "year": 2020,
    }


def test_stratified_sample_empty():
    assert _stratified_sample([]) == []


def test_stratified_sample_size_clamped_to_min():
    # Few papers → all of them returned (n cannot exceed input size)
    papers = [_mk_cut_paper(f"p{i}", 0.5) for i in range(5)]
    out = _stratified_sample(papers)
    assert len(out) == 5


def test_stratified_sample_includes_borderline_and_obvious():
    # 100 papers with monotonic scores; check we keep both ends
    papers = [_mk_cut_paper(f"p{i}", 1.0 - i / 100) for i in range(100)]
    out = _stratified_sample(papers)
    out_ids = {p["paper_id"] for p in out}
    # Highest scores are "borderline" (closest to keeping); lowest are obvious cuts
    assert any(p["paper_id"] == "p0" for p in out), "missing borderline (highest score)"
    assert any(p["paper_id"] == "p99" for p in out), "missing obvious (lowest score)"


def test_stratified_sample_respects_min_floor():
    # 30 papers × 0.15 = 4.5 → floor below _SAMPLE_MIN(=20) → forced to 20
    papers = [_mk_cut_paper(f"p{i}", 0.5) for i in range(30)]
    out = _stratified_sample(papers)
    assert len(out) == 20


def test_stratified_sample_respects_max_ceiling():
    # 2000 × 0.15 = 300 → above _SAMPLE_MAX(=150) → clamped to 150
    papers = [_mk_cut_paper(f"p{i}", 0.5) for i in range(2000)]
    out = _stratified_sample(papers)
    assert len(out) == 150
