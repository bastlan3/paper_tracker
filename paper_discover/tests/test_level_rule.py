"""
Tests for the deterministic level mapping in stage3_judge/level_rule.py.
These are the most critical unit tests — the level rule must be provably correct.
"""

import pytest

from paper_discover.pipeline.stage3_judge.level_rule import (
    compute_level,
    gate_b_from_dimensions,
)

# ── gate_b_from_dimensions ────────────────────────────────────────────────────

def _dims(*presences: str) -> list[dict]:
    return [{"dimension": f"d{i}", "presence": p, "evidence": "x"} for i, p in enumerate(presences)]


def test_gate_b_all_present():
    assert gate_b_from_dimensions(_dims("present", "present", "present")) == 4


def test_gate_b_majority_present_none_absent():
    assert gate_b_from_dimensions(_dims("present", "present", "partial")) == 3


def test_gate_b_majority_partial():
    assert gate_b_from_dimensions(_dims("present", "partial", "partial", "absent")) == 2


def test_gate_b_majority_absent():
    assert gate_b_from_dimensions(_dims("absent", "absent", "partial")) == 1


def test_gate_b_empty():
    assert gate_b_from_dimensions([]) == 0


# ── compute_level ─────────────────────────────────────────────────────────────

def _score(a: int, b: int, flags: list | None = None, dims: list | None = None) -> str:
    bdims = dims or _dims(*["present"] * b if b >= 3 else ["partial"] * max(b, 1))
    return compute_level(
        gate_a_score=a,
        gate_b_overall=b,
        gate_b_dimension_scores=bdims,
        flags=flags or [],
    )


class TestCore:
    def test_a4_is_core(self):
        assert _score(4, 0) == "CORE"

    def test_b4_all_present_is_core(self):
        dims = _dims("present", "present", "present")
        assert compute_level(0, 4, dims, []) == "CORE"

    def test_a3_b3_is_core(self):
        assert _score(3, 3) == "CORE"

    def test_a3_b2_not_core(self):
        assert _score(3, 2) != "CORE"


class TestSupporting:
    def test_a3_alone_is_supporting(self):
        dims = _dims("absent", "absent")
        assert compute_level(3, 0, dims, []) == "SUPPORTING"

    def test_b3_alone_is_supporting(self):
        dims = _dims("present", "present", "partial")
        assert compute_level(1, 3, dims, []) == "SUPPORTING"

    def test_methods_paper_a2_is_supporting(self):
        dims = _dims("partial", "partial")
        result = compute_level(2, 1, dims, ["methods"])
        assert result == "SUPPORTING"


class TestContext:
    def test_review_b2_is_context(self):
        dims = _dims("present", "partial")
        result = compute_level(0, 2, dims, ["review"])
        assert result == "CONTEXT"

    def test_meta_analysis_b2_is_context(self):
        dims = _dims("present", "partial")
        result = compute_level(0, 2, dims, ["meta_analysis"])
        assert result == "CONTEXT"


class TestAdjacent:
    def test_cross_domain_b1_is_adjacent(self):
        dims = _dims("partial")
        result = compute_level(0, 1, dims, [], is_cross_domain=True)
        assert result == "ADJACENT"

    def test_cross_domain_a0_b0_is_cut(self):
        dims = _dims("absent")
        result = compute_level(0, 0, dims, [], is_cross_domain=True)
        assert result == "CUT"


class TestCut:
    def test_all_zero_is_cut(self):
        assert _score(0, 0, [], _dims("absent", "absent")) == "CUT"

    def test_a1_b1_is_cut(self):
        dims = _dims("partial")
        result = compute_level(1, 1, dims, [])
        assert result == "CUT"


class TestCriticalDimension:
    def test_critical_absent_caps_gate_b(self):
        """A critical dimension absent must prevent CORE even with high b score."""
        plan_dims = [
            {"name": "d0", "value": "x", "essential": True, "critical": True},
            {"name": "d1", "value": "y", "essential": True, "critical": False},
        ]
        score_dims = [
            {"dimension": "d0", "presence": "absent", "evidence": ""},
            {"dimension": "d1", "presence": "present", "evidence": "found"},
        ]
        result = compute_level(
            gate_a_score=3,
            gate_b_overall=3,
            gate_b_dimension_scores=score_dims,
            flags=[],
            plan_dimensions=plan_dims,
        )
        # gate_b gets capped to 1 because d0 is critical and absent → A=3,B=1 → SUPPORTING
        assert result == "SUPPORTING"
