"""
Deterministic level mapping from LLM gate scores + flags.

The level is NEVER chosen by the LLM directly — the LLM supplies gate scores
and the code here decides the level. This prevents back-rationalisation (FLAG F5).

Rules (from the design doc §4.4.3):

  A=4                              → CORE   (decisive anchor match alone)
  B=4, all essentials present      → CORE   (decisive question coverage alone)
  A≥3 AND B≥3                     → CORE
  A=3 alone (B<3)                  → SUPPORTING
  B≥3 without strong anchor match  → SUPPORTING
  methods paper AND A≥2            → SUPPORTING
  foundational/review AND B≥2      → CONTEXT
  cross-domain AND (A≥1 OR B≥1)   → ADJACENT  [set by cross-domain judge]
  otherwise                        → CUT

Soft-forced-disagreement rule (FLAG F5 mitigation):
  Uniform-high scores are allowed when they're genuinely justified.
  The forced-disagreement logic is in the judge PROMPT, not here.
  The code only applies the threshold rules — it does not penalise uniform scores.
"""

from __future__ import annotations


def compute_level(
    gate_a_score: int,
    gate_b_overall: int,
    gate_b_dimension_scores: list[dict],
    flags: list[str],
    is_cross_domain: bool = False,
    plan_dimensions: list[dict] | None = None,
) -> str:
    """
    Return one of: CORE, SUPPORTING, CONTEXT, ADJACENT, CUT.

    Args:
        gate_a_score:             max over anchors (0–4)
        gate_b_overall:           LLM-assessed overall B score (0–4)
        gate_b_dimension_scores:  [{dimension, presence, evidence}, …]
        flags:                    list of flag strings from the judge output
        is_cross_domain:          True only when routed through the cross-domain judge
        plan_dimensions:          original plan dimensions (to check criticality)
    """
    is_methods = "methods" in flags
    is_foundational = "review" in flags or "meta_analysis" in flags

    # Check if any critical dimension is absent — caps Gate B
    effective_gate_b = gate_b_overall
    if plan_dimensions:
        critical_absent = _any_critical_absent(gate_b_dimension_scores, plan_dimensions)
        if critical_absent:
            effective_gate_b = min(effective_gate_b, 1)

    # ── CORE ─────────────────────────────────────────────────────────────────
    if gate_a_score == 4:
        return "CORE"

    all_essentials_present = _all_essentials_present(gate_b_dimension_scores, plan_dimensions)
    if effective_gate_b == 4 and all_essentials_present:
        return "CORE"

    if gate_a_score >= 3 and effective_gate_b >= 3:
        return "CORE"

    # ── SUPPORTING ────────────────────────────────────────────────────────────
    if gate_a_score == 3:                      # decisive anchor, weak question coverage
        return "SUPPORTING"

    if effective_gate_b >= 3:                  # good question coverage, weak anchor match
        return "SUPPORTING"

    if is_methods and gate_a_score >= 2:       # methods paper closely related to anchor
        return "SUPPORTING"

    # ── CONTEXT ───────────────────────────────────────────────────────────────
    if is_foundational and effective_gate_b >= 2:
        return "CONTEXT"

    # ── ADJACENT ─────────────────────────────────────────────────────────────
    if is_cross_domain and (gate_a_score >= 1 or effective_gate_b >= 1):
        return "ADJACENT"

    # ── CUT ───────────────────────────────────────────────────────────────────
    return "CUT"


def gate_b_from_dimensions(
    dimension_scores: list[dict],
    plan_dimensions: list[dict] | None = None,
) -> int:
    """
    Compute Gate B score from per-dimension presence scores (Soft AND rule).

    Soft AND:
      All essentials present      → 4
      Majority present, none absent → 3
      Majority at least partial, ≤1 absent → 2
      Otherwise → ≤1
    """
    if not dimension_scores:
        return 0

    essential_scores = dimension_scores
    if plan_dimensions:
        essential_names = {d["name"] for d in plan_dimensions if d.get("essential")}
        if essential_names:
            essential_scores = [
                s for s in dimension_scores if s["dimension"] in essential_names
            ]
    if not essential_scores:
        essential_scores = dimension_scores

    presences = [s["presence"] for s in essential_scores]
    n = len(presences)
    n_present = presences.count("present")
    n_partial = presences.count("partial")
    n_absent = presences.count("absent")

    if n_present == n:
        return 4
    if n_present > n / 2 and n_absent == 0:
        return 3
    if (n_present + n_partial) > n / 2 and n_absent <= 1:
        return 2
    return 1


def _all_essentials_present(
    dimension_scores: list[dict],
    plan_dimensions: list[dict] | None,
) -> bool:
    if not plan_dimensions:
        return all(s["presence"] == "present" for s in dimension_scores)
    essential_names = {d["name"] for d in plan_dimensions if d.get("essential")}
    for s in dimension_scores:
        if s["dimension"] in essential_names and s["presence"] != "present":
            return False
    return True


def _any_critical_absent(
    dimension_scores: list[dict],
    plan_dimensions: list[dict],
) -> bool:
    critical_names = {d["name"] for d in plan_dimensions if d.get("critical")}
    for s in dimension_scores:
        if s["dimension"] in critical_names and s["presence"] == "absent":
            return True
    return False
