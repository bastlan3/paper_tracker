"""
Stage 3 — Cross-domain analogy judge (M3).

Routed candidates are those retrieved by the concept-translation reframing
pass. They live in a different field than the user's question, so the main
judge's anchor-proximity rubric does not apply. This judge instead checks
whether the paper is a real *structural analogue* — same mathematical or
conceptual shape — or just a coincidental keyword match.

Default verdict: superficial overlap → CUT.
ADJACENT is granted only when ALL of these hold:
  - superficial_overlap_only is False
  - concept_correspondence is named (non-empty)
  - at least one essential dimension is mapped through the analogy
  - analogy_strength ≥ 2
  - evidence quote is non-empty
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...models.structured_output import CROSS_DOMAIN_SCHEMA
from ...models.vllm_client import get_client

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "cross_domain.txt").read_text()
_ABSTRACT_TRUNCATE = 2000


# ── Channel detection ─────────────────────────────────────────────────────────

_CROSS_DOMAIN_MARKER = "concept_translation"
_DIRECT_MARKERS = ("citation", "lexical", "semantic")


def is_cross_domain_candidate(seen_by: list[str]) -> bool:
    """
    Route to the cross-domain judge when the paper was found ONLY by the
    concept-translation pass — not via citation, lexical, or direct semantic
    channels. A paper that surfaces from both is treated as in-domain.
    """
    if not seen_by:
        return False
    has_cd = any(_CROSS_DOMAIN_MARKER in ch for ch in seen_by)
    has_direct = any(
        any(m in ch for m in _DIRECT_MARKERS) and _CROSS_DOMAIN_MARKER not in ch
        for ch in seen_by
    )
    return has_cd and not has_direct


# ── Decision logic ────────────────────────────────────────────────────────────

def decide_cross_domain(judge_out: dict, plan_dimensions: list[dict] | None = None) -> str:
    """
    Map a cross-domain judge output to a level. Only ADJACENT or CUT.
    """
    if judge_out.get("superficial_overlap_only", True):
        return "CUT"

    if not (judge_out.get("concept_correspondence") or "").strip():
        return "CUT"

    if not judge_out.get("dimensions_addressed"):
        return "CUT"

    if int(judge_out.get("analogy_strength") or 0) < 2:
        return "CUT"

    if not (judge_out.get("evidence") or "").strip():
        return "CUT"

    # Optional: only count dimensions that match essential dimension NAMES.
    if plan_dimensions:
        essential_names = {d["name"] for d in plan_dimensions if d.get("essential")}
        addressed = set(judge_out.get("dimensions_addressed", []))
        if essential_names and not (addressed & essential_names):
            return "CUT"

    return "ADJACENT"


# ── LLM call ──────────────────────────────────────────────────────────────────

async def cross_domain_judge(
    candidate: dict,
    intent: str,
    dimensions: list[dict],
    cross_domain_field: str,
    cross_domain_query: str,
    cross_domain_rationale: str,
) -> tuple[str, dict, float]:
    """
    Call the cross-domain LLM judge. Returns (level, raw_output, confidence).
    Level is computed by decide_cross_domain() — never chosen by the LLM.
    """
    abstract = (candidate.get("abstract") or "")[:_ABSTRACT_TRUNCATE]
    authors_list = json.loads(candidate.get("authors_json") or "[]")
    authors_str = ", ".join(authors_list[:5]) + (" et al." if len(authors_list) > 5 else "")

    dims_block = _format_dimensions(dimensions)
    prompt = _PROMPT.format(
        intent=intent,
        dimensions=dims_block,
        cross_domain_field=cross_domain_field or "(unknown)",
        cross_domain_query=cross_domain_query or "(unknown)",
        cross_domain_rationale=cross_domain_rationale or "(none)",
        title=candidate.get("title", ""),
        authors=authors_str,
        year=candidate.get("year") or "n.d.",
        venue=candidate.get("venue") or "unknown",
        abstract=abstract or "(no abstract)",
    )

    messages = [{"role": "user", "content": prompt}]
    llm = get_client()
    judge_out = await llm.judge(messages, CROSS_DOMAIN_SCHEMA)

    level = decide_cross_domain(judge_out, dimensions)
    confidence = float(judge_out.get("confidence") or 0.5)
    return level, judge_out, confidence


def _format_dimensions(dimensions: list[dict]) -> str:
    if not dimensions:
        return "(none)"
    return "\n".join(
        f"- {d['name']}: {d['value']}" + (" ★" if d.get("critical") else "")
        for d in dimensions
        if d.get("essential")
    )
