"""
T1 — Hard-rule triage: fast, zero-LLM cuts.
Returns (should_cut: bool, reason: str).

Catches obvious negatives before the reranker and LLM are invoked,
reducing downstream cost by ~10×.
"""

from __future__ import annotations

import re
from datetime import datetime


# Study types that are always excluded unless the plan overrides
_DEFAULT_EXCLUDED_TYPES = {"editorial", "news", "comment", "letter", "errata"}
_ABSTRACT_MIN_CHARS = 50


def triage(paper: dict, plan: dict) -> tuple[bool, str]:
    """
    Returns:
        (True, reason)  — cut this paper
        (False, "")     — passes triage; send to T2 reranker
    """
    scope = plan.get("scope", {})

    # No title
    if not (paper.get("title") or "").strip():
        return True, "no_title"

    # No abstract or too short
    abstract = (paper.get("abstract") or "").strip()
    if len(abstract) < _ABSTRACT_MIN_CHARS:
        return True, "no_abstract"

    # Year out of scope
    date_from = scope.get("date_from")
    year = paper.get("year")
    if date_from and year:
        try:
            cutoff_year = int(str(date_from)[:4])
            if int(year) < cutoff_year:
                return True, f"before_date_from:{cutoff_year}"
        except (ValueError, TypeError):
            pass

    date_to = scope.get("date_to")
    if date_to and year:
        try:
            cutoff_year = int(str(date_to)[:4])
            if int(year) > cutoff_year:
                return True, f"after_date_to:{cutoff_year}"
        except (ValueError, TypeError):
            pass

    # Study type exclusions
    excluded_types = set(scope.get("exclude_study_types", [])) | _DEFAULT_EXCLUDED_TYPES
    item_type = (paper.get("item_type") or paper.get("pub_type") or "").lower()
    for excl in excluded_types:
        if excl.lower() in item_type:
            return True, f"excluded_type:{excl}"

    # Language check (heuristic: flag if title has >30% non-ASCII chars)
    allowed_langs = scope.get("languages", ["en"])
    if "en" in allowed_langs and paper.get("title"):
        title = paper["title"]
        non_ascii = sum(1 for c in title if ord(c) > 127)
        if non_ascii / max(len(title), 1) > 0.35:
            return True, "likely_non_english"

    # Retracted (hard exclude from judging; kept in DB with flag)
    # We do NOT cut retracted papers silently — we flag them for special handling.
    # Only skip here if the paper itself is superseded with no scholarly value.

    return False, ""
