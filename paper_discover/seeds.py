"""
M6 — Seed modes.

Three independent ways to start a discovery run:

  1. Anchor mode      — DOIs, arXiv IDs, or Zotero item keys.
  2. Collection mode  — a Zotero collection key; theme inferred from contents.
  3. Structured mode  — PICO / boolean / filter dict converted directly to a
                        plan without a planner LLM call.

This module covers (a) anchor identifier resolution across the four ID
families and (b) the structured → plan conversion. Collection mode is
already handled by stage0_plan via the Zotero client.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Anchor identifier classification ─────────────────────────────────────────

_DOI_RE = re.compile(r"^(?:doi:)?10\.\d{4,9}/\S+$", re.IGNORECASE)
_ARXIV_RE = re.compile(r"^(?:arxiv:|arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$")
_ARXIV_OLD_RE = re.compile(r"^(?:arxiv:|arXiv:)?([a-z\-]+/\d{7})$", re.IGNORECASE)
_ZOTERO_KEY_RE = re.compile(r"^(?:zotero:)?([A-Z0-9]{8})$")
_S2_ID_RE = re.compile(r"^(?:s2:|S2:)?([0-9a-f]{40})$")  # S2 paper IDs are 40-hex
_OPENALEX_RE = re.compile(r"^(?:openalex:)?(W\d+)$")


def classify_anchor_id(raw: str) -> tuple[str, str]:
    """
    Identify which family an anchor identifier belongs to.

    Returns (family, normalized_id) where family is one of:
      'doi' | 'arxiv' | 'zotero' | 's2' | 'openalex' | 'unknown'
    """
    s = raw.strip()
    if not s:
        return "unknown", s

    if m := _DOI_RE.match(s):
        return "doi", s.lower().removeprefix("doi:")

    if m := _ARXIV_RE.match(s):
        return "arxiv", m.group(1)
    if m := _ARXIV_OLD_RE.match(s):
        return "arxiv", m.group(1)

    if m := _OPENALEX_RE.match(s):
        return "openalex", m.group(1)

    if m := _S2_ID_RE.match(s):
        return "s2", m.group(1)

    if m := _ZOTERO_KEY_RE.match(s):
        return "zotero", m.group(1)

    return "unknown", s


async def resolve_anchor(raw_id: str) -> dict | None:
    """
    Resolve an anchor identifier to a paper dict via whichever API knows it.
    Returns None if not resolvable.
    """
    family, norm = classify_anchor_id(raw_id)

    if family == "doi":
        from .pipeline.stage2_retrieve.openalex import fetch_paper_by_doi as oa
        from .pipeline.stage2_retrieve.semantic_scholar import fetch_paper_by_doi as s2
        return (await oa(norm)) or (await s2(norm))

    if family == "arxiv":
        from .pipeline.stage2_retrieve.semantic_scholar import fetch_paper_by_id
        return await fetch_paper_by_id(f"arXiv:{norm}")

    if family == "openalex":
        from .pipeline.stage2_retrieve.openalex import fetch_paper_by_openalex_id
        return await fetch_paper_by_openalex_id(norm)

    if family == "s2":
        from .pipeline.stage2_retrieve.semantic_scholar import fetch_paper_by_id
        return await fetch_paper_by_id(norm)

    if family == "zotero":
        from .zotero.client import get_client
        client = get_client()
        if not await client.is_available():
            logger.warning("Zotero not reachable; cannot resolve %s", raw_id)
            return None
        item = await client.get_item_by_key(norm)
        if not item:
            return None
        # Promote DOI/arXiv to a real metadata record if available
        doi = item.get("doi")
        if doi:
            from .pipeline.stage2_retrieve.openalex import fetch_paper_by_doi
            paper = await fetch_paper_by_doi(doi)
            if paper:
                return paper
        return item

    logger.warning("Unrecognised anchor id: %s", raw_id)
    return None


async def resolve_anchors(raw_ids: list[str]) -> list[dict]:
    """Resolve a list of anchor IDs, dropping those that fail."""
    out: list[dict] = []
    for rid in raw_ids:
        paper = await resolve_anchor(rid)
        if paper:
            out.append(paper)
        else:
            logger.warning("Could not resolve anchor: %s", rid)
    return out


# ── Structured → plan conversion ─────────────────────────────────────────────
#
# A structured query is one of:
#
#   PICO:
#     {"format": "pico", "population": "...", "intervention": "...",
#      "comparison": "...", "outcome": "...",
#      "scope": {...}, "anchors": [...], "depth": "deep"}
#
#   Boolean:
#     {"format": "boolean", "intent": "...", "boolean": "(A OR B) AND C NOT D",
#      "scope": {...}, "anchors": [...]}
#
#   Filter-only (no question text — keyword filter run):
#     {"format": "filter", "intent": "...", "concepts": [...],
#      "scope": {date_from, date_to, languages, ...}}
#
# All three output the same plan shape that downstream stages consume.

def plan_from_structured(query: dict) -> dict:
    """
    Build a plan dict from a structured query. Bypasses the planner LLM
    so the user can drive an exact PICO / boolean / filter run when they
    already know what they want.
    """
    fmt = (query.get("format") or "").lower()
    if fmt == "pico":
        return _plan_from_pico(query)
    if fmt == "boolean":
        return _plan_from_boolean(query)
    if fmt == "filter":
        return _plan_from_filter(query)
    raise ValueError(
        f"unsupported structured query format: {fmt!r} "
        f"(expected one of 'pico', 'boolean', 'filter')"
    )


def _plan_from_pico(q: dict) -> dict:
    """
    PICO → plan: each non-empty field becomes an essential dimension.
    Population and Intervention default to critical (most studies pivot on them).
    """
    pop = (q.get("population") or "").strip()
    intervention = (q.get("intervention") or "").strip()
    comparison = (q.get("comparison") or "").strip()
    outcome = (q.get("outcome") or "").strip()

    if not (pop and intervention and outcome):
        raise ValueError("PICO query requires population, intervention, and outcome")

    intent = q.get("intent") or _format_pico_intent(pop, intervention, comparison, outcome)

    dimensions = [
        {"name": "population",    "value": pop,
         "essential": True, "critical": True},
        {"name": "intervention",  "value": intervention,
         "essential": True, "critical": True},
        {"name": "outcome",       "value": outcome,
         "essential": True, "critical": False},
    ]
    if comparison:
        dimensions.append({
            "name": "comparison", "value": comparison,
            "essential": False, "critical": False,
        })

    concepts = [pop, intervention, outcome] + ([comparison] if comparison else [])

    return _finalise_plan(q, intent=intent, concepts=concepts, dimensions=dimensions)


def _format_pico_intent(p: str, i: str, c: str, o: str) -> str:
    base = f"In {p}, does {i}"
    if c:
        base += f" compared with {c}"
    return f"{base} affect {o}?"


def _plan_from_boolean(q: dict) -> dict:
    intent = (q.get("intent") or "").strip()
    boolean = (q.get("boolean") or "").strip()
    if not boolean:
        raise ValueError("boolean query requires a non-empty 'boolean' field")
    if not intent:
        intent = f"Find papers matching: {boolean}"

    # Extract concepts from boolean expression — each non-operator token is a concept
    concepts = _extract_boolean_concepts(boolean)

    dimensions = [
        {"name": f"concept_{i+1}", "value": c, "essential": True, "critical": False}
        for i, c in enumerate(concepts)
    ] or [{"name": "topic", "value": boolean, "essential": True, "critical": False}]

    return _finalise_plan(q, intent=intent, concepts=concepts or [boolean],
                          dimensions=dimensions, boolean=boolean)


def _extract_boolean_concepts(expr: str) -> list[str]:
    """
    Pull leaf tokens from a boolean expression — anything that isn't an
    operator or a parenthesis. Quoted phrases stay together.
    """
    # Capture quoted phrases first, then individual tokens
    tokens: list[str] = []
    for match in re.finditer(r'"([^"]+)"|([A-Za-z][\w\-]+)', expr):
        tok = match.group(1) or match.group(2)
        if tok.upper() in {"AND", "OR", "NOT"}:
            continue
        tokens.append(tok)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _plan_from_filter(q: dict) -> dict:
    intent = (q.get("intent") or "").strip()
    concepts = q.get("concepts") or []
    if not (intent or concepts):
        raise ValueError("filter query requires either 'intent' or 'concepts'")

    if not intent:
        intent = "Survey papers matching: " + ", ".join(concepts)

    dimensions = [
        {"name": f"concept_{i+1}", "value": c, "essential": True, "critical": False}
        for i, c in enumerate(concepts)
    ] or [{"name": "topic", "value": intent, "essential": True, "critical": False}]

    return _finalise_plan(q, intent=intent, concepts=concepts, dimensions=dimensions)


def _finalise_plan(q: dict, **fields: Any) -> dict:
    """Common plan-tail assembly: stitch in scope, anchors, depth."""
    plan: dict = dict(fields)
    plan["scope"] = q.get("scope") or {}
    plan["anchors"] = q.get("anchors") or []
    plan["depth"] = q.get("depth") or "deep"
    plan["_source"] = "structured"
    return plan
