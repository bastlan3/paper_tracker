"""
M6 unit tests — seed modes (anchor classification, structured-query → plan).

No network: classify_anchor_id is pure; resolver tests use the path that
falls through to "unknown" without making API calls.
"""

from __future__ import annotations

import json

import pytest

from paper_discover.seeds import (
    classify_anchor_id,
    plan_from_structured,
    _extract_boolean_concepts,
)


# ── classify_anchor_id ───────────────────────────────────────────────────────

def test_classify_doi_with_prefix():
    fam, norm = classify_anchor_id("doi:10.1056/NEJMoa2024816")
    assert fam == "doi"
    assert norm == "10.1056/nejmoa2024816"


def test_classify_doi_without_prefix():
    fam, norm = classify_anchor_id("10.1038/s41586-021-03828-1")
    assert fam == "doi"


def test_classify_arxiv_new_format():
    fam, norm = classify_anchor_id("2401.12345")
    assert fam == "arxiv"
    assert norm == "2401.12345"


def test_classify_arxiv_with_prefix():
    fam, norm = classify_anchor_id("arXiv:2401.12345v2")
    assert fam == "arxiv"
    assert norm == "2401.12345v2"


def test_classify_arxiv_old_format():
    fam, norm = classify_anchor_id("hep-th/0501001")
    assert fam == "arxiv"
    assert norm == "hep-th/0501001"


def test_classify_openalex():
    fam, norm = classify_anchor_id("W2741809807")
    assert fam == "openalex"
    assert norm == "W2741809807"


def test_classify_zotero_key():
    fam, norm = classify_anchor_id("ABC23XYZ")
    assert fam == "zotero"
    assert norm == "ABC23XYZ"


def test_classify_zotero_with_prefix():
    fam, norm = classify_anchor_id("zotero:ABC23XYZ")
    assert fam == "zotero"


def test_classify_unknown_garbage():
    fam, _ = classify_anchor_id("just some random string")
    assert fam == "unknown"


def test_classify_empty():
    fam, _ = classify_anchor_id("   ")
    assert fam == "unknown"


# ── plan_from_structured: PICO ───────────────────────────────────────────────

def test_pico_full_query_builds_plan():
    q = {
        "format": "pico",
        "population": "adults with CKD",
        "intervention": "GLP-1 receptor agonists",
        "comparison": "placebo",
        "outcome": "cardiovascular mortality",
        "depth": "deep",
    }
    plan = plan_from_structured(q)
    assert plan["intent"].startswith("In adults with CKD")
    assert any(d["name"] == "population" and d["critical"] for d in plan["dimensions"])
    assert any(d["name"] == "intervention" and d["critical"] for d in plan["dimensions"])
    assert any(d["name"] == "outcome" for d in plan["dimensions"])
    assert any(d["name"] == "comparison" for d in plan["dimensions"])
    assert plan["depth"] == "deep"
    assert plan["_source"] == "structured"


def test_pico_without_comparison_omits_dimension():
    q = {
        "format": "pico",
        "population": "X", "intervention": "Y", "outcome": "Z",
    }
    plan = plan_from_structured(q)
    names = {d["name"] for d in plan["dimensions"]}
    assert "comparison" not in names


def test_pico_missing_required_field_raises():
    with pytest.raises(ValueError):
        plan_from_structured({"format": "pico", "population": "X", "intervention": "Y"})


def test_pico_uses_explicit_intent_when_provided():
    q = {
        "format": "pico",
        "population": "X", "intervention": "Y", "outcome": "Z",
        "intent": "Custom phrasing of the question.",
    }
    plan = plan_from_structured(q)
    assert plan["intent"] == "Custom phrasing of the question."


# ── plan_from_structured: boolean ────────────────────────────────────────────

def test_boolean_query_builds_plan():
    q = {
        "format": "boolean",
        "intent": "long-COVID neurological symptoms",
        "boolean": '"long covid" AND (neurological OR cognitive) NOT pediatric',
    }
    plan = plan_from_structured(q)
    assert plan["intent"] == "long-COVID neurological symptoms"
    concepts_lower = [c.lower() for c in plan["concepts"]]
    assert "long covid" in concepts_lower
    assert "neurological" in concepts_lower
    assert "cognitive" in concepts_lower
    assert "pediatric" in concepts_lower
    # Operators must NOT appear as concepts
    assert "and" not in concepts_lower
    assert "or" not in concepts_lower
    assert "not" not in concepts_lower


def test_boolean_without_intent_uses_expression():
    q = {"format": "boolean", "boolean": "diabetes AND mortality"}
    plan = plan_from_structured(q)
    assert "diabetes" in plan["intent"].lower() or "Find papers" in plan["intent"]


def test_boolean_empty_expression_raises():
    with pytest.raises(ValueError):
        plan_from_structured({"format": "boolean", "boolean": ""})


def test_extract_boolean_concepts_dedupes():
    out = _extract_boolean_concepts("(diabetes OR diabetes) AND mortality")
    lower = [t.lower() for t in out]
    # "diabetes" appears once
    assert lower.count("diabetes") == 1
    assert "mortality" in lower


def test_extract_boolean_concepts_keeps_quoted_phrases():
    out = _extract_boolean_concepts('"long covid" AND fatigue')
    assert "long covid" in out
    assert "fatigue" in out


# ── plan_from_structured: filter ─────────────────────────────────────────────

def test_filter_query_with_concepts():
    q = {
        "format": "filter",
        "concepts": ["diabetes", "metformin"],
        "scope": {"date_from": "2020-01-01", "languages": ["en"]},
    }
    plan = plan_from_structured(q)
    assert plan["scope"]["date_from"] == "2020-01-01"
    assert len(plan["dimensions"]) == 2
    assert {d["value"] for d in plan["dimensions"]} == {"diabetes", "metformin"}


def test_filter_query_with_intent_only():
    q = {"format": "filter", "intent": "Survey RCTs in oncology"}
    plan = plan_from_structured(q)
    assert plan["intent"] == "Survey RCTs in oncology"


def test_filter_missing_intent_and_concepts_raises():
    with pytest.raises(ValueError):
        plan_from_structured({"format": "filter"})


# ── format dispatch ──────────────────────────────────────────────────────────

def test_unknown_format_raises():
    with pytest.raises(ValueError):
        plan_from_structured({"format": "magic", "intent": "x"})


def test_missing_format_raises():
    with pytest.raises(ValueError):
        plan_from_structured({"intent": "x"})


# ── Plan shape consistency across modes ──────────────────────────────────────

def test_plan_shape_has_required_keys_in_all_modes():
    pico = plan_from_structured({
        "format": "pico", "population": "p", "intervention": "i", "outcome": "o",
    })
    boolean = plan_from_structured({
        "format": "boolean", "boolean": "x AND y", "intent": "t",
    })
    filter_q = plan_from_structured({
        "format": "filter", "concepts": ["a"],
    })
    for plan in (pico, boolean, filter_q):
        for k in ("intent", "concepts", "dimensions", "scope", "anchors", "depth"):
            assert k in plan, f"missing {k!r} in plan"


def test_anchors_passed_through():
    plan = plan_from_structured({
        "format": "pico", "population": "p", "intervention": "i", "outcome": "o",
        "anchors": ["10.x/y", "10.x/z"],
    })
    assert plan["anchors"] == ["10.x/y", "10.x/z"]
