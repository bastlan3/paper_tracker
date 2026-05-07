"""
M3 unit tests — cross-domain analogy judge + LanceDB-backed embedding cache.

No LLM calls and no LanceDB required: the cache falls back to an in-memory
backend when LanceDB isn't installed, which is the path these tests exercise.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper_discover.models.structured_output import CROSS_DOMAIN_SCHEMA
from paper_discover.models.vector_store import EmbeddingCache
from paper_discover.pipeline.stage3_judge.cross_domain import (
    decide_cross_domain,
    is_cross_domain_candidate,
)


# ── Channel detection ────────────────────────────────────────────────────────

def test_concept_translation_only_is_cross_domain():
    assert is_cross_domain_candidate(["openalex:concept_translation"]) is True


def test_mixed_with_lexical_is_not_cross_domain():
    assert is_cross_domain_candidate(
        ["openalex:concept_translation", "semantic_scholar:lexical"]
    ) is False


def test_mixed_with_citation_is_not_cross_domain():
    assert is_cross_domain_candidate(
        ["openalex:concept_translation", "s2:citation"]
    ) is False


def test_pure_lexical_is_not_cross_domain():
    assert is_cross_domain_candidate(["openalex:lexical"]) is False


def test_empty_seen_by_is_not_cross_domain():
    assert is_cross_domain_candidate([]) is False


def test_two_concept_translation_channels_still_cross_domain():
    """A paper found via two reframings (and only those) is still cross-domain."""
    assert is_cross_domain_candidate(
        ["openalex:concept_translation", "semantic_scholar:concept_translation"]
    ) is True


# ── decide_cross_domain ──────────────────────────────────────────────────────

def _strong_cross_domain_verdict(**overrides) -> dict:
    """A baseline judge output that should map to ADJACENT."""
    out = {
        "superficial_overlap_only": False,
        "concept_correspondence": "predator-prey ↔ tumor-immune dynamics",
        "source_concept": "predator-prey",
        "target_concept": "tumor-immune",
        "dimensions_addressed": ["population_dynamics"],
        "analogy_strength": 3,
        "evidence": "We model competing populations under shared resources.",
        "confidence": 0.8,
        "two_sentence_summary": "...",
        "why_this_level": "...",
    }
    out.update(overrides)
    return out


def test_strong_analogy_maps_to_adjacent():
    verdict = _strong_cross_domain_verdict()
    assert decide_cross_domain(verdict) == "ADJACENT"


def test_superficial_overlap_is_cut():
    verdict = _strong_cross_domain_verdict(superficial_overlap_only=True)
    assert decide_cross_domain(verdict) == "CUT"


def test_missing_correspondence_is_cut():
    verdict = _strong_cross_domain_verdict(concept_correspondence="")
    assert decide_cross_domain(verdict) == "CUT"


def test_no_dimensions_addressed_is_cut():
    verdict = _strong_cross_domain_verdict(dimensions_addressed=[])
    assert decide_cross_domain(verdict) == "CUT"


def test_weak_analogy_strength_is_cut():
    verdict = _strong_cross_domain_verdict(analogy_strength=1)
    assert decide_cross_domain(verdict) == "CUT"


def test_missing_evidence_is_cut():
    verdict = _strong_cross_domain_verdict(evidence="")
    assert decide_cross_domain(verdict) == "CUT"


def test_dimensions_must_match_essentials():
    """If essential dim names are known, addressed must intersect them."""
    plan_dims = [
        {"name": "population", "value": "CKD", "essential": True, "critical": False},
        {"name": "outcome", "value": "mortality", "essential": True, "critical": False},
    ]
    verdict = _strong_cross_domain_verdict(dimensions_addressed=["unrelated_dim"])
    assert decide_cross_domain(verdict, plan_dims) == "CUT"

    verdict = _strong_cross_domain_verdict(dimensions_addressed=["population"])
    assert decide_cross_domain(verdict, plan_dims) == "ADJACENT"


def test_no_essentials_known_falls_back_permissively():
    """If we don't know essential names, decide_cross_domain doesn't filter."""
    plan_dims = [{"name": "x", "value": "y", "essential": False, "critical": False}]
    verdict = _strong_cross_domain_verdict(dimensions_addressed=["something"])
    assert decide_cross_domain(verdict, plan_dims) == "ADJACENT"


# ── Schema sanity ────────────────────────────────────────────────────────────

def test_cross_domain_schema_required_fields():
    required = set(CROSS_DOMAIN_SCHEMA["required"])
    expected = {
        "superficial_overlap_only", "concept_correspondence",
        "source_concept", "target_concept", "dimensions_addressed",
        "analogy_strength", "evidence", "confidence",
        "two_sentence_summary", "why_this_level",
    }
    assert required == expected


def test_cross_domain_schema_analogy_strength_bounded():
    s = CROSS_DOMAIN_SCHEMA["properties"]["analogy_strength"]
    assert s["minimum"] == 0 and s["maximum"] == 4


# ── Embedding cache (in-memory fallback) ─────────────────────────────────────

def test_embedding_cache_falls_back_when_no_path():
    cache = EmbeddingCache.open(path=None)
    assert cache.persistent is False


def test_embedding_cache_get_put_roundtrip():
    cache = EmbeddingCache.open(path=None)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.put("paper:abc", v)
    got = cache.get("paper:abc")
    assert got is not None
    assert np.array_equal(got, v)


def test_embedding_cache_miss_returns_none():
    cache = EmbeddingCache.open(path=None)
    assert cache.get("paper:nope") is None


def test_embedding_cache_get_many():
    cache = EmbeddingCache.open(path=None)
    cache.put("k1", np.array([1.0], dtype=np.float32))
    cache.put("k2", np.array([2.0], dtype=np.float32))
    got = cache.get_many(["k1", "k2", "k3"])
    assert set(got.keys()) == {"k1", "k2"}


def test_embedding_cache_lance_failure_falls_back(monkeypatch, tmp_path):
    """If LanceDB import fails, EmbeddingCache.open() falls back silently."""
    import paper_discover.models.vector_store as vs

    class _BoomBackend:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated lancedb failure")

    monkeypatch.setattr(vs, "_LanceBackend", _BoomBackend)
    cache = vs.EmbeddingCache.open(str(tmp_path / "emb.lance"))
    assert cache.persistent is False
    cache.put("k", np.array([1.0], dtype=np.float32))
    assert cache.get("k") is not None
