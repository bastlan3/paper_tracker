"""
Stage 1 — Query planning: turns the approved plan into concrete search queries
across lexical, semantic, and concept-translation families.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models.structured_output import QUERY_PLAN_SCHEMA
from ..models.vllm_client import get_client

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent / "prompts" / "query_planner.txt").read_text()


async def build_query_plan(plan: dict) -> dict:
    """
    Call the planner LLM to produce a structured query plan.
    Returns a dict with keys: lexical_queries, semantic_queries, concept_translation_queries.
    """
    anchors = plan.get("anchors", [])
    anchor_titles = ", ".join(anchors[:5]) if anchors else "(none)"

    messages = [
        {
            "role": "user",
            "content": _PROMPT.format(
                plan_json=json.dumps(plan, indent=2),
                anchor_titles=anchor_titles,
            ),
        }
    ]
    llm = get_client()
    return await llm.plan_json(messages, QUERY_PLAN_SCHEMA)


def flatten_queries(query_plan: dict) -> list[dict]:
    """
    Return a flat list of query dicts, each with:
      family, source, query_text, dimensions_targeted
    Ready to be handed to Stage 2 retrieval workers.
    """
    out: list[dict] = []

    for q in query_plan.get("lexical_queries", []):
        out.append({
            "family": "lexical",
            "source": q["source"],
            "query_text": q["query"],
            "dimensions_targeted": q.get("dimensions_targeted", []),
        })

    for q in query_plan.get("semantic_queries", []):
        out.append({
            "family": "semantic",
            "source": "openalex",   # semantic search via OpenAlex embedding endpoint
            "query_text": q["query"],
            "dimensions_targeted": q.get("dimensions_targeted", []),
        })
        out.append({
            "family": "semantic",
            "source": "semantic_scholar",
            "query_text": q["query"],
            "dimensions_targeted": q.get("dimensions_targeted", []),
        })

    for q in query_plan.get("concept_translation_queries", []):
        out.append({
            "family": "concept_translation",
            "source": "openalex",
            "query_text": q["query"],
            "dimensions_targeted": [],
            "cross_domain_field": q.get("field"),
            "cross_domain_rationale": q.get("rationale"),
        })

    return out
