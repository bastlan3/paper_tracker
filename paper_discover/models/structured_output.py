"""
JSON schema definitions for guided decoding (vLLM extra_body['guided_json']).
Each schema is a plain dict; pass it to LLMClient.judge() or .plan_json().
"""

from __future__ import annotations

# ── Plan artifact ─────────────────────────────────────────────────────────────

PLAN_SCHEMA: dict = {
    "type": "object",
    "required": ["intent", "concepts", "dimensions", "scope", "anchors", "depth"],
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string"},
        "concepts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "dimensions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name", "value", "essential", "critical"],
                "additionalProperties": False,
                "properties": {
                    "name":      {"type": "string"},
                    "value":     {"type": "string"},
                    "essential": {"type": "boolean"},
                    "critical":  {"type": "boolean"},
                },
            },
        },
        "scope": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "date_from":           {"type": "string"},
                "date_to":             {"type": "string"},
                "languages":           {"type": "array", "items": {"type": "string"}},
                "exclude_study_types": {"type": "array", "items": {"type": "string"}},
            },
        },
        "anchors": {"type": "array", "items": {"type": "string"}},
        "depth":   {"type": "string", "enum": ["fast", "standard", "deep", "unlimited"]},
    },
}

# ── Clarification questions from the planner ──────────────────────────────────

CLARIFICATIONS_SCHEMA: dict = {
    "type": "object",
    "required": ["questions"],
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["question", "default"],
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "default":  {"type": "string"},
                },
            },
        }
    },
}

# ── Query plan from Stage 1 ───────────────────────────────────────────────────

QUERY_PLAN_SCHEMA: dict = {
    "type": "object",
    "required": ["lexical_queries", "semantic_queries", "concept_translation_queries"],
    "additionalProperties": False,
    "properties": {
        "lexical_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "query", "dimensions_targeted"],
                "additionalProperties": False,
                "properties": {
                    "source":             {"type": "string"},
                    "query":              {"type": "string"},
                    "dimensions_targeted": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "semantic_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["query", "dimensions_targeted"],
                "additionalProperties": False,
                "properties": {
                    "query":              {"type": "string"},
                    "dimensions_targeted": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "concept_translation_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["field", "query", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "field":    {"type": "string"},
                    "query":    {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

# ── Main judge rubric ─────────────────────────────────────────────────────────

JUDGE_SCHEMA: dict = {
    "type": "object",
    "required": [
        "gate_A_score",
        "gate_A_evidence",
        "gate_A_per_anchor",
        "gate_B_dimension_scores",
        "gate_B_overall",
        "flags",
        "confidence",
        "two_sentence_summary",
        "why_this_level",
    ],
    "additionalProperties": False,
    "properties": {
        "gate_A_score": {"type": "integer", "minimum": 0, "maximum": 4},
        "gate_A_evidence": {"type": "string"},
        "gate_A_per_anchor": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["anchor_id", "score", "reason"],
                "additionalProperties": False,
                "properties": {
                    "anchor_id": {"type": "string"},
                    "score":     {"type": "integer", "minimum": 0, "maximum": 4},
                    "reason":    {"type": "string"},
                },
            },
        },
        "gate_B_dimension_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["dimension", "presence", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "dimension": {"type": "string"},
                    "presence":  {"type": "string", "enum": ["absent", "partial", "present"]},
                    "evidence":  {"type": "string"},
                },
            },
        },
        "gate_B_overall": {"type": "integer", "minimum": 0, "maximum": 4},
        "flags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "review", "meta_analysis", "methods", "negative_result",
                    "replication", "preregistered", "retracted",
                ],
            },
        },
        "confidence":           {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "two_sentence_summary": {"type": "string"},
        "why_this_level":       {"type": "string"},
    },
}

# ── Cross-domain analogy judge rubric (M3) ───────────────────────────────────
#
# The cross-domain judge looks at candidates retrieved by the concept-
# translation pass — papers from a distant field that may be a structural
# analogue of the user's question. Its default verdict is "superficial
# keyword overlap" → CUT. To pass ADJACENT it must name the concept
# correspondence AND identify which essential dimensions the analogue
# addresses (FLAG F14). Level mapping is in level_rule.compute_level()
# with kwargs cross_domain=True, gate_b=analogy_strength.

GAP_LIST_SCHEMA: dict = {
    "type": "object",
    "required": ["gaps"],
    "additionalProperties": False,
    "properties": {
        "gaps": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["question", "category", "motivated_by", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "question":     {"type": "string"},
                    "category":     {
                        "type": "string",
                        "enum": ["methodological", "population", "outcome",
                                 "mechanism", "replication"],
                    },
                    "motivated_by": {"type": "array", "items": {"type": "string"}},
                    "rationale":    {"type": "string"},
                },
            },
        },
    },
}

CROSS_DOMAIN_SCHEMA: dict = {
    "type": "object",
    "required": [
        "superficial_overlap_only",
        "concept_correspondence",
        "source_concept",
        "target_concept",
        "dimensions_addressed",
        "analogy_strength",
        "evidence",
        "confidence",
        "two_sentence_summary",
        "why_this_level",
    ],
    "additionalProperties": False,
    "properties": {
        # If true, the paper just shares vocabulary with the question — CUT.
        "superficial_overlap_only": {"type": "boolean"},
        # One sentence naming the structural analogy. Empty string if superficial.
        "concept_correspondence": {"type": "string"},
        # E.g. "predator-prey dynamics in ecology"
        "source_concept": {"type": "string"},
        # E.g. "tumor-immune cell dynamics in oncology"
        "target_concept": {"type": "string"},
        # Which essential dimensions of the user's question the analogue addresses.
        "dimensions_addressed": {"type": "array", "items": {"type": "string"}},
        # 0=no analogy, 4=tight isomorphism. 2+ required for ADJACENT.
        "analogy_strength": {"type": "integer", "minimum": 0, "maximum": 4},
        # Required quote from the abstract supporting the analogy claim.
        "evidence": {"type": "string"},
        "confidence":           {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "two_sentence_summary": {"type": "string"},
        "why_this_level":       {"type": "string"},
    },
}
