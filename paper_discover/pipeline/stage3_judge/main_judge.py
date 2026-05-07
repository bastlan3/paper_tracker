"""
Stage 3 — Main judging pipeline.

For each pending candidate:
  T1: triage (hard rules, no LLM)
  T2: reranker (cross-encoder, no LLM)
  T4: full LLM judge with Gate A/B rubric

Level assignment is deterministic (level_rule.py), not LLM-chosen.
All decisions written back to candidates table with full evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...candidates.db import fetch_pending_candidates, read_conn
from ...candidates.queries import get_concept_translation_query
from ...candidates.signals import compute_signals, signals_to_prompt_block
from ...models.embedding import embed_batch
from ...models.reranker import rerank, threshold as rerank_threshold
from ...models.structured_output import JUDGE_SCHEMA
from ...models.vllm_client import get_client
from .cross_domain import (
    cross_domain_judge,
    is_cross_domain_candidate,
)
from .level_rule import compute_level, gate_b_from_dimensions
from .triage import triage

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "main_judge.txt").read_text()

_MAX_ANCHOR_ABSTRACTS = 5          # anchors shown to judge (longest first)
_BATCH_SIZE = 50                   # candidates per judging batch
_ABSTRACT_TRUNCATE = 800           # chars; longer abstracts are trimmed


# ── Main entry point ──────────────────────────────────────────────────────────

async def run_judging(
    db_path: str,
    run_id: str,
    plan: dict,
    conn_write: sqlite3.Connection,
) -> dict:
    """
    Judge all pending candidates for the given run.
    Returns stats dict with counts per level.
    """
    anchors = _load_anchors(db_path, run_id)
    intent = plan.get("intent", "")
    dimensions = plan.get("dimensions", [])

    # Precompute anchor and dimension embeddings once (reused for every candidate)
    anchor_vecs: list[tuple[str, np.ndarray]] = []
    intent_vec: np.ndarray | None = None
    dim_vecs: list[tuple[str, np.ndarray]] = []

    try:
        texts = [intent] + [d["value"] for d in dimensions]
        anchor_texts = [f"{a.get('title','')} {a.get('abstract','')[:300]}" for a in anchors]
        all_texts = texts + anchor_texts
        all_vecs = await embed_batch(all_texts)
        intent_vec = all_vecs[0]
        dim_vecs = [(dimensions[i]["value"], all_vecs[i + 1]) for i in range(len(dimensions))]
        anchor_vecs = [(anchors[i]["paper_id"], all_vecs[len(texts) + i]) for i in range(len(anchors))]
    except Exception as exc:
        logger.warning("Embedding precomputation failed (signals will be partial): %s", exc)

    # Format anchors block for judge prompt (reused for every candidate)
    anchors_block = _format_anchors_block(anchors)

    stats = {"CORE": 0, "SUPPORTING": 0, "CONTEXT": 0, "ADJACENT": 0, "CUT": 0, "error": 0}
    total_judged = 0

    while True:
        batch = fetch_pending_candidates(db_path, run_id, limit=_BATCH_SIZE)
        if not batch:
            break

        logger.info("Judging batch of %d candidates (total so far: %d)", len(batch), total_judged)

        # T2: reranker on the whole batch at once (GPU-efficient)
        reranker_scores = await _rerank_batch(intent, batch)

        for i, candidate in enumerate(batch):
            paper_id = candidate["paper_id"]
            r_score = float(reranker_scores[i]) if reranker_scores is not None else None

            # T1: hard rules
            should_cut, cut_reason = triage(candidate, plan)
            if should_cut:
                await _write_decision(
                    conn_write, run_id, paper_id, "CUT", "T1",
                    judge_score=None, confidence=1.0, judged_by="triage",
                    cut_reason=cut_reason,
                )
                stats["CUT"] += 1
                total_judged += 1
                continue

            # T2: reranker threshold
            rth = rerank_threshold()
            if r_score is not None and r_score < rth:
                await _write_decision(
                    conn_write, run_id, paper_id, "CUT", "T2",
                    judge_score=None, confidence=0.9, judged_by="reranker",
                    cut_reason=f"reranker_score:{r_score:.3f}<{rth}",
                )
                stats["CUT"] += 1
                total_judged += 1
                continue

            # T4: full LLM judge — route cross-domain candidates separately
            try:
                seen_by = json.loads(candidate.get("seen_by_json") or "[]")
                if is_cross_domain_candidate(seen_by):
                    cd_query = get_concept_translation_query(db_path, run_id, paper_id)
                    level, judge_out, confidence = await cross_domain_judge(
                        candidate=candidate,
                        intent=intent,
                        dimensions=dimensions,
                        cross_domain_field=(cd_query or {}).get("source", ""),
                        cross_domain_query=(cd_query or {}).get("query_text", ""),
                        cross_domain_rationale="(retrieved by concept-translation reframing)",
                    )
                    await _write_decision(
                        conn_write, run_id, paper_id, level, "T4",
                        judge_score=judge_out, confidence=confidence,
                        judged_by="cross_domain_judge",
                    )
                    stats[level] += 1
                    total_judged += 1
                    continue

                cand_vec: np.ndarray | None = None
                if intent_vec is not None:
                    text = f"{candidate.get('title','')} {candidate.get('abstract','')[:300]}"
                    vecs = await embed_batch([text])
                    cand_vec = vecs[0]

                signals = compute_signals(
                    candidate, anchor_vecs, intent_vec or np.zeros(1),
                    dim_vecs, cand_vec, r_score,
                )
                level, judge_out, confidence = await _llm_judge(
                    candidate, anchors_block, intent, dimensions, signals, plan,
                )

                await _write_decision(
                    conn_write, run_id, paper_id, level, "T4",
                    judge_score=judge_out, confidence=confidence,
                    judged_by="main_judge",
                )
                stats[level] += 1

            except Exception as exc:
                logger.error("LLM judge failed for %s: %s", paper_id, exc)
                await _write_decision(
                    conn_write, run_id, paper_id, "CUT", "T4",
                    judge_score=None, confidence=0.0, judged_by="main_judge",
                    cut_reason=f"judge_error:{exc}",
                    status="parse_error",
                )
                stats["error"] += 1

            total_judged += 1

    logger.info("Judging complete. %s", stats)
    return stats


# ── LLM judge ────────────────────────────────────────────────────────────────

async def _llm_judge(
    candidate: dict,
    anchors_block: str,
    intent: str,
    dimensions: list[dict],
    signals: dict,
    plan: dict,
) -> tuple[str, dict, float]:
    """
    Call the main LLM judge. Returns (level, raw_judge_output, confidence).
    Level is derived deterministically from gate scores + flags.
    """
    dims_block = _format_dimensions(dimensions)
    abstract = (candidate.get("abstract") or "")[:_ABSTRACT_TRUNCATE]
    authors_list = json.loads(candidate.get("authors_json") or "[]")
    authors_str = ", ".join(authors_list[:5]) + (" et al." if len(authors_list) > 5 else "")

    prompt = _JUDGE_PROMPT.format(
        intent=intent,
        dimensions=dims_block,
        anchors_block=anchors_block,
        title=candidate.get("title", ""),
        authors=authors_str,
        year=candidate.get("year") or "n.d.",
        venue=candidate.get("venue") or "unknown",
        abstract=abstract or "(no abstract)",
        signals_block=signals_to_prompt_block(signals),
    )

    messages = [{"role": "user", "content": prompt}]
    llm = get_client()
    judge_out = await llm.judge(messages, JUDGE_SCHEMA)

    # Derive Gate B from dimension scores (overrides LLM's gate_B_overall with code)
    gate_b_computed = gate_b_from_dimensions(
        judge_out.get("gate_B_dimension_scores", []),
        plan.get("dimensions"),
    )
    # We use the LLM's gate_B_overall as a cross-check; if they differ by >1, log it
    llm_gate_b = judge_out.get("gate_B_overall", 0)
    if abs(gate_b_computed - llm_gate_b) > 1:
        logger.debug(
            "Gate B mismatch for %s: code=%d llm=%d",
            candidate["paper_id"], gate_b_computed, llm_gate_b,
        )

    level = compute_level(
        gate_a_score=judge_out.get("gate_A_score", 0),
        gate_b_overall=gate_b_computed,
        gate_b_dimension_scores=judge_out.get("gate_B_dimension_scores", []),
        flags=judge_out.get("flags", []),
        is_cross_domain=False,
        plan_dimensions=plan.get("dimensions"),
    )

    confidence = float(judge_out.get("confidence", 0.5))
    return level, judge_out, confidence


# ── Batch reranking ───────────────────────────────────────────────────────────

async def _rerank_batch(intent: str, candidates: list[dict]) -> np.ndarray | None:
    passages = [
        f"{c.get('title','')} {(c.get('abstract') or '')[:400]}"
        for c in candidates
    ]
    try:
        return await rerank(intent, passages)
    except Exception as exc:
        logger.warning("Reranker failed; skipping T2 for this batch: %s", exc)
        return None


# ── Write decision ────────────────────────────────────────────────────────────

async def _write_decision(
    conn: sqlite3.Connection,
    run_id: str,
    paper_id: str,
    level: str,
    tier: str,
    judge_score: dict | None,
    confidence: float,
    judged_by: str,
    cut_reason: str = "",
    status: str = "judged",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    actual_status = status if level == "CUT" and status == "parse_error" else (
        "cut" if level == "CUT" else "judged"
    )
    evidence = (
        judge_score.get("gate_A_evidence", "") if judge_score else cut_reason
    )
    flags = json.dumps(judge_score.get("flags", [])) if judge_score else "[]"

    conn.execute(
        """
        UPDATE candidates
        SET judge_status    = ?,
            judge_tier      = ?,
            level           = ?,
            judge_score_json = ?,
            judge_confidence = ?,
            judged_by       = ?,
            judged_at       = ?,
            flags_json      = ?,
            evidence_span   = ?
        WHERE run_id = ? AND paper_id = ?
        """,
        (
            actual_status,
            tier,
            level,
            json.dumps(judge_score) if judge_score else None,
            confidence,
            judged_by,
            now,
            flags,
            evidence,
            run_id,
            paper_id,
        ),
    )
    conn.commit()


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_anchors_block(anchors: list[dict]) -> str:
    if not anchors:
        return "(no anchor papers provided)"
    lines: list[str] = []
    for a in anchors[:_MAX_ANCHOR_ABSTRACTS]:
        authors = json.loads(a.get("authors_json") or "[]")
        first = authors[0] if authors else "Unknown"
        lines.append(
            f"[{a['paper_id']}] {a.get('title', 'Unknown')} — {first} et al. ({a.get('year','n.d.')})\n"
            f"Abstract: {(a.get('abstract') or '')[:400]}…"
        )
    return "\n\n".join(lines)


def _format_dimensions(dimensions: list[dict]) -> str:
    if not dimensions:
        return "(no dimensions specified)"
    lines: list[str] = []
    for d in dimensions:
        flag = " ★" if d.get("critical") else ("" if not d.get("essential") else " (essential)")
        lines.append(f"- {d['name']}: {d['value']}{flag}")
    return "\n".join(lines)


def _load_anchors(db_path: str, run_id: str) -> list[dict]:
    from ...candidates.queries import get_anchors
    return get_anchors(db_path, run_id)
