"""
Compute per-candidate signal block stored in candidates.signals_json.
Signals are fed verbatim to the judge prompt as context.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from ..models.embedding import cosine_sim


def _load_vector(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _bin_seen_count(seen_count: int) -> str:
    if seen_count == 1:
        return "1 channel"
    if seen_count <= 3:
        return "2–3 channels"
    return "4+ channels"


def compute_signals(
    candidate: dict,
    anchor_vecs: list[tuple[str, np.ndarray]],
    intent_vec: np.ndarray,
    dim_vecs: list[tuple[str, np.ndarray]],
    candidate_vec: np.ndarray | None,
    reranker_score: float | None,
) -> dict:
    """
    Build the signals dict that goes into candidates.signals_json and the judge prompt.

    Args:
        candidate:      row from candidates JOIN papers (includes seen_count, hop_distance)
        anchor_vecs:    list of (paper_id, embedding_vector) for each anchor
        intent_vec:     embedding of the plan's intent statement
        dim_vecs:       list of (dimension_name, embedding_vector)
        candidate_vec:  embedding of the candidate's title+abstract (None if unavailable)
        reranker_score: T2 cross-encoder score (None if not yet computed)
    """
    signals: dict = {
        "seen_count_bin": _bin_seen_count(candidate.get("seen_count", 1)),
        "seen_by":        json.loads(candidate.get("seen_by_json") or "[]"),
        "hop_distance":   candidate.get("hop_distance_to_anchor"),
        "reranker_score": round(reranker_score, 4) if reranker_score is not None else None,
    }

    if candidate_vec is not None:
        # Similarity to intent
        signals["intent_sim"] = round(cosine_sim(candidate_vec, intent_vec), 4)

        # Per-anchor similarity
        anchor_sims = {}
        for aid, avec in anchor_vecs:
            anchor_sims[aid] = round(cosine_sim(candidate_vec, avec), 4)
        signals["anchor_sims"] = anchor_sims
        signals["max_anchor_sim"] = round(max(anchor_sims.values()), 4) if anchor_sims else None

        # Per-dimension similarity
        dim_sims = {}
        for dname, dvec in dim_vecs:
            dim_sims[dname] = round(cosine_sim(candidate_vec, dvec), 4)
        signals["dimension_sims"] = dim_sims

    return signals


def signals_to_prompt_block(signals: dict) -> str:
    """Format signals dict as the text block shown in the judge prompt."""
    lines = [
        f"- Found by: {signals.get('seen_count_bin', 'unknown')} "
        f"({', '.join(signals.get('seen_by', [])[:4])})",
    ]
    if signals.get("hop_distance") is not None:
        lines.append(f"- Citation hops to nearest anchor: {signals['hop_distance']}")
    else:
        lines.append("- Citation hops to nearest anchor: unknown")
    if signals.get("reranker_score") is not None:
        lines.append(f"- Reranker (cross-encoder) score: {signals['reranker_score']:.3f}")
    if signals.get("intent_sim") is not None:
        lines.append(f"- Embedding similarity to intent: {signals['intent_sim']:.3f}")
    if signals.get("max_anchor_sim") is not None:
        lines.append(f"- Max similarity to any anchor: {signals['max_anchor_sim']:.3f}")
    if signals.get("dimension_sims"):
        dim_str = ", ".join(
            f"{k}={v:.2f}" for k, v in signals["dimension_sims"].items()
        )
        lines.append(f"- Per-dimension similarities: {dim_str}")
    return "\n".join(lines)
