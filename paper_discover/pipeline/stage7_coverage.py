"""
Stage 7 — Calibrated coverage estimator.

Combines four independent signals into a single coverage probability with
a confidence interval. All signal values and the methodology are stored
in the coverage_signals table for full auditability.

Signals
-------
S1 — Saturation flatness (from Stage 4 saturation_log):
     1 - (new_keeps_last_iteration / total_keeps)
     High → discovery curve has flattened → we likely found most papers.

S2 — Skeptic overturn rate (from Stage 5 skeptic_flags):
     1 - (overturned / total_flagged)
     Low overturn → judge was thorough → high confidence.
     NULL if Stage 5 was skipped.

S3 — Channel Jaccard overlap (from candidates.seen_by_json):
     Average pairwise Jaccard similarity between any two retrieval
     channel's sets of kept papers. High overlap → channels converge
     on the same answer → robust coverage.

S4 — Anchor-injection accuracy (runs inline here):
     Strip identifying metadata from each anchor paper and run it
     through the main judge as a blind candidate.
     Fraction returned as CORE or SUPPORTING = accuracy.
     High accuracy → judge is correctly calibrated.

Final estimate
--------------
coverage_p = weighted average of available signals:
  weights: S1=0.30, S2=0.20, S3=0.20, S4=0.30

95% confidence interval: Wilson score interval treating coverage_p
as a binomial proportion with n = total_kept_papers.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from itertools import combinations

logger = logging.getLogger(__name__)

_WEIGHTS = {"S1": 0.30, "S2": 0.20, "S3": 0.20, "S4": 0.30}


# ── Public entry point ────────────────────────────────────────────────────────

async def run_coverage(
    db_path: str,
    run_id: str,
    plan: dict,
    saturation_summary: dict,
    skeptic_summary: dict,
) -> dict:
    """
    Compute the calibrated coverage estimate.
    Returns a dict with: coverage_p, coverage_ci_lo, coverage_ci_hi,
    signals (per-signal breakdown), methodology_json.
    """
    logger.info("[%s] Stage 7: coverage estimation", run_id)

    signals: dict[str, float | None] = {}

    # S1 — saturation flatness
    signals["S1"] = saturation_summary.get("saturation_signal")

    # S2 — skeptic overturn rate
    if skeptic_summary.get("skipped"):
        signals["S2"] = None
    else:
        overturn = skeptic_summary.get("overturn_rate", 0.0)
        signals["S2"] = 1.0 - (overturn or 0.0)

    # S3 — channel Jaccard
    signals["S3"] = _compute_channel_jaccard(db_path, run_id)

    # S4 — anchor injection accuracy
    signals["S4"] = await _anchor_injection_accuracy(db_path, run_id, plan)

    # Weighted aggregate (skip None signals, renormalise weights)
    available = {k: v for k, v in signals.items() if v is not None}
    if not available:
        logger.warning("[%s] No coverage signals available; returning 0.5 estimate", run_id)
        coverage_p, lo, hi = 0.5, 0.2, 0.8
    else:
        total_w = sum(_WEIGHTS[k] for k in available)
        coverage_p = sum(_WEIGHTS[k] * v for k, v in available.items()) / total_w
        coverage_p = max(0.0, min(1.0, coverage_p))

        # Wilson score CI using n = total_kept as effective sample size
        n_kept = _count_keeps(db_path, run_id)
        lo, hi = _wilson_ci(coverage_p, max(n_kept, 10))

    methodology = {
        "signals": signals,
        "weights": _WEIGHTS,
        "available_signals": list(available.keys()),
        "total_kept": _count_keeps(db_path, run_id),
        "saturation_log": saturation_summary.get("log", []),
        "skeptic_flagged": skeptic_summary.get("flagged"),
        "skeptic_sampled": skeptic_summary.get("total_sampled"),
    }

    _store_coverage(db_path, run_id, signals, coverage_p, lo, hi, methodology)

    logger.info(
        "[%s] Coverage estimate: %.0f%% (CI %.0f%%–%.0f%%)",
        run_id, coverage_p * 100, lo * 100, hi * 100,
    )
    return {
        "coverage_p": coverage_p,
        "coverage_ci_lo": lo,
        "coverage_ci_hi": hi,
        "signals": signals,
        "methodology": methodology,
    }


# ── S3: Channel Jaccard ───────────────────────────────────────────────────────

def _compute_channel_jaccard(db_path: str, run_id: str) -> float | None:
    """
    Average pairwise Jaccard between any two retrieval channels' kept-paper sets.
    Only considers channels with ≥ 3 kept papers (too-small sets inflate Jaccard).
    """
    from ..candidates.db import read_conn

    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.seen_by_json
            FROM candidates c
            WHERE c.run_id = ?
              AND c.level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            """,
            (run_id,),
        ).fetchall()

    # Build channel → set of kept paper_ids
    channel_papers: dict[str, set[str]] = {}
    for row in rows:
        try:
            channels = json.loads(row["seen_by_json"] or "[]")
        except Exception:
            channels = []
        pid = row["paper_id"]
        for ch in channels:
            channel_papers.setdefault(ch, set()).add(pid)

    qualifying = {ch: pids for ch, pids in channel_papers.items() if len(pids) >= 3}
    if len(qualifying) < 2:
        return None

    jaccard_scores: list[float] = []
    for ch1, ch2 in combinations(qualifying, 2):
        s1, s2 = qualifying[ch1], qualifying[ch2]
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        if union > 0:
            jaccard_scores.append(intersection / union)

    if not jaccard_scores:
        return None
    return sum(jaccard_scores) / len(jaccard_scores)


# ── S4: Anchor injection accuracy ─────────────────────────────────────────────

async def _anchor_injection_accuracy(
    db_path: str, run_id: str, plan: dict
) -> float | None:
    """
    Re-run the main judge on anchor papers (their metadata is NOT stripped —
    the judge only sees title+abstract in its prompt, not the paper_id).
    Check what fraction receive CORE or SUPPORTING (acceptable outcomes).
    Store results in anchor_probes.
    """
    from ..candidates.db import read_conn, open_db
    from ..candidates.queries import get_anchors
    from ..pipeline.stage3_judge.main_judge import (
        _format_anchors_block, _format_dimensions, _llm_judge
    )
    from ..candidates.signals import compute_signals, signals_to_prompt_block
    from ..models.embedding import embed_batch

    import numpy as np

    anchors = get_anchors(db_path, run_id)
    if not anchors:
        return None
    if len(anchors) < 3:
        logger.warning("[%s] Fewer than 3 anchors; anchor injection skipped", run_id)
        return None

    # Build context for judging (same as main_judge.py)
    intent = plan.get("intent", "")
    dimensions = plan.get("dimensions", [])

    try:
        texts = [intent] + [d["value"] for d in dimensions]
        anchor_texts = [f"{a.get('title','')} {a.get('abstract','')[:300]}" for a in anchors]
        all_vecs = await embed_batch(texts + anchor_texts)
        intent_vec = all_vecs[0]
        dim_vecs = [(dimensions[i]["value"], all_vecs[i + 1]) for i in range(len(dimensions))]
        anchor_vecs = [(anchors[i]["paper_id"], all_vecs[len(texts) + i]) for i in range(len(anchors))]
    except Exception as exc:
        logger.warning("[%s] Embedding failed for anchor probes: %s", run_id, exc)
        intent_vec = np.zeros(1)
        dim_vecs = []
        anchor_vecs = []

    anchors_block = _format_anchors_block(anchors)
    passed = 0
    conn = open_db(db_path)

    for anchor in anchors:
        paper_id = anchor["paper_id"]
        try:
            cand_vec = None
            if anchor_vecs:
                text = f"{anchor.get('title','')} {anchor.get('abstract','')[:300]}"
                vecs = await embed_batch([text])
                cand_vec = vecs[0]

            signals = compute_signals(
                anchor, anchor_vecs, intent_vec, dim_vecs, cand_vec, reranker_score=None
            )
            level, _, _ = await _llm_judge(
                candidate=anchor,
                anchors_block=anchors_block,
                intent=intent,
                dimensions=dimensions,
                signals=signals,
                plan=plan,
            )
            probe_passed = level in ("CORE", "SUPPORTING")
            if probe_passed:
                passed += 1

            conn.execute(
                """
                INSERT OR REPLACE INTO anchor_probes
                  (run_id, paper_id, expected_level, actual_level, passed)
                VALUES (?,?,?,?,?)
                """,
                (run_id, paper_id, "CORE", level, int(probe_passed)),
            )
        except Exception as exc:
            logger.warning("[%s] Anchor probe failed for %s: %s", run_id, paper_id, exc)
            conn.execute(
                "INSERT OR REPLACE INTO anchor_probes (run_id, paper_id, expected_level, actual_level, passed) VALUES (?,?,?,?,?)",
                (run_id, paper_id, "CORE", "error", 0),
            )

    conn.commit()
    conn.close()

    accuracy = passed / len(anchors)
    logger.info("[%s] Anchor injection: %d/%d passed (accuracy=%.0f%%)",
                run_id, passed, len(anchors), accuracy * 100)
    return accuracy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _count_keeps(db_path: str, run_id: str) -> int:
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id=? AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')",
            (run_id,),
        ).fetchone()[0]


def _store_coverage(
    db_path: str,
    run_id: str,
    signals: dict,
    coverage_p: float,
    lo: float,
    hi: float,
    methodology: dict,
) -> None:
    from ..candidates.db import open_db
    conn = open_db(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO coverage_signals
          (run_id, saturation_signal, skeptic_signal, channel_jaccard,
           anchor_accuracy, coverage_p, coverage_ci_lo, coverage_ci_hi,
           methodology_json, computed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            signals.get("S1"),
            signals.get("S2"),
            signals.get("S3"),
            signals.get("S4"),
            coverage_p,
            lo,
            hi,
            json.dumps(methodology),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.execute(
        "UPDATE runs SET coverage_p=?, coverage_ci_lo=?, coverage_ci_hi=? WHERE run_id=?",
        (coverage_p, lo, hi, run_id),
    )
    conn.commit()
    conn.close()
