"""
Stage 8 — PRISMA flow diagram (M7).

Renders a Mermaid flowchart of the screening funnel using counts pulled
straight from the candidates and queries tables. The numbers are
authoritative: every transition in the diagram corresponds to a SQL
COUNT, never a hand-tracked variable.

Layers
------
  1. Identified through retrieval     = candidate_sources rows
  2. After dedup                      = unique paper_ids in candidates
  3. Screened (T1 + T2 hard cuts)     = T1 ∪ T2 cuts in candidates
  4. Assessed by full LLM judge       = T3 + T4 + T5 rows
  5. Included (kept levels)           = level NOT IN ('CUT')
"""

from __future__ import annotations

import logging
from pathlib import Path

from ...candidates.db import read_conn

logger = logging.getLogger(__name__)


# ── Counts ────────────────────────────────────────────────────────────────────

def collect_prisma_counts(db_path: str, run_id: str) -> dict:
    with read_conn(db_path) as conn:
        identified = conn.execute(
            "SELECT COUNT(*) FROM candidate_sources WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]

        unique_after_dedup = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]

        t1_cuts = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id=? AND judge_tier='T1' AND level='CUT'",
            (run_id,),
        ).fetchone()[0]
        t2_cuts = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE run_id=? AND judge_tier='T2' AND level='CUT'",
            (run_id,),
        ).fetchone()[0]

        assessed_full = conn.execute(
            """
            SELECT COUNT(*) FROM candidates
             WHERE run_id=? AND judge_tier IN ('T3','T4','T5')
            """,
            (run_id,),
        ).fetchone()[0]

        kept_by_level = dict(
            conn.execute(
                """
                SELECT level, COUNT(*) FROM candidates
                 WHERE run_id=? AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
                 GROUP BY level
                """,
                (run_id,),
            ).fetchall()
        )

        excluded_after_assessment = conn.execute(
            """
            SELECT COUNT(*) FROM candidates
             WHERE run_id=? AND judge_tier IN ('T3','T4','T5') AND level='CUT'
            """,
            (run_id,),
        ).fetchone()[0]

    kept_total = sum(kept_by_level.values())
    return {
        "identified":               identified,
        "after_dedup":              unique_after_dedup,
        "duplicates_removed":       max(identified - unique_after_dedup, 0),
        "t1_cuts":                  t1_cuts,
        "t2_cuts":                  t2_cuts,
        "screened":                 unique_after_dedup,
        "assessed_full":            assessed_full,
        "excluded_after_assessment": excluded_after_assessment,
        "kept_by_level":            kept_by_level,
        "kept_total":               kept_total,
    }


# ── Mermaid renderer ──────────────────────────────────────────────────────────

def generate_prisma_mermaid(counts: dict) -> str:
    by_level = counts.get("kept_by_level") or {}
    kept_lines = "<br/>".join(
        f"{level}: {n}" for level, n in by_level.items()
    ) or "(none)"

    return (
        "flowchart TD\n"
        f"  A[Records identified through retrieval<br/>n = {counts['identified']}] --> B\n"
        f"  B[After dedup<br/>n = {counts['after_dedup']}<br/>"
        f"<i>{counts['duplicates_removed']} duplicates removed</i>] --> C\n"
        f"  C[Screened<br/>n = {counts['screened']}] --> D\n"
        f"  D[After hard rules / reranker<br/>"
        f"<i>T1 cuts: {counts['t1_cuts']}, T2 cuts: {counts['t2_cuts']}</i>] --> E\n"
        f"  E[Assessed by full LLM judge<br/>n = {counts['assessed_full']}] --> F\n"
        f"  F[Excluded after assessment<br/>n = {counts['excluded_after_assessment']}]\n"
        f"  E --> G[Included<br/>n = {counts['kept_total']}<br/>{kept_lines}]\n"
    )


def write_prisma(output_dir: str, db_path: str, run_id: str) -> dict:
    counts = collect_prisma_counts(db_path, run_id)
    mermaid = generate_prisma_mermaid(counts)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prisma.mmd").write_text(mermaid)
    logger.info("PRISMA → %s (kept=%d / identified=%d)",
                out / "prisma.mmd", counts["kept_total"], counts["identified"])
    return counts
