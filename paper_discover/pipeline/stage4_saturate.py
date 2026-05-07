"""
Stage 4 — Saturation pass.

After the main judging pass, expand the citation graph 1 hop from every
kept paper, judge the newcomers, and repeat until the rate of new keeps
drops below a threshold or the iteration cap is hit.

Tracks a discovery curve (new keeps per iteration) stored in saturation_log.
The curve is used by Stage 7 to estimate coverage.

Design notes
------------
- Citation edges already in the DB (deposited by Stage 2 writer) are used first.
- For newly kept papers whose neighbors we haven't fetched yet, a live API
  call retrieves the next hop.
- Judging reuses run_judging() from main_judge.py — it processes all
  pending candidates regardless of when they were added.
- Saturation signal (used by Stage 7): flatness of the discovery curve,
  measured as new_keeps_last_iter / total_keeps.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone

from ..candidates.db import DBWriter, open_db, upsert_candidate
from ..candidates.queries import citation_neighborhood
from ..pipeline.stage2_retrieve.openalex import (
    fetch_paper_by_openalex_id,
    get_citation_neighborhood as oa_citation,
)
from ..pipeline.stage2_retrieve.semantic_scholar import (
    fetch_paper_by_id as s2_fetch,
    get_citation_neighborhood as s2_citation,
)
from ..pipeline.stage2_retrieve.writer import RetrievalQueue, canonical_id, normalise_authors
from ..pipeline.stage3_judge.main_judge import run_judging

import json

logger = logging.getLogger(__name__)

SATURATION_THRESHOLD = 0.05   # stop when new_keeps / total_keeps < 5%
MAX_ITERATIONS = 5
_SAT_CHANNEL = "saturation"


# ── Public entry point ────────────────────────────────────────────────────────

async def run_saturation(
    db_path: str,
    run_id: str,
    plan: dict,
) -> dict:
    """
    Run the saturation loop. Returns a summary dict with:
      iterations, total_keeps, converged, saturation_signal
    """
    logger.info("[%s] Stage 4: saturation pass (max %d iterations)", run_id, MAX_ITERATIONS)

    summary = {
        "iterations": 0,
        "total_keeps": 0,
        "converged": False,
        "saturation_signal": 0.0,
        "log": [],
    }

    for iteration in range(1, MAX_ITERATIONS + 1):
        summary["iterations"] = iteration
        logger.info("[%s] Saturation iteration %d", run_id, iteration)

        kept_ids = _get_kept_ids(db_path, run_id)
        if not kept_ids:
            logger.info("[%s] No kept papers yet; stopping saturation", run_id)
            break

        existing_candidate_ids = _get_all_candidate_ids(db_path, run_id)

        # Expand via citation edges already in the DB (fast, free)
        db_neighbors = citation_neighborhood(db_path, kept_ids, direction="both")
        truly_new_from_db = db_neighbors - existing_candidate_ids

        # Expand via live API for kept papers added in the PREVIOUS iteration
        # (their edges weren't fetched during Stage 2 retrieval)
        prev_iter_kept = _get_prev_iteration_kept(db_path, run_id, iteration)
        live_papers = await _fetch_live_neighbors(prev_iter_kept, existing_candidate_ids)

        new_candidates_count = len(truly_new_from_db) + len(live_papers)
        if new_candidates_count == 0:
            logger.info("[%s] No new candidates found; saturation reached at iteration %d",
                        run_id, iteration)
            summary["converged"] = True
            _log_saturation(db_path, run_id, iteration, 0, 0,
                            _count_keeps(db_path, run_id), converged=True)
            break

        # Upsert new candidates from DB neighbors
        async with DBWriter(db_path) as writer:
            q_id = f"sat:{run_id}:{iteration}:db"
            for pid in truly_new_from_db:
                await upsert_candidate(writer, run_id, pid, _SAT_CHANNEL,
                                       hop_distance=iteration + 1)

            # Upsert live-fetched papers
            queue = RetrievalQueue(writer, run_id)
            await queue.start()
            for paper in live_papers:
                refs = paper.pop("_references", [])
                citers = paper.pop("_cited_by", [])
                await queue.submit(
                    paper, _SAT_CHANNEL, f"sat:{run_id}:{iteration}:live",
                    hop_distance=iteration + 1,
                    references=refs, cited_by=citers,
                )
            await queue.stop()

        # Re-run judging on all newly pending candidates
        conn = open_db(db_path)
        try:
            stats = await run_judging(db_path, run_id, plan, conn)
        finally:
            conn.close()

        new_keeps = sum(stats.get(l, 0) for l in ("CORE", "SUPPORTING", "CONTEXT", "ADJACENT"))
        total_keeps = _count_keeps(db_path, run_id)

        log_entry = {
            "iteration": iteration,
            "new_candidates": new_candidates_count,
            "new_keeps": new_keeps,
            "total_keeps": total_keeps,
        }
        summary["log"].append(log_entry)
        summary["total_keeps"] = total_keeps

        converged = (new_keeps / max(total_keeps, 1)) < SATURATION_THRESHOLD
        _log_saturation(db_path, run_id, iteration, new_candidates_count,
                        new_keeps, total_keeps, converged=converged)

        logger.info(
            "[%s] Iteration %d: +%d candidates, +%d keeps, total=%d (converged=%s)",
            run_id, iteration, new_candidates_count, new_keeps, total_keeps, converged,
        )

        if converged:
            summary["converged"] = True
            break

    # Compute saturation signal for Stage 7
    log = summary["log"]
    if log:
        last_new_keeps = log[-1]["new_keeps"]
        total = max(summary["total_keeps"], 1)
        summary["saturation_signal"] = max(0.0, 1.0 - (last_new_keeps / total))
    else:
        summary["saturation_signal"] = 1.0  # no iterations ran → assumed saturated

    logger.info("[%s] Saturation done. signal=%.3f", run_id, summary["saturation_signal"])
    return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_kept_ids(db_path: str, run_id: str) -> list[str]:
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT paper_id FROM candidates
            WHERE run_id = ? AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            """,
            (run_id,),
        ).fetchall()
    return [r[0] for r in rows]


def _get_all_candidate_ids(db_path: str, run_id: str) -> set[str]:
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT paper_id FROM candidates WHERE run_id = ?", (run_id,)
        ).fetchall()
    return {r[0] for r in rows}


def _get_prev_iteration_kept(db_path: str, run_id: str, current_iter: int) -> list[str]:
    """
    Return paper_ids of papers kept in the PREVIOUS saturation iteration.
    These are candidates added by the saturation channel in the previous iteration
    that were then promoted above CUT.
    On iteration 1, returns all kept papers from the main judging pass (Stage 3).
    """
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        if current_iter == 1:
            rows = conn.execute(
                """
                SELECT paper_id FROM candidates
                WHERE run_id = ?
                  AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
                  AND judge_tier IN ('T1','T2','T3','T4','T5')
                """,
                (run_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT paper_id FROM candidates
                WHERE run_id = ?
                  AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
                  AND seen_by_json LIKE ?
                """,
                (run_id, f'%{_SAT_CHANNEL}%'),
            ).fetchall()
    return [r[0] for r in rows]


async def _fetch_live_neighbors(
    paper_ids: list[str],
    existing_ids: set[str],
    max_per_paper: int = 50,
) -> list[dict]:
    """
    Fetch citation neighbors live from S2/OpenAlex for papers not yet expanded.
    Returns only papers whose canonical_id is not in existing_ids.
    """
    if not paper_ids:
        return []

    # Separate S2 IDs from OpenAlex IDs
    s2_ids = [pid.split(":")[-1] for pid in paper_ids if pid.startswith("s2:")]
    oa_ids = [pid.replace("W", "").lstrip("0") if not pid.startswith("W") else pid
              for pid in paper_ids if pid.startswith("W") or "openalex" in pid]

    results: list[dict] = []
    if s2_ids:
        papers = await s2_citation(s2_ids, direction="both", max_per_paper=max_per_paper)
        results.extend(papers)
    if oa_ids:
        papers = await oa_citation(oa_ids[:10], direction="both", max_results=100)
        results.extend(papers)

    # Deduplicate against existing candidates
    filtered: list[dict] = []
    seen: set[str] = set()
    for p in results:
        pid = canonical_id(p)
        if pid not in existing_ids and pid not in seen:
            seen.add(pid)
            filtered.append(p)
    return filtered


def _count_keeps(db_path: str, run_id: str) -> int:
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        return conn.execute(
            """
            SELECT COUNT(*) FROM candidates
            WHERE run_id = ? AND level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            """,
            (run_id,),
        ).fetchone()[0]


def _log_saturation(
    db_path: str,
    run_id: str,
    iteration: int,
    new_candidates: int,
    new_keeps: int,
    total_keeps: int,
    converged: bool,
) -> None:
    conn = open_db(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO saturation_log
          (run_id, iteration, new_candidates, new_keeps, total_keeps, converged, logged_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            run_id, iteration, new_candidates, new_keeps,
            total_keeps, int(converged),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
