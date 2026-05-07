"""
Stage 5 — Adversarial skeptic.

A second LLM agent — on a DIFFERENT model family than the main judge
(decided in design: main=Qwen3, skeptic=Llama/Mistral) — reviews a
stratified sample of CUT papers and flags any that should be reconsidered.

Flagged papers re-enter the judge queue and are re-judged by the main
judge with their skeptic_flag set. The overturn rate (fraction of flagged
papers that the main judge then promotes) is the key metric fed to
Stage 7 coverage estimator.

Why a different model family?
  If both judge and skeptic share the same base model, their failure modes
  are correlated and the adversarial review adds little signal. A different
  family's biases are largely independent (FLAG F7 from the design doc).

  In practice the skeptic uses the same vLLM server but a different model
  name — this requires either (a) multi-model vLLM serving or (b) a second
  vLLM instance on a different port. The config exposes a dedicated
  skeptic_base_url. If the skeptic model is unavailable, Stage 5 is
  skipped gracefully and Stage 7 notes the missing signal.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI, APIConnectionError

logger = logging.getLogger(__name__)

_SKEPTIC_PROMPT = (Path(__file__).parent.parent / "prompts" / "skeptic.txt").read_text()
_CFG_PATH = os.path.join(os.path.dirname(__file__), "../config/models.yaml")

# Fraction of CUT papers to sample for skeptic review
_SAMPLE_FRAC = 0.15
_SAMPLE_MIN = 20
_SAMPLE_MAX = 150
# Papers per batch sent to skeptic (keep prompt size manageable)
_BATCH_SIZE = 15


def _skeptic_cfg() -> dict:
    path = os.environ.get("PAPER_DISCOVER_MODELS_CONFIG", _CFG_PATH)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("skeptic", cfg["local"])


def _get_skeptic_client() -> AsyncOpenAI:
    cfg = _skeptic_cfg()
    base_url = os.environ.get(
        "PAPER_DISCOVER_SKEPTIC_URL",
        cfg.get("base_url", "http://localhost:8001/v1"),
    )
    api_key = cfg.get("api_key", "token-skeptic")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _skeptic_model() -> str:
    return _skeptic_cfg().get("skeptic_model", "Llama-3.3-8B-Instruct")


# ── Verdict schema for structured output ─────────────────────────────────────

_VERDICT_SCHEMA: dict = {
    "type": "object",
    "required": ["verdicts"],
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["paper_id", "reconsider", "reason"],
                "additionalProperties": False,
                "properties": {
                    "paper_id":    {"type": "string"},
                    "reconsider":  {"type": "boolean"},
                    "reason":      {"type": "string"},
                },
            },
        }
    },
}


# ── Public entry point ────────────────────────────────────────────────────────

async def run_skeptic(
    db_path: str,
    run_id: str,
    plan: dict,
) -> dict:
    """
    Run the adversarial skeptic pass. Returns a summary dict:
      flagged, total_sampled, overturn_rate (filled in after re-judging),
      skipped (True if skeptic model unreachable)
    """
    logger.info("[%s] Stage 5: adversarial skeptic", run_id)

    # Check availability before loading any heavy context
    available = await _check_skeptic_available()
    if not available:
        logger.warning(
            "[%s] Skeptic model unreachable; skipping Stage 5. "
            "Coverage estimate will note the missing signal.",
            run_id,
        )
        return {"flagged": 0, "total_sampled": 0, "overturn_rate": None, "skipped": True}

    cut_papers = _get_cut_papers(db_path, run_id)
    sample = _stratified_sample(cut_papers)

    if not sample:
        logger.info("[%s] No CUT papers to review", run_id)
        return {"flagged": 0, "total_sampled": 0, "overturn_rate": 0.0, "skipped": False}

    logger.info("[%s] Skeptic reviewing %d / %d CUT papers", run_id, len(sample), len(cut_papers))

    anchors_block = _format_anchors(db_path, run_id)
    intent = plan.get("intent", "")
    dims_block = _format_dimensions(plan.get("dimensions", []))

    flagged_ids: list[str] = []

    for i in range(0, len(sample), _BATCH_SIZE):
        batch = sample[i : i + _BATCH_SIZE]
        papers_block = _format_papers_block(batch)

        prompt = _SKEPTIC_PROMPT.format(
            intent=intent,
            dimensions=dims_block,
            anchors_block=anchors_block,
            papers_block=papers_block,
        )
        verdicts = await _call_skeptic(prompt)
        for v in verdicts:
            if v.get("reconsider"):
                flagged_ids.append(v["paper_id"])
                _write_skeptic_flag(
                    db_path, run_id, v["paper_id"],
                    reason=v.get("reason", ""),
                )

    logger.info("[%s] Skeptic flagged %d papers for reconsideration", run_id, len(flagged_ids))

    # Re-queue flagged papers for main judging
    if flagged_ids:
        _requeue_flagged(db_path, run_id, flagged_ids)
        from .stage3_judge.main_judge import run_judging
        from ..candidates.db import open_db
        conn = open_db(db_path)
        try:
            stats = await run_judging(db_path, run_id, plan, conn)
        finally:
            conn.close()

        overturned = _count_overturned(db_path, run_id, flagged_ids)
        overturn_rate = overturned / max(len(flagged_ids), 1)
        _update_resolutions(db_path, run_id, flagged_ids)
    else:
        stats = {}
        overturn_rate = 0.0

    return {
        "flagged": len(flagged_ids),
        "total_sampled": len(sample),
        "overturn_rate": overturn_rate,
        "skipped": False,
    }


# ── Skeptic LLM call ──────────────────────────────────────────────────────────

async def _check_skeptic_available() -> bool:
    try:
        client = _get_skeptic_client()
        # Lightweight probe: list models
        await client.models.list()
        return True
    except (APIConnectionError, Exception):
        return False


async def _call_skeptic(prompt: str) -> list[dict]:
    client = _get_skeptic_client()
    model = _skeptic_model()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
            extra_body={"guided_json": _VERDICT_SCHEMA},
        )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw).get("verdicts", [])
    except Exception as exc:
        logger.error("Skeptic call failed: %s", exc)
        return []


# ── Sampling ──────────────────────────────────────────────────────────────────

def _stratified_sample(cut_papers: list[dict]) -> list[dict]:
    """
    Stratified by reranker score percentile so the skeptic sees a mix of
    obvious cuts and borderline cuts — not just low-score papers.
    """
    if not cut_papers:
        return []

    n = min(
        max(_SAMPLE_MIN, int(len(cut_papers) * _SAMPLE_FRAC)),
        _SAMPLE_MAX,
        len(cut_papers),
    )

    # Sort by reranker score descending (higher = more borderline)
    def _score(p: dict) -> float:
        try:
            sig = json.loads(p.get("signals_json") or "{}")
            return float(sig.get("reranker_score") or 0.0)
        except Exception:
            return 0.0

    sorted_papers = sorted(cut_papers, key=_score, reverse=True)

    # Take top-third borderline, bottom-third obvious cuts, middle random
    n_borderline = n // 3
    n_obvious = n // 3
    n_random = n - n_borderline - n_obvious

    borderline = sorted_papers[:n_borderline]
    obvious = sorted_papers[-n_obvious:] if n_obvious else []
    middle = sorted_papers[n_borderline : len(sorted_papers) - n_obvious]
    random_sample = random.sample(middle, min(n_random, len(middle)))

    return borderline + random_sample + obvious


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_cut_papers(db_path: str, run_id: str) -> list[dict]:
    from ..candidates.db import read_conn
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.signals_json, c.judge_tier,
                   p.title, p.abstract, p.year, p.venue, p.authors_json
            FROM candidates c JOIN papers p USING (paper_id)
            WHERE c.run_id = ? AND c.level = 'CUT'
              AND c.judge_tier IN ('T2','T3','T4')   -- exclude T1 hard-rule cuts
            """,
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _format_papers_block(papers: list[dict]) -> str:
    lines: list[str] = []
    for p in papers:
        authors = json.loads(p.get("authors_json") or "[]")
        auth_str = (authors[0] if authors else "Unknown") + (" et al." if len(authors) > 1 else "")
        abstract = (p.get("abstract") or "")[:400]
        lines.append(
            f"[{p['paper_id']}] {p.get('title', 'Unknown title')} — {auth_str} ({p.get('year','n.d.')})\n"
            f"  {abstract}"
        )
    return "\n\n".join(lines)


def _format_anchors(db_path: str, run_id: str) -> str:
    from ..candidates.queries import get_anchors
    anchors = get_anchors(db_path, run_id)
    if not anchors:
        return "(none)"
    lines: list[str] = []
    for a in anchors[:4]:
        lines.append(f"- {a.get('title','?')} ({a.get('year','n.d.')})")
    return "\n".join(lines)


def _format_dimensions(dimensions: list[dict]) -> str:
    if not dimensions:
        return "(none)"
    return "\n".join(
        f"- {d['name']}: {d['value']}" + (" ★" if d.get("critical") else "")
        for d in dimensions
        if d.get("essential")
    )


def _write_skeptic_flag(db_path: str, run_id: str, paper_id: str, reason: str) -> None:
    from ..candidates.db import open_db
    conn = open_db(db_path)
    conn.execute(
        """
        INSERT OR IGNORE INTO skeptic_flags
          (run_id, paper_id, flagged_at, skeptic_model, skeptic_reason, resolution)
        VALUES (?,?,?,?,?,'pending')
        """,
        (
            run_id, paper_id,
            datetime.now(timezone.utc).isoformat(),
            _skeptic_model(),
            reason,
        ),
    )
    conn.commit()
    conn.close()


def _requeue_flagged(db_path: str, run_id: str, paper_ids: list[str]) -> None:
    """Reset judge_status to 'pending' so run_judging() will re-process them."""
    from ..candidates.db import open_db
    conn = open_db(db_path)
    placeholders = ",".join("?" * len(paper_ids))
    conn.execute(
        f"""
        UPDATE candidates
        SET judge_status = 'pending', judge_tier = NULL,
            level = NULL, judge_score_json = NULL,
            judge_confidence = NULL, judged_at = NULL
        WHERE run_id = ? AND paper_id IN ({placeholders})
        """,
        [run_id] + paper_ids,
    )
    conn.commit()
    conn.close()


def _count_overturned(db_path: str, run_id: str, paper_ids: list[str]) -> int:
    """Count how many re-judged papers are now NOT CUT (the skeptic was right)."""
    from ..candidates.db import read_conn
    placeholders = ",".join("?" * len(paper_ids))
    with read_conn(db_path) as conn:
        return conn.execute(
            f"""
            SELECT COUNT(*) FROM candidates
            WHERE run_id = ? AND paper_id IN ({placeholders})
              AND level != 'CUT' AND level IS NOT NULL
            """,
            [run_id] + paper_ids,
        ).fetchone()[0]


def _update_resolutions(db_path: str, run_id: str, paper_ids: list[str]) -> None:
    from ..candidates.db import open_db, read_conn
    conn = open_db(db_path)
    for pid in paper_ids:
        with read_conn(db_path) as rc:
            level = rc.execute(
                "SELECT level FROM candidates WHERE run_id=? AND paper_id=?", (run_id, pid)
            ).fetchone()
        resolution = "overturned" if (level and level[0] != "CUT") else "sustained"
        conn.execute(
            "UPDATE skeptic_flags SET resolution=? WHERE run_id=? AND paper_id=?",
            (resolution, run_id, pid),
        )
    conn.commit()
    conn.close()
