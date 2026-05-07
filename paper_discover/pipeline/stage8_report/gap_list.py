"""
Stage 8 — Gap list (M7).

Synthesises open questions grounded ONLY in the kept papers — no outside
knowledge. The LLM call is isolated so the formatting / rendering is
unit-testable without a model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...candidates.db import read_conn
from ...models.structured_output import GAP_LIST_SCHEMA

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "gap_list.txt").read_text()
_MAX_PAPERS_IN_PROMPT = 60


# ── Public entry point ───────────────────────────────────────────────────────

async def generate_gap_list(
    db_path: str,
    run_id: str,
    plan: dict,
) -> dict:
    """
    Run the gap-extraction LLM call. Returns {"gaps": [...]}. Returns
    {"gaps": []} on any failure — the report should still render.
    """
    papers = _load_kept_with_summaries(db_path, run_id)
    if not papers:
        return {"gaps": []}

    prompt = _PROMPT.format(
        intent=plan.get("intent", ""),
        dimensions=_format_dimensions(plan.get("dimensions", [])),
        papers_block=format_papers_block(papers),
    )

    try:
        from ...models.vllm_client import get_client
        llm = get_client()
        out = await llm.plan_json([{"role": "user", "content": prompt}], GAP_LIST_SCHEMA)
        return {"gaps": out.get("gaps", [])}
    except Exception as exc:
        logger.warning("Gap-list LLM call failed: %s", exc)
        return {"gaps": []}


def write_gap_list(
    output_dir: str,
    gaps: dict,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "gaps.md"
    md_path.write_text(render_gap_list_md(gaps))
    return md_path


# ── Pure helpers (testable without an LLM) ───────────────────────────────────

def format_papers_block(papers: list[dict]) -> str:
    lines = []
    for p in papers[:_MAX_PAPERS_IN_PROMPT]:
        title = (p.get("title") or "").strip() or "(no title)"
        year = p.get("year") or "n.d."
        summary = (p.get("summary") or "").strip()
        lines.append(f"[{p['paper_id']}] {title} ({year})\n  {summary}")
    return "\n\n".join(lines)


def _format_dimensions(dimensions: list[dict]) -> str:
    if not dimensions:
        return "(none)"
    return "\n".join(
        f"- {d['name']}: {d['value']}" + (" ★" if d.get("critical") else "")
        for d in dimensions
        if d.get("essential")
    )


def render_gap_list_md(gaps: dict) -> str:
    items = gaps.get("gaps") or []
    if not items:
        return "# Open questions / gaps\n\n_No gaps identified in this run._\n"

    lines = ["# Open questions / gaps", ""]
    known = {"methodological", "population", "outcome", "mechanism", "replication"}
    by_cat: dict[str, list[dict]] = {}
    for g in items:
        cat = g.get("category", "uncategorised")
        if cat not in known:
            cat = "uncategorised"
        by_cat.setdefault(cat, []).append(g)

    cat_titles = {
        "methodological": "Methodological",
        "population":     "Population",
        "outcome":        "Outcome",
        "mechanism":      "Mechanism",
        "replication":    "Replication",
        "uncategorised":  "Other",
    }
    for cat in ("methodological", "population", "outcome", "mechanism",
                "replication", "uncategorised"):
        bucket = by_cat.get(cat) or []
        if not bucket:
            continue
        lines += [f"## {cat_titles[cat]}", ""]
        for g in bucket:
            ref = ", ".join(f"`{pid}`" for pid in g.get("motivated_by") or [])
            ref_str = f" — motivated by {ref}" if ref else ""
            lines += [
                f"### {g.get('question', '(no question)')}",
                f"",
                f"{g.get('rationale', '').strip()}{ref_str}",
                f"",
            ]
    return "\n".join(lines)


# ── DB load ──────────────────────────────────────────────────────────────────

def _load_kept_with_summaries(db_path: str, run_id: str) -> list[dict]:
    """
    Pull kept papers plus their two_sentence_summary from judge_score_json.
    Falls back to the abstract first sentence if the summary is missing.
    """
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.judge_score_json,
                   p.title, p.year, p.abstract
            FROM candidates c JOIN papers p USING (paper_id)
            WHERE c.run_id = ?
              AND c.level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            ORDER BY c.judge_confidence DESC NULLS LAST
            """,
            (run_id,),
        ).fetchall()

    papers: list[dict] = []
    for r in rows:
        summary = ""
        if r["judge_score_json"]:
            try:
                summary = json.loads(r["judge_score_json"]).get(
                    "two_sentence_summary", ""
                )
            except json.JSONDecodeError:
                pass
        if not summary:
            abstract = (r["abstract"] or "").strip()
            summary = abstract.split(". ")[0][:200] if abstract else ""
        papers.append({
            "paper_id": r["paper_id"],
            "title":    r["title"],
            "year":     r["year"],
            "summary":  summary,
        })
    return papers
