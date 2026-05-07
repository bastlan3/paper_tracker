"""
LangGraph pipeline definition for paper-discover.

Each node corresponds to one pipeline stage. The state carries only the
run_id and plan — all substantive data lives in SQLite. This keeps the
LangGraph state small and enables checkpointing/replay cheaply.

M2 topology (linear):
  START → retrieval → judging → saturation → skeptic → coverage → hygiene → report → END

Each stage reads/writes the candidates DB; the LangGraph state itself only
carries identifiers and small summary dicts (saturation_summary,
skeptic_summary, coverage, hygiene_summary) so checkpoints stay small.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..candidates.db import DBWriter, fetch_bibliography, open_db, run_sql_invariants
from ..candidates.queries import get_anchors, zotero_paper_ids
from ..pipeline.stage1_query import build_query_plan, flatten_queries
from ..pipeline.stage2_retrieve.openalex import (
    get_citation_neighborhood as oa_citation,
    search_by_concept,
    search_keyword as oa_keyword,
)
from ..pipeline.stage2_retrieve.semantic_scholar import (
    get_citation_neighborhood as s2_citation,
    search_keyword as s2_keyword,
)
from ..pipeline.stage2_retrieve.writer import RetrievalQueue, canonical_id
from ..pipeline.stage3_judge.main_judge import run_judging
from ..pipeline.stage4_saturate import run_saturation
from ..pipeline.stage5_skeptic import run_skeptic
from ..pipeline.stage6_hygiene import run_hygiene
from ..pipeline.stage7_coverage import run_coverage
from ..pipeline.stage8_report.bibliography import write_report
from ..zotero.client import get_client as get_zotero

logger = logging.getLogger(__name__)


# ── Pipeline state ────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    run_id: str
    db_path: str
    plan: dict
    output_dir: str
    judge_stats: dict
    saturation_summary: dict
    skeptic_summary: dict
    coverage: dict
    hygiene_summary: dict
    error: str | None


# ── Node: retrieval ───────────────────────────────────────────────────────────

async def node_retrieval(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db_path = state["db_path"]
    plan = state["plan"]
    logger.info("[%s] Stage 2: retrieval", run_id)

    anchors = get_anchors(db_path, run_id)
    has_anchors = bool(anchors)

    async with DBWriter(db_path) as writer:
        queue = RetrievalQueue(writer, run_id)
        await queue.start()

        # ── Load Zotero library for dedup ──────────────────────────────────
        zotero = get_zotero()
        if await zotero.is_available():
            zotero_items = await zotero.get_all_items()
            for item in zotero_items:
                pid = canonical_id(item)
                if pid:
                    await writer.execute(
                        """
                        INSERT OR IGNORE INTO zotero_items
                          (run_id, zotero_key, paper_id, collection, tags_json)
                        VALUES (?,?,?,?,?)
                        """,
                        (
                            run_id,
                            item.get("zotero_key", ""),
                            pid,
                            item.get("collection"),
                            json.dumps(item.get("tags", [])),
                        ),
                    )

        # ── Citation-first when anchors exist ──────────────────────────────
        if has_anchors:
            logger.info("[%s] Citation-first retrieval for %d anchors", run_id, len(anchors))
            anchor_s2_ids = [a.get("s2_id") for a in anchors if a.get("s2_id")]
            anchor_oa_ids = [a.get("openalex_id") for a in anchors if a.get("openalex_id")]

            if anchor_s2_ids:
                s2_neighbors = await s2_citation(anchor_s2_ids, direction="both", max_per_paper=100)
                qid = _make_qid(run_id, "citation", "semantic_scholar", "anchor_neighborhood")
                await _log_q(writer, qid, run_id, "citation", "semantic_scholar",
                             "anchor neighborhood", [], len(s2_neighbors))
                for i, paper in enumerate(s2_neighbors):
                    await queue.submit(paper, "s2:citation", qid, rank=i, hop_distance=1,
                                       references=paper.pop("_references", []),
                                       cited_by=paper.pop("_cited_by", []))

            if anchor_oa_ids:
                oa_neighbors = await oa_citation(anchor_oa_ids, direction="both", max_results=200)
                qid = _make_qid(run_id, "citation", "openalex", "anchor_neighborhood")
                await _log_q(writer, qid, run_id, "citation", "openalex",
                             "anchor neighborhood", [], len(oa_neighbors))
                for i, paper in enumerate(oa_neighbors):
                    await queue.submit(paper, "oa:citation", qid, rank=i, hop_distance=1)

        # ── Query plan: lexical + semantic + concept-translation ───────────
        query_plan = await build_query_plan(plan)
        flat_queries = flatten_queries(query_plan)
        logger.info("[%s] Issuing %d planned queries", run_id, len(flat_queries))

        retrieval_tasks = []
        for q in flat_queries:
            retrieval_tasks.append(
                _run_query(queue, writer, run_id, q, plan)
            )

        await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        await queue.stop()

    state["error"] = None
    return state


async def _run_query(
    queue: RetrievalQueue,
    writer: DBWriter,
    run_id: str,
    q: dict,
    plan: dict,
) -> None:
    family = q["family"]
    source = q["source"]
    query_text = q["query_text"]
    max_results = 100
    cfg = plan.get("depth", "deep")

    qid = _make_qid(run_id, family, source, query_text[:40])
    results: list[dict] = []

    try:
        if source == "openalex":
            if family == "lexical":
                results = await oa_keyword(query_text, max_results)
            else:
                results = await search_by_concept(query_text, max_results)
        elif source == "semantic_scholar":
            results = await s2_keyword(query_text, max_results)
        # Future: pubmed, arxiv, crossref workers (M6 seed modes)

        await _log_q(writer, qid, run_id, family, source, query_text,
                     q.get("dimensions_targeted", []), len(results))

        for i, paper in enumerate(results):
            await queue.submit(
                paper, f"{source}:{family}", qid, rank=i,
                references=paper.pop("_references", []),
                cited_by=paper.pop("_cited_by", []),
            )
    except Exception as exc:
        logger.warning("[%s] Query failed [%s/%s]: %s", run_id, source, family, exc)
        await _log_q(writer, qid, run_id, family, source, query_text,
                     q.get("dimensions_targeted", []), 0, status="error")


# ── Node: judging ─────────────────────────────────────────────────────────────

async def node_judging(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db_path = state["db_path"]
    plan = state["plan"]
    logger.info("[%s] Stage 3: judging", run_id)

    conn = open_db(db_path)
    try:
        stats = await run_judging(db_path, run_id, plan, conn)
    finally:
        conn.close()

    # Validate SQL invariants before proceeding to report
    violations = run_sql_invariants(db_path, run_id)
    if violations:
        logger.warning("[%s] SQL invariant violations: %s", run_id, violations)

    state["judge_stats"] = stats
    return state


# ── Node: saturation (Stage 4) ────────────────────────────────────────────────

async def node_saturation(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    summary = await run_saturation(state["db_path"], run_id, state["plan"])
    state["saturation_summary"] = summary
    return state


# ── Node: skeptic (Stage 5) ───────────────────────────────────────────────────

async def node_skeptic(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    try:
        summary = await run_skeptic(state["db_path"], run_id, state["plan"])
    except Exception as exc:
        logger.warning("[%s] Skeptic stage failed: %s", run_id, exc)
        summary = {"flagged": 0, "total_sampled": 0, "overturn_rate": None, "skipped": True}
    state["skeptic_summary"] = summary
    return state


# ── Node: coverage (Stage 7) ──────────────────────────────────────────────────

async def node_coverage(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    coverage = await run_coverage(
        state["db_path"], run_id, state["plan"],
        saturation_summary=state.get("saturation_summary", {}),
        skeptic_summary=state.get("skeptic_summary", {"skipped": True}),
    )
    state["coverage"] = coverage
    return state


# ── Node: hygiene (Stage 6) ───────────────────────────────────────────────────

async def node_hygiene(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    try:
        summary = await run_hygiene(state["db_path"], run_id, state["plan"])
    except Exception as exc:
        logger.warning("[%s] Hygiene stage failed: %s", run_id, exc)
        summary = {"checked": 0, "retracted": 0, "errata": 0, "oa_resolved": 0,
                   "errors": 1, "skipped": True}
    state["hygiene_summary"] = summary
    return state


# ── Node: report ──────────────────────────────────────────────────────────────

async def node_report(state: PipelineState) -> PipelineState:
    run_id = state["run_id"]
    db_path = state["db_path"]
    plan = state["plan"]
    output_dir = state["output_dir"]
    logger.info("[%s] Stage 8: report", run_id)

    rows = fetch_bibliography(db_path, run_id)

    # M7: gap-list synthesis (LLM call). Failures are non-fatal — we still
    # write the bibliography.
    try:
        from ..pipeline.stage8_report.gap_list import generate_gap_list
        gaps = await generate_gap_list(db_path, run_id, plan)
    except Exception as exc:
        logger.warning("[%s] Gap-list generation failed: %s", run_id, exc)
        gaps = {"gaps": []}

    write_report(
        output_dir=output_dir,
        rows=rows,
        plan=plan,
        run_id=run_id,
        judge_stats=state.get("judge_stats", {}),
        coverage=state.get("coverage"),
        hygiene=state.get("hygiene_summary"),
        db_path=db_path,
        gaps=gaps,
    )

    # Mark run as done
    conn = open_db(db_path)
    conn.execute(
        "UPDATE runs SET status = 'done', finished_at = ? WHERE run_id = ?",
        (datetime.now(timezone.utc).isoformat(), run_id),
    )
    conn.commit()
    conn.close()

    return state


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(PipelineState)
    g.add_node("retrieval",  node_retrieval)
    g.add_node("judging",    node_judging)
    g.add_node("saturation", node_saturation)
    g.add_node("skeptic",    node_skeptic)
    g.add_node("coverage",   node_coverage)
    g.add_node("hygiene",    node_hygiene)
    g.add_node("report",     node_report)

    g.add_edge(START,        "retrieval")
    g.add_edge("retrieval",  "judging")
    g.add_edge("judging",    "saturation")
    g.add_edge("saturation", "skeptic")
    g.add_edge("skeptic",    "coverage")
    g.add_edge("coverage",   "hygiene")
    g.add_edge("hygiene",    "report")
    g.add_edge("report",     END)

    return g.compile()


# ── Utilities ─────────────────────────────────────────────────────────────────

def _make_qid(run_id: str, family: str, source: str, hint: str) -> str:
    import hashlib
    raw = f"{run_id}:{family}:{source}:{hint}"
    return "q" + hashlib.sha1(raw.encode()).hexdigest()[:12]


async def _log_q(
    writer: DBWriter,
    query_id: str,
    run_id: str,
    family: str,
    source: str,
    query_text: str,
    dimensions: list[str],
    result_count: int,
    status: str = "ok",
) -> None:
    from ..candidates.db import log_query
    await log_query(writer, query_id, run_id, family, source, query_text,
                    dimensions, result_count, status)
