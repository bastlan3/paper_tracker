"""
Stage 0 — Plan mode: interactive terminal dialogue that produces an approved
plan artifact (plan.json) before any retrieval starts.

Flow:
  1. Parse seed (NL text, anchor IDs, Zotero collection key, or structured dict)
  2. Planner LLM drafts an intent object
  3. Clarification loop (up to 5 questions, user may skip each with Enter)
  4. Present plan for approval / edit
  5. Save to plan.json

For daily-digest runs Stage 0 is skipped; the saved plan is reused directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from ..models.structured_output import CLARIFICATIONS_SCHEMA, PLAN_SCHEMA
from ..models.vllm_client import get_client
from ..zotero.client import get_client as get_zotero

logger = logging.getLogger(__name__)
console = Console()

_PLANNER_PROMPT = (Path(__file__).parent.parent / "prompts" / "planner.txt").read_text()
_CLARIF_PROMPT = (Path(__file__).parent.parent / "prompts" / "clarification.txt").read_text()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_plan_mode(
    seed: str | None = None,
    anchor_ids: list[str] | None = None,
    zotero_collection: str | None = None,
    out_path: str = "plan.json",
) -> dict:
    """
    Interactive plan mode. Returns the approved plan dict and writes it to out_path.
    Raises SystemExit if the user aborts.
    """
    console.print(Panel("[bold cyan]paper-discover — plan mode[/]", expand=False))

    # ── Collect seed ──────────────────────────────────────────────────────────
    if seed is None and not anchor_ids and not zotero_collection:
        seed = Prompt.ask("[bold]Describe your research question or topic[/]")

    anchor_meta: list[dict] = []
    if anchor_ids:
        anchor_meta = await _resolve_anchors(anchor_ids)
        if not anchor_meta:
            console.print("[yellow]⚠ Could not resolve any anchor IDs; proceeding without anchors.[/]")

    zotero_items: list[dict] = []
    if zotero_collection:
        zotero_items = await _fetch_zotero_collection(zotero_collection)
        if zotero_items:
            console.print(f"[dim]Loaded {len(zotero_items)} items from Zotero collection.[/]")

    # ── Draft plan ────────────────────────────────────────────────────────────
    console.print("[dim]Drafting plan…[/]")
    plan = await _draft_plan(seed or "", anchor_meta, zotero_items)

    # ── Clarification loop ────────────────────────────────────────────────────
    plan = await _clarification_loop(plan)

    # ── Present and approve ───────────────────────────────────────────────────
    while True:
        _display_plan(plan)
        action = Prompt.ask(
            "[bold]Approve plan?[/]",
            choices=["y", "edit", "abort"],
            default="y",
        )
        if action == "abort":
            console.print("[red]Aborted.[/]")
            raise SystemExit(1)
        if action == "y":
            break
        plan = await _edit_plan(plan)

    # ── Stamp and save ────────────────────────────────────────────────────────
    plan["_approved_at"] = datetime.now(timezone.utc).isoformat()
    plan["_anchor_ids"] = anchor_ids or []

    Path(out_path).write_text(json.dumps(plan, indent=2))
    console.print(f"[green]✓ Plan saved to {out_path}[/]")
    return plan


# ── Draft plan ────────────────────────────────────────────────────────────────

async def _draft_plan(
    seed: str,
    anchors: list[dict],
    zotero_items: list[dict],
) -> dict:
    anchor_block = _format_anchors(anchors)

    # If we have Zotero items and no explicit seed, let the planner infer theme
    if not seed and zotero_items:
        seed = (
            "The user has provided a Zotero collection. "
            "Infer the research theme from the titles and abstracts below:\n"
            + "\n".join(
                f"- {it['title']} ({it.get('year', 'n.d.')})"
                for it in zotero_items[:30]
            )
        )

    messages = [
        {"role": "user", "content": _PLANNER_PROMPT.format(seed=seed, anchors=anchor_block)},
    ]
    llm = get_client()
    return await llm.plan_json(messages, PLAN_SCHEMA)


# ── Clarification loop ────────────────────────────────────────────────────────

async def _clarification_loop(plan: dict) -> dict:
    llm = get_client()
    messages = [
        {
            "role": "user",
            "content": _CLARIF_PROMPT.format(plan_json=json.dumps(plan, indent=2)),
        }
    ]
    result = await llm.plan_json(messages, CLARIFICATIONS_SCHEMA)
    questions = result.get("questions", [])

    if not questions:
        return plan

    console.print("\n[bold]A few clarifying questions (press Enter to accept defaults):[/]")
    answers: list[str] = []
    for item in questions:
        answer = Prompt.ask(
            f"  {item['question']}",
            default=item.get("default", ""),
        )
        answers.append(answer.strip())

    if any(answers):
        refinement_prompt = (
            f"Refine this search plan based on the user's answers.\n\n"
            f"Original plan:\n{json.dumps(plan, indent=2)}\n\n"
            f"Questions and answers:\n"
            + "\n".join(
                f"Q: {q['question']}\nA: {a}"
                for q, a in zip(questions, answers)
                if a
            )
            + "\n\nOutput the updated plan JSON only."
        )
        messages = [{"role": "user", "content": refinement_prompt}]
        plan = await llm.plan_json(messages, PLAN_SCHEMA)

    return plan


# ── Display ───────────────────────────────────────────────────────────────────

def _display_plan(plan: dict) -> None:
    dims = plan.get("dimensions", [])
    dim_lines = "\n".join(
        f"  {'★' if d.get('critical') else '•'} **{d['name']}**: {d['value']}"
        f"{' _(essential)_' if d.get('essential') else ''}"
        for d in dims
    )
    scope = plan.get("scope", {})
    scope_lines = "\n".join(f"  - {k}: {v}" for k, v in scope.items() if v)
    md = f"""## Intent
{plan.get('intent', '')}

## Key Concepts
{', '.join(plan.get('concepts', []))}

## Dimensions
{dim_lines}

## Scope
{scope_lines or '  (none)'}

## Depth
{plan.get('depth', 'deep')}

## Anchors
{', '.join(plan.get('anchors', [])) or '(none)'}
"""
    console.print(Panel(Markdown(md), title="[bold]Draft plan[/]", expand=False))


# ── Edit plan ─────────────────────────────────────────────────────────────────

async def _edit_plan(plan: dict) -> dict:
    console.print("[dim]Describe what to change:[/]")
    change = Prompt.ask("> ")
    llm = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                f"Update this search plan per the user's request.\n\n"
                f"Current plan:\n{json.dumps(plan, indent=2)}\n\n"
                f"Change requested: {change}\n\n"
                "Output the updated plan JSON only."
            ),
        }
    ]
    return await llm.plan_json(messages, PLAN_SCHEMA)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _resolve_anchors(anchor_ids: list[str]) -> list[dict]:
    """Fetch metadata for anchor IDs from OpenAlex / Semantic Scholar."""
    from .stage2_retrieve.openalex import fetch_paper_by_doi
    from .stage2_retrieve.semantic_scholar import fetch_paper_by_id as s2_fetch

    results: list[dict] = []
    for aid in anchor_ids:
        paper: dict | None = None
        if aid.startswith("10."):
            paper = await fetch_paper_by_doi(aid)
        elif aid.startswith("2") and len(aid) > 10:
            paper = await s2_fetch(aid)
        if paper:
            results.append(paper)
        else:
            logger.warning("Could not resolve anchor %s", aid)
    return results


async def _fetch_zotero_collection(collection_key: str) -> list[dict]:
    zotero = get_zotero()
    return await zotero.get_collection_items(collection_key)


def _format_anchors(anchors: list[dict]) -> str:
    if not anchors:
        return "(none provided)"
    lines = []
    for a in anchors:
        authors = json.loads(a.get("authors_json") or "[]")
        first_author = authors[0] if authors else "Unknown"
        lines.append(
            f"[{a.get('paper_id')}] {a.get('title', 'Unknown title')} "
            f"— {first_author} et al. ({a.get('year', 'n.d.')})\n"
            f"  Abstract: {(a.get('abstract') or '')[:300]}…"
        )
    return "\n\n".join(lines)
