"""
paper-discover CLI entry point.

Commands:
  plan    Interactive plan mode → produces plan.json
  run     Execute the full pipeline against an approved plan
  digest  (M5) Run incremental digest for saved searches
  list    Show past runs in a given database
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="paper-discover",
    help="Multi-agent literature discovery with calibrated coverage.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )


# ── plan ─────────────────────────────────────────────────────────────────────

@app.command()
def plan(
    seed: str = typer.Option(None, "--seed", "-s", help="Research question or topic (prompted if omitted)"),
    anchors: str = typer.Option(None, "--anchors", "-a", help="Comma-separated DOIs or arXiv IDs"),
    collection: str = typer.Option(None, "--collection", "-c", help="Zotero collection key"),
    out: str = typer.Option("plan.json", "--out", "-o", help="Output path for plan.json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive plan mode: dialogue with LLM → approved plan.json."""
    _setup_logging(verbose)
    from .pipeline.stage0_plan import run_plan_mode

    anchor_ids = [a.strip() for a in anchors.split(",")] if anchors else None

    asyncio.run(
        run_plan_mode(
            seed=seed,
            anchor_ids=anchor_ids,
            zotero_collection=collection,
            out_path=out,
        )
    )


# ── run ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    plan_path: str = typer.Argument(..., help="Path to the approved plan.json"),
    db: str = typer.Option(None, "--db", help="SQLite database path (default: runs/<run_id>/run.db)"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Report output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Execute the full discovery pipeline against an approved plan."""
    _setup_logging(verbose)
    asyncio.run(_run_async(plan_path, db, output_dir))


async def _run_async(
    plan_path: str,
    db_path: str | None,
    output_dir: str | None,
) -> None:
    from ulid import ULID

    plan_file = Path(plan_path)
    if not plan_file.exists():
        console.print(f"[red]Plan file not found: {plan_path}[/]")
        raise typer.Exit(1)

    plan = json.loads(plan_file.read_text())
    run_id = str(ULID())

    # Default paths
    run_dir = Path(f"runs/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_path or str(run_dir / "run.db")
    output_dir = output_dir or str(run_dir / "report")

    console.print(f"[bold cyan]Run ID:[/] {run_id}")
    console.print(f"[dim]DB: {db_path}[/]")
    console.print(f"[dim]Output: {output_dir}[/]")

    from .candidates.db import open_db
    conn = open_db(db_path)

    # Insert run record
    conn.execute(
        "INSERT INTO runs (run_id, started_at, mode, plan_json, status) VALUES (?,?,?,?,?)",
        (run_id, datetime.now(timezone.utc).isoformat(), "deep", json.dumps(plan), "running"),
    )

    # Insert anchors into DB
    anchor_ids: list[str] = plan.get("_anchor_ids") or plan.get("anchors") or []
    if anchor_ids:
        from .pipeline.stage0_plan import _resolve_anchors
        from .pipeline.stage2_retrieve.writer import canonical_id
        from .candidates.db import upsert_paper

        from .candidates.db import DBWriter
        async with DBWriter(db_path) as writer:
            anchor_papers = await _resolve_anchors(anchor_ids)
            for ap in anchor_papers:
                ap["paper_id"] = canonical_id(ap)
                ap["title_norm"] = ap.get("title", "").lower()
                from .pipeline.stage2_retrieve.writer import normalise_authors
                ap["authors"] = normalise_authors(ap.get("authors") or [])
                import json as _json
                ap["authors_json"] = _json.dumps(ap["authors"])
                ap["first_author"] = ap["authors"][0] if ap["authors"] else None
                from datetime import datetime as dt, timezone as tz
                ap["fetched_at"] = dt.now(tz.utc).isoformat()
                await upsert_paper(writer, ap)
                await writer.execute(
                    "INSERT OR IGNORE INTO run_anchors (run_id, paper_id) VALUES (?,?)",
                    (run_id, ap["paper_id"]),
                )

    conn.commit()
    conn.close()

    # Run the LangGraph pipeline
    from .pipeline.graph import build_graph

    graph = build_graph()
    initial_state: dict = {
        "run_id": run_id,
        "db_path": db_path,
        "plan": plan,
        "output_dir": output_dir,
        "judge_stats": {},
        "saturation_summary": {},
        "skeptic_summary": {"skipped": True},
        "coverage": {},
        "hygiene_summary": {},
        "error": None,
    }

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        console.print(f"[red]Pipeline error: {final_state['error']}[/]")
        raise typer.Exit(1)

    stats = final_state.get("judge_stats", {})
    console.print("\n[bold green]✓ Run complete[/]")
    console.print(f"  CORE: {stats.get('CORE', 0)}  SUPPORTING: {stats.get('SUPPORTING', 0)}"
                  f"  CONTEXT: {stats.get('CONTEXT', 0)}  ADJACENT: {stats.get('ADJACENT', 0)}"
                  f"  CUT: {stats.get('CUT', 0)}")
    cov = final_state.get("coverage") or {}
    if cov.get("coverage_p") is not None:
        console.print(
            f"  Coverage: {cov['coverage_p']:.0%}"
            f" (CI {cov.get('coverage_ci_lo', 0):.0%}–{cov.get('coverage_ci_hi', 0):.0%})"
        )
    console.print(f"  Report: [cyan]{output_dir}[/]")


# ── list ──────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_runs(
    db: str = typer.Option("run.db", "--db", help="SQLite database path"),
) -> None:
    """List past runs stored in a database."""
    from .candidates.queries import list_runs as _list_runs

    runs = _list_runs(db)
    if not runs:
        console.print("[dim]No runs found.[/]")
        return

    table = Table(title="Past runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Coverage")

    for r in runs:
        cov = f"{r['coverage_p']:.0%}" if r.get("coverage_p") else "—"
        started = (r.get("started_at") or "")[:16].replace("T", " ")
        table.add_row(r["run_id"], r["mode"], r["status"], started, cov)

    console.print(table)


# ── M5: saved-search CRUD ─────────────────────────────────────────────────────

@app.command()
def save(
    plan_path: str = typer.Argument(..., help="Path to an approved plan.json"),
    name: str = typer.Option(..., "--name", help="Human-readable name for the saved search"),
    db: str = typer.Option("digest.db", "--db", help="SQLite DB to store the saved search"),
    cadence: str = typer.Option("daily", "--cadence", help="daily | weekly"),
) -> None:
    """Persist an approved plan as a saved search for digest mode."""
    from .digest import save_search

    pf = Path(plan_path)
    if not pf.exists():
        console.print(f"[red]Plan not found: {plan_path}[/]")
        raise typer.Exit(1)
    plan = json.loads(pf.read_text())

    s = save_search(db, name=name, plan=plan, cadence=cadence)
    console.print(f"[green]Saved:[/] {s.name} ({s.search_id})")
    console.print(f"  cadence: {s.cadence}")
    console.print(f"  db: {db}")


@app.command()
def searches(
    db: str = typer.Option("digest.db", "--db"),
) -> None:
    """List saved searches."""
    from .digest import list_searches

    items = list_searches(db)
    if not items:
        console.print("[dim]No saved searches.[/]")
        return

    table = Table(title="Saved searches")
    table.add_column("Search ID", style="cyan")
    table.add_column("Name")
    table.add_column("Cadence")
    table.add_column("Last run")
    for s in items:
        last = (s.last_run_at or "—")[:16].replace("T", " ")
        table.add_row(s.search_id, s.name, s.cadence, last)
    console.print(table)


@app.command(name="forget")
def forget_search(
    search_id: str = typer.Argument(...),
    db: str = typer.Option("digest.db", "--db"),
) -> None:
    """Delete a saved search."""
    from .digest import delete_search

    if delete_search(db, search_id):
        console.print(f"[green]Deleted[/] {search_id}")
    else:
        console.print(f"[red]Not found:[/] {search_id}")
        raise typer.Exit(1)


# ── M5: digest run ────────────────────────────────────────────────────────────

@app.command()
def digest(
    search_id: str = typer.Argument(None, help="Saved search ID (omit to run all enabled)"),
    db: str = typer.Option("digest.db", "--db", help="DB containing saved searches"),
    out: str = typer.Option(None, "--out", help="Output directory for digests"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run an incremental digest for one or all saved searches."""
    _setup_logging(verbose)
    asyncio.run(_digest_async(search_id, db, out))


async def _digest_async(search_id: str | None, db_path: str, out: str | None) -> None:
    from .digest import (
        get_search, list_searches, make_incremental_plan, mark_run,
        write_digest_md,
    )

    targets = (
        [get_search(db_path, search_id)] if search_id
        else [s for s in list_searches(db_path) if s.enabled]
    )
    targets = [t for t in targets if t]
    if not targets:
        console.print("[yellow]No saved searches to run.[/]")
        return

    for search in targets:
        console.print(f"[bold cyan]Digest:[/] {search.name} ({search.search_id})")
        plan = make_incremental_plan(search.plan, search.last_run_at)
        await _run_digest_pipeline(search, plan, db_path, out)
        mark_run(db_path, search.search_id)


async def _run_digest_pipeline(
    search: "SavedSearch",  # noqa: F821
    plan: dict,
    db_path: str,
    out: str | None,
) -> None:
    """
    Minimal incremental pipeline: retrieval → judging → write digest.
    Saturation, skeptic, coverage, and hygiene are all skipped — digest
    mode is meant to be cheap and fast.
    """
    from ulid import ULID

    from .candidates.db import open_db
    from .candidates.queries import zotero_paper_ids  # noqa: F401  (sanity)
    from .digest import write_digest_md
    from .pipeline.graph import node_judging, node_retrieval

    run_id = str(ULID())
    run_dir = Path(out or f"digest_runs/{search.search_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_db = str(run_dir / "run.db")
    digest_md = str(run_dir / f"{run_id}.md")

    # Insert the run row so foreign keys are satisfied.
    conn = open_db(run_db)
    conn.execute(
        "INSERT INTO runs (run_id, started_at, mode, plan_json, status) VALUES (?,?,?,?,?)",
        (run_id, datetime.now(timezone.utc).isoformat(),
         "digest", json.dumps(plan), "running"),
    )
    conn.commit()
    conn.close()

    state: dict = {
        "run_id": run_id,
        "db_path": run_db,
        "plan": plan,
        "output_dir": str(run_dir),
        "judge_stats": {},
        "saturation_summary": {},
        "skeptic_summary": {"skipped": True},
        "coverage": {},
        "hygiene_summary": {"skipped": True},
        "error": None,
    }

    state = await node_retrieval(state)
    state = await node_judging(state)

    # Pull kept rows directly for the digest writer.
    from .candidates.db import fetch_bibliography
    rows = fetch_bibliography(run_db, run_id)

    write_digest_md(
        output_path=digest_md,
        search=search,
        rows=rows,
        judge_stats=state.get("judge_stats", {}),
        incremental_from=plan.get("scope", {}).get("date_from"),
    )

    conn = open_db(run_db)
    conn.execute(
        "UPDATE runs SET status='done', finished_at=? WHERE run_id=?",
        (datetime.now(timezone.utc).isoformat(), run_id),
    )
    conn.commit()
    conn.close()

    console.print(f"  [green]→[/] {digest_md}")


if __name__ == "__main__":
    app()
