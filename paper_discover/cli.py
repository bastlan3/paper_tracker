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


# ── digest (stub for M5) ──────────────────────────────────────────────────────

@app.command()
def digest(
    search_id: str = typer.Option(None, "--search-id", help="Saved search ID (all if omitted)"),
    db: str = typer.Option("digest.db", "--db"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run incremental digest for saved searches. (Implemented in M5)"""
    _setup_logging(verbose)
    console.print("[yellow]Digest mode not yet implemented (M5).[/]")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
