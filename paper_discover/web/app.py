"""
M8 — FastAPI web app.

Wraps the same Python pipeline as the CLI. Offers a tiny HTML dashboard
plus a JSON API that other tools (or M9's MCP server) can call.

Endpoints
---------
  GET  /                           HTML dashboard (run list)
  GET  /api/runs                   list runs (most recent first)
  GET  /api/runs/{run_id}          run details + coverage
  GET  /api/runs/{run_id}/papers   bibliography (kept papers)
  GET  /api/runs/{run_id}/concept_map  concept-map JSON
  GET  /api/runs/{run_id}/prisma   PRISMA counts
  GET  /api/saved_searches         list saved searches
  POST /api/saved_searches         create a saved search
  DELETE /api/saved_searches/{id}  delete a saved search

The DB path is fixed at startup via the PAPER_DISCOVER_DB env var (or
the `db_path` argument to `make_app`). One process serves one DB.

Why this is small
-----------------
The CLI is the source of truth for running the pipeline. The web app
exists so reports, history, and saved-searches can be browsed without
shell access — and so an MCP server (M9) has a stable HTTP surface to
call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


try:
    from pydantic import BaseModel

    class SavedSearchIn(BaseModel):
        name: str
        plan: dict
        cadence: str = "daily"
except ImportError:  # pragma: no cover
    SavedSearchIn = None  # type: ignore


def make_app(db_path: str | None = None):
    """
    Build and return a FastAPI app. FastAPI is imported lazily so the
    package as a whole doesn't require it.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is not installed. Run `pip install -e '.[web]'` to enable "
            "the web app."
        ) from exc

    from ..candidates.db import open_db, read_conn
    from ..candidates.queries import list_runs as _list_runs, get_run
    from ..digest import (
        delete_search, list_searches, save_search,
    )
    from ..pipeline.stage8_report.concept_map import build_concept_map
    from ..pipeline.stage8_report.prisma import collect_prisma_counts

    db_path = db_path or os.environ.get("PAPER_DISCOVER_DB", "run.db")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app):
        # Ensure DB is initialised so reads don't trip schema errors.
        if not Path(db_path).exists():
            logger.info("DB %s not found; initialising.", db_path)
            open_db(db_path).close()
        yield

    app = FastAPI(
        title="paper-discover",
        description="Calibrated multi-agent literature discovery.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # ── HTML dashboard ───────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        runs = _list_runs(db_path)
        rows_html = "".join(
            f"<tr><td><code>{r['run_id'][:12]}</code></td>"
            f"<td>{r['mode']}</td>"
            f"<td>{r['status']}</td>"
            f"<td>{(r.get('started_at') or '')[:16].replace('T',' ')}</td>"
            f"<td>{(r['coverage_p']*100):.0f}%</td>"
            "</tr>"
            for r in runs if r.get("coverage_p") is not None
        ) or '<tr><td colspan="5"><em>No runs yet.</em></td></tr>'
        return _DASHBOARD_HTML.format(rows=rows_html, db=db_path)

    # ── Runs ─────────────────────────────────────────────────────────────────

    @app.get("/api/runs")
    def api_list_runs():
        return {"runs": _list_runs(db_path)}

    @app.get("/api/runs/{run_id}")
    def api_get_run(run_id: str):
        run = get_run(db_path, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        # Load coverage signals if present
        with read_conn(db_path) as conn:
            cov = conn.execute(
                "SELECT * FROM coverage_signals WHERE run_id = ?", (run_id,)
            ).fetchone()
        run["coverage_signals"] = dict(cov) if cov else None
        return run

    @app.get("/api/runs/{run_id}/papers")
    def api_get_papers(run_id: str, level: str | None = None):
        sql = (
            "SELECT * FROM v_bibliography WHERE run_id = ?"
            + (" AND level = ?" if level else "")
        )
        params: tuple = (run_id, level) if level else (run_id,)
        with read_conn(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return {"run_id": run_id, "papers": [dict(r) for r in rows]}

    @app.get("/api/runs/{run_id}/concept_map")
    def api_concept_map(run_id: str):
        return build_concept_map(db_path, run_id)

    @app.get("/api/runs/{run_id}/prisma")
    def api_prisma(run_id: str):
        return collect_prisma_counts(db_path, run_id)

    # ── Saved searches ───────────────────────────────────────────────────────

    @app.get("/api/saved_searches")
    def api_list_saved():
        return {"searches": [s.__dict__ for s in list_searches(db_path)]}

    @app.post("/api/saved_searches", status_code=201)
    def api_create_saved(body: SavedSearchIn):
        try:
            s = save_search(db_path, name=body.name, plan=body.plan, cadence=body.cadence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return s.__dict__

    @app.delete("/api/saved_searches/{search_id}")
    def api_delete_saved(search_id: str):
        if not delete_search(db_path, search_id):
            raise HTTPException(status_code=404, detail="not found")
        return {"deleted": search_id}

    # ── Health ───────────────────────────────────────────────────────────────

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "db": db_path}

    return app


# ── Dashboard HTML (kept simple to avoid template-engine deps) ───────────────

_DASHBOARD_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>paper-discover dashboard</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 900px;
          margin: 32px auto; padding: 0 16px; color: #111; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ color: #666; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px;
           font-size: 14px; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #eee; text-align: left; }}
  th {{ background: #fafafa; }}
  code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
  .nav {{ margin-top: 16px; font-size: 13px; }}
  .nav a {{ margin-right: 12px; }}
</style>
</head><body>
<h1>paper-discover</h1>
<div class="sub">DB: <code>{db}</code></div>
<div class="nav">
  <a href="/api/runs">/api/runs</a>
  <a href="/api/saved_searches">/api/saved_searches</a>
  <a href="/healthz">/healthz</a>
</div>
<table>
  <thead><tr><th>Run ID</th><th>Mode</th><th>Status</th><th>Started</th><th>Coverage</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>
"""
