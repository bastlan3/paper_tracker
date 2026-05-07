"""
M9 — MCP server.

Exposes paper-discover's read-side surface as MCP tools so other agents
can pull bibliographies, coverage estimates, and concept maps from a
local SQLite database without going through the CLI or HTTP layer.

The actual MCP runtime is a soft dependency: importing this module
without the `mcp` package installed still works — the pure
tool-handler functions stay callable so other Python code (and the
test suite) can use them without the SDK.

Design
------
Every tool is split in two:

  - A *pure handler* (`tool_<name>`) that takes structured arguments,
    returns plain JSON-serialisable Python values, and never touches
    the MCP transport.
  - A small wrapper inside `serve()` that registers the handler with
    the `mcp` SDK.

Tests exercise the handlers directly. The wiring layer is exercised
manually via `paper-discover mcp`.

Why read-side only
------------------
Triggering a long-running pipeline run over MCP is doable (return a
task ID, poll status) but introduces durability and resumability
complexity that's better solved by the CLI scheduler. The minimum
useful M9 is "let an agent see what paper-discover has found."
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ── Pure tool handlers (testable without the MCP SDK) ────────────────────────

def tool_list_runs(db_path: str) -> dict:
    """Return all runs (most recent first)."""
    from .candidates.queries import list_runs as _list_runs
    runs = _list_runs(db_path)
    return {"runs": runs}


def tool_get_run(db_path: str, run_id: str) -> dict:
    """Return run metadata + coverage signals (or null)."""
    from .candidates.db import read_conn
    from .candidates.queries import get_run

    run = get_run(db_path, run_id)
    if not run:
        return {"error": "run not found", "run_id": run_id}
    with read_conn(db_path) as conn:
        cov = conn.execute(
            "SELECT * FROM coverage_signals WHERE run_id = ?", (run_id,)
        ).fetchone()
    run["coverage_signals"] = dict(cov) if cov else None
    return run


def tool_get_bibliography(
    db_path: str,
    run_id: str,
    level: str | None = None,
) -> dict:
    """Return kept papers for a run, optionally filtered by level."""
    from .candidates.db import read_conn

    sql = "SELECT * FROM v_bibliography WHERE run_id = ?"
    params: tuple = (run_id,)
    if level:
        if level not in ("CORE", "SUPPORTING", "CONTEXT", "ADJACENT"):
            return {"error": f"invalid level: {level}"}
        sql += " AND level = ?"
        params = (run_id, level)

    with read_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {"run_id": run_id, "papers": [dict(r) for r in rows]}


def tool_get_concept_map(db_path: str, run_id: str) -> dict:
    """Return the run's kept-paper citation graph."""
    from .pipeline.stage8_report.concept_map import build_concept_map
    return build_concept_map(db_path, run_id)


def tool_get_prisma(db_path: str, run_id: str) -> dict:
    """Return PRISMA funnel counts for a run."""
    from .pipeline.stage8_report.prisma import collect_prisma_counts
    return collect_prisma_counts(db_path, run_id)


def tool_list_saved_searches(db_path: str) -> dict:
    from .digest import list_searches
    return {"searches": [s.__dict__ for s in list_searches(db_path)]}


def tool_save_plan(
    db_path: str,
    name: str,
    plan: dict,
    cadence: str = "daily",
) -> dict:
    from .digest import save_search
    try:
        s = save_search(db_path, name=name, plan=plan, cadence=cadence)
    except ValueError as exc:
        return {"error": str(exc)}
    return s.__dict__


def tool_plan_from_pico(query: dict) -> dict:
    """Build a plan dict from PICO/boolean/filter input. No LLM call."""
    from .seeds import plan_from_structured
    try:
        return plan_from_structured(query)
    except ValueError as exc:
        return {"error": str(exc)}


# ── Tool registry (used by serve() and tests) ────────────────────────────────

TOOLS: dict[str, dict] = {
    "list_runs": {
        "handler": tool_list_runs,
        "description": "List runs stored in a paper-discover SQLite database.",
        "schema": {
            "type": "object",
            "required": ["db_path"],
            "properties": {"db_path": {"type": "string"}},
        },
    },
    "get_run": {
        "handler": tool_get_run,
        "description": "Get run metadata and coverage_signals for a run_id.",
        "schema": {
            "type": "object",
            "required": ["db_path", "run_id"],
            "properties": {
                "db_path": {"type": "string"},
                "run_id":  {"type": "string"},
            },
        },
    },
    "get_bibliography": {
        "handler": tool_get_bibliography,
        "description": "List kept papers for a run, optionally filtered by level.",
        "schema": {
            "type": "object",
            "required": ["db_path", "run_id"],
            "properties": {
                "db_path": {"type": "string"},
                "run_id":  {"type": "string"},
                "level":   {"type": "string",
                            "enum": ["CORE", "SUPPORTING", "CONTEXT", "ADJACENT"]},
            },
        },
    },
    "get_concept_map": {
        "handler": tool_get_concept_map,
        "description": "Return the kept-paper citation graph for a run.",
        "schema": {
            "type": "object",
            "required": ["db_path", "run_id"],
            "properties": {
                "db_path": {"type": "string"},
                "run_id":  {"type": "string"},
            },
        },
    },
    "get_prisma": {
        "handler": tool_get_prisma,
        "description": "Return PRISMA funnel counts for a run.",
        "schema": {
            "type": "object",
            "required": ["db_path", "run_id"],
            "properties": {
                "db_path": {"type": "string"},
                "run_id":  {"type": "string"},
            },
        },
    },
    "list_saved_searches": {
        "handler": tool_list_saved_searches,
        "description": "List saved searches (for digest mode).",
        "schema": {
            "type": "object",
            "required": ["db_path"],
            "properties": {"db_path": {"type": "string"}},
        },
    },
    "save_plan": {
        "handler": tool_save_plan,
        "description": "Save an approved plan as a saved search.",
        "schema": {
            "type": "object",
            "required": ["db_path", "name", "plan"],
            "properties": {
                "db_path": {"type": "string"},
                "name":    {"type": "string"},
                "plan":    {"type": "object"},
                "cadence": {"type": "string",
                            "enum": ["daily", "weekly"]},
            },
        },
    },
    "plan_from_pico": {
        "handler": tool_plan_from_pico,
        "description": "Build a plan dict from a PICO / boolean / filter query (no LLM).",
        "schema": {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "object"}},
        },
    },
}


def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch to a registered tool handler. Used by both the MCP wrapper and tests."""
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    handler = TOOLS[name]["handler"]
    try:
        return handler(**arguments)
    except TypeError as exc:
        return {"error": f"bad arguments: {exc}"}


# ── MCP runtime wrapper ──────────────────────────────────────────────────────

def serve(default_db: str | None = None) -> None:
    """
    Start an MCP stdio server. Requires the `mcp` package; otherwise
    raises ImportError with a hint.

    The server speaks MCP's JSON-RPC over stdio and registers every
    handler in TOOLS. `default_db` is injected as the db_path argument
    when the caller omits it (so an agent doesn't need to know the
    file path).
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MCP server requires `mcp`. Run `pip install -e '.[mcp]'`."
        ) from exc

    import asyncio

    default_db = default_db or os.environ.get("PAPER_DISCOVER_DB", "run.db")
    server = Server("paper-discover")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=meta["description"],
                inputSchema=meta["schema"],
            )
            for name, meta in TOOLS.items()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        args = dict(arguments or {})
        # Inject default db_path when the caller omits it.
        if "db_path" not in args and "db_path" in TOOLS[name]["schema"].get("properties", {}):
            args["db_path"] = default_db
        result = call_tool(name, args)
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())
