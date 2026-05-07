"""
M5 — Daily-digest mode.

A saved search is a `(name, plan_json, cadence)` triple. Running a digest
re-uses the previously approved plan, restricts the retrieval window to
publications since the last run, and produces a short Markdown digest
instead of a full annotated bibliography.

What changes vs. a deep run
---------------------------
- Stage 0 (plan mode) is skipped entirely; the saved plan is reused.
- Stage 2 retrieval inherits a `scope.date_from` of `last_run_at` (or the
  plan's own date_from, whichever is later).
- Stage 4 saturation is shallow (single iteration) — we are not trying
  to canvas the whole field again, just to surface what's new.
- Stage 5 skeptic is opt-in; for the common case the digest just lists
  what passed the main judge.
- Stage 7 coverage is skipped (saturation curve is too short to fit).
- Output is `digest.md` listing only papers promoted above CUT during
  this incremental window, with their level + confidence + summary.

CRUD on saved_searches
----------------------
Saved searches live in the same SQLite DB as runs. The schema is
identical to the rest (see schema.sql `saved_searches`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ulid

from .candidates.db import open_db, read_conn

logger = logging.getLogger(__name__)


# ── Saved-search CRUD ─────────────────────────────────────────────────────────

@dataclass
class SavedSearch:
    search_id: str
    name: str
    plan: dict
    cadence: str
    enabled: bool
    created_at: str
    last_run_at: str | None


def save_search(
    db_path: str,
    name: str,
    plan: dict,
    cadence: str = "daily",
) -> SavedSearch:
    """Persist a new saved search. Returns the SavedSearch."""
    if cadence not in {"daily", "weekly"}:
        raise ValueError(f"unsupported cadence: {cadence!r}")

    sid = str(ulid.ULID())
    now = datetime.now(timezone.utc).isoformat()
    conn = open_db(db_path)
    conn.execute(
        """INSERT INTO saved_searches
             (search_id, name, plan_json, created_at, last_run_at, cadence, enabled)
           VALUES (?,?,?,?,?,?,1)""",
        (sid, name, json.dumps(plan), now, None, cadence),
    )
    conn.commit()
    conn.close()
    return SavedSearch(
        search_id=sid, name=name, plan=plan, cadence=cadence,
        enabled=True, created_at=now, last_run_at=None,
    )


def delete_search(db_path: str, search_id: str) -> bool:
    conn = open_db(db_path)
    cur = conn.execute("DELETE FROM saved_searches WHERE search_id = ?", (search_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted > 0


def list_searches(db_path: str) -> list[SavedSearch]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM saved_searches ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_search(r) for r in rows]


def get_search(db_path: str, search_id: str) -> SavedSearch | None:
    with read_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM saved_searches WHERE search_id = ?", (search_id,)
        ).fetchone()
    return _row_to_search(row) if row else None


def _row_to_search(row) -> SavedSearch:
    return SavedSearch(
        search_id=row["search_id"],
        name=row["name"],
        plan=json.loads(row["plan_json"]),
        cadence=row["cadence"],
        enabled=bool(row["enabled"]),
        created_at=row["created_at"],
        last_run_at=row["last_run_at"],
    )


def mark_run(db_path: str, search_id: str, when: str | None = None) -> None:
    when = when or datetime.now(timezone.utc).isoformat()
    conn = open_db(db_path)
    conn.execute(
        "UPDATE saved_searches SET last_run_at = ? WHERE search_id = ?",
        (when, search_id),
    )
    conn.commit()
    conn.close()


# ── Plan adaptation for incremental retrieval ────────────────────────────────

def make_incremental_plan(plan: dict, last_run_at: str | None) -> dict:
    """
    Return a copy of `plan` whose `scope.date_from` is the latest of:
      - the plan's own existing scope.date_from
      - the saved-search last_run_at (truncated to YYYY-MM-DD)
    so retrieval only considers publications since the last digest.
    """
    if not last_run_at:
        return plan

    last_date = last_run_at[:10]  # YYYY-MM-DD prefix
    incr = json.loads(json.dumps(plan))  # deep copy via JSON
    scope = incr.setdefault("scope", {})
    existing = scope.get("date_from") or ""
    scope["date_from"] = max(existing, last_date)
    return incr


# ── Digest writing ────────────────────────────────────────────────────────────

def write_digest_md(
    output_path: str,
    search: SavedSearch,
    rows: list[dict],
    judge_stats: dict,
    incremental_from: str | None,
) -> None:
    """
    Render a short Markdown digest. Rows are the kept papers from the
    incremental run (level != CUT).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Daily digest — {search.name}",
        f"",
        f"**Search ID**: `{search.search_id}`  ",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Window**: papers published since {incremental_from or '(initial run)'}",
        f"",
        f"## Counts",
        f"- 🔴 CORE: {judge_stats.get('CORE', 0)}",
        f"- 🟠 SUPPORTING: {judge_stats.get('SUPPORTING', 0)}",
        f"- 🟡 CONTEXT: {judge_stats.get('CONTEXT', 0)}",
        f"- 🔵 ADJACENT: {judge_stats.get('ADJACENT', 0)}",
        f"",
    ]

    if not rows:
        lines += [f"_No new papers passed the judge in this window._"]
        out.write_text("\n".join(lines))
        return

    levels = ["CORE", "SUPPORTING", "CONTEXT", "ADJACENT"]
    by_level: dict[str, list[dict]] = {lv: [] for lv in levels}
    for r in rows:
        if r.get("level") in by_level:
            by_level[r["level"]].append(r)

    for level in levels:
        if not by_level[level]:
            continue
        lines += [f"## {level}", f""]
        for row in by_level[level]:
            title = row.get("title") or "(no title)"
            year = row.get("year") or "n.d."
            doi = row.get("doi")
            link = f" · [DOI](https://doi.org/{doi})" if doi else ""
            confidence = row.get("judge_confidence")
            conf = f" · {confidence:.0%}" if confidence is not None else ""
            authors = _first_author_string(row.get("authors_json"))
            evidence = (row.get("evidence_span") or "").strip()
            lines += [f"- **{title}** — {authors} ({year}){conf}{link}"]
            if evidence:
                lines += [f"  > {evidence}"]
        lines.append("")

    out.write_text("\n".join(lines))


def _first_author_string(authors_json: str | None) -> str:
    if not authors_json:
        return "Unknown"
    try:
        authors = json.loads(authors_json)
    except json.JSONDecodeError:
        return "Unknown"
    if not authors:
        return "Unknown"
    return authors[0] + (" et al." if len(authors) > 1 else "")
