"""
Stage 6 — Hygiene pass.

Runs after coverage and before the report. For every kept paper
(CORE / SUPPORTING / CONTEXT / ADJACENT) we:

  1. Check Crossref's relation feed for retractions and errata that
     reference the DOI (Retraction Watch is exposed CC-BY through the
     Crossref REST API — FLAG F8 attribution string is added to the
     report by stage8_report).
  2. Resolve an open-access URL via Unpaywall, falling back to Europe
     PMC's OA mirror search.

Decisions are written back to the global `papers` table so that all runs
referring to a given DOI benefit. `flags_json` on the run-scoped
`candidates` row gains a "retracted" flag where applicable.

NEVER cuts a paper for being retracted. The level is preserved with a
loud RETRACTED flag in the bibliography (the user can decide whether to
include it). FLAG F8 attribution is rendered in summary.md by Stage 8.

Network failure handling
------------------------
Every external call has bounded retries and a short timeout. On total
failure the paper is left as-is — Stage 6 is best-effort and never
blocks the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..candidates.db import open_db, read_conn

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────

_CROSSREF_BASE = "https://api.crossref.org/works"
_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_HTTP_TIMEOUT = 10.0
_HTTP_RETRIES = 2
_CONCURRENCY = 8  # max simultaneous HTTP requests


# ── Public entry point ────────────────────────────────────────────────────────

async def run_hygiene(
    db_path: str,
    run_id: str,
    plan: dict | None = None,
) -> dict:
    """
    Update kept papers with retraction/errata/OA info. Returns a summary:
      checked, retracted, errata, oa_resolved, errors
    """
    logger.info("[%s] Stage 6: hygiene pass", run_id)

    rows = _kept_with_doi(db_path, run_id)
    if not rows:
        logger.info("[%s] No kept papers with DOIs; hygiene skipped", run_id)
        return {"checked": 0, "retracted": 0, "errata": 0, "oa_resolved": 0, "errors": 0}

    summary = {"checked": 0, "retracted": 0, "errata": 0, "oa_resolved": 0, "errors": 0}
    sem = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        async def _process(row: dict) -> dict:
            async with sem:
                return await _check_one(client, row)

        results = await asyncio.gather(
            *[_process(r) for r in rows], return_exceptions=True
        )

    conn = open_db(db_path)
    try:
        for row, result in zip(rows, results):
            summary["checked"] += 1
            if isinstance(result, Exception):
                summary["errors"] += 1
                logger.warning("[%s] hygiene error for %s: %s", run_id, row["paper_id"], result)
                continue
            updated = _persist(conn, run_id, row, result)
            if updated.get("retracted"):
                summary["retracted"] += 1
            if updated.get("errata"):
                summary["errata"] += 1
            if updated.get("oa_url"):
                summary["oa_resolved"] += 1
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "[%s] Hygiene done: %d checked, %d retracted, %d errata, %d OA resolved, %d errors",
        run_id, summary["checked"], summary["retracted"], summary["errata"],
        summary["oa_resolved"], summary["errors"],
    )
    return summary


# ── Per-paper check ──────────────────────────────────────────────────────────

async def _check_one(client: httpx.AsyncClient, row: dict) -> dict:
    """
    Look up retraction, errata, and OA URL for a single paper.
    Returns a dict with optional keys: retracted (bool), retraction_notice (dict),
    errata (list[dict]), oa_url (str).
    """
    doi = (row.get("doi") or "").strip().lower()
    if not doi:
        return {}

    out: dict[str, Any] = {}

    # Crossref work record holds retraction relations
    cr = await _fetch_crossref(client, doi)
    if cr:
        retraction = parse_retraction(cr)
        if retraction:
            out["retracted"] = True
            out["retraction_notice"] = retraction
        errata = parse_errata(cr)
        if errata:
            out["errata"] = errata

    # OA URL: prefer Unpaywall (more authoritative for green OA)
    if not row.get("oa_url"):
        oa = await _fetch_unpaywall(client, doi)
        oa_url = parse_unpaywall_oa(oa) if oa else None
        if not oa_url:
            ep = await _fetch_europepmc(client, doi)
            oa_url = parse_europepmc_oa(ep) if ep else None
        if oa_url:
            out["oa_url"] = oa_url

    return out


# ── Crossref ─────────────────────────────────────────────────────────────────

async def _fetch_crossref(client: httpx.AsyncClient, doi: str) -> dict | None:
    url = f"{_CROSSREF_BASE}/{doi}"
    headers = {"User-Agent": "paper-discover/0.1 (mailto:" + os.environ.get(
        "OPENALEX_EMAIL", "researcher@example.com") + ")"}
    return await _get_json(client, url, headers=headers)


def parse_retraction(crossref_work: dict) -> dict | None:
    """
    Crossref encodes retractions as a `relation` of type
    'is-retraction-of' or via `update-to` entries with type 'retraction'.
    Returns the retraction notice dict, or None.
    """
    msg = crossref_work.get("message") or crossref_work
    # Variant A: top-level relation map
    relation = msg.get("relation") or {}
    for rtype in ("is-retraction-of", "is-retracted-by"):
        rels = relation.get(rtype) or []
        if rels:
            return {
                "source": "crossref.relation",
                "type": rtype,
                "details": rels,
            }
    # Variant B: update-to list (more common for the retracted side)
    for upd in msg.get("update-to") or []:
        if (upd.get("type") or "").lower() == "retraction":
            return {
                "source": "crossref.update-to",
                "type": "retraction",
                "details": upd,
            }
    return None


def parse_errata(crossref_work: dict) -> list[dict]:
    msg = crossref_work.get("message") or crossref_work
    out: list[dict] = []
    relation = msg.get("relation") or {}
    for rtype in ("has-correction", "is-corrected-by", "has-erratum"):
        rels = relation.get(rtype) or []
        for r in rels:
            out.append({"type": rtype, **r})
    for upd in msg.get("update-to") or []:
        if (upd.get("type") or "").lower() in ("correction", "erratum"):
            out.append({"type": upd.get("type"), **upd})
    return out


# ── Unpaywall ────────────────────────────────────────────────────────────────

async def _fetch_unpaywall(client: httpx.AsyncClient, doi: str) -> dict | None:
    email = os.environ.get("UNPAYWALL_EMAIL")
    if not email:
        return None  # Unpaywall requires a contact email; skip rather than 422
    url = f"{_UNPAYWALL_BASE}/{doi}"
    return await _get_json(client, url, params={"email": email})


def parse_unpaywall_oa(unpaywall: dict) -> str | None:
    best = unpaywall.get("best_oa_location") or {}
    for key in ("url_for_pdf", "url"):
        if best.get(key):
            return best[key]
    for loc in unpaywall.get("oa_locations") or []:
        for key in ("url_for_pdf", "url"):
            if loc.get(key):
                return loc[key]
    return None


# ── Europe PMC (preprint + OA mirror routing) ────────────────────────────────

async def _fetch_europepmc(client: httpx.AsyncClient, doi: str) -> dict | None:
    return await _get_json(
        client, _EUROPEPMC_BASE,
        params={"query": f"DOI:{doi}", "format": "json", "resultType": "core"},
    )


def parse_europepmc_oa(epmc: dict) -> str | None:
    res = (epmc.get("resultList") or {}).get("result") or []
    if not res:
        return None
    r = res[0]
    if r.get("isOpenAccess") == "Y":
        for link in (r.get("fullTextUrlList") or {}).get("fullTextUrl") or []:
            if link.get("availability", "").startswith("Open access") and link.get("url"):
                return link["url"]
        if r.get("pmcid"):
            return f"https://europepmc.org/article/PMC/{r['pmcid'].replace('PMC','')}"
    return None


# ── HTTP helper ──────────────────────────────────────────────────────────────

async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
) -> dict | None:
    backoff = 0.5
    for attempt in range(_HTTP_RETRIES + 1):
        try:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (404, 422):
                return None  # not found / bad request — don't retry
            if r.status_code == 429:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return None
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == _HTTP_RETRIES:
                return None
            await asyncio.sleep(backoff)
            backoff *= 2
    return None


# ── DB helpers ───────────────────────────────────────────────────────────────

def _kept_with_doi(db_path: str, run_id: str) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.flags_json, p.doi, p.oa_url, p.retracted
            FROM candidates c
            JOIN papers p USING (paper_id)
            WHERE c.run_id = ?
              AND c.level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
              AND p.doi IS NOT NULL AND p.doi != ''
            """,
            (run_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _persist(conn, run_id: str, row: dict, update: dict) -> dict:
    """Apply hygiene update to papers + candidates. Returns the dict of fields actually changed."""
    if not update:
        return {}

    paper_id = row["paper_id"]
    changed: dict[str, Any] = {}

    if update.get("retracted") and not row.get("retracted"):
        conn.execute(
            "UPDATE papers SET retracted = 1, retraction_notice_json = ? WHERE paper_id = ?",
            (json.dumps(update.get("retraction_notice") or {}), paper_id),
        )
        # Add retracted to candidates.flags_json (idempotent)
        flags = json.loads(row.get("flags_json") or "[]")
        if "retracted" not in flags:
            flags.append("retracted")
            conn.execute(
                "UPDATE candidates SET flags_json = ? WHERE run_id = ? AND paper_id = ?",
                (json.dumps(flags), run_id, paper_id),
            )
        changed["retracted"] = True

    if update.get("errata"):
        # Errata is informational; we stash it in retraction_notice_json under a
        # different key only if there's no actual retraction notice.
        if not row.get("retracted") and not update.get("retracted"):
            conn.execute(
                "UPDATE papers SET retraction_notice_json = ? WHERE paper_id = ?",
                (json.dumps({"errata": update["errata"]}), paper_id),
            )
        changed["errata"] = True

    if update.get("oa_url") and not row.get("oa_url"):
        conn.execute(
            "UPDATE papers SET oa_url = ? WHERE paper_id = ?",
            (update["oa_url"], paper_id),
        )
        changed["oa_url"] = True

    return changed
