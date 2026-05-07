"""
Stage 8 — Report generation: annotated bibliography.

Reads the bibliography view from SQLite and emits:
  - bibliography.md   (human-readable, grouped by level)
  - bibliography.bib  (BibTeX)
  - bibliography.csl.json  (CSL JSON for citation managers)
  - summary.md        (run stats, coverage, flags)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _level_emoji(level: str) -> str:
    return {"CORE": "🔴", "SUPPORTING": "🟠", "CONTEXT": "🟡", "ADJACENT": "🔵"}.get(level, "⚪")


def _format_authors(authors_json: str, max_authors: int = 5) -> str:
    try:
        authors = json.loads(authors_json or "[]")
    except json.JSONDecodeError:
        return ""
    if len(authors) <= max_authors:
        return ", ".join(authors)
    return ", ".join(authors[:max_authors]) + " et al."


def _flags_str(flags_json: str | None) -> str:
    if not flags_json:
        return ""
    try:
        flags = json.loads(flags_json)
    except json.JSONDecodeError:
        return ""
    labels = {
        "review": "Review", "meta_analysis": "Meta-analysis",
        "methods": "Methods", "negative_result": "Negative result",
        "replication": "Replication", "preregistered": "Pre-registered",
        "retracted": "⚠ RETRACTED",
    }
    return " · ".join(labels.get(f, f) for f in flags if f in labels)


def _safe_bibtex_key(row: dict) -> str:
    pid = row.get("paper_id", "unknown")
    first_author = ""
    try:
        authors = json.loads(row.get("authors_json") or "[]")
        if authors:
            name_parts = authors[0].split()
            first_author = name_parts[-1].lower()
    except Exception:
        pass
    year = str(row.get("year") or "")
    base = "".join(c for c in first_author if c.isalnum()) + year
    slug = pid.split(":")[-1][:6].replace("/", "")
    return f"{base or 'unknown'}{slug}"


def generate_bibliography_md(rows: list[dict], plan: dict, run_id: str) -> str:
    """Render the annotated bibliography as Markdown."""
    lines = [
        f"# Literature Review",
        f"",
        f"**Run ID**: `{run_id}`  ",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Intent**: {plan.get('intent', '')}",
        f"",
    ]

    levels = ["CORE", "SUPPORTING", "CONTEXT", "ADJACENT"]
    level_headers = {
        "CORE":       "## Core Papers",
        "SUPPORTING": "## Supporting Papers",
        "CONTEXT":    "## Background / Context",
        "ADJACENT":   "## Cross-domain Analogues",
    }

    by_level: dict[str, list[dict]] = {lv: [] for lv in levels}
    for row in rows:
        lv = row.get("level")
        if lv in by_level:
            by_level[lv].append(row)

    for level in levels:
        papers = by_level[level]
        if not papers:
            continue
        lines.append(level_headers[level])
        lines.append("")
        for row in papers:
            em = _level_emoji(level)
            title = row.get("title", "Unknown title")
            authors = _format_authors(row.get("authors_json") or "[]")
            year = row.get("year") or "n.d."
            venue = row.get("venue") or ""
            doi = row.get("doi")
            doi_link = f" · [DOI](https://doi.org/{doi})" if doi else ""
            oa_url = row.get("oa_url")
            oa_link = f" · [Full text]({oa_url})" if oa_url else ""
            conf = row.get("judge_confidence")
            conf_str = f" · Confidence: {conf:.0%}" if conf is not None else ""
            flags = _flags_str(row.get("flags_json"))
            flags_str = f" · *{flags}*" if flags else ""
            preprint_str = " · **Preprint**" if row.get("is_preprint") else ""
            retracted_str = "\n\n> ⚠ **This paper has been retracted.**" if row.get("retracted") else ""
            evidence = row.get("evidence_span") or ""
            seen_bin = ""
            try:
                seen_by = json.loads(row.get("seen_by_json") or "[]")
                n = row.get("seen_count", 1)
                if n >= 4:
                    seen_bin = " · Found by 4+ channels"
                elif n >= 2:
                    seen_bin = f" · Found by {n} channels"
            except Exception:
                pass

            lines.extend([
                f"### {em} {title}",
                f"",
                f"**{authors}** ({year}){' · ' + venue if venue else ''}{doi_link}{oa_link}",
                f"{conf_str}{flags_str}{preprint_str}{seen_bin}",
                f"",
            ])
            if evidence:
                lines.extend([f"> {evidence}", f""])
            if retracted_str:
                lines.extend([retracted_str, ""])
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_bibtex(rows: list[dict]) -> str:
    """Emit a .bib file for all kept papers."""
    entries: list[str] = []
    for row in rows:
        key = _safe_bibtex_key(row)
        title = row.get("title", "").replace("{", "\\{").replace("}", "\\}")
        authors_raw = json.loads(row.get("authors_json") or "[]")
        authors_bib = " and ".join(authors_raw)
        year = row.get("year") or ""
        journal = (row.get("venue") or "").replace("{", "\\{").replace("}", "\\}")
        doi = row.get("doi") or ""
        url = row.get("oa_url") or (f"https://doi.org/{doi}" if doi else "")

        entry_type = "article"
        entry = (
            f"@{entry_type}{{{key},\n"
            f"  title   = {{{title}}},\n"
            f"  author  = {{{authors_bib}}},\n"
            f"  year    = {{{year}}},\n"
            + (f"  journal = {{{journal}}},\n" if journal else "")
            + (f"  doi     = {{{doi}}},\n" if doi else "")
            + (f"  url     = {{{url}}},\n" if url else "")
            + f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)


def generate_csl_json(rows: list[dict]) -> list[dict]:
    """Emit CSL JSON list (importable into Zotero, Mendeley, etc.)."""
    items: list[dict] = []
    for row in rows:
        authors_raw = json.loads(row.get("authors_json") or "[]")
        csl_authors = []
        for name in authors_raw:
            parts = name.rsplit(" ", 1)
            if len(parts) == 2:
                csl_authors.append({"family": parts[1], "given": parts[0]})
            else:
                csl_authors.append({"literal": name})

        item: dict = {
            "type": "article-journal",
            "id": row.get("paper_id"),
            "title": row.get("title", ""),
            "author": csl_authors,
            "issued": {"date-parts": [[row["year"]]]} if row.get("year") else {},
            "container-title": row.get("venue") or "",
        }
        if row.get("doi"):
            item["DOI"] = row["doi"]
        if row.get("oa_url"):
            item["URL"] = row["oa_url"]
        items.append(item)
    return items


_RETRACTION_ATTRIBUTION = (
    "_Retraction status sourced via Crossref's relation feed, which incorporates "
    "Retraction Watch data (CC-BY 4.0; © The Center for Scientific Integrity)._"
)


def generate_summary_md(
    stats: dict,
    plan: dict,
    run_id: str,
    coverage: dict | None,
    hygiene: dict | None = None,
) -> str:
    """Short statistics summary."""
    total_kept = sum(stats.get(lv, 0) for lv in ["CORE", "SUPPORTING", "CONTEXT", "ADJACENT"])
    total_cut = stats.get("CUT", 0)
    total = total_kept + total_cut
    lines = [
        f"# Run Summary — `{run_id}`",
        f"",
        f"**Intent**: {plan.get('intent','')}",
        f"",
        f"## Counts",
        f"| Level | Papers |",
        f"|---|---|",
        f"| 🔴 CORE | {stats.get('CORE', 0)} |",
        f"| 🟠 SUPPORTING | {stats.get('SUPPORTING', 0)} |",
        f"| 🟡 CONTEXT | {stats.get('CONTEXT', 0)} |",
        f"| 🔵 ADJACENT | {stats.get('ADJACENT', 0)} |",
        f"| ⚪ CUT | {total_cut} |",
        f"| **Total evaluated** | **{total}** |",
        f"",
    ]
    if coverage:
        p = coverage.get("coverage_p")
        lo = coverage.get("coverage_ci_lo")
        hi = coverage.get("coverage_ci_hi")
        if p is not None:
            lines += [
                f"## Coverage estimate",
                f"≈ **{p:.0%}** (CI {lo:.0%}–{hi:.0%})" if lo and hi else f"≈ **{p:.0%}**",
                f"",
            ]
    if hygiene:
        lines += [
            f"## Hygiene",
            f"| Check | Count |",
            f"|---|---|",
            f"| Papers checked | {hygiene.get('checked', 0)} |",
            f"| Retractions flagged | {hygiene.get('retracted', 0)} |",
            f"| Errata attached | {hygiene.get('errata', 0)} |",
            f"| OA URLs resolved | {hygiene.get('oa_resolved', 0)} |",
            f"",
            _RETRACTION_ATTRIBUTION,
            f"",
        ]
    return "\n".join(lines)


def write_report(
    output_dir: str,
    rows: list[dict],
    plan: dict,
    run_id: str,
    judge_stats: dict,
    coverage: dict | None = None,
    hygiene: dict | None = None,
    db_path: str | None = None,
    gaps: dict | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bib_md = generate_bibliography_md(rows, plan, run_id)
    (out / "bibliography.md").write_text(bib_md)
    logger.info("Wrote bibliography.md (%d papers)", len(rows))

    bib_tex = generate_bibtex(rows)
    (out / "bibliography.bib").write_text(bib_tex)

    csl = generate_csl_json(rows)
    (out / "bibliography.csl.json").write_text(json.dumps(csl, indent=2, ensure_ascii=False))

    summary_md = generate_summary_md(judge_stats, plan, run_id, coverage, hygiene)
    (out / "summary.md").write_text(summary_md)

    # M7: concept map + PRISMA + gap list (concept_map and PRISMA need DB access)
    if db_path:
        from .concept_map import write_concept_map
        from .prisma import write_prisma
        write_concept_map(output_dir, db_path, run_id)
        write_prisma(output_dir, db_path, run_id)
    if gaps is not None:
        from .gap_list import write_gap_list
        write_gap_list(output_dir, gaps)

    logger.info("Report written to %s", output_dir)
