"""
Stage 8 — Concept map (M7).

Builds a graph of kept papers and their citation edges from the run DB.

Outputs
-------
- concept_map.json : nodes (papers) + edges (citations) in a Cosmograph-
  / Sigma-friendly shape. Designed to be opened directly in any graph
  viewer; we never embed a renderer in this module.
- concept_map.html : a tiny self-contained HTML viewer that loads the
  JSON via Cosmograph from a CDN. Optional convenience.

Why we don't depend on networkx
-------------------------------
The graph we emit is shallow (nodes + edges + a couple of attributes).
A dict-based representation is enough, keeps tests pure, and avoids a
heavy import for a feature most users will visualise externally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...candidates.db import read_conn

logger = logging.getLogger(__name__)


_LEVEL_COLOUR = {
    "CORE":       "#dc2626",  # red-600
    "SUPPORTING": "#ea580c",  # orange-600
    "CONTEXT":    "#ca8a04",  # yellow-600
    "ADJACENT":   "#2563eb",  # blue-600
}


# ── Public API ───────────────────────────────────────────────────────────────

def build_concept_map(db_path: str, run_id: str) -> dict:
    """
    Read the run's kept papers + citation edges and return a graph dict:
      {
        "nodes": [{"id", "label", "level", "year", "color", "size"}],
        "edges": [{"source", "target"}]
      }
    Edges are restricted to (kept_paper → kept_paper) — citations to
    papers outside the kept set are dropped to keep the graph readable.
    """
    nodes = _fetch_nodes(db_path, run_id)
    if not nodes:
        return {"nodes": [], "edges": []}

    kept_ids = {n["id"] for n in nodes}
    edges = _fetch_edges(db_path, kept_ids)

    return {"nodes": nodes, "edges": edges}


def write_concept_map(
    output_dir: str,
    db_path: str,
    run_id: str,
) -> dict:
    """
    Write concept_map.json (and concept_map.html). Returns the graph dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = build_concept_map(db_path, run_id)

    (out / "concept_map.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False)
    )
    (out / "concept_map.html").write_text(_html_viewer())

    logger.info(
        "Concept map: %d nodes, %d edges → %s",
        len(graph["nodes"]), len(graph["edges"]), out / "concept_map.json",
    )
    return graph


# ── Internals ────────────────────────────────────────────────────────────────

def _fetch_nodes(db_path: str, run_id: str) -> list[dict]:
    with read_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.paper_id, c.level, c.judge_confidence, c.seen_count,
                   p.title, p.year
            FROM candidates c JOIN papers p USING (paper_id)
            WHERE c.run_id = ?
              AND c.level IN ('CORE','SUPPORTING','CONTEXT','ADJACENT')
            """,
            (run_id,),
        ).fetchall()

    return [_to_node(r) for r in rows]


def _to_node(row) -> dict:
    level = row["level"]
    confidence = row["judge_confidence"] or 0.5
    seen = row["seen_count"] or 1
    # Size scales with confidence + log of channel convergence (capped).
    size = 6 + 12 * confidence + min(seen, 6)
    title = (row["title"] or "")[:120]
    return {
        "id":    row["paper_id"],
        "label": title,
        "level": level,
        "year":  row["year"],
        "color": _LEVEL_COLOUR.get(level, "#94a3b8"),
        "size":  round(size, 2),
    }


def _fetch_edges(db_path: str, kept_ids: set[str]) -> list[dict]:
    if len(kept_ids) < 2:
        return []
    with read_conn(db_path) as conn:
        placeholders = ",".join("?" * len(kept_ids))
        rows = conn.execute(
            f"""
            SELECT src_paper_id, dst_paper_id FROM citations
             WHERE src_paper_id IN ({placeholders})
               AND dst_paper_id IN ({placeholders})
            """,
            list(kept_ids) + list(kept_ids),
        ).fetchall()
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["src_paper_id"], r["dst_paper_id"])
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": r["src_paper_id"], "target": r["dst_paper_id"]})
    return edges


def _html_viewer() -> str:
    """Minimal Cosmograph viewer that loads concept_map.json from disk."""
    return """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Concept map</title>
<style>
  html, body, #graph { margin:0; padding:0; height:100%; width:100%; background:#0b1220; }
  body { font-family: -apple-system, system-ui, sans-serif; color:#e5e7eb; }
  #legend { position:fixed; top:8px; left:8px; padding:8px 10px; border-radius:4px;
            background:rgba(0,0,0,0.6); font-size:12px; line-height:1.6; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
         vertical-align:middle; margin-right:6px; }
</style>
</head><body>
<div id="graph"></div>
<div id="legend">
  <strong>Concept map</strong><br>
  <span class="dot" style="background:#dc2626"></span>CORE
  <span class="dot" style="background:#ea580c;margin-left:8px"></span>SUPPORTING<br>
  <span class="dot" style="background:#ca8a04"></span>CONTEXT
  <span class="dot" style="background:#2563eb;margin-left:8px"></span>ADJACENT
</div>
<script type="module">
  import { Cosmograph } from "https://cdn.jsdelivr.net/npm/@cosmograph/cosmograph/dist/index.js";
  const data = await fetch('concept_map.json').then(r => r.json());
  const cg = new Cosmograph(document.getElementById('graph'), {
    nodeColor: n => n.color,
    nodeSize:  n => n.size,
    nodeLabelAccessor: n => n.label,
    showDynamicLabels: true,
  });
  cg.setData(data.nodes, data.edges);
</script>
</body></html>
"""
