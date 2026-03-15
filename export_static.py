"""Export DB and UMAP data to docs/ for GitHub Pages."""

import json
import os
from pathlib import Path

from sqlmodel import Session

from app.database import init_db, engine, get_all_papers

DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

init_db()

with Session(engine) as session:
    papers = get_all_papers(session)

papers_json = [
    {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "authors": p.get_authors_list(),
        "abstract": p.abstract,
        "summary": p.summary,
        "keywords": p.get_keywords_list(),
        "similarity_score": p.similarity_score,
        "published": p.published.isoformat() if p.published else None,
        "url": p.url,
        "pdf_url": p.pdf_url,
        "source": p.source,
    }
    for p in papers
]

(DOCS_DIR / "papers.json").write_text(json.dumps(papers_json, indent=2))
print(f"Exported {len(papers_json)} papers to docs/papers.json")

umap_src = Path(__file__).parent / "data" / "umap_viz.json"
if umap_src.exists():
    umap_data = json.loads(umap_src.read_text())
    (DOCS_DIR / "umap.json").write_text(json.dumps(umap_data))
    print("Exported UMAP data to docs/umap.json")
else:
    print("No UMAP data found, skipping")
