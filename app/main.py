"""
EBM Paper Tracker — FastAPI backend with scheduled jobs.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlmodel import Session
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings, DATA_DIR, BASE_DIR
from app.database import (
    engine, init_db, Paper,
    get_all_papers, get_paper_by_arxiv_id, get_unnotified_papers,
)
from app.embeddings import embed_texts, compute_seed_embeddings, recompute_umap
from app.arxiv_scraper import ingest_papers, ingest_papers_deep
from app.summarizer import enrich_papers
from app.notifications import send_daily_digest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────

seed_embeddings = None
scheduler = AsyncIOScheduler()


# ── Scheduled jobs ────────────────────────────────────────────────────────────

async def daily_ingest_job():
    """Daily job: fetch, filter, summarize, notify."""
    global seed_embeddings
    logger.info("Running daily ingest job...")

    try:
        with Session(engine) as session:
            if seed_embeddings is None:
                seed_embeddings = compute_seed_embeddings()
                logger.info(f"Seed embeddings shape: {seed_embeddings.shape}")

            # Fetch and filter new papers (7 days back to catch weekend gaps)
            new_papers = ingest_papers(session, seed_embeddings, embed_texts, days_back=7)
            logger.info(f"Ingest result: {len(new_papers)} new papers added")

            # Summarize new papers
            if new_papers:
                enrich_papers(session, new_papers)

            # Send notifications only for papers found in this run
            await send_daily_digest(session, new_papers)

        logger.info("Daily ingest job complete")
    except Exception as e:
        logger.error(f"Daily ingest job FAILED: {e}", exc_info=True)


async def weekly_umap_job():
    """Weekly job: recompute UMAP visualization."""
    logger.info("Running weekly UMAP recomputation...")
    with Session(engine) as session:
        recompute_umap(session)
    logger.info("Weekly UMAP job complete")


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global seed_embeddings

    init_db()
    logger.info("Database initialized")

    # Pre-load seed embeddings
    try:
        seed_embeddings = compute_seed_embeddings()
        logger.info("Seed embeddings loaded")
    except Exception as e:
        logger.error(f"Failed to load seed embeddings: {e}")

    # Schedule jobs
    scheduler.add_job(
        daily_ingest_job,
        "cron",
        hour=settings.DAILY_DIGEST_HOUR,
        minute=0,
        id="daily_ingest",
    )
    scheduler.add_job(
        weekly_umap_job,
        "cron",
        day_of_week=settings.WEEKLY_UMAP_DAY[:3].lower(),
        hour=3,
        minute=0,
        id="weekly_umap",
    )
    scheduler.start()
    logger.info(f"Scheduler started: daily at {settings.DAILY_DIGEST_HOUR}:00, UMAP on {settings.WEEKLY_UMAP_DAY}")

    yield

    scheduler.shutdown()
    logger.info("Scheduler shut down")


app = FastAPI(title="EBM Paper Tracker", lifespan=lifespan)

# Serve static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/papers")
async def list_papers(
    search: str = Query(None, description="Search title/summary/keywords"),
    source: str = Query(None, description="Filter by source: arxiv or manual"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with Session(engine) as session:
        papers = get_all_papers(session)

        # Filter
        if source:
            papers = [p for p in papers if p.source == source]
        if search:
            q = search.lower()
            papers = [
                p for p in papers
                if q in p.title.lower()
                or (p.summary and q in p.summary.lower())
                or (p.keywords and q in p.keywords.lower())
            ]

        total = len(papers)
        papers = papers[offset: offset + limit]

        return {
            "total": total,
            "papers": [
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
                    "added_at": p.added_at.isoformat() if p.added_at else None,
                }
                for p in papers
            ],
        }


class AddPaperRequest(BaseModel):
    arxiv_id: str  # e.g., "2301.12345" or full URL


@app.post("/api/papers/add")
async def add_paper_manually(req: AddPaperRequest):
    """Manually add a paper by arxiv ID or URL."""
    import re
    import arxiv as arxiv_lib

    # Extract arxiv ID from URL or raw ID
    arxiv_id = req.arxiv_id.strip()
    match = re.search(r"(\d{4}\.\d{4,5})", arxiv_id)
    if match:
        arxiv_id = match.group(1)
    else:
        raise HTTPException(400, f"Could not parse arxiv ID from: {req.arxiv_id}")

    with Session(engine) as session:
        existing = get_paper_by_arxiv_id(session, arxiv_id)
        if existing:
            return {"status": "exists", "paper": {"arxiv_id": existing.arxiv_id, "title": existing.title}}

        # Fetch from arxiv
        client = arxiv_lib.Client()
        search = arxiv_lib.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            raise HTTPException(404, f"Paper {arxiv_id} not found on arxiv")

        r = results[0]
        text = f"{r.title}. {r.summary}"
        embedding = embed_texts([text])[0]

        paper = Paper(
            arxiv_id=arxiv_id,
            title=r.title.strip(),
            authors=json.dumps([a.name for a in r.authors]),
            abstract=r.summary.strip(),
            categories=",".join(r.categories),
            published=r.published.replace(tzinfo=None),
            url=r.entry_id,
            pdf_url=r.pdf_url,
            embedding=json.dumps(embedding.tolist()),
            source="manual",
        )
        session.add(paper)
        session.commit()
        session.refresh(paper)

        # Summarize in background
        enrich_papers(session, [paper])

        return {
            "status": "added",
            "paper": {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "summary": paper.summary,
                "keywords": paper.get_keywords_list(),
            },
        }


@app.get("/api/viz")
async def get_visualization():
    """Get UMAP visualization data."""
    viz_path = DATA_DIR / "umap_viz.json"
    if viz_path.exists():
        return JSONResponse(content=json.loads(viz_path.read_text()))

    # Compute on the fly if no cache
    with Session(engine) as session:
        result = recompute_umap(session)
        return result


@app.post("/api/ingest")
async def trigger_ingest():
    """Manually trigger the ingest pipeline."""
    await daily_ingest_job()
    return {"status": "ok", "message": "Ingest pipeline completed"}


@app.post("/api/deep-search")
async def trigger_deep_search(months: int = Query(12, ge=1, le=24)):
    """
    Deep search: fetch up to a year (or more) of EBM papers from arxiv.
    This takes a while — ~3 seconds per month due to arxiv rate limits.
    Fetches, filters by similarity, summarizes, then recomputes UMAP.
    """
    global seed_embeddings
    logger.info(f"Starting deep search: {months} months back")

    with Session(engine) as session:
        if seed_embeddings is None:
            seed_embeddings = compute_seed_embeddings()

        new_papers = ingest_papers_deep(
            session, seed_embeddings, embed_texts, months_back=months
        )

        if new_papers:
            enrich_papers(session, new_papers)
            recompute_umap(session)

        # Send notifications only for papers found in this run
        await send_daily_digest(session, new_papers)

    return {
        "status": "ok",
        "new_papers": len(new_papers),
        "message": f"Deep search complete: {len(new_papers)} new papers from {months} months",
    }


@app.post("/api/umap/recompute")
async def trigger_umap():
    """Manually trigger UMAP recomputation."""
    with Session(engine) as session:
        result = recompute_umap(session)
    return {"status": "ok", "papers_count": len(result.get("papers", []))}


@app.get("/api/stats")
async def get_stats():
    """Get tracker statistics."""
    with Session(engine) as session:
        papers = get_all_papers(session)
        return {
            "total_papers": len(papers),
            "arxiv_papers": sum(1 for p in papers if p.source == "arxiv"),
            "manual_papers": sum(1 for p in papers if p.source == "manual"),
            "summarized": sum(1 for p in papers if p.summary),
            "with_embeddings": sum(1 for p in papers if p.embedding),
        }