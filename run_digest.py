"""Standalone daily digest runner — used by GitHub Actions."""

import asyncio
import logging

from sqlmodel import Session

from app.database import init_db, engine
from app.embeddings import embed_texts, compute_seed_embeddings
from app.arxiv_scraper import ingest_papers
from app.summarizer import enrich_papers
from app.notifications import send_daily_digest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


async def main():
    init_db()
    seed_embeddings = compute_seed_embeddings()

    with Session(engine) as session:
        # days_back=2 catches weekend gaps when job runs on Monday
        new_papers = ingest_papers(session, seed_embeddings, embed_texts, days_back=2)
        if new_papers:
            enrich_papers(session, new_papers)
            await send_daily_digest(session, new_papers)


asyncio.run(main())
