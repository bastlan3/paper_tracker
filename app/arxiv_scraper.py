"""
Two-stage arxiv paper filtering:
  Stage 1: keyword-based search across relevant categories
  Stage 2: embedding similarity against seed paper set
"""

import json
import logging
import re
from datetime import datetime, timedelta

import arxiv
import numpy as np
from sqlmodel import Session

from app.config import settings
from app.database import Paper, get_paper_by_arxiv_id

logger = logging.getLogger(__name__)


def _clean_arxiv_id(entry_id: str) -> str:
    """Extract clean arxiv ID from full URL."""
    # e.g., http://arxiv.org/abs/2301.12345v1 -> 2301.12345
    match = re.search(r"(\d{4}\.\d{4,5})", entry_id)
    return match.group(1) if match else entry_id


def _matches_keywords(title: str, abstract: str) -> bool:
    """Stage 1: check if paper matches any EBM keywords."""
    text = (title + " " + abstract).lower()
    for kw in settings.ARXIV_KEYWORDS:
        if kw.lower() in text:
            return True
    return False


def fetch_recent_papers(days_back: int = 2) -> list[dict]:
    """
    Fetch recent arxiv papers matching EBM keywords.
    Returns raw paper dicts before similarity filtering.
    """
    query_parts = []
    for kw in settings.ARXIV_KEYWORDS:
        escaped = kw.replace('"', '\\"')
        query_parts.append(f'all:"{escaped}"')

    cat_filter = " OR ".join(f"cat:{c}" for c in settings.ARXIV_CATEGORIES)
    keyword_filter = " OR ".join(query_parts)
    query = f"({keyword_filter}) AND ({cat_filter})"

    logger.info(f"Arxiv query: {query}")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    papers = []

    for result in client.results(search):
        if result.published.replace(tzinfo=None) < cutoff:
            continue

        if not _matches_keywords(result.title, result.summary):
            continue

        papers.append({
            "arxiv_id": _clean_arxiv_id(result.entry_id),
            "title": result.title.strip(),
            "authors": json.dumps([a.name for a in result.authors]),
            "abstract": result.summary.strip(),
            "categories": ",".join(result.categories),
            "published": result.published.replace(tzinfo=None),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
        })

    logger.info(f"Stage 1 (keyword filter): {len(papers)} papers found")
    return papers


def filter_by_similarity(
    papers: list[dict],
    seed_embeddings: np.ndarray,
    embed_fn,
    threshold: float = None,
) -> list[tuple[dict, float, np.ndarray]]:
    """
    Stage 2: filter papers by embedding similarity to seed set.
    Returns (paper_dict, similarity_score, embedding) tuples above threshold.
    """
    if threshold is None:
        threshold = settings.SIMILARITY_THRESHOLD

    if len(papers) == 0:
        return []

    # Embed all candidate paper abstracts
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = embed_fn(texts)

    # Compute mean similarity to seed set
    # seed_embeddings: (n_seeds, dim), embeddings: (n_papers, dim)
    seed_mean = seed_embeddings.mean(axis=0, keepdims=True)  # (1, dim)
    seed_mean = seed_mean / np.linalg.norm(seed_mean, axis=1, keepdims=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)

    similarities = (normed @ seed_mean.T).squeeze()  # (n_papers,)

    results = []
    for i, (paper, sim) in enumerate(zip(papers, similarities)):
        if sim >= threshold:
            results.append((paper, float(sim), embeddings[i]))

    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        f"Stage 2 (similarity filter): {len(results)}/{len(papers)} papers above threshold {threshold:.2f}"
    )
    return results


def fetch_papers_deep(months_back: int = 12, progress_cb=None) -> list[dict]:
    """
    Deep search: fetch up to a year of arxiv papers matching EBM keywords.
    Batches by month to work around arxiv API limits (~2000 results max per query).
    progress_cb(month_index, total_months, papers_so_far) is called after each month.
    """
    import time

    query_parts = []
    for kw in settings.ARXIV_KEYWORDS:
        escaped = kw.replace('"', '\\"')
        query_parts.append(f'all:"{escaped}"')

    cat_filter = " OR ".join(f"cat:{c}" for c in settings.ARXIV_CATEGORIES)
    keyword_filter = " OR ".join(query_parts)
    query = f"({keyword_filter}) AND ({cat_filter})"

    all_papers = []
    seen_ids = set()
    now = datetime.utcnow()

    for month_offset in range(months_back):
        # Define month window
        end_date = now - timedelta(days=30 * month_offset)
        start_date = now - timedelta(days=30 * (month_offset + 1))

        logger.info(
            f"Deep search month {month_offset + 1}/{months_back}: "
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=300,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        month_count = 0
        for result in client.results(search):
            pub = result.published.replace(tzinfo=None)
            if pub > end_date:
                continue
            if pub < start_date:
                break

            if not _matches_keywords(result.title, result.summary):
                continue

            aid = _clean_arxiv_id(result.entry_id)
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            all_papers.append({
                "arxiv_id": aid,
                "title": result.title.strip(),
                "authors": json.dumps([a.name for a in result.authors]),
                "abstract": result.summary.strip(),
                "categories": ",".join(result.categories),
                "published": pub,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
            })
            month_count += 1

        logger.info(f"  Found {month_count} papers in this month window")

        if progress_cb:
            progress_cb(month_offset + 1, months_back, len(all_papers))

        # Be nice to arxiv API — 3 second pause between month batches
        if month_offset < months_back - 1:
            time.sleep(3)

    logger.info(f"Deep search total: {len(all_papers)} papers across {months_back} months")
    return all_papers


def ingest_papers(
    session: Session,
    seed_embeddings: np.ndarray,
    embed_fn,
    days_back: int = 2,
) -> list[Paper]:
    """Full pipeline: fetch, filter, store new papers."""
    raw_papers = fetch_recent_papers(days_back=days_back)
    return _filter_and_store(session, seed_embeddings, embed_fn, raw_papers)


def ingest_papers_deep(
    session: Session,
    seed_embeddings: np.ndarray,
    embed_fn,
    months_back: int = 12,
    progress_cb=None,
) -> list[Paper]:
    """Deep search pipeline: fetch a year of papers, filter, store."""
    raw_papers = fetch_papers_deep(months_back=months_back, progress_cb=progress_cb)
    return _filter_and_store(session, seed_embeddings, embed_fn, raw_papers)


def _filter_and_store(
    session: Session,
    seed_embeddings: np.ndarray,
    embed_fn,
    raw_papers: list[dict],
) -> list[Paper]:
    """Shared logic: dedupe, similarity filter, store."""
    # Skip papers we already have
    new_papers = []
    for p in raw_papers:
        if get_paper_by_arxiv_id(session, p["arxiv_id"]) is None:
            new_papers.append(p)

    logger.info(f"New papers (not in DB): {len(new_papers)}")

    if not new_papers:
        return []

    # Stage 2: similarity filter — batch to avoid OOM on large sets
    BATCH_SIZE = 50
    stored = []

    for batch_start in range(0, len(new_papers), BATCH_SIZE):
        batch = new_papers[batch_start : batch_start + BATCH_SIZE]
        accepted = filter_by_similarity(batch, seed_embeddings, embed_fn)

        for paper_dict, sim_score, embedding in accepted:
            paper = Paper(
                **paper_dict,
                similarity_score=sim_score,
                embedding=json.dumps(embedding.tolist()),
                source="arxiv",
            )
            session.add(paper)
            stored.append(paper)

        # Commit per batch to avoid giant transactions
        session.commit()

    for p in stored:
        session.refresh(p)

    logger.info(f"Stored {len(stored)} new papers")
    return stored
