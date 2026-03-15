"""
Paper embeddings using Mistral Embed API (no local PyTorch needed)
and UMAP for 2D projection.
"""

import json
import logging
import time

import numpy as np
from sqlmodel import Session

from app.config import settings, DATA_DIR
from app.database import Paper, get_papers_with_embeddings

logger = logging.getLogger(__name__)

# Mistral embed output dimension is 1024
EMBED_BATCH_SIZE = 16  # Mistral API batch limit


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using Mistral Embed API.
    Returns (n, dim) numpy array, L2-normalized.
    Batches automatically to stay within API limits.
    """
    from mistralai import Mistral

    client = Mistral(api_key=settings.MISTRAL_API_KEY)
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        logger.info(f"Embedding batch {i // EMBED_BATCH_SIZE + 1}/{(len(texts) - 1) // EMBED_BATCH_SIZE + 1}")

        response = client.embeddings.create(
            model="mistral-embed",
            inputs=batch,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Rate limit courtesy
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(0.5)

    result = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.clip(norms, 1e-8, None)

    return result


def compute_seed_embeddings() -> np.ndarray:
    """
    Compute or load cached embeddings for the seed paper set.
    Uses arxiv to fetch seed paper metadata, then embeds title + abstract.
    """
    cache_path = DATA_DIR / "seed_embeddings.npy"
    ids_path = DATA_DIR / "seed_ids.json"

    # Check cache
    if cache_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text())
        if cached_ids == settings.SEED_PAPER_IDS:
            logger.info("Loading cached seed embeddings")
            return np.load(cache_path)

    logger.info(f"Computing seed embeddings for {len(settings.SEED_PAPER_IDS)} papers")

    import arxiv

    client = arxiv.Client()
    texts = []

    for paper_id in settings.SEED_PAPER_IDS:
        search = arxiv.Search(id_list=[paper_id])
        results = list(client.results(search))
        if results:
            r = results[0]
            texts.append(f"{r.title}. {r.summary}")
            logger.info(f"  Seed: {r.title[:80]}...")
        else:
            logger.warning(f"  Seed paper {paper_id} not found on arxiv")

    if not texts:
        raise RuntimeError("No seed papers could be fetched from arxiv")

    embeddings = embed_texts(texts)

    # Cache
    np.save(cache_path, embeddings)
    ids_path.write_text(json.dumps(settings.SEED_PAPER_IDS))

    logger.info(f"Seed embeddings computed: shape {embeddings.shape}")
    return embeddings


def recompute_umap(session: Session) -> dict:
    """
    Recompute UMAP 2D projection for all papers with embeddings.
    Updates paper records with umap_x, umap_y.
    Returns the viz data dict.
    """
    import umap

    papers = get_papers_with_embeddings(session)
    if len(papers) < 5:
        logger.warning(f"Only {len(papers)} papers with embeddings, need at least 5 for UMAP")
        return {"papers": [], "message": "Need at least 5 papers for visualization"}

    embeddings = np.array([p.get_embedding_list() for p in papers])

    n_neighbors = min(15, len(papers) - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="cosine",
        min_dist=0.1,
        random_state=42,
    )

    coords = reducer.fit_transform(embeddings)

    viz_data = []
    for i, paper in enumerate(papers):
        paper.umap_x = float(coords[i, 0])
        paper.umap_y = float(coords[i, 1])
        session.add(paper)

        viz_data.append({
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.get_authors_list()[:3],
            "summary": paper.summary or paper.abstract[:200],
            "keywords": paper.get_keywords_list(),
            "similarity_score": paper.similarity_score,
            "published": paper.published.isoformat(),
            "url": paper.url,
            "x": paper.umap_x,
            "y": paper.umap_y,
            "source": paper.source,
        })

    session.commit()

    # Cache viz data to JSON for fast frontend loading
    viz_path = DATA_DIR / "umap_viz.json"
    viz_path.write_text(json.dumps(viz_data, indent=2))

    logger.info(f"UMAP recomputed for {len(papers)} papers")
    return {"papers": viz_data}
