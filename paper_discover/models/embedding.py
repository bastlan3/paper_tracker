"""
Embedding service: BGE-M3 via sentence_transformers (local) or Infinity server.
Returns float32 numpy arrays, L2-normalized.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/models.yaml")


def _cfg() -> dict:
    path = os.environ.get("PAPER_DISCOVER_MODELS_CONFIG", _CONFIG_PATH)
    with open(path) as f:
        return yaml.safe_load(f)["embedding"]


@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer

    cfg = _cfg()
    logger.info("Loading embedding model %s locally", cfg["model"])
    return SentenceTransformer(cfg["model"], trust_remote_code=True)


def embed_batch_sync(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts locally. Returns shape (N, dim), float32, L2-norm."""
    cfg = _cfg()
    if cfg.get("use_server"):
        raise RuntimeError("Server embedding not implemented in M1; set use_server: false")
    model = _local_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=cfg["batch_size"])
    return vecs.astype(np.float32)


async def embed_batch(texts: list[str]) -> np.ndarray:
    """Async wrapper: runs embedding in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_batch_sync, texts)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def top_k_similar(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    paper_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """Return top-k (paper_id, score) pairs from a matrix of row vectors."""
    scores = matrix @ query_vec
    idx = np.argsort(scores)[::-1][:k]
    return [(paper_ids[i], float(scores[i])) for i in idx]
