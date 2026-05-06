"""
Cross-encoder reranker: BGE-Reranker-v2-M3 via sentence_transformers.
Used for T2 pre-LLM filtering in Stage 3.
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
        return yaml.safe_load(f)["reranker"]


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    cfg = _cfg()
    logger.info("Loading reranker model %s locally", cfg["model"])
    return CrossEncoder(cfg["model"], max_length=512, trust_remote_code=True)


def rerank_sync(query: str, passages: list[str]) -> np.ndarray:
    """Score (query, passage) pairs. Returns float32 array of shape (N,) in [0, 1]."""
    cfg = _cfg()
    ce = _cross_encoder()
    pairs = [(query, p) for p in passages]
    scores = ce.predict(
        pairs,
        batch_size=cfg["batch_size"],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return scores.astype(np.float32)


async def rerank(query: str, passages: list[str]) -> np.ndarray:
    """Async wrapper around the synchronous cross-encoder."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank_sync, query, passages)


def threshold() -> float:
    return float(_cfg()["threshold"])
