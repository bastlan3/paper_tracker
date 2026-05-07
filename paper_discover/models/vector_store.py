"""
M3 — Persistent embedding cache backed by LanceDB (with in-memory fallback).

Why this exists
---------------
Stages 3 (judging), 4 (saturation), 5 (skeptic re-judging) and 7 (anchor
injection probes) all embed the same anchors and dimension texts repeatedly.
Saturation also re-embeds candidate abstracts on every iteration. With deep
runs hitting tens of thousands of candidates, this cost dominates.

LanceDB gives us:
  - Single-file Lance dataset under runs/<run_id>/embeddings.lance
  - Columnar layout that survives across runs
  - Optional approximate-NN search later (Stage 4 saturation could use it
    to find semantically-near papers without a citation edge — future work)

Soft dependency
---------------
LanceDB is imported lazily. If the import fails (or if the caller passes
no path), we fall back to an in-memory dict that gets dropped at process
exit. All callers see the same `EmbeddingCache` interface either way.

Usage
-----
    cache = EmbeddingCache.open("runs/<run_id>/embeddings.lance", model="bge-m3")
    vec = cache.get("paper_id:doi:10.x/y")
    if vec is None:
        vec = await embed_batch([text])[0]
        cache.put("paper_id:doi:10.x/y", vec)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class _Backend(Protocol):
    def get(self, key: str) -> np.ndarray | None: ...
    def put(self, key: str, vec: np.ndarray) -> None: ...
    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]: ...


# ── In-memory fallback ────────────────────────────────────────────────────────

class _MemoryBackend:
    def __init__(self) -> None:
        self._store: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._store.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._store[key] = vec.astype(np.float32, copy=False)

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        return {k: self._store[k] for k in keys if k in self._store}


# ── LanceDB backend ───────────────────────────────────────────────────────────

class _LanceBackend:
    """
    Single Lance table per (path, model). Schema: key STRING, vector FLOAT32[dim].
    """

    def __init__(self, path: str, model: str, dim: int) -> None:
        import lancedb  # imported lazily so the dep is optional

        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        self._table_name = f"emb_{_sanitize(model)}"
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}  # in-process LRU-lite
        self._table = self._open_or_create()

    def _open_or_create(self):
        import pyarrow as pa
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self._dim)),
        ])
        return self._db.create_table(self._table_name, schema=schema)

    def get(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            return self._cache[key]
        df = self._table.search().where(f"key = '{_sql_escape(key)}'").limit(1).to_list()
        if not df:
            return None
        vec = np.asarray(df[0]["vector"], dtype=np.float32)
        self._cache[key] = vec
        return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        v = vec.astype(np.float32, copy=False)
        if v.shape[0] != self._dim:
            raise ValueError(f"vector dim {v.shape[0]} != expected {self._dim}")
        self._cache[key] = v
        self._table.add([{"key": key, "vector": v.tolist()}])

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        # Fetch all matching rows in one query
        in_clause = ",".join(f"'{_sql_escape(k)}'" for k in keys)
        rows = self._table.search().where(f"key IN ({in_clause})").to_list()
        out: dict[str, np.ndarray] = {}
        for r in rows:
            v = np.asarray(r["vector"], dtype=np.float32)
            out[r["key"]] = v
            self._cache[r["key"]] = v
        return out


# ── Public facade ─────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    Interface used by the rest of the pipeline.

    The `EmbeddingCache.open()` constructor tries LanceDB first; on any
    import or instantiation failure it transparently falls back to the
    in-memory backend so the pipeline never breaks because of a missing
    optional dep.
    """

    def __init__(self, backend: _Backend, *, persistent: bool) -> None:
        self._backend = backend
        self.persistent = persistent

    @classmethod
    def open(
        cls,
        path: str | None,
        *,
        model: str = "bge-m3",
        dim: int = 1024,
    ) -> "EmbeddingCache":
        if path:
            try:
                return cls(_LanceBackend(path, model=model, dim=dim), persistent=True)
            except Exception as exc:  # ImportError, file errors, etc.
                logger.info(
                    "LanceDB unavailable (%s); using in-memory embedding cache.", exc
                )
        return cls(_MemoryBackend(), persistent=False)

    def get(self, key: str) -> np.ndarray | None:
        return self._backend.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._backend.put(key, vec)

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        return self._backend.get_many(keys)

    async def embed_with_cache(
        self,
        keys_and_texts: list[tuple[str, str]],
    ) -> dict[str, np.ndarray]:
        """
        Look up keys in the cache; embed any misses; return a {key: vec} map.
        Uses paper_discover.models.embedding.embed_batch for the misses.
        """
        from .embedding import embed_batch

        cached = self.get_many([k for k, _ in keys_and_texts])
        misses = [(k, t) for k, t in keys_and_texts if k not in cached]
        if misses:
            vecs = await embed_batch([t for _, t in misses])
            for (k, _), v in zip(misses, vecs):
                self.put(k, v)
                cached[k] = v
        return cached


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:40]


def _sql_escape(s: str) -> str:
    return s.replace("'", "''")
