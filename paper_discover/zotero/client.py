"""
Zotero read-only client for paper-discover.

Tries the local Zotero API (port 23119) first; falls back to the
zotero-mcp MCP server (54yyyu/zotero-mcp) if the local API is unavailable.

Only reads structured fields: DOI, title, authors, abstract, tags, collection.
User notes and highlights are intentionally excluded (FLAG F11 — privacy).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_LOCAL_PORT = int(os.environ.get("ZOTERO_LOCAL_PORT", "23119"))
_LOCAL_BASE = f"http://127.0.0.1:{_LOCAL_PORT}/api"
_API_KEY = os.environ.get("ZOTERO_API_KEY", "")
_LIBRARY_ID = os.environ.get("ZOTERO_LIBRARY_ID", "")
_LIBRARY_TYPE = os.environ.get("ZOTERO_LIBRARY_TYPE", "user")

# Fields from Zotero we expose to the agent
_SAFE_FIELDS = {"DOI", "title", "abstractNote", "date", "publicationTitle", "itemType"}


class ZoteroClient:
    """Read-only client for a local Zotero installation."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=10)

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{_LOCAL_BASE}/")
            return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def get_all_items(self) -> list[dict]:
        """Return all library items as normalised paper dicts (no notes/highlights)."""
        if not await self.is_available():
            logger.warning("Zotero local API not available (port %d); skipping dedup", _LOCAL_PORT)
            return []
        items = await self._fetch_paged(f"{_LOCAL_BASE}/items?format=json&itemType=-attachment-note")
        return [self._normalise(it) for it in items if it.get("data", {}).get("itemType") not in ("note", "attachment")]

    async def get_collection_items(self, collection_key: str) -> list[dict]:
        """Items in a specific Zotero collection (used when the seed is a collection)."""
        url = f"{_LOCAL_BASE}/collections/{collection_key}/items?format=json"
        items = await self._fetch_paged(url)
        return [self._normalise(it) for it in items]

    async def get_item_by_key(self, key: str) -> dict | None:
        try:
            resp = await self._client.get(f"{_LOCAL_BASE}/items/{key}?format=json")
            resp.raise_for_status()
            return self._normalise(resp.json())
        except Exception as exc:
            logger.debug("Zotero item %s fetch failed: %s", key, exc)
            return None

    async def _fetch_paged(self, base_url: str, page_size: int = 100) -> list[dict]:
        items: list[dict] = []
        start = 0
        while True:
            sep = "&" if "?" in base_url else "?"
            resp = await self._client.get(f"{base_url}{sep}limit={page_size}&start={start}")
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            items.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
        return items

    def _normalise(self, raw: dict) -> dict:
        """Strip unsafe fields; return a minimal paper-like dict."""
        data = raw.get("data", raw)
        authors = []
        for creator in data.get("creators", []):
            name = creator.get("name") or f"{creator.get('lastName', '')} {creator.get('firstName', '')}".strip()
            authors.append(name)

        doi = data.get("DOI", "").strip() or None
        year_str = (data.get("date") or "")[:4]
        year = int(year_str) if year_str.isdigit() else None

        return {
            "zotero_key":  data.get("key"),
            "title":       (data.get("title") or "").strip(),
            "abstract":    (data.get("abstractNote") or "").strip(),
            "doi":         doi,
            "year":        year,
            "venue":       data.get("publicationTitle") or data.get("conferenceName") or "",
            "authors":     authors,
            "tags":        [t.get("tag") for t in data.get("tags", []) if t.get("tag")],
            "item_type":   data.get("itemType", ""),
            "collection":  data.get("collections", [None])[0],
        }


_instance: ZoteroClient | None = None


def get_client() -> ZoteroClient:
    global _instance
    if _instance is None:
        _instance = ZoteroClient()
    return _instance
