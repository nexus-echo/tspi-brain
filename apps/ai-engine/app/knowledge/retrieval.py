"""Phase 2 RAG retrieval — async + cached, strictly additive and fail-safe.

- aretrieve(): async; embeds the query without blocking the event loop and runs the (sync) DB
  search in a worker thread. Use this at request time.
- retrieve(): sync; for offline scripts (e.g. verify_phase2).
Both share an LRU cache of query embeddings (keyed by backend+model+dim+text), so repeated
queries (axis names recur across patients) skip the embedder entirely. Both return [] whenever
vectors aren't available (SQLite/no pgvector/no embeddings/any error).
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

_log = logging.getLogger(__name__)

_CACHE: "OrderedDict[tuple, list[float]]" = OrderedDict()
_CACHE_MAX = 2048


def available() -> bool:
    from app.knowledge.db import IS_POSTGRES
    return IS_POSTGRES


def clear_cache() -> None:
    _CACHE.clear()


def _key(emb, text: str) -> tuple:
    return (emb.backend, emb.model_id, emb.dim, text)


def _cache_get(key: tuple):
    v = _CACHE.get(key)
    if v is not None:
        _CACHE.move_to_end(key)
    return v


def _cache_put(key: tuple, vec: list[float]) -> None:
    _CACHE[key] = vec
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _search_sync(vec: list[float], kind: str | None, k: int) -> list[dict]:
    from app.knowledge.db import get_session
    from app.knowledge.vectors import get_table, search
    s = get_session()
    try:
        return search(s, get_table(len(vec)), vec, kind=kind, k=k)
    finally:
        s.close()


def retrieve(query: str, kind: str | None = None, k: int = 5) -> list[dict]:
    """Synchronous retrieval (offline scripts)."""
    if not available():
        return []
    try:
        from app.knowledge.embeddings import EmbeddingProvider
        emb = EmbeddingProvider()
        key = _key(emb, query)
        vec = _cache_get(key)
        if vec is None:
            vec = emb.embed(query)
            _cache_put(key, vec)
        return _search_sync(vec, kind, k)
    except Exception as e:  # noqa: BLE001 — RAG is optional
        _log.warning("RAG retrieve failed: %s", e, exc_info=True)
        return []


async def aretrieve(query: str, kind: str | None = None, k: int = 5) -> list[dict]:
    """Async retrieval for request time: non-blocking embed + threaded DB search + cache."""
    if not available():
        return []
    try:
        from app.knowledge.embeddings import EmbeddingProvider
        emb = EmbeddingProvider()
        key = _key(emb, query)
        vec = _cache_get(key)
        if vec is None:
            vec = await emb.aembed(query)
            _cache_put(key, vec)
        return await asyncio.to_thread(_search_sync, vec, kind, k)
    except Exception as e:  # noqa: BLE001 — RAG is optional
        _log.warning("RAG aretrieve failed: %s", e, exc_info=True)
        return []
