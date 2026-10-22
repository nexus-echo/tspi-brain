"""Embeddings table + cosine search (Postgres + pgvector only).

Same Postgres DB holds the embeddings alongside the relational catalog. pgvector is imported
lazily so the SQLite dev path never needs it. Used by Phase 2 RAG.
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, select

_TABLE: Table | None = None


def get_table(dim: int = 384) -> Table:
    """Define/cache the kb_embeddings table. Imports pgvector lazily (Postgres path only)."""
    global _TABLE
    if _TABLE is None:
        from pgvector.sqlalchemy import Vector  # local import: not needed on SQLite
        md = MetaData()
        _TABLE = Table(
            "kb_embeddings", md,
            Column("id", Integer, primary_key=True),
            Column("kind", String(20)),        # 'axis' | 'module' | 'product'
            Column("ref_code", String(64)),    # e.g. 'A17' | 'KS' | product id
            Column("content", Text),
            Column("embedding", Vector(dim)),
        )
    return _TABLE


def create_embeddings_table(eng, dim: int = 384) -> Table:
    t = get_table(dim)
    t.create(eng, checkfirst=True)
    return t


def search(session, table: Table, query_vec: list[float], kind: str | None = None, k: int = 5,
           kind_prefix: str | None = None) -> list[dict]:
    """Cosine-distance nearest neighbours. Returns [{kind, ref_code, content, score}].
    `kind_prefix` (e.g. 'kb:') selects the Q&A corpus rows written by embed_knowledge."""
    dist = table.c.embedding.cosine_distance(query_vec).label("dist")
    stmt = select(table.c.kind, table.c.ref_code, table.c.content, dist)
    if kind:
        stmt = stmt.where(table.c.kind == kind)
    if kind_prefix:
        stmt = stmt.where(table.c.kind.startswith(kind_prefix))
    stmt = stmt.order_by(dist).limit(k)
    rows = session.execute(stmt).all()
    return [{"kind": r.kind, "ref_code": r.ref_code, "content": r.content,
             "score": round(1.0 - float(r.dist), 4)} for r in rows]
