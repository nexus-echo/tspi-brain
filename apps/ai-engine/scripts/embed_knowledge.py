"""Embed the TSPI knowledge base into kb_embeddings (Postgres + pgvector only).

Writes two groups of rows (the table is dropped & rebuilt to match the current backend/dimension):

  1. LEGACY rows — kind 'axis' (ref_code = axis code, e.g. 'A17') and kind 'product' — exactly as
     before. The analysis pipeline (axis_mapper RAG fallback, report_composer context) and
     scripts/verify_phase2.py depend on these; their content/format is unchanged.
  2. Q&A CORPUS rows — kind 'kb:<kind>', ref_code = corpus document id (app/knowledge/corpus.py):
     framework, keys, steps, domains, 39 axes (+ monographs), module registry, product monographs,
     180 networks, red-flag & safety rules, dosing protocol, official expert answers. Used only by
     POST /ask and GET /knowledge/search.

Run AFTER seed_db, on Postgres:
    python -m scripts.embed_knowledge
"""
from __future__ import annotations

from app.knowledge.corpus import build_corpus, stats
from app.knowledge.db import IS_POSTGRES, engine, get_session
from app.knowledge.embeddings import EmbeddingProvider
from app.knowledge.models import Axis, Product, SubAxis
from app.knowledge.search import VECTOR_KIND_PREFIX
from app.knowledge.vectors import get_table

_BATCH = 64
_EMBED_CHARS = 2000           # keep each input well inside embedding-model context limits


def _legacy_rows(s) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []  # (kind, ref_code, content)
    for a in s.query(Axis).all():
        subs = " ".join(x.description for x in s.query(SubAxis).filter_by(axis_id=a.id))
        rows.append(("axis", a.code, f"{a.code} {a.name}. {subs}"))
    for p in s.query(Product).all():
        rows.append(("product", str(p.tspi_id or p.name), p.name or ""))
    return rows


def _corpus_rows() -> list[tuple[str, str, str]]:
    return [(f"{VECTOR_KIND_PREFIX}{d['kind']}"[:20], d["id"][:64],
             f"{d['title']}\n{d['content']}"[:_EMBED_CHARS]) for d in build_corpus()]


def embed_all() -> dict:
    if not IS_POSTGRES:
        print("SKIP: embeddings require PostgreSQL + pgvector (set DATABASE_URL). "
              "Q&A still works on SQLite via exact-code + BM25 retrieval.")
        return {"embedded": 0}

    emb = EmbeddingProvider()
    table = get_table(emb.dim)
    eng = engine()
    table.drop(eng, checkfirst=True)     # rebuild to match current dimension/backend
    table.create(eng, checkfirst=True)

    s = get_session()
    try:
        legacy = _legacy_rows(s)
        rows = legacy + _corpus_rows()
        n = 0
        for i in range(0, len(rows), _BATCH):
            chunk = rows[i:i + _BATCH]
            vecs = emb.embed_many([c[2] for c in chunk])
            s.execute(table.insert(), [
                {"kind": k, "ref_code": rc, "content": ct, "embedding": v}
                for (k, rc, ct), v in zip(chunk, vecs)
            ])
            n += len(chunk)
            print(f"  embedded {n}/{len(rows)}")
        s.commit()
        return {"embedded": n, "legacy_rows": len(legacy), "qa_corpus_rows": n - len(legacy),
                "backend": emb.backend, "model": emb.model_id, "dim": emb.dim,
                "corpus": stats()["by_kind"]}
    finally:
        s.close()


if __name__ == "__main__":
    print("embedded:", embed_all())
