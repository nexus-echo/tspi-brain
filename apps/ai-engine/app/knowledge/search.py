"""Hybrid knowledge search for the Q&A API.

Three retrievers, fused with Reciprocal Rank Fusion (RRF):
  1. exact   — codes named in the question (A17, D3, H-003, LBN-001, RF-E01, KEY-B, S4) and
               module/product names (e.g. "KS", "ABSORN") resolve straight to their documents.
  2. vector  — pgvector cosine search over kb_embeddings (Postgres only; built by
               scripts/embed_knowledge.py). Silently skipped when unavailable.
  3. lexical — in-process BM25 over the same corpus. Always available (SQLite dev, CI, or when the
               embedding service is down), so Q&A never hard-depends on the vector store.
Thai text has no word spaces, so Thai runs are indexed as character bigrams.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from app.knowledge import corpus

_RRF_K = 60
VECTOR_KIND_PREFIX = "kb:"
_WEIGHTS = {"exact": 2.0, "vector": 1.0, "lexical": 1.0}
_LATIN = re.compile(r"[a-z0-9]+")
_THAI = re.compile(r"[฀-๿]+")
_GREEK = str.maketrans({"κ": "k", "α": "a", "β": "b", "γ": "g", "δ": "d", "µ": "u", "μ": "u"})
_STOP = frozenset("""a an the of and or to in on for with by is are was be what which who how why
when where does do did can could should would this that these those it its my our your their about
from as at into me i we you he she they tell explain give list show any all there please plan""".split())


def tokenize(text: str) -> list[str]:
    low = (text or "").lower().translate(_GREEK)
    toks = [t for t in _LATIN.findall(low) if t not in _STOP and (len(t) > 1 or t.isdigit())]
    for run in _THAI.findall(low):
        toks.extend(run[i:i + 2] for i in range(max(1, len(run) - 1)))
    return toks


class _BM25:
    def __init__(self, docs: tuple[dict, ...], k1: float = 1.4, b: float = 0.75) -> None:
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs: list[Counter] = []
        df: Counter = Counter()
        for d in docs:
            # title counted twice — titles are short and highly discriminative
            tf = Counter(tokenize(f"{d['title']} {d['title']} {d['content']}"))
            self.tfs.append(tf)
            df.update(tf.keys())
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int, kinds: set[str] | None) -> list[tuple[dict, float]]:
        q = set(tokenize(query))
        if not q:
            return []
        scored = []
        for d, tf, ln in zip(self.docs, self.tfs, self.lens):
            if kinds and d["kind"] not in kinds:
                continue
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s > 0:
                scored.append((d, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


@lru_cache(maxsize=1)
def _index() -> _BM25:
    return _BM25(corpus.build_corpus())


# ----------------------------------------------------------------------------- exact matching
_CODE_RULES = (
    (re.compile(r"\b(?:axis|แกน)\s*(\d{1,2})\b|\bA(\d{1,2})\b", re.I), lambda n: f"axis:A{int(n)}"),
    (re.compile(r"\b(?:domain|โดเมน)\s*(\d{1,2})\b|\bD(\d{1,2})\b", re.I), lambda n: f"domain:D{int(n)}"),
    (re.compile(r"\bLBN-?(\d{1,3})\b|\bN(\d{3})\b", re.I), lambda n: f"network:LBN-{int(n):03d}"),
    (re.compile(r"\bS([1-9])\b"), lambda n: "step:S1-S9"),
)
_MODULE_CODE = re.compile(r"\b([A-Z]{1,6}-\d{2,3}(?:-\d{2,3})?)\b", re.I)
_RF_CODE = re.compile(r"\bRF-[A-Z]{1,3}\d{2}\b", re.I)
_KEY = re.compile(r"\bKEY[- ]?([ABC])\b", re.I)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


@lru_cache(maxsize=1)
def _name_index() -> dict[str, list[str]]:
    """normalized module/product name (and its first word, e.g. 'absorn') -> doc ids."""
    idx: dict[str, list[str]] = {}
    for d in corpus.build_corpus():
        if d["kind"] not in ("module", "product"):
            continue
        name = d["title"].split("—", 1)[-1] if d["kind"] == "module" else d["title"].split(" - ", 1)[-1]
        n = _norm(name)
        if not n or n in ("none", "n a"):
            continue
        for key in {n, n.split()[0]}:
            if len(key) >= 2 and key not in _STOP and not key.isdigit():
                idx.setdefault(key, []).append(d["id"])
    return idx


def exact_ids(question: str) -> list[str]:
    ids: list[str] = []
    q = question or ""
    for rx, fmt in _CODE_RULES:
        for m in rx.finditer(q):
            n = next((g for g in m.groups() if g), None)
            if n:
                ids.append(fmt(n))
    for m in _MODULE_CODE.finditer(q):
        code = m.group(1).upper()
        if code.startswith("LBN") or code.startswith("RF"):
            continue
        ids.append(f"module:{code}")
    ids += [f"safety:{m.group(0).upper()}" for m in _RF_CODE.finditer(q)]
    ids += [f"key:KEY-{m.group(1).upper()}" for m in _KEY.finditer(q)]
    # module / product names as whole words (n-grams up to 4 words)
    words = _norm(q).split()
    names = _name_index()
    for size in (4, 3, 2, 1):
        for i in range(len(words) - size + 1):
            ids.extend(names.get(" ".join(words[i:i + size]), [])[:3])
    if "ks" in words:                                  # KS has its own bowel-performance protocol
        ids.append("dosing:ks-protocol")
    known = corpus.by_id()
    out, seen = [], set()
    for i in ids:
        if i in known and i not in seen:
            seen.add(i)
            out.append(i)
    return out[:12]


# ----------------------------------------------------------------------------- hybrid
async def _vector(question: str, k: int, kinds: set[str] | None) -> list[dict]:
    from app.knowledge.retrieval import aretrieve
    # Only the Q&A corpus rows ('kb:<kind>', ref_code = corpus doc id). The legacy 'axis'/'product'
    # rows stay reserved for the analysis pipeline (axis_mapper / report_composer).
    hits = await aretrieve(question, kind=None, k=k * 3 if kinds else k, kind_prefix=VECTOR_KIND_PREFIX)
    known = corpus.by_id()
    out = []
    for h in hits:
        d = known.get(h.get("ref_code") or "")
        if d is None:                                  # corpus changed since the last embed run
            continue
        if kinds and d["kind"] not in kinds:
            continue
        out.append({**d, "_score": h.get("score")})
    return out[:k]


async def hybrid_search(question: str, k: int = 8, kinds: list[str] | None = None,
                        extra_ids: list[str] | None = None) -> dict:
    """-> {hits: [doc + {score, retrieval}], vector_used: bool}. `extra_ids` (e.g. modules in the
    patient's plan) are fused in at exact-match priority."""
    kset = {x for x in (kinds or []) if x in corpus.KINDS} or None
    known = corpus.by_id()
    exact = [i for i in exact_ids(question) + list(extra_ids or [])
             if i in known and (not kset or known[i]["kind"] in kset)]
    vec = await _vector(question, k, kset)
    lex = _index().search(question, k * 2, kset)

    fused: dict[str, dict] = {}

    def add(doc: dict, rank: int, src: str, raw: float | None) -> None:
        e = fused.setdefault(doc["id"], {**{x: doc[x] for x in ("id", "kind", "ref_code", "title", "content")},
                                         "rrf": 0.0, "retrieval": [], "score": None})
        e["rrf"] += _WEIGHTS[src] / (_RRF_K + rank)
        if src not in e["retrieval"]:
            e["retrieval"].append(src)
        if src == "vector" and raw is not None:
            e["score"] = raw

    for r, i in enumerate(dict.fromkeys(exact)):
        add(known[i], r, "exact", None)
    for r, d in enumerate(vec):
        add(d, r, "vector", d.get("_score"))
    for r, (d, s) in enumerate(lex):
        add(d, r, "lexical", None)
    hits = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]
    for h in hits:
        h["score"] = round(h["rrf"] * 1000, 2) if h["score"] is None else h["score"]
        h.pop("rrf", None)
    return {"hits": hits, "vector_used": bool(vec), "lexical_used": bool(lex), "exact_matches": len(exact)}
