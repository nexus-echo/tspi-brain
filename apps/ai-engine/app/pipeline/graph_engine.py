"""Stage 3 -- Root-cause chain via a directed graph of axis dependencies.

Builds a NetworkX DiGraph of upstream->downstream axis edges, propagates the patient's
severities, and uses PageRank to rank DRIVER vs AMPLIFIER axes. The edge list below is a
seed; Phase 1 loads the curated dependency graph from the knowledge base.
"""
from __future__ import annotations

import networkx as nx

from app.schemas import AxisScore, Severity

import logging

_log = logging.getLogger(__name__)

_SEVERITY_WEIGHT = {
    Severity.optimal: 0.0,
    Severity.subclinical: 0.33,
    Severity.functional_impairment: 0.66,
    Severity.pathological: 1.0,
}

# Seed upstream -> downstream edges (microbiome/inflammation are common upstream hubs).
_SEED_EDGES: list[tuple[str, str]] = [
    ("A17", "A18"), ("A18", "A1"), ("A1", "A5"), ("A1", "A26"),
    ("A1", "A20"), ("A1", "A33"), ("A5", "A6"), ("A34", "A1"),
    ("A1", "A31"), ("A31", "A25"),
]


def _graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(_SEED_EDGES)
    return g


def root_cause(axis_scores: list[AxisScore]) -> tuple[list[str], list[AxisScore]]:
    g = _graph()
    weights = {a.axis_code: _SEVERITY_WEIGHT[a.severity] for a in axis_scores}
    for code in weights:
        g.add_node(code)

    # Personalized PageRank biased by severity; fall back if SciPy is unavailable.
    personalization = {n: weights.get(n, 0.0) + 1e-6 for n in g.nodes}
    try:
        ranks = nx.pagerank(g, personalization=personalization) if g.number_of_edges() else {}
    except Exception:  # noqa: BLE001
        _log.warning("PageRank failed; using degree-based fallback ranking", exc_info=True)
        ranks = {n: weights.get(n, 0.0) + 0.1 * g.out_degree(n) for n in g.nodes}

    present = list(axis_scores)
    present.sort(key=lambda a: ranks.get(a.axis_code, 0.0), reverse=True)
    for i, a in enumerate(present):
        a.is_driver = i < max(1, len(present) // 2) and weights[a.axis_code] >= 0.66

    sub = g.subgraph([a.axis_code for a in axis_scores]).copy()
    try:
        ordered = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        ordered = [a.axis_code for a in present]
    chain = [c for c in ordered if c in weights]

    return chain, present
