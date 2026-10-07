"""DB-backed knowledge repository (Phase 0) + official module registry (post domain-expert answers).

Axis catalog (39 axes / 12 domains / keys / products) comes from the relational DB. The
axis->module map now reads the OFFICIAL compiled registry (data/axis_module_official.json +
data/module_registry.json): 1 module = 1 product, keyed by H-Code, with a per-axis primary/secondary
role, dose_type, contraindications and status. Falls back to the provisional map if the official
files are absent, so older setups still run.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.knowledge.db import get_session, init_db
from app.knowledge.models import Axis, Domain, Product

import logging

_log = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent.parent / "data"


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


@lru_cache
def _provisional_axis_modules() -> dict:
    return json.loads((_DATA / "axis_module_provisional.json").read_text(encoding="utf-8"))["map"]


@lru_cache
def _official() -> dict | None:
    try:
        return json.loads((_DATA / "axis_module_official.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


@lru_cache
def _registry() -> dict:
    try:
        reg = json.loads((_DATA / "module_registry.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {"_meta": {"registry_version": reg.get("registry_version"),
                      "framework_version": reg.get("framework_version")},
            **{m["module_code"]: m for m in reg.get("modules", [])}}


@lru_cache
def _registry_norms() -> frozenset:
    try:
        reg = json.loads((_DATA / "product_registry.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return frozenset()
    names: set[str] = set()
    for r in reg:
        names.add(_norm(r.get("norm_key", "")))
        names.add(_norm(r.get("canonical_name", "")))
    return frozenset(n for n in names if n)


class KnowledgeRepo:
    """Stable interface over the knowledge DB + official module registry."""

    def __init__(self) -> None:
        init_db()
        self._ensure_seeded()

    def _ensure_seeded(self) -> None:
        s = get_session()
        try:
            empty = s.query(Axis).count() == 0
        finally:
            s.close()
        if empty:
            try:
                from scripts.seed_db import seed_all
                seed_all()
            except Exception:  # noqa: BLE001 — DB stays empty; axis() degrades gracefully
                _log.warning("Knowledge DB auto-seed failed; axis() will degrade", exc_info=True)

    def axis(self, code: str) -> dict:
        s = get_session()
        try:
            a = s.query(Axis).filter_by(code=code).first()
            if not a:
                return {"name": code}
            dom = s.get(Domain, a.domain_id) if a.domain_id else None
            return {"name": a.name, "domain_code": dom.code if dom else None}
        finally:
            s.close()

    def modules_for_axis(self, axis_code: str) -> list[dict]:
        official = _official()
        if official is not None:
            out: list[dict] = []
            for e in official.get("map", {}).get(axis_code, []):
                if e.get("status", "active") != "active":
                    continue
                out.append({
                    "code": e["code"], "name": e.get("name"),
                    "phytocore": e.get("phytocore"), "dose_type": e.get("dose_type", "severity"),
                    "status": e.get("status", "active"), "role": e.get("role", "primary"),
                    "resolved": True, "provisional": False,
                })
            return out
        # fallback: provisional map (legacy)
        reg = _registry_norms()
        return [{"code": c, "name": c, "resolved": _norm(c) in reg, "provisional": True,
                 "role": "primary", "dose_type": "severity", "status": "active"}
                for c in _provisional_axis_modules().get(axis_code, [])]

    def module_contraindications(self, code: str) -> list[str]:
        m = _registry().get(code)
        return list(m.get("contraindications", [])) if m else []

    def registry_meta(self) -> dict:
        return dict(_registry().get("_meta", {}))

    def is_contraindicated(self, module_code: str, medications: list[str]) -> bool:
        # Governance: we FLAG contraindications (safety.py), never silently drop a module.
        return False

    def counts(self) -> dict:
        s = get_session()
        try:
            return {
                "axes": s.query(Axis).count(),
                "domains": s.query(Domain).count(),
                "products": s.query(Product).count(),
            }
        finally:
            s.close()
