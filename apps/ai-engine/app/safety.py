"""Phase 3 safety layer — flags contraindications; NEVER silently drops a module.

Two rule kinds in data/safety_rules.json:
- module-name rule  : `match` names a module; flags that module if a listed med/condition is
  present (or unconditionally if none are listed).
- med-keyword rule  : has `avoid_with_meds` and the `match`/`applies_to_ingredient` is a drug;
  emits ONE report-level alert when that med is present.
Official contraindication data is pending; the seed rules demonstrate the mechanism. Conservative:
it warns for clinician review, it never auto-prescribes or auto-removes.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path

import logging

_log = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent / "data" / "safety_rules.json"


class SafetyOutcome(str, Enum):
    """Categorical safety result (31 Jul Directive 8). Safety is a GATE, not a numeric score.

    CONTRAINDICATED / HOLD exclude a module BEFORE ranking; INSUFFICIENT_SAFETY_DATA means we
    could not clear it, so it is held rather than assumed safe.
    """
    PASS = "PASS"
    PASS_WITH_MONITORING = "PASS_WITH_MONITORING"
    HOLD = "HOLD"
    CONTRAINDICATED = "CONTRAINDICATED"
    INSUFFICIENT_SAFETY_DATA = "INSUFFICIENT_SAFETY_DATA"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@lru_cache
def _rules() -> dict:
    try:
        return json.loads(_DATA.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"global_alerts": [], "rules": []}


def check_modules(modules: list, medications: list[str], conditions: list[str]) -> list[dict]:
    meds = {_norm(m) for m in (medications or [])}
    conds = {_norm(c) for c in (conditions or [])}
    alerts: list[dict] = []
    seen: set = set()

    def _code(m):
        return getattr(m, "module_code", None) if not isinstance(m, dict) else m.get("module_code")

    def _name(m):
        return getattr(m, "module_name", None) if not isinstance(m, dict) else m.get("module_name")

    for rule in _rules().get("rules", []):
        rmatch = _norm(rule.get("match", ""))
        amed = {_norm(x) for x in rule.get("avoid_with_meds", [])}
        acond = {_norm(x) for x in rule.get("avoid_with_conditions", [])}
        note = rule.get("note", "Potential contraindication — review.")
        is_med_rule = bool(amed) and (rmatch in amed or bool(rule.get("applies_to_ingredient")))

        if is_med_rule:
            if amed & meds:
                key = ("*", note)
                if key not in seen:
                    seen.add(key)
                    alerts.append({"module": None, "reason": note, "severity": "review"})
            continue

        for m in modules:
            mc = _norm(_code(m) or "")
            mn = _norm(_name(m) or "")
            if rmatch and (rmatch in mc or rmatch in mn):
                cond_ok = (not amed and not acond) or bool(amed & meds) or bool(acond & conds)
                if cond_ok and (mc, note) not in seen:
                    seen.add((mc, note))
                    alerts.append({"module": _code(m), "reason": note, "severity": "review"})
                    try:
                        m.safety = ((m.safety + " | ") if getattr(m, "safety", None) else "") + note
                    except Exception:  # noqa: BLE001
                        _log.warning("Could not annotate module %s with safety note", _code(m), exc_info=True)

    # per-module contraindications carried in the registry: flag when a patient med/condition
    # token (>=4 chars) literally appears in the module's contraindication note (conservative).
    tokens = {t for t in (meds | conds) if len(t) >= 4}
    for m in modules:
        for cnote in (getattr(m, "contraindications", None) or []):
            cn = _norm(cnote)
            if any(t in cn for t in tokens):
                key = (_norm(_code(m) or ""), "reg-contra")
                if key not in seen:
                    seen.add(key)
                    alerts.append({"module": _code(m), "severity": "review",
                                   "reason": f"Registry contraindication note matches a patient med/condition: {cnote[:140]}"})
    return alerts


def global_alerts() -> list[str]:
    return _rules().get("global_alerts", [])
