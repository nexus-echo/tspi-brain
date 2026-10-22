"""Clinician step-by-step treatment-plan workflow ("plan sessions").

The one-shot /report runs the whole pipeline in one call. For clinicians, the SAME pipeline stages
(app/pipeline/orchestrator.py) run one step at a time; after each step the engine VERIFIES the
outcome (checks) and the clinician must CONFIRM it — acknowledging every check that needs it —
or ADJUST it (structured, reason-coded) or GO BACK, before the next step is computed:

    1 intake     data check: which labs the engine used / could not use, symptoms recognised, consent
    2 red_flags  deterministic red-flag + critical-value screen
    3 analysis   39 axes, NSS + severity level, root-cause chain, candidate networks
                 adjust: exclude_axis / include_axis  (affects module matching only; NSS unchanged)
    4 modules    matched + selected modules, alternatives with reasons, safety outcomes
                 adjust: remove_module / add_module
    5 dosing     default dose per module (NSS level / KS bowel protocol), safety alerts
                 adjust: change_dose
    -> confirming dosing composes and saves the DRAFT report (still needs physician approval).

The engine is the only source of clinical content: steps return engine output; clinician
adjustments are recorded with actor, time, before/after and reason code (never silent).
Entering a step computes it fresh; going back re-opens an earlier step as it was and clears
every later step.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone

from app import phenotype, red_flags
from app.pipeline import normalizer
from app.schemas import AnalysisResult, AxisScore, ModulePick, PatientInput

STEPS = ["intake", "red_flags", "analysis", "modules", "dosing"]
TITLES = {
    "intake": "Case data check",
    "red_flags": "Red-flag screen",
    "analysis": "Axes, severity & networks",
    "modules": "Module selection",
    "dosing": "Dosing & safety",
}
ADJUST_ACTIONS = {
    "intake": ["update_case"],
    "red_flags": [],
    "analysis": ["exclude_axis", "include_axis"],
    "modules": ["remove_module", "add_module"],
    "dosing": ["change_dose"],
}
REASON_CODES = ["NEW_CLINICAL_INFORMATION", "PATIENT_PREFERENCE", "SAFETY_CONCERN", "REGISTRY_ERROR",
                "ALGORITHM_ERROR", "CLINICAL_JUDGMENT", "DIAGNOSTIC_UNCERTAINTY", "TREATMENT_RESPONSE"]
DRAFT_NOTICE = ("Step-by-step AI draft. Each step needs your confirmation; the final plan is a DRAFT "
                "that still requires physician approval before any patient use.")


class WorkflowError(Exception):
    """A client-correctable workflow problem -> HTTP status + message (+ optional detail)."""

    def __init__(self, status: int, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail or {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(code: str, severity: str, message: str, requires_ack: bool = False, **extra) -> dict:
    """severity: info | warning | critical | blocking. blocking -> cannot confirm at all."""
    return {"code": code, "severity": severity, "message": message,
            "requires_ack": requires_ack or severity == "critical", **extra}


def _patient(state: dict) -> PatientInput:
    return PatientInput.model_validate(state["patient"])


def _analysis(state: dict) -> AnalysisResult:
    return AnalysisResult.model_validate(state["analysis"])


def _mods(rows: list[dict]) -> list[ModulePick]:
    return [ModulePick.model_validate(m) for m in rows]


def _dump(mods: list[ModulePick]) -> list[dict]:
    return [m.model_dump(mode="json") for m in mods]


def _log(state: dict, actor: dict, event: str, step: str, **detail) -> None:
    state.setdefault("history", []).append(
        {"at": _now(), "actor": actor, "event": event, "step": step, **detail})


# ============================================================================ step computations
def _compute_intake(state: dict) -> dict:
    p = _patient(state)
    signals = normalizer.normalize(p)
    labs, checks = [], []
    for lab in p.labs:
        mine = [s for s in signals if s.source.startswith(f"{lab.analyte}=")]
        if not mine:
            status = "not_used"
        elif all(s.concept.startswith("lab:") for s in mine):
            status = "not_recognised"
        else:
            status = "used"
        labs.append({"analyte": lab.analyte, "value": lab.value, "unit": lab.unit,
                     "ref_low": lab.ref_low, "ref_high": lab.ref_high,
                     "flag": next((s.source.rsplit("(", 1)[-1].rstrip(")") for s in mine), lab.flag),
                     "status": status,
                     "interpreted_as": sorted({f"{s.concept} ({s.direction})" for s in mine
                                               if not s.concept.startswith("lab:")})})
        if status == "not_used":
            checks.append(_check(f"LAB_NOT_USED:{lab.analyte}", "warning",
                                 f"{lab.analyte} will not be used by the engine (e.g. a context-dependent "
                                 "marker without its companion test, such as ferritin without CRP).",
                                 requires_ack=True))
        elif status == "not_recognised":
            checks.append(_check(f"LAB_NOT_RECOGNISED:{lab.analyte}", "warning",
                                 f"{lab.analyte} is not in the TSPI marker registry; it is kept but cannot be "
                                 "interpreted. Rename it to the TSPI analyte name if it is a known test.",
                                 requires_ack=True))
        if isinstance(lab.value, (int, float)) and not lab.unit:
            checks.append(_check(f"LAB_NO_UNIT:{lab.analyte}", "warning",
                                 f"{lab.analyte} has no unit; critical-value and cut-off checks may not apply.",
                                 requires_ack=True))
    pheno = phenotype.evaluate(p)
    syms = [s.get("name") for s in pheno.get("symptoms_identified", []) if isinstance(s, dict)]
    if not p.consent.get("ai_analysis"):
        checks.append(_check("CONSENT_MISSING", "blocking",
                             "Patient consent to AI analysis is not recorded. Update the case with consent."))
    if not p.labs and not (p.symptoms or p.structured_symptoms):
        checks.append(_check("NO_CLINICAL_DATA", "blocking", "No labs and no symptoms were provided."))
    elif not p.labs:
        checks.append(_check("NO_LABS", "warning",
                             "No lab results: the analysis will rely on symptoms only and be less complete.",
                             requires_ack=True))
    if (p.symptoms or p.structured_symptoms) and not syms:
        checks.append(_check("NO_SYMPTOMS_RECOGNISED", "info",
                             "None of the described symptoms matched the TSPI symptom dictionary."))
    if not p.medications:
        checks.append(_check("NO_MEDICATIONS", "info",
                             "No medications listed: drug–herb safety checks can only use what is recorded."))
    used = sum(1 for x in labs if x["status"] == "used")
    return {
        "summary": f"{used} of {len(labs)} lab result(s) will be used; {len(syms)} symptom(s) recognised.",
        "case": {"case_id": p.case_id, "age_band": p.age_band, "sex": p.sex, "symptoms": p.symptoms,
                 "medications": p.medications, "conditions": p.conditions, "imaging": p.imaging,
                 "consent_ai_analysis": bool(p.consent.get("ai_analysis"))},
        "labs": labs,
        "symptoms_identified": syms,
        "checks": checks,
    }


def _compute_red_flags(state: dict) -> dict:
    scr = red_flags.screen(_patient(state))
    state["red_flag_screen"] = scr
    checks = []
    for f in scr.get("red_flags", []):
        checks.append(_check(f"RED_FLAG:{f['red_flag_code']}", "critical",
                             f"RED FLAG [{f['action_class']}] {f['name']} ({f['matched']}). "
                             f"Required action: {scr.get('required_action')}."))
    if not scr.get("module_plan_allowed", True):
        checks.append(_check("MODULE_PLAN_BLOCKED", "critical",
                             "Emergency red flag: the module plan is blocked. The workflow can continue to "
                             "document the analysis, but no modules will be selected."))
    elif scr.get("module_plan_requires_clinician_override"):
        checks.append(_check("MODULE_PLAN_NEEDS_OVERRIDE", "critical",
                             "Urgent red flag: a module plan is allowed only with your explicit clinical override."))
    if scr.get("release_block"):
        checks.append(_check("RELEASE_BLOCK", "warning",
                             "The resulting plan will be blocked from patient release.", requires_ack=True))
    return {
        "summary": (f"{len(scr.get('red_flags', []))} red flag(s) found — action class "
                    f"{scr.get('action_class')}." if scr.get("red_flags") else "No red flags or critical values found."),
        "screen": scr,
        "checks": checks,
    }


def _effective_axes(state: dict) -> list[AxisScore]:
    """Axis set used for module matching after the clinician's axis decisions."""
    dec = state.get("axis_decisions", {})
    out = []
    for a in _analysis(state).axis_scores:
        if a.axis_code in dec.get("excluded", {}):
            continue
        if a.axis_code in dec.get("included", {}) and not a.scoring_effect:
            a = a.model_copy(update={"scoring_effect": True})
        out.append(a)
    return out


def _analysis_view(state: dict) -> dict:
    from app.pipeline.orchestrator import candidate_networks, nss_breakdown_for
    a = _analysis(state)
    dec = state.get("axis_decisions", {"excluded": {}, "included": {}})
    axes = []
    for x in sorted(a.axis_scores, key=lambda x: (x.score is None, -(x.score or 0))):
        axes.append({"axis_code": x.axis_code, "axis_name": x.axis_name, "domain_code": x.domain_code,
                     "score": x.score, "severity": x.severity.value if hasattr(x.severity, "value") else x.severity,
                     "status": getattr(x.status, "value", x.status), "confidence": x.confidence_label,
                     "driver": x.is_driver, "evidence": x.evidence[:6], "missing_evidence": x.missing_evidence[:6],
                     "hypothesis_only": not x.scoring_effect,
                     "clinician_decision": ("excluded" if x.axis_code in dec["excluded"]
                                            else "included" if x.axis_code in dec["included"] else None)})
    checks = []
    if not a.axis_scores:
        checks.append(_check("NO_AXES", "warning", "No axis could be assessed from the data; no modules "
                             "will be matched.", requires_ack=True))
    low = [x["axis_code"] for x in axes if x["confidence"] in ("Very Low", "Low") and not x["hypothesis_only"]
           and x["clinician_decision"] != "excluded"]
    if low:
        checks.append(_check("LOW_CONFIDENCE_AXES", "warning",
                             f"Low-confidence axes will drive module matching: {', '.join(low)}. "
                             "Confirm them or exclude any you do not accept.", requires_ack=True, axes=low))
    hyp = [x["axis_code"] for x in axes if x["hypothesis_only"] and x["clinician_decision"] != "included"]
    if hyp:
        checks.append(_check("HYPOTHESIS_AXES", "info",
                             f"Hypothesis-only axes (not used for modules unless you include them): {', '.join(hyp)}.",
                             axes=hyp))
    if a.not_assessed:
        checks.append(_check("NOT_ASSESSED_AXES", "info",
                             f"{len(a.not_assessed)} axes could not be assessed (insufficient data — NOT normal)."))
    if dec["excluded"] or dec["included"]:
        checks.append(_check("AXIS_DECISIONS", "info",
                             "Your axis decisions change module matching only; NSS and the severity level "
                             "remain the engine's values (dose can be changed at the dosing step)."))
    drivers = [x["axis_code"] for x in axes if x["driver"]]
    return {
        "summary": (f"NSS {a.nss}/100 — severity level {a.severity_level} {a.severity_name}. "
                    f"{len(axes)} axes assessed" + (f"; drivers {', '.join(drivers)}" if drivers else "") + "."),
        "nss": a.nss, "severity_level": a.severity_level, "severity_name": a.severity_name,
        "nss_detail": nss_breakdown_for(a),
        "root_cause_chain": a.root_cause_chain, "system_priority": a.system_priority,
        "nine_step_position": a.nine_step_position, "prakati_gap": a.prakati_gap,
        "axes": axes,
        "not_assessed": [{"axis_code": x.axis_code, "axis_name": x.axis_name,
                          "required_evidence": x.required_evidence[:4]} for x in a.not_assessed],
        "networks": candidate_networks(a),
        "notes": a.notes,
        "checks": checks,
    }


async def _compute_analysis(state: dict) -> dict:
    from app.pipeline.orchestrator import analyze
    state["analysis"] = (await analyze(_patient(state))).model_dump(mode="json")
    state["axis_decisions"] = {"excluded": {}, "included": {}}
    return _analysis_view(state)


def _modules_view(state: dict) -> dict:
    from app.pipeline.orchestrator import safety_alerts
    plan = state["plan"]
    sel, con = _mods(plan["modules"]), _mods(plan["considered"])
    alerts = safety_alerts(copy.deepcopy(sel), copy.deepcopy(con), _patient(state))
    sel_codes = {m.module_code for m in sel}
    checks = []
    scr = state.get("red_flag_screen") or {}
    if not scr.get("module_plan_allowed", True):
        checks.append(_check("MODULE_PLAN_BLOCKED", "info", "Module plan blocked by the red-flag screen."))
    elif not sel:
        checks.append(_check("NO_MODULES", "warning", "No module was selected.", requires_ack=True))
    for m in sel:
        if m.safety_outcome not in ("PASS",):
            checks.append(_check(f"SAFETY_OUTCOME:{m.module_code}", "warning",
                                 f"{m.module_code} {m.module_name}: safety outcome {m.safety_outcome}"
                                 + (f" — {'; '.join(m.contraindications)[:200]}" if m.contraindications else "") + ".",
                                 requires_ack=True))
        if not m.resolved:
            checks.append(_check(f"UNRESOLVED_MODULE:{m.module_code}", "warning",
                                 f"{m.module_code} is not resolved in the product registry.", requires_ack=True))
    rel = [x for x in alerts if not x.get("module") or x.get("module") in sel_codes]
    for i, x in enumerate(rel, 1):
        checks.append(_check(f"SAFETY_ALERT:{x.get('module') or 'general'}:{i}", "warning",
                             f"Safety alert ({x.get('module') or 'general'}): {x.get('reason')}", requires_ack=True))
    selection = plan.get("selection") or {}
    if selection.get("requires_senior_approval"):
        checks.append(_check("SENIOR_APPROVAL", "warning", "This selection requires senior clinician approval.",
                             requires_ack=True))
    if selection.get("requires_explicit_justification"):
        checks.append(_check("JUSTIFICATION_REQUIRED", "warning",
                             "This selection exceeds typical limits and needs an explicit justification "
                             "(add it as the confirmation note).", requires_ack=True))

    def row(m: ModulePick) -> dict:
        return {"module_code": m.module_code, "module_name": m.module_name, "tier": m.tier, "phase": m.phase,
                "target_axes": m.target_axes, "match_score": m.module_match_score,
                "safety_outcome": m.safety_outcome, "contraindications": m.contraindications,
                "reason_not_selected": m.reason_not_selected, "notes": m.match_notes[:4]}
    return {
        "summary": f"{len(sel)} module(s) selected from {len(sel) + len(con)} matched.",
        "selected": [row(m) for m in sel],
        "considered": [row(m) for m in con][:15],
        "considered_total": len(con),
        "selection": selection,
        "safety_alerts": alerts,
        "checks": checks,
    }


def _compute_modules(state: dict) -> dict:
    from app.pipeline.orchestrator import plan_modules
    a = _analysis(state)
    sel, con, selection = plan_modules(a, _patient(state), axis_scores=_effective_axes(state))
    state["plan"] = {"modules": _dump(sel), "considered": _dump(con), "selection": selection}
    state["dose_overrides"] = {}
    return _modules_view(state)


def _dosing_view(state: dict) -> dict:
    from app.knowledge import dosing
    from app.pipeline.orchestrator import apply_default_doses, safety_alerts
    a = _analysis(state)
    mods = apply_default_doses(_mods(state["plan"]["modules"]), a.nss)
    overrides = state.get("dose_overrides", {})
    rows, any_bowel = [], False
    for m in mods:
        src = "bowel_protocol" if (m.dose_type == "bowel" or dosing.is_bowel(m.module_code)
                                   or dosing.is_bowel(m.module_name or "")) else "severity_level"
        any_bowel |= src == "bowel_protocol"
        if m.module_code in overrides:
            m.dose, src = overrides[m.module_code]["dose"], "clinician"
        rows.append({"module_code": m.module_code, "module_name": m.module_name, "phase": m.phase,
                     "dose": m.dose, "dose_source": src})
    state["plan"]["dosed"] = _dump(mods)
    alerts = safety_alerts(copy.deepcopy(mods), [], _patient(state))
    checks = [_check(f"DOSE_SAFETY:{x.get('module') or 'general'}:{i}", "warning",
                     f"Safety alert ({x.get('module') or 'general'}): {x.get('reason')}", requires_ack=True)
              for i, x in enumerate(alerts, 1)]
    if overrides:
        checks.append(_check("DOSE_CHANGED", "info", "Doses changed by you: "
                             + ", ".join(f"{c} -> {o['dose']}" for c, o in overrides.items()) + "."))
    from app import safety
    for g in safety.global_alerts():
        checks.append(_check("GLOBAL_SAFETY", "info", g))
    level = dosing.level_for_nss(a.nss)
    return {
        "summary": f"Severity level {level['level']} ({level['name']}): default dose {dosing.dose_string(level)}.",
        "nss": a.nss, "severity_level": level["level"], "severity_name": level["name"],
        "default_dose": dosing.dose_string(level),
        "modules": rows,
        "ks_protocol": dosing.ks_protocol() if any_bowel else None,
        "reassess_every_days": dosing.reassessment_days(),
        "safety_alerts": alerts,
        "checks": checks,
    }


def _compute_dosing(state: dict) -> dict:
    return _dosing_view(state)


async def _compute(state: dict, step: str) -> dict:
    if step == "intake":
        return _compute_intake(state)
    if step == "red_flags":
        return _compute_red_flags(state)
    if step == "analysis":
        return await _compute_analysis(state)
    if step == "modules":
        return _compute_modules(state)
    if step == "dosing":
        return _compute_dosing(state)
    raise WorkflowError(422, f"Unknown step '{step}'.")


async def _enter(state: dict, step: str) -> None:
    res = await _compute(state, step)
    state["steps"][step] = {"status": "awaiting_confirmation", "result": res, "adjustments": [],
                            "confirmation": None}
    state["current_step"] = step


# ============================================================================ public operations
async def start(patient: PatientInput, report_language: str | None, actor: dict) -> dict:
    state = {"version": 1, "patient": patient.model_dump(mode="json"), "report_language": report_language,
             "steps": {s: {"status": "pending"} for s in STEPS}, "current_step": "intake",
             "history": [], "report_id": None}
    await _enter(state, "intake")
    _log(state, actor, "started", "intake")
    return state


def _require_current(state: dict, step: str) -> None:
    if state.get("current_step") == "completed":
        raise WorkflowError(409, "This plan session is complete; the draft plan has been generated.",
                            {"report_id": state.get("report_id")})
    if step != state.get("current_step"):
        raise WorkflowError(409, f"Step '{step}' is not the current step ('{state.get('current_step')}'). "
                                 "Refresh the session and act on the current step.",
                            {"current_step": state.get("current_step")})


async def confirm(state: dict, step: str, acknowledge: list[str], note: str | None, actor: dict,
                  finalize) -> dict:
    """Confirm the current step. `finalize(state)` is awaited after the last step and must return a
    report id (the router persists the draft report with ownership)."""
    _require_current(state, step)
    checks = state["steps"][step]["result"].get("checks", [])
    blocking = [c for c in checks if c["severity"] == "blocking"]
    if blocking:
        raise WorkflowError(409, "This step cannot be confirmed until these issues are fixed: "
                                 + "; ".join(c["message"] for c in blocking), {"blocking": blocking})
    needed = [c["code"] for c in checks if c.get("requires_ack")]
    missing = [c for c in needed if c not in set(acknowledge or [])]
    if missing:
        raise WorkflowError(409, "Please review and acknowledge these checks before confirming: "
                                 + ", ".join(missing), {"missing_acknowledgements": missing})
    if step == "modules" and any(c["code"] == "JUSTIFICATION_REQUIRED" for c in checks) and not (note or "").strip():
        raise WorkflowError(409, "This module selection needs a justification: add it as the confirmation note.")
    state["steps"][step]["status"] = "confirmed"
    state["steps"][step]["confirmation"] = {"actor": actor, "at": _now(), "acknowledged": needed,
                                            "note": note}
    _log(state, actor, "confirmed", step, acknowledged=needed, note=note)
    idx = STEPS.index(step)
    if idx + 1 < len(STEPS):
        await _enter(state, STEPS[idx + 1])
    else:
        state["report_id"] = await finalize(state)
        state["current_step"] = "completed"
        _log(state, actor, "completed", step, report_id=state["report_id"])
    return state


def _reason(act: dict) -> tuple[str, str | None]:
    rc = act.get("reason_code")
    if rc not in REASON_CODES:
        raise WorkflowError(422, f"Every adjustment needs a reason_code, one of: {', '.join(REASON_CODES)}.")
    return rc, (act.get("rationale") or None)


async def adjust(state: dict, step: str, actions: list[dict], case: PatientInput | None, actor: dict) -> dict:
    _require_current(state, step)
    allowed = ADJUST_ACTIONS[step]
    if step == "intake":
        if case is None:
            raise WorkflowError(422, "To adjust the case data, send the corrected case.")
        before = state["patient"]
        state["patient"] = case.model_dump(mode="json")
        changed = sorted(k for k in state["patient"] if state["patient"].get(k) != before.get(k))
        await _enter(state, "intake")
        state["steps"]["intake"]["adjustments"].append(
            {"action": "update_case", "changed_fields": changed, "actor": actor, "at": _now()})
        _log(state, actor, "adjusted", step, action="update_case", changed_fields=changed)
        return state
    if not actions:
        raise WorkflowError(422, f"No adjustment actions given. Allowed at this step: {', '.join(allowed) or 'none'}.")
    records = []
    for act in actions:
        kind = act.get("action")
        if kind not in allowed:
            raise WorkflowError(422, f"'{kind}' is not allowed at step '{step}'. Allowed: {', '.join(allowed) or 'none'}"
                                     + ("" if allowed else " — go back to the case data step to correct the input."))
        rc, why = _reason(act)
        rec = {"action": kind, "reason_code": rc, "rationale": why, "actor": actor, "at": _now()}
        if step == "analysis":
            code = (act.get("axis_code") or "").upper()
            codes = {a["axis_code"] for a in state["analysis"]["axis_scores"]}
            if code not in codes:
                raise WorkflowError(422, f"Axis {code or '(missing)'} is not in this analysis. Assessed axes: "
                                         + ", ".join(sorted(codes)))
            dec = state["axis_decisions"]
            if kind == "exclude_axis":
                dec["included"].pop(code, None)
                dec["excluded"][code] = {"reason_code": rc, "rationale": why}
            else:
                if code in dec["excluded"]:
                    dec["excluded"].pop(code)
                else:
                    dec["included"][code] = {"reason_code": rc, "rationale": why}
            rec["axis_code"] = code
        elif step == "modules":
            rec.update(_adjust_module(state, kind, (act.get("module_code") or "").strip()))
        elif step == "dosing":
            code = (act.get("module_code") or "").strip()
            new = (act.get("new_dose") or "").strip()
            cur = {m["module_code"]: m for m in state["plan"].get("dosed", [])}
            if code not in cur:
                raise WorkflowError(422, f"Module {code or '(missing)'} is not in the plan. Plan modules: "
                                         + ", ".join(cur))
            if not new:
                raise WorkflowError(422, "change_dose needs new_dose.")
            prev = state["dose_overrides"].get(code, {}).get("dose") or cur[code].get("dose")
            state["dose_overrides"][code] = {"dose": new, "reason_code": rc, "rationale": why}
            rec.update({"module_code": code, "before": prev, "after": new})
        records.append(rec)
    view = {"analysis": _analysis_view, "modules": _modules_view, "dosing": _dosing_view}[step](state)
    st = state["steps"][step]
    st["result"], st["status"] = view, "awaiting_confirmation"
    st["adjustments"].extend(records)
    _log(state, actor, "adjusted", step, actions=[{k: v for k, v in r.items() if k not in ("actor", "at")}
                                                  for r in records])
    return state


def _adjust_module(state: dict, kind: str, code: str) -> dict:
    plan = state["plan"]
    sel, con = plan["modules"], plan["considered"]
    if kind == "remove_module":
        hit = next((m for m in sel if m["module_code"] == code), None)
        if not hit:
            raise WorkflowError(422, f"Module {code or '(missing)'} is not selected. Selected: "
                                     + ", ".join(m["module_code"] for m in sel))
        sel.remove(hit)
        hit["reason_not_selected"] = "REMOVED_BY_CLINICIAN"
        con.insert(0, hit)
        return {"module_code": code, "module_name": hit.get("module_name")}
    # add_module
    scr = state.get("red_flag_screen") or {}
    if not scr.get("module_plan_allowed", True):
        raise WorkflowError(409, "The module plan is blocked by an emergency red flag; modules cannot be added.")
    if any(m["module_code"] == code for m in sel):
        raise WorkflowError(422, f"Module {code} is already selected.")
    hit = next((m for m in con if m["module_code"] == code), None)
    if hit:
        if hit.get("module_status") == "EXCLUDED":
            raise WorkflowError(409, f"Module {code} was excluded by the safety gate ({hit.get('reason_not_selected')}) "
                                     "and cannot be added here.")
        con.remove(hit)
        hit["reason_not_selected"] = None
        hit["match_notes"] = [*(hit.get("match_notes") or []), "Selected by clinician"]
    else:
        hit = _registry_pick(state, code)
    sel.append(hit)
    from app.pipeline import sequencer
    plan["modules"] = _dump(sequencer.sequence(_mods(sel)))
    return {"module_code": code, "module_name": hit.get("module_name")}


def _registry_pick(state: dict, code: str) -> dict:
    from app.knowledge.repository import _registry
    reg = _registry()
    m = reg.get(code)
    if not m or code == "_meta":
        raise WorkflowError(422, f"Module {code or '(missing)'} is not in the module registry.")
    if m.get("status", "active") != "active":
        raise WorkflowError(409, f"Module {code} is {m.get('status')} in the registry and cannot be prescribed.")
    active = {a.axis_code for a in _effective_axes(state)}
    axes = [x for x in (m.get("primary_axes", []) + m.get("secondary_axes", [])) if x in active]
    pick = ModulePick(
        module_code=code, module_name=m.get("name_en"), target_axes=axes,
        phytocore_code=m.get("phytocore_code"), dose_type=m.get("dose_type", "severity"),
        status=m.get("status", "active"),
        axis_roles={x: ("primary" if x in m.get("primary_axes", []) else "secondary") for x in axes},
        contraindications=list(m.get("contraindications") or []), module_match_score=None,
        safety_outcome="PASS_WITH_MONITORING" if m.get("contraindications") else "PASS",
        tier="supporting", match_notes=["Added by clinician from the registry (not engine-matched)"]
        + ([] if axes else ["Targets none of the patient's active axes"]))
    return pick.model_dump(mode="json")


async def back(state: dict, to_step: str, actor: dict) -> dict:
    if state.get("current_step") == "completed":
        raise WorkflowError(409, "This plan session is complete. Edit the generated draft plan instead, "
                                 "or start a new plan session.", {"report_id": state.get("report_id")})
    if to_step not in STEPS:
        raise WorkflowError(422, f"Unknown step '{to_step}'. Steps: {', '.join(STEPS)}.")
    cur = STEPS.index(state["current_step"])
    tgt = STEPS.index(to_step)
    if tgt >= cur:
        raise WorkflowError(409, f"You can only go back to an earlier step (current: {state['current_step']}).")
    st = state["steps"][to_step]
    st["status"], st["confirmation"] = "awaiting_confirmation", None
    discarded = []
    for s in STEPS[tgt + 1:]:
        if state["steps"][s].get("adjustments"):
            discarded.append(s)
        state["steps"][s] = {"status": "pending"}
    state["current_step"] = to_step
    _log(state, actor, "went_back", to_step, from_step=STEPS[cur], cleared_steps=STEPS[tgt + 1:],
         discarded_adjustments_in=discarded)
    return state


async def build_draft(state: dict):
    """Compose + persist the draft report from the confirmed steps (same finalize stage as /report)."""
    from app.pipeline.orchestrator import finalize_report
    patient, analysis = _patient(state), _analysis(state)
    dec = state.get("axis_decisions", {})
    for code, d in dec.get("excluded", {}).items():
        analysis.notes.append(f"Clinician excluded axis {code} from module matching ({d['reason_code']}"
                              + (f": {d['rationale']}" if d.get("rationale") else "") + ").")
    for code, d in dec.get("included", {}).items():
        analysis.notes.append(f"Clinician included hypothesis axis {code} in module matching ({d['reason_code']}).")
    mods = _mods(state["plan"]["dosed"])
    for m in mods:
        o = state.get("dose_overrides", {}).get(m.module_code)
        if o:
            m.match_notes = [*m.match_notes, f"Dose set by clinician ({o['reason_code']})"]
    return await finalize_report(patient, analysis, mods, _mods(state["plan"]["considered"]),
                                 state["plan"].get("selection") or {}, state.get("report_language"),
                                 workflow=workflow_trail(state))


def workflow_trail(state: dict) -> dict:
    """What the final report carries: who confirmed/adjusted what, step by step."""
    return {"mode": "stepwise", "steps": [
        {"step": s, "title": TITLES[s], "confirmation": state["steps"][s].get("confirmation"),
         "adjustments": [{k: v for k, v in a.items()} for a in state["steps"][s].get("adjustments", [])]}
        for s in STEPS],
        "axis_decisions": state.get("axis_decisions", {}),
        "dose_overrides": state.get("dose_overrides", {})}


def view(session: dict) -> dict:
    """Public representation of a session for the API / MCP."""
    state = session["state"]
    cur = state.get("current_step")
    steps = [{"step": s, "number": i + 1, "title": TITLES[s], "status": state["steps"][s].get("status")}
             for i, s in enumerate(STEPS)]
    out = {"session_id": session["id"], "case_id": session["case_id"], "status": session["status"],
           "current_step": cur, "total_steps": len(STEPS), "steps": steps,
           "report_id": state.get("report_id"), "notice": DRAFT_NOTICE}
    if cur in STEPS:
        st = state["steps"][cur]
        res = st.get("result", {})
        checks = res.get("checks", [])
        out["step"] = {
            "step": cur, "number": STEPS.index(cur) + 1, "title": TITLES[cur], "status": st.get("status"),
            "summary": res.get("summary"),
            "result": {k: v for k, v in res.items() if k not in ("checks", "summary")},
            "checks": checks,
            "acknowledgements_required": [c["code"] for c in checks if c.get("requires_ack")],
            "can_confirm": not any(c["severity"] == "blocking" for c in checks),
            "adjust_actions": ADJUST_ACTIONS[cur],
            "adjustments": [{k: v for k, v in a.items() if k != "actor"} for a in st.get("adjustments", [])],
            "next_step": STEPS[STEPS.index(cur) + 1] if STEPS.index(cur) + 1 < len(STEPS) else "generate_draft_plan",
            "can_go_back_to": STEPS[:STEPS.index(cur)],
        }
    return out
