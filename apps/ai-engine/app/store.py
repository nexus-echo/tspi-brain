"""Phase 3 persistence + audit (de-identified). Works on SQLite (dev) and Postgres (prod)."""
from __future__ import annotations

import uuid

from app.knowledge.db import get_session, init_db
from app.knowledge.models import AuditLog, OutcomeRow, PlanSession, Report, ReportValidation


def new_id() -> str:
    return uuid.uuid4().hex[:32]


def audit(action: str, *, case_id: str | None = None, report_id: str | None = None,
          actor: str | None = "mihealth", allowed: bool = True, detail: dict | None = None,
          role: str | None = None, source: str | None = None,
          request_id: str | None = None) -> None:
    init_db()
    s = get_session()
    try:
        s.add(AuditLog(action=action, case_id=case_id, report_id=report_id,
                       actor=actor, allowed=allowed, detail=detail or {},
                       role=role, source=source, request_id=request_id))
        s.commit()
    finally:
        s.close()


def list_audit(*, case_id: str | None = None, report_id: str | None = None,
               actor: str | None = None, action: str | None = None, limit: int = 200) -> list[dict]:
    """Read the append-only audit trail (auditor/reviewer only, enforced at the route)."""
    s = get_session()
    try:
        q = s.query(AuditLog)
        if case_id:
            q = q.filter(AuditLog.case_id == case_id)
        if report_id:
            q = q.filter(AuditLog.report_id == report_id)
        if actor:
            q = q.filter(AuditLog.actor == actor)
        if action:
            q = q.filter(AuditLog.action == action)
        rows = q.order_by(AuditLog.id.desc()).limit(min(limit, 1000)).all()
        return [{"id": r.id, "action": r.action, "case_id": r.case_id, "report_id": r.report_id,
                 "actor": r.actor, "role": getattr(r, "role", None),
                 "source": getattr(r, "source", None), "request_id": getattr(r, "request_id", None),
                 "allowed": r.allowed, "detail": r.detail,
                 "at": r.created_at.isoformat() if r.created_at else None} for r in rows]
    finally:
        s.close()


def save_report(case_id: str, status: str, nss: int, severity_level: int,
                payload: dict, safety_alerts: list | None) -> str:
    init_db()
    rid = new_id()
    s = get_session()
    try:
        s.add(Report(id=rid, case_id=case_id, status=status, nss=nss,
                     severity_level=severity_level, payload=payload,
                     safety_alerts={"alerts": safety_alerts or []}))
        s.commit()
    finally:
        s.close()
    return rid


def get_report(report_id: str) -> dict | None:
    s = get_session()
    try:
        r = s.get(Report, report_id)
        if not r:
            return None
        return {"id": r.id, "case_id": r.case_id, "status": r.status, "nss": r.nss,
                "severity_level": r.severity_level, "payload": r.payload,
                "safety_alerts": (r.safety_alerts or {}).get("alerts", []),
                "owner_clinician_id": getattr(r, "owner_clinician_id", None),
                "owner_case_subject": getattr(r, "owner_case_subject", None),
                "clinic_id": getattr(r, "clinic_id", None),
                "deliverable": r.status == "validated"}
    finally:
        s.close()


# ----------------------------------------------------------------------------- plan sessions
def create_plan_session(case_id: str, state: dict, current_step: str, *,
                        owner_clinician_id: str | None, clinic_id: str | None) -> str:
    init_db()
    sid = new_id()
    s = get_session()
    try:
        s.add(PlanSession(id=sid, case_id=case_id, status="active", current_step=current_step,
                          state=state, owner_clinician_id=owner_clinician_id, clinic_id=clinic_id))
        s.commit()
    finally:
        s.close()
    return sid


def get_plan_session(session_id: str) -> dict | None:
    s = get_session()
    try:
        r = s.get(PlanSession, session_id)
        if not r:
            return None
        return {"id": r.id, "case_id": r.case_id, "status": r.status,
                "current_step": r.current_step, "state": r.state,
                "owner_clinician_id": r.owner_clinician_id, "clinic_id": r.clinic_id,
                "report_id": r.report_id, "deliverable": False,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None}
    finally:
        s.close()


def save_plan_session(session_id: str, *, state: dict, current_step: str, status: str,
                      report_id: str | None = None) -> None:
    s = get_session()
    try:
        r = s.get(PlanSession, session_id)
        if not r:
            return
        r.state = state
        r.current_step = current_step
        r.status = status
        if report_id:
            r.report_id = report_id
        r.updated_at = _now()
        try:
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(r, "state")
        except Exception:  # noqa: BLE001
            pass
        s.commit()
    finally:
        s.close()


def latest_report_id_for_case(case_id: str) -> str | None:
    """Most recent report id for a de-identified case code (None if the case has no report)."""
    s = get_session()
    try:
        r = (s.query(Report.id).filter(Report.case_id == case_id)
             .order_by(Report.created_at.desc()).first())
        return r[0] if r else None
    finally:
        s.close()


def save_identity(case_id: str, blob: str, encrypted: bool) -> None:
    """P4 — upsert encrypted patient identity (PHI) keyed by case_id."""
    from app.knowledge.models import PatientIdentity as _PI
    init_db()
    s = get_session()
    try:
        row = s.get(_PI, case_id)
        if row:
            row.blob, row.encrypted, row.updated_at = blob, encrypted, _now()
        else:
            s.add(_PI(case_id=case_id, blob=blob, encrypted=encrypted))
        s.commit()
    finally:
        s.close()


def get_identity(case_id: str) -> dict | None:
    from app.knowledge.models import PatientIdentity as _PI
    s = get_session()
    try:
        row = s.get(_PI, case_id)
        return {"blob": row.blob, "encrypted": row.encrypted} if row else None
    finally:
        s.close()


def set_report_owner(report_id: str, *, clinician_id: str | None = None,
                     case_subject: str | None = None, clinic_id: str | None = None) -> None:
    """Stamp ownership / tenant scope on a report (Phase B). Nulls are left unchanged."""
    s = get_session()
    try:
        r = s.get(Report, report_id)
        if not r:
            return
        if clinician_id is not None:
            r.owner_clinician_id = clinician_id
        if case_subject is not None:
            r.owner_case_subject = case_subject
        if clinic_id is not None:
            r.clinic_id = clinic_id
        s.commit()
    finally:
        s.close()


def stamp_report_actor(report_id: str, field: str, actor: dict) -> None:
    """Record WHO acted on a report (e.g. created_by) in the report payload (JSON, no migration).
    `actor` is {id, name?, email?}; a timestamp is added here."""
    s = get_session()
    try:
        r = s.get(Report, report_id)
        if not r:
            return
        payload = dict(r.payload or {})
        payload[field] = {**actor, "at": _now().isoformat()}
        r.payload = payload
        try:
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(r, "payload")
        except Exception:  # noqa: BLE001
            pass
        s.commit()
    finally:
        s.close()


def record_validation(report_id: str, doctor_id: str, decision: str,
                      edits: dict | None, approver: dict | None = None) -> dict | None:
    s = get_session()
    try:
        r = s.get(Report, report_id)
        if not r:
            return None
        r.status = {"approve": "validated", "reject": "rejected", "edit": "validated"}.get(decision, r.status)
        edits = dict(edits or {})
        if approver:
            stamped = {**approver, "at": _now().isoformat(), "decision": decision}
            edits["_approved_by"] = stamped
            if decision in ("approve", "edit"):
                payload = dict(r.payload or {})
                payload["approved_by"] = stamped
                r.payload = payload
                try:
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(r, "payload")
                except Exception:  # noqa: BLE001
                    pass
        s.add(ReportValidation(report_id=report_id, doctor_id=doctor_id,
                               decision=decision, edits=edits))
        s.commit()
        return {"report_id": report_id, "status": r.status, "deliverable": r.status == "validated"}
    finally:
        s.close()


def record_override(report_id: str, clinician_id: str, actions: list[dict],
                    editor: dict | None = None) -> dict | None:
    """Apply structured physician edits to a report's plan. NON-DESTRUCTIVE and audited.

    - The original plan is preserved; every edit is appended to payload['clinician_overrides']
      with before/after + reason (5 Aug 'structured clinician overrides' requirement).
    - Editing a plan invalidates any prior approval: the report returns to 'draft' and must be
      re-approved before it is deliverable again.
    """
    s = get_session()
    try:
        r = s.get(Report, report_id)
        if not r:
            return None
        payload = dict(r.payload or {})
        overrides = list(payload.get("clinician_overrides", []))
        modules = payload.get("modules", []) or []
        considered = payload.get("considered_modules", []) or []
        by_code = {m.get("module_code"): m for m in (modules + considered) if isinstance(m, dict)}

        applied: list[dict] = []
        for a in actions:
            act = a.get("action")
            code = a.get("module_code")
            reason_code = a.get("reason_code")
            rationale = a.get("rationale")
            mod = by_code.get(code) if code else None
            before, after = None, None

            if act in ("REMOVE_MODULE", "REJECT_MODULE"):
                if not mod:
                    applied.append({"action": act, "module_code": code, "status": "NOT_FOUND"})
                    continue
                before = {"module_status": mod.get("module_status")}
                mod["module_status"] = "PHYSICIAN_EXCLUDED"
                mod["reason_not_selected"] = "PHYSICIAN_REJECTED"
                mod["physician_removed"] = True
                after = {"module_status": "PHYSICIAN_EXCLUDED"}
            elif act == "CHANGE_DOSE":
                if not mod:
                    applied.append({"action": act, "module_code": code, "status": "NOT_FOUND"}); continue
                before = {"dose": mod.get("dose")}
                mod["dose"] = a.get("new_dose")
                mod["dose_overridden_by_clinician"] = True
                after = {"dose": a.get("new_dose")}
            elif act == "OVERRIDE_SAFETY":
                if not mod:
                    applied.append({"action": act, "module_code": code, "status": "NOT_FOUND"}); continue
                before = {"safety_outcome": mod.get("safety_outcome")}
                mod["safety_outcome"] = "PASS_WITH_MONITORING"
                mod["safety_overridden_by_clinician"] = True
                after = {"safety_outcome": "PASS_WITH_MONITORING"}
            elif act in ("ADD_SECONDARY_AXIS", "REMOVE_SECONDARY_AXIS"):
                if not mod:
                    applied.append({"action": act, "module_code": code, "status": "NOT_FOUND"}); continue
                axes = list(mod.get("target_axes", []) or [])
                before = {"target_axes": list(axes)}
                ax = a.get("axis_code")
                if act == "ADD_SECONDARY_AXIS" and ax and ax not in axes:
                    axes.append(ax)
                if act == "REMOVE_SECONDARY_AXIS" and ax in axes:
                    axes.remove(ax)
                mod["target_axes"] = axes
                after = {"target_axes": axes}
            elif act == "ADD_NOTE":
                pass  # free-text note captured in the override record below
            else:
                applied.append({"action": act, "status": "UNSUPPORTED_ACTION"}); continue

            entry = {"action": act, "module_code": code, "axis_code": a.get("axis_code"),
                     "reason_code": reason_code, "rationale": rationale,
                     "before": before, "after": after,
                     "clinician_id": clinician_id,
                     "clinician_name": (editor or {}).get("name"),
                     "clinician_email": (editor or {}).get("email"),
                     "at": _now().isoformat()}
            overrides.append(entry)
            applied.append({"action": act, "module_code": code, "status": "APPLIED"})

        payload["clinician_overrides"] = overrides
        if editor:
            payload["last_edited_by"] = {**editor, "at": _now().isoformat()}
        payload["plan_version"] = int(payload.get("plan_version", 1)) + 1
        # editing invalidates prior approval -> must be re-approved
        was_validated = r.status == "validated"
        r.status = "draft"
        r.payload = payload
        try:
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(r, "payload")
        except Exception:  # noqa: BLE001
            pass
        s.commit()
        return {"report_id": report_id, "plan_version": payload["plan_version"],
                "applied": applied, "overrides_total": len(overrides),
                "status": r.status, "deliverable": False,
                "reapproval_required": was_validated,
                "notice": "Plan edited by clinician. Re-approval required before it is deliverable."}
    finally:
        s.close()


def save_outcome(report_id: str, marker: str, baseline: float, followup: float) -> dict:
    init_db()
    delta = followup - baseline
    s = get_session()
    try:
        s.add(OutcomeRow(report_id=report_id, marker=marker, baseline=baseline,
                         followup=followup, delta=delta))
        s.commit()
    finally:
        s.close()
    return {"report_id": report_id, "marker": marker, "delta": delta}


# --- Phase 4: axis weights (learning loop) ---
from app.knowledge.models import AxisWeight  # noqa: E402

_WEIGHTS_CACHE: dict[str, float] | None = None


def get_axis_weights() -> dict[str, float]:
    """Cached {axis_code: weight}. Missing axes default to 1.0 at the call site."""
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is None:
        init_db()
        s = get_session()
        try:
            _WEIGHTS_CACHE = {w.axis_code: w.weight for w in s.query(AxisWeight).all()}
        finally:
            s.close()
    return dict(_WEIGHTS_CACHE)


def set_axis_weight(axis_code: str, weight: float, samples: int = 0) -> None:
    init_db()
    s = get_session()
    try:
        w = s.get(AxisWeight, axis_code)
        if w:
            w.weight = weight
            w.samples = samples
        else:
            s.add(AxisWeight(axis_code=axis_code, weight=weight, samples=samples))
        s.commit()
    finally:
        s.close()
    invalidate_weights_cache()


def invalidate_weights_cache() -> None:
    global _WEIGHTS_CACHE
    _WEIGHTS_CACHE = None


def outcomes_for_case(case_id: str) -> list[dict]:
    """All outcomes for a case (joined via its reports), oldest first."""
    s = get_session()
    try:
        rows = (s.query(OutcomeRow, Report.case_id)
                .join(Report, OutcomeRow.report_id == Report.id)
                .filter(Report.case_id == case_id)
                .order_by(OutcomeRow.created_at.asc()).all())
        return [{"marker": o.marker, "baseline": o.baseline, "followup": o.followup,
                 "delta": o.delta, "report_id": o.report_id} for o, _cid in rows]
    finally:
        s.close()


def all_outcomes() -> list[dict]:
    s = get_session()
    try:
        return [{"marker": o.marker, "baseline": o.baseline, "followup": o.followup,
                 "delta": o.delta} for o in s.query(OutcomeRow).all()]
    finally:
        s.close()


def unlearned_outcomes() -> list[dict]:
    s = get_session()
    try:
        rows = s.query(OutcomeRow).filter(OutcomeRow.learned == False).all()  # noqa: E712
        return [{"id": o.id, "marker": o.marker, "delta": o.delta} for o in rows]
    finally:
        s.close()


def mark_learned(ids: list[int]) -> None:
    s = get_session()
    try:
        for oid in ids:
            o = s.get(OutcomeRow, oid)
            if o:
                o.learned = True
        s.commit()
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Phase 12 — 3-level learning store (patient-specific / network behaviour / patient model)
# + population proposals (never auto-applied).
# ---------------------------------------------------------------------------
from datetime import datetime, timezone  # noqa: E402

from app.knowledge.models import (  # noqa: E402
    LearningRecord,
    ModelUpdateProposal,
    NetworkObservation,
    PatientAxisWeight,
    PatientModel,
)


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --- Level 1: patient-specific axis weights -------------------------------
def get_patient_axis_weights(case_id: str) -> dict[str, float]:
    init_db()
    s = get_session()
    try:
        return {w.axis_code: w.weight
                for w in s.query(PatientAxisWeight).filter_by(case_id=case_id).all()}
    finally:
        s.close()


def get_patient_axis_row(case_id: str, axis_code: str) -> dict | None:
    init_db()
    s = get_session()
    try:
        w = s.get(PatientAxisWeight, {"case_id": case_id, "axis_code": axis_code})
        if not w:
            return None
        return {"case_id": w.case_id, "axis_code": w.axis_code, "weight": w.weight,
                "cumulative_adjustment": w.cumulative_adjustment,
                "consecutive_improvements": w.consecutive_improvements, "samples": w.samples}
    finally:
        s.close()


def upsert_patient_axis_weight(case_id: str, axis_code: str, weight: float,
                               cumulative_adjustment: float, consecutive_improvements: int,
                               samples: int) -> None:
    init_db()
    s = get_session()
    try:
        w = s.get(PatientAxisWeight, {"case_id": case_id, "axis_code": axis_code})
        if w:
            w.weight = weight
            w.cumulative_adjustment = cumulative_adjustment
            w.consecutive_improvements = consecutive_improvements
            w.samples = samples
            w.updated_at = _now()
        else:
            s.add(PatientAxisWeight(case_id=case_id, axis_code=axis_code, weight=weight,
                                    cumulative_adjustment=cumulative_adjustment,
                                    consecutive_improvements=consecutive_improvements,
                                    samples=samples, updated_at=_now()))
        s.commit()
    finally:
        s.close()


# --- Level 2: network behaviour -------------------------------------------
def add_network_observation(case_id: str, changed_first: str, direction: str,
                            downstream: str | None = None, magnitude: float | None = None,
                            time_to_response_days: int | None = None, repeated: bool = False,
                            confounders: str | None = None) -> int:
    init_db()
    s = get_session()
    try:
        row = NetworkObservation(case_id=case_id, changed_first=changed_first, downstream=downstream,
                                 direction=direction, magnitude=magnitude,
                                 time_to_response_days=time_to_response_days, repeated=repeated,
                                 confounders=confounders, created_at=_now())
        s.add(row)
        s.commit()
        return row.id
    finally:
        s.close()


def network_observations(case_id: str | None = None) -> list[dict]:
    init_db()
    s = get_session()
    try:
        q = s.query(NetworkObservation)
        if case_id:
            q = q.filter_by(case_id=case_id)
        return [{"id": r.id, "case_id": r.case_id, "changed_first": r.changed_first,
                 "downstream": r.downstream, "direction": r.direction, "magnitude": r.magnitude,
                 "time_to_response_days": r.time_to_response_days, "repeated": r.repeated}
                for r in q.order_by(NetworkObservation.created_at).all()]
    finally:
        s.close()


# --- Level 3: versioned patient biological model ---------------------------
def latest_patient_model(case_id: str) -> dict | None:
    init_db()
    s = get_session()
    try:
        m = (s.query(PatientModel).filter_by(case_id=case_id)
             .order_by(PatientModel.version.desc()).first())
        if not m:
            return None
        return {"case_id": m.case_id, "version": m.version, "change_kind": m.change_kind,
                "axis_state": m.axis_state, "network_state": m.network_state,
                "trigger": m.trigger, "superseded": m.superseded}
    finally:
        s.close()


def save_patient_model(case_id: str, axis_state: dict, network_state: dict,
                       change_kind: str = "CONFIRM", trigger: str | None = None) -> int:
    """Append a NEW version. Old versions are never deleted (audit history)."""
    init_db()
    s = get_session()
    try:
        prev = (s.query(PatientModel).filter_by(case_id=case_id)
                .order_by(PatientModel.version.desc()).first())
        version = (prev.version + 1) if prev else 1
        if prev:
            prev.superseded = True                 # marked, NOT deleted
        s.add(PatientModel(case_id=case_id, version=version, change_kind=change_kind,
                           axis_state=axis_state, network_state=network_state,
                           trigger=trigger, created_at=_now()))
        s.commit()
        return version
    finally:
        s.close()


def patient_model_history(case_id: str) -> list[dict]:
    init_db()
    s = get_session()
    try:
        return [{"version": m.version, "change_kind": m.change_kind, "trigger": m.trigger,
                 "superseded": m.superseded, "created_at": str(m.created_at)}
                for m in (s.query(PatientModel).filter_by(case_id=case_id)
                          .order_by(PatientModel.version).all())]
    finally:
        s.close()


# --- Level 4: population proposals (propose-only) ---------------------------
def add_proposal(axis_code: str, current_weight: float, proposed_weight: float, sample_size: int,
                 improved: int, worsened: int, rationale: str, blocking_reasons: str) -> int:
    init_db()
    s = get_session()
    try:
        p = ModelUpdateProposal(axis_code=axis_code, current_weight=current_weight,
                                proposed_weight=proposed_weight, sample_size=sample_size,
                                improved=improved, worsened=worsened, rationale=rationale,
                                blocking_reasons=blocking_reasons,
                                status="PROPOSE_MODEL_UPDATE", created_at=_now())
        s.add(p)
        s.commit()
        return p.id
    finally:
        s.close()


def list_proposals(status: str | None = None) -> list[dict]:
    init_db()
    s = get_session()
    try:
        q = s.query(ModelUpdateProposal)
        if status:
            q = q.filter_by(status=status)
        return [{"id": p.id, "axis_code": p.axis_code, "current_weight": p.current_weight,
                 "proposed_weight": p.proposed_weight, "sample_size": p.sample_size,
                 "improved": p.improved, "worsened": p.worsened, "rationale": p.rationale,
                 "blocking_reasons": p.blocking_reasons, "status": p.status,
                 "reviewed_by": p.reviewed_by}
                for p in q.order_by(ModelUpdateProposal.created_at.desc()).all()]
    finally:
        s.close()


def decide_proposal(proposal_id: int, reviewer: str, approve: bool) -> dict | None:
    """Apply a proposal to the GLOBAL model only after explicit clinical approval."""
    init_db()
    s = get_session()
    try:
        p = s.get(ModelUpdateProposal, proposal_id)
        if not p:
            return None
        p.status = "APPROVED" if approve else "REJECTED"
        p.reviewed_by = reviewer
        p.review_date = _now()
        s.commit()
        out = {"id": p.id, "axis_code": p.axis_code, "status": p.status,
               "proposed_weight": p.proposed_weight, "reviewed_by": reviewer}
    finally:
        s.close()
    if approve:
        set_axis_weight(out["axis_code"], out["proposed_weight"], samples=0)   # now (and only now)
    return out


# --- learning records ------------------------------------------------------
def add_learning_record(case_id: str, **kw) -> int:
    init_db()
    s = get_session()
    try:
        r = LearningRecord(case_id=case_id, created_at=_now(), **kw)
        s.add(r)
        s.commit()
        return r.id
    finally:
        s.close()
