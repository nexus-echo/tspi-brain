"""HTTP surface of the brain. Thin layer — real work lives in app/pipeline + app/store.

Phase 3: consent enforcement (+ audit), report persistence, doctor-validation workflow,
deliverable gating, outcome capture.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import deid, learning, red_flags, store
from app.auth import Principal, check_report_access, get_principal, require_role
from app.config import settings
from app.pipeline import extraction
from app.pipeline.orchestrator import analyze, build_report
from app.schemas import (
    AnalysisResult,
    AskRequest,
    AskResponse,
    CaseReport,
    OutcomeRecord,
    OverrideRequest,
    PatientInput,
    PlanSessionStart,
    PlanStepAdjust,
    PlanStepBack,
    PlanStepConfirm,
    ValidationRequest,
    ExtractionResult,
)

router = APIRouter()


def _require_consent(p: PatientInput, scope: str) -> None:
    if not p.consent.get(scope, scope == "ai_analysis" and False):
        store.audit("consent_denied", case_id=p.case_id, allowed=False, detail={"scope": scope})
        raise HTTPException(status_code=403, detail=f"Consent '{scope}' not granted.")


def _audit_ctx(p: Principal) -> dict:
    return {"role": p.role, "source": p.source, "request_id": p.request_id}


def _actor(p: Principal, fallback_id: str | None = None) -> dict:
    """Who performed an action — for report attribution (created/approved/edited by).
    Uses the authenticated principal; falls back to the request-supplied id when auth is off."""
    return {"id": (p.id if settings.auth_enabled else (fallback_id or p.id)),
            "name": p.name, "email": p.email, "role": p.role}


@router.post("/screen", tags=["safety"])
async def screen_endpoint(patient: PatientInput,
                          p: Principal = Depends(require_role(
                              "patient", "clinic_staff", "clinician", "reviewer",
                              "service"))) -> dict:
    """Phase 7 — deterministic red-flag screening. Runs before any TSPI reasoning.

    Conservative by design: it may only escalate care, never withhold it.
    """
    result = red_flags.screen(patient)
    store.audit("red_flag_screen", case_id=patient.case_id, actor=p.id,
                detail={"action_class": result.get("action_class"),
                        "count": len(result.get("red_flags", []))}, **_audit_ctx(p))
    return result


@router.post("/analyze", response_model=AnalysisResult, tags=["diagnosis"])
async def analyze_endpoint(patient: PatientInput,
                           p: Principal = Depends(require_role(
                               "patient", "clinic_staff", "clinician", "reviewer",
                               "service"))) -> AnalysisResult:
    _require_consent(patient, "ai_analysis")
    patient = deid.intake(patient)          # P4: strip + store PII before any pipeline/LLM step
    result = await analyze(patient)
    store.audit("analyze", case_id=patient.case_id, actor=p.id,
                detail={"nss": result.nss}, **_audit_ctx(p))
    return result


@router.post("/report", response_model=CaseReport, tags=["diagnosis"])
async def report_endpoint(patient: PatientInput,
                          report_language: str | None = None,
                          p: Principal = Depends(require_role(
                              "patient", "clinic_staff", "clinician", "reviewer",
                              "service"))) -> CaseReport:
    _require_consent(patient, "ai_analysis")
    patient = deid.intake(patient)          # P4: strip + store PII before any pipeline/LLM step
    report = await build_report(patient, report_language=report_language)
    # Phase B — stamp ownership/tenant scope (de-identified keys, never PII).
    # Self-service patients (TSPI Digital): the patient's own user id is the case subject, so only
    # they (and reviewing clinicians) can read it. The report stays an unvalidated AI draft
    # (deliverable=false) until a TSPI clinician approves it; patients can never write to it.
    store.set_report_owner(
        report.report_id,
        clinician_id=(p.id if p.role in ("clinician", "reviewer") else None),
        case_subject=(p.id if p.role == "patient" else patient.case_id),
        clinic_id=p.clinic_id)
    store.stamp_report_actor(report.report_id, "created_by", _actor(p))   # who generated the plan
    store.audit("report", case_id=patient.case_id, report_id=report.report_id,
                actor=p.id, **_audit_ctx(p))
    return report


@router.get("/reports/{report_id}", tags=["workflow"])
async def get_report_endpoint(report_id: str,
                              p: Principal = Depends(require_role(
                                  "patient", "clinic_staff", "clinician", "reviewer",
                                  "auditor", "service"))) -> dict:
    rep = store.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")
    check_report_access(p, rep, write=False)          # tenant / approved-only gate
    return rep


@router.get("/reports/{report_id}/identified", tags=["workflow"])
async def get_identified_report_endpoint(
        report_id: str,
        p: Principal = Depends(require_role(
            "patient", "clinic_staff", "clinician", "reviewer", "service"))) -> dict:
    """Return the report with PII re-attached (render-time re-identification). NEVER served to the
    chat/MCP surface — that would leak PII to the chat provider. MiHealth/portal only."""
    if p.source == "mcp":
        raise HTTPException(status_code=403,
                            detail="Identified reports are not available over the chat/MCP surface.")
    rep = store.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")
    check_report_access(p, rep, write=False)
    rep["report_markdown"] = deid.reidentify(rep.get("payload", {}).get("report_markdown", ""),
                                             rep["case_id"])
    rep["identified"] = True
    store.audit("identified_report", report_id=report_id, actor=p.id, **_audit_ctx(p))
    return rep


@router.get("/reports/{report_id}/render", tags=["workflow"])
async def render_report_endpoint(
        report_id: str, language: str | None = None, audience: str = "physician",
        identified: bool = False,
        p: Principal = Depends(require_role(
            "patient", "clinic_staff", "clinician", "reviewer", "service"))) -> dict:
    """Render the report from the common template. language: en|th · audience: physician|patient.
    De-identified by default (placeholders). identified=true re-attaches PII — never over chat/MCP,
    and the patient audience requires an approved (deliverable) report."""
    from app import report_render
    rep = store.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")
    check_report_access(p, rep, write=False)
    if audience == "patient" and not rep.get("deliverable"):
        raise HTTPException(status_code=409, detail="Patient report requires physician approval first.")
    md = report_render.render(rep.get("payload", {}), language=language, audience=audience)
    if identified:
        if p.source == "mcp":
            raise HTTPException(status_code=403, detail="Identified render not available over chat/MCP.")
        md = deid.reidentify(md, rep["case_id"])
    store.audit("render_report", report_id=report_id, actor=p.id,
                detail={"language": language, "audience": audience, "identified": identified}, **_audit_ctx(p))
    return {"report_id": report_id, "language": (language or rep.get("payload", {}).get("report_language", "en")),
            "audience": audience, "identified": identified, "markdown": md}


@router.post("/validate", tags=["workflow"])
async def validate_endpoint(req: ValidationRequest,
                            p: Principal = Depends(require_role(
                                "clinician", "reviewer", "service"))) -> dict:
    existing = store.get_report(req.report_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Report not found.")
    check_report_access(p, existing, write=True)      # only owner clinician / reviewer
    approver = _actor(p, fallback_id=req.doctor_id)
    result = store.record_validation(req.report_id, req.doctor_id, req.decision, req.edits,
                                     approver=approver)
    store.audit("validate", report_id=req.report_id, actor=approver["id"],
                detail={"decision": req.decision, "by_name": p.name, "by_email": p.email},
                **_audit_ctx(p))
    return result


# ----------------------------------------------------------------------------- plan sessions
# Clinician step-by-step plan workflow. The SAME pipeline as /report, one verified step at a time,
# each confirmed (or adjusted / rolled back) by the clinician. /report stays one-shot for patients,
# staff, MiHealth and the portal.
_PLAN_ROLES = ("clinician", "reviewer", "service")


def _plan_principal(p: Principal = Depends(get_principal)) -> Principal:
    if settings.auth_enabled and p.role not in _PLAN_ROLES:
        raise HTTPException(status_code=403, detail=(
            "Step-by-step plan sessions are for clinicians and reviewers. Use the standard plan "
            "generation instead."))
    return p


def _wf_error(e) -> HTTPException:
    return HTTPException(status_code=e.status, detail={"message": e.message, **e.detail})


def _load_session(session_id: str, p: Principal) -> dict:
    sess = store.get_plan_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Plan session not found.")
    check_report_access(p, sess, write=True)          # owner clinician / reviewer only
    if sess["status"] == "cancelled":
        raise HTTPException(status_code=409, detail="This plan session was cancelled.")
    return sess


def _save_session(sess: dict, state: dict) -> dict:
    status = "completed" if state.get("current_step") == "completed" else "active"
    store.save_plan_session(sess["id"], state=state, current_step=state["current_step"],
                            status=status, report_id=state.get("report_id"))
    return {**sess, "state": state, "status": status, "current_step": state["current_step"]}


@router.post("/plan-sessions", tags=["plan-sessions"])
async def start_plan_session(req: PlanSessionStart, p: Principal = Depends(_plan_principal)) -> dict:
    """Start a clinician step-by-step plan for a de-identified case. Returns step 1 (case data
    check) with its verification checks. Nothing is analysed until the clinician confirms."""
    from app import workflow
    _require_consent(req.patient, "ai_analysis")
    patient = deid.intake(req.patient)
    actor = _actor(p)
    state = await workflow.start(patient, req.report_language, actor)
    sid = store.create_plan_session(
        patient.case_id, state, state["current_step"],
        owner_clinician_id=(p.id if p.role in ("clinician", "reviewer") else None), clinic_id=p.clinic_id)
    store.audit("plan_session_start", case_id=patient.case_id, actor=p.id,
                detail={"session_id": sid}, **_audit_ctx(p))
    return workflow.view(store.get_plan_session(sid))


@router.get("/plan-sessions/{session_id}", tags=["plan-sessions"])
async def get_plan_session(session_id: str, p: Principal = Depends(_plan_principal)) -> dict:
    from app import workflow
    return workflow.view(_load_session(session_id, p))


@router.post("/plan-sessions/{session_id}/confirm", tags=["plan-sessions"])
async def confirm_plan_step(session_id: str, req: PlanStepConfirm,
                            p: Principal = Depends(_plan_principal)) -> dict:
    """Confirm the CURRENT step (acknowledging every check that requires it). The engine then
    computes and verifies the next step. Confirming the last step (dosing) generates the DRAFT
    report — it still needs physician approval (POST /validate)."""
    from app import workflow
    sess = _load_session(session_id, p)
    actor = _actor(p)

    async def _finalize(state: dict) -> str:
        report = await workflow.build_draft(state)
        store.set_report_owner(report.report_id,
                               clinician_id=(p.id if p.role in ("clinician", "reviewer") else None),
                               case_subject=sess["case_id"], clinic_id=p.clinic_id)
        store.stamp_report_actor(report.report_id, "created_by", actor)
        store.audit("report", case_id=sess["case_id"], report_id=report.report_id, actor=p.id,
                    detail={"stepwise": True, "session_id": session_id}, **_audit_ctx(p))
        return report.report_id

    try:
        state = await workflow.confirm(sess["state"], req.step, req.acknowledge, req.note, actor, _finalize)
    except workflow.WorkflowError as e:
        raise _wf_error(e) from e
    sess = _save_session(sess, state)
    store.audit("plan_step_confirm", case_id=sess["case_id"], report_id=state.get("report_id"), actor=p.id,
                detail={"session_id": session_id, "step": req.step,
                        "acknowledged": len(req.acknowledge)}, **_audit_ctx(p))
    return workflow.view(sess)


@router.post("/plan-sessions/{session_id}/adjust", tags=["plan-sessions"])
async def adjust_plan_step(session_id: str, req: PlanStepAdjust,
                           p: Principal = Depends(_plan_principal)) -> dict:
    """Apply reason-coded clinician adjustments to the CURRENT step, then re-verify it.
    intake: send the corrected `case` · analysis: exclude_axis / include_axis · modules:
    remove_module / add_module · dosing: change_dose. red_flags has no adjustments (go back to
    intake to correct the data)."""
    from app import workflow
    sess = _load_session(session_id, p)
    case = None
    if req.case is not None:
        if req.case.case_id != sess["case_id"]:
            raise HTTPException(status_code=422, detail="The corrected case must keep the same case_id.")
        _require_consent(req.case, "ai_analysis")
        case = deid.intake(req.case)
    try:
        state = await workflow.adjust(sess["state"], req.step, [a.model_dump() for a in req.actions],
                                      case, _actor(p))
    except workflow.WorkflowError as e:
        raise _wf_error(e) from e
    sess = _save_session(sess, state)
    store.audit("plan_step_adjust", case_id=sess["case_id"], actor=p.id,
                detail={"session_id": session_id, "step": req.step,
                        "actions": [a.action for a in req.actions] or (["update_case"] if case else [])},
                **_audit_ctx(p))
    return workflow.view(sess)


@router.post("/plan-sessions/{session_id}/back", tags=["plan-sessions"])
async def back_plan_step(session_id: str, req: PlanStepBack,
                         p: Principal = Depends(_plan_principal)) -> dict:
    """Re-open an earlier step as it was; every later step is cleared and recomputed on confirm."""
    from app import workflow
    sess = _load_session(session_id, p)
    try:
        state = await workflow.back(sess["state"], req.to_step, _actor(p))
    except workflow.WorkflowError as e:
        raise _wf_error(e) from e
    sess = _save_session(sess, state)
    store.audit("plan_step_back", case_id=sess["case_id"], actor=p.id,
                detail={"session_id": session_id, "to_step": req.to_step}, **_audit_ctx(p))
    return workflow.view(sess)


@router.post("/plan-sessions/{session_id}/cancel", tags=["plan-sessions"])
async def cancel_plan_session(session_id: str, p: Principal = Depends(_plan_principal)) -> dict:
    from app import workflow
    sess = _load_session(session_id, p)
    if sess["status"] == "completed":
        raise HTTPException(status_code=409, detail="A completed session cannot be cancelled.")
    store.save_plan_session(session_id, state=sess["state"], current_step=sess["current_step"],
                            status="cancelled")
    store.audit("plan_session_cancel", case_id=sess["case_id"], actor=p.id,
                detail={"session_id": session_id}, **_audit_ctx(p))
    return {**workflow.view({**sess, "status": "cancelled"}), "status": "cancelled"}


@router.post("/reports/{report_id}/override", tags=["workflow"])
async def override_endpoint(report_id: str, req: OverrideRequest,
                            p: Principal = Depends(require_role(
                                "clinician", "reviewer", "service"))) -> dict:
    """Apply structured physician edits to a draft plan (remove/reject module, change dose,
    override safety with justification, add/remove secondary axis, add note). NON-DESTRUCTIVE:
    the original plan and every edit are preserved with reason codes. Editing invalidates any
    prior approval — the plan returns to draft and must be re-approved."""
    existing = store.get_report(report_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    check_report_access(p, existing, write=True)
    editor = _actor(p, fallback_id=req.clinician_id)
    result = store.record_override(
        report_id, req.clinician_id, [a.model_dump() for a in req.actions], editor=editor)
    store.audit("override", report_id=report_id, actor=editor["id"],
                detail={"actions": [a.action for a in req.actions],
                        "plan_version": result.get("plan_version"),
                        "by_name": p.name, "by_email": p.email}, **_audit_ctx(p))
    return result


@router.get("/audit", tags=["workflow"])
async def audit_endpoint(case_id: str | None = None, report_id: str | None = None,
                         actor: str | None = None, action: str | None = None, limit: int = 200,
                         p: Principal = Depends(require_role("auditor", "reviewer"))) -> dict:
    """Read the append-only audit trail. Auditor/reviewer only."""
    return {"entries": store.list_audit(case_id=case_id, report_id=report_id,
                                        actor=actor, action=action, limit=limit)}


@router.post("/outcome", tags=["learning"])
async def outcome_endpoint(rec: OutcomeRecord,
                           p: Principal = Depends(require_role(
                               "clinician", "reviewer", "service"))) -> dict:
    result = store.save_outcome(rec.report_id, rec.marker, rec.baseline, rec.followup)
    store.audit("outcome", report_id=rec.report_id, detail=result)
    return result


@router.post("/learning/recalibrate", tags=["learning"])
async def recalibrate_endpoint(
        p: Principal = Depends(require_role("reviewer", "service"))) -> dict:
    """Phase 12: PROPOSE-ONLY. Population learning may never auto-change the global model.

    Creates PROPOSE_MODEL_UPDATE proposals for clinical review (expert rule Q11). The previous
    Phase-4 behaviour (auto-recalibrating global axis weights) was non-compliant and is removed.
    """
    result = learning.propose_model_update()
    store.audit("recalibrate", actor="system",
                detail={"proposals_created": result["proposals_created"],
                        "global_model_changed": False})
    return result


@router.get("/learning/proposals", tags=["learning"])
async def list_proposals_endpoint(status: str | None = None) -> dict:
    """Population model-update proposals awaiting clinical review."""
    return learning.list_proposals(status)


@router.post("/learning/proposals/{proposal_id}/decide", tags=["learning"])
async def decide_proposal_endpoint(proposal_id: int, reviewer: str, approve: bool = True,
                                   p: Principal = Depends(require_role("reviewer", "service"))) -> dict:
    """Clinician-gated. This is the ONLY path that may change the global clinical model."""
    result = learning.approve_proposal(proposal_id, reviewer, approve)
    if not result:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    return result


@router.post("/learning/adapt/{case_id}", tags=["learning"])
async def adapt_patient_endpoint(case_id: str) -> dict:
    """Level 1 — patient-specific axis adaptation (bounded +-5%/cycle, +-20% cumulative)."""
    return learning.adapt_patient(case_id)


@router.post("/learning/networks/{case_id}", tags=["learning"])
async def observe_networks_endpoint(case_id: str) -> dict:
    """Level 2 — learn the BEHAVIOUR of biological networks (what moved first, what followed)."""
    return learning.observe_networks(case_id)


@router.get("/patient-model/{case_id}", tags=["learning"])
async def patient_model_endpoint(case_id: str) -> dict:
    """Level 3 — the patient's versioned living biological model (history never deleted)."""
    return learning.patient_model(case_id)


@router.post("/patient-model/{case_id}/rebuild", tags=["learning"])
async def rebuild_patient_model_endpoint(case_id: str, change_kind: str = "REFINE",
                                         trigger: str | None = None,
                                         p: Principal = Depends(require_role("reviewer", "service"))) -> dict:
    """Level 3 — reconstruct the model as a NEW version (CONFIRM/REFINE/REDIRECT/OVERRIDE/CONTRADICT)."""
    return learning.rebuild_patient_model(case_id, change_kind=change_kind, trigger=trigger)


@router.get("/learning/weights", tags=["learning"])
async def weights_endpoint() -> dict:
    return {"axis_weights": store.get_axis_weights()}


@router.get("/temporal/{case_id}", tags=["learning"])
async def temporal_endpoint(case_id: str) -> dict:
    """Phase 4: per-marker trajectory for a case across its outcomes."""
    return learning.temporal(case_id)


@router.get("/knowledge/health", tags=["knowledge"])
async def knowledge_health() -> dict:
    from app.knowledge import corpus
    try:
        kb = corpus.stats()
    except Exception:  # noqa: BLE001 — health must never fail on the KB
        kb = {"documents": 0}
    return {
        "deidentification_enforced": settings.enforce_deidentification,
        "doctor_validation_required": settings.require_doctor_validation,
        "knowledge_base": kb,
    }


_ASK_ROLES = ("patient", "clinic_staff", "clinician", "reviewer", "auditor", "service")


@router.post("/ask", response_model=AskResponse, tags=["knowledge"])
async def ask_endpoint(req: AskRequest,
                       p: Principal = Depends(require_role(*_ASK_ROLES))) -> AskResponse:
    """Ask TSPI Brain a question about a treatment plan and/or the TSPI knowledge base
    (39 axes, 12 domains, 3 keys, modules & products, 180 networks, red-flag/safety rules, dosing,
    official expert answers). Retrieves the relevant knowledge (exact codes + pgvector embeddings +
    BM25) and writes a grounded, cited answer. Without an LLM it returns the retrieved passages.

    Plan scope: pass `report_id` (or a `case_id` to use its latest report). Access follows the same
    tenant rules as GET /reports/{id}. Questions must be de-identified (422 otherwise)."""
    from app.knowledge import qa

    report, outcomes = None, None
    if req.report_id or req.case_id:
        rid = req.report_id or store.latest_report_id_for_case(req.case_id)
        report = store.get_report(rid) if rid else None
        if not report:
            raise HTTPException(status_code=404, detail="No treatment plan found for that report/case.")
        check_report_access(p, report, write=False)
        outcomes = store.outcomes_for_case(report["case_id"])

    audience = req.audience or ("patient" if p.role == "patient" else "clinician")
    result = await qa.answer(req.question, report=report, outcomes=outcomes,
                             history=[h.model_dump() for h in req.history], kinds=req.kinds,
                             top_k=req.top_k, language=req.language, audience=audience,
                             use_llm=req.include_answer)
    # Audit WHO asked and WHAT was in scope — not the question text.
    store.audit("ask", case_id=(report or {}).get("case_id"), report_id=(report or {}).get("id"),
                actor=p.id, detail={"answer_mode": result["answer_mode"],
                                    "sources": [s["doc_id"] for s in result["sources"]][:20],
                                    "kinds": req.kinds}, **_audit_ctx(p))
    return AskResponse(**result)


@router.get("/knowledge/search", tags=["knowledge"])
async def knowledge_search_endpoint(q: str, kinds: str | None = None, k: int = 8,
                                    p: Principal = Depends(require_role(*_ASK_ROLES))) -> dict:
    """Retrieval only: the knowledge-base passages that best match `q` (no LLM).
    `kinds` is a comma-separated filter, e.g. `module,product`."""
    from app.knowledge import qa
    from app.knowledge.search import hybrid_search
    qa.guard_text(q)
    res = await hybrid_search(q, k=max(1, min(k, 50)),
                              kinds=[x.strip() for x in kinds.split(",")] if kinds else None)
    return {"query": q, "vector_used": res["vector_used"], "hits": [
        {"doc_id": h["id"], "kind": h["kind"], "ref_code": h["ref_code"], "title": h["title"],
         "content": h["content"], "score": h["score"], "retrieval": h["retrieval"]} for h in res["hits"]]}


@router.post("/extract", response_model=ExtractionResult, tags=["extraction"])
async def extract_endpoint(
    file: UploadFile = File(...),
    doc_type: str = Form("lab"),
    p: Principal = Depends(require_role(
        "patient", "clinic_staff", "clinician", "reviewer", "service")),
) -> ExtractionResult:
    """Read ONE uploaded report file (image / scanned PDF / text PDF) with LOCAL models and
    return candidate labs + imaging narrative. Governance: local-only; confirmed=False (a human
    must review before it feeds diagnosis). Raw file bytes are processed in-memory, not stored."""
    data = await file.read()
    result = await extraction.extract_from_file(
        data, file.content_type or "", file.filename or "", doc_type
    )
    store.audit("extract", detail={"source": result.source, "engine": result.engine,
                                   "labs": len(result.labs)})
    return result
