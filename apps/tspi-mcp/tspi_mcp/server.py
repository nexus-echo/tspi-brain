"""TSPI AI Brain — FastMCP server (Phase A).

Exposes the existing engine endpoints as MCP tools for a clinician's AI chat client. Phase A uses
ONLY endpoints that already exist (/health, /screen, /analyze, /report, /reports, /validate,
/outcome) — the engine needs no changes and keeps working standalone for MiHealth and any other
REST client.

Governance baked into this layer (not the engine):
  * De-identification gate on every case input (PII never forwarded).
  * Every generated plan is an AI DRAFT — not deliverable until a clinician approves it.
  * The chat/LLM is a presentation layer only: it must relay engine output, never invent
    axis scores, networks, modules, doses, or evidence.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from . import engine_client as engine
from . import ui
from .config import settings
from .deident import PIIError, _scrub_text, deidentify
from .error_logging import ToolErrorLoggingMiddleware, configure_logging

configure_logging()

INSTRUCTIONS = """TSPI AI Brain — deterministic clinical decision-support.

RULES FOR THE ASSISTANT:
- Collect DE-IDENTIFIED data only: a case code, an age band, sex, symptoms, and lab values.
  NEVER ask for or submit a patient's name, date of birth, MRN, phone, email, or address.
- The TSPI engine is the sole source of clinical conclusions. Present its structured output as-is.
  Do NOT invent or estimate axis scores, networks, modules, doses, evidence grades, or outcomes.
- Always run tspi_screen_red_flags first for a new case.
- CLINICIANS/REVIEWERS build plans STEP BY STEP (plan session): tspi_plan_start, then for each step
  show the step's summary, result and checks, and WAIT for the clinician to confirm
  (tspi_plan_confirm_step), adjust (tspi_plan_adjust_step) or go back (tspi_plan_go_back). Never
  confirm a step or acknowledge a check on your own initiative; only acknowledge the check codes the
  clinician has reviewed in this conversation. Steps: 1 case data check, 2 red flags, 3 axes/NSS/
  networks, 4 modules, 5 dosing; confirming step 5 generates the draft plan.
  Patients and clinic staff keep the one-shot tspi_generate_treatment_plan.
- A generated treatment plan is an AI DRAFT pending physician review; it is NOT for patient use
  until a clinician approves it with tspi_approve_treatment_plan.
- If tspi_whoami returns role 'patient' (self-service user), the patient may see their own draft,
  but it must be labelled UNREVIEWED AI DRAFT (not reviewed or approved by a TSPI doctor, not
  medical advice) and they must be told to contact TSPI Digital for doctor review. Never call
  approve, update or record_outcome for a patient.
- For questions about a treatment plan or the TSPI knowledge base (axes, domains, keys, modules,
  products, networks, red-flag/safety rules, dosing), call tspi_ask. Relay its answer with the [S#]
  / [PLAN] citations and its notice; do not add clinical facts that are not in the answer or its
  sources. If answer_mode is 'retrieval_only', summarise ONLY the returned passages.
- Some results include a UI resource (buttons: Generate report, Download PDF, Approve). Place its
  \\ui{...} marker at the end of your reply. A message saying the user clicked a button is a real
  request from the signed-in user: carry it out. "Confirmed with the Approve button" counts as the
  clinician's explicit approval confirmation.
"""

from .identity import build_auth


auth_provider = build_auth()

# FAIL-CLOSED: a network transport (streamable-http/http/sse) is a PUBLIC surface and must never
# run without auth. If auth didn't build (TSPI_MCP_AUTH unset/none, e.g. .env not loaded), refuse
# to start over HTTP instead of silently exposing every tool. stdio (local Claude Desktop) is exempt.
if settings.transport in ("streamable-http", "http", "sse") and auth_provider is None:
    raise RuntimeError(
        "Public MCP transport "
        f"({settings.transport}) requires authentication, but no auth provider was built "
        "(TSPI_MCP_AUTH is 'none' or unset). Set TSPI_MCP_AUTH=workos (with "
        "TSPI_WORKOS_AUTHKIT_DOMAIN + TSPI_MCP_BASE_URL) or =jwt. Refusing to start unauthenticated."
    )

# One concise startup log line — confirms auth actually loaded (the usual failure mode is
# TSPI_MCP_AUTH defaulting to "none" when the .env wasn't found). Logged, not printed.
logging.getLogger("tspi_mcp").info(
    "TSPI MCP starting: transport=%s auth=%s provider=%s base_url=%s",
    settings.transport, settings.mcp_auth,
    type(auth_provider).__name__ if auth_provider else None, settings.mcp_base_url,
)


mcp = FastMCP(
    "tspi-ai-brain",
    instructions=INSTRUCTIONS,
    auth=auth_provider,
)
# Log the full traceback (+ error_id) for any exception escaping a tool. Arguments never logged.
mcp.add_middleware(ToolErrorLoggingMiddleware())

# --------------------------------------------------------------------------- input models
class LabResultIn(BaseModel):
    analyte: str = Field(description="Lab test name, e.g. 'CRP', 'HbA1c', 'Ferritin'.")
    value: float | str = Field(description="Measured value (number, or text for qualitative).")
    unit: str | None = Field(default=None, description="Unit, e.g. 'mg/L'. Include it for derived markers.")
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str | None = Field(default=None, description="Optional 'high' | 'low' if already flagged.")


class PatientCase(BaseModel):
    """A DE-IDENTIFIED patient case. There are deliberately no identity fields."""
    case_id: str = Field(description="A non-identifying case code (NOT a name or MRN), e.g. 'CASE-0912'.")
    age_band: str | None = Field(default=None, description="Age band, e.g. '40s', not an exact DOB.")
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    symptoms: str = Field(default="", description="Free-text clinical symptoms (no identifiers).")
    labs: list[LabResultIn] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    imaging: list[str] = Field(default_factory=list)
    consent_ai_analysis: bool = Field(
        default=True,
        description="Clinician attests the patient consented to AI analysis. Audited; required by the engine.",
    )


_DRAFT_NOTICE = ("AI-GENERATED CLINICAL DRAFT — PENDING PHYSICIAN REVIEW — NOT FOR PATIENT RELEASE. "
                 "Module recommendations are PROVISIONAL until the official Module Registry is approved.")


def _to_engine_payload(case: PatientCase) -> dict:
    raw = case.model_dump(exclude_none=False)
    raw["consent"] = {"ai_analysis": bool(raw.pop("consent_ai_analysis", True))}
    clean, findings = deidentify(raw, strict=settings.strict_deident)
    clean["_deident_findings"] = findings   # popped before sending; surfaced to caller
    return clean


def _forwardable(payload: dict) -> tuple[dict, list[str]]:
    findings = payload.pop("_deident_findings", [])
    return payload, findings


def _role() -> str:
    from .identity import current_identity_ext
    return current_identity_ext()["role"]


def _report_ui(rep: dict, report_id: str | None = None):
    """Button panel for a report dict from the engine (or the generate wrapper)."""
    rid = rep.get("report_id") or rep.get("id") or report_id
    return ui.with_panel(rep, ui.report_panel(rid, rep.get("status"), bool(rep.get("deliverable")),
                                              _role()), f"report/{rid}")


# --------------------------------------------------------------------------- tools
@mcp.tool()
async def tspi_whoami() -> dict:
    """Show the identity the server resolved for the current authenticated session — the fields
    read from your WorkOS token (user id, role, clinic, name, email). Use this to verify the JWT
    template is delivering the expected claims. Read-only; returns only the caller's OWN identity,
    never patient data."""
    from .identity import current_identity_ext
    ident = current_identity_ext()
    return {
        "authenticated": settings.mcp_auth != "none",
        "identity": ident,
        "note": ("These come from your WorkOS token claims (or the static pilot identity on stdio). "
                 "role must be an engine-valid value (e.g. 'clinician') to pass RBAC."),
    }


@mcp.tool()
async def tspi_engine_health() -> dict:
    """Check that the TSPI engine is reachable and report its governance/registry status."""
    try:
        health = await engine.get("/health")
        know = await engine.get("/knowledge/health")
        return {"ok": True, "engine": health, "knowledge": know,
                "note": "Module recommendations are provisional until the Module Registry is approved."}
    except engine.EngineError as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
async def tspi_screen_red_flags(case: PatientCase) -> dict:
    """Run deterministic red-flag / critical-value screening. ALWAYS run this first for a new case.
    It can only escalate care, never withhold it."""
    try:
        payload, findings = _forwardable(_to_engine_payload(case))
    except PIIError as e:
        return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
    try:
        result = await engine.post("/screen", json=payload)
    except engine.EngineError as e:
        return {"error": str(e)}
    if findings:
        result["deidentification_warning"] = f"Redacted possible identifiers: {', '.join(findings)}"
    return ui.with_panel(result, ui.screen_panel(case.case_id, result.get("red_flags"), _role()),
                         f"screen/{case.case_id}")


@mcp.tool()
async def tspi_analyze_case(case: PatientCase) -> dict:
    """Run the TSPI biological analysis (39 axes, differential networks, NSS) WITHOUT producing a
    full treatment plan. Returns the engine's structured assessment. Presentation only — do not
    add or alter any scores."""
    try:
        payload, findings = _forwardable(_to_engine_payload(case))
    except PIIError as e:
        return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
    try:
        result = await engine.post("/analyze", json=payload)
    except engine.EngineError as e:
        return {"error": str(e)}
    out = {"analysis": result}
    if findings:
        out["deidentification_warning"] = f"Redacted possible identifiers: {', '.join(findings)}"
    return ui.with_panel(out, ui.analysis_panel(case.case_id, _role()), f"analysis/{case.case_id}")


@mcp.tool()
async def tspi_generate_treatment_plan(case: PatientCase) -> dict:
    """Generate the full TSPI case report + candidate module plan for a de-identified case.
    The result is an AI DRAFT (deliverable=false) and must be reviewed and approved by a clinician
    before any patient use. Do not present it as a final prescription."""
    try:
        payload, findings = _forwardable(_to_engine_payload(case))
    except PIIError as e:
        return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
    if ui.is_clinical(_role()):
        # Clinicians build plans step by step: start a plan session instead of a one-shot plan.
        return await _start_session(payload, findings, redirected=True)
    try:
        report = await engine.post("/report", json=payload)
    except engine.EngineError as e:
        return {"error": str(e)}
    out = {
        "notice": _DRAFT_NOTICE,
        "report_id": report.get("report_id"),
        "deliverable": report.get("deliverable", False),
        "status": report.get("status", "draft"),
        "plan": report,
        "next_step": "Review, then call tspi_approve_treatment_plan to approve or reject.",
    }
    if findings:
        out["deidentification_warning"] = f"Redacted possible identifiers: {', '.join(findings)}"
    return _report_ui(out)


@mcp.tool()
async def tspi_get_treatment_plan(report_id: str) -> dict:
    """Fetch a previously generated plan by its report_id (to review or continue)."""
    try:
        rep = await engine.get(f"/reports/{report_id}")
    except engine.EngineError as e:
        return {"error": str(e)}
    return _report_ui(rep, report_id)


@mcp.tool()
async def tspi_approve_treatment_plan(
    report_id: str,
    decision: Literal["approve", "edit", "reject"],
    edits: dict | None = None,
    reason: str | None = None,
) -> dict:
    """Record the CLINICIAN's decision on a draft plan. 'approve' makes it deliverable; 'reject'
    blocks it; 'edit' records edits. The clinician identity comes from the authenticated session.
    This is the physician-approval gate — nothing reaches a patient without it.
    Only clinicians and reviewers may call this; the engine enforces the same rule."""
    if not ui.is_clinical(_role()):
        return {"error": "Only a clinician or reviewer can approve, edit or reject a plan.",
                "action_required": "Ask TSPI Digital for doctor review of this report."}
    body = {"report_id": report_id, "doctor_id": settings.clinician_id, "decision": decision,
            "edits": ({"reason": reason, **(edits or {})} if (edits or reason) else None)}
    try:
        result = await engine.post("/validate", json=body)
    except engine.EngineError as e:
        return {"error": str(e)}
    out = {"decision": decision, "by": settings.clinician_id, "result": result}
    try:
        rep = await engine.get(f"/reports/{report_id}")
    except engine.EngineError:
        return out
    return ui.with_panel(out, ui.report_panel(report_id, rep.get("status"),
                                              bool(rep.get("deliverable")), _role()),
                         f"report/{report_id}/{decision}")


class OverrideActionIn(BaseModel):
    """One structured physician edit. Every edit REQUIRES a reason_code."""
    action: Literal["REMOVE_MODULE", "REJECT_MODULE", "CHANGE_DOSE", "OVERRIDE_SAFETY",
                    "ADD_SECONDARY_AXIS", "REMOVE_SECONDARY_AXIS", "ADD_NOTE"]
    module_code: str | None = Field(default=None, description="Target module (for module actions).")
    axis_code: str | None = Field(default=None, description="Axis code (for axis actions).")
    new_dose: str | None = Field(default=None, description="New dose text (for CHANGE_DOSE).")
    reason_code: Literal["NEW_CLINICAL_INFORMATION", "PATIENT_PREFERENCE", "SAFETY_CONCERN",
                         "REGISTRY_ERROR", "ALGORITHM_ERROR", "CLINICAL_JUDGMENT",
                         "DIAGNOSTIC_UNCERTAINTY", "TREATMENT_RESPONSE"]
    rationale: str | None = Field(default=None, description="Short free-text justification.")


@mcp.tool()
async def tspi_update_treatment_plan(report_id: str, actions: list[OverrideActionIn]) -> dict:
    """Apply the CLINICIAN's structured edits to a draft plan: remove/reject a module, change a
    dose, override safety (with justification), add/remove a secondary axis, or add a note. Every
    edit needs a reason_code and is recorded non-destructively. Editing invalidates any prior
    approval — the plan returns to draft and must be re-approved with tspi_approve_treatment_plan."""
    if not ui.is_clinical(_role()):
        return {"error": "Only a clinician or reviewer can edit a plan."}
    body = {"clinician_id": settings.clinician_id, "actions": [a.model_dump() for a in actions]}
    try:
        result = await engine.post(f"/reports/{report_id}/override", json=body)
    except engine.EngineError as e:
        return {"error": str(e)}
    result["by"] = settings.clinician_id
    return ui.with_panel(result, ui.report_panel(report_id, "draft", False, _role()),
                         f"report/{report_id}/edited")


@mcp.tool()
async def tspi_record_outcome(report_id: str, marker: str, baseline: float, followup: float) -> dict:
    """Record a follow-up marker change for a case (feeds the propose-only learning loop). This
    never auto-changes the model; it only contributes evidence for later clinical review."""
    body = {"report_id": report_id, "marker": marker, "baseline": baseline, "followup": followup}
    try:
        return await engine.post("/outcome", json=body)
    except engine.EngineError as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- clinician plan sessions
PlanStepName = Literal["intake", "red_flags", "analysis", "modules", "dosing"]

_STEP_GUIDE = ("Show the clinician this step's summary, the key results and every check (verbatim), then "
               "ask them to confirm, adjust or go back. Do NOT confirm or acknowledge checks yourself.")
_NOT_CLINICAL = {"error": "Step-by-step plans are for clinicians and reviewers.",
                 "action_required": "Use tspi_generate_treatment_plan for a self-service draft report."}


def _session_out(view: dict, extra: dict | None = None):
    out = dict(view)
    if view.get("current_step") == "completed":
        out["next_step"] = ("The DRAFT plan was generated (report_id). Review it with tspi_get_treatment_plan; "
                            "approve, edit or reject it with tspi_approve_treatment_plan / tspi_update_treatment_plan.")
    else:
        out["assistant_guidance"] = _STEP_GUIDE
    if extra:
        out.update(extra)
    return ui.with_panel(out, ui.step_panel(view, _role()),
                         f"plan/{view.get('session_id')}/{view.get('current_step')}")


async def _start_session(payload: dict, findings: list[str], redirected: bool = False):
    try:
        view = await engine.post("/plan-sessions", json={"patient": payload})
    except engine.EngineError as e:
        return {"error": str(e)}
    extra = {}
    if redirected:
        extra["note"] = ("For clinicians the plan is built step by step: a plan session was started "
                         "instead of a one-shot plan. Step 1 of 5 is below.")
    if findings:
        extra["deidentification_warning"] = f"Redacted possible identifiers: {', '.join(findings)}"
    return _session_out(view, extra)


class PlanAdjustmentIn(BaseModel):
    """One reason-coded adjustment to the CURRENT step."""
    action: Literal["exclude_axis", "include_axis", "remove_module", "add_module", "change_dose"] = Field(
        description="analysis step: exclude_axis | include_axis · modules step: remove_module | add_module "
                    "· dosing step: change_dose")
    axis_code: str | None = Field(default=None, description="For axis actions, e.g. 'A5'.")
    module_code: str | None = Field(default=None, description="For module/dose actions, e.g. 'H-003'.")
    new_dose: str | None = Field(default=None, description="For change_dose, e.g. '2 caps x 2/day'.")
    reason_code: Literal["NEW_CLINICAL_INFORMATION", "PATIENT_PREFERENCE", "SAFETY_CONCERN", "REGISTRY_ERROR",
                         "ALGORITHM_ERROR", "CLINICAL_JUDGMENT", "DIAGNOSTIC_UNCERTAINTY", "TREATMENT_RESPONSE"]
    rationale: str | None = Field(default=None, description="Short justification in the clinician's words.")


@mcp.tool()
async def tspi_plan_start(case: PatientCase) -> dict:
    """CLINICIANS: start a step-by-step treatment plan for a de-identified case. Returns step 1 of 5
    (case data check: which labs the engine will use, symptoms recognised) with its checks. Each step
    must be confirmed by the clinician before the engine computes the next one. Run
    tspi_screen_red_flags first for a new case, as always."""
    if not ui.is_clinical(_role()):
        return _NOT_CLINICAL
    try:
        payload, findings = _forwardable(_to_engine_payload(case))
    except PIIError as e:
        return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
    return await _start_session(payload, findings)


@mcp.tool()
async def tspi_plan_get(session_id: str) -> dict:
    """CLINICIANS: show the current step of a plan session (to resume or re-display it)."""
    if not ui.is_clinical(_role()):
        return _NOT_CLINICAL
    try:
        return _session_out(await engine.get(f"/plan-sessions/{session_id}"))
    except engine.EngineError as e:
        return {"error": str(e)}


@mcp.tool()
async def tspi_plan_confirm_step(
    session_id: str,
    step: PlanStepName = Field(description="The CURRENT step being confirmed."),
    acknowledge: list[str] | None = Field(default=None, description=(
        "Codes of the checks the CLINICIAN has reviewed and accepted (the step's "
        "acknowledgements_required). Only include codes the clinician has seen.")),
    note: str | None = Field(default=None, description="Optional clinician note / justification."),
) -> dict:
    """CLINICIANS: confirm the current step after the clinician has reviewed it. The engine verifies
    that every required check is acknowledged, then computes and returns the next step. Confirming the
    last step (dosing) generates the DRAFT plan, which still needs approval. Call ONLY when the
    clinician explicitly confirmed this step in the conversation (or with the Confirm button)."""
    if not ui.is_clinical(_role()):
        return _NOT_CLINICAL
    body = {"step": step, "acknowledge": list(acknowledge or []), "note": note}
    try:
        return _session_out(await engine.post(f"/plan-sessions/{session_id}/confirm", json=body))
    except engine.EngineError as e:
        return {"error": str(e)}


@mcp.tool()
async def tspi_plan_adjust_step(
    session_id: str,
    step: PlanStepName = Field(description="The CURRENT step being adjusted."),
    adjustments: list[PlanAdjustmentIn] | None = Field(default=None, description=(
        "Reason-coded changes for the analysis, modules or dosing step.")),
    case: PatientCase | None = Field(default=None, description=(
        "intake step only: the corrected de-identified case (same case_id).")),
) -> dict:
    """CLINICIANS: apply the clinician's changes to the CURRENT step, then show the re-verified step
    for confirmation. intake: corrected case data · analysis: exclude/include an axis for module
    matching · modules: remove/add a module · dosing: change a module's dose. Confirm the changes with
    the clinician (table: action, target, new value, reason code) before calling."""
    if not ui.is_clinical(_role()):
        return _NOT_CLINICAL
    body: dict[str, Any] = {"step": step, "actions": [a.model_dump() for a in (adjustments or [])]}
    findings: list[str] = []
    if case is not None:
        try:
            body["case"], findings = _forwardable(_to_engine_payload(case))
        except PIIError as e:
            return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
    try:
        view = await engine.post(f"/plan-sessions/{session_id}/adjust", json=body)
    except engine.EngineError as e:
        return {"error": str(e)}
    return _session_out(view, {"deidentification_warning": f"Redacted possible identifiers: {', '.join(findings)}"}
                        if findings else None)


@mcp.tool()
async def tspi_plan_go_back(
    session_id: str,
    to_step: PlanStepName = Field(description="An earlier step to re-open."),
) -> dict:
    """CLINICIANS: re-open an earlier step of a plan session as it was. Every later step is cleared and
    recomputed when the clinician confirms again (adjustments made in later steps are discarded)."""
    if not ui.is_clinical(_role()):
        return _NOT_CLINICAL
    try:
        return _session_out(await engine.post(f"/plan-sessions/{session_id}/back", json={"to_step": to_step}))
    except engine.EngineError as e:
        return {"error": str(e)}


KnowledgeKind = Literal["framework", "key", "step", "domain", "axis", "module", "product", "network",
                        "safety", "dosing", "faq", "doc"]


class AskTurnIn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(description="A previous question or answer in this conversation.")


@mcp.tool()
async def tspi_ask(
    question: str = Field(description=(
        "The question, in English or Thai. De-identified only — no patient names, phone numbers, "
        "emails or ID numbers. E.g. 'Why is H-003 in this plan?', 'Which modules target A17?', "
        "'How is KS dosed?', 'What does red flag RF-CV01 mean?'")),
    report_id: str | None = Field(default=None, description=(
        "Ask about this generated treatment plan (from tspi_generate_treatment_plan).")),
    case_id: str | None = Field(default=None, description=(
        "De-identified case code; the engine uses that case's latest plan. Ignored if report_id is set.")),
    kinds: list[KnowledgeKind] | None = Field(default=None, description=(
        "Optionally restrict which knowledge sources are searched.")),
    language: Literal["en", "th"] | None = Field(default=None, description="Answer language (default: the question's)."),
    history: list[AskTurnIn] | None = Field(default=None, description=(
        "Earlier turns of this Q&A, for follow-up questions (most recent last, max 12).")),
    retrieval_only: bool = Field(default=False, description=(
        "True -> return only the matching knowledge passages, without a generated answer.")),
) -> dict:
    """Ask TSPI Brain a question about a TREATMENT PLAN and/or the TSPI KNOWLEDGE BASE (3 Keys,
    9 Restoration Steps, 12 Domains, 39 Axes, module registry & product monographs, 180 biological
    networks, red-flag & safety rules, severity dosing, official expert answers). The engine
    retrieves the relevant knowledge (code lookup + vector embeddings + keyword search) and writes a
    grounded answer that cites its sources as [S#] and the plan as [PLAN]. Read-only; audited.
    Present the answer as returned — it explains the engine's output, it does not change the plan."""
    texts = [question] + [h.content for h in (history or [])]
    found = sorted({f for t in texts for f in _scrub_text(t, case_id)[1]})
    if found:
        return {"error": f"Possible patient identifiers detected in the question ({', '.join(found)}).",
                "action_required": "Remove names, phone numbers, emails or ID numbers and ask again."}
    body: dict[str, Any] = {"question": question, "include_answer": not retrieval_only}
    if report_id:
        body["report_id"] = report_id
    elif case_id:
        body["case_id"] = case_id
    if kinds:
        body["kinds"] = list(kinds)
    if language:
        body["language"] = language
    if history:
        body["history"] = [h.model_dump() for h in history][-12:]
    try:
        result = await engine.post("/ask", json=body)
    except engine.EngineError as e:
        return {"error": str(e)}
    if _role() == "patient" and result.get("plan") and not result["plan"].get("deliverable"):
        result["notice"] = ("UNREVIEWED AI DRAFT — not reviewed or approved by a TSPI doctor and not "
                            "medical advice. Contact TSPI Digital for doctor review.")
    return result


def main() -> None:
    """Run the server. FastMCP launches its own uvicorn for HTTP transports and wires the WorkOS
    auth (middleware + /.well-known routes + the /mcp guard) from the FastMCP instance — no separate
    uvicorn call or custom ASGI app is needed. The fail-closed guard at import time already ensures
    an HTTP transport never starts without an auth provider."""
    t = settings.transport
    if t in ("streamable-http", "http", "sse"):
        import os
        port = int(os.getenv("TSPI_MCP_PORT", "8080"))
        mcp.run(transport=t, host="0.0.0.0", port=port)                 # container: bind all interfaces
    else:
        mcp.run(transport=t)                                            # stdio (local Claude Desktop)


if __name__ == "__main__":
    main()
