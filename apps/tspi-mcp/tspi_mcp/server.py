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
from .deident import PIIError, deidentify
from .error_logging import ToolErrorLoggingMiddleware, configure_logging

configure_logging()

INSTRUCTIONS = """TSPI AI Brain — deterministic clinical decision-support.

RULES FOR THE ASSISTANT:
- Collect DE-IDENTIFIED data only: a case code, an age band, sex, symptoms, and lab values.
  NEVER ask for or submit a patient's name, date of birth, MRN, phone, email, or address.
- The TSPI engine is the sole source of clinical conclusions. Present its structured output as-is.
  Do NOT invent or estimate axis scores, networks, modules, doses, evidence grades, or outcomes.
- Always run tspi_screen_red_flags first for a new case.
- A generated treatment plan is an AI DRAFT pending physician review; it is NOT for patient use
  until a clinician approves it with tspi_approve_treatment_plan.
- If tspi_whoami returns role 'patient' (self-service user), the patient may see their own draft,
  but it must be labelled UNREVIEWED AI DRAFT (not reviewed or approved by a TSPI doctor, not
  medical advice) and they must be told to contact TSPI Digital for doctor review. Never call
  approve, update or record_outcome for a patient.
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
    return ui.with_panel(out, ui.analysis_panel(case.case_id), f"analysis/{case.case_id}")


@mcp.tool()
async def tspi_generate_treatment_plan(case: PatientCase) -> dict:
    """Generate the full TSPI case report + candidate module plan for a de-identified case.
    The result is an AI DRAFT (deliverable=false) and must be reviewed and approved by a clinician
    before any patient use. Do not present it as a final prescription."""
    try:
        payload, findings = _forwardable(_to_engine_payload(case))
    except PIIError as e:
        return {"error": str(e), "action_required": "Remove patient identifiers and resend."}
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
