"""Phase B — authentication, roles, and access checks.

Design (see docs/01-plans/TSPI_Phase_B_Auth_RBAC_Audit_Plan.md):
  * A trusted caller (MCP server / MiHealth) presents a SERVICE bearer token and forwards the
    end-user identity via X-TSPI-* headers. The engine trusts those headers ONLY when the service
    token is valid — the browser/LLM can never set them directly.
  * When settings.auth_enabled is False (default), everything is permissive so the local pilot,
    MiHealth, and the existing tests keep working unchanged.
  * The engine is the authority: role checks + tenant/ownership checks are enforced here,
    server-side, regardless of what any UI shows.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

from app.config import settings

# Roles (mirror the plan). 'service' is the machine caller acting on behalf of a user.
ROLES = {"patient", "clinic_staff", "clinician", "reviewer", "auditor", "service"}


@dataclass
class Principal:
    id: str                      # user identity: clinician id, patient subject key, or staff id
    role: str                    # one of ROLES
    clinic_id: str | None = None
    source: str = "system"       # mcp | mihealth | system
    request_id: str | None = None
    name: str | None = None      # human display name (for report attribution; never PII-gating)
    email: str | None = None     # human email (for report attribution / audit)


def _valid_service_tokens() -> set[str]:
    return {t.strip() for t in (settings.service_tokens or "").split(",") if t.strip()}


def get_principal(
    authorization: str | None = Header(default=None),
    x_tspi_user_id: str | None = Header(default=None),
    x_tspi_role: str | None = Header(default=None),
    x_tspi_clinic_id: str | None = Header(default=None),
    x_tspi_source: str | None = Header(default=None),
    x_request_id: str | None = Header(default=None),
    x_tspi_name: str | None = Header(default=None),
    x_tspi_email: str | None = Header(default=None),
) -> Principal:
    """Resolve the caller into a Principal. Fail-closed when auth is enabled."""
    if not settings.auth_enabled:
        # Permissive dev/pilot principal — reviewer passes all role checks; reads across tenants.
        return Principal(id="dev", role="reviewer", source="dev")

    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token not in _valid_service_tokens():
        raise HTTPException(status_code=401, detail="Invalid or missing service token.")

    role = (x_tspi_role or "").strip()
    if role not in ROLES:
        raise HTTPException(status_code=401, detail="Missing or invalid X-TSPI-Role.")
    if not x_tspi_user_id:
        raise HTTPException(status_code=401, detail="Missing X-TSPI-User-Id.")

    return Principal(id=x_tspi_user_id, role=role, clinic_id=x_tspi_clinic_id,
                     source=(x_tspi_source or "mihealth"), request_id=x_request_id,
                     name=(x_tspi_name or None), email=(x_tspi_email or None))


def require_role(*allowed: str):
    """Dependency factory: allow only the given roles (no-op when auth is disabled)."""
    allowed_set = set(allowed)

    def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not settings.auth_enabled:
            return principal
        if principal.role not in allowed_set:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{principal.role}' may not perform this action.")
        return principal

    return _dep


def check_report_access(principal: Principal, report: dict, *, write: bool) -> None:
    """Tenant/ownership gate for a specific report. Raises 403 when not permitted.

    Rules (§4.3):
      patient      -> read own case subject when approved (deliverable), OR an unapproved draft
                      the patient generated themselves (self-service, marked AI draft); no writes
      clinic_staff -> read within own clinic; no clinical writes
      clinician    -> read/write own cases
      reviewer     -> read/write across
      auditor      -> read across; no writes
    """
    if not settings.auth_enabled:
        return
    role = principal.role
    deliverable = bool(report.get("deliverable"))

    if role == "reviewer":
        return
    if role == "auditor":
        if write:
            raise HTTPException(status_code=403, detail="Auditor is read-only.")
        return
    if role == "clinician":
        if report.get("owner_clinician_id") in (None, principal.id):
            return
        raise HTTPException(status_code=403, detail="Not your case.")
    if role == "clinic_staff":
        if write:
            raise HTTPException(status_code=403, detail="Clinic staff cannot perform clinical writes.")
        if report.get("clinic_id") and report.get("clinic_id") == principal.clinic_id:
            return
        raise HTTPException(status_code=403, detail="Report is outside your clinic.")
    if role == "patient":
        if write:
            raise HTTPException(status_code=403, detail="Patients cannot modify plans.")
        if report.get("owner_case_subject") == principal.id and deliverable:
            return
        created_by = (report.get("payload") or {}).get("created_by") or {}
        if (report.get("owner_case_subject") == principal.id
                and created_by.get("role") == "patient" and created_by.get("id") == principal.id):
            return                                # own self-service draft (unreviewed)
        raise HTTPException(status_code=403,
                            detail="Patients may only view their own APPROVED plan.")
    raise HTTPException(status_code=403, detail="Not permitted.")
