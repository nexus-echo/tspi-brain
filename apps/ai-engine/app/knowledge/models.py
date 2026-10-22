"""SQLAlchemy models for the knowledge base (relational source of truth).

This is the 'exact catalog' half of storage. The 'semantic librarian' half lives in the
vector store (pgvector/Chroma), embedded FROM these rows so there is one source of truth.

Run migrations in Phase 0 to load: 39 axes (from the Thai summary PDFs / axes/*.md) and the
module/product catalog, then populate `axis_module_map`.
"""
from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Key(Base):                      # 3 Keys
    __tablename__ = "keys"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(1), unique=True)   # A / B / C
    name: Mapped[str] = mapped_column(String(120))


class Step(Base):                     # 9 Steps of Wellness
    __tablename__ = "steps"
    id: Mapped[int] = mapped_column(primary_key=True)
    order: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(120))


class Domain(Base):                   # 12 Domains
    __tablename__ = "domains"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(16), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    key_id: Mapped[int | None] = mapped_column(ForeignKey("keys.id"))
    axes: Mapped[list["Axis"]] = relationship(back_populates="domain")


class Axis(Base):                     # 39 Axes
    __tablename__ = "axes"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(8), unique=True)   # A1..A39
    name: Mapped[str] = mapped_column(String(200))
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"))
    domain: Mapped[Domain] = relationship(back_populates="axes")
    sub_axes: Mapped[list["SubAxis"]] = relationship(back_populates="axis")


class SubAxis(Base):
    __tablename__ = "sub_axes"
    id: Mapped[int] = mapped_column(primary_key=True)
    axis_id: Mapped[int] = mapped_column(ForeignKey("axes.id"))
    letter: Mapped[str] = mapped_column(String(2))             # A / B / C / D
    description: Mapped[str] = mapped_column(Text)
    axis: Mapped[Axis] = relationship(back_populates="sub_axes")


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    tspi_id: Mapped[str | None] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    ingredients: Mapped[str | None] = mapped_column(Text)
    mechanisms: Mapped[str | None] = mapped_column(Text)
    active_compounds: Mapped[str | None] = mapped_column(Text)
    dosage: Mapped[str | None] = mapped_column(Text)
    safety: Mapped[str | None] = mapped_column(Text)
    therapeutic_focus: Mapped[str | None] = mapped_column(Text)


class Module(Base):                   # therapeutic unit = ONE product (1 module = 1 product)
    __tablename__ = "modules"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)          # H-Code (primary business key)
    name: Mapped[str | None] = mapped_column(String(200))               # English canonical name
    name_th: Mapped[str | None] = mapped_column(String(200))            # localized display name
    phytocore_code: Mapped[str | None] = mapped_column(String(120))     # long-term scientific ID
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))   # 1:1 product link
    dose_type: Mapped[str] = mapped_column(String(16), default="severity")      # severity | bowel
    status: Mapped[str] = mapped_column(String(16), default="active")           # active|deprecated|archived|draft
    replaced_by: Mapped[str | None] = mapped_column(String(32))         # successor H-Code (supersede)
    registry_version: Mapped[str | None] = mapped_column(String(32))
    contraindications: Mapped[str | None] = mapped_column(Text)         # JSON-encoded list of notes


class ModuleProduct(Base):            # kept for history; module<->product is now 1:1
    __tablename__ = "module_products"
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"), primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), primary_key=True)


class AxisModuleMap(Base):            # THE key mapping table (axis -> module, with role)
    __tablename__ = "axis_module_map"
    axis_id: Mapped[int] = mapped_column(ForeignKey("axes.id"), primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id"), primary_key=True)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)
    role: Mapped[str] = mapped_column(String(16), default="primary")    # primary | secondary


class RegistryMeta(Base):             # which registry/framework versions are loaded
    __tablename__ = "registry_meta"
    id: Mapped[int] = mapped_column(primary_key=True)
    registry_version: Mapped[str | None] = mapped_column(String(32))
    framework_version: Mapped[str | None] = mapped_column(String(32))
    effective_date: Mapped[str | None] = mapped_column(String(32))


# ---------------------------------------------------------------------------
# Phase 3 — workflow, audit & outcomes (de-identified; no PII stored)
# ---------------------------------------------------------------------------
from datetime import datetime, timezone  # noqa: E402

from sqlalchemy import JSON, Boolean, DateTime  # noqa: E402


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Report(Base):
    """A generated TSPI Case Report (de-identified). Persisted for validation + audit."""
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)        # uuid
    case_id: Mapped[str] = mapped_column(String(64))                     # de-identified handle
    status: Mapped[str] = mapped_column(String(16), default="draft")    # draft|validated|rejected
    nss: Mapped[int] = mapped_column(Integer, default=0)
    severity_level: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON)                          # full CaseReport JSON
    safety_alerts: Mapped[dict | None] = mapped_column(JSON)
    # Phase B — ownership / tenant scope (nullable for legacy rows). NOT PII: de-identified keys.
    owner_clinician_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    owner_case_subject: Mapped[str | None] = mapped_column(String(64), nullable=True)
    clinic_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PlanSession(Base):
    """Clinician step-by-step plan workflow (intake -> red flags -> analysis -> modules -> dosing
    -> draft report). `state` holds the de-identified case, each step's result/checks, clinician
    confirmations and adjustments; the finished draft is a normal Report (report_id)."""
    __tablename__ = "plan_sessions"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="active")      # active|completed|cancelled
    current_step: Mapped[str] = mapped_column(String(24), default="intake")
    state: Mapped[dict] = mapped_column(JSON)
    owner_clinician_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    clinic_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ReportValidation(Base):
    __tablename__ = "report_validations"
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"))
    doctor_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))                    # approve|edit|reject
    edits: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class OutcomeRow(Base):
    __tablename__ = "outcomes"
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.id"))
    marker: Mapped[str] = mapped_column(String(64))
    baseline: Mapped[float] = mapped_column(Float)
    followup: Mapped[float] = mapped_column(Float)
    delta: Mapped[float] = mapped_column(Float)
    learned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PatientIdentity(Base):
    """P4 — encrypted patient identity (PHI), keyed by the de-identified case_id.

    Only the de-id/re-id layer reads this; the brain + LLM never do. `blob` is Fernet-encrypted
    (or plaintext JSON in the dev fallback when no key is configured — never in production)."""
    __tablename__ = "patient_identities"
    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    blob: Mapped[str] = mapped_column(Text)
    encrypted: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AuditLog(Base):
    """Append-only audit trail. Records actions + consent decisions; no PII."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(String(40))     # analyze|report|validate|outcome|consent_denied
    case_id: Mapped[str | None] = mapped_column(String(64))
    report_id: Mapped[str | None] = mapped_column(String(40))
    actor: Mapped[str | None] = mapped_column(String(64))   # 'mihealth' | doctor_id | 'system'
    allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    detail: Mapped[dict | None] = mapped_column(JSON)
    # Phase B — richer, reconstructable audit
    role: Mapped[str | None] = mapped_column(String(24), nullable=True)
    source: Mapped[str | None] = mapped_column(String(24), nullable=True)   # mcp|mihealth|system
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AxisWeight(Base):
    """Phase 4 — learned per-axis weight (1.0 = neutral). Adjusted by the learning loop from
    outcomes; consumed by the severity engine (NSS/SPS)."""
    __tablename__ = "axis_weights"
    axis_code: Mapped[str] = mapped_column(String(8), primary_key=True)   # A1..A39
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# ---------------------------------------------------------------------------
# Phase 12 — Adaptive Biological Intelligence Loop (3 levels; population = propose-only)
# Per TSPI 3 Q11: the AI must NOT modify the global clinical model from individual cases.
# ---------------------------------------------------------------------------
class PatientAxisWeight(Base):
    """Level 1 — patient-specific axis adaptation (bounded, never global)."""
    __tablename__ = "patient_axis_weights"
    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    axis_code: Mapped[str] = mapped_column(String(8), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    cumulative_adjustment: Mapped[float] = mapped_column(Float, default=0.0)  # bounded +-0.20
    consecutive_improvements: Mapped[int] = mapped_column(Integer, default=0)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class NetworkObservation(Base):
    """Level 2 — network behaviour learning: which network moved first, what followed, how fast."""
    __tablename__ = "network_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64))
    changed_first: Mapped[str] = mapped_column(String(64))          # network/axis code
    downstream: Mapped[str | None] = mapped_column(String(64))
    direction: Mapped[str] = mapped_column(String(16))              # improved | worsened
    magnitude: Mapped[float | None] = mapped_column(Float)
    time_to_response_days: Mapped[int | None] = mapped_column(Integer)
    repeated: Mapped[bool] = mapped_column(Boolean, default=False)
    confounders: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PatientModel(Base):
    """Level 3 — versioned living patient biological model. Old versions are NEVER deleted."""
    __tablename__ = "patient_models"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    change_kind: Mapped[str] = mapped_column(String(16), default="CONFIRM")  # CONFIRM/REFINE/REDIRECT/OVERRIDE/CONTRADICT
    axis_state: Mapped[dict] = mapped_column(JSON, default=dict)
    network_state: Mapped[dict] = mapped_column(JSON, default=dict)
    trigger: Mapped[str | None] = mapped_column(String(200))
    superseded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelUpdateProposal(Base):
    """Level 4 — population learning. AI may only PROPOSE; a clinician must approve."""
    __tablename__ = "model_update_proposals"
    id: Mapped[int] = mapped_column(primary_key=True)
    axis_code: Mapped[str] = mapped_column(String(8))
    current_weight: Mapped[float] = mapped_column(Float, default=1.0)
    proposed_weight: Mapped[float] = mapped_column(Float, default=1.0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    improved: Mapped[int] = mapped_column(Integer, default=0)
    worsened: Mapped[int] = mapped_column(Integer, default=0)
    rationale: Mapped[str | None] = mapped_column(Text)
    blocking_reasons: Mapped[str | None] = mapped_column(Text)      # why it cannot auto-apply
    status: Mapped[str] = mapped_column(String(24), default="PROPOSE_MODEL_UPDATE")
    reviewed_by: Mapped[str | None] = mapped_column(String(64))
    review_date: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class LearningRecord(Base):
    """The minimum learning record specified by the experts (Q11)."""
    __tablename__ = "learning_records"
    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), index=True)
    patient_model_version: Mapped[int | None] = mapped_column(Integer)
    baseline_axis_state: Mapped[dict] = mapped_column(JSON, default=dict)
    baseline_network_state: Mapped[dict] = mapped_column(JSON, default=dict)
    followup_axis_state: Mapped[dict] = mapped_column(JSON, default=dict)
    followup_network_state: Mapped[dict] = mapped_column(JSON, default=dict)
    intervention_modules: Mapped[dict] = mapped_column(JSON, default=dict)
    dose: Mapped[str | None] = mapped_column(String(120))
    duration: Mapped[str | None] = mapped_column(String(64))
    adherence: Mapped[str | None] = mapped_column(String(32))
    lifestyle_interventions: Mapped[dict] = mapped_column(JSON, default=dict)
    concurrent_medications: Mapped[dict] = mapped_column(JSON, default=dict)
    symptom_response: Mapped[str | None] = mapped_column(Text)
    laboratory_response: Mapped[str | None] = mapped_column(Text)
    adverse_response: Mapped[str | None] = mapped_column(Text)
    time_to_response: Mapped[str | None] = mapped_column(String(64))
    confounders: Mapped[dict] = mapped_column(JSON, default=dict)
    physician_interpretation: Mapped[str | None] = mapped_column(Text)
    learning_approval_status: Mapped[str] = mapped_column(String(24), default="UNREVIEWED")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
