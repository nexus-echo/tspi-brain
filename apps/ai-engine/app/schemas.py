"""Pydantic contracts for the whole pipeline.

input -> signals -> axis scores (+NSS/level/SPS) -> dosed modules -> de-identified report
(+ safety alerts, report_id, deliverable). Phase 3 adds validation/consent/audit fields.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from app.evidence import (
    AxisStatus,
    EvidenceGrade,
    EvidenceItem,
    EvidenceStatus,
)


class Severity(str, Enum):
    optimal = "optimal"
    subclinical = "subclinical"
    functional_impairment = "functional_impairment"
    pathological = "pathological"


class KeyCode(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class LabResult(BaseModel):
    analyte: str
    value: float | str
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str | None = None


class Symptom(BaseModel):
    """Phase 8 — a STRUCTURED symptom. Symptoms are first-class biological evidence, not
    narrative text (expert Q4). Field set per the experts' required structure."""
    symptom_code: str | None = None
    canonical_name_en: str
    display_name_th: str | None = None
    onset: str | None = None
    duration: str | None = None
    frequency: str | None = None
    severity: int | None = Field(None, ge=0, le=10)
    progression: str | None = None
    location: str | None = None
    quality: str | None = None
    triggering_factors: list[str] = []
    relieving_factors: list[str] = []
    meal_relationship: str | None = None
    bowel_relationship: str | None = None
    sleep_relationship: str | None = None
    stress_relationship: str | None = None
    activity_relationship: str | None = None
    circadian_pattern: str | None = None
    functional_impact: str | None = None
    associated_symptoms: list[str] = []
    negative_findings: list[str] = []          # e.g. "gastroscopy normal" -- narrows, never erases
    red_flags: list[str] = []


class PatientInput(BaseModel):
    """De-identified by contract: case_id + minimal demographics only."""
    case_id: str = Field(..., examples=["TSPI-EX-001"])
    age_band: str | None = Field(None, examples=["mid-40s"])
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    symptoms: str = ""                          # free text (back-compat; parsed deterministically)
    structured_symptoms: list[Symptom] = []     # Phase 8: preferred, richer clinical phenotype
    labs: list[LabResult] = []
    imaging: list[str] = []
    medications: list[str] = []
    conditions: list[str] = []          # for safety/contraindication checks
    lifestyle: dict[str, str] = {}
    omics: dict[str, str] = {}
    consent: dict[str, bool] = {}
    # P4 — optional PII; the engine strips + stores it encrypted at intake so the pipeline/LLM
    # never see it. De-identified clients (MCP, existing MiHealth) simply omit this.
    patient_identity: "PatientIdentity | None" = None       # ai_analysis, doctor_sharing, tspi_connection, ...


class PatientIdentity(BaseModel):
    """P4 — patient PII. Accepted ONLY at the engine's intake boundary; stripped + stored encrypted
    before any pipeline/LLM step. Never persisted in the report; re-attached at render only."""
    full_name: str | None = None
    dob: str | None = None
    mrn: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None


class BiologicalSignal(BaseModel):
    concept: str
    direction: Literal["up", "down", "abnormal", "normal"] = "abnormal"
    magnitude: float | None = None
    source: str
    evidence_status: EvidenceStatus = EvidenceStatus.CLINICAL_INFERENCE
    value_source_type: str = "MEASURED"   # MEASURED | DERIVED (Marker Registry, 25 Jul Q5a)


class AxisScore(BaseModel):
    """An assessed axis. Phase 6: carries the four required dimensions.

    `score` is NULLABLE on purpose -- absence of data is never 0 and never "normal".
    """
    axis_code: str
    axis_name: str
    domain_code: str | None = None
    key: KeyCode | None = None
    severity: Severity
    is_driver: bool = False
    inferred: bool = False
    evidence: list[str] = []

    # --- Phase 6: the four dimensions (expert Q7) ---
    score: float | None = None                      # 0-100; None when NOT_ASSESSED
    status: AxisStatus = AxisStatus.AXIS_CANDIDATE
    evidence_status: EvidenceStatus = EvidenceStatus.CLINICAL_INFERENCE
    evidence_grade: EvidenceGrade | None = None
    confidence: float = 0.0                         # 0..1
    confidence_label: str = "Very Low"
    data_completeness: float = 0.0                  # 0..1 weighted
    biological_uncertainty: float = 0.0             # 0..1
    uncertainty_label: str = "Low"
    scoring_effect: bool = True                     # False when evidence is HYPOTHESIS-only
    reason: str | None = None
    evidence_items: list[EvidenceItem] = []
    missing_evidence: list[str] = []


class NotAssessedAxis(BaseModel):
    """An axis we could NOT assess. Explicit by design -- no data must never read as normal."""
    axis_code: str
    axis_name: str
    domain_code: str | None = None
    score: None = None
    status: AxisStatus = AxisStatus.NOT_ASSESSED
    confidence: str = "INSUFFICIENT_DATA"
    reason: str = "required markers unavailable"
    required_evidence: list[str] = []


class ModulePick(BaseModel):
    module_code: str
    module_name: str | None = None
    target_axes: list[str] = []
    mechanism: str | None = None
    dose: str | None = None
    clinical_role: str | None = None
    safety: str | None = None
    phase: Literal["step_1", "step_2", "step_3"] = "step_1"
    resolved: bool = True
    phytocore_code: str | None = None
    dose_type: str = "severity"                 # severity | bowel
    status: str = "active"
    axis_roles: dict[str, str] = {}             # axis_code -> "primary" | "secondary"
    contraindications: list[str] = []
    # --- Phase 10: Module Match v1.0 ---
    module_match_score: float | None = None     # None when EXCLUDED (safety gate) -> not_calculated
    match_breakdown: dict = {}
    module_status: str = "ELIGIBLE"             # ELIGIBLE | EXCLUDED
    safety_outcome: str = "PASS"                # PASS | PASS_WITH_MONITORING | HOLD | CONTRAINDICATED | INSUFFICIENT_SAFETY_DATA
    tier: str | None = None                     # core | supporting
    reason_not_selected: str | None = None
    duplicate_of: str | None = None
    match_notes: list[str] = []


class AnalysisResult(BaseModel):
    case_id: str
    signals: list[BiologicalSignal]
    axis_scores: list[AxisScore]
    root_cause_chain: list[str]
    nss: int = 0
    severity_level: int = 0
    severity_name: str = ""
    system_priority: list[dict] = []
    nine_step_position: int | None = None
    prakati_gap: str | None = None
    notes: list[str] = []
    not_assessed: list[NotAssessedAxis] = []        # Phase 6: explicit, never silently absent
    axis_master_version: str | None = None
    nss_algorithm_version: str | None = None
    red_flag_screen: dict = {}                      # Phase 7: runs BEFORE network/axis reasoning
    phenotype: dict = {}                            # Phase 8: clinical phenotype + differential


class CaseReport(BaseModel):
    case_id: str
    report_id: str | None = None              # persisted id (Phase 3)
    analysis: AnalysisResult
    modules: list[ModulePick]
    monitoring: list[dict] = []
    safety_alerts: list[dict] = []            # contraindication flags (Phase 3)
    report_markdown: str = ""
    disclaimer: str = "For clinician review and approval — not a substitute for medical judgment."
    status: Literal["draft", "validated", "rejected"] = "draft"
    deliverable: bool = False                 # only true once a doctor validates
    registry_version: str | None = None       # which module-registry version produced this
    framework_version: str | None = None      # which 39-axis framework version
    # --- Phase 7: red-flag gate ---
    red_flag_screen: dict = {}
    release_block: bool = False               # true -> must not reach the patient
    escalation: str | None = None             # EMERGENCY_REFERRAL / URGENT_MEDICAL_ASSESSMENT / ...
    # --- Phase 10: matching is not prescribing ---
    considered_modules: list[ModulePick] = [] # matched but NOT selected, each with a reason
    module_selection: dict = {}               # counts, limits, approval flags, match_version
    nss_detail: dict = {}                     # explainable NSS v0.1 breakdown
    # --- Production pilot ---
    pilot_mode: bool = False                  # produced with provisional/candidate data
    provisional: bool = False                 # not clinically validated
    pilot_notice: str | None = None           # watermark shown on the report
    report_language: str = "en"               # "en" | "th"
    networks: list[dict] = []                 # candidate mechanistic networks for the assessed axes


class ValidationRequest(BaseModel):
    report_id: str
    doctor_id: str
    decision: Literal["approve", "edit", "reject"]
    edits: dict | None = None


class OverrideAction(BaseModel):
    """One structured physician edit to a plan (5 Aug 'structured clinician overrides')."""
    action: Literal[
        "REMOVE_MODULE", "REJECT_MODULE", "CHANGE_DOSE", "OVERRIDE_SAFETY",
        "ADD_SECONDARY_AXIS", "REMOVE_SECONDARY_AXIS", "ADD_NOTE",
    ]
    module_code: str | None = None
    axis_code: str | None = None
    new_dose: str | None = None
    reason_code: Literal[
        "NEW_CLINICAL_INFORMATION", "PATIENT_PREFERENCE", "SAFETY_CONCERN", "REGISTRY_ERROR",
        "ALGORITHM_ERROR", "CLINICAL_JUDGMENT", "DIAGNOSTIC_UNCERTAINTY", "TREATMENT_RESPONSE",
    ]
    rationale: str | None = None


class OverrideRequest(BaseModel):
    clinician_id: str
    actions: list[OverrideAction]


class OutcomeRecord(BaseModel):
    report_id: str
    marker: str
    baseline: float
    followup: float


# --- Lab/imaging extraction (POST /extract) ---
class ExtractedLab(BaseModel):
    analyte: str
    value: float | str
    unit: str | None = None
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str | None = None                 # H/L/normal/abnormal if the report states one


class ExtractionResult(BaseModel):
    """Candidate structured data read from ONE uploaded file. Always confirmed=False:
    a human must review before it feeds diagnosis (governance gate preserved)."""
    source: str                              # image | pdf-text | pdf-scan | text | unknown
    engine: str                              # e.g. ollama-vision:qwen2.5vl | ollama-text:qwen2.5:14b | heuristic
    labs: list[ExtractedLab] = []
    imaging_modality: str | None = None      # e.g. "Ultrasound (pelvic)"
    imaging_impression: str | None = None    # radiologist impression / conclusion
    imaging_findings: list[str] = []         # bulletable findings (sizes, locations, ...)
    narrative: str | None = None             # any free-text summary the model produced
    confirmed: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- Knowledge Q&A (/ask)
class AskTurn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    """Ask TSPI Brain a question about a treatment plan and/or the TSPI knowledge base.
    De-identified only: never put names, phone numbers, emails or ID numbers in the question."""
    question: str = Field(min_length=2, max_length=2000)
    report_id: str | None = Field(default=None, description="Answer about this stored plan/report.")
    case_id: str | None = Field(default=None, description="De-identified case code; uses its latest report.")
    kinds: list[str] | None = Field(default=None, description=(
        "Restrict knowledge sources: framework, key, step, domain, axis, module, product, network, "
        "safety, dosing, faq, doc."))
    top_k: int | None = Field(default=None, ge=1, le=20)
    language: Literal["en", "th"] | None = None
    audience: Literal["clinician", "patient"] | None = None
    history: list[AskTurn] = Field(default_factory=list, max_length=12,
                                   description="Prior turns for follow-up questions (most recent last).")
    include_answer: bool = Field(default=True, description="False -> retrieval only (no LLM).")


class AskSource(BaseModel):
    id: str                       # S1..Sn — what the answer cites
    doc_id: str
    kind: str
    ref_code: str
    title: str
    snippet: str
    score: float | None = None
    retrieval: list[str] = []     # exact | vector | lexical
    cited: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_mode: Literal["llm_grounded", "retrieval_only"]
    sources: list[AskSource] = []
    plan: dict | None = None
    notice: str | None = None
    retrieval: dict = {}
    disclaimer: str
