"""Wires the official TSPI-AI decision flow (Phase 0-3):
collect -> 12 Systems -> 39 Axes -> NSS -> SPS -> Level -> modules -> dosing -> KS ->
safety flags -> compose -> persist (de-identified) -> doctor validation gate.
"""
from __future__ import annotations

from app import phenotype, red_flags, safety, store
from app.config import settings
from app.knowledge import dosing
from app.knowledge.repository import KnowledgeRepo
from app.llm.provider import LLMProvider
from app.pipeline import (
    axis_mapper,
    graph_engine,
    module_matcher,
    normalizer,
    report_composer,
    sequencer,
    severity,
    step_locator,
)
from app.schemas import AnalysisResult, CaseReport, PatientInput

import logging

_log = logging.getLogger(__name__)

_repo = KnowledgeRepo()
_llm = LLMProvider()


def nss_breakdown_for(analysis) -> dict:
    """Explainable NSS v0.1 breakdown for the report (auditable, not a black box)."""
    return severity.nss_detail(analysis.axis_scores, store.get_axis_weights())


def _axis_master_version() -> str | None:
    """Stamp which axis ontology produced this analysis (expert: version-stamp everything)."""
    import json
    from pathlib import Path
    try:
        p = Path(__file__).resolve().parent.parent.parent / "data" / "tspi_axes_39.json"
        return json.loads(p.read_text(encoding="utf-8")).get("framework_version")
    except Exception:  # noqa: BLE001
        _log.warning("Could not read axis master version", exc_info=True)
        return None


async def analyze(patient: PatientInput) -> AnalysisResult:
    # Phase 7 — MANDATORY red-flag screening BEFORE any network/axis/module reasoning.
    screen = red_flags.screen(patient)
    # Phase 8 — build the clinical phenotype BEFORE axis reasoning (symptoms are core evidence).
    pheno = phenotype.evaluate(patient)
    signals = normalizer.normalize(patient)
    axis_scores, not_assessed = await axis_mapper.map_axes(signals, _repo, pheno)
    chain, axis_scores = graph_engine.root_cause(axis_scores)
    weights = store.get_axis_weights()                 # Phase 4: learned weights (neutral if none)
    nss = severity.compute_nss(axis_scores, weights)
    nss_breakdown = severity.nss_detail(axis_scores, weights)
    level = dosing.level_for_nss(nss)
    sps = severity.compute_sps(axis_scores, weights)
    position, gap = step_locator.locate(axis_scores)
    return AnalysisResult(
        case_id=patient.case_id, signals=signals, axis_scores=axis_scores,
        root_cause_chain=chain, nss=nss, severity_level=level["level"],
        severity_name=level["name"], system_priority=sps,
        nine_step_position=position, prakati_gap=gap,
        not_assessed=not_assessed,
        axis_master_version=_axis_master_version(),
        nss_algorithm_version=severity.algorithm_version(),
        red_flag_screen=screen,
        phenotype=pheno,
        notes=["NSS uses a documented heuristic pending the official formula.",
               dosing.safety_note()],
    )


async def build_report(patient: PatientInput, report_language: str | None = None) -> CaseReport:
    analysis = await analyze(patient)
    screen = analysis.red_flag_screen or {}

    # Phase 7 gate: an emergency red flag blocks the module plan entirely.
    considered: list = []
    selection: dict = {}
    if screen.get("module_plan_allowed", True):
        matched = module_matcher.match(analysis.axis_scores, _repo, patient.medications,
                                       conditions=patient.conditions,
                                       step_position=analysis.nine_step_position)
        modules = sequencer.sequence(matched["selected"])
        considered = matched["considered"]
        selection = matched["selection"]
    else:
        modules = []
        selection = {"note": "module plan blocked by red-flag screening",
                     "counts": {"selected": 0}}

    # Phase 10 dosing safeguard (expert Q3): NSS guides the DEFAULT dose only. It must never
    # override module-specific rules (e.g. the bowel group), contraindications, organ impairment,
    # age adjustments or physician judgment.
    level = dosing.level_for_nss(analysis.nss)
    level_dose = dosing.dose_string(level)
    for m in modules:
        if m.dose_type == "bowel" or dosing.is_bowel(m.module_code) or dosing.is_bowel(m.module_name or ""):
            m.dose = dosing.bowel_dose(m.module_name or m.module_code)
        else:
            m.dose = level_dose

    # Phase 3 — safety flags (never drop a module; flag for clinician)
    # Safety flags cover the SELECTED plan AND the considered alternatives: a clinician may
    # override-select a considered module, so a contraindication must never be hidden by ranking.
    alerts = safety.check_modules(modules + considered, patient.medications, patient.conditions)

    report = await report_composer.compose(analysis, modules, _llm)
    report.monitoring = [{"reassess_every_days": dosing.reassessment_days()}]
    report.safety_alerts = alerts
    _meta = _repo.registry_meta()
    report.registry_version = _meta.get("registry_version")
    report.framework_version = _meta.get("framework_version")

    # Phase 7 — surface the gate on the report
    report.considered_modules = considered
    report.module_selection = selection
    report.nss_detail = nss_breakdown_for(analysis)
    report.red_flag_screen = screen
    report.release_block = bool(screen.get("release_block", False))
    report.escalation = screen.get("required_action")
    if screen.get("red_flags"):
        for f in screen["red_flags"]:
            report.safety_alerts.append({
                "module": None, "severity": "red_flag",
                "action_class": f["action_class"],
                "reason": f"RED FLAG [{f['action_class']}] {f['name']} ({f['matched']})"})
        store.audit("red_flag", case_id=analysis.case_id,
                    detail={"action_class": screen.get("action_class"),
                            "count": len(screen["red_flags"]),
                            "release_block": report.release_block})

    # Candidate network layer — mechanistic context for the assessed axes (explains, never ranks)
    from app import networks as _net
    seen_ax: list[str] = []
    for a in analysis.axis_scores:
        if getattr(a, "scoring_effect", True) and a.axis_code not in seen_ax:
            seen_ax.append(a.axis_code)
    report.networks = [n for ax in seen_ax[:6] for n in _net.networks_for_axis(ax, limit=2)]

    # Production pilot — stamp provisional watermark + language; provisional never auto-reaches a patient
    from app import pilot
    report.report_language = pilot.report_language(report_language)
    if pilot.is_pilot():
        report.pilot_mode = True
        report.provisional = True
        report.pilot_notice = pilot.PILOT_NOTICE
        report.release_block = True          # provisional plans must not reach a patient
        report.disclaimer = f"{pilot.PILOT_NOTICE} {report.disclaimer}"
        for m in report.modules:
            if pilot.PILOT_NOTICE[:5] not in " ".join(m.match_notes):
                m.match_notes = [*m.match_notes, "PROVISIONAL — pilot candidate mapping"]

    # Phase 3 — persist (de-identified) + doctor-validation gate
    status = "draft" if settings.require_doctor_validation else "validated"
    report.status = status
    report.deliverable = (status == "validated") and not report.release_block
    report.report_id = store.save_report(
        analysis.case_id, status, analysis.nss, analysis.severity_level,
        report.model_dump(mode="json"), alerts)
    store.audit("report", case_id=analysis.case_id, report_id=report.report_id,
                detail={"nss": analysis.nss, "level": analysis.severity_level,
                        "safety_alerts": len(alerts)})
    return report
