"""Smoke tests: the pipeline boots and produces a grounded, de-identified report."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_PATIENT = {
    "case_id": "TSPI-EX-001",
    "age_band": "mid-40s",
    "sex": "female",
    "symptoms": "fibroid, hypertension, body inflammation, stress",
    "labs": [
        {"analyte": "CRP", "value": 21.57, "unit": "mg/L", "ref_high": 5.0},
        {"analyte": "HbA1c", "value": 6.5, "unit": "%", "ref_high": 5.6},
        {"analyte": "Hemoglobin", "value": 10.8, "unit": "g/dL", "ref_low": 12.0},
        {"analyte": "TSH", "value": 0.416, "unit": "uIU/mL", "ref_low": 0.54},
    ],
    "medications": ["antihypertensive"],
    "consent": {"ai_analysis": True},
}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_analyze_maps_axes():
    r = client.post("/analyze", json=_PATIENT).json()
    codes = {a["axis_code"] for a in r["axis_scores"]}
    assert "A1" in codes            # CRP -> systemic inflammation
    assert r["root_cause_chain"]    # a chain was produced


def test_report_is_grounded_and_deidentified():
    r = client.post("/report", json=_PATIENT).json()
    assert r["case_id"] == "TSPI-EX-001"
    assert "clinician review" in r["disclaimer"].lower()
    # modules are resolved against the real product registry; unresolved ones are FLAGGED
    assert all(isinstance(m["resolved"], bool) for m in r["modules"])
    ks = [m for m in r["modules"] if (m["module_name"] or "").strip().upper() == "KS"]
    assert ks and ks[0]["resolved"]  # core KS module resolves to the catalog


def test_consent_blocks_analysis():
    p = {**_PATIENT, "consent": {"ai_analysis": False}}
    assert client.post("/analyze", json=p).status_code == 403


def test_severity_and_dosing_present():
    r = client.post("/report", json=_PATIENT).json()
    a = r["analysis"]
    assert 0 <= a["nss"] <= 100
    assert a["severity_level"] in (0, 1, 2, 3)
    assert a["severity_name"]
    assert a["system_priority"]                      # SPS ranked systems
    assert all(m["dose"] for m in r["modules"])      # every module got a dose
    ks = [m for m in r["modules"] if (m["module_name"] or "").strip().upper() == "KS"]
    assert ks and "bowel" in ks[0]["dose"].lower()   # KS uses its special protocol
    assert ks[0]["dose_type"] == "bowel"


def test_async_embed_matches_sync():
    import asyncio
    from app.knowledge.embeddings import EmbeddingProvider
    e = EmbeddingProvider()                       # hash backend in tests
    assert e.embed("chronic inflammation") == asyncio.run(e.aembed("chronic inflammation"))


def test_retrieval_cache_helpers():
    from app.knowledge import retrieval as R
    R.clear_cache()
    R._cache_put(("k",), [1.0, 2.0])
    assert R._cache_get(("k",)) == [1.0, 2.0]
    R.clear_cache()
    assert R._cache_get(("k",)) is None


def test_retrieve_noop_on_sqlite():
    import asyncio
    from app.knowledge import retrieval as R
    assert R.retrieve("x") == []                  # sync no-op
    assert asyncio.run(R.aretrieve("x")) == []    # async no-op


# ---- Phase 3: validation, persistence, consent/audit, safety ----
def test_report_persisted_not_deliverable():
    r = client.post("/report", json=_PATIENT).json()
    assert r["report_id"]                         # persisted
    assert r["status"] == "draft"
    assert r["deliverable"] is False              # needs doctor validation first


def test_doctor_validation_makes_deliverable():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    v = client.post("/validate", json={"report_id": rid, "doctor_id": "dr.smith",
                                       "decision": "approve"}).json()
    assert v["status"] == "validated" and v["deliverable"] is True
    fetched = client.get(f"/reports/{rid}").json()
    assert fetched["deliverable"] is True


def test_validate_unknown_report_404():
    assert client.post("/validate", json={"report_id": "nope", "doctor_id": "d",
                                          "decision": "approve"}).status_code == 404


def test_outcome_recorded():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    o = client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                      "baseline": 21.5, "followup": 6.0}).json()
    assert round(o["delta"], 1) == -15.5


def test_safety_flags_for_condition():
    p = {**_PATIENT, "conditions": ["pregnancy"]}   # KERRA rule -> flag (not dropped)
    r = client.post("/report", json=p).json()
    # Phase 10: the selection ceiling means KERRA may be a CONSIDERED alternative rather than
    # selected -- a contraindication must still be flagged, never hidden by ranking.
    presented = r["modules"] + r["considered_modules"]
    kerra_codes = {m["module_code"] for m in presented if "KERRA" in (m["module_name"] or "").upper()}
    assert kerra_codes, "KERRA module expected among the presented modules for this case"
    flagged = [a for a in r["safety_alerts"] if a.get("module") in kerra_codes]
    assert flagged and all(isinstance(m["resolved"], bool) for m in r["modules"])


def test_safety_med_keyword_alert():
    p = {**_PATIENT, "medications": ["warfarin"]}
    r = client.post("/report", json=p).json()
    assert any("anticoagulant" in a["reason"].lower() or "bleeding" in a["reason"].lower()
               for a in r["safety_alerts"])


# ---- Phase 4: learning loop + temporal ----
def test_weights_default_neutral():
    # no learning yet -> weights empty -> NSS identical to Phase 1-3 behaviour
    assert client.get("/learning/weights").json()["axis_weights"] == {}
    r = client.post("/report", json=_PATIENT).json()
    assert r["analysis"]["nss"] >= 60          # high-CRP case stays high-severity


def test_recalibrate_is_propose_only_and_never_changes_global_model():
    """Phase 12 compliance: population learning must NEVER auto-modify the global model.

    (This replaces the old Phase-4 test, which asserted the now-forbidden auto-recalibration.)
    """
    client.post("/learning/recalibrate")                       # drain any prior outcomes
    w0 = client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0)
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 10.0, "followup": 25.0})   # CRP up = worse
    out = client.post("/learning/recalibrate").json()
    assert out["status"] == "PROPOSE_MODEL_UPDATE"
    assert out["global_model_changed"] is False
    assert out["proposals_created"] >= 1
    # the global weight must be untouched until a clinician approves
    w1 = client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0)
    assert w1 == w0, "global model must not change without clinical approval"


def test_proposal_requires_clinical_approval_before_it_applies():
    client.post("/learning/recalibrate")
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 10.0, "followup": 30.0})
    props = client.post("/learning/recalibrate").json()["proposals"]
    a1 = [p for p in props if p["axis_code"] == "A1"]
    assert a1, "expected a proposal for A1"
    pid = a1[0]["id"]
    assert a1[0]["blocking_reasons"], "a proposal must state why it cannot auto-apply"
    before = client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0)
    # approve -> only now may the global model change
    d = client.post(f"/learning/proposals/{pid}/decide",
                    params={"reviewer": "dr.smith", "approve": True}).json()
    assert d["status"] == "APPROVED" and d["reviewed_by"] == "dr.smith"
    after = client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0)
    assert after != before, "approved proposal should apply to the global model"


def test_patient_specific_learning_is_bounded_and_not_global():
    """Level 1: bounded +-5%/cycle, +-20% cumulative, and scoped to ONE patient."""
    from app.learning import MAX_CUMULATIVE, MAX_STEP
    cid = "P12-BOUNDS"
    p = {**_PATIENT, "case_id": cid}
    rid = client.post("/report", json=p).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 10.0, "followup": 25.0})   # worsening
    g0 = client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0)
    out = client.post(f"/learning/adapt/{cid}").json()
    assert out["level"] == "PATIENT_SPECIFIC"
    a1 = out["updated"].get("A1")
    assert a1 and abs(a1["step"]) <= MAX_STEP + 1e-9          # never more than 5% per cycle
    assert abs(a1["cumulative_adjustment"]) <= MAX_CUMULATIVE + 1e-9
    # patient-specific learning must not touch the global model
    assert client.get("/learning/weights").json()["axis_weights"].get("A1", 1.0) == g0


def test_improvement_must_be_sustained_before_relaxing():
    """One good visit must not immediately de-prioritise an axis."""
    cid = "P12-SUSTAIN"
    p = {**_PATIENT, "case_id": cid}
    rid = client.post("/report", json=p).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 21.0, "followup": 6.0})    # improved once
    out = client.post(f"/learning/adapt/{cid}").json()["updated"]["A1"]
    assert out["step"] == 0.0                                  # held, not relaxed
    assert out["consecutive_improvements"] == 1
    assert "not yet sustained" in (out["note"] or "")


def test_patient_model_is_versioned_and_never_deleted():
    cid = "P12-MODEL"
    p = {**_PATIENT, "case_id": cid}
    client.post("/report", json=p)
    v1 = client.post(f"/patient-model/{cid}/rebuild", params={"change_kind": "CONFIRM"}).json()
    v2 = client.post(f"/patient-model/{cid}/rebuild",
                     params={"change_kind": "CONTRADICT", "trigger": "new lab"}).json()
    assert v1["version"] == 1 and v2["version"] == 2
    hist = client.get(f"/patient-model/{cid}").json()
    assert len(hist["history"]) == 2                           # v1 retained, not deleted
    assert hist["current"]["version"] == 2
    assert hist["current"]["change_kind"] == "CONTRADICT"
    assert hist["history"][0]["superseded"] is True            # marked, still present


def test_network_behaviour_is_recorded():
    """Level 2: the AI learns network behaviour, not disease labels."""
    cid = "P12-NET"
    p = {**_PATIENT, "case_id": cid}
    rid = client.post("/report", json=p).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 21.0, "followup": 6.0})
    out = client.post(f"/learning/networks/{cid}").json()
    assert out["level"] == "NETWORK_BEHAVIOUR"
    assert out["observations"] and out["observations"][0]["changed_first"] == "A1"
    assert out["observations"][0]["direction"] == "improved"


def test_temporal_trajectory():
    cid = "TEMPORAL-1"
    p = {**_PATIENT, "case_id": cid}
    rid = client.post("/report", json=p).json()["report_id"]
    client.post("/outcome", json={"report_id": rid, "marker": "CRP",
                                  "baseline": 21.0, "followup": 6.0})    # CRP down = improving
    t = client.get(f"/temporal/{cid}").json()
    crp = [m for m in t["markers"] if m["marker"] == "CRP"][0]
    assert crp["trend"] == "improving"


# ---- Extraction: /extract reads an uploaded file into candidate labs ----
def test_extract_endpoint_parses_labs_and_drops_pii():
    content = b"CRP 21.57 mg/L (ref < 5.0)\nHbA1c: 6.5 % (4.0-5.6)\nName: Jane Doe\n"
    files = {"file": ("labs.txt", content, "text/plain")}
    r = client.post("/extract", files=files, data={"doc_type": "lab"}).json()
    analytes = {l["analyte"] for l in r["labs"]}
    assert "CRP" in analytes                 # value extracted
    assert "Name" not in analytes            # PII line not treated as a lab
    assert r["confirmed"] is False           # human-review gate preserved
    assert r["source"] == "text"


# ---- Official module registry (domain-expert answers) ----
def test_report_stamps_registry_versions():
    r = client.post("/report", json=_PATIENT).json()
    assert r["registry_version"] and r["framework_version"]   # traceable to the data that made it


def test_modules_carry_role_and_dose_type():
    r = client.post("/report", json=_PATIENT).json()
    assert r["modules"], "expected modules for a high-severity case"
    m = r["modules"][0]
    assert m["dose_type"] in ("severity", "bowel")
    assert set(m["axis_roles"].values()) <= {"primary", "secondary"}   # per-axis role tagged


def test_bowel_group_uses_bowel_dose():
    r = client.post("/report", json=_PATIENT).json()
    bowel = [m for m in r["modules"] if m["dose_type"] == "bowel"]
    assert bowel and all("bowel" in (m["dose"] or "").lower() for m in bowel)


def test_a37_has_modules():
    # A37 (Protein Quality Control) — interim support assigned by the experts
    from app.knowledge.repository import KnowledgeRepo
    mods = KnowledgeRepo().modules_for_axis("A37")
    names = {(m.get("name") or "").upper() for m in mods}
    assert mods and any(n in names for n in ("KS", "KERRA CAPSULE", "VITALPLUS"))


# ---- Phase 6: Evidence & Confidence model (expert Q7/Q9) ----
def test_every_assessed_axis_carries_the_four_dimensions():
    r = client.post("/analyze", json=_PATIENT).json()
    assert r["axis_scores"]
    for a in r["axis_scores"]:
        assert a["evidence_status"] in ("MEASURED", "DERIVED", "CLINICAL_INFERENCE",
                                        "HYPOTHESIS", "NOT_AVAILABLE")
        assert 0.0 <= a["confidence"] <= 1.0
        assert 0.0 <= a["data_completeness"] <= 1.0
        assert 0.0 <= a["biological_uncertainty"] <= 1.0
        assert a["status"] in ("AXIS_CANDIDATE", "PROVISIONALLY_ASSESSED",
                                "ASSESSED", "HIGH_CONFIDENCE_ASSESSED")


def test_unassessed_axes_are_explicit_never_normal():
    """No data must NEVER be interpreted as normal (and never defaulted to 50)."""
    r = client.post("/analyze", json=_PATIENT).json()
    na = r["not_assessed"]
    assert na, "axes without evidence must be declared, not silently omitted"
    for a in na:
        assert a["score"] is None                      # never 0, never 50
        assert a["status"] == "NOT_ASSESSED"
        assert a["confidence"] == "INSUFFICIENT_DATA"
        assert a["reason"]
    # every one of the 39 axes is accounted for exactly once
    codes = {a["axis_code"] for a in r["axis_scores"]} | {a["axis_code"] for a in na}
    assert len(codes) == 39


def test_hypothesis_evidence_never_raises_a_score():
    from app.evidence import EvidenceStatus, EvidenceItem, axis_status, contributes_to_score
    assert contributes_to_score(EvidenceStatus.HYPOTHESIS) is False
    assert contributes_to_score(EvidenceStatus.NOT_AVAILABLE) is False
    assert contributes_to_score(EvidenceStatus.MEASURED) is True
    only_hypo = [EvidenceItem(source="rag guess", status=EvidenceStatus.HYPOTHESIS)]
    status, reason = axis_status(only_hypo, 0.0)
    assert status.value == "NOT_ASSESSED" and reason == "INSUFFICIENT_DATA"


def test_data_completeness_is_weighted_not_a_raw_ratio():
    """Missing a CORE marker must not yield high completeness (expert rejected present/expected)."""
    from app.evidence import EvidenceItem, EvidenceStatus, EvidenceWeight, data_completeness
    expected = [EvidenceWeight.CORE, EvidenceWeight.OPTIONAL]
    only_optional = [EvidenceItem(source="x", status=EvidenceStatus.MEASURED,
                                  weight_class=EvidenceWeight.OPTIONAL)]
    only_core = [EvidenceItem(source="y", status=EvidenceStatus.MEASURED,
                              weight_class=EvidenceWeight.CORE)]
    # 1-of-2 items either way, but the CORE one must score far higher than the OPTIONAL one
    assert data_completeness(only_optional, expected) == 0.25
    assert data_completeness(only_core, expected) == 0.75


def test_measured_evidence_beats_symptom_only_on_confidence():
    r = client.post("/analyze", json=_PATIENT).json()
    by = {a["axis_code"]: a for a in r["axis_scores"]}
    a1 = by.get("A1")                                  # from measured CRP
    assert a1 and a1["evidence_status"] == "MEASURED"
    sym = [a for a in r["axis_scores"] if a["evidence_status"] == "CLINICAL_INFERENCE"]
    if sym:                                            # symptom-only axes must be less certain
        assert a1["confidence"] >= sym[0]["confidence"]
        assert a1["biological_uncertainty"] <= sym[0]["biological_uncertainty"]


def test_analysis_is_version_stamped():
    r = client.post("/analyze", json=_PATIENT).json()
    assert r["axis_master_version"] == "39-axis-master-260715"
    assert r["nss_algorithm_version"]


# ---- Phase 7: Red-Flag screening (expert Q5 — mandatory, deterministic, runs FIRST) ----
def test_clean_patient_has_no_red_flags_and_keeps_module_plan():
    r = client.post("/report", json=_PATIENT).json()
    assert r["red_flag_screen"]["red_flags"] == []
    assert r["release_block"] is False
    assert r["modules"], "a patient without red flags must still get a plan"


def test_class1_emergency_blocks_module_plan_and_release():
    p = {**_PATIENT, "case_id": "RF-EMERG",
         "symptoms": "sudden slurred speech and limb weakness since this morning"}
    r = client.post("/report", json=p).json()
    s = r["red_flag_screen"]
    assert s["action_class"] == "CLASS_1_EMERGENCY"
    assert s["required_action"] == "EMERGENCY_REFERRAL"
    assert s["module_plan_allowed"] is False
    assert r["modules"] == [], "emergency must block the module plan entirely"
    assert r["release_block"] is True and r["deliverable"] is False
    assert r["escalation"] == "EMERGENCY_REFERRAL"


def test_class2_urgent_blocks_release_but_allows_plan_with_override():
    p = {**_PATIENT, "case_id": "RF-URG", "symptoms": "black stool and severe abdominal pain"}
    r = client.post("/report", json=p).json()
    s = r["red_flag_screen"]
    assert s["action_class"] == "CLASS_2_URGENT"
    assert s["release_block"] is True
    assert s["module_plan_requires_clinician_override"] is True
    assert r["deliverable"] is False


def test_critical_value_survives_a_low_nss():
    """A critical alert must NEVER be diluted by averaging (NSS 28 + critical K+ is still urgent)."""
    p = {"case_id": "RF-CRIT", "age_band": "50s", "sex": "male", "symptoms": "mild tiredness",
         "labs": [{"analyte": "Potassium", "value": 6.9, "unit": "mmol/L"}],
         "consent": {"ai_analysis": True}}
    r = client.post("/report", json=p).json()
    s = r["red_flag_screen"]
    assert s["critical_values"], "critical potassium must be detected"
    assert s["action_class"] == "CLASS_1_EMERGENCY"
    assert r["release_block"] is True
    # the low overall score must NOT suppress the critical alert
    assert r["analysis"]["nss"] < 61 and r["escalation"] == "EMERGENCY_REFERRAL"


def test_critical_value_rules_are_unit_aware():
    """5.5 mmol/L glucose is normal; it must not be read against the mg/dL rule."""
    p = {"case_id": "RF-UNIT", "symptoms": "none",
         "labs": [{"analyte": "Glucose", "value": 5.5, "unit": "mmol/L"}],
         "consent": {"ai_analysis": True}}
    assert client.post("/screen", json=p).json()["red_flags"] == []


def test_screen_endpoint_is_deterministic_and_repeatable():
    p = {**_PATIENT, "symptoms": "worst headache of my life"}
    a = client.post("/screen", json=p).json()
    b = client.post("/screen", json=p).json()
    assert a["action_class"] == b["action_class"] == "CLASS_1_EMERGENCY"
    assert [f["red_flag_code"] for f in a["red_flags"]] == [f["red_flag_code"] for f in b["red_flags"]]
    assert a["registry_authoritative"] is False      # provisional list, honestly labelled


# ---- Phase 10: NSS v0.1 + Module Match v1.0 (expert Q3 / Q10) ----
def test_module_ceiling_matching_is_not_prescribing():
    """A severe patient matches dozens of modules; the PLAN must stay clinically realistic."""
    r = client.post("/report", json=_PATIENT).json()
    sel = r["module_selection"]
    counts = sel["counts"]
    assert counts["matched"] > 20, "broad axis maps should still match many candidates"
    assert counts["selected"] <= sel["limits"]["typical_total_max"] == 6
    assert counts["core"] <= 3 and counts["supporting"] <= 3
    assert len(r["modules"]) == counts["selected"]
    assert sel["match_version"] == "TSPI-Module-Match-v1.0"


def test_every_unselected_module_has_a_structured_reason():
    r = client.post("/report", json=_PATIENT).json()
    assert r["considered_modules"], "unselected candidates must be surfaced, not dropped"
    allowed = {"CONTRAINDICATED", "MAJOR_INTERACTION", "DUPLICATE_MECHANISM",
               "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_PATIENT_DATA", "NOT_STEP_COMPATIBLE",
               "EXCESSIVE_MODULE_BURDEN", "LOWER_RANKED_THAN_SELECTED_ALTERNATIVE",
               "PHYSICIAN_EXCLUDED", "DIAGNOSTIC_WORKUP_REQUIRED_FIRST"}
    for m in r["considered_modules"]:
        assert m["reason_not_selected"] in allowed, m["module_code"]


def test_mechanism_deduplication_happens():
    r = client.post("/report", json=_PATIENT).json()
    dups = [m for m in r["considered_modules"] if m["reason_not_selected"] == "DUPLICATE_MECHANISM"]
    assert dups, "modules sharing a dominant mechanism should be de-duplicated"
    assert all(m["duplicate_of"] for m in dups), "a duplicate must name its representative"


def test_selected_modules_are_ranked_and_scored():
    r = client.post("/report", json=_PATIENT).json()
    scores = [m["module_match_score"] for m in r["modules"]]
    assert all(s is not None for s in scores)
    for m in r["modules"]:
        b = m["match_breakdown"]
        assert {"axis_match", "network_match", "evidence_grade", "phenotype_fit",
                "safety", "step_compatibility", "historical_response"} <= set(b)


def test_safety_is_a_gate_not_a_ten_point_penalty():
    """An EXCLUDED module must have score not_calculated (None) -- not merely -10."""
    from app.module_match import select
    cands = [
        {"module_code": "X1", "module_status": "EXCLUDED", "module_match_score": -1, "dedupe_key": None},
        {"module_code": "X2", "module_status": "ELIGIBLE", "module_match_score": 80.0, "dedupe_key": "k1"},
    ]
    out = select(cands)
    assert out["counts"]["excluded_by_safety_gate"] == 1
    assert [c["module_code"] for c in out["selected"]] == ["X2"]   # excluded never ranks


def test_nss_v03_reports_reliability_separately_and_is_explainable():
    """5 Aug §14: NSS reports Observed value + coverage/confidence/uncertainty/status separately."""
    r = client.post("/report", json=_PATIENT).json()
    d = r["nss_detail"]
    assert d["algorithm_version"] == "TSPI-NSS-v0.3"
    assert d["final_nss"] == r["analysis"]["nss"] == d["observed_nss"]
    assert d["assessment_status"] in ("NOT_ASSESSED", "PROVISIONAL", "ASSESSED")
    assert 0 <= d["assessment_coverage"] <= 100
    assert 0.0 <= d["overall_confidence"] <= 1.0
    assert 0.0 <= d["biological_uncertainty"] <= 1.0
    assert d["network_factor"] in (1.0, 1.05, 1.10)
    for c in d["contributors"]:
        assert c["status"] in ("PROVISIONALLY_ASSESSED", "ASSESSED", "HIGH_CONFIDENCE_ASSESSED")
        assert 0.0 <= c["confidence"] <= 1.0 and 0.0 <= c["data_completeness"] <= 1.0


def test_nss_severity_is_not_reduced_by_low_completeness():
    """5 Aug FINAL: a severe abnormality stays severe even when data completeness is low.

    `Severity x Completeness` (e.g. 90 x 0.30 = 27) is explicitly prohibited.
    """
    from app.pipeline import severity
    from app.schemas import AxisScore, Severity
    severe_but_incomplete = AxisScore(
        axis_code="A1", axis_name="Systemic Inflammatory Load", domain_code="D1",
        severity=Severity.pathological, score=90.0,
        status="ASSESSED", confidence=0.2, data_completeness=0.30, scoring_effect=True)
    nss = severity.compute_nss([severe_but_incomplete])
    assert nss >= 90, "low completeness must NOT drag a severe axis down to a mild score"


def test_axis_candidate_and_not_assessed_are_excluded_from_nss():
    from app.pipeline import severity
    from app.schemas import AxisScore, Severity
    scored = AxisScore(axis_code="A1", axis_name="x", severity=Severity.pathological, score=100.0,
                       status="ASSESSED", scoring_effect=True)
    candidate = AxisScore(axis_code="A5", axis_name="y", severity=Severity.pathological, score=100.0,
                          status="AXIS_CANDIDATE", scoring_effect=True)
    d = severity.nss_detail([scored, candidate])
    assert d["included_axis_count"] == 1                # candidate excluded
    assert {c["axis"] for c in d["contributors"]} == {"A1"}


def test_network_factor_bands():
    from app.pipeline.severity import network_factor
    assert network_factor(0) == 1.00 and network_factor(2) == 1.00
    assert network_factor(3) == 1.05 and network_factor(5) == 1.05
    assert network_factor(6) == 1.10 and network_factor(12) == 1.10


def test_no_universal_inflammatory_multiplier():
    """The old 0.6*burden + 0.4*inflammation heuristic must be gone (expert warning)."""
    import inspect
    from app.pipeline import severity
    src = inspect.getsource(severity)
    assert "0.4 * infl" not in src and "_INFLAMMATORY_AXES" not in src


# ---- Phase 8: Clinical Phenotype Engine (expert Part 1 + Q4) ----
_SYMPTOM_ONLY = {
    "case_id": "PH-NOLABS", "age_band": "40s", "sex": "female",
    "symptoms": ("bloating after meals, early satiety, belching, constipation, "
                 "worse when stressed. Gastroscopy normal."),
    "consent": {"ai_analysis": True},
}


def test_axes_can_be_assessed_from_symptoms_without_any_labs():
    """The original failure: symptoms-only produced an almost empty report."""
    r = client.post("/report", json=_SYMPTOM_ONLY).json()
    a = r["analysis"]
    assert not _SYMPTOM_ONLY.get("labs")
    assert len(a["axis_scores"]) >= 3, "symptoms alone must be able to assess axes"
    assert a["nss"] > 0
    assert r["modules"], "a symptom-only patient must still receive a plan"
    assert any(x["evidence_status"] == "CLINICAL_INFERENCE" for x in a["axis_scores"])


def test_one_symptom_never_maps_to_exactly_one_axis():
    """Expert hard rule: never `one symptom = one axis`."""
    p = {"case_id": "PH-1", "symptoms": "dizziness", "consent": {"ai_analysis": True}}
    ph = client.post("/analyze", json=p).json()["phenotype"]
    assert len(ph["differential"]) >= 3, "a nonspecific symptom must yield a differential"
    assert ph["axis_candidates"]


def test_co_symptoms_shift_the_weighting():
    """bloating alone is nonspecific; + early satiety + belching -> A16 becomes high."""
    alone = client.post("/analyze", json={"case_id": "PH-A", "symptoms": "bloating",
                                          "consent": {"ai_analysis": True}}).json()["phenotype"]
    cluster = client.post("/analyze", json={
        "case_id": "PH-B", "symptoms": "bloating, early satiety, belching after meals",
        "consent": {"ai_analysis": True}}).json()["phenotype"]
    assert cluster["axis_candidates"]["A16"] > alone["axis_candidates"]["A16"]
    assert any(c["cluster_code"] == "CL-UPPER-GI" for c in cluster["clusters"])


def test_normal_test_narrows_but_does_not_erase_the_symptom():
    ph = client.post("/analyze", json=_SYMPTOM_ONLY).json()["phenotype"]
    neg = [n for n in ph["negative_findings"] if n["test"] == "gastroscopy"]
    assert neg, "a normal gastroscopy should be interpreted, not ignored"
    assert "A16" in neg[0]["axes_still_possible"]     # symptom survives the normal test
    assert neg[0]["does_not_exclude"]
    assert ph["axis_candidates"].get("A16", 0) > 0


def test_symptom_burden_is_separate_from_attribution_confidence():
    """Severe symptoms must not imply a confident mechanism."""
    r = client.post("/analyze", json=_SYMPTOM_ONLY).json()
    ph = r["phenotype"]
    assert ph["symptom_burden"] > 0
    sym_axes = [a for a in r["axis_scores"] if a["evidence_status"] == "CLINICAL_INFERENCE"]
    assert sym_axes
    # burden can be high while confidence stays low and uncertainty high
    assert all(a["confidence"] <= 0.75 for a in sym_axes)
    assert any(a["biological_uncertainty"] >= 0.25 for a in sym_axes)


def test_adaptive_questions_are_offered_for_missing_discriminators():
    ph = client.post("/analyze", json=_SYMPTOM_ONLY).json()["phenotype"]
    assert len(ph["adaptive_questions"]) >= 3
    assert ph["dictionary_authoritative"] is False       # provisional dictionary, honestly flagged


def test_structured_symptoms_are_accepted():
    p = {"case_id": "PH-STRUCT", "consent": {"ai_analysis": True}, "symptoms": "",
         "structured_symptoms": [{
             "symptom_code": "SYM-BLOAT", "canonical_name_en": "Bloating", "duration": "2 years",
             "frequency": "daily", "severity": 7, "meal_relationship": "after meals",
             "stress_relationship": "worse with stress",
             "associated_symptoms": ["early satiety", "belching"],
             "negative_findings": ["gastroscopy normal"]}]}
    r = client.post("/analyze", json=p).json()
    ph = r["phenotype"]
    assert any(s["matched_on"] in ("structured", "associated") for s in ph["symptoms_identified"])
    assert ph["axis_candidates"].get("A16", 0) > 0
    assert ph["negative_findings"], "structured negative_findings must be interpreted"


# ---- Marker Registry (25 Jul Q1 + Q5): value_source_type, direction rules, ferritin=A1 ----
def test_marker_value_source_is_separate_from_direction():
    from app.markers import lookup, ValueSourceType, DirectionRule
    homa = lookup("HOMA-IR")
    assert homa.value_source_type is ValueSourceType.DERIVED
    assert homa.direction_rule is DirectionRule.LOWER_BETTER
    assert homa.calculation_rules and homa.required_inputs   # DERIVED carries its formula(s)
    crp = lookup("crp")
    assert crp.value_source_type is ValueSourceType.MEASURED


def test_optimal_range_is_u_shape_family():
    from app.markers import lookup, DirectionRule, CurveType, interpret
    tsh = lookup("tsh")
    assert tsh.direction_rule is DirectionRule.TARGET_RANGE
    assert tsh.curve_type is CurveType.U_SHAPE          # 25 Jul Q5b: Optimal Range == U-shape
    assert interpret(tsh, 0.1)["abnormal"] and interpret(tsh, 9)["abnormal"]   # both tails bad
    assert not interpret(tsh, 1.5)["abnormal"]


def test_ferritin_inflammatory_axis_is_A1_not_A34():
    from app.markers import lookup, resolve_context_axes, DirectionRule
    f = lookup("ferritin")
    assert f.direction_rule is DirectionRule.CONTEXT_DEPENDENT
    assert resolve_context_axes(f, "down", set()) == ["A26"]              # low -> hematopoiesis
    assert resolve_context_axes(f, "up", {"CRP"}) == ["A1"]               # high + inflammation -> A1
    assert resolve_context_axes(f, "up", set()) == []                     # high alone -> assert nothing
    # A34 must never be the ferritin inflammatory axis
    assert "A34" not in resolve_context_axes(f, "up", {"CRP"})


def test_derived_marker_reads_as_derived_evidence():
    from app.pipeline.normalizer import normalize
    from app.schemas import PatientInput, LabResult
    p = PatientInput(case_id="M-1", age_band="40s", sex="male",
                     labs=[LabResult(analyte="HOMA-IR", value=4.2, ref_low=0.5, ref_high=2.0)])
    sig = [s for s in normalize(p) if "HOMA" in s.source][0]
    assert sig.value_source_type == "DERIVED"


# ---- Phase 11: legacy->current axis converter (Blocker 1, 25 Jul mapping) ----
def test_tier_a_axes_auto_convert():
    from app.registry_convert import convert_axis_ref
    a = convert_axis_ref("Axis 16 - Liver Detoxification")
    assert a["current_code"] == "A15" and a["auto_applied"] is True
    assert convert_axis_ref("Axis 21 – Endothelial Function")["current_code"] == "A20"


def test_same_number_different_meaning_is_quarantined_never_shifted():
    """Legacy 'Metabolic Balance' must NOT become current A27 automatically (different concept)."""
    from app.registry_convert import convert_axis_ref
    m = convert_axis_ref("Axis 27 - Metabolic Balance")
    assert m["auto_applied"] is False and m["review_required"] is True
    assert m["current_code"] is None                       # never auto-applied
    assert m["conversion_type"] == "SAME_NUMBER_DIFFERENT_MEANING"


def test_cela_converts_to_network_not_axis():
    from app.registry_convert import convert_axis_ref
    c = convert_axis_ref("Axis 39 Cognitive-Emotional Loop")
    assert c["conversion_type"] == "DEPRECATED_TO_NETWORK"
    assert c["network_code"] == "N120" and c["auto_applied"] is False


def test_registry_conversion_quarantines_modules_with_unresolved_refs():
    from app.registry_convert import convert_registry
    mods = [
        {"module_code": "H-A", "name_en": "SAFE", "legacy_target_axes":
            ["Axis 16 - Liver Detoxification", "Axis 21 – Endothelial Function"]},
        {"module_code": "H-B", "name_en": "NEEDS REVIEW", "legacy_target_axes":
            ["Axis 16 - Liver Detoxification", "Axis 27 - Metabolic Balance"]},
    ]
    res = convert_registry(mods)
    safe = [m for m in res["modules"] if m["module_code"] == "H-A"][0]
    needs = [m for m in res["modules"] if m["module_code"] == "H-B"][0]
    assert safe["production_allowed"] is True and safe["target_axes"] == ["A15", "A20"]
    assert needs["production_allowed"] is False              # a quarantined ref blocks production
    assert res["production_allowed"] is False and res["review_sheet"]


# ---- Phase 8.3b: Named Phenotype Registry (25 Jul Q3, Option A) ----
def test_phenotype_registry_is_named_versioned_and_candidate():
    from app import phenotype_registry as PR
    m = PR.match({"SYM-BLOAT", "SYM-EARLY-SATIETY", "SYM-BELCHING"})
    assert m, "expected a named phenotype to fire"
    up = [p for p in m if p["phenotype_code"] == "PH-UPPER-GI-DYS"][0]
    assert up["phenotype_name_en"] == "Upper-Digestive Dysfunction"
    assert up["status"] == "CANDIDATE" and up["version"]     # named + versioned + candidate
    assert PR.is_authoritative() is False                    # dev-proposed, not yet approved
    assert "A16" in up["candidate_axis_codes"]


def test_phenotype_requires_its_required_feature_and_minimum_count():
    from app import phenotype_registry as PR
    assert PR.match({"SYM-EARLY-SATIETY"}) == []             # required SYM-BLOAT absent
    assert PR.match({"SYM-BLOAT"}) == []                     # only 1 feature < minimum 2


def test_symptom_to_phenotype_to_axis_path_in_evaluate():
    """The engine must expose the Symptom -> Phenotype -> Axis path, not just Symptom -> Axis."""
    r = client.post("/report", json=_SYMPTOM_ONLY).json()
    ph = r["analysis"].get("phenotype") or {}
    names = {p["phenotype_code"] for p in ph.get("phenotypes", [])}
    assert "PH-UPPER-GI-DYS" in names or "PH-INTESTINAL-MOTILITY" in names
    assert ph.get("phenotype_path", "").startswith("Symptom -> Phenotype -> Axis")


# ---- 31 Jul refinements (§3) ----
def test_ferritin_primary_axis_is_A26_secondary_A1():
    from app.markers import lookup
    f = lookup("ferritin")
    assert f.primary_axis == "A26"           # 31 Jul §6.3
    assert "A1" in f.secondary_axes


def test_homa_ir_formula_is_unit_specific_and_guarded():
    from app.markers import lookup, formula_for_unit
    h = lookup("HOMA-IR")
    assert "405" in formula_for_unit(h, "mg/dL")
    assert "22.5" in formula_for_unit(h, "mmol/L")
    assert formula_for_unit(h, None) is None          # units MUST be validated before calc
    assert formula_for_unit(h, "bogus") is None


def test_ungraded_module_is_not_defaulted_to_D():
    from app.module_match import score_module
    sc = score_module({"target_axes": ["A1"], "primary_axes": ["A1"]}, {"A1": 1.0})
    assert sc["evidence_grade_letter"] == "EVIDENCE_NOT_GRADED"   # not "D"
    assert sc["breakdown"]["evidence_grade"] == 0.5              # neutral, not weakest (0.25)


def test_network_component_is_validation_not_zero():
    from app.module_match import score_module
    sc = score_module({"target_axes": ["A1"], "primary_axes": ["A1"]}, {"A1": 1.0})
    assert sc["network_validation_status"] == "NOT_ASSESSED"     # not silently 0
    assert sc["provisional"] is True
    assert 0 < sc["validated_score_coverage"] <= 100


def test_axis_status_uses_official_five_states():
    from app.evidence import AxisStatus
    assert {s.value for s in AxisStatus} == {
        "NOT_ASSESSED", "AXIS_CANDIDATE", "PROVISIONALLY_ASSESSED",
        "ASSESSED", "HIGH_CONFIDENCE_ASSESSED"}


def test_safety_gate_is_categorical():
    """Safety is expressed as a categorical outcome, never a numeric score (31 Jul Directive 8)."""
    from app.safety import SafetyOutcome
    valid = {o.value for o in SafetyOutcome}
    assert valid == {"PASS", "PASS_WITH_MONITORING", "HOLD",
                     "CONTRAINDICATED", "INSUFFICIENT_SAFETY_DATA"}
    p = {**_PATIENT, "conditions": ["pregnancy"]}
    r = client.post("/report", json=p).json()
    presented = r["modules"] + r["considered_modules"]
    # every presented module carries a valid categorical safety_outcome
    assert presented and all(m.get("safety_outcome") in valid for m in presented)
    # an absolutely contraindicated module is CONTRAINDICATED, not merely low-scored
    excluded = [m for m in presented if m.get("module_status") == "EXCLUDED"]
    assert all(m["safety_outcome"] == "CONTRAINDICATED" for m in excluded)


# ---- Clinician override / update plan (POST /reports/{id}/override) ----
def test_override_edits_plan_and_requires_reapproval():
    # generate + approve a plan
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    client.post("/validate", json={"report_id": rid, "doctor_id": "dr.smith", "decision": "approve"})
    assert client.get(f"/reports/{rid}").json()["deliverable"] is True
    # pick a presented module to remove
    rep = client.get(f"/reports/{rid}").json()
    mods = rep["payload"].get("modules", [])
    assert mods, "expected at least one module to edit"
    code = mods[0]["module_code"]
    # apply a structured override
    out = client.post(f"/reports/{rid}/override", json={
        "clinician_id": "dr.smith",
        "actions": [{"action": "REMOVE_MODULE", "module_code": code,
                     "reason_code": "CLINICAL_JUDGMENT", "rationale": "not indicated"}]}).json()
    assert out["applied"][0]["status"] == "APPLIED"
    assert out["reapproval_required"] is True
    assert out["deliverable"] is False
    # editing invalidated the approval
    after = client.get(f"/reports/{rid}").json()
    assert after["deliverable"] is False
    ov = after["payload"]["clinician_overrides"]
    assert ov and ov[0]["reason_code"] == "CLINICAL_JUDGMENT" and ov[0]["before"] is not None


def test_override_unknown_report_404():
    r = client.post("/reports/nope/override", json={
        "clinician_id": "d", "actions": [{"action": "ADD_NOTE", "reason_code": "CLINICAL_JUDGMENT"}]})
    assert r.status_code == 404


def test_override_rejects_action_without_reason_code():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    r = client.post(f"/reports/{rid}/override", json={
        "clinician_id": "d", "actions": [{"action": "ADD_NOTE"}]})  # missing reason_code
    assert r.status_code == 422


# ---- Phase B: auth + RBAC + tenant isolation ----
import contextlib
from app.config import settings as _S

@contextlib.contextmanager
def _auth_on(token="svc-test"):
    old_e, old_t = _S.auth_enabled, _S.service_tokens
    _S.auth_enabled, _S.service_tokens = True, token
    try:
        yield
    finally:
        _S.auth_enabled, _S.service_tokens = old_e, old_t

def _hdr(role, user="u1", clinic=None, token="svc-test"):
    h = {"Authorization": f"Bearer {token}", "X-TSPI-Role": role, "X-TSPI-User-Id": user}
    if clinic:
        h["X-TSPI-Clinic-Id"] = clinic
    return h

_PT = {**_PATIENT, "case_id": "CASE-P1"}

def test_auth_disabled_by_default_keeps_endpoints_open():
    assert _S.auth_enabled is False
    assert client.post("/analyze", json=_PATIENT).status_code == 200   # no headers needed

def test_missing_token_is_401_when_auth_on():
    with _auth_on():
        assert client.post("/analyze", json=_PATIENT).status_code == 401

def test_patient_self_service_generates_own_unreviewed_draft():
    """TSPI Digital: a patient may screen, analyse and generate their OWN draft, read it while it
    is still an unreviewed AI draft, and never write to it (approval stays with clinicians)."""
    with _auth_on():
        assert client.post("/screen", json=_PT, headers=_hdr("patient", "user_pat1")).status_code == 200
        assert client.post("/analyze", json=_PT, headers=_hdr("patient", "user_pat1")).status_code == 200
        r = client.post("/report", json=_PT, headers=_hdr("patient", "user_pat1"))
        assert r.status_code == 200
        rid = r.json()["report_id"]
        own = client.get(f"/reports/{rid}", headers=_hdr("patient", "user_pat1"))
        assert own.status_code == 200 and own.json()["deliverable"] is False
        # another patient cannot read it
        assert client.get(f"/reports/{rid}", headers=_hdr("patient", "user_pat2")).status_code == 403
        # patient cannot approve or edit
        assert client.post("/validate", json={"report_id": rid, "doctor_id": "user_pat1",
                                              "decision": "approve"},
                           headers=_hdr("patient", "user_pat1")).status_code == 403
        # a clinician can still review it
        assert client.get(f"/reports/{rid}", headers=_hdr("clinician", "dr.a")).status_code == 200

def test_clinician_can_generate():
    with _auth_on():
        r = client.post("/report", json=_PT, headers=_hdr("clinician", "dr.a", clinic="clinicA"))
        assert r.status_code == 200

def test_patient_sees_only_own_approved_plan():
    with _auth_on():
        rid = client.post("/report", json=_PT,
                          headers=_hdr("clinician", "dr.a", clinic="clinicA")).json()["report_id"]
        # draft -> patient blocked
        assert client.get(f"/reports/{rid}", headers=_hdr("patient", "CASE-P1")).status_code == 403
        # another patient blocked
        assert client.get(f"/reports/{rid}", headers=_hdr("patient", "CASE-OTHER")).status_code == 403
        # approve, then own patient can read
        client.post("/validate", json={"report_id": rid, "doctor_id": "dr.a", "decision": "approve"},
                    headers=_hdr("clinician", "dr.a"))
        assert client.get(f"/reports/{rid}", headers=_hdr("patient", "CASE-P1")).status_code == 200

def test_clinic_staff_reads_clinic_but_cannot_write():
    with _auth_on():
        rid = client.post("/report", json=_PT,
                          headers=_hdr("clinician", "dr.a", clinic="clinicA")).json()["report_id"]
        assert client.get(f"/reports/{rid}", headers=_hdr("clinic_staff", "s1", clinic="clinicA")).status_code == 200
        assert client.get(f"/reports/{rid}", headers=_hdr("clinic_staff", "s2", clinic="clinicB")).status_code == 403
        # staff cannot approve/override
        assert client.post("/validate", json={"report_id": rid, "doctor_id": "s1", "decision": "approve"},
                           headers=_hdr("clinic_staff", "s1", clinic="clinicA")).status_code == 403

def test_learning_admin_is_reviewer_only_and_audit_is_gated():
    with _auth_on():
        assert client.post("/learning/recalibrate", headers=_hdr("clinician", "dr.a")).status_code == 403
        assert client.post("/learning/recalibrate", headers=_hdr("reviewer", "rv")).status_code == 200
        assert client.get("/audit", headers=_hdr("clinician", "dr.a")).status_code == 403
        assert client.get("/audit", headers=_hdr("auditor", "au")).status_code == 200


# ---- P1: PILOT_MODE + report language ----
import contextlib as _ctx
from app.config import settings as _PS

@_ctx.contextmanager
def _pilot_on():
    old = _PS.pilot_mode
    _PS.pilot_mode = True
    try:
        yield
    finally:
        _PS.pilot_mode = old

def test_report_language_defaults_to_en():
    r = client.post("/report", json=_PATIENT).json()
    assert r["report_language"] == "en"
    assert r["pilot_mode"] is False and r["provisional"] is False

def test_report_language_override_th():
    r = client.post("/report?report_language=th", json=_PATIENT).json()
    assert r["report_language"] == "th"

def test_pilot_mode_watermarks_and_blocks_patient_release():
    with _pilot_on():
        r = client.post("/report", json=_PATIENT).json()
        assert r["pilot_mode"] is True and r["provisional"] is True
        assert "PROVISIONAL" in (r["pilot_notice"] or "")
        assert r["release_block"] is True           # provisional must not reach a patient
        assert r["deliverable"] is False


# ---- P2: candidate clinical data (cut-offs, steps, expanded registries) ----
def test_cutoffs_classify_bands_and_critical():
    from app.cutoffs import classify
    assert classify("CRP", 21)["flag"] == "high"
    assert classify("Potassium", 7.0)["band"] == "CRITICAL"
    assert classify("Hemoglobin", 6.5)["band"] == "CRITICAL"
    assert classify("Zonulin", 5)["classifiable"] is False

def test_cutoffs_fallback_when_no_ref_range():
    # lab with NO ref range -> normalizer derives the flag from the cut-off registry
    from app.pipeline.normalizer import normalize
    from app.schemas import PatientInput, LabResult
    p = PatientInput(case_id="CUT-1", age_band="40s", sex="male",
                     labs=[LabResult(analyte="CRP", value=21.0)])   # no ref_high/low
    sig = [s for s in normalize(p) if "CRP" in s.source][0]
    assert sig.direction == "up"

def test_restoration_steps_canonical():
    from app.steps import steps, name
    s = steps()
    assert len(s) == 9 and s[0]["code"] == "S1"
    assert name("S9").startswith("Prakati")

def test_registries_expanded():
    import json
    assert len(json.load(open("data/marker_registry.json"))["markers"]) >= 28
    assert len(json.load(open("data/phenotype_registry.json"))["phenotypes"]) >= 12
    ame = json.load(open("data/axis_min_evidence.json"))["axes"]
    assert len(ame) == 39
    onco = [a for a in ame if a["axis_code"] == "A37"][0]
    assert onco["allow_provisional_assessment"] is False   # oncology never provisional from symptoms


# ---- P3: candidate network layer + pilot provisional module notes ----
def test_report_carries_candidate_networks():
    r = client.post("/report", json=_PATIENT).json()
    nets = r.get("networks", [])
    assert nets and any(n["primary_axis"] == "A1" for n in nets)   # CRP -> A1 networks present
    assert all(n["mapping_status"] in ("CANDIDATE", "APPROVED") for n in nets)

def test_pilot_marks_modules_provisional():
    with _pilot_on():
        r = client.post("/report", json=_PATIENT).json()
        assert r["modules"]
        assert all(any("PROVISIONAL" in note for note in m["match_notes"]) for m in r["modules"])


# ---- P4: PII de-id boundary, encrypted storage, re-id at render, LLM guard ----
def test_engine_intake_strips_and_stores_pii_deidentified_pipeline():
    from app import deid, store
    p = {**_PATIENT, "case_id": "PII-1",
         "patient_identity": {"full_name": "John Q Patient", "dob": "1980-05-01"}}
    r = client.post("/report", json=p).json()
    # PII never appears in the returned (de-identified) report
    blob = str(r).lower()
    assert "john q patient" not in blob and "1980-05-01" not in blob
    # identity is stored (dev fallback plaintext here since no key) and re-id works
    assert deid.get_identity("PII-1")["full_name"] == "John Q Patient"

def test_reidentify_substitutes_placeholders():
    from app import deid
    deid.store_identity("PII-2", {"full_name": "Jane Roe"})
    assert deid.reidentify("Patient: {{patient.full_name}} — plan", "PII-2") == "Patient: Jane Roe — plan"

def test_llm_prompt_guard_blocks_pii():
    import pytest, asyncio
    from app import deid
    from app.pipeline.report_composer import compose
    from app.schemas import AnalysisResult
    deid.store_identity("PII-3", {"full_name": "Secret Name"})
    class _LLM:
        enabled = True
        async def complete(self, prompt): return "should not run"
    # craft an analysis whose case_id has stored PII and inject the name to simulate a leak
    a = AnalysisResult(case_id="PII-3", signals=[], axis_scores=[], root_cause_chain=[])
    a_json = a.model_copy()
    # monkey a leak: put the name into the symptoms-like field via model_dump patch is hard;
    # instead assert the guard raises when the name is present in a prompt string
    with pytest.raises(deid.PIILeak):
        deid.assert_prompt_deidentified("... Secret Name ...", "PII-3")

def test_identified_report_blocked_over_mcp_and_available_to_clinician():
    with _auth_on():
        rid = client.post("/report", json={**_PATIENT, "case_id": "PII-4",
                          "patient_identity": {"full_name": "Owen Owner"}},
                          headers=_hdr("clinician", "dr.a", clinic="clinicA")).json()["report_id"]
        # chat/MCP surface must NOT get an identified report
        mcp_hdr = {**_hdr("clinician", "dr.a", clinic="clinicA"), "X-TSPI-Source": "mcp"}
        assert client.get(f"/reports/{rid}/identified", headers=mcp_hdr).status_code == 403
        # clinician (portal source) can
        ok = client.get(f"/reports/{rid}/identified", headers=_hdr("clinician", "dr.a", clinic="clinicA"))
        assert ok.status_code == 200 and ok.json()["identified"] is True


# ---- P5: bilingual report renderer (physician + patient, en/th) ----
def test_render_physician_en():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    r = client.get(f"/reports/{rid}/render?language=en&audience=physician").json()
    md = r["markdown"]
    assert r["language"] == "en"
    assert "TSPI Integrative Case Report" in md
    assert "Axis Analysis" in md and "Module Plan" in md
    assert "{{patient.full_name}}" in md          # de-identified: placeholder, not real PII

def test_render_thai_localises_labels():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    md = client.get(f"/reports/{rid}/render?language=th").json()["markdown"]
    assert "รายงานกรณีศึกษาเชิงบูรณาการ TSPI" in md   # Thai title
    assert "A1" in md                                  # canonical codes stay stable

def test_render_patient_requires_approval():
    rid = client.post("/report", json=_PATIENT).json()["report_id"]
    # draft -> patient render blocked
    assert client.get(f"/reports/{rid}/render?audience=patient").status_code == 409
    client.post("/validate", json={"report_id": rid, "doctor_id": "dr.a", "decision": "approve"})
    r = client.get(f"/reports/{rid}/render?audience=patient")
    assert r.status_code == 200 and "Your plan" in r.json()["markdown"]

def test_render_identified_blocked_over_mcp():
    with _auth_on():
        rid = client.post("/report", json={**_PATIENT, "case_id": "REN-1",
                          "patient_identity": {"full_name": "Rendered Name"}},
                          headers=_hdr("clinician", "dr.a", clinic="clinicA")).json()["report_id"]
        mcp_hdr = {**_hdr("clinician", "dr.a", clinic="clinicA"), "X-TSPI-Source": "mcp"}
        assert client.get(f"/reports/{rid}/render?identified=true", headers=mcp_hdr).status_code == 403
        ok = client.get(f"/reports/{rid}/render?identified=true",
                        headers=_hdr("clinician", "dr.a", clinic="clinicA")).json()
        assert "Rendered Name" in ok["markdown"]      # re-identified for the clinician/portal
