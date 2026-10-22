"""Clinician step-by-step plan workflow (/plan-sessions): verified steps, confirmations,
reason-coded adjustments, go-back, RBAC, and parity with the one-shot /report."""
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

CASE = {
    "case_id": "TSPI-PS-001",
    "age_band": "40s",
    "sex": "female",
    "symptoms": "body inflammation, stress, fatigue",
    "labs": [{"analyte": "CRP", "value": 21.57, "unit": "mg/L", "ref_high": 5.0},
             {"analyte": "HbA1c", "value": 6.5, "unit": "%", "ref_high": 5.6},
             {"analyte": "Ferritin", "value": 600, "unit": "ng/mL", "ref_high": 300},
             {"analyte": "Zorblax", "value": 3, "unit": "U"}],
    "medications": ["warfarin"],
    "consent": {"ai_analysis": True},
}


def _start(case=None):
    r = client.post("/plan-sessions", json={"patient": case or CASE})
    assert r.status_code == 200, r.text
    return r.json()


def _confirm(s, note=None):
    st = s["step"]
    r = client.post(f"/plan-sessions/{s['session_id']}/confirm",
                    json={"step": st["step"], "acknowledge": st["acknowledgements_required"], "note": note})
    assert r.status_code == 200, r.text
    return r.json()


def test_intake_verifies_labs_and_waits_for_confirmation():
    s = _start()
    assert s["current_step"] == "intake" and s["step"]["number"] == 1 and s["total_steps"] == 5
    labs = {x["analyte"]: x["status"] for x in s["step"]["result"]["labs"]}
    assert labs["CRP"] == "used"
    assert labs["Zorblax"] == "not_recognised"
    assert "LAB_NOT_RECOGNISED:Zorblax" in s["step"]["acknowledgements_required"]
    assert s["steps"][1]["status"] == "pending"                 # nothing analysed yet


def test_confirm_requires_acknowledgements_and_current_step():
    s = _start()
    sid = s["session_id"]
    r = client.post(f"/plan-sessions/{sid}/confirm", json={"step": "intake", "acknowledge": []})
    assert r.status_code == 409 and "Zorblax" in str(r.json()["detail"]["missing_acknowledgements"])
    r = client.post(f"/plan-sessions/{sid}/confirm", json={"step": "analysis", "acknowledge": []})
    assert r.status_code == 409 and r.json()["detail"]["current_step"] == "intake"


def test_full_walkthrough_produces_draft_with_trail():
    s = _start()
    s = _confirm(s)
    assert s["current_step"] == "red_flags"
    s = _confirm(s)
    assert s["current_step"] == "analysis"
    res = s["step"]["result"]
    assert res["nss"] >= 0 and res["axes"] and "networks" in res
    s = _confirm(s)
    assert s["current_step"] == "modules" and s["step"]["result"]["selected"]
    s = _confirm(s)
    assert s["current_step"] == "dosing"
    assert all(m["dose"] for m in s["step"]["result"]["modules"])
    s = _confirm(s, note="reviewed")
    assert s["status"] == "completed" and s["current_step"] == "completed" and s["report_id"]
    rep = client.get(f"/reports/{s['report_id']}").json()
    assert rep["status"] == "draft" and rep["deliverable"] is False
    trail = rep["payload"]["workflow"]
    assert trail["mode"] == "stepwise" and all(x["confirmation"] for x in trail["steps"])
    # a completed session cannot be driven further
    r = client.post(f"/plan-sessions/{s['session_id']}/confirm", json={"step": "dosing"})
    assert r.status_code == 409


def test_stepwise_without_adjustments_matches_one_shot():
    s = _start()
    for _ in range(5):
        s = _confirm(s)
    step_rep = client.get(f"/reports/{s['report_id']}").json()["payload"]
    one = client.post("/report", json=CASE).json()
    assert step_rep["analysis"]["nss"] == one["analysis"]["nss"]
    assert [(m["module_code"], m["dose"]) for m in step_rep["modules"]] == \
        [(m["module_code"], m["dose"]) for m in one["modules"]]


def test_adjustments_axis_module_dose_are_applied_and_recorded():
    s = _confirm(_confirm(_start()))                     # -> analysis
    sid = s["session_id"]
    axis = s["step"]["result"]["axes"][0]["axis_code"]
    r = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "analysis", "actions": [
        {"action": "exclude_axis", "axis_code": axis, "reason_code": "CLINICAL_JUDGMENT", "rationale": "x"}]})
    assert r.status_code == 200
    s = r.json()
    assert next(a for a in s["step"]["result"]["axes"] if a["axis_code"] == axis)["clinician_decision"] == "excluded"
    s = _confirm(s)                                      # -> modules
    sel = s["step"]["result"]["selected"]
    assert all(axis not in m["target_axes"] or len(m["target_axes"]) > 1 for m in sel)
    code = sel[0]["module_code"]
    s = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "modules", "actions": [
        {"action": "remove_module", "module_code": code, "reason_code": "PATIENT_PREFERENCE"}]}).json()
    assert code not in [m["module_code"] for m in s["step"]["result"]["selected"]]
    assert s["step"]["result"]["considered"][0]["reason_not_selected"] == "REMOVED_BY_CLINICIAN"
    s = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "modules", "actions": [
        {"action": "add_module", "module_code": code, "reason_code": "CLINICAL_JUDGMENT"}]}).json()
    assert code in [m["module_code"] for m in s["step"]["result"]["selected"]]
    s = _confirm(s)                                      # -> dosing
    s = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "dosing", "actions": [
        {"action": "change_dose", "module_code": code, "new_dose": "1 cap x 2/day",
         "reason_code": "SAFETY_CONCERN"}]}).json()
    row = next(m for m in s["step"]["result"]["modules"] if m["module_code"] == code)
    assert row["dose"] == "1 cap x 2/day" and row["dose_source"] == "clinician"
    s = _confirm(s)
    rep = client.get(f"/reports/{s['report_id']}").json()["payload"]
    assert next(m for m in rep["modules"] if m["module_code"] == code)["dose"] == "1 cap x 2/day"
    assert any(axis in n for n in rep["analysis"]["notes"])
    assert rep["workflow"]["dose_overrides"][code]["reason_code"] == "SAFETY_CONCERN"


def test_adjust_validation():
    s = _confirm(_confirm(_start()))
    sid = s["session_id"]
    r = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "analysis", "actions": [
        {"action": "remove_module", "module_code": "H-003", "reason_code": "CLINICAL_JUDGMENT"}]})
    assert r.status_code == 422                       # wrong action for this step
    r = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "analysis", "actions": [
        {"action": "exclude_axis", "axis_code": "A99", "reason_code": "CLINICAL_JUDGMENT"}]})
    assert r.status_code == 422
    r = client.post(f"/plan-sessions/{sid}/adjust", json={"step": "analysis", "actions": [
        {"action": "exclude_axis", "axis_code": "A1"}]})
    assert r.status_code == 422                       # reason_code is mandatory


def test_intake_case_correction_reverifies():
    s = _start()
    fixed = {**CASE, "labs": [x for x in CASE["labs"] if x["analyte"] != "Zorblax"]}
    r = client.post(f"/plan-sessions/{s['session_id']}/adjust", json={"step": "intake", "case": fixed})
    s = r.json()
    assert "LAB_NOT_RECOGNISED:Zorblax" not in s["step"]["acknowledgements_required"]
    assert s["step"]["adjustments"][0]["action"] == "update_case"
    r = client.post(f"/plan-sessions/{s['session_id']}/adjust",
                    json={"step": "intake", "case": {**fixed, "case_id": "OTHER-1"}})
    assert r.status_code == 422


def test_go_back_reopens_and_clears_later_steps():
    s = _confirm(_confirm(_confirm(_start())))          # -> modules
    r = client.post(f"/plan-sessions/{s['session_id']}/back", json={"to_step": "analysis"})
    s = r.json()
    assert s["current_step"] == "analysis" and s["step"]["status"] == "awaiting_confirmation"
    assert [x["status"] for x in s["steps"][3:]] == ["pending", "pending"]
    r = client.post(f"/plan-sessions/{s['session_id']}/back", json={"to_step": "dosing"})
    assert r.status_code == 409


def test_red_flags_need_acknowledgement():
    case = {**CASE, "case_id": "TSPI-PS-RF", "symptoms": "sudden unilateral weakness, fatigue"}
    s = _confirm(_start(case))
    assert s["current_step"] == "red_flags"
    codes = s["step"]["acknowledgements_required"]
    assert any(c.startswith("RED_FLAG:") for c in codes)
    r = client.post(f"/plan-sessions/{s['session_id']}/confirm", json={"step": "red_flags", "acknowledge": []})
    assert r.status_code == 409


def test_consent_required():
    r = client.post("/plan-sessions", json={"patient": {**CASE, "consent": {"ai_analysis": False}}})
    assert r.status_code == 403


def test_rbac_patients_and_staff_cannot_use_and_other_clinician_blocked(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "service_tokens", "svc-test")

    def h(role, uid):
        return {"Authorization": "Bearer svc-test", "X-TSPI-User-Id": uid, "X-TSPI-Role": role}

    for role in ("patient", "clinic_staff"):
        r = client.post("/plan-sessions", json={"patient": CASE}, headers=h(role, "u1"))
        assert r.status_code == 403
    r = client.post("/plan-sessions", json={"patient": CASE}, headers=h("clinician", "doc-1"))
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert client.get(f"/plan-sessions/{sid}", headers=h("clinician", "doc-2")).status_code == 403
    assert client.get(f"/plan-sessions/{sid}", headers=h("reviewer", "rev-1")).status_code == 200
    # patients keep the one-shot plan
    assert client.post("/report", json=CASE, headers=h("patient", "pat-1")).status_code == 200


def test_cancel():
    s = _start()
    r = client.post(f"/plan-sessions/{s['session_id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.get(f"/plan-sessions/{s['session_id']}").status_code == 409
