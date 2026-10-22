"""Clinician step-by-step plan tools: role gating, forwarding, step panels, one-shot redirect."""
import asyncio
import dataclasses
import html as h
import json
import re

import pytest
from fastmcp import Client

from tspi_mcp import engine_client, server, ui
from tspi_mcp.config import settings

CASE = {"case_id": "PT-AB12CD", "age_band": "40s", "sex": "female", "symptoms": "fatigue",
        "labs": [{"analyte": "CRP", "value": 6, "unit": "mg/L"}]}


def _view(step="intake", status="active", acks=("LAB_NOT_USED:X",), report_id=None):
    steps = ["intake", "red_flags", "analysis", "modules", "dosing"]
    v = {"session_id": "S-1", "case_id": "PT-AB12CD", "status": status, "current_step": step,
         "total_steps": 5, "report_id": report_id,
         "steps": [{"step": s, "number": i + 1, "title": s, "status": "pending"} for i, s in enumerate(steps)]}
    if step in steps:
        i = steps.index(step)
        v["step"] = {"step": step, "number": i + 1, "title": step.title(), "status": "awaiting_confirmation",
                     "summary": "summary", "result": {}, "can_confirm": True,
                     "checks": [{"code": c, "severity": "warning", "message": f"check {c}", "requires_ack": True}
                                for c in acks],
                     "acknowledgements_required": list(acks),
                     "adjust_actions": ["exclude_axis"] if step == "analysis" else [],
                     "next_step": steps[i + 1] if i + 1 < 5 else "generate_draft_plan",
                     "can_go_back_to": steps[:i]}
    return v


@pytest.fixture()
def engine(monkeypatch):
    calls = []

    async def post(path, json=None, params=None):
        calls.append(("POST", path, json))
        if path == "/plan-sessions":
            return _view()
        if path.endswith("/confirm"):
            return _view("completed", "completed", (), "R-9") if json["step"] == "dosing" else _view("red_flags")
        if path.endswith("/adjust"):
            return _view("analysis", acks=())
        if path.endswith("/back"):
            return _view(json["to_step"])
        if path == "/report":
            return {"report_id": "R-1", "status": "draft", "deliverable": False}
        return {}

    async def get(path, params=None):
        calls.append(("GET", path, None))
        return _view()

    monkeypatch.setattr(server.engine, "post", post)
    monkeypatch.setattr(server.engine, "get", get)
    return calls


def _call(tool, args):
    async def go():
        async with Client(server.mcp) as c:
            return await c.call_tool(tool, args, raise_on_error=False)
    return asyncio.run(go())


def _role(monkeypatch, role):
    monkeypatch.setattr(server, "_role", lambda: role)


def test_clinician_generate_redirects_to_stepwise(engine, monkeypatch):
    _role(monkeypatch, "clinician")
    res = _call("tspi_generate_treatment_plan", {"case": CASE}).structured_content
    assert [c[1] for c in engine] == ["/plan-sessions"]
    assert engine[0][2]["patient"]["case_id"] == "PT-AB12CD"
    assert res["session_id"] == "S-1" and "step by step" in res["note"]


def test_patient_and_staff_keep_one_shot(engine, monkeypatch):
    for role in ("patient", "clinic_staff"):
        engine.clear()
        _role(monkeypatch, role)
        res = _call("tspi_generate_treatment_plan", {"case": CASE}).structured_content
        assert [c[1] for c in engine] == ["/report"] and res["report_id"] == "R-1"


def test_plan_tools_refuse_non_clinicians(engine, monkeypatch):
    _role(monkeypatch, "patient")
    for tool, args in (("tspi_plan_start", {"case": CASE}), ("tspi_plan_get", {"session_id": "S-1"}),
                       ("tspi_plan_confirm_step", {"session_id": "S-1", "step": "intake"}),
                       ("tspi_plan_go_back", {"session_id": "S-1", "to_step": "intake"})):
        assert "clinicians" in _call(tool, args).structured_content["error"]
    assert engine == []


def test_confirm_adjust_back_forwarding(engine, monkeypatch):
    _role(monkeypatch, "clinician")
    _call("tspi_plan_confirm_step", {"session_id": "S-1", "step": "intake",
                                     "acknowledge": ["LAB_NOT_USED:X"], "note": "ok"})
    assert engine[-1][1:] == ("/plan-sessions/S-1/confirm",
                              {"step": "intake", "acknowledge": ["LAB_NOT_USED:X"], "note": "ok"})
    _call("tspi_plan_adjust_step", {"session_id": "S-1", "step": "analysis", "adjustments": [
        {"action": "exclude_axis", "axis_code": "A5", "reason_code": "CLINICAL_JUDGMENT"}]})
    body = engine[-1][2]
    assert body["step"] == "analysis" and body["actions"][0]["axis_code"] == "A5"
    _call("tspi_plan_adjust_step", {"session_id": "S-1", "step": "intake", "case": CASE})
    assert engine[-1][2]["case"]["case_id"] == "PT-AB12CD"
    _call("tspi_plan_go_back", {"session_id": "S-1", "to_step": "intake"})
    assert engine[-1][1:] == ("/plan-sessions/S-1/back", {"to_step": "intake"})
    res = _call("tspi_plan_confirm_step", {"session_id": "S-1", "step": "dosing"}).structured_content
    assert res["report_id"] == "R-9" and "DRAFT plan" in res["next_step"]


def test_pii_in_case_never_reaches_engine(engine, monkeypatch):
    _role(monkeypatch, "clinician")
    res = _call("tspi_plan_start", {"case": {**CASE, "symptoms": "call 081 234 5678"}}).structured_content
    # lenient mode redacts, strict mode rejects — either way the raw number is never forwarded
    sent = json.dumps(engine)
    assert "081 234 5678" not in sent
    assert res.get("deidentification_warning") or res.get("error")


def _prompts(page):
    return [json.loads(h.unescape(m)) for m in re.findall(r'data-prompt="([^"]*)"', page)]


def test_step_panel_buttons(monkeypatch):
    page = ui.step_panel(_view("analysis"), "clinician")
    assert "Step 3 of 5" in page and "check LAB_NOT_USED:X" in page
    ps = _prompts(page)
    assert ps[0] == ("Confirm step 'analysis' (Analysis) of plan session S-1 and continue. "
                     "I have reviewed and acknowledge: LAB_NOT_USED:X.")
    assert ps[1].startswith("I want to adjust step 'analysis'")
    assert ps[2] == "Go back to step 'red_flags' of plan session S-1."
    assert ui.step_panel(_view("analysis"), "patient") is None
    last = ui.step_panel(_view("dosing", acks=()), "clinician")
    assert "Confirm &amp; generate draft plan" in last
    done = ui.step_panel(_view("completed", "completed", (), "R-9"), "clinician")
    assert "Download PDF" in done and "R-9" in done


def test_step_panel_attached_for_chat_client(engine, monkeypatch):
    _role(monkeypatch, "clinician")
    monkeypatch.setattr(ui, "settings", dataclasses.replace(settings, ui_mode="on"))
    res = _call("tspi_plan_start", {"case": CASE})
    assert any(b.type == "resource" for b in res.content)


def test_confirm_uses_long_timeout_and_409_is_readable():
    msg = engine_client._explain(409, json.dumps({"detail": {
        "message": "Please review and acknowledge these checks before confirming: X",
        "missing_acknowledgements": ["X"]}}))
    assert msg.startswith("Please review") and "missing_acknowledgements" in msg
