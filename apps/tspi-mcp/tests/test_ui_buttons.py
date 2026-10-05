"""Interactive button panels (MCP-UI): who sees which button, and when panels are attached."""
import asyncio
import dataclasses
import json

import pytest
from fastmcp import Client

from tspi_mcp import server, ui
from tspi_mcp.config import settings


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def ui_on(monkeypatch):
    monkeypatch.setattr(ui, "settings", dataclasses.replace(settings, ui_mode="on"))


@pytest.fixture()
def engine(monkeypatch):
    """Fake engine: records calls, returns canned responses."""
    calls = []
    reports = {"R-1": {"id": "R-1", "status": "draft", "deliverable": False},
               "R-OK": {"id": "R-OK", "status": "validated", "deliverable": True}}

    async def post(path, json=None, params=None):
        calls.append(("POST", path, json))
        if path == "/screen":
            return {"red_flags": json.get("_flags", []) if isinstance(json, dict) else []}
        if path == "/analyze":
            return {"axes": []}
        if path == "/report":
            return {"report_id": "R-1", "status": "draft", "deliverable": False}
        if path == "/validate":
            reports["R-1"] = {"id": "R-1", "status": "validated", "deliverable": True}
            return {"ok": True}
        return {}

    async def get(path, params=None):
        calls.append(("GET", path, None))
        return dict(reports[path.rsplit("/", 1)[-1]])

    monkeypatch.setattr(server.engine, "post", post)
    monkeypatch.setattr(server.engine, "get", get)
    return calls


def as_role(monkeypatch, role):
    monkeypatch.setattr(server, "_role", lambda: role)


CASE = {"case_id": "PT-AB12CD", "age_band": "40s", "sex": "female", "symptoms": "fatigue",
        "labs": [{"analyte": "CRP", "value": 6, "unit": "mg/L"}]}


async def _call(tool, args):
    async with Client(server.mcp) as c:
        return await c.call_tool(tool, args, raise_on_error=False)


def _html(res):
    found = [b for b in res.content if b.type == "resource"]
    if not found:
        return None
    r = found[0].resource
    assert str(r.uri).startswith("ui://tspi/")
    assert r.mimeType == "text/html"
    return r.text


def _prompts(page):
    """Prompts carried by buttons (data-prompt holds a JSON string, HTML-escaped)."""
    import html as h
    import re
    return [json.loads(h.unescape(m)) for m in re.findall(r'data-prompt="([^"]*)"', page)]


# ------------------------------------------------------------------ gating
def test_no_panel_without_chat_client(engine, monkeypatch):
    as_role(monkeypatch, "clinician")
    res = _run(_call("tspi_generate_treatment_plan", {"case": CASE}))
    assert _html(res) is None                      # Claude connector / stdio: plain JSON only
    assert res.structured_content["report_id"] == "R-1"


def test_auto_mode_attaches_for_chat_client_id(engine, monkeypatch):
    as_role(monkeypatch, "clinician")
    monkeypatch.setattr(ui, "settings", dataclasses.replace(
        settings, ui_mode="auto", ui_client_ids=("client_CHAT",)))
    monkeypatch.setattr(ui, "_token_client_id", lambda: "client_CHAT")
    assert _html(_run(_call("tspi_generate_treatment_plan", {"case": CASE}))) is not None
    monkeypatch.setattr(ui, "_token_client_id", lambda: "client_OTHER")
    assert _html(_run(_call("tspi_generate_treatment_plan", {"case": CASE}))) is None


def test_off_mode_never_attaches(engine, monkeypatch):
    as_role(monkeypatch, "clinician")
    monkeypatch.setattr(ui, "settings", dataclasses.replace(settings, ui_mode="off"))
    monkeypatch.setattr(ui, "_token_client_id", lambda: "anything")
    assert _html(_run(_call("tspi_generate_treatment_plan", {"case": CASE}))) is None


# ------------------------------------------------------------------ generate / download
def test_patient_report_has_download_but_no_approve(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "patient")
    res = _run(_call("tspi_generate_treatment_plan", {"case": CASE}))
    page = _html(res)
    assert "Download PDF" in page and "Unreviewed AI draft" in page
    assert 'id="approve-ask"' not in page and 'id="confirm"' not in page
    assert _prompts(page) == ["Create and give me the PDF for report R-1."]
    assert json.loads(res.content[0].text)["report_id"] == "R-1"   # model still gets the JSON


def test_screen_and_analysis_offer_generate(engine, ui_on, monkeypatch):
    for role in ("patient", "clinician"):
        as_role(monkeypatch, role)
        page = _html(_run(_call("tspi_screen_red_flags", {"case": CASE})))
        assert "Generate report" in page
        assert "case PT-AB12CD" in _prompts(page)[0]
        page = _html(_run(_call("tspi_analyze_case", {"case": CASE})))
        assert _prompts(page) == [ui.generate_prompt("PT-AB12CD")]


def test_red_flags_block_generate_button_for_patients():
    page = ui.screen_panel("PT-1", [{"flag": "chest pain"}], "patient")
    assert "Urgent" in page and 'data-prompt="' not in page
    page = ui.screen_panel("C-1", [{"flag": "chest pain"}], "clinician")
    assert "Generate report anyway" in page


# ------------------------------------------------------------------ approve
def test_clinician_draft_has_two_step_approve(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "clinician")
    page = _html(_run(_call("tspi_get_treatment_plan", {"report_id": "R-1"})))
    assert 'id="approve-ask"' in page and 'id="confirm"' in page
    assert 'id="approve-ask" data-prompt' not in page        # first click only reveals the confirm
    assert ui.approve_prompt("R-1") in _prompts(page)


def test_reviewer_also_gets_approve(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "reviewer")
    assert 'id="approve-ask"' in _html(_run(_call("tspi_get_treatment_plan", {"report_id": "R-1"})))


def test_approved_report_has_no_approve_button(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "clinician")
    page = _html(_run(_call("tspi_get_treatment_plan", {"report_id": "R-OK"})))
    assert "Approved by clinician" in page and 'id="approve-ask"' not in page
    assert "Download PDF" in page


def test_patient_cannot_approve_even_by_typing(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "patient")
    res = _run(_call("tspi_approve_treatment_plan", {"report_id": "R-1", "decision": "approve"}))
    assert "Only a clinician or reviewer" in res.structured_content["error"]
    assert not [c for c in engine if c[1] == "/validate"]   # never reached the engine


def test_patient_cannot_edit(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "patient")
    res = _run(_call("tspi_update_treatment_plan", {"report_id": "R-1", "actions": [
        {"action": "ADD_NOTE", "reason_code": "CLINICAL_JUDGMENT", "rationale": "x"}]}))
    assert "error" in res.structured_content
    assert not [c for c in engine if "override" in c[1]]


def test_clinician_approval_returns_approved_panel(engine, ui_on, monkeypatch):
    as_role(monkeypatch, "clinician")
    res = _run(_call("tspi_approve_treatment_plan", {"report_id": "R-1", "decision": "approve"}))
    assert res.structured_content["decision"] == "approve"
    page = _html(res)
    assert "Approved by clinician" in page and 'id="approve-ask"' not in page


# ------------------------------------------------------------------ safety of the HTML
def test_values_are_escaped():
    page = ui.report_panel('R"><script>alert(1)</script>', "draft", False, "clinician")
    assert "<script>alert(1)" not in page
    page = ui.analysis_panel("<img src=x onerror=alert(1)>")
    assert "<img src=x" not in page
