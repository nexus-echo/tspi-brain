"""Knowledge Q&A API (/ask, /knowledge/search): hybrid retrieval, plan grounding, governance."""
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)

_PATIENT = {
    "case_id": "TSPI-ASK-001",
    "age_band": "40s",
    "sex": "female",
    "symptoms": "body inflammation, stress",
    "labs": [{"analyte": "CRP", "value": 21.57, "unit": "mg/L", "ref_high": 5.0},
             {"analyte": "HbA1c", "value": 6.5, "unit": "%", "ref_high": 5.6}],
    "medications": ["warfarin"],
    "consent": {"ai_analysis": True},
}


def _report_id() -> str:
    return client.post("/report", json=_PATIENT).json()["report_id"]


def test_kb_question_exact_code_retrieval():
    r = client.post("/ask", json={"question": "Which modules target axis A17?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer_mode"] == "retrieval_only"          # no LLM in tests
    assert body["sources"][0]["doc_id"] == "axis:A17"
    assert "exact" in body["sources"][0]["retrieval"]
    assert body["plan"] is None and body["notice"] is None


def test_kb_question_lexical_and_module_name():
    body = client.post("/ask", json={"question": "How is KS dosed?"}).json()
    ids = [s["doc_id"] for s in body["sources"]]
    assert "dosing:ks-protocol" in ids
    assert any(i.startswith("module:") or i.startswith("product:") for i in ids)


def test_kinds_filter():
    body = client.post("/ask", json={"question": "critical potassium", "kinds": ["safety"]}).json()
    assert body["sources"] and all(s["kind"] == "safety" for s in body["sources"])
    assert body["sources"][0]["doc_id"] == "safety:RF-CV01"


def test_plan_question_uses_report_and_marks_draft():
    rid = _report_id()
    body = client.post("/ask", json={"question": "Why was this module chosen and what is the dose?",
                                     "report_id": rid}).json()
    assert body["plan"]["report_id"] == rid and body["plan"]["deliverable"] is False
    assert "DRAFT" in body["notice"]
    assert "[PLAN]" in body["answer"] and "NSS" in body["answer"]


def test_case_id_resolves_latest_report():
    rid = _report_id()
    body = client.post("/ask", json={"question": "Summarise the plan", "case_id": "TSPI-ASK-001"}).json()
    assert body["plan"]["report_id"] == rid


def test_unknown_report_404():
    assert client.post("/ask", json={"question": "what?", "report_id": "nope"}).status_code == 404


def test_pii_in_question_rejected():
    r = client.post("/ask", json={"question": "Plan for patient, phone 081 234 5678?"})
    assert r.status_code == 422 and r.json()["error"] == "pii_detected"


def test_llm_grounded_answer_and_prompt(monkeypatch):
    seen = {}

    class FakeLLM:
        enabled = True

        async def complete(self, prompt):
            seen["prompt"] = prompt
            return "A17 is Microbiome Ecology [S1]. The draft plan is pending review [PLAN]."

    monkeypatch.setattr("app.llm.provider.LLMProvider", FakeLLM)
    rid = _report_id()
    body = client.post("/ask", json={"question": "What is axis A17 and is it in this plan?",
                                     "report_id": rid}).json()
    assert body["answer_mode"] == "llm_grounded"
    assert body["answer"].startswith("A17 is Microbiome Ecology")
    assert any(s["cited"] for s in body["sources"])
    p = seen["prompt"]
    assert "=== PLAN" in p and "[S1]" in p and "Never invent" in p
    assert "QUESTION: What is axis A17" in p


def test_llm_failure_degrades_to_retrieval(monkeypatch):
    class BrokenLLM:
        enabled = True

        async def complete(self, prompt):
            raise RuntimeError("down")

    monkeypatch.setattr("app.llm.provider.LLMProvider", BrokenLLM)
    body = client.post("/ask", json={"question": "What is Domain D1?"}).json()
    assert body["answer_mode"] == "retrieval_only"
    assert body["retrieval"]["llm_error"] == "RuntimeError"
    assert body["sources"][0]["doc_id"] == "domain:D1"


def test_knowledge_search_endpoint():
    body = client.get("/knowledge/search", params={"q": "NF-kB inflammatory network", "kinds": "network"}).json()
    assert body["hits"][0]["doc_id"] == "network:LBN-001"


def test_knowledge_health_reports_corpus():
    kb = client.get("/knowledge/health").json()["knowledge_base"]
    assert kb["documents"] > 300 and kb["by_kind"]["axis"] >= 39 and kb["by_kind"]["module"] > 50


def test_patient_cannot_ask_about_other_patients_plan(monkeypatch):
    rid = _report_id()                                   # created with auth off -> not this patient's
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(settings, "service_tokens", "svc-test")
    h = {"Authorization": "Bearer svc-test", "X-TSPI-User-Id": "patient-9", "X-TSPI-Role": "patient"}
    r = client.post("/ask", json={"question": "What is my dose?", "report_id": rid}, headers=h)
    assert r.status_code == 403
    r = client.post("/ask", json={"question": "What is axis A1?"}, headers=h)   # KB-only is fine
    assert r.status_code == 200


def test_vector_hits_fuse_with_corpus(monkeypatch):
    """pgvector path (mocked): only 'kb:' rows are requested and they join back to corpus docs."""
    seen = {}

    async def fake_aretrieve(query, kind=None, k=5, kind_prefix=None):
        seen["prefix"] = kind_prefix
        return [{"kind": "kb:network", "ref_code": "network:LBN-001", "content": "x", "score": 0.91},
                {"kind": "kb:axis", "ref_code": "gone:doc", "content": "stale", "score": 0.5}]

    monkeypatch.setattr("app.knowledge.retrieval.aretrieve", fake_aretrieve)
    body = client.post("/ask", json={"question": "cytokine switch"}).json()
    assert seen["prefix"] == "kb:"
    assert body["retrieval"]["vector_used"] is True
    top = {s["doc_id"]: s for s in body["sources"]}
    assert "vector" in top["network:LBN-001"]["retrieval"] and "gone:doc" not in top
