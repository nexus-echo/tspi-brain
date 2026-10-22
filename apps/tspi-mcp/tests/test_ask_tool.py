"""tspi_ask MCP tool: forwards de-identified questions to the engine's /ask endpoint."""
import asyncio

import pytest
from fastmcp import Client

from tspi_mcp import engine_client, server


@pytest.fixture()
def engine(monkeypatch):
    calls = []

    async def post(path, json=None, params=None):
        calls.append((path, json))
        plan = ({"report_id": json["report_id"], "status": "draft", "deliverable": False}
                if json.get("report_id") else None)
        return {"question": json["question"], "answer": "A17 is Microbiome Ecology [S1].",
                "answer_mode": "llm_grounded", "plan": plan, "notice": None,
                "sources": [{"id": "S1", "doc_id": "axis:A17", "title": "Axis A17"}]}

    monkeypatch.setattr(server.engine, "post", post)
    return calls


def _call(args):
    async def go():
        async with Client(server.mcp) as c:
            return await c.call_tool("tspi_ask", args, raise_on_error=False)
    return asyncio.run(go())


def test_forwards_question_and_scope(engine, monkeypatch):
    monkeypatch.setattr(server, "_role", lambda: "clinician")
    res = _call({"question": "Which modules target A17?", "report_id": "R-1", "case_id": "CASE-1",
                 "kinds": ["axis", "module"], "language": "th",
                 "history": [{"role": "user", "content": "What is A17?"}]})
    path, body = engine[0]
    assert path == "/ask"
    assert body == {"question": "Which modules target A17?", "include_answer": True, "report_id": "R-1",
                    "kinds": ["axis", "module"], "language": "th",
                    "history": [{"role": "user", "content": "What is A17?"}]}
    assert res.structured_content["answer"].startswith("A17")


def test_case_id_and_retrieval_only(engine, monkeypatch):
    monkeypatch.setattr(server, "_role", lambda: "clinician")
    _call({"question": "Summarise the plan", "case_id": "CASE-0912", "retrieval_only": True})
    body = engine[0][1]
    assert body["case_id"] == "CASE-0912" and body["include_answer"] is False


def test_pii_question_never_reaches_engine(engine, monkeypatch):
    monkeypatch.setattr(server, "_role", lambda: "clinician")
    res = _call({"question": "Email the plan to jane@example.com"})
    assert engine == [] and "identifiers" in res.structured_content["error"]


def test_patient_draft_gets_unreviewed_notice(engine, monkeypatch):
    monkeypatch.setattr(server, "_role", lambda: "patient")
    res = _call({"question": "What is my dose?", "report_id": "R-1"})
    assert res.structured_content["notice"].startswith("UNREVIEWED AI DRAFT")


def test_ask_uses_long_timeout():
    assert "/ask" in engine_client._SLOW_PATHS
