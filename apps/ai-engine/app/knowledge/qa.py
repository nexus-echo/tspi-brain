"""TSPI Brain Q&A — answer questions about a treatment plan and the TSPI knowledge base.

Flow:  question (+ optional report) -> PII guard -> hybrid retrieval (exact + pgvector + BM25)
       -> grounded prompt (numbered sources + the plan's own structured data) -> LLM answer with
       [S#]/[PLAN] citations.  If no LLM is available or it times out, a retrieval-only answer
       (the best matching passages) is returned instead — the endpoint never invents content.

Governance:
  * The question/history are scanned for identifiers and rejected (PIILeak -> 422) if any are
    found; the final prompt is re-checked with deid.assert_prompt_deidentified when a case is in
    scope. Clinician/patient names and emails stamped on a report are never put in the prompt.
  * The answer writer may only EXPLAIN what the engine and knowledge base already say; it must not
    invent scores, modules, doses or evidence, and must say when the sources do not contain it.
  * Plans that are not physician-approved are always labelled as AI drafts.
"""
from __future__ import annotations

import asyncio
import logging
import re

from app import deid
from app.config import settings
from app.knowledge import corpus
from app.knowledge.search import hybrid_search

_log = logging.getLogger(__name__)

DRAFT_NOTICE = ("This plan is an AI-generated DRAFT pending physician review — it is not approved "
                "for patient use.")
PATIENT_DRAFT_NOTICE = ("UNREVIEWED AI DRAFT — this plan has not been reviewed or approved by a TSPI "
                        "doctor and is not medical advice. Please contact TSPI Digital for doctor review.")
DISCLAIMER = ("Answers are generated from the TSPI knowledge base and the engine's structured output "
              "for clinician decision support. They do not replace clinical judgment.")
_SNIPPET = 320
_SRC_CHARS = 1400


# ----------------------------------------------------------------------------- PII guard
def guard_text(text: str, case_id: str | None = None) -> None:
    _, findings = deid.scrub_text(text or "", case_id)
    if findings:
        raise deid.PIILeak(f"The question contains possible patient identifiers ({', '.join(findings)}). "
                           "Remove names, phone numbers, emails or ID numbers and ask again.")


# ----------------------------------------------------------------------------- plan context
def _fmt_list(xs, n=None) -> str:
    xs = [str(x) for x in (xs or []) if x not in (None, "")]
    return ", ".join(xs[:n] if n else xs)


def plan_context(report: dict, outcomes: list[dict] | None = None) -> tuple[str, list[str]]:
    """Compact, de-identified text of a stored report + KB doc ids for its modules/axes."""
    p = report.get("payload") or {}
    a = p.get("analysis") or {}
    approved = bool(report.get("deliverable"))
    L: list[str] = []
    # (report id deliberately omitted: a hex id can contain digit runs the PII guard would flag)
    L.append(f"Treatment plan for case {report.get('case_id')} — status: {report.get('status')}"
             + (" (physician-APPROVED)" if approved else " (AI DRAFT, NOT physician-approved)")
             + (f", plan version {p['plan_version']}" if p.get("plan_version") else "") + ".")
    if p.get("pilot_notice"):
        L.append(f"Pilot notice: {p['pilot_notice']}")
    L.append(f"Network Severity Score (NSS): {a.get('nss', report.get('nss'))}/100; severity level "
             f"{a.get('severity_level', report.get('severity_level'))} {a.get('severity_name', '')}.".rstrip())
    rf = p.get("red_flag_screen") or a.get("red_flag_screen") or {}
    flags = [f.get("name_en") or f.get("red_flag_code") for f in (rf.get("red_flags") or []) + (rf.get("critical_values") or [])
             if isinstance(f, dict)]
    if flags or p.get("escalation") or p.get("release_block"):
        L.append(f"Red flags: {_fmt_list(flags) or 'none'}; escalation: {p.get('escalation') or 'none'}; "
                 f"release blocked: {bool(p.get('release_block'))}.")
    if a.get("root_cause_chain"):
        L.append("Root-cause chain: " + " -> ".join(str(x) for x in a["root_cause_chain"]))
    if a.get("system_priority"):
        L.append("System priority: " + _fmt_list([f"{s.get('system')} ({s.get('priority_score')})"
                                                   for s in a["system_priority"]], 6))
    axes = sorted(a.get("axis_scores") or [], key=lambda x: (x.get("score") is None, -(x.get("score") or 0)))
    if axes:
        L.append("Assessed axes:")
        for x in axes[:14]:
            ev = _fmt_list(x.get("evidence"), 4)
            L.append(f"  - {x.get('axis_code')} {x.get('axis_name')}: score {x.get('score')}, severity "
                     f"{x.get('severity')}, {x.get('status')}, confidence {x.get('confidence_label')}"
                     + (", DRIVER" if x.get("is_driver") else "") + (f"; evidence: {ev}" if ev else "")
                     + (f"; missing: {_fmt_list(x.get('missing_evidence'), 4)}" if x.get("missing_evidence") else ""))
    if a.get("not_assessed"):
        L.append("Not assessed (insufficient data — NOT normal): "
                 + _fmt_list([f"{x.get('axis_code')}" for x in a["not_assessed"]], 39))
    ph = a.get("phenotype") or {}
    syms = [s.get("name") for s in ph.get("symptoms_identified", []) if isinstance(s, dict)]
    if syms:
        L.append("Symptoms identified: " + _fmt_list(syms))
    mods = p.get("modules") or []
    if mods:
        L.append("Selected modules (candidate plan):")
        for m in mods:
            L.append(f"  - {m.get('module_code')} {m.get('module_name')}: dose {m.get('dose')}; {m.get('phase')}; "
                     f"tier {m.get('tier')}; dose type {m.get('dose_type')}; targets {_fmt_list(m.get('target_axes'))}; "
                     f"match score {m.get('module_match_score')}; safety {m.get('safety_outcome')}"
                     + (f"; contraindications: {_fmt_list(m.get('contraindications'))}" if m.get("contraindications") else ""))
    considered = [m for m in (p.get("considered_modules") or []) if m.get("reason_not_selected")]
    if considered:
        L.append("Considered but not selected: " + "; ".join(
            f"{m.get('module_code')} {m.get('module_name')} ({m.get('reason_not_selected')})" for m in considered[:10]))
    alerts = report.get("safety_alerts") or p.get("safety_alerts") or []
    if alerts:
        L.append("Safety alerts: " + "; ".join(f"{x.get('module') or 'general'}: {x.get('reason')} [{x.get('severity')}]"
                                               for x in alerts if isinstance(x, dict)))
    if p.get("monitoring"):
        L.append("Monitoring: " + "; ".join(corpus._json_text(m).replace("\n", " ") for m in p["monitoring"][:5]))
    if p.get("networks"):
        L.append("Candidate networks: " + _fmt_list([f"{n.get('network_code')} {n.get('name')} ({n.get('primary_axis')})"
                                                    for n in p["networks"]], 12))
    for o in p.get("clinician_overrides") or []:
        L.append(f"Clinician edit: {o.get('action')} {o.get('module_code') or o.get('axis_code') or ''} "
                 f"reason {o.get('reason_code')}" + (f" — {o.get('rationale')}" if o.get("rationale") else ""))
    if outcomes:
        L.append("Follow-up outcomes: " + "; ".join(f"{o['marker']} {o['baseline']} -> {o['followup']} (delta {o['delta']})"
                                                     for o in outcomes[-10:]))
    known = corpus.by_id()
    ids = [f"module:{m.get('module_code')}" for m in mods]
    ids += [f"axis:{x.get('axis_code')}" for x in axes if x.get("is_driver")][:4]
    ids += [f"dosing:level-{a.get('severity_level')}"]
    return "\n".join(L), [i for i in ids if i in known]


# ----------------------------------------------------------------------------- prompt
def _language_hint(question: str, language: str | None) -> str:
    if language:
        return {"th": "Thai", "en": "English"}.get(language, language)
    return "Thai" if re.search(r"[฀-๿]", question or "") else "the same language as the question"


def build_prompt(question: str, sources: list[dict], plan_text: str | None,
                 history: list[dict] | None, language: str | None, audience: str) -> str:
    parts = [
        "You are TSPI Brain, the knowledge assistant of the TSPI clinical decision-support system "
        "(3 Keys -> 9 Restoration Steps -> 12 Domains -> 39 Biological Axes -> 180 Living Biological "
        "Networks -> TSPI herbal modules).",
        "RULES:",
        "1. Answer ONLY from the SOURCES and the PLAN below. If they do not contain the answer, say so "
        "plainly and state what information is missing. Never use outside knowledge to fill gaps.",
        "2. Never invent or change axis scores, NSS, modules, doses, contraindications, evidence grades "
        "or outcomes. Quote numbers exactly as written.",
        "3. Cite every factual sentence with [S#] for knowledge sources or [PLAN] for the treatment plan.",
        "4. If the plan is not physician-approved, say it is an AI draft pending physician review.",
        "5. Safety first: mention relevant contraindications, red flags or safety alerts present in the sources.",
        f"6. Audience: {'a patient — use plain language, no jargon, and do not give instructions to change treatment; refer them to their TSPI doctor' if audience == 'patient' else 'a clinician — be precise and concise'}.",
        f"7. Write the answer in {_language_hint(question, language)}. Keep it focused (short paragraphs or bullets).",
        "",
    ]
    if plan_text:
        parts += ["=== PLAN (engine output for this case) ===", plan_text, ""]
    parts.append("=== SOURCES (TSPI knowledge base) ===")
    for i, s in enumerate(sources, 1):
        parts.append(f"[S{i}] {corpus.KINDS.get(s['kind'], s['kind'])}: {s['title']}\n{s['content'][:_SRC_CHARS]}")
    if not sources:
        parts.append("(no matching knowledge-base passages)")
    if history:
        parts += ["", "=== CONVERSATION SO FAR ==="]
        for h in history[-6:]:
            parts.append(f"{h.get('role', 'user').upper()}: {str(h.get('content', ''))[:800]}")
    parts += ["", f"QUESTION: {question}", "ANSWER:"]
    return "\n".join(parts)


def _snippet(text: str) -> str:
    t = re.sub(r"[#*`>]+", "", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:_SNIPPET] + ("…" if len(t) > _SNIPPET else "")


def _retrieval_only_answer(sources: list[dict], plan_text: str | None, reason: str) -> str:
    lines = [f"(Retrieval-only answer — {reason}. The most relevant TSPI knowledge is listed below; "
             "no text was generated.)"]
    if plan_text:
        lines += ["", "Treatment plan summary [PLAN]:", plan_text[:1500]]
    if sources:
        lines += ["", "Relevant knowledge:"]
        lines += [f"[S{i}] {s['title']}: {_snippet(s['content'])}" for i, s in enumerate(sources[:5], 1)]
    else:
        lines.append("No matching passages were found in the TSPI knowledge base.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main entry
async def answer(question: str, *, report: dict | None = None, outcomes: list[dict] | None = None,
                 history: list[dict] | None = None, kinds: list[str] | None = None,
                 top_k: int | None = None, language: str | None = None,
                 audience: str = "clinician", use_llm: bool = True) -> dict:
    case_id = report.get("case_id") if report else None
    guard_text(question, case_id)
    for h in history or []:
        guard_text(str(h.get("content", "")), case_id)

    k = max(1, min(int(top_k or settings.ask_top_k), 20))
    plan_text, plan_ids = (plan_context(report, outcomes) if report else (None, []))
    res = await hybrid_search(question, k=k, kinds=kinds, extra_ids=plan_ids[:4])
    sources = res["hits"]

    text, mode, llm_error = None, "retrieval_only", None
    if use_llm:
        from app.llm.provider import LLMProvider
        llm = LLMProvider()
        if llm.enabled:
            prompt = build_prompt(question, sources, plan_text, history, language, audience)
            if case_id:
                # Fail closed (PIILeak -> 422) on the CASE-DERIVED parts. Knowledge-base passages are
                # reference data (product codes, LC-MS figures) and are not patient data.
                case_part = "\n".join([plan_text or "", question] + [str(h.get("content", "")) for h in history or []])
                deid.assert_prompt_deidentified(case_part, case_id)
            try:
                text = (await asyncio.wait_for(llm.complete(prompt), timeout=settings.ask_llm_budget_s)).strip()
                mode = "llm_grounded" if text else "retrieval_only"
            except Exception as e:  # noqa: BLE001 — degrade to retrieval-only, never fail the request
                _log.warning("Ask: LLM answer failed (%s); returning retrieval-only answer",
                             type(e).__name__, exc_info=True)
                llm_error = type(e).__name__
        else:
            llm_error = "no LLM provider configured"
    if not text:
        text = _retrieval_only_answer(sources, plan_text,
                                      "the answer writer is unavailable" if use_llm else "generation disabled")
        mode = "retrieval_only"

    notice = None
    if report and not report.get("deliverable"):
        notice = PATIENT_DRAFT_NOTICE if audience == "patient" else DRAFT_NOTICE

    cited = set(re.findall(r"\[S(\d+)\]", text or ""))
    return {
        "question": question,
        "answer": text,
        "answer_mode": mode,
        "sources": [{"id": f"S{i}", "doc_id": s["id"], "kind": s["kind"], "ref_code": s["ref_code"],
                     "title": s["title"], "snippet": _snippet(s["content"]), "score": s.get("score"),
                     "retrieval": s.get("retrieval", []), "cited": str(i) in cited}
                    for i, s in enumerate(sources, 1)],
        "plan": ({"report_id": report.get("id"), "case_id": report.get("case_id"),
                  "status": report.get("status"), "deliverable": bool(report.get("deliverable")),
                  "used": True} if report else None),
        "notice": notice,
        "retrieval": {"vector_used": res["vector_used"], "lexical_used": res["lexical_used"],
                      "exact_matches": res["exact_matches"], "top_k": k,
                      **({"llm_error": llm_error} if llm_error else {})},
        "disclaimer": DISCLAIMER,
    }
