"""Stage 7 — Compose the de-identified TSPI Case Report.

The LLM writes prose strictly from the structured analysis. When RAG is available, retrieved
axis context (gathered concurrently, cached) is added to the prompt. A deterministic Markdown
fallback always works.
"""
from __future__ import annotations

import asyncio

from app.knowledge import retrieval
from app.llm.provider import LLMProvider
from app.schemas import AnalysisResult, CaseReport, ModulePick

import logging

_log = logging.getLogger(__name__)

_SECTION_ORDER = (
    "Short Clinical Protocol (Step-1)",
    "Introduction (network-disorder framing)",
    "Clinical Overview",
    "Network Disease",
    "39-Axis Mapping",
    "Therapeutic Architecture (Step 1->2->3)",
    "Monitoring, Outcomes & Prakati",
)


def _deterministic_markdown(analysis: AnalysisResult, modules: list[ModulePick]) -> str:
    lines = [f"# TSPI Case Report — {analysis.case_id} (de-identified)", ""]
    lines.append("## Axis map")
    for a in analysis.axis_scores:
        tag = "driver" if a.is_driver else "amplifier"
        inf = " *(inferred)*" if a.inferred else ""
        lines.append(f"- **{a.axis_code} {a.axis_name}** — {a.severity.value} ({tag}){inf}")
    lines.append("")
    lines.append(f"**Severity:** NSS {analysis.nss}/100 -> Level {analysis.severity_level} ({analysis.severity_name})")
    lines.append("")
    lines.append("## Root-cause chain")
    lines.append(" → ".join(analysis.root_cause_chain) or "_n/a_")
    lines.append("")
    lines.append(f"**9-Steps position:** {analysis.nine_step_position}  ·  "
                 f"**Prakati gap:** {analysis.prakati_gap}")
    lines.append("")
    lines.append("## Module plan (sequenced)")
    for m in modules:
        flag = "" if m.resolved else "  ⚠️ UNRESOLVED — confirm in catalog"
        lines.append(f"- [{m.phase}] **{m.module_code}** → {', '.join(m.target_axes)} — {m.dose or ''}{flag}")
    lines.append("")
    lines.append("> For clinician review and approval — not a substitute for medical judgment.")
    return "\n".join(lines)


async def _rag_context(analysis: AnalysisResult) -> str:
    """Concurrent retrieved grounding for the driver axes (empty unless vectors available)."""
    if not retrieval.available():
        return ""
    drivers = [a for a in analysis.axis_scores if a.is_driver]
    results = await asyncio.gather(
        *[retrieval.aretrieve(a.axis_name, kind="axis", k=1) for a in drivers])
    seen, ctx = set(), []
    for hits in results:
        for hit in hits:
            key = hit.get("ref_code")
            if key and key not in seen:
                seen.add(key)
                ctx.append(f"- {hit['ref_code']}: {hit['content']}")
    return ("\n\nRETRIEVED CONTEXT (grounding):\n" + "\n".join(ctx)) if ctx else ""


async def compose(analysis: AnalysisResult, modules: list[ModulePick], llm: LLMProvider) -> CaseReport:
    markdown = _deterministic_markdown(analysis, modules)

    if llm.enabled:
        context = await _rag_context(analysis)
        prompt = (
            "Write a de-identified TSPI Case Report. Use ONLY the structured facts below; "
            "do not invent modules or labs. Sections in order: "
            + "; ".join(_SECTION_ORDER)
            + ".\n\nSTRUCTURED ANALYSIS:\n"
            + analysis.model_dump_json(indent=2)
            + "\n\nMODULE PLAN:\n"
            + "\n".join(m.model_dump_json() for m in modules)
            + context
        )
        # P4 guard: the LLM must never see PII. Fail closed if any identifier is in the prompt.
        from app import deid
        deid.assert_prompt_deidentified(prompt, analysis.case_id)
        try:
            markdown = await llm.complete(prompt)
        except Exception:  # noqa: BLE001 — fall back to deterministic report
            _log.warning("LLM report composition failed for case %s; using deterministic report",
                         analysis.case_id, exc_info=True)

    return CaseReport(case_id=analysis.case_id, analysis=analysis, modules=modules,
                      report_markdown=markdown)
