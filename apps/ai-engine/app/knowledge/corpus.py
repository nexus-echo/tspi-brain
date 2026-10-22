"""TSPI knowledge corpus — ONE list of retrievable documents shared by the Q&A API, the lexical
index and the embedding loader (scripts/embed_knowledge.py).

Sources (all de-identified reference knowledge — never patient data):
  * Structured masters in data/: 3 Keys, 9 Restoration Steps, 12 Domains, 39 Axes (+ sub-axes and
    mapped modules), Module Registry, 180 Living Biological Networks, red-flag rules, safety rules,
    severity dosing protocol, framework principles.
  * Optional curated markdown/text under KB_DOCS_DIR (product monographs, axis monographs, official
    domain-expert answers). Only the allow-listed globs in KB_DOCS_GLOBS are read, so folders such as
    patient samples or stakeholder chats can never be indexed by accident.

Each document: {id, kind, ref_code, title, content}. `id` is stable and <=64 chars so it can be
stored as kb_embeddings.ref_code and joined back after a vector search.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

_log = logging.getLogger(__name__)

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
_CHUNK_CHARS = 1600

# kind -> human label (also the allowed values for the `kinds` filter on /ask)
KINDS = {
    "framework": "TSPI framework",
    "key": "Fundamental Key",
    "step": "Restoration Step",
    "domain": "Domain",
    "axis": "Biological Axis",
    "module": "Module (registry)",
    "product": "Product monograph",
    "network": "Living Biological Network",
    "safety": "Safety / red-flag rule",
    "dosing": "Dosing protocol",
    "faq": "Official domain-expert answer",
    "doc": "Reference document",
}


def _load(name: str):
    try:
        return json.loads((_DATA / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        _log.info("KB corpus: %s not found; skipped", name)
        return None


def _doc(kind: str, ref: str, title: str, content: str, did: str | None = None) -> dict:
    did = (did or f"{kind}:{ref}")[:64]
    return {"id": did, "kind": kind, "ref_code": str(ref), "title": title.strip(),
            "content": re.sub(r"\n{3,}", "\n\n", content).strip()}


def _lines(*parts) -> str:
    return "\n".join(p for p in parts if p)


def _json_text(obj, indent: int = 0) -> str:
    """Compact human-readable rendering of a small JSON reference file."""
    pad = "  " * indent
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.append(f"{pad}{k}:")
                out.append(_json_text(v, indent + 1))
            else:
                out.append(f"{pad}{k}: {v}")
        return "\n".join(out)
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            return f"{pad}{', '.join(str(x) for x in obj)}"
        return "\n".join(f"{pad}-\n{_json_text(x, indent + 1)}" for x in obj)
    return f"{pad}{obj}"


# ----------------------------------------------------------------------------- structured masters
def _framework_docs() -> list[dict]:
    docs: list[dict] = []
    fw = _load("tspi_framework.json") or {}
    if fw:
        docs.append(_doc("framework", "overview", "TSPI framework overview", _lines(
            f"Architecture: {fw.get('architecture', '')}",
            f"Core principle: {fw.get('principle', '')}",
            "Master authority order: " + " > ".join(fw.get("master_authority_order", [])),
            "Decisions:\n" + _json_text(fw.get("decisions", {})) if fw.get("decisions") else "")))
        for k in fw.get("keys_3", []):
            docs.append(_doc("key", f"KEY-{k['code']}", f"Key {k['code']} — {k['name']}", _lines(
                f"Fundamental Key {k['code']} ({k['name']}).",
                f"Tridosha correspondence: {k.get('tridosha')}" if k.get("tridosha") else "",
                f"Focus: {k.get('focus')}" if k.get("focus") else "")))
    dk = _load("domain_key_master.json") or {}
    for k in dk.get("keys", []):
        docs.append(_doc("key", f"{k['key_code']}-meaning", f"{k['key_code']} — {k['official_name']}",
                         f"{k['key_code']} {k['official_name']}: {k.get('meaning', '')}"))
    steps = (_load("restoration_steps.json") or {}).get("steps", [])
    if steps:
        docs.append(_doc("step", "S1-S9", "The 9 Restoration Steps", "\n".join(
            f"{s['code']} (step {s['order']}): {s['name_en']}" for s in steps)))
    return docs


def _axis_and_domain_docs() -> list[dict]:
    axes_doc = _load("tspi_axes_39.json") or {}
    axes = axes_doc.get("axes", []) if isinstance(axes_doc, dict) else axes_doc
    key_names = {k["key_code"]: k["official_name"] for k in (axes_doc.get("keys", []) if isinstance(axes_doc, dict) else [])}
    amap = (_load("axis_module_official.json") or {}).get("map", {})
    nets = (_load("tspi_networks_180.json") or {}).get("networks", [])
    nets_by_axis: dict[str, list[str]] = {}
    for n in nets:
        nets_by_axis.setdefault(n.get("primary_axis_code"), []).append(
            f"{n['network_code']} {n.get('network_name_en', '')}")

    docs: list[dict] = []
    axes_by_domain: dict[str, list[str]] = {}
    for a in axes:
        code = a["axis_id"]
        axes_by_domain.setdefault(a.get("domain", ""), []).append(f"{code} {a['name']}")
        subs = "\n".join(f"  {s['code']}: {s.get('name', '')}" for s in a.get("sub_axes", []))
        mods = amap.get(code, [])
        prim = ", ".join(f"{m['code']} {m.get('name', '')}" for m in mods
                         if m.get("role", "primary") == "primary" and m.get("status", "active") == "active")
        sec = ", ".join(f"{m['code']} {m.get('name', '')}" for m in mods
                        if m.get("role") == "secondary" and m.get("status", "active") == "active")
        sec_keys = ", ".join(f"{k} {key_names.get(k, '')}".strip() for k in a.get("key_secondary", []))
        docs.append(_doc("axis", code, f"Axis {code} — {a['name']}", _lines(
            f"Biological axis {code} (axis {a.get('axis_num', '')}): {a['name']}.",
            f"Domain {a.get('domain_num', '')}: {a.get('domain', '')}.",
            f"Primary key: {a.get('key_primary', '')} {key_names.get(a.get('key_primary'), '')}".rstrip()
            + (f"; secondary keys: {sec_keys}" if sec_keys else "") + ".",
            f"Sub-axes:\n{subs}" if subs else "",
            f"Primary modules mapped to {code}: {prim}." if prim else "",
            f"Secondary modules mapped to {code}: {sec}." if sec else "",
            (f"Living biological networks on {code}: " + "; ".join(nets_by_axis[code][:20]) + ".")
            if nets_by_axis.get(code) else "")))

    dk = _load("domain_key_master.json") or {}
    for d in dk.get("domains", []):
        name = d.get("domain_name", "")
        docs.append(_doc("domain", d["domain_code"], f"Domain {d['domain_code']} — {name}", _lines(
            f"Domain {d['domain_code']}: {name}.",
            f"Primary key: {d.get('primary_key_code')} {key_names.get(d.get('primary_key_code'), '')}".rstrip() + ".",
            "Secondary keys: " + ", ".join(d.get("secondary_key_codes", [])) + "." if d.get("secondary_key_codes") else "",
            "Axes in this domain: " + "; ".join(axes_by_domain.get(name, [])) + "." if axes_by_domain.get(name) else "")))
    return docs


def _module_docs() -> list[dict]:
    reg = _load("module_registry.json") or {}
    axes_doc = _load("tspi_axes_39.json") or {}
    axis_names = {a["axis_id"]: a["name"] for a in (axes_doc.get("axes", []) if isinstance(axes_doc, dict) else [])}

    def ax(codes):
        return ", ".join(f"{c} {axis_names.get(c, '')}".strip() for c in codes)

    docs = []
    for m in reg.get("modules", []):
        code = m["module_code"]
        name = m.get("name_en") or code
        ci = m.get("contraindications") or []
        docs.append(_doc("module", code, f"Module {code} — {name}", _lines(
            f"TSPI module {code}: {name}" + (f" ({m['name_th']})" if m.get("name_th") else "") + ".",
            f"PhytoCore code: {m['phytocore_code']}." if str(m.get("phytocore_code") or "None") not in ("None", "nan") else "",
            f"Status: {m.get('status', 'active')}. Dose type: {m.get('dose_type', 'severity')}"
            + (" (KS bowel-performance protocol)." if m.get("dose_type") == "bowel" else " (dose follows Network Severity level)."),
            f"Primary target axes: {ax(m.get('primary_axes', []))}." if m.get("primary_axes") else "",
            f"Secondary target axes: {ax(m.get('secondary_axes', []))}." if m.get("secondary_axes") else "",
            "Contraindications / cautions: " + " ".join(ci) if ci else "Contraindications: none recorded in the registry.",
            f"Registry version: {reg.get('registry_version', '')}.")))
    return docs


def _network_docs() -> list[dict]:
    docs = []
    for n in (_load("tspi_networks_180.json") or {}).get("networks", []):
        sec = ", ".join(n.get("secondary_axis_codes", []) or [])
        docs.append(_doc("network", n["network_code"], f"Network {n['network_code']} — {n.get('network_name_en', '')}", _lines(
            f"Living biological network {n['network_code']} ({n.get('legacy_display_id', '')}): {n.get('network_name_en', '')}.",
            f"Biological axis: {n.get('primary_axis_code')} {n.get('biological_axis_label', '')}" + (f"; secondary axes {sec}" if sec else "") + ".",
            f"Domain: {n.get('primary_domain_code')} ({n.get('domain_group_label_from_file', '')}).",
            f"Scope: {n.get('scope')}" if n.get("scope") else "",
            f"Mapping status: {n.get('mapping_status', '')}; code status: {n.get('network_code_status', '')}.")))
    return docs


def _safety_docs() -> list[dict]:
    docs = []
    rf = _load("red_flag_rules.json") or {}
    classes = rf.get("action_classes", {})
    if classes:
        docs.append(_doc("safety", "red-flag-classes", "Red-flag action classes",
                         "Red-flag screening runs before any TSPI reasoning and can only escalate care.\n"
                         + _json_text(classes)))
    for group in ("symptom_rules", "condition_rules", "critical_value_rules"):
        for r in rf.get(group, []):
            cls = r.get("action_class", "")
            extra = ""
            if group == "critical_value_rules":
                extra = (f"Analyte {r.get('analyte')}: critical low {r.get('critical_low')}, "
                         f"critical high {r.get('critical_high')} {', '.join(r.get('units', []))}.")
            else:
                extra = "Triggers: " + ", ".join(r.get("match_any", [])) + "."
            docs.append(_doc("safety", r["red_flag_code"], f"Red flag {r['red_flag_code']} — {r.get('name_en', '')}", _lines(
                f"Red-flag rule {r['red_flag_code']}: {r.get('name_en', '')}.", extra,
                f"Action class: {cls} — required action {classes.get(cls, {}).get('required_action', '')}.")))
    sr = _load("safety_rules.json") or {}
    for i, r in enumerate(sr.get("rules", []), 1):
        docs.append(_doc("safety", f"rule-{i}", f"Safety rule — {r.get('match', i)}", _json_text(r)))
    if sr.get("global_alerts"):
        docs.append(_doc("safety", "global-alerts", "Global safety alerts", "\n".join(sr["global_alerts"])))
    return docs


def _dosing_docs() -> list[dict]:
    d = _load("tspi_severity_dosing.json")
    if not d:
        return []
    docs = []
    for lv in d.get("levels", []):
        dose = lv.get("dose", {})
        docs.append(_doc("dosing", f"level-{lv['level']}", f"Severity level {lv['level']} — {lv.get('name', '')}", _lines(
            f"Network Severity Score (NSS) {lv.get('nss_min')}–{lv.get('nss_max')} = severity level {lv['level']} ({lv.get('name', '')}).",
            f"Standard module dose: {dose.get('caps')} caps × {dose.get('times_per_day')} times/day = "
            f"{dose.get('daily_total')} caps/day ({', '.join(dose.get('schedule', []))}).",
            f"Objective: {lv.get('objective', '')}.")))
    if d.get("ks_protocol"):
        docs.append(_doc("dosing", "ks-protocol", "KS bowel-performance dosing protocol", _json_text(d["ks_protocol"])))
    rest = {k: v for k, v in d.items() if k not in ("levels", "ks_protocol")}
    if rest:
        docs.append(_doc("dosing", "nss-scale", "NSS scale and dosing source", _json_text(rest)))
    return docs


# ----------------------------------------------------------------------------- curated text docs
def _docs_root() -> Path | None:
    if settings.kb_docs_dir:
        p = Path(settings.kb_docs_dir)
        return p if p.is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:                       # repo checkout: <repo>/knowledge-sources
        cand = parent / "knowledge-sources"
        if cand.is_dir():
            return cand
    return None


_NONE_LINE = re.compile(r"^.*:\*{0,2}\s*(None|nan|N/A)?\s*$", re.IGNORECASE)


def _clean_md(text: str) -> str:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s in ("None", "`None`", "nan"):
            continue
        if _NONE_LINE.match(s) and ("**" in s or s.endswith(":")) and not s.startswith("#"):
            continue
        out.append(line.rstrip())
    return "\n".join(out)


def _chunks(text: str, sep: str = r"\n-{3,}\n") -> list[str]:
    parts = [p.strip() for p in re.split(sep, text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        # drop sections that are only a heading after cleaning
        if not re.sub(r"^#.*$", "", p, flags=re.MULTILINE).strip():
            continue
        if cur and len(cur) + len(p) > _CHUNK_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > _CHUNK_CHARS * 1.5:          # hard-split very long sections
            chunks.append(cur[:_CHUNK_CHARS])
            cur = cur[_CHUNK_CHARS:]
    if cur:
        chunks.append(cur)
    return chunks


def _file_docs() -> list[dict]:
    root = _docs_root()
    if root is None:
        return []
    docs: list[dict] = []
    globs = [g.strip() for g in (settings.kb_docs_globs or "").split(",") if g.strip()]
    for g in globs:
        for f in sorted(root.glob(g)):
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            stem = f.stem
            heading = next((l.lstrip("# ").strip() for l in raw.splitlines() if l.startswith("#")), stem)
            heading = re.sub(r"^[^\w]+", "", heading)          # drop leading emoji
            if stem.startswith("product_"):
                kind, ref = "product", stem[len("product_"):]
                body, sep = _clean_md(raw), r"\n-{3,}\n"
            elif stem.startswith("axis_"):
                m = re.match(r"axis_(\d+)", stem)
                kind, ref = "axis", (f"A{int(m.group(1))}-monograph" if m else stem)
                body, sep = _clean_md(raw), r"\n-{3,}\n"
            elif "question_answers" in stem:
                kind, ref, heading = "faq", "expert-answers", "Official TSPI domain-expert answers"
                body, sep = raw, r"\n(?=Q\d+\.)"
            else:
                kind, ref = "doc", stem
                body, sep = raw, r"\n-{3,}\n"
            for i, ch in enumerate(_chunks(body, sep), 1):
                title = heading
                if kind == "faq":
                    q = re.match(r"(Q\d+\..*)", ch)
                    title = f"{heading} — {q.group(1)[:90]}" if q else heading
                docs.append(_doc(kind, ref, title, ch, did=f"{kind}:{ref}#{i}"[:64]))
    return docs


@lru_cache(maxsize=1)
def build_corpus() -> tuple[dict, ...]:
    """All documents (cached for the process lifetime; restart after changing data/ or docs)."""
    docs: list[dict] = []
    for fn in (_framework_docs, _axis_and_domain_docs, _module_docs, _network_docs,
               _safety_docs, _dosing_docs, _file_docs):
        try:
            docs.extend(fn())
        except Exception:  # noqa: BLE001 — one bad source must not take down Q&A
            _log.warning("KB corpus source %s failed", fn.__name__, exc_info=True)
    seen: set[str] = set()
    out = []
    for d in docs:
        if d["id"] in seen or not d["content"]:
            continue
        seen.add(d["id"])
        out.append(d)
    return tuple(out)


def by_id() -> dict[str, dict]:
    return {d["id"]: d for d in build_corpus()}


def stats() -> dict:
    counts: dict[str, int] = {}
    for d in build_corpus():
        counts[d["kind"]] = counts.get(d["kind"], 0) + 1
    return {"documents": sum(counts.values()), "by_kind": counts,
            "docs_dir": str(_docs_root()) if _docs_root() else None}
