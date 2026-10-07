"""Local-only extraction: uploaded report file -> candidate labs + imaging narrative.

Governance: runs entirely on-prem. Images and scanned PDFs are read by a LOCAL Ollama
VISION model (settings.vision_model); text PDFs are structured by a LOCAL Ollama TEXT model.
Nothing is sent to any cloud LLM. Every result is confirmed=False -> a human must review it
before it feeds diagnosis. If the models are unreachable, a deterministic regex parser runs so
the endpoint never hard-fails.
"""
from __future__ import annotations

import base64
import json
import re

from app.config import settings
from app.llm.provider import OllamaLocal
from app.schemas import ExtractedLab, ExtractionResult

import logging

_log = logging.getLogger(__name__)

_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")

_PROMPT = (
    "You are a careful medical data extractor. Read the attached patient report EXACTLY as "
    "printed; do not invent, infer, or fill in values that are not shown. Return ONLY a JSON "
    "object with this shape:\n"
    '{"labs":[{"analyte":str,"value":number|string,"unit":str|null,'
    '"ref_low":number|null,"ref_high":number|null,"flag":"H"|"L"|"normal"|"abnormal"|null}],'
    '"imaging":{"modality":str|null,"impression":str|null,"findings":[str]},'
    '"narrative":str|null}\n'
    "Rules: numbers as numbers when possible, else the printed string; use null when a field is "
    "absent. Include a lab row only if an analyte name AND a value are both printed. For an "
    "ultrasound/sonography/X-ray/CT/MRI report, fill imaging.modality, imaging.impression "
    "(the conclusion) and imaging.findings (each measurement/observation as one string); labs "
    "may be empty. Put any other useful free text in narrative."
)


# ---- helpers -------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _is_image(mime: str, name: str) -> bool:
    return name.endswith(_IMG_EXT) or "image" in (mime or "")


def _is_pdf(mime: str, name: str) -> bool:
    return name.endswith(".pdf") or "pdf" in (mime or "")


def _pdf_text(data: bytes) -> str:
    try:
        import io
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    except Exception:
        _log.warning("PDF text extraction failed", exc_info=True)
        return ""


def _pdf_page_images(data: bytes, max_pages: int) -> list[str]:
    """Rasterize PDF pages to PNG base64 (for scanned PDFs). Uses PyMuPDF if available."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    out: list[str] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # ~144 dpi
            out.append(_b64(pix.tobytes("png")))
    except Exception:
        _log.warning("PDF page rasterization failed after %d page(s)", len(out), exc_info=True)
        return out
    return out


def _loads(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


def _to_labs(rows) -> list[ExtractedLab]:
    labs: list[ExtractedLab] = []
    for x in rows or []:
        if not isinstance(x, dict):
            continue
        analyte = str(x.get("analyte", "")).strip()
        value = x.get("value")
        if not analyte or value in (None, ""):
            continue
        labs.append(ExtractedLab(
            analyte=analyte,
            value=value,
            unit=(x.get("unit") or None),
            ref_low=_num(x.get("ref_low")),
            ref_high=_num(x.get("ref_high")),
            flag=(x.get("flag") or None),
        ))
    return labs


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_imaging(res: ExtractionResult, obj: dict) -> None:
    img = obj.get("imaging") or {}
    if isinstance(img, dict):
        res.imaging_modality = res.imaging_modality or (img.get("modality") or None)
        res.imaging_impression = res.imaging_impression or (img.get("impression") or None)
        f = img.get("findings") or []
        if isinstance(f, list):
            res.imaging_findings.extend(str(s) for s in f if str(s).strip())
    if obj.get("narrative"):
        res.narrative = ((res.narrative + "\n") if res.narrative else "") + str(obj["narrative"])


# ---- deterministic fallback (no model) -----------------------------------
_LINE = re.compile(
    r"^\s*(?P<analyte>[A-Za-z][A-Za-z0-9 \-/().]{1,40}?)\s*[:\-]?\s+"
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|[A-Za-zµ/%\^0-9.\-]{1,12})?\s*"
    r"(?:\(?\s*(?:ref\.?\s*)?(?:(?P<op><|>)\s*(?P<bound>\d+(?:\.\d+)?)|"
    r"(?P<low>\d+(?:\.\d+)?)\s*[-–]\s*(?P<high>\d+(?:\.\d+)?))\s*\)?)?\s*$",
    re.IGNORECASE,
)
_STOP = {"name", "date", "patient", "report", "page", "address", "phone", "dob", "age", "sex"}


def _heuristic(text: str) -> list[ExtractedLab]:
    labs: list[ExtractedLab] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) > 80:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        analyte = m.group("analyte").strip(" :-").strip()
        if not analyte or analyte.lower() in _STOP or len(analyte) < 2:
            continue
        lo = hi = None
        if m.group("op") == "<":
            hi = float(m.group("bound"))
        elif m.group("op") == ">":
            lo = float(m.group("bound"))
        elif m.group("low"):
            lo, hi = float(m.group("low")), float(m.group("high"))
        labs.append(ExtractedLab(analyte=analyte, value=m.group("value"),
                                 unit=(m.group("unit") or None), ref_low=lo, ref_high=hi))
    return labs


# ---- main entrypoint -----------------------------------------------------
async def extract_from_file(data: bytes, mime: str, filename: str,
                            doc_type: str = "lab") -> ExtractionResult:
    name = (filename or "").lower()
    use_models = settings.extraction_backend == "ollama"
    ol = OllamaLocal()

    # 1) IMAGE -> vision model
    if _is_image(mime, name):
        res = ExtractionResult(source="image", engine=f"ollama-vision:{settings.vision_model}")
        if not use_models:
            res.engine = "heuristic"
            res.notes = "extraction_backend=heuristic: images cannot be read without a vision model."
            return res
        try:
            obj = _loads(await ol.vision_json(_PROMPT, [_b64(data)]))
            res.labs = _to_labs(obj.get("labs"))
            _apply_imaging(res, obj)
        except Exception as e:  # noqa: BLE001
            _log.warning("Vision extraction failed", exc_info=True)
            res.notes = f"vision-unavailable:{type(e).__name__}"
        return res

    # 2) PDF -> text layer if present, else rasterize -> vision
    if _is_pdf(mime, name):
        text = _pdf_text(data)
        if len(text.strip()) >= settings.pdf_text_min_chars:
            res = ExtractionResult(source="pdf-text",
                                   engine=f"ollama-text:{settings.extraction_text_model or settings.ollama_model}")
            if use_models:
                try:
                    obj = _loads(await ol.text_json(_PROMPT + "\n\nREPORT TEXT:\n" + text[:8000]))
                    res.labs = _to_labs(obj.get("labs"))
                    _apply_imaging(res, obj)
                except Exception as e:  # noqa: BLE001
                    _log.warning("Text-LLM extraction failed (PDF); using heuristic", exc_info=True)
                    res.notes = f"text-llm-unavailable:{type(e).__name__}; used heuristic"
            if not res.labs and not res.imaging_impression:
                res.labs = _heuristic(text)
                if res.engine.startswith("ollama") and res.labs:
                    res.engine = "heuristic"
            return res
        # scanned PDF -> images -> vision (all pages in one call)
        res = ExtractionResult(source="pdf-scan", engine=f"ollama-vision:{settings.vision_model}")
        imgs = _pdf_page_images(data, settings.extraction_max_pages)
        if not imgs:
            res.notes = "pdf-scan: PyMuPDF not installed; cannot rasterize pages for vision."
            return res
        if not use_models:
            res.engine = "heuristic"
            res.notes = "extraction_backend=heuristic: scanned PDF needs a vision model."
            return res
        try:
            obj = _loads(await ol.vision_json(_PROMPT + f"\n\n(These {len(imgs)} images are pages "
                                              "of ONE report.)", imgs))
            res.labs = _to_labs(obj.get("labs"))
            _apply_imaging(res, obj)
        except Exception as e:  # noqa: BLE001
            _log.warning("Vision extraction failed", exc_info=True)
            res.notes = f"vision-unavailable:{type(e).__name__}"
        return res

    # 3) text / csv / unknown -> decode -> text model or heuristic
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    res = ExtractionResult(source="text",
                           engine=f"ollama-text:{settings.extraction_text_model or settings.ollama_model}")
    if use_models and text.strip():
        try:
            obj = _loads(await ol.text_json(_PROMPT + "\n\nREPORT TEXT:\n" + text[:8000]))
            res.labs = _to_labs(obj.get("labs"))
            _apply_imaging(res, obj)
        except Exception as e:  # noqa: BLE001
            _log.warning("Text-LLM extraction failed; using heuristic", exc_info=True)
            res.notes = f"text-llm-unavailable:{type(e).__name__}; used heuristic"
    if not res.labs and not res.imaging_impression:
        res.labs = _heuristic(text)
        res.engine = "heuristic"
    return res
