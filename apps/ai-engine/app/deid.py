"""P4 — de-identification / re-identification boundary (engine side).

Centralises ALL PII handling in one place so clients never re-implement it. The rule kept
absolutely: **the brain and the LLM only ever see de-identified data + placeholders.** PII enters
here, is stored **encrypted** keyed by the de-identified `case_id`, and is re-attached ONLY at report
render time — never in the LLM prompt, never in a chat transcript.
"""
from __future__ import annotations

import json
import re

from app import store
from app.config import settings

# Placeholder tokens the report template uses; re-id substitutes these at render.
_FIELDS = ["full_name", "dob", "mrn", "phone", "email", "address"]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\d.])(?:\+?\d[\d\s().-]{7,}\d)(?![\d.])")
# Long bare digit runs (MRN etc.). Not preceded/followed by '.' so float fractions don't match.
_LONGID = re.compile(r"(?<![\d.])\d{7,}(?![\d.])")
# A standalone decimal number (one dot between digit runs), e.g. 0.0833333 or 3.9 — never a phone.
_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


def _looks_like_phone(m: str) -> bool:
    """Filter regex hits from structured/clinical text: lab values, ranges ("3.9 - 6.1") and long
    floats (0.08333333333) look like phone numbers to the raw pattern but are not."""
    digits = sum(c.isdigit() for c in m)
    if not 9 <= digits <= 15:            # E.164 max 15; real local numbers (IN/TH) have >= 9
        return False
    if _DECIMAL.search(m):               # contains a decimal number -> measurement, not a phone
        return False
    return True


def _mask(m: str) -> str:
    """Shape of a hit with letters/digits hidden (safe to log): '+66 81 234 5678' -> '+dd dd ddd dddd'."""
    return re.sub(r"\d", "d", re.sub(r"[A-Za-z]", "x", m))[:40]


# No date pattern: lab/collection dates are legitimate clinical data. The stored DOB is still
# blocked by the exact-value identity check in assert_prompt_deidentified().
_RULES = (
    ("email", _EMAIL, None),
    ("phone", _PHONE, _looks_like_phone),
    ("identifier", _LONGID, None),
)


# ---- case-code exemption -------------------------------------------------------------------
# A case code like "TSPI-261008-0151-01" contains 12 digits joined by dashes, which the raw phone
# pattern reads as a phone number. Case codes are the de-identified key by design, so they are
# shielded from the scan: (a) the request's own case_id, when it is shaped like a code (letter
# prefix + dash-joined alphanumerics), and (b) any TSPI-/CASE- prefixed code in free text.
# A bare-digit case_id (e.g. "9876543210") is NOT shielded and is still scanned.
_CASE_CODE = re.compile(r"\b(?:TSPI|CASE)(?:-[A-Za-z0-9]+)+\b", re.IGNORECASE)
_CODE_SHAPE = re.compile(r"[A-Za-z]{2,12}(?:[-_][A-Za-z0-9]{1,12}){1,6}")
_CONTACT_PREFIX = {"tel", "ph", "phone", "mob", "mobile", "cell", "call", "whatsapp", "wa",
                   "contact", "fax", "mrn", "hn", "id", "dob"}
_SHIELD = "\uE000"   # private-use char: no digits/letters, so no rule can match it


def is_case_code(value) -> bool:
    """True if `value` is a letter-prefixed case code (safe to exempt from identifier scans)."""
    if not isinstance(value, str) or len(value) > 60 or not _CODE_SHAPE.fullmatch(value.strip()):
        return False
    prefix = re.split(r"[-_]", value.strip(), 1)[0].lower()
    return prefix not in _CONTACT_PREFIX


def _shield_codes(text: str, case_id: str | None = None) -> tuple[str, list[str]]:
    """Swap case codes for a placeholder char; returns (text, codes) for _unshield."""
    codes: list[str] = []

    def _keep(mo):
        codes.append(mo.group(0))
        return _SHIELD

    if case_id and is_case_code(case_id):
        text = re.sub(re.escape(case_id.strip()), _keep, text)
    return _CASE_CODE.sub(_keep, text), codes


def _unshield(text: str, codes: list[str]) -> str:
    it = iter(codes)
    return re.sub(_SHIELD, lambda _m: next(it), text)


class PIILeak(RuntimeError):
    """Raised if identifiable information is about to reach the LLM."""


def _fernet():
    key = settings.encryption_key
    if not key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(key.encode() if isinstance(key, str) else key)


# ---------------------------------------------------------------- store / load (encrypted)
def store_identity(case_id: str, identity: dict) -> None:
    payload = json.dumps({k: identity.get(k) for k in _FIELDS if identity.get(k)}, ensure_ascii=False)
    f = _fernet()
    if f:
        store.save_identity(case_id, f.encrypt(payload.encode()).decode(), encrypted=True)
    else:
        store.save_identity(case_id, payload, encrypted=False)   # dev fallback (flagged; not for prod)


def get_identity(case_id: str) -> dict | None:
    row = store.get_identity(case_id)
    if not row:
        return None
    blob, enc = row["blob"], row["encrypted"]
    if enc:
        f = _fernet()
        if not f:
            return None                          # encrypted at rest but no key here -> cannot read
        blob = f.decrypt(blob.encode()).decode()
    try:
        return json.loads(blob)
    except ValueError:
        return None


# ---------------------------------------------------------------- scrub / guard
def _scan(text: str, case_id: str | None = None) -> tuple[str, list[str], list[str]]:
    """-> (redacted text, finding labels, masked shapes of each hit for diagnostics)."""
    findings, shapes = [], []
    out, codes = _shield_codes(text or "", case_id)
    for label, rx, accept in _RULES:
        hit = False

        def _repl(mo, label=label, accept=accept):
            nonlocal hit
            if accept and not accept(mo.group(0)):
                return mo.group(0)
            hit = True
            shapes.append(f"{label}:{_mask(mo.group(0))}")
            return f"[REDACTED_{label.upper()}]"

        out = rx.sub(_repl, out)
        if hit:
            findings.append(label)
    return _unshield(out, codes), findings, shapes


def scrub_text(text: str, case_id: str | None = None) -> tuple[str, list[str]]:
    out, findings, _ = _scan(text, case_id)
    return out, findings


def _digits(v) -> str:
    return re.sub(r"\D", "", str(v))


def assert_prompt_deidentified(prompt: str, case_id: str) -> None:
    """Guard: the stored identity's values must NOT appear in an LLM prompt. Fail closed."""
    ident = get_identity(case_id) or {}
    low = prompt.lower()
    compact = None
    for k, v in ident.items():
        if v and str(v).strip() and str(v).lower() in low:
            raise PIILeak(f"PII ({k}) detected in LLM prompt for case {case_id}. Blocked.")
        # phone/MRN may be re-formatted (spaces, dashes, parens) — also compare with those removed
        if k in ("phone", "mrn") and len(_digits(v)) >= 7:
            compact = compact if compact is not None else re.sub(r"[\s()-]", "", prompt)
            if re.search(rf"(?<![\d.]){_digits(v)}(?![\d.])", compact):
                raise PIILeak(f"PII ({k}) detected in LLM prompt for case {case_id}. Blocked.")
    # also catch raw identifiers that slipped into free text
    _, findings, shapes = _scan(prompt, case_id)
    if findings:
        # shapes have digits masked (e.g. 'phone:dd ddd dddd') — safe to log, enough to locate the source
        raise PIILeak(f"Identifier(s) {findings} detected in LLM prompt for case {case_id}. "
                      f"Blocked. Matched shapes: {shapes[:5]}")


# ---------------------------------------------------------------- re-identify (render only)
def placeholders(identity: dict) -> dict[str, str]:
    return {f"{{{{patient.{k}}}}}": str(identity.get(k) or "") for k in _FIELDS}


def reidentify(text: str, case_id: str) -> str:
    ident = get_identity(case_id)
    if not ident or not text:
        return text
    for token, value in placeholders(ident).items():
        text = text.replace(token, value)
    return text


# ---------------------------------------------------------------- intake (strip before pipeline)
def intake(patient):
    """If the caller sent PII, store it encrypted + strip it so only de-identified data proceeds.
    Also scrubs free-text symptoms. Returns the de-identified patient object."""
    ident = getattr(patient, "patient_identity", None)
    if ident is not None:
        store_identity(patient.case_id, ident.model_dump() if hasattr(ident, "model_dump") else dict(ident))
        patient.patient_identity = None
        store.audit("pii_intake", case_id=patient.case_id, detail={"stored": True})
    # scrub any identifiers that leaked into free-text symptoms
    if getattr(patient, "symptoms", None):
        clean, findings = scrub_text(patient.symptoms, patient.case_id)
        if findings:
            patient.symptoms = clean
            store.audit("pii_scrub", case_id=patient.case_id, detail={"fields": findings})
    return patient
