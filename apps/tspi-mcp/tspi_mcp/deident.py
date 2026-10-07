"""De-identification gate.

Governance invariant (all TSPI expert rulings): **PII must never reach the engine or the LLM.**
The MCP tool schemas deliberately have NO name / DOB / MRN / contact fields, which steers the
model to collect only de-identified data. This module is the second line of defence: it scans
free-text fields for identifiers and either redacts them (lenient) or rejects the call (strict).

It is intentionally conservative and self-contained (no engine import) so it can be unit-tested
and can never itself leak data.
"""
from __future__ import annotations

import re

# Fields the engine legitimately accepts. Anything else is dropped before forwarding.
ALLOWED_FIELDS = {
    "case_id", "age_band", "sex", "symptoms", "structured_symptoms", "labs", "imaging",
    "medications", "conditions", "lifestyle", "omics", "consent",
}

# Field names that indicate a caller is trying to pass identity — always rejected.
FORBIDDEN_FIELDS = {
    "name", "patient_name", "full_name", "first_name", "last_name", "surname",
    "dob", "date_of_birth", "birthdate", "mrn", "hn", "national_id", "ssn",
    "passport", "phone", "mobile", "tel", "email", "address", "photo",
}

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\d.])(?:\+?\d[\d\s().-]{7,}\d)(?![\d.])")
# long digit runs (MRN / national id); not preceded/followed by '.' so float fractions don't match
_LONG_ID = re.compile(r"(?<![\d.])\d{7,}(?![\d.])")
# a standalone decimal number (one dot between digit runs), e.g. 0.0833333 or 3.9 — never a phone
_DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")
_NAME_HINT = re.compile(r"\b(?:name|patient|mr|mrs|ms|dr)[.:]\s+[A-Z][a-z]+", re.IGNORECASE)


def _looks_like_phone(m: str) -> bool:
    """Lab values, ranges ("3.9 - 6.1") and long floats match the raw phone pattern but are not
    phones. Mirrors the engine's app/deid.py so both layers agree."""
    digits = sum(c.isdigit() for c in m)
    if not 9 <= digits <= 15:            # E.164 max 15; real local numbers (IN/TH) have >= 9
        return False
    return not _DECIMAL.search(m)        # contains a decimal number -> measurement, not a phone


# No date pattern: lab/collection dates are legitimate clinical data (removed to match the engine).
# Explicit DOB fields are still rejected via FORBIDDEN_FIELDS.
_REDACTIONS = (
    ("email", _EMAIL, None),
    ("phone", _PHONE, _looks_like_phone),
    ("identifier", _LONG_ID, None),
    ("name", _NAME_HINT, None),
)


class PIIError(ValueError):
    """Raised in strict mode when identifiable information is detected."""


def _scrub_text(text: str) -> tuple[str, list[str]]:
    findings: list[str] = []
    out = text
    for label, rx, accept in _REDACTIONS:
        hit = False

        def _repl(mo, label=label, accept=accept):
            nonlocal hit
            if accept and not accept(mo.group(0)):
                return mo.group(0)
            hit = True
            return f"[REDACTED_{label.upper()}]"

        out = rx.sub(_repl, out)
        if hit:
            findings.append(label)
    return out, findings


def _walk(value):
    """Recursively scrub strings inside dict/list structures. Returns (clean, findings)."""
    findings: list[str] = []
    if isinstance(value, str):
        clean, f = _scrub_text(value)
        return clean, f
    if isinstance(value, dict):
        clean_d = {}
        for k, v in value.items():
            cv, f = _walk(v)
            clean_d[k] = cv
            findings += f
        return clean_d, findings
    if isinstance(value, list):
        clean_l = []
        for v in value:
            cv, f = _walk(v)
            clean_l.append(cv)
            findings += f
        return clean_l, findings
    return value, findings


def deidentify(payload: dict, *, strict: bool) -> tuple[dict, list[str]]:
    """Return a de-identified copy of `payload` plus a list of what was removed.

    - Drops any field not in ALLOWED_FIELDS.
    - Rejects any FORBIDDEN_FIELDS outright.
    - Scrubs identifiers from all remaining free-text.
    In strict mode, ANY finding raises PIIError (fail-closed).
    """
    forbidden_present = sorted(set(payload) & FORBIDDEN_FIELDS)
    if forbidden_present:
        raise PIIError(
            "Identity fields are not allowed in TSPI input: "
            f"{', '.join(forbidden_present)}. Submit de-identified data only "
            "(case_id + age_band + sex + clinical findings)."
        )

    kept = {k: v for k, v in payload.items() if k in ALLOWED_FIELDS}
    clean, findings = _walk(kept)
    findings = sorted(set(findings))

    if findings and strict:
        raise PIIError(
            "Possible patient identifiers detected in the input "
            f"({', '.join(findings)}). Remove them and resend — TSPI must never receive PII. "
            "Use a case code instead of any name/number that could identify the patient."
        )
    return clean, findings
