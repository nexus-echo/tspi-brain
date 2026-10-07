"""Regression: case codes like TSPI-YYMMDD-HHMM-NN must not be flagged as phone numbers."""
import pytest

from app import deid


def test_case_code_not_flagged_in_scan():
    _, findings = deid.scrub_text("Case TSPI-261008-0151-01: insomnia, TSH 8.81", "TSPI-261008-0151-01")
    assert findings == []


def test_prompt_guard_allows_case_code(monkeypatch):
    monkeypatch.setattr(deid, "get_identity", lambda _cid: None)
    deid.assert_prompt_deidentified("Report for TSPI-261008-0151-01. HbA1c 6.0 (4.0 - 5.6)",
                                    "TSPI-261008-0151-01")


def test_prompt_guard_still_blocks_phone(monkeypatch):
    monkeypatch.setattr(deid, "get_identity", lambda _cid: None)
    with pytest.raises(deid.PIILeak):
        deid.assert_prompt_deidentified("TSPI-261008-0151-01 call +66 81 234 5678",
                                        "TSPI-261008-0151-01")
