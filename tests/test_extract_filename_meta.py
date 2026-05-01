"""Filename metadata: handles quarterly + annual-call naming conventions."""

from pathlib import Path

from bpclassifier.extract import _filename_meta, _parse_quarter_token


def test_quarterly_filename():
    ticker, quarter = _filename_meta(Path("AMD_Q4-2025.txt"))
    assert ticker == "AMD"
    assert quarter == "Q4-2025"


def test_annual_call_filename():
    ticker, quarter = _filename_meta(Path("FAST_2026_01_20.txt"))
    assert ticker == "FAST"
    assert quarter == "2026_01_20"


def test_quarter_token_quarterly():
    qn, yr = _parse_quarter_token("Q4-2025")
    assert qn == 4
    assert yr == 2025


def test_quarter_token_annual():
    qn, yr = _parse_quarter_token("2026_01_20")
    assert qn is None
    assert yr == 2026


def test_quarter_token_unparseable():
    qn, yr = _parse_quarter_token("garbage")
    assert qn is None
    assert yr is None
