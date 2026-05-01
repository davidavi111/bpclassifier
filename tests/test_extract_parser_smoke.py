"""Smoke test: parse all ECT transcripts without errors."""

from pathlib import Path

from bpclassifier.extract import parse_all

PROJECT_ROOT = Path(__file__).parent.parent


def test_parse_all_smoke():
    """Every .txt in ECT/ parses without raising."""
    transcripts = parse_all(PROJECT_ROOT / "ECT")
    assert len(transcripts) >= 50, f"Expected at least 50 transcripts, got {len(transcripts)}"


def test_parse_all_have_company_and_ticker():
    """Every parsed transcript has a non-empty ticker and company."""
    transcripts = parse_all(PROJECT_ROOT / "ECT")
    for t in transcripts:
        assert t.ticker, f"Empty ticker in {t.raw_path}"
        assert t.company, f"Empty company in {t.raw_path}"


def test_parse_all_have_some_content():
    """No transcript should be entirely empty (no prepared remarks AND no Q&A)."""
    transcripts = parse_all(PROJECT_ROOT / "ECT")
    empty = [t for t in transcripts if not t.prepared and not t.qa]
    assert not empty, f"Empty transcripts found: {[t.raw_path for t in empty]}"
