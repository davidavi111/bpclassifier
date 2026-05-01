"""Orphan Q&A pairs (no preceding question) must not be malformed."""

from pathlib import Path

from bpclassifier.extract import parse_all

PROJECT_ROOT = Path(__file__).parent.parent


def test_no_qa_pair_is_fully_empty():
    """Each parsed QAPair has at least a question or an answer."""
    transcripts = parse_all(PROJECT_ROOT / "ECT")
    for t in transcripts:
        for pair in t.qa:
            assert (
                pair.question or pair.answer
            ), f"Empty QAPair (no question and no answer) in {t.raw_path}"
