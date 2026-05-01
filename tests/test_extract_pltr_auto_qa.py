"""PLTR transcripts skip the Q&A operator marker; parser auto-enters Q&A mode."""

from pathlib import Path

from bpclassifier.extract import parse_transcript

PROJECT_ROOT = Path(__file__).parent.parent


def test_pltr_has_qa_pairs():
    """PLTR transcripts should still parse Q&A pairs despite the missing marker."""
    pltr_files = sorted((PROJECT_ROOT / "ECT").glob("PLTR_*.txt"))
    assert pltr_files, "No PLTR transcripts found in ECT/"

    pairs_per_file = {p.name: len(parse_transcript(p).qa) for p in pltr_files}
    files_with_qa = [name for name, n in pairs_per_file.items() if n > 0]
    assert files_with_qa, (
        f"No PLTR transcript had Q&A pairs — auto-enter likely broken. " f"Counts: {pairs_per_file}"
    )
