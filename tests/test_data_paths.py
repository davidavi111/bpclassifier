"""Smoke test: required data folders and at least one transcript exist."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_ect_directory_exists():
    ect = PROJECT_ROOT / "ECT"
    assert ect.is_dir(), f"ECT/ directory not found at {ect}"


def test_ect_contains_transcripts():
    ect = PROJECT_ROOT / "ECT"
    transcripts = list(ect.glob("*.txt"))
    assert (
        len(transcripts) >= 10
    ), f"Expected at least 10 transcripts in ECT/, found {len(transcripts)}"


def test_data_subdirectories_exist():
    for sub in ("raw", "interim", "gold", "splits"):
        path = PROJECT_ROOT / "data" / sub
        assert path.is_dir(), f"data/{sub}/ missing"
