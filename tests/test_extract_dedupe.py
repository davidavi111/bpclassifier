"""Within-transcript line deduplication."""

from bpclassifier.extract import DEDUPE_MIN_LINE_CHARS, _dedupe_lines


def test_repeated_long_lines_collapsed():
    line = "This is a long boilerplate disclaimer that should be deduped."
    assert len(line) >= DEDUPE_MIN_LINE_CHARS
    text = f"{line}\nUnique content one.\n{line}\nUnique content two.\n"
    out = _dedupe_lines(text)
    assert out.count(line) == 1


def test_short_lines_preserved():
    """Short lines are NOT deduped (e.g. 'Yes.' from multiple speakers)."""
    text = "Yes.\nNo.\nYes.\nNo.\n"
    out = _dedupe_lines(text)
    assert out.count("Yes.") == 2
    assert out.count("No.") == 2


def test_dedupe_keeps_first_occurrence_order():
    """First occurrence is kept; ordering is preserved otherwise."""
    a = "X" * (DEDUPE_MIN_LINE_CHARS + 5)
    b = "Y" * (DEDUPE_MIN_LINE_CHARS + 5)
    text = f"{a}\n{b}\n{a}\n"
    out = _dedupe_lines(text)
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines == [a, b]
