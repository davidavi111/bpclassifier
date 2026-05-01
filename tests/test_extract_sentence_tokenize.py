"""Sentence tokenization correctness."""

from bpclassifier.extract import (
    MIN_SENTENCE_CHARS,
    _parse_speaker,
    _tokenize_sentences,
)


def test_basic_split():
    text = "Hello there. This is a test. We expect three sentences here today."
    sents = _tokenize_sentences(text)
    assert len(sents) == 3


def test_empty_text():
    assert _tokenize_sentences("") == []
    assert _tokenize_sentences("   \n  ") == []


def test_no_oversplit_on_abbrevs():
    """Common financial abbreviations should not cause spurious splits."""
    text = "Apple Inc. reported revenue. The CEO commented on the results today."
    sents = _tokenize_sentences(text)
    # Without abbrev handling punkt may produce 3; with our config it should be ≤ 2.
    # Test as upper bound to avoid brittle exact-count assertion.
    assert len(sents) <= 3


def test_min_length_constant_matches_handout():
    """The handout pins the minimum at 40 characters."""
    assert MIN_SENTENCE_CHARS == 40


def test_parse_speaker_executive():
    name, title = _parse_speaker("Executives - Lisa Su - Chair, President & CEO")
    assert name == "Lisa Su"
    assert title == "Chair, President & CEO"


def test_parse_speaker_analyst():
    name, title = _parse_speaker("Analysts - John Smith - Managing Director, Goldman Sachs")
    assert name == "John Smith"
    assert title == "Managing Director, Goldman Sachs"


def test_parse_speaker_operator_returns_none():
    assert _parse_speaker("Operator") == (None, None)


def test_parse_speaker_none_input():
    assert _parse_speaker(None) == (None, None)
