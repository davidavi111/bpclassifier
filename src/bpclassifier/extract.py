"""extract.py — earnings-call transcript parsing and sentence tokenization.

Architecture (two layers):
  Layer 1: file → structured Transcript (parser logic adapted from David's
           prior earnings-NLP project, see attribution comment in this file).
  Layer 2: Transcript → list[Sentence] with full metadata, ready for the
           gold-labeling pipeline.

Layer 1 attribution: regex patterns, section markers, dataclasses, and the
parsing functions ``parse_transcript`` / ``parse_all`` are adapted from
David's prior project earnings_nlp/parser.py. Edge cases preserved:
  - utf-8-sig BOM stripping (FAST_2026_01_20.txt)
  - PLTR transcripts that omit the Q&A operator marker (auto-enter Q&A mode)
  - Orphan answer blocks (Answer with no preceding Question)
  - Annual-call filenames in TICKER_YYYY_MM_DD form

Layer 2 is new for this project. It splits each text block into sentences
using NLTK punkt with a small custom-abbreviation layer, applies the
handout's <40-character drop rule, and emits Sentence rows.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants — lifted from prior project, unchanged.
# ──────────────────────────────────────────────────────────────────────────

SECTION_HEADERS: frozenset[str] = frozenset(
    {
        "Presentation Operator Message",
        "Presenter Speech",
        "Question and Answer Operator Message",
        "Question",
        "Answer",
    }
)

_ROLE_LINE_RE: re.Pattern[str] = re.compile(r"^(Executives|Analysts|Operator)\b.*$")

# Primary: "Company, Q4 2025 Earnings Call, Jan 01, 2026 -"
# Fallback: "Company, 2025 Earnings Call, Jan 01, 2026 -"  (annual calls)
_HEADER_RE: re.Pattern[str] = re.compile(
    r"^(?P<company>.+?),\s*(?:Q(?P<q>\d)\s*)?(?P<y>\d{4}).*?Earnings Call.*?"
    r"(?P<date>[A-Z][a-z]+ \d{1,2},\s*\d{4})"
)

# ──────────────────────────────────────────────────────────────────────────
# Constants — new for this project.
# ──────────────────────────────────────────────────────────────────────────

MIN_SENTENCE_CHARS: int = 40
"""Per the assignment handout: sentences shorter than this are dropped."""

DEDUPE_MIN_LINE_CHARS: int = 30
"""Lines shorter than this are NOT deduplicated within a transcript
(short utterances like 'Yes.' or 'Thank you.' should be preserved)."""

_FINANCIAL_ABBREVS: frozenset[str] = frozenset(
    {
        "inc",
        "corp",
        "co",
        "ltd",
        "llc",
        "lp",
        "plc",
        "no",
        "vs",
        "etc",
        "u.s",
        "u.k",
        "e.g",
        "i.e",
        "approx",
        "incl",
        "mr",
        "mrs",
        "ms",
        "dr",
        "jr",
        "sr",
    }
)
"""Abbreviations that punkt sometimes mis-splits on; treated as known abbrevs."""


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QAPair:
    """A single question-answer exchange in the Q&A section."""

    q_role: str | None
    question: str | None
    a_role: str | None
    answer: str | None


@dataclass(frozen=True)
class PreparedBlock:
    """A single speaker block from the prepared-remarks section."""

    role: str
    text: str


@dataclass(frozen=True)
class Transcript:
    """Parsed structure of one earnings-call transcript."""

    ticker: str
    quarter: str
    call_date: str | None
    company: str
    prepared: tuple[PreparedBlock, ...]
    qa: tuple[QAPair, ...]
    raw_path: str


@dataclass(frozen=True)
class Sentence:
    """One sentence with full provenance metadata, ready for gold labeling."""

    sentence_id: str  # f"{ticker}_{quarter}_{position_in_doc:05d}"
    text: str
    ticker: str
    quarter: str  # raw quarter token from filename, e.g. "Q4-2025" or "2026_01_20"
    quarter_num: int | None  # 1..4, or None for annual calls
    year: int | None
    call_date: str | None
    company: str
    section_type: str  # "prepared_remarks" | "question" | "answer"
    speaker_role: str | None  # raw role line from the transcript
    speaker_name: str | None  # best-effort parsed
    speaker_title: str | None
    position_in_block: int  # 0-indexed sentence position within its block
    position_in_doc: int  # 0-indexed sentence position within whole transcript


class TranscriptParseError(ValueError):
    """Raised when a transcript header cannot be parsed."""


# ──────────────────────────────────────────────────────────────────────────
# Private helpers — lifted from prior project.
# ──────────────────────────────────────────────────────────────────────────


def _filename_meta(path: Path) -> tuple[str, str]:
    """Extract ticker and quarter label from a transcript filename.

    Examples
    --------
    >>> _filename_meta(Path("AMD_Q4-2025.txt"))
    ('AMD', 'Q4-2025')
    >>> _filename_meta(Path("FAST_2026_01_20.txt"))
    ('FAST', '2026_01_20')
    """
    stem = path.stem
    ticker, _, quarter = stem.partition("_")
    return ticker, quarter


def _blocks(text: str) -> Generator[tuple[str, str, str], None, None]:
    """Iterate (section, role, body) tuples from a raw transcript text."""
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line in SECTION_HEADERS:
            section = line
            i += 1
            role = lines[i].strip() if i < n else ""
            i += 1
            buf: list[str] = []
            while i < n and lines[i].strip() not in SECTION_HEADERS:
                buf.append(lines[i])
                i += 1
            yield section, role, "\n".join(buf).strip()
        else:
            i += 1


def _split_prepared(role: str, body: str) -> list[PreparedBlock]:
    """Split a Presenter Speech body into one PreparedBlock per speaker."""
    blocks: list[PreparedBlock] = []
    current_role = role
    current_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if _ROLE_LINE_RE.match(stripped) and stripped != current_role:
            if "".join(current_lines).strip():
                blocks.append(
                    PreparedBlock(role=current_role, text="\n".join(current_lines).strip())
                )
            current_role = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if "".join(current_lines).strip():
        blocks.append(PreparedBlock(role=current_role, text="\n".join(current_lines).strip()))
    return blocks


# ──────────────────────────────────────────────────────────────────────────
# Private helpers — new for this project.
# ──────────────────────────────────────────────────────────────────────────

_PUNKT_TOKENIZER: Any = None


def _ensure_punkt() -> Any:
    """Load NLTK punkt sentence tokenizer, downloading data if needed.

    NLTK 3.9+ uses 'punkt_tab'; older versions used 'punkt'. Try both.
    The tokenizer is configured with our custom financial-abbrev set.
    """
    global _PUNKT_TOKENIZER
    if _PUNKT_TOKENIZER is not None:
        return _PUNKT_TOKENIZER

    import nltk
    from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            break
        except LookupError:
            data_name = resource.split("/", 1)[1]
            try:
                nltk.download(data_name, quiet=True)
                nltk.data.find(resource)
                break
            except (LookupError, ValueError, OSError):
                continue

    params = PunktParameters()
    params.abbrev_types = set(_FINANCIAL_ABBREVS)
    _PUNKT_TOKENIZER = PunktSentenceTokenizer(params)
    return _PUNKT_TOKENIZER


def _tokenize_sentences(text: str) -> list[str]:
    """Split a paragraph of text into sentences using punkt + custom abbrevs.

    Returns trimmed, non-empty sentence strings. Returns [] for empty input.
    """
    if not text or not text.strip():
        return []
    tokenizer = _ensure_punkt()
    return [s.strip() for s in tokenizer.tokenize(text) if s.strip()]


def _dedupe_lines(text: str) -> str:
    """Within a single transcript, collapse globally-repeated long lines.

    Lines with fewer than ``DEDUPE_MIN_LINE_CHARS`` are NEVER deduplicated,
    so short utterances like "Yes." are preserved. The first occurrence of
    each long line is kept; subsequent occurrences are dropped.
    """
    seen: set[str] = set()
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= DEDUPE_MIN_LINE_CHARS:
            if stripped in seen:
                continue
            seen.add(stripped)
        out_lines.append(line)
    return "\n".join(out_lines)


_SPEAKER_NAME_RE: re.Pattern[str] = re.compile(
    r"^(?:Executives|Analysts)\s*-\s*(?P<name>[^-]+?)\s*-\s*(?P<title>.+)$"
)


def _parse_speaker(role_line: str | None) -> tuple[str | None, str | None]:
    """Best-effort parse of a role line into (name, title).

    Examples
    --------
    >>> _parse_speaker("Executives - Lisa Su - Chair, President & CEO")
    ('Lisa Su', 'Chair, President & CEO')
    >>> _parse_speaker("Operator")
    (None, None)
    >>> _parse_speaker(None)
    (None, None)
    """
    if not role_line:
        return None, None
    m = _SPEAKER_NAME_RE.match(role_line.strip())
    if not m:
        return None, None
    return m.group("name").strip(), m.group("title").strip()


def _resolve_role_and_body(role: str, body: str) -> tuple[str | None, str]:
    """Disambiguate the role-line vs. body output of ``_blocks``.

    ``_blocks`` always treats the first line after a section header as the
    role line. But many transcripts skip the role line entirely and start
    directly with content. Detect this by checking whether the captured
    "role" matches the speaker-label pattern (Executives / Analysts /
    Operator). If it doesn't, it's actually body text and should be merged
    back into the body, with role set to None.

    Edge cases handled:
      - AVGO_Q1-2024: very short Q&A where the entire question fits on one
        line right after the ``Question`` header (body is empty).
      - Q&A sections that begin directly with the question/answer text
        without a role line (body is non-empty but role is sentence text).
      - Presenter Speech sections that begin with content.

    Returns
    -------
    (resolved_role, resolved_body) : tuple
        ``resolved_role`` is ``None`` if the captured role was sentence text;
        otherwise the original role string. ``resolved_body`` always contains
        all of the original text (role and body re-joined when role wasn't
        actually a role line).
    """
    if role and not _ROLE_LINE_RE.match(role):
        combined = (role + "\n" + body).strip() if body else role
        return None, combined
    return role, body


def _parse_quarter_token(quarter: str) -> tuple[int | None, int | None]:
    """Extract (quarter_num, year) from a filename quarter token.

    Examples
    --------
    >>> _parse_quarter_token("Q4-2025")
    (4, 2025)
    >>> _parse_quarter_token("2026_01_20")
    (None, 2026)
    """
    m = re.match(r"^Q(\d)-(\d{4})$", quarter)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})", quarter)
    if m:
        return None, int(m.group(1))
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Public API — Layer 1 (parsing). Adapted from prior project.
# ──────────────────────────────────────────────────────────────────────────


def parse_transcript(path: Path) -> Transcript:
    """Parse a single S&P-format earnings-call transcript file.

    Adapted from David's prior project earnings_nlp/parser.py.
    Adds within-transcript line deduplication before structure parsing
    (per assignment handout's "de-duplicate repeated lines" step).
    """
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")  # utf-8-sig strips BOM
    raw = _dedupe_lines(raw)

    first_line = raw.splitlines()[0].strip() if raw.strip() else ""
    m = _HEADER_RE.match(first_line)
    if not m:
        raise TranscriptParseError(
            f"Cannot parse header in '{path.name}': first line was: {first_line!r}"
        )

    company = m.group("company").strip()
    try:
        call_date: str | None = datetime.strptime(
            m.group("date").replace("  ", " "), "%b %d, %Y"
        ).strftime("%Y-%m-%d")
    except ValueError:
        call_date = None

    prepared: list[PreparedBlock] = []
    qa: list[dict] = []
    current_q: dict | None = None
    in_qa = False

    for section, role, body in _blocks(raw):
        if section == "Question and Answer Operator Message":
            in_qa = True
            continue
        # PLTR transcripts omit the Q&A operator marker; auto-enter Q&A mode.
        if not in_qa and section in ("Question", "Answer"):
            in_qa = True

        if not in_qa and section == "Presenter Speech":
            resolved_role, resolved_body = _resolve_role_and_body(role, body)
            prepared.extend(_split_prepared(resolved_role or "", resolved_body))
        elif in_qa and section == "Question":
            q_role, question = _resolve_role_and_body(role, body)
            current_q = {
                "q_role": q_role,
                "question": question,
                "a_role": None,
                "answer": None,
            }
            qa.append(current_q)
        elif in_qa and section == "Answer":
            a_role, answer = _resolve_role_and_body(role, body)
            if current_q is None or current_q["answer"] is not None:
                # Orphan answer (no preceding question, or previous Q already answered).
                current_q = {
                    "q_role": None,
                    "question": None,
                    "a_role": a_role,
                    "answer": answer,
                }
                qa.append(current_q)
            else:
                current_q["a_role"] = a_role
                current_q["answer"] = answer

    ticker, quarter = _filename_meta(path)
    return Transcript(
        ticker=ticker,
        quarter=quarter,
        call_date=call_date,
        company=company,
        prepared=tuple(prepared),
        qa=tuple(
            QAPair(
                q_role=q["q_role"],
                question=q["question"],
                a_role=q["a_role"],
                answer=q["answer"],
            )
            for q in qa
        ),
        raw_path=str(path),
    )


def parse_all(ect_dir: Path) -> list[Transcript]:
    """Parse every .txt file in *ect_dir*, sorted by filename."""
    paths = sorted(p for p in ect_dir.iterdir() if p.suffix == ".txt")
    results: list[Transcript] = []
    for p in paths:
        results.append(parse_transcript(p))
    logger.info("Parsed %d transcripts from %s", len(results), ect_dir)
    return results


# ──────────────────────────────────────────────────────────────────────────
# Public API — Layer 2 (sentence tokenization). New for this project.
# ──────────────────────────────────────────────────────────────────────────


def transcript_to_sentences(t: Transcript) -> list[Sentence]:
    """Flatten a Transcript into Sentence rows with full metadata.

    Drops sentences shorter than ``MIN_SENTENCE_CHARS``.
    """
    quarter_num, year = _parse_quarter_token(t.quarter)
    out: list[Sentence] = []
    pos_doc = 0

    def _emit(text: str, section_type: str, speaker_role: str | None) -> None:
        nonlocal pos_doc
        speaker_name, speaker_title = _parse_speaker(speaker_role)
        sentences = _tokenize_sentences(text)
        for pos_block, s in enumerate(sentences):
            if len(s) < MIN_SENTENCE_CHARS:
                continue
            out.append(
                Sentence(
                    sentence_id=f"{t.ticker}_{t.quarter}_{pos_doc:05d}",
                    text=s,
                    ticker=t.ticker,
                    quarter=t.quarter,
                    quarter_num=quarter_num,
                    year=year,
                    call_date=t.call_date,
                    company=t.company,
                    section_type=section_type,
                    speaker_role=speaker_role,
                    speaker_name=speaker_name,
                    speaker_title=speaker_title,
                    position_in_block=pos_block,
                    position_in_doc=pos_doc,
                )
            )
            pos_doc += 1

    for block in t.prepared:
        _emit(block.text, "prepared_remarks", block.role)

    for pair in t.qa:
        if pair.question:
            _emit(pair.question, "question", pair.q_role)
        if pair.answer:
            _emit(pair.answer, "answer", pair.a_role)

    return out


def extract_all(ect_dir: Path) -> tuple[list[Transcript], list[Sentence]]:
    """End-to-end: parse all transcripts and produce flat Sentence list."""
    transcripts = parse_all(ect_dir)
    sentences: list[Sentence] = []
    for t in transcripts:
        sentences.extend(transcript_to_sentences(t))
    logger.info("Extracted %d sentences from %d transcripts", len(sentences), len(transcripts))
    return transcripts, sentences
