"""tests/test_gold_no_duplicates.py — Verify gold labels have no duplicate sentence_ids."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

GOLD_PATH = Path(__file__).parent.parent / "data" / "gold" / "labeled.parquet"
EXPECTED_ROWS = 2500
VALID_SOURCES = {"unanimous", "meta_judge", "audit"}


@pytest.fixture(scope="module")
def gold_df() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        pytest.skip("labeled.parquet not yet generated — run 04_freeze_gold.py first")
    return pd.read_parquet(GOLD_PATH)


def test_row_count(gold_df: pd.DataFrame) -> None:
    assert len(gold_df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(gold_df)}"


def test_no_duplicate_sentence_ids(gold_df: pd.DataFrame) -> None:
    dupes = gold_df["sentence_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate sentence_ids found"


def test_no_null_gold_label(gold_df: pd.DataFrame) -> None:
    nulls = gold_df["gold_label"].isna().sum()
    assert nulls == 0, f"{nulls} rows have null gold_label"


def test_no_null_source(gold_df: pd.DataFrame) -> None:
    nulls = gold_df["source"].isna().sum()
    assert nulls == 0, f"{nulls} rows have null source"


def test_valid_source_values(gold_df: pd.DataFrame) -> None:
    bad = set(gold_df["source"].unique()) - VALID_SOURCES
    assert not bad, f"Unexpected source values: {bad}"


def test_required_columns_present(gold_df: pd.DataFrame) -> None:
    required = {"sentence_id", "gold_label", "source"}
    missing = required - set(gold_df.columns)
    assert not missing, f"Missing columns: {missing}"
