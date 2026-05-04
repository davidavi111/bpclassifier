"""tests/test_gold_class_balance.py — Verify gold label class balance constraints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

GOLD_PATH = Path(__file__).parent.parent / "data" / "gold" / "labeled.parquet"
VALID_LABELS = {"boilerplate", "substantive"}


@pytest.fixture(scope="module")
def gold_df() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        pytest.skip("labeled.parquet not yet generated — run 04_freeze_gold.py first")
    return pd.read_parquet(GOLD_PATH)


def test_both_classes_present(gold_df: pd.DataFrame) -> None:
    found = set(gold_df["gold_label"].unique())
    assert found == VALID_LABELS, f"Expected both classes; found: {found}"


def test_substantive_majority(gold_df: pd.DataFrame) -> None:
    counts = gold_df["gold_label"].value_counts()
    assert (
        counts["substantive"] > counts["boilerplate"]
    ), "Expected substantive to outnumber boilerplate in earnings-call corpus"


def test_boilerplate_at_least_10_pct(gold_df: pd.DataFrame) -> None:
    frac = (gold_df["gold_label"] == "boilerplate").mean()
    assert frac >= 0.10, f"Boilerplate fraction {frac:.2%} is suspiciously low (< 10%)"


def test_substantive_at_least_50_pct(gold_df: pd.DataFrame) -> None:
    frac = (gold_df["gold_label"] == "substantive").mean()
    assert frac >= 0.50, f"Substantive fraction {frac:.2%} is unexpectedly low (< 50%)"


def test_no_invalid_labels(gold_df: pd.DataFrame) -> None:
    bad = set(gold_df["gold_label"].unique()) - VALID_LABELS
    assert not bad, f"Unexpected labels found: {bad}"
