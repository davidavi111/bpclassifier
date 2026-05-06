"""tests/test_stage_6_winner.py — Stage 6 output contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
EVAL_DIR = _ROOT / "data" / "eval"
SPLITS_DIR = _ROOT / "data" / "splits"

TEST_N = (
    len(json.loads((SPLITS_DIR / "test.json").read_text()))
    if (SPLITS_DIR / "test.json").exists()
    else 0
)


@pytest.fixture(scope="module")
def leaderboard() -> pd.DataFrame:
    path = EVAL_DIR / "leaderboard.parquet"
    if not path.exists():
        pytest.skip("leaderboard.parquet not found — run 12_ensembles_winner.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def test_winner() -> pd.DataFrame:
    path = EVAL_DIR / "test_winner.parquet"
    if not path.exists():
        pytest.skip("test_winner.parquet not found — run 12_ensembles_winner.py first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def cm() -> dict:
    path = EVAL_DIR / "confusion_matrix_test.json"
    if not path.exists():
        pytest.skip("confusion_matrix_test.json not found — run 12_ensembles_winner.py first")
    return json.loads(path.read_text())


def test_leaderboard_has_8_rows(leaderboard: pd.DataFrame) -> None:
    assert len(leaderboard) == 8, f"Expected 8 leaderboard rows, got {len(leaderboard)}"


def test_leaderboard_has_required_columns(leaderboard: pd.DataFrame) -> None:
    required = {"model", "threshold", "threshold_std", "val_recall", "val_macro_f1", "eligible"}
    assert required.issubset(set(leaderboard.columns))


def test_winner_is_eligible(leaderboard: pd.DataFrame) -> None:
    winner = leaderboard[leaderboard["eligible"]].iloc[0]
    assert winner["eligible"] is True or winner["eligible"] == True  # noqa: E712


def test_winner_val_recall_meets_floor(leaderboard: pd.DataFrame) -> None:
    winner = leaderboard[leaderboard["eligible"]].iloc[0]
    assert (
        winner["val_recall"] >= 0.96
    ), f"Winner val recall {winner['val_recall']:.4f} < 0.96 floor"


def test_test_recall_floor(cm: dict) -> None:
    recall = cm["tp"] / (cm["tp"] + cm["fn"])
    assert recall >= 0.96, f"Test substantive_recall={recall:.4f} < 0.96 floor"


def test_test_winner_row_count(test_winner: pd.DataFrame) -> None:
    assert len(test_winner) == TEST_N, f"Expected {TEST_N} test rows, got {len(test_winner)}"


def test_test_winner_no_nulls(test_winner: pd.DataFrame) -> None:
    assert test_winner["prob_substantive"].isna().sum() == 0


def test_test_winner_probs_in_range(test_winner: pd.DataFrame) -> None:
    assert test_winner["prob_substantive"].between(0.0, 1.0).all()
