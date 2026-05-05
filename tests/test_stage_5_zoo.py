"""tests/test_stage_5_zoo.py — Stage 5 classifier output contract tests.

Tests validate the OOF and val-prediction parquets written by the 4 local CPU models.
Skipped when output files don't exist yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
SPLITS_DIR = _ROOT / "data" / "splits"
INTERIM = _ROOT / "data" / "interim"

MODELS = ["rules", "logreg", "histgbm", "fasttext", "finbert", "setfit"]

TRAIN_N = (
    json.loads((SPLITS_DIR / "train.json").read_text())
    if (SPLITS_DIR / "train.json").exists()
    else []
)
VAL_N = (
    json.loads((SPLITS_DIR / "val.json").read_text()) if (SPLITS_DIR / "val.json").exists() else []
)
TRAIN_SIZE = len(TRAIN_N)
VAL_SIZE = len(VAL_N)


@pytest.fixture(scope="module", params=MODELS)
def model_name(request):
    return request.param


@pytest.fixture(scope="module", params=MODELS)
def oof_df(request) -> pd.DataFrame:
    model = request.param
    path = INTERIM / f"oof_train_{model}.parquet"
    if not path.exists():
        pytest.skip(f"oof_train_{model}.parquet not found — run the model script first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module", params=MODELS)
def val_df(request) -> pd.DataFrame:
    model = request.param
    path = INTERIM / f"pred_val_{model}.parquet"
    if not path.exists():
        pytest.skip(f"pred_val_{model}.parquet not found — run the model script first")
    return pd.read_parquet(path)


def test_oof_row_count(oof_df: pd.DataFrame) -> None:
    assert len(oof_df) == TRAIN_SIZE, f"Expected {TRAIN_SIZE} OOF rows, got {len(oof_df)}"


def test_oof_no_nulls(oof_df: pd.DataFrame) -> None:
    assert oof_df["prob_substantive"].isna().sum() == 0


def test_oof_prob_in_range(oof_df: pd.DataFrame) -> None:
    assert oof_df["prob_substantive"].between(0.0, 1.0).all(), "OOF probs outside [0, 1]"


def test_oof_sentence_id_unique(oof_df: pd.DataFrame) -> None:
    assert oof_df["sentence_id"].is_unique, "Duplicate sentence_ids in OOF"


def test_val_row_count(val_df: pd.DataFrame) -> None:
    assert len(val_df) == VAL_SIZE, f"Expected {VAL_SIZE} val rows, got {len(val_df)}"


def test_val_no_nulls(val_df: pd.DataFrame) -> None:
    assert val_df["prob_substantive"].isna().sum() == 0


def test_val_prob_in_range(val_df: pd.DataFrame) -> None:
    assert val_df["prob_substantive"].between(0.0, 1.0).all(), "Val probs outside [0, 1]"


def test_val_sentence_id_unique(val_df: pd.DataFrame) -> None:
    assert val_df["sentence_id"].is_unique, "Duplicate sentence_ids in val predictions"


def test_oof_not_all_same_value(oof_df: pd.DataFrame) -> None:
    """Model must produce varied predictions — all-same signals a constant predictor."""
    std = oof_df["prob_substantive"].std()
    assert std > 0.01, f"OOF probs have near-zero std ({std:.4f}) — likely a constant predictor"


def test_val_not_all_same_value(val_df: pd.DataFrame) -> None:
    std = val_df["prob_substantive"].std()
    assert std > 0.01, f"Val probs have near-zero std ({std:.4f}) — likely a constant predictor"
