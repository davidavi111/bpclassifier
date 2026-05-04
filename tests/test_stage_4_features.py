"""tests/test_stage_4_features.py — Stage 4 feature matrix shape, NaN, and flag tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FEATURES_PATH = Path(__file__).parent.parent / "data" / "interim" / "features.parquet"
EXPECTED_ROWS = 2500
EMBED_DIM = 768
N_FLAGS = 27


@pytest.fixture(scope="module")
def feat_df() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip("features.parquet not yet generated — run 07_features.py first")
    return pd.read_parquet(FEATURES_PATH)


def test_row_count(feat_df: pd.DataFrame) -> None:
    assert len(feat_df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(feat_df)}"


def test_sentence_id_unique(feat_df: pd.DataFrame) -> None:
    dupes = feat_df["sentence_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate sentence_ids"


def test_embed_columns_present(feat_df: pd.DataFrame) -> None:
    for i in (0, 1, EMBED_DIM - 1):
        assert f"embed_{i}" in feat_df.columns, f"Missing embed_{i}"


def test_embed_column_count(feat_df: pd.DataFrame) -> None:
    embed_cols = [c for c in feat_df.columns if c.startswith("embed_")]
    assert len(embed_cols) == EMBED_DIM, f"Expected {EMBED_DIM} embed cols, got {len(embed_cols)}"


def test_no_nan_in_embeddings(feat_df: pd.DataFrame) -> None:
    embed_cols = [c for c in feat_df.columns if c.startswith("embed_")]
    n_nan = feat_df[embed_cols].isna().sum().sum()
    assert n_nan == 0, f"{n_nan} NaN values in embedding columns"


def test_flag_columns_present(feat_df: pd.DataFrame) -> None:
    flag_cols = [c for c in feat_df.columns if c.startswith("flag_")]
    assert len(flag_cols) == N_FLAGS, f"Expected {N_FLAGS} flag cols, got {len(flag_cols)}"


def test_flags_are_binary(feat_df: pd.DataFrame) -> None:
    flag_cols = [c for c in feat_df.columns if c.startswith("flag_")]
    for col in flag_cols:
        bad = set(feat_df[col].unique()) - {0, 1}
        assert not bad, f"{col} has non-binary values: {bad}"


def test_no_nan_in_flags(feat_df: pd.DataFrame) -> None:
    flag_cols = [c for c in feat_df.columns if c.startswith("flag_")]
    n_nan = feat_df[flag_cols].isna().sum().sum()
    assert n_nan == 0, f"{n_nan} NaN values in flag columns"


def test_embeddings_normalized(feat_df: pd.DataFrame) -> None:
    embed_cols = [c for c in feat_df.columns if c.startswith("embed_")]
    norms = np.linalg.norm(feat_df[embed_cols].values, axis=1)
    assert np.allclose(
        norms, 1.0, atol=1e-4
    ), f"Embeddings not unit-normalized; mean norm={norms.mean():.4f}"


def test_at_least_one_flag_fires_per_major_flag(feat_df: pd.DataFrame) -> None:
    always_expected = [
        "flag_dollar_amount",
        "flag_percent_change",
        "flag_thanks_greeting",
        "flag_short_sentence",
    ]
    for flag in always_expected:
        if flag in feat_df.columns:
            assert feat_df[flag].sum() > 0, f"{flag} never fires — likely a bug"
