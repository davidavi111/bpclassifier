"""tests/test_stage_3_meta_judge.py — Stage 3 meta-judge output contract tests.

These tests validate the parquet artifact written by scripts/05_meta_judge.py.
They are skipped when the output file does not yet exist (script not yet run).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "interim" / "meta_judge_outputs.parquet"
EXPECTED_ROWS = 332
VALID_LABELS = {"boilerplate", "substantive"}
EXPECTED_MODEL = "Qwen/Qwen3.5-27B"


@pytest.fixture(scope="module")
def meta_df() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        pytest.skip("meta_judge_outputs.parquet not yet generated — run 05_meta_judge.py first")
    return pd.read_parquet(OUTPUT_PATH)


def test_row_count(meta_df: pd.DataFrame) -> None:
    assert len(meta_df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(meta_df)}"


def test_no_null_meta_label(meta_df: pd.DataFrame) -> None:
    nulls = meta_df["meta_label"].isna().sum()
    assert nulls == 0, f"{nulls} rows have null meta_label"


def test_no_null_meta_reasoning(meta_df: pd.DataFrame) -> None:
    nulls = meta_df["meta_reasoning"].isna().sum()
    assert nulls == 0, f"{nulls} rows have null meta_reasoning"


def test_meta_label_values(meta_df: pd.DataFrame) -> None:
    bad = set(meta_df["meta_label"].unique()) - VALID_LABELS
    assert not bad, f"Unexpected meta_label values: {bad}"


def test_model_id_uniform(meta_df: pd.DataFrame) -> None:
    bad = meta_df[meta_df["model_id"] != EXPECTED_MODEL]
    assert (
        len(bad) == 0
    ), f"{len(bad)} rows have unexpected model_id: {bad['model_id'].unique().tolist()}"


def test_sentence_id_unique(meta_df: pd.DataFrame) -> None:
    dupes = meta_df["sentence_id"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate sentence_ids found"


def test_required_columns_present(meta_df: pd.DataFrame) -> None:
    required = {"sentence_id", "meta_label", "meta_reasoning", "model_id"}
    missing = required - set(meta_df.columns)
    assert not missing, f"Missing columns: {missing}"
