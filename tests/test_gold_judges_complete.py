"""tests/test_gold_judges_complete.py — Verify gold label source coverage and judge completeness."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

GOLD_PATH = Path(__file__).parent.parent / "data" / "gold" / "labeled.parquet"
AUDIT_PATH = Path(__file__).parent.parent / "data" / "gold" / "audit_decisions.jsonl"


@pytest.fixture(scope="module")
def gold_df() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        pytest.skip("labeled.parquet not yet generated — run 04_freeze_gold.py first")
    return pd.read_parquet(GOLD_PATH)


def test_unanimous_count_majority(gold_df: pd.DataFrame) -> None:
    n_unanimous = (gold_df["source"] == "unanimous").sum()
    assert (
        n_unanimous > 2000
    ), f"Expected >2000 unanimous rows (judges agreed on most); got {n_unanimous}"


def test_audit_source_present(gold_df: pd.DataFrame) -> None:
    n_audit = (gold_df["source"] == "audit").sum()
    assert n_audit > 0, "Expected at least some audit-sourced labels"


def test_meta_judge_source_present(gold_df: pd.DataFrame) -> None:
    n_meta = (gold_df["source"] == "meta_judge").sum()
    assert n_meta > 0, "Expected at least some meta_judge-sourced labels"


def test_audit_labels_match_decisions(gold_df: pd.DataFrame) -> None:
    """Audit-sourced rows must match david_label in audit_decisions.jsonl."""
    if not AUDIT_PATH.exists():
        pytest.skip("audit_decisions.jsonl not found")
    import json

    decisions: dict[str, str] = {}
    with open(AUDIT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                decisions[entry["sentence_id"]] = entry["david_label"]
            except (json.JSONDecodeError, KeyError):
                pass

    audit_rows = gold_df[gold_df["source"] == "audit"]
    for _, row in audit_rows.iterrows():
        sid = row["sentence_id"]
        assert sid in decisions, f"Audit-sourced row {sid} not found in audit_decisions.jsonl"
        assert (
            row["gold_label"] == decisions[sid]
        ), f"Label mismatch for {sid}: gold={row['gold_label']!r}, david={decisions[sid]!r}"


def test_disagreement_rows_have_non_unanimous_source(gold_df: pd.DataFrame) -> None:
    """All non-unanimous rows must be sourced from audit or meta_judge."""
    non_unanimous = gold_df[gold_df["source"] != "unanimous"]
    bad = non_unanimous[~non_unanimous["source"].isin({"audit", "meta_judge"})]
    assert len(bad) == 0, f"{len(bad)} non-unanimous rows have unexpected source"
