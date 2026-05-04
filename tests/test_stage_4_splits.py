"""tests/test_stage_4_splits.py — Stage 4 split integrity and leakage tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
SPLIT_NAMES = ("train", "val", "test")
VALID_LABELS = {"boilerplate", "substantive"}


@pytest.fixture(scope="module")
def splits() -> dict[str, list[dict]]:
    for name in SPLIT_NAMES:
        path = SPLITS_DIR / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{name}.json not yet generated — run 06_split.py first")
    return {name: json.loads((SPLITS_DIR / f"{name}.json").read_text()) for name in SPLIT_NAMES}


def test_all_split_files_exist(splits: dict) -> None:
    for name in SPLIT_NAMES:
        assert (SPLITS_DIR / f"{name}.json").exists(), f"{name}.json missing"


def test_total_row_count(splits: dict) -> None:
    total = sum(len(splits[name]) for name in SPLIT_NAMES)
    assert total == 2500, f"Expected 2500 total rows across splits, got {total}"


def test_no_train_val_leakage(splits: dict) -> None:
    train_ids = {r["sentence_id"] for r in splits["train"]}
    val_ids = {r["sentence_id"] for r in splits["val"]}
    overlap = train_ids & val_ids
    assert not overlap, f"Train/val overlap: {len(overlap)} sentence_ids"


def test_no_train_test_leakage(splits: dict) -> None:
    train_ids = {r["sentence_id"] for r in splits["train"]}
    test_ids = {r["sentence_id"] for r in splits["test"]}
    overlap = train_ids & test_ids
    assert not overlap, f"Train/test overlap: {len(overlap)} sentence_ids"


def test_no_val_test_leakage(splits: dict) -> None:
    val_ids = {r["sentence_id"] for r in splits["val"]}
    test_ids = {r["sentence_id"] for r in splits["test"]}
    overlap = val_ids & test_ids
    assert not overlap, f"Val/test overlap: {len(overlap)} sentence_ids"


def test_all_labels_valid(splits: dict) -> None:
    for name in SPLIT_NAMES:
        bad = {r["gold_label"] for r in splits[name]} - VALID_LABELS
        assert not bad, f"{name}: invalid labels {bad}"


def test_train_is_largest_split(splits: dict) -> None:
    sizes = {name: len(splits[name]) for name in SPLIT_NAMES}
    assert sizes["train"] > sizes["val"], "Train should be larger than val"
    assert sizes["train"] > sizes["test"], "Train should be larger than test"


def test_both_classes_in_each_split(splits: dict) -> None:
    for name in SPLIT_NAMES:
        labels = {r["gold_label"] for r in splits[name]}
        assert labels == VALID_LABELS, f"{name}: missing class — found {labels}"


def test_test_set_has_required_columns(splits: dict) -> None:
    for row in splits["test"][:5]:
        assert "sentence_id" in row
        assert "gold_label" in row
