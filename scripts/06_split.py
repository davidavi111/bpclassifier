"""06_split.py — Stage 4: Create stratified group-aware 60/20/20 train/val/test splits.

Group = ticker + quarter (one transcript per group). All sentences from the same
transcript stay in the same split. Stratification is on gold_label at the group level
(majority label per group used as the stratum key).

Inputs:
  data/gold/labeled.parquet          (2,500 rows: sentence_id, gold_label, source)
  data/interim/labeling_sample.parquet  (2,500 rows: sentence_id, ticker, quarter, text, …)

Outputs:
  data/splits/train.json  — list of {"sentence_id": …, "gold_label": …}
  data/splits/val.json
  data/splits/test.json
  W&B run: project=Boilerplate_Classifier,
           tags=[stage:4-features, purpose:extract, split:train]

Usage:
  conda activate Boilerplate_Classifier
  python scripts/06_split.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))

from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("06_split")

GOLD_PATH = _ROOT / "data" / "gold" / "labeled.parquet"
SAMPLE_PATH = _ROOT / "data" / "interim" / "labeling_sample.parquet"
SPLITS_DIR = _ROOT / "data" / "splits"

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# test = 1 - TRAIN_FRAC - VAL_FRAC = 0.20
SEED = 42


def main() -> None:
    import wandb

    log.info("Loading gold labels → %s", GOLD_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["sentence_id", "gold_label"]]

    log.info("Loading sample metadata → %s", SAMPLE_PATH)
    sample = pd.read_parquet(SAMPLE_PATH)[["sentence_id", "ticker", "quarter"]]

    merged = gold.merge(sample, on="sentence_id", how="inner")
    log.info("Merged: %d rows", len(merged))

    # Group key: one transcript per (ticker, quarter) pair
    merged["group_id"] = merged["ticker"] + "_" + merged["quarter"].astype(str)

    # Per-group majority label for stratification
    group_labels = (
        merged.groupby("group_id")["gold_label"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
        .rename(columns={"gold_label": "majority_label"})
    )
    log.info("Unique groups (transcripts): %d", len(group_labels))

    # Split groups: first carve out test (20%), then split remainder into train/val
    # val fraction within train+val pool: 0.20 / (0.60 + 0.20) = 0.25
    test_frac = 1.0 - TRAIN_FRAC - VAL_FRAC
    val_in_trainval = VAL_FRAC / (1.0 - test_frac)  # 0.20 / 0.80 = 0.25

    train_val_groups, test_groups = train_test_split(
        group_labels,
        test_size=0.20,
        stratify=group_labels["majority_label"],
        random_state=SEED,
    )
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=val_in_trainval,
        stratify=train_val_groups["majority_label"],
        random_state=SEED,
    )

    train_ids = set(train_groups["group_id"])
    val_ids = set(val_groups["group_id"])
    test_ids = set(test_groups["group_id"])

    def _rows(group_set: set) -> list[dict]:
        mask = merged["group_id"].isin(group_set)
        return merged[mask][["sentence_id", "gold_label"]].to_dict(orient="records")

    train_rows = _rows(train_ids)
    val_rows = _rows(val_ids)
    test_rows = _rows(test_ids)

    log.info(
        "Split sizes — train: %d  val: %d  test: %d  total: %d",
        len(train_rows),
        len(val_rows),
        len(test_rows),
        len(train_rows) + len(val_rows) + len(test_rows),
    )

    # Verify no leakage
    train_sids = {r["sentence_id"] for r in train_rows}
    val_sids = {r["sentence_id"] for r in val_rows}
    test_sids = {r["sentence_id"] for r in test_rows}
    assert not (train_sids & val_sids), "Train/val sentence_id overlap!"
    assert not (train_sids & test_sids), "Train/test sentence_id overlap!"
    assert not (val_sids & test_sids), "Val/test sentence_id overlap!"
    log.info("Leakage checks passed.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = SPLITS_DIR / f"{name}.json"
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        log.info("Saved %s → %s (%d rows)", name, path, len(rows))

    def _balance(rows: list[dict]) -> dict:
        from collections import Counter

        return dict(Counter(r["gold_label"] for r in rows))

    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage4-split",
        config={
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": 1 - TRAIN_FRAC - VAL_FRAC,
            "seed": SEED,
            "n_groups": len(group_labels),
            "n_train_groups": len(train_groups),
            "n_val_groups": len(val_groups),
            "n_test_groups": len(test_groups),
        },
        tags=["stage:4-features", "purpose:extract", "split:train"],
        job_type="extract",
    )

    wandb.log(
        {
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "train_boilerplate": _balance(train_rows).get("boilerplate", 0),
            "train_substantive": _balance(train_rows).get("substantive", 0),
            "val_boilerplate": _balance(val_rows).get("boilerplate", 0),
            "val_substantive": _balance(val_rows).get("substantive", 0),
            "test_boilerplate": _balance(test_rows).get("boilerplate", 0),
            "test_substantive": _balance(test_rows).get("substantive", 0),
        }
    )
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
