"""08_model_rules.py — Stage 5: Rules-based classifier.

No training. Boilerplate score = fraction of boilerplate flags that fire.
Substantive score = fraction of substantive flags that fire.
p_substantive = sigmoid(substantive_score - boilerplate_score, scale=4)

Since rules have no learned parameters, OOF = full-train predictions.

Outputs:
  data/interim/oof_train_rules.parquet   (1,476 rows: sentence_id, prob_substantive)
  data/interim/pred_val_rules.parquet    (503 rows)
  W&B run: tags=[stage:5-zoo, purpose:eval, model:rules]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import (  # noqa: E402
    BOILERPLATE_FLAGS,
    SUBSTANTIVE_FLAGS,
    load_features,
    load_split,
    merge_split_features,
    save_oof,
    save_val_preds,
)
from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("08_model_rules")


def rules_score(df: pd.DataFrame) -> np.ndarray:
    """Return p_substantive via sigmoid(sub_score - bp_score)."""
    bp_cols = [c for c in BOILERPLATE_FLAGS if c in df.columns]
    sub_cols = [c for c in SUBSTANTIVE_FLAGS if c in df.columns]
    bp_score = df[bp_cols].mean(axis=1).values
    sub_score = df[sub_cols].mean(axis=1).values
    net = sub_score - bp_score
    return 1.0 / (1.0 + np.exp(-4.0 * net))


def main() -> None:
    import wandb
    from sklearn.metrics import f1_score, recall_score

    feat_df = load_features()
    train_df = merge_split_features(load_split("train"), feat_df)
    val_df = merge_split_features(load_split("val"), feat_df)

    t0 = time.perf_counter()
    train_probs = rules_score(train_df)
    val_probs = rules_score(val_df)
    infer_time = time.perf_counter() - t0
    throughput = (len(train_df) + len(val_df)) / infer_time

    train_preds = (train_probs >= 0.5).astype(int)
    train_y = (train_df["gold_label"] == "substantive").astype(int).values
    val_preds = (val_probs >= 0.5).astype(int)
    val_y = (val_df["gold_label"] == "substantive").astype(int).values

    log.info(
        "Train macro-F1: %.4f  substantive recall: %.4f",
        f1_score(train_y, train_preds, average="macro"),
        recall_score(train_y, train_preds, pos_label=1),
    )
    log.info(
        "Val   macro-F1: %.4f  substantive recall: %.4f",
        f1_score(val_y, val_preds, average="macro"),
        recall_score(val_y, val_preds, pos_label=1),
    )
    log.info("Throughput: %.0f sentences/sec", throughput)

    save_oof(train_df["sentence_id"], train_probs, "rules")
    save_val_preds(val_df["sentence_id"], val_probs, "rules")

    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage5-rules",
        config={
            "model": "rules",
            "n_boilerplate_flags": len(BOILERPLATE_FLAGS),
            "n_substantive_flags": len(SUBSTANTIVE_FLAGS),
            "sigmoid_scale": 4.0,
        },
        tags=["stage:5-zoo", "purpose:eval", "model:rules"],
        job_type="eval",
    )
    wandb.log(
        {
            "val_macro_f1": float(f1_score(val_y, val_preds, average="macro")),
            "val_substantive_recall": float(recall_score(val_y, val_preds, pos_label=1)),
            "train_macro_f1": float(f1_score(train_y, train_preds, average="macro")),
            "infer_throughput_sps": throughput,
            "training_time_sec": 0.0,
        }
    )
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
