"""10_model_histgbm.py — Stage 5: HistGradientBoosting on embeddings + flags.

Features: all-mpnet-base-v2 768-dim embeddings + 27 regex flags (795 total)
CV: 5-fold StratifiedGroupKFold on train split

Outputs:
  data/interim/oof_train_histgbm.parquet   (1,476 rows)
  data/interim/pred_val_histgbm.parquet    (503 rows)
  data/models/histgbm/model.pkl
  W&B run: tags=[stage:5-zoo, purpose:train, model:histgbm, split:oof]
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import (  # noqa: E402
    MODELS_DIR,
    binary_labels,
    get_embed_cols,
    get_flag_cols,
    load_features,
    load_split,
    load_text,
    make_group_ids,
    merge_split_features,
    run_group_cv,
    save_oof,
    save_val_preds,
)
from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("10_model_histgbm")


def main() -> None:
    import wandb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, recall_score

    feat_df = load_features()
    text_df = load_text()
    feature_cols = get_embed_cols(feat_df) + get_flag_cols(feat_df)

    train_df = merge_split_features(load_split("train"), feat_df)
    val_df = merge_split_features(load_split("val"), feat_df)

    x_train = train_df[feature_cols].values.astype(np.float32)
    y_train = binary_labels(train_df)
    groups = make_group_ids(train_df, text_df)
    x_val = val_df[feature_cols].values.astype(np.float32)
    y_val = binary_labels(val_df)

    hgb_params = dict(
        max_iter=300, max_leaf_nodes=31, learning_rate=0.1, random_state=42, early_stopping=False
    )

    def train_fn(x_mat, y):
        return HistGradientBoostingClassifier(**hgb_params).fit(x_mat, y)

    def predict_fn(model, x_mat):
        return model.predict_proba(x_mat)[:, 1]

    log.info("Running 5-fold group-aware CV…")
    oof_probs, fold_metrics = run_group_cv(x_train, y_train, groups, train_fn, predict_fn)

    for fm in fold_metrics:
        log.info(
            "Fold %d — macro-F1=%.4f  sub-recall=%.4f",
            fm["fold"],
            fm["macro_f1"],
            fm["substantive_recall"],
        )

    mean_f1 = float(np.mean([f["macro_f1"] for f in fold_metrics]))
    mean_recall = float(np.mean([f["substantive_recall"] for f in fold_metrics]))
    log.info("OOF macro-F1=%.4f  sub-recall=%.4f", mean_f1, mean_recall)

    t0 = time.perf_counter()
    final_model = train_fn(x_train, y_train)
    train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    val_probs = predict_fn(final_model, x_val)
    infer_sec = time.perf_counter() - t1
    throughput = len(x_val) / infer_sec

    val_preds = (val_probs >= 0.5).astype(int)
    log.info(
        "Val macro-F1=%.4f  sub-recall=%.4f  throughput=%.0f sps",
        f1_score(y_val, val_preds, average="macro"),
        recall_score(y_val, val_preds, pos_label=1),
        throughput,
    )

    save_oof(train_df["sentence_id"], oof_probs, "histgbm")
    save_val_preds(val_df["sentence_id"], val_probs, "histgbm")

    model_dir = MODELS_DIR / "histgbm"
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as fh:
        pickle.dump({"model": final_model, "feature_cols": feature_cols}, fh)

    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage5-histgbm",
        config={"model": "histgbm", **hgb_params, "n_features": len(feature_cols), "n_folds": 5},
        tags=["stage:5-zoo", "purpose:train", "model:histgbm", "split:oof"],
        job_type="train",
    )
    wandb.log(
        {
            "oof_macro_f1": mean_f1,
            "oof_substantive_recall": mean_recall,
            "val_macro_f1": float(f1_score(y_val, val_preds, average="macro")),
            "val_substantive_recall": float(recall_score(y_val, val_preds, pos_label=1)),
            "training_time_sec": train_sec,
            "infer_throughput_sps": throughput,
            **{f"fold_{f['fold']}_macro_f1": f["macro_f1"] for f in fold_metrics},
        }
    )

    artifact = wandb.Artifact(
        "model-histgbm",
        type="model",
        description="HistGBM on embeddings+flags",
        metadata={"val_macro_f1": float(f1_score(y_val, val_preds, average="macro"))},
    )
    artifact.add_file(str(model_dir / "model.pkl"))
    run.log_artifact(artifact)
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
