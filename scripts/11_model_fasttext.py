"""11_model_fasttext.py — Stage 5: FastText text classifier.

Features: raw sentence text
CV: 5-fold StratifiedGroupKFold on train split
FastText format: __label__substantive / __label__boilerplate + text

Outputs:
  data/interim/oof_train_fasttext.parquet   (1,476 rows)
  data/interim/pred_val_fasttext.parquet    (503 rows)
  data/models/fasttext/model.bin
  W&B run: tags=[stage:5-zoo, purpose:train, model:fasttext, split:oof]
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import (  # noqa: E402
    MODELS_DIR,
    N_FOLDS,
    SEED,
    binary_labels,
    load_split,
    load_text,
    make_group_ids,
    save_oof,
    save_val_preds,
)
from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("11_model_fasttext")

FT_PARAMS = dict(epoch=15, lr=0.5, wordNgrams=2, dim=100, minCount=1, loss="softmax", thread=4)


def _write_ft_file(
    texts: pd.Series, labels: np.ndarray, path: Path, oversample_minority: bool = True
) -> None:
    """Write FastText-format training file, optionally oversampling the minority class."""
    lines = []
    for text, label in zip(texts, labels, strict=False):
        label_str = "__label__substantive" if label == 1 else "__label__boilerplate"
        lines.append(f"{label_str} {text.strip()}\n")

    if oversample_minority:
        minority_label = 0 if labels.mean() > 0.5 else 1
        minority_lines = [
            line for line, lbl in zip(lines, labels, strict=False) if lbl == minority_label
        ]
        n_majority = sum(1 for lbl in labels if lbl != minority_label)
        n_minority = len(minority_lines)
        if n_minority > 0:
            repeat = max(1, round(n_majority / n_minority)) - 1
            lines += minority_lines * repeat

    rng = np.random.default_rng(SEED)
    rng.shuffle(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _train_ft(train_file: Path) -> object:
    import fasttext

    return fasttext.train_supervised(str(train_file), **FT_PARAMS)


def _predict_ft(model: object, texts: pd.Series) -> np.ndarray:
    """Return p(substantive) by calling model.f.predict() directly (NumPy-2 safe).

    model.f.predict() returns list[(score, label)] — bypass the broken Python wrapper.
    """
    probs = []
    for text in texts:
        raw = model.f.predict(text.strip().replace("\n", " "), 2, 0.0, "strict")
        label_score = {label: float(score) for score, label in raw}
        probs.append(label_score.get("__label__substantive", 0.0))
    return np.clip(np.array(probs, dtype=float), 0.0, 1.0)


def main() -> None:
    import wandb
    from sklearn.metrics import f1_score, recall_score
    from sklearn.model_selection import StratifiedGroupKFold

    text_df = load_text()

    train_split = load_split("train")
    val_split = load_split("val")

    # Merge to get text for train sentences
    train_merged = train_split.merge(
        text_df[["sentence_id", "text", "ticker", "quarter"]], on="sentence_id"
    )
    val_merged = val_split.merge(
        text_df[["sentence_id", "text", "ticker", "quarter"]], on="sentence_id"
    )

    y_train = binary_labels(train_merged.rename(columns={"gold_label": "gold_label"}))
    y_val = binary_labels(val_merged.rename(columns={"gold_label": "gold_label"}))

    # Re-add gold_label column needed by binary_labels (already there via load_split)
    train_merged["gold_label"] = (
        train_split.set_index("sentence_id").loc[train_merged["sentence_id"], "gold_label"].values
    )
    y_train = (train_merged["gold_label"] == "substantive").astype(int).values
    y_val = (val_merged["gold_label"] == "substantive").astype(int).values

    groups = make_group_ids(train_merged, text_df)

    # 5-fold CV
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(train_merged), dtype=float)
    fold_metrics = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_merged, y_train, groups)):
            tr_texts = train_merged["text"].iloc[tr_idx]
            tr_labels = y_train[tr_idx]
            val_texts = train_merged["text"].iloc[val_idx]
            val_labels = y_train[val_idx]

            train_file = tmpdir / f"fold_{fold_idx}_train.txt"
            _write_ft_file(tr_texts, tr_labels, train_file)

            model = _train_ft(train_file)
            probs = _predict_ft(model, val_texts)
            oof_probs[val_idx] = probs

            preds = (probs >= 0.5).astype(int)
            fold_metrics.append(
                {
                    "fold": fold_idx,
                    "macro_f1": float(f1_score(val_labels, preds, average="macro")),
                    "substantive_recall": float(recall_score(val_labels, preds, pos_label=1)),
                }
            )
            log.info(
                "Fold %d — macro-F1=%.4f  sub-recall=%.4f",
                fold_idx,
                fold_metrics[-1]["macro_f1"],
                fold_metrics[-1]["substantive_recall"],
            )

        mean_f1 = float(np.mean([f["macro_f1"] for f in fold_metrics]))
        mean_recall = float(np.mean([f["substantive_recall"] for f in fold_metrics]))
        log.info("OOF macro-F1=%.4f  sub-recall=%.4f", mean_f1, mean_recall)

        # Final model on full train
        full_train_file = tmpdir / "full_train.txt"
        _write_ft_file(train_merged["text"], y_train, full_train_file)

        t0 = time.perf_counter()
        final_model = _train_ft(full_train_file)
        train_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    val_probs = _predict_ft(final_model, val_merged["text"])
    infer_sec = time.perf_counter() - t1
    throughput = len(val_merged) / infer_sec

    val_preds = (val_probs >= 0.5).astype(int)
    log.info(
        "Val macro-F1=%.4f  sub-recall=%.4f  throughput=%.0f sps",
        f1_score(y_val, val_preds, average="macro"),
        recall_score(y_val, val_preds, pos_label=1),
        throughput,
    )

    save_oof(train_merged["sentence_id"], oof_probs, "fasttext")
    save_val_preds(val_merged["sentence_id"], val_probs, "fasttext")

    model_dir = MODELS_DIR / "fasttext"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.bin"
    final_model.save_model(str(model_path))
    log.info("Model saved → %s", model_path)

    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage5-fasttext",
        config={"model": "fasttext", **FT_PARAMS, "n_folds": N_FOLDS},
        tags=["stage:5-zoo", "purpose:train", "model:fasttext", "split:oof"],
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
        "model-fasttext",
        type="model",
        description="FastText supervised on raw text",
        metadata={"val_macro_f1": float(f1_score(y_val, val_preds, average="macro"))},
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
