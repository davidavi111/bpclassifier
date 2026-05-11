"""13_switch_winner_ensemble.py — Switch winner to FinBERT+SetFit 50/50 ensemble.

Steps:
  1. Load SetFit model from W&B (model-setfit:v0), predict on test set.
  2. Load FinBERT test probs from data/eval/test_winner.parquet.
  3. Ensemble: new_prob = 0.5*finbert + 0.5*setfit, threshold=0.16.
  4. Compute OOF fold-std on train split using mean of finbert+setfit OOF probs.
  5. Compute test metrics; STOP if substantive_recall < 0.96.
  6. Overwrite: test_winner.parquet, confusion_matrix_test.json, leaderboard.parquet.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import SEED, load_split, load_text  # noqa: E402
from gpu_runner import get_wandb_key  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("13_switch_winner_ensemble")

ENTITY = "david-avichzer-hebrew-university-of-jerusalem"
PROJECT = "Boilerplate_Classifier"
INTERIM = _ROOT / "data" / "interim"
EVAL_DIR = _ROOT / "data" / "eval"
RECALL_FLOOR = 0.96
WINNER_THRESHOLD = 0.16
N_FOLDS = 5
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 4)


# ─── Threshold helpers ────────────────────────────────────────────────────────


def tune_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    from sklearn.metrics import f1_score, recall_score

    best: tuple[float, float, float] | None = None
    for t in THRESHOLDS:
        preds = (probs >= t).astype(int)
        recall = float(recall_score(y, preds, pos_label=1, zero_division=0))
        if recall >= RECALL_FLOOR:
            f1 = float(f1_score(y, preds, average="macro", zero_division=0))
            if best is None or f1 > best[2]:
                best = (float(t), recall, f1)
    return best


def threshold_std_oof(
    probs_oof: np.ndarray,
    y_oof: np.ndarray,
    groups: np.ndarray,
) -> float:
    from sklearn.model_selection import StratifiedGroupKFold

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_thresholds: list[float] = []
    for _, vl_idx in skf.split(probs_oof, y_oof, groups):
        result = tune_threshold(probs_oof[vl_idx], y_oof[vl_idx])
        if result is not None:
            fold_thresholds.append(result[0])
    return float(np.std(fold_thresholds)) if len(fold_thresholds) >= 2 else 0.0


def main() -> None:
    import wandb
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    get_wandb_key()

    # ── 1. Load test split + text ──────────────────────────────────────────────
    test_split = load_split("test")
    text_df = load_text()
    test_merged = test_split.merge(text_df[["sentence_id", "text"]], on="sentence_id", how="inner")
    log.info("Test split: %d sentences.", len(test_merged))

    # ── 2. Load SetFit model from W&B, predict on test ────────────────────────
    log.info("Downloading model-setfit:v0 from W&B…")
    api = wandb.Api()
    art = api.artifact(f"{ENTITY}/{PROJECT}/model-setfit:v0")
    model_dir = Path(art.download())
    log.info("Model downloaded to %s", model_dir)

    from setfit import SetFitModel  # noqa: PLC0415

    sf_model = SetFitModel.from_pretrained(str(model_dir))
    log.info("SetFit model loaded. Running test inference…")
    raw = sf_model.predict_proba(test_merged["text"].tolist())
    setfit_probs = np.clip(
        raw.numpy() if hasattr(raw, "numpy") else np.array(raw, dtype=float),
        0.0,
        1.0,
    )[:, 1]
    log.info("SetFit test probs: shape=%s, mean=%.4f", setfit_probs.shape, setfit_probs.mean())

    pred_test_setfit = pd.DataFrame(
        {"sentence_id": test_merged["sentence_id"].values, "prob_substantive": setfit_probs}
    )
    pred_test_setfit.to_parquet(INTERIM / "pred_test_setfit.parquet", index=False)
    log.info("Saved → data/interim/pred_test_setfit.parquet")

    # ── 3. Load FinBERT test probs ────────────────────────────────────────────
    tw_df = pd.read_parquet(EVAL_DIR / "test_winner.parquet")
    # Merge to align sentence_ids (test_winner has finbert probs as prob_substantive)
    finbert_test = tw_df[["sentence_id", "prob_substantive"]].rename(
        columns={"prob_substantive": "finbert_prob"}
    )

    # ── 4. Build ensemble on test ─────────────────────────────────────────────
    ensemble_df = pred_test_setfit.rename(columns={"prob_substantive": "setfit_prob"}).merge(
        finbert_test, on="sentence_id", how="inner"
    )
    # Also need gold labels
    ensemble_df = ensemble_df.merge(
        test_split[["sentence_id", "gold_label"]], on="sentence_id", how="inner"
    )
    log.info("Ensemble test set: %d sentences.", len(ensemble_df))

    ensemble_df["prob_substantive"] = (
        0.5 * ensemble_df["finbert_prob"] + 0.5 * ensemble_df["setfit_prob"]
    )
    ensemble_df["pred_label"] = np.where(
        ensemble_df["prob_substantive"] >= WINNER_THRESHOLD, "substantive", "boilerplate"
    )

    # ── 5. OOF fold-std ──────────────────────────────────────────────────────
    log.info("Computing OOF fold-std for FinBERT+SetFit ensemble…")
    train_split = load_split("train")
    y_train = (train_split["gold_label"] == "substantive").astype(int).values

    oof_finbert = pd.read_parquet(INTERIM / "oof_train_finbert.parquet").merge(
        train_split[["sentence_id"]], on="sentence_id", how="inner"
    )
    oof_setfit = pd.read_parquet(INTERIM / "oof_train_setfit.parquet").merge(
        train_split[["sentence_id"]], on="sentence_id", how="inner"
    )
    oof_ensemble = (
        0.5 * oof_finbert["prob_substantive"].values + 0.5 * oof_setfit["prob_substantive"].values
    )

    groups_train = (
        train_split.merge(
            text_df.assign(group=text_df["ticker"] + "_" + text_df["quarter"].astype(str))[
                ["sentence_id", "group"]
            ],
            on="sentence_id",
            how="left",
        )["group"]
        .fillna("unknown")
        .values
    )

    t_std = threshold_std_oof(oof_ensemble, y_train, groups_train)
    log.info("OOF threshold std: %.4f", t_std)

    # ── 6. Test metrics ───────────────────────────────────────────────────────
    log.info("=== FINAL TEST EVAL — FinBERT+SetFit ensemble @ threshold=%.2f ===", WINNER_THRESHOLD)

    y_test = (ensemble_df["gold_label"] == "substantive").astype(int).values
    test_probs = ensemble_df["prob_substantive"].values
    test_preds = (test_probs >= WINNER_THRESHOLD).astype(int)

    test_sub_recall = float(recall_score(y_test, test_preds, pos_label=1, zero_division=0))
    test_sub_prec = float(precision_score(y_test, test_preds, pos_label=1, zero_division=0))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro", zero_division=0))

    log.info(
        "Test  sub-recall=%.4f  sub-prec=%.4f  macro-F1=%.4f",
        test_sub_recall,
        test_sub_prec,
        test_macro_f1,
    )

    if test_sub_recall < RECALL_FLOOR:
        log.error(
            "STOP: Test substantive_recall=%.4f < %.2f floor. NOT updating any artifacts.",
            test_sub_recall,
            RECALL_FLOOR,
        )
        sys.exit(1)

    cm = confusion_matrix(y_test, test_preds).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    log.info("Confusion matrix: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

    # ── 7. Overwrite artifacts ────────────────────────────────────────────────

    # confusion_matrix_test.json
    cm_dict = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "winner": "finbert+setfit",
        "threshold": WINNER_THRESHOLD,
        "test_macro_f1": test_macro_f1,
        "test_substantive_recall": test_sub_recall,
        "test_substantive_precision": test_sub_prec,
    }
    (EVAL_DIR / "confusion_matrix_test.json").write_text(
        json.dumps(cm_dict, indent=2), encoding="utf-8"
    )
    log.info("Overwrote confusion_matrix_test.json")

    # test_winner.parquet
    new_tw = ensemble_df[["sentence_id", "prob_substantive", "pred_label", "gold_label"]].copy()
    new_tw.to_parquet(EVAL_DIR / "test_winner.parquet", index=False)
    log.info("Overwrote test_winner.parquet")

    # leaderboard.parquet — add ensemble row marked as winner
    lb = pd.read_parquet(EVAL_DIR / "leaderboard.parquet")
    # Mark old rows as not winner
    if "is_winner" not in lb.columns:
        lb["is_winner"] = False
    else:
        lb["is_winner"] = False

    new_row = pd.DataFrame(
        [
            {
                "model": "finbert+setfit",
                "threshold": WINNER_THRESHOLD,
                "threshold_std": t_std,
                "val_recall": 0.962,  # from val sweep (provided externally)
                "val_macro_f1": 0.876,  # from val sweep (provided externally)
                "eligible": True,
                "is_winner": True,
            }
        ]
    )
    lb = pd.concat([new_row, lb], ignore_index=True)
    lb.to_parquet(EVAL_DIR / "leaderboard.parquet", index=False)
    log.info("Updated leaderboard.parquet — finbert+setfit row added as winner.")

    # ── 8. Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENSEMBLE SWITCH COMPLETE")
    print(f"  Winner: finbert+setfit @ threshold={WINNER_THRESHOLD:.2f}")
    print(f"  OOF threshold std: {t_std:.4f}")
    print(f"  Test substantive recall: {test_sub_recall:.4f}")
    print(f"  Test macro-F1: {test_macro_f1:.4f}")
    print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
