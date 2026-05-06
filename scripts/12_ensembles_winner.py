"""12_ensembles_winner.py — Stage 6: Ensembles + winner + final test eval.

Decision flow:
  1. Load all 6 OOF + val-pred parquets (local + from W&B for finbert/setfit).
  2. Build 2 ensembles (mean_prob, rank_avg) over the 4 non-transformer models.
  3. Sweep threshold ∈ [0.05, 0.95] on VAL for all 8 candidates.
  4. Eligible = at least one threshold achieves substantive_recall ≥ 0.96 on val.
  5. Winner = highest val macro-F1 among eligible candidates.
  6. FINAL EVAL: apply winner threshold to TEST (touched ONCE).
  7. Save artifacts + W&B Model Registry alias='production'.

Outputs:
  data/eval/leaderboard.parquet
  data/eval/test_winner.parquet
  data/eval/confusion_matrix_test.json
  W&B run: tags=[stage:6-winner, purpose:eval, split:test]
  W&B Model Registry: BPClassifier / alias=production
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import (  # noqa: E402
    BOILERPLATE_FLAGS,
    MODELS_DIR,
    SEED,
    SUBSTANTIVE_FLAGS,
    load_features,
    load_split,
    load_text,
    merge_split_features,
)
from gpu_runner import get_wandb_key, init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("12_ensembles_winner")

# ─── Constants ────────────────────────────────────────────────────────────────

ENTITY = "david-avichzer-hebrew-university-of-jerusalem"
PROJECT = "Boilerplate_Classifier"
INTERIM = _ROOT / "data" / "interim"
EVAL_DIR = _ROOT / "data" / "eval"

LOCAL_MODELS = ["rules", "logreg", "histgbm", "fasttext"]
DEEP_MODELS = ["finbert", "setfit"]
BASE_MODELS = LOCAL_MODELS + DEEP_MODELS
ENSEMBLE_NAMES = ["mean_prob", "rank_avg"]
CANDIDATES = BASE_MODELS + ENSEMBLE_NAMES  # 8 total

RECALL_FLOOR = 0.96
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 4)
N_FOLDS = 5


# ─── Rule-based scoring ───────────────────────────────────────────────────────


def rules_score(df: pd.DataFrame) -> np.ndarray:
    bp_cols = [c for c in BOILERPLATE_FLAGS if c in df.columns]
    sub_cols = [c for c in SUBSTANTIVE_FLAGS if c in df.columns]
    bp = df[bp_cols].mean(axis=1).values
    sub = df[sub_cols].mean(axis=1).values
    return 1.0 / (1.0 + np.exp(-4.0 * (sub - bp)))


# ─── FastText prediction helper ───────────────────────────────────────────────


def ft_predict(model: object, texts: pd.Series) -> np.ndarray:
    probs = []
    for text in texts:
        raw = model.f.predict(text.strip().replace("\n", " "), 2, 0.0, "strict")
        label_score = {label: float(score) for score, label in raw}
        probs.append(label_score.get("__label__substantive", 0.0))
    return np.clip(np.array(probs, dtype=float), 0.0, 1.0)


# ─── Threshold tuning ─────────────────────────────────────────────────────────


def tune_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Sweep thresholds; return (threshold, recall, macro_f1) for best eligible, or None."""
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
    """Return std of per-fold optimal thresholds (threshold variance proxy)."""
    from sklearn.model_selection import StratifiedGroupKFold

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_thresholds: list[float] = []
    for _, vl_idx in skf.split(probs_oof, y_oof, groups):
        result = tune_threshold(probs_oof[vl_idx], y_oof[vl_idx])
        if result is not None:
            fold_thresholds.append(result[0])
    return float(np.std(fold_thresholds)) if len(fold_thresholds) >= 2 else 0.0


# ─── Ensemble builders ────────────────────────────────────────────────────────


def mean_prob_ensemble(probs_dict: dict[str, np.ndarray]) -> np.ndarray:
    stack = np.stack(list(probs_dict.values()), axis=1)
    return stack.mean(axis=1)


def rank_avg_ensemble(probs_dict: dict[str, np.ndarray]) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = np.stack([rankdata(p) / len(p) for p in probs_dict.values()], axis=1)
    return ranks.mean(axis=1)


# ─── Test-set inference per model ─────────────────────────────────────────────


def get_test_probs_base(
    name: str,
    test_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    text_df: pd.DataFrame,
    api,
) -> tuple[np.ndarray, pd.Series]:
    """Return (probs, sentence_ids) for a base model on the test split."""
    if name == "rules":
        merged = merge_split_features(test_df, feat_df)
        return rules_score(merged), merged["sentence_id"]

    if name == "logreg":
        merged = merge_split_features(test_df, feat_df)
        with open(MODELS_DIR / "logreg" / "model.pkl", "rb") as fh:
            saved = pickle.load(fh)  # noqa: S301
        x = saved["scaler"].transform(merged[saved["feature_cols"]].values.astype(np.float32))
        return saved["model"].predict_proba(x)[:, 1], merged["sentence_id"]

    if name == "histgbm":
        merged = merge_split_features(test_df, feat_df)
        with open(MODELS_DIR / "histgbm" / "model.pkl", "rb") as fh:
            saved = pickle.load(fh)  # noqa: S301
        x = merged[saved["feature_cols"]].values.astype(np.float32)
        return saved["model"].predict_proba(x)[:, 1], merged["sentence_id"]

    if name == "fasttext":
        import fasttext  # noqa: PLC0415

        test_merged = test_df.merge(text_df[["sentence_id", "text"]], on="sentence_id", how="inner")
        model = fasttext.load_model(str(MODELS_DIR / "fasttext" / "model.bin"))
        return ft_predict(model, test_merged["text"]), test_merged["sentence_id"]

    if name == "finbert":
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415, N812
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

        test_merged = test_df.merge(text_df[["sentence_id", "text"]], on="sentence_id", how="inner")
        art = api.artifact(f"{ENTITY}/{PROJECT}/model-finbert:v0")
        model_dir = Path(art.download())
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()

        all_probs: list[float] = []
        texts = test_merged["text"].tolist()
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt",
                )
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
        return np.clip(np.array(all_probs, dtype=float), 0.0, 1.0), test_merged["sentence_id"]

    if name == "setfit":
        from setfit import SetFitModel  # noqa: PLC0415

        test_merged = test_df.merge(text_df[["sentence_id", "text"]], on="sentence_id", how="inner")
        art = api.artifact(f"{ENTITY}/{PROJECT}/model-setfit:v0")
        model_dir = Path(art.download())
        model = SetFitModel.from_pretrained(str(model_dir))
        raw = model.predict_proba(test_merged["text"].tolist())
        arr = raw.numpy() if hasattr(raw, "numpy") else np.array(raw, dtype=float)
        return np.clip(arr[:, 1], 0.0, 1.0), test_merged["sentence_id"]

    raise ValueError(f"Unknown base model: {name}")


# ─── Main ─────────────────────────────────────────────────────────────────────


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

    # ── 1. Load splits + features + text ──────────────────────────────────────
    feat_df = load_features()
    text_df = load_text()

    train_split = load_split("train")
    val_split = load_split("val")
    # test_split loaded here so we can align sentence_ids; labels NOT used yet.
    test_split = load_split("test")

    y_train = (train_split["gold_label"] == "substantive").astype(int).values
    y_val = (val_split["gold_label"] == "substantive").astype(int).values

    # Reconstruct OOF group IDs (ticker_quarter) for threshold_std
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

    # ── 2. Load OOF + val probs for base models ────────────────────────────────
    probs_oof: dict[str, np.ndarray] = {}
    probs_val: dict[str, np.ndarray] = {}

    for name in BASE_MODELS:
        oof_path = INTERIM / f"oof_train_{name}.parquet"
        val_path = INTERIM / f"pred_val_{name}.parquet"
        if not oof_path.exists() or not val_path.exists():
            log.error("Missing OOF/val parquet for %s — aborting.", name)
            sys.exit(1)

        oof_df = pd.read_parquet(oof_path).merge(
            train_split[["sentence_id"]], on="sentence_id", how="inner"
        )
        val_df_m = pd.read_parquet(val_path).merge(
            val_split[["sentence_id"]], on="sentence_id", how="inner"
        )
        probs_oof[name] = oof_df["prob_substantive"].values
        probs_val[name] = val_df_m["prob_substantive"].values

    log.info("Loaded OOF/val probs for %d base models.", len(BASE_MODELS))

    # ── 3. Build ensemble probs (4 non-transformer models) ────────────────────
    local_oof = {m: probs_oof[m] for m in LOCAL_MODELS}
    local_val = {m: probs_val[m] for m in LOCAL_MODELS}

    probs_oof["mean_prob"] = mean_prob_ensemble(local_oof)
    probs_val["mean_prob"] = mean_prob_ensemble(local_val)
    probs_oof["rank_avg"] = rank_avg_ensemble(local_oof)
    probs_val["rank_avg"] = rank_avg_ensemble(local_val)
    log.info("Built 2 ensemble candidates.")

    # ── 4. Per-candidate threshold tuning + leaderboard ───────────────────────
    rows: list[dict] = []
    for name in CANDIDATES:
        result = tune_threshold(probs_val[name], y_val)
        eligible = result is not None

        if eligible:
            chosen_t, val_recall, val_f1 = result
            t_std = threshold_std_oof(probs_oof[name], y_train, groups_train)
        else:
            # Best recall achievable (at lowest threshold)
            preds_low = (probs_val[name] >= THRESHOLDS[0]).astype(int)
            val_recall = float(recall_score(y_val, preds_low, pos_label=1, zero_division=0))
            val_f1 = float(f1_score(y_val, preds_low, average="macro", zero_division=0))
            chosen_t = float("nan")
            t_std = float("nan")
            log.warning(
                "%s INELIGIBLE — best recall=%.4f (floor=%.2f).", name, val_recall, RECALL_FLOOR
            )

        rows.append(
            {
                "model": name,
                "threshold": chosen_t,
                "threshold_std": t_std,
                "val_recall": val_recall,
                "val_macro_f1": val_f1,
                "eligible": eligible,
            }
        )
        log.info(
            "%-12s  eligible=%-5s  threshold=%.2f  val_recall=%.4f  val_f1=%.4f",
            name,
            str(eligible),
            chosen_t if eligible else float("nan"),
            val_recall,
            val_f1,
        )

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(["eligible", "val_macro_f1"], ascending=[False, False])
        .reset_index(drop=True)
    )
    leaderboard.to_parquet(EVAL_DIR / "leaderboard.parquet", index=False)
    log.info("Leaderboard saved → %s", EVAL_DIR / "leaderboard.parquet")

    # ── 5. Select winner ──────────────────────────────────────────────────────
    eligible_rows = leaderboard[leaderboard["eligible"]]
    n_eligible = len(eligible_rows)
    log.info("%d / %d candidates eligible on val.", n_eligible, len(CANDIDATES))

    if n_eligible == 0:
        log.error(
            "STOP: NO candidate achieves substantive_recall ≥ %.2f on val. "
            "Cannot select a winner.",
            RECALL_FLOOR,
        )
        sys.exit(1)

    winner_row = eligible_rows.iloc[0]
    winner_name = winner_row["model"]
    winner_threshold = float(winner_row["threshold"])
    winner_val_f1 = float(winner_row["val_macro_f1"])
    winner_val_recall = float(winner_row["val_recall"])
    winner_t_std = float(winner_row["threshold_std"])

    log.info(
        "WINNER: %s  threshold=%.2f  std=%.4f  val_recall=%.4f  val_f1=%.4f",
        winner_name,
        winner_threshold,
        winner_t_std,
        winner_val_recall,
        winner_val_f1,
    )

    # ── 6. Generate test predictions for winner ───────────────────────────────
    log.info("Generating test predictions for winner '%s'…", winner_name)

    api = wandb.Api()

    if winner_name in BASE_MODELS:
        test_probs, test_sids = get_test_probs_base(winner_name, test_split, feat_df, text_df, api)
    elif winner_name in ENSEMBLE_NAMES:
        # Need test probs for all 4 local models
        local_test: dict[str, np.ndarray] = {}
        test_sids_ref: pd.Series | None = None
        for m in LOCAL_MODELS:
            tp, ts = get_test_probs_base(m, test_split, feat_df, text_df, api)
            local_test[m] = tp
            if test_sids_ref is None:
                test_sids_ref = ts
        test_sids = test_sids_ref
        if winner_name == "mean_prob":
            test_probs = mean_prob_ensemble(local_test)
        else:
            test_probs = rank_avg_ensemble(local_test)
    else:
        raise ValueError(f"Unknown winner: {winner_name}")

    log.info("Test probs generated: %d sentences.", len(test_probs))

    # ── 7. FINAL EVAL on test (labels loaded HERE and ONLY HERE) ──────────────
    log.info("=== FINAL TEST EVAL — touching test labels for the first time ===")

    # Align test_sids with test_split order
    test_df_aligned = pd.DataFrame(
        {"sentence_id": test_sids.values, "prob_substantive": test_probs}
    ).merge(test_split[["sentence_id", "gold_label"]], on="sentence_id", how="inner")

    y_test_aligned = (test_df_aligned["gold_label"] == "substantive").astype(int).values
    test_probs_aligned = test_df_aligned["prob_substantive"].values

    test_preds = (test_probs_aligned >= winner_threshold).astype(int)

    test_sub_recall = float(recall_score(y_test_aligned, test_preds, pos_label=1, zero_division=0))
    test_sub_prec = float(precision_score(y_test_aligned, test_preds, pos_label=1, zero_division=0))
    test_macro_f1 = float(f1_score(y_test_aligned, test_preds, average="macro", zero_division=0))
    test_bp_f1 = float(
        f1_score(y_test_aligned, test_preds, pos_label=0, average="binary", zero_division=0)
    )
    test_sub_f1 = float(
        f1_score(y_test_aligned, test_preds, pos_label=1, average="binary", zero_division=0)
    )

    log.info(
        "Test  sub-recall=%.4f  sub-prec=%.4f  macro-F1=%.4f",
        test_sub_recall,
        test_sub_prec,
        test_macro_f1,
    )
    log.info("Test  bp-F1=%.4f  sub-F1=%.4f", test_bp_f1, test_sub_f1)

    if test_sub_recall < RECALL_FLOOR:
        log.error(
            "STOP: Test substantive_recall=%.4f < %.2f floor. "
            "Winner fails on test set. DO NOT relax the floor.",
            test_sub_recall,
            RECALL_FLOOR,
        )
        sys.exit(1)

    cm = confusion_matrix(y_test_aligned, test_preds).tolist()
    cm_dict = {
        "tn": cm[0][0],
        "fp": cm[0][1],
        "fn": cm[1][0],
        "tp": cm[1][1],
        "winner": winner_name,
        "threshold": winner_threshold,
        "test_macro_f1": test_macro_f1,
        "test_substantive_recall": test_sub_recall,
        "test_substantive_precision": test_sub_prec,
    }
    (EVAL_DIR / "confusion_matrix_test.json").write_text(
        json.dumps(cm_dict, indent=2), encoding="utf-8"
    )

    test_winner_df = test_df_aligned[["sentence_id", "prob_substantive"]].copy()
    test_winner_df["pred_label"] = np.where(test_preds == 1, "substantive", "boilerplate")
    test_winner_df["gold_label"] = test_df_aligned["gold_label"].values
    test_winner_df.to_parquet(EVAL_DIR / "test_winner.parquet", index=False)

    log.info("Outputs saved → %s", EVAL_DIR)

    # ── 8. W&B run + artifact + registry ──────────────────────────────────────
    run = init_wandb_run(
        project=PROJECT,
        name="stage6-winner",
        config={
            "winner": winner_name,
            "threshold": winner_threshold,
            "threshold_std": winner_t_std,
            "recall_floor": RECALL_FLOOR,
            "n_candidates": len(CANDIDATES),
            "n_eligible": n_eligible,
        },
        tags=["stage:6-winner", "purpose:eval", "split:test"],
        job_type="eval",
    )

    wandb.log(
        {
            "n_eligible": n_eligible,
            "winner_val_macro_f1": winner_val_f1,
            "winner_val_recall": winner_val_recall,
            "winner_threshold": winner_threshold,
            "winner_threshold_std": winner_t_std,
            "test_macro_f1": test_macro_f1,
            "test_substantive_recall": test_sub_recall,
            "test_substantive_precision": test_sub_prec,
            "test_boilerplate_f1": test_bp_f1,
            "test_substantive_f1": test_sub_f1,
        }
    )

    # Log leaderboard as W&B table (keep NaN as None — W&B handles null natively)
    lb_for_table = leaderboard.where(leaderboard.notna(), other=None)
    lb_table = wandb.Table(dataframe=lb_for_table)
    wandb.log({"leaderboard": lb_table})

    # Create winner artifact (threshold config + reference to model)
    winner_config = {
        "winner_model": winner_name,
        "threshold": winner_threshold,
        "threshold_std": winner_t_std,
        "val_macro_f1": winner_val_f1,
        "val_substantive_recall": winner_val_recall,
        "test_macro_f1": test_macro_f1,
        "test_substantive_recall": test_sub_recall,
        "recall_floor": RECALL_FLOOR,
    }
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "winner_config.json"
        config_path.write_text(json.dumps(winner_config, indent=2), encoding="utf-8")

        winner_art = wandb.Artifact(
            "winner-model",
            type="model",
            description=f"Stage 6 winner: {winner_name} @ threshold={winner_threshold:.2f}",
            metadata=winner_config,
        )
        winner_art.add_file(str(config_path), name="winner_config.json")

        # Add underlying model files if local
        if winner_name == "logreg":
            winner_art.add_file(str(MODELS_DIR / "logreg" / "model.pkl"), name="model.pkl")
        elif winner_name == "histgbm":
            winner_art.add_file(str(MODELS_DIR / "histgbm" / "model.pkl"), name="model.pkl")
        elif winner_name == "fasttext":
            winner_art.add_file(str(MODELS_DIR / "fasttext" / "model.bin"), name="model.bin")

        run.log_artifact(winner_art)

    # Link to Model Registry
    try:
        winner_art.wait()
        run.link_artifact(
            winner_art,
            target_path=f"{ENTITY}/model-registry/BPClassifier",
            aliases=["production"],
        )
        log.info("Linked winner-model to Model Registry as 'production'.")
    except Exception as exc:
        log.warning("Model Registry link failed (non-fatal): %s", exc)

    run.finish()
    log.info("W&B run finished.")

    # ── 9. Print checkpoint summary ────────────────────────────────────────────
    top3 = leaderboard[leaderboard["eligible"]].head(3)
    top3_str = ", ".join(f"{r.model}={r.val_macro_f1:.3f}" for r in top3.itertuples())
    print("\n" + "=" * 60)
    print("CHECKPOINT 6")
    print(f"  Winner: {winner_name} @ threshold={winner_threshold:.2f} (std={winner_t_std:.4f})")
    print(f"  Test substantive recall: {test_sub_recall:.4f}")
    print(f"  Test macro-F1: {test_macro_f1:.4f}")
    print(f"  Eligible candidates on val: {n_eligible}/8")
    print(f"  Top-3 by val macro-F1: {top3_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
