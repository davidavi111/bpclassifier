"""train_utils.py — Shared helpers for Stage 5 classifier training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

_ROOT = Path(__file__).parent.parent.parent

SPLITS_DIR = _ROOT / "data" / "splits"
FEATURES_PATH = _ROOT / "data" / "interim" / "features.parquet"
SAMPLE_PATH = _ROOT / "data" / "interim" / "labeling_sample.parquet"
MODELS_DIR = _ROOT / "data" / "models"

LABEL_COL = "gold_label"
POS_CLASS = "substantive"
N_FOLDS = 5
SEED = 42

BOILERPLATE_FLAGS = [
    "flag_safe_harbor",
    "flag_sec_filing_ref",
    "flag_please_see_filing",
    "flag_risk_factor_ref",
    "flag_legal_disclaimer",
    "flag_operator_phrase",
    "flag_operator_closing",
    "flag_thanks_greeting",
    "flag_replay_info",
    "flag_speaker_handoff",
]
SUBSTANTIVE_FLAGS = [
    "flag_dollar_amount",
    "flag_percent_change",
    "flag_yoy_comparison",
    "flag_qoq_comparison",
    "flag_guidance_language",
    "flag_per_share_metric",
    "flag_gaap_nongaap",
    "flag_organic_growth",
    "flag_strategic_language",
    "flag_ma_language",
    "flag_cost_reduction",
    "flag_market_share",
    "flag_product_launch",
]
ALL_FLAGS = (
    BOILERPLATE_FLAGS
    + SUBSTANTIVE_FLAGS
    + [
        "flag_short_sentence",
        "flag_all_caps_ratio",
        "flag_question_mark",
        "flag_numeric_density",
    ]
)


def load_split(name: str) -> pd.DataFrame:
    """Load a split JSON → DataFrame with sentence_id and gold_label."""
    path = SPLITS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(data)


def load_features() -> pd.DataFrame:
    """Load features.parquet (sentence_id, embed_*, flag_*)."""
    return pd.read_parquet(FEATURES_PATH)


def load_text() -> pd.DataFrame:
    """Load sentence text + group metadata from labeling_sample.parquet."""
    return pd.read_parquet(SAMPLE_PATH)[["sentence_id", "text", "ticker", "quarter"]]


def get_embed_cols(feat_df: pd.DataFrame) -> list[str]:
    return [c for c in feat_df.columns if c.startswith("embed_")]


def get_flag_cols(feat_df: pd.DataFrame) -> list[str]:
    return [c for c in feat_df.columns if c.startswith("flag_")]


def merge_split_features(split_df: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Join split rows with feature matrix; returns merged DataFrame."""
    return split_df.merge(feat_df, on="sentence_id", how="inner")


def make_group_ids(df: pd.DataFrame, text_df: pd.DataFrame) -> np.ndarray:
    """Return group IDs (ticker_quarter) aligned with df.sentence_id."""
    mapping = text_df.assign(
        group_id=text_df["ticker"] + "_" + text_df["quarter"].astype(str)
    ).set_index("sentence_id")["group_id"]
    return df["sentence_id"].map(mapping).values


def binary_labels(df: pd.DataFrame) -> np.ndarray:
    """Return binary label array: 1=substantive, 0=boilerplate."""
    return (df[LABEL_COL] == POS_CLASS).astype(int).values


def run_group_cv(
    x_mat: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_fn: callable,
    predict_proba_fn: callable,
    n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, list[dict]]:
    """
    5-fold stratified group-aware CV.

    Returns:
        oof_probs: shape (n,) — OOF predicted probability of positive class
        fold_metrics: list of per-fold metric dicts
    """
    from sklearn.metrics import f1_score, recall_score

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_mat, y, groups)):
        x_tr, x_val = x_mat[train_idx], x_mat[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = train_fn(x_tr, y_tr)
        probs = predict_proba_fn(model, x_val)
        oof_probs[val_idx] = probs

        preds = (probs >= 0.5).astype(int)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "macro_f1": float(f1_score(y_val, preds, average="macro")),
                "substantive_recall": float(recall_score(y_val, preds, pos_label=1)),
            }
        )

    return oof_probs, fold_metrics


def measure_speed(model: Any, x_mat: np.ndarray, predict_fn: callable) -> tuple[float, float]:
    """Return (train_sec, infer_sentences_per_sec) measured on x_mat."""
    t0 = time.perf_counter()
    predict_fn(model, x_mat)
    infer_sec = time.perf_counter() - t0
    throughput = len(x_mat) / infer_sec if infer_sec > 0 else float("inf")
    return infer_sec, throughput


def save_oof(sentence_ids: pd.Series, oof_probs: np.ndarray, model_name: str) -> Path:
    """Save OOF probabilities for train split → data/interim/oof_train_{model}.parquet."""
    out = _ROOT / "data" / "interim" / f"oof_train_{model_name}.parquet"
    pd.DataFrame({"sentence_id": sentence_ids, "prob_substantive": oof_probs}).to_parquet(
        out, index=False
    )
    return out


def save_val_preds(sentence_ids: pd.Series, probs: np.ndarray, model_name: str) -> Path:
    """Save val predictions → data/interim/pred_val_{model}.parquet."""
    out = _ROOT / "data" / "interim" / f"pred_val_{model_name}.parquet"
    pd.DataFrame({"sentence_id": sentence_ids, "prob_substantive": probs}).to_parquet(
        out, index=False
    )
    return out
