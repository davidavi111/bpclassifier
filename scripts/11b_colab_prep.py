"""11b_colab_prep.py — Stage 5: Upload Colab bridge data + local OOF artifacts to W&B.

Pushes two W&B Artifacts:
  colab-data:v0
    train_with_text.parquet  (1,476 rows: sentence_id, text, gold_label, ticker, quarter)
    val_with_text.parquet    (503 rows: same schema)
  oof-local:v0
    oof_train_{rules,logreg,histgbm,fasttext}.parquet
    pred_val_{rules,logreg,histgbm,fasttext}.parquet

Usage:
  conda activate Boilerplate_Classifier
  python scripts/11b_colab_prep.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.train_utils import (  # noqa: E402
    SAMPLE_PATH,
    SPLITS_DIR,
)
from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("11b_colab_prep")

INTERIM = _ROOT / "data" / "interim"
LOCAL_MODELS = ["rules", "logreg", "histgbm", "fasttext"]


def build_split_with_text(split_name: str, text_df: pd.DataFrame) -> pd.DataFrame:
    import json

    rows = json.loads((SPLITS_DIR / f"{split_name}.json").read_text(encoding="utf-8"))
    split_df = pd.DataFrame(rows)
    merged = split_df.merge(
        text_df[["sentence_id", "text", "ticker", "quarter"]], on="sentence_id", how="inner"
    )
    log.info("%s split: %d rows after merge", split_name, len(merged))
    return merged


def main() -> None:
    import wandb

    text_df = pd.read_parquet(SAMPLE_PATH)[["sentence_id", "text", "ticker", "quarter"]]
    log.info("Loaded labeling_sample: %d rows", len(text_df))

    train_with_text = build_split_with_text("train", text_df)
    val_with_text = build_split_with_text("val", text_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        train_path = tmpdir / "train_with_text.parquet"
        val_path = tmpdir / "val_with_text.parquet"
        train_with_text.to_parquet(train_path, index=False)
        val_with_text.to_parquet(val_path, index=False)
        log.info("Written train/val parquets to temp dir")

        run = init_wandb_run(
            project="Boilerplate_Classifier",
            name="stage5-colab-prep",
            config={
                "n_train": len(train_with_text),
                "n_val": len(val_with_text),
                "local_models": LOCAL_MODELS,
            },
            tags=["stage:5-zoo", "purpose:prep", "split:colab"],
            job_type="prep",
        )

        colab_art = wandb.Artifact(
            "colab-data",
            type="dataset",
            description="Train + val splits with sentence text for Colab fine-tuning",
            metadata={
                "n_train": len(train_with_text),
                "n_val": len(val_with_text),
                "columns": list(train_with_text.columns),
            },
        )
        colab_art.add_file(str(train_path), name="train_with_text.parquet")
        colab_art.add_file(str(val_path), name="val_with_text.parquet")
        run.log_artifact(colab_art)
        log.info("Uploaded colab-data artifact")

        oof_art = wandb.Artifact(
            "oof-local",
            type="predictions",
            description="OOF + val predictions from the 4 local CPU classifiers",
            metadata={"models": LOCAL_MODELS},
        )
        for model in LOCAL_MODELS:
            oof_path = INTERIM / f"oof_train_{model}.parquet"
            val_path_m = INTERIM / f"pred_val_{model}.parquet"
            if not oof_path.exists():
                raise FileNotFoundError(f"Missing {oof_path} — run 08/09/10/11 scripts first")
            if not val_path_m.exists():
                raise FileNotFoundError(f"Missing {val_path_m} — run 08/09/10/11 scripts first")
            oof_df = pd.read_parquet(oof_path)
            val_df = pd.read_parquet(val_path_m)
            log.info(
                "%s: oof=%d rows, val=%d rows",
                model,
                len(oof_df),
                len(val_df),
            )
            oof_art.add_file(str(oof_path), name=f"oof_train_{model}.parquet")
            oof_art.add_file(str(val_path_m), name=f"pred_val_{model}.parquet")

        run.log_artifact(oof_art)
        log.info("Uploaded oof-local artifact (%d models)", len(LOCAL_MODELS))

        wandb.log(
            {
                "n_train": len(train_with_text),
                "n_val": len(val_with_text),
                "n_local_models": len(LOCAL_MODELS),
            }
        )
        run.finish()
        log.info("Done — colab-data:v0 and oof-local:v0 pushed to W&B")


if __name__ == "__main__":
    main()
