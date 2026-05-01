"""01_extract.py — Stage 2 pipeline: parse ECT transcripts to a sentence pool.

Outputs:
  - data/raw/sentences.parquet           (regenerable, gitignored)
  - W&B run (project: Boilerplate_Classifier, tags: stage:2-extract)
  - W&B Artifact: sentence-pool:v<n>     (versioned, with full metadata)

Usage:
  conda activate Boilerplate_Classifier
  python scripts/01_extract.py
  # or, with a custom ECT path:
  python scripts/01_extract.py --ect-dir path/to/ECT --output data/raw/sentences.parquet
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts/ to path for gpu_runner import
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS.resolve()))

from gpu_runner import init_wandb_run  # noqa: E402

# Add src/ for project package
_PROJECT_ROOT = _SCRIPTS.parent
sys.path.insert(0, str((_PROJECT_ROOT / "src").resolve()))

from bpclassifier.extract import extract_all  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("01_extract")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def sentences_to_dataframe(sentences) -> pd.DataFrame:
    """Convert list[Sentence] → DataFrame with stable column order."""
    rows = [asdict(s) for s in sentences]
    df = pd.DataFrame(rows)
    column_order = [
        "sentence_id",
        "text",
        "ticker",
        "quarter",
        "quarter_num",
        "year",
        "call_date",
        "company",
        "section_type",
        "speaker_role",
        "speaker_name",
        "speaker_title",
        "position_in_block",
        "position_in_doc",
    ]
    return df[column_order]


def summarize(df: pd.DataFrame) -> dict:
    """Compute summary stats for W&B logging."""
    text_lens = df["text"].str.len()
    return {
        "n_sentences": int(len(df)),
        "n_transcripts": int(df["sentence_id"].str.rsplit("_", n=1).str[0].nunique()),
        "n_tickers": int(df["ticker"].nunique()),
        "n_unique_speakers": int(df["speaker_name"].dropna().nunique()),
        "section_prepared_remarks": int((df["section_type"] == "prepared_remarks").sum()),
        "section_question": int((df["section_type"] == "question").sum()),
        "section_answer": int((df["section_type"] == "answer").sum()),
        "len_mean": float(text_lens.mean()),
        "len_median": float(text_lens.median()),
        "len_p95": float(text_lens.quantile(0.95)),
        "len_max": int(text_lens.max()),
        "len_min": int(text_lens.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--ect-dir",
        type=Path,
        default=_PROJECT_ROOT / "ECT",
        help="Directory of raw .txt transcripts (default: ./ECT)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw" / "sentences.parquet",
        help="Output parquet path (default: data/raw/sentences.parquet)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B run (useful for quick local-only checks)",
    )
    args = parser.parse_args()

    if not args.ect_dir.is_dir():
        logger.error("ECT directory not found: %s", args.ect_dir)
        return 1

    logger.info("Extracting from %s", args.ect_dir)
    transcripts, sentences = extract_all(args.ect_dir)
    df = sentences_to_dataframe(sentences)
    stats = summarize(df)

    logger.info("Summary: %s", stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, engine="pyarrow", index=False)
    logger.info("Wrote %s (%d rows)", args.output, len(df))

    if args.no_wandb:
        logger.info("Skipping W&B logging (--no-wandb)")
        return 0

    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="extract-v0",
        job_type="extract",
        tags=["stage:2-extract", "purpose:extract"],
        config={
            "seed": SEED,
            "ect_dir": str(args.ect_dir),
            "min_sentence_chars": 40,
            "n_input_transcripts": len(transcripts),
        },
        notes="Stage 2: parse ECT/ transcripts and tokenize into sentence pool.",
    )

    run.summary.update(stats)

    # Top-N tables for easy dashboard inspection
    per_ticker = (
        df.groupby("ticker")
        .size()
        .reset_index(name="n_sentences")
        .sort_values("n_sentences", ascending=False)
    )
    per_section = df.groupby("section_type").size().reset_index(name="n_sentences")

    import wandb

    run.log(
        {
            "per_ticker": wandb.Table(dataframe=per_ticker),
            "per_section": wandb.Table(dataframe=per_section),
            "length_histogram": wandb.Histogram(df["text"].str.len().values),
        }
    )

    artifact = wandb.Artifact(
        name="sentence-pool",
        type="dataset",
        description="Stage 2 output: tokenized sentences from ECT/ with full metadata.",
        metadata=stats,
    )
    artifact.add_file(str(args.output), name="sentences.parquet")
    run.log_artifact(artifact)

    run.finish()
    logger.info("Stage 2 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
