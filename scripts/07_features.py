"""07_features.py — Stage 4: Compute embeddings + regex feature flags for all 2,500 sentences.

Embedding model: all-mpnet-base-v2 (768-dim, sentence-transformers)
Regex flags: 27 binary features (0/1) per sentence

Inputs:
  data/interim/labeling_sample.parquet  (2,500 rows: sentence_id, text, …)

Outputs:
  data/interim/features.parquet  (2,500 rows: sentence_id + 768 embed dims + 27 flags)
  W&B Artifact: embeddings:v0
  W&B run: project=Boilerplate_Classifier,
           tags=[stage:4-features, purpose:extract, model:all-mpnet-base-v2]

Resume-safe: if features.parquet already exists and has the right row count, skip re-encoding.

Usage:
  conda activate Boilerplate_Classifier
  python scripts/07_features.py
"""

from __future__ import annotations

import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
log = logging.getLogger("07_features")

SAMPLE_PATH = _ROOT / "data" / "interim" / "labeling_sample.parquet"
OUTPUT_PATH = _ROOT / "data" / "interim" / "features.parquet"
EMBED_MODEL = "all-mpnet-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 64


# ── Regex flag definitions ────────────────────────────────────────────────────


def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_FLAGS: list[tuple[str, re.Pattern | None, callable]] = [
    # -- Boilerplate signal flags --
    (
        "flag_safe_harbor",
        _re(
            r"forward.looking|safe.harbor|actual results may differ|future results|"
            r"actual results to differ"
        ),
        None,
    ),
    (
        "flag_sec_filing_ref",
        _re(
            r"\b(10-K|10-Q|8-K|annual report|quarterly report|SEC filing|"
            r"Form 10|proxy statement)\b"
        ),
        None,
    ),
    (
        "flag_please_see_filing",
        _re(
            r"please (refer|see|review).{0,60}(press release|presentation|website|"
            r"filing|SEC|investor relations)"
        ),
        None,
    ),
    (
        "flag_risk_factor_ref",
        _re(r"\brisk factors?\b|\buncertainties\b|\brisks related to\b"),
        None,
    ),
    (
        "flag_legal_disclaimer",
        _re(
            r"should not be relied|no guarantee|no assurance|no warranty|"
            r"cannot guarantee|does not guarantee"
        ),
        None,
    ),
    (
        "flag_operator_phrase",
        _re(
            r"^(operator[:,]?\s)|please (go ahead|hold|stand by|press)|"
            r"thank you[,.]?\s*(operator|next caller)|"
            r"your (line is|lines are) (open|now open)"
        ),
        None,
    ),
    (
        "flag_operator_closing",
        _re(
            r"(that|this) (concludes|ends|completes|concludes today).{0,40}(call|conference)|"
            r"conference (has|is now) (ended|concluded|complete)|"
            r"you may now disconnect"
        ),
        None,
    ),
    (
        "flag_thanks_greeting",
        _re(
            r"^(thank you|thanks|good (morning|afternoon|evening)|"
            r"welcome (everyone|to|back)|hello everyone|hi everyone)"
        ),
        None,
    ),
    (
        "flag_replay_info",
        _re(
            r"\b(replay|recording|playback|archive|webcast)\b.{0,60}"
            r"(available|access|listen|view|watch)"
        ),
        None,
    ),
    (
        "flag_speaker_handoff",
        _re(
            r"(i('ll| will)|let me) now (turn|hand|pass|give).{0,30}"
            r"(call|floor|line|over)|joining us today (is|are)|"
            r"let me (introduce|hand it to|turn it over)"
        ),
        None,
    ),
    # -- Substantive signal flags --
    (
        "flag_dollar_amount",
        _re(r"\$[\d,]+(\.\d+)?\s*(million|billion|thousand|M\b|B\b|K\b)"),
        None,
    ),
    (
        "flag_percent_change",
        _re(r"\d+(\.\d+)?\s*%"),
        None,
    ),
    (
        "flag_yoy_comparison",
        _re(
            r"year.over.year|year-over-year|compared to.{0,20}(prior|last|same)\s+year|"
            r"versus.{0,20}prior year|prior.year period"
        ),
        None,
    ),
    (
        "flag_qoq_comparison",
        _re(
            r"quarter.over.quarter|quarter-over-quarter|sequentially|"
            r"from.{0,20}(prior|last|previous)\s+quarter|versus.{0,20}prior quarter"
        ),
        None,
    ),
    (
        "flag_guidance_language",
        _re(
            r"\bguidance\b|\boutlook\b|\bforecast\b|"
            r"(expect|anticipate).{0,40}(fiscal|quarter|year|full.year)"
        ),
        None,
    ),
    (
        "flag_per_share_metric",
        _re(r"\bper share\b|\bEPS\b|\bearnings per share\b|\bdiluted\b|\bbasic\b.{0,20}share"),
        None,
    ),
    (
        "flag_gaap_nongaap",
        _re(
            r"non.GAAP|non GAAP|adjusted EBITDA|adjusted earnings|adjusted revenue|"
            r"adjusted (gross|operating|net)|reconciliation"
        ),
        None,
    ),
    (
        "flag_organic_growth",
        _re(r"\borganic (growth|revenue|sales)\b"),
        None,
    ),
    (
        "flag_strategic_language",
        _re(r"\bstrategic\b|\binitiative\b|\bpriority\b|\blong.term\b|\binvest(ing|ment)?\b"),
        None,
    ),
    (
        "flag_ma_language",
        _re(
            r"\bacquisition\b|\bmerger\b|\bdivest(iture)?\b|\bjoint venture\b|"
            r"\bstrategic combination\b|\btransaction\b.{0,20}(close|complet|acqui)"
        ),
        None,
    ),
    (
        "flag_cost_reduction",
        _re(
            r"cost (reduction|saving|efficiency|optimization|structure)|"
            r"\brestructuring\b|\bheadcount\b|\blayoff\b|\bworkforce reduction\b"
        ),
        None,
    ),
    (
        "flag_market_share",
        _re(r"\bmarket share\b|\bcompetitive position\b|\bgained share\b|\bmarket leadership\b"),
        None,
    ),
    (
        "flag_product_launch",
        _re(r"\blaunch(ed|ing)?\b|\bintroduc(ed|ing)\b|\bnew product\b|\breleased\b"),
        None,
    ),
    # -- Structural / length flags --
    (
        "flag_short_sentence",
        None,
        lambda text: int(len(text) < 60),
    ),
    (
        "flag_all_caps_ratio",
        None,
        lambda text: int(
            (sum(1 for w in text.split() if w.isupper() and len(w) > 1) / max(len(text.split()), 1))
            > 0.30
        ),
    ),
    (
        "flag_question_mark",
        None,
        lambda text: int(text.rstrip().endswith("?")),
    ),
    (
        "flag_numeric_density",
        None,
        lambda text: int(
            sum(bool(re.search(r"\d", w)) for w in text.split()) / max(len(text.split()), 1) > 0.20
        ),
    ),
]


def compute_flags(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for text in texts:
        row = {}
        for name, pattern, fn in _FLAGS:
            if pattern is not None:
                row[name] = int(bool(pattern.search(text)))
            else:
                row[name] = fn(text)
        rows.append(row)
    return pd.DataFrame(rows, index=texts.index)


def main() -> None:
    import wandb
    from sentence_transformers import SentenceTransformer

    log.info("Loading sample → %s", SAMPLE_PATH)
    sample = pd.read_parquet(SAMPLE_PATH)[["sentence_id", "text"]]
    log.info("Loaded %d sentences", len(sample))

    # ── Embeddings ────────────────────────────────────────────────────────────
    log.info("Loading embedding model: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    log.info("Encoding sentences (batch_size=%d)…", BATCH_SIZE)
    embeddings = model.encode(
        sample["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    log.info("Embeddings shape: %s", embeddings.shape)

    embed_df = pd.DataFrame(
        embeddings,
        columns=[f"embed_{i}" for i in range(EMBED_DIM)],
        index=sample.index,
    )

    # ── Regex flags ───────────────────────────────────────────────────────────
    log.info("Computing %d regex feature flags…", len(_FLAGS))
    flag_df = compute_flags(sample["text"])
    flag_names = [name for name, _, _ in _FLAGS]
    log.info(
        "Flag means:\n%s",
        flag_df[flag_names].mean().round(3).to_string(),
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    features = pd.concat(
        [
            sample[["sentence_id"]].reset_index(drop=True),
            embed_df.reset_index(drop=True),
            flag_df.reset_index(drop=True),
        ],
        axis=1,
    )
    features.to_parquet(OUTPUT_PATH, index=False)
    log.info(
        "Saved features → %s  (%d rows, %d cols)", OUTPUT_PATH, len(features), len(features.columns)
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage4-features",
        config={
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "n_flags": len(_FLAGS),
            "n_sentences": len(sample),
            "batch_size": BATCH_SIZE,
            "normalize_embeddings": True,
            "seed": 42,
        },
        tags=["stage:4-features", "purpose:extract", f"model:{EMBED_MODEL}"],
        job_type="extract",
    )

    wandb.log(
        {
            "n_sentences": len(features),
            "embed_dim": EMBED_DIM,
            "n_flag_cols": len(flag_names),
            **{f"flag_mean/{n}": float(flag_df[n].mean()) for n in flag_names},
        }
    )

    artifact = wandb.Artifact(
        "embeddings",
        type="dataset",
        description=f"{EMBED_MODEL} embeddings + {len(_FLAGS)} regex flags for {len(features)} sentences",
        metadata={"model": EMBED_MODEL, "embed_dim": EMBED_DIM, "n_flags": len(_FLAGS)},
    )
    artifact.add_file(str(OUTPUT_PATH))
    run.log_artifact(artifact)
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
