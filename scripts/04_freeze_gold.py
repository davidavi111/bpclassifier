"""04_freeze_gold.py — Stage 3: Freeze gold labels from majority vote + meta-judge + audit.

Label hierarchy per sentence_id:
  1. David's hand-audit decision (audit_decisions.jsonl)  → source = "audit"
  2. Qwen3.5-27B meta-judge (meta_judge_outputs.parquet)  → source = "meta_judge"
     (only for the 332 disagreement cases not covered by 1)
  3. Unanimous majority vote (all 3 judges agree)         → source = "unanimous"

Inputs:
  data/interim/judge_outputs.parquet       (7,500-row long-form)
  data/interim/meta_judge_outputs.parquet  (332 disagreement rows)
  data/gold/audit_decisions.jsonl          (≤50 David decisions)

Outputs:
  data/gold/labeled.parquet                (2,500 rows: sentence_id, gold_label, source)
  W&B Artifact: gold-labels:v0
  W&B run: project=Boilerplate_Classifier,
           tags=[stage:3-gold, purpose:label, split:train]

Usage:
  conda activate Boilerplate_Classifier
  python scripts/04_freeze_gold.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.audit import find_disagreements, pivot_judgments  # noqa: E402
from gpu_runner import init_wandb_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("04_freeze_gold")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_INTERIM = _ROOT / "data" / "interim"
DATA_GOLD = _ROOT / "data" / "gold"
JUDGE_OUTPUTS_PATH = DATA_INTERIM / "judge_outputs.parquet"
META_JUDGE_PATH = DATA_INTERIM / "meta_judge_outputs.parquet"
AUDIT_DECISIONS_PATH = DATA_GOLD / "audit_decisions.jsonl"
OUTPUT_PATH = DATA_GOLD / "labeled.parquet"


def _load_audit_decisions(path: Path) -> dict[str, str]:
    """Load David's hand-audit decisions → {sentence_id: label}."""
    decisions: dict[str, str] = {}
    if not path.exists():
        return decisions
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                decisions[entry["sentence_id"]] = entry["david_label"]
            except (json.JSONDecodeError, KeyError):
                pass
    return decisions


def main() -> None:
    import wandb

    log.info("Loading judge outputs → %s", JUDGE_OUTPUTS_PATH)
    judge_outputs = pd.read_parquet(JUDGE_OUTPUTS_PATH)
    wide = pivot_judgments(judge_outputs)
    log.info("Wide form: %d rows", len(wide))

    disagreements = find_disagreements(wide)
    dis_ids = set(disagreements["sentence_id"])
    log.info("Disagreements: %d  Unanimous: %d", len(dis_ids), len(wide) - len(dis_ids))

    log.info("Loading meta-judge outputs → %s", META_JUDGE_PATH)
    meta_df = pd.read_parquet(META_JUDGE_PATH)
    meta_labels: dict[str, str] = dict(
        zip(meta_df["sentence_id"], meta_df["meta_label"], strict=False)
    )

    log.info("Loading audit decisions → %s", AUDIT_DECISIONS_PATH)
    audit_labels = _load_audit_decisions(AUDIT_DECISIONS_PATH)
    log.info("Audit decisions loaded: %d", len(audit_labels))

    # ── Build gold labels ──────────────────────────────────────────────────────
    records: list[dict] = []

    for _, row in wide.iterrows():
        sid = row["sentence_id"]
        if sid not in dis_ids:
            # Unanimous — all three agree; take any label col
            gold_label = row["anthropic_label"]
            source = "unanimous"
        elif sid in audit_labels:
            gold_label = audit_labels[sid]
            source = "audit"
        elif sid in meta_labels:
            gold_label = meta_labels[sid]
            source = "meta_judge"
        else:
            log.warning("No label found for disagreement %s — skipping", sid)
            continue

        records.append({"sentence_id": sid, "gold_label": gold_label, "source": source})

    gold_df = pd.DataFrame(records)
    log.info(
        "Gold labels: %d rows  (unanimous=%d, meta_judge=%d, audit=%d)",
        len(gold_df),
        (gold_df["source"] == "unanimous").sum(),
        (gold_df["source"] == "meta_judge").sum(),
        (gold_df["source"] == "audit").sum(),
    )

    dist = gold_df["gold_label"].value_counts().to_dict()
    log.info("Class balance: %s", dist)

    DATA_GOLD.mkdir(parents=True, exist_ok=True)
    gold_df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)

    # ── W&B run + artifact ─────────────────────────────────────────────────────
    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage3-freeze-gold",
        config={
            "n_total": len(gold_df),
            "n_unanimous": int((gold_df["source"] == "unanimous").sum()),
            "n_meta_judge": int((gold_df["source"] == "meta_judge").sum()),
            "n_audit": int((gold_df["source"] == "audit").sum()),
            "boilerplate": dist.get("boilerplate", 0),
            "substantive": dist.get("substantive", 0),
        },
        tags=["stage:3-gold", "purpose:label", "split:train"],
        job_type="label",
    )

    wandb.log(
        {
            "total_gold_labels": len(gold_df),
            "boilerplate_count": dist.get("boilerplate", 0),
            "substantive_count": dist.get("substantive", 0),
            "unanimous_count": int((gold_df["source"] == "unanimous").sum()),
            "meta_judge_count": int((gold_df["source"] == "meta_judge").sum()),
            "audit_count": int((gold_df["source"] == "audit").sum()),
        }
    )

    artifact = wandb.Artifact(
        "gold-labels",
        type="dataset",
        description=f"Frozen gold labels: {len(gold_df)} sentences",
        metadata={
            "n_unanimous": int((gold_df["source"] == "unanimous").sum()),
            "n_meta_judge": int((gold_df["source"] == "meta_judge").sum()),
            "n_audit": int((gold_df["source"] == "audit").sum()),
            "boilerplate": dist.get("boilerplate", 0),
            "substantive": dist.get("substantive", 0),
        },
    )
    artifact.add_file(str(OUTPUT_PATH))
    run.log_artifact(artifact)
    run.finish()
    log.info("Done.")


if __name__ == "__main__":
    main()
