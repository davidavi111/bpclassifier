"""03_audit.py — Stage 3: Human review of 50 LLM-judge disagreements.

Inputs:
  data/interim/judge_outputs.parquet     (7,500 rows — Stage 3 labeling output)
  data/interim/labeling_sample.parquet   (2,500-row sample with full text)
  data/gold/audit_decisions.jsonl        (resume cache — created on first save)

Outputs:
  data/gold/audit_sample.parquet         (50-row stratified disagreement sample)
  data/gold/audit_decisions.jsonl        (JSONL decision log, appended per review)

Usage:
  conda activate Boilerplate_Classifier
  python scripts/03_audit.py

Controls:
  b  → boilerplate
  s  → substantive
  k  → skip (uncertain; excluded from gold labels)
  q  → quit and save progress (resume next time)

The script is safe to interrupt and restart: completed decisions are cached
in audit_decisions.jsonl and skipped on the next run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.audit import (  # noqa: E402
    find_disagreements,
    load_decisions,
    pivot_judgments,
    save_decision,
    stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("03_audit")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_INTERIM = _ROOT / "data" / "interim"
DATA_GOLD = _ROOT / "data" / "gold"
JUDGE_OUTPUTS_PATH = DATA_INTERIM / "judge_outputs.parquet"
SAMPLE_PATH = DATA_INTERIM / "labeling_sample.parquet"
AUDIT_SAMPLE_PATH = DATA_GOLD / "audit_sample.parquet"
DECISIONS_PATH = DATA_GOLD / "audit_decisions.jsonl"

AUDIT_N = 50
SEED = 42


# ── Display helper ────────────────────────────────────────────────────────────


def print_case(
    idx: int,
    total: int,
    row: pd.Series,
    sentence_text: str,
) -> None:
    """Print one disagreement case to stdout."""
    separator = "─" * 70
    print(f"\n{separator}")
    print(f"  Case {idx}/{total}  │  sentence_id: {row['sentence_id']}")
    print(separator)
    print(f"  Ticker  : {row.get('ticker', '?')}")
    print(f"  Quarter : {row.get('quarter', '?')}")
    print(f"  Section : {row.get('section_type', '?')}")
    print(f"  Minority: {row.get('minority_judge', '?')}")
    print()
    print(f"  TEXT: {sentence_text}")
    print()
    print(f"  Anthropic : {row.get('anthropic_label', '?')}")
    print(f"    reason  : {str(row.get('anthropic_reasoning', ''))[:120]}")
    print(f"  DeepSeek  : {row.get('deepseek_label', '?')}")
    print(f"    reason  : {str(row.get('deepseek_reasoning', ''))[:120]}")
    print(f"  Llama     : {row.get('llama_label', '?')}")
    print(f"    reason  : {str(row.get('llama_reasoning', ''))[:120]}")
    print()
    print("  [b] boilerplate   [s] substantive   [k] skip   [q] quit")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_GOLD.mkdir(parents=True, exist_ok=True)

    # Build or load audit sample
    if AUDIT_SAMPLE_PATH.exists():
        log.info("Loading existing audit sample from %s", AUDIT_SAMPLE_PATH)
        audit_sample = pd.read_parquet(AUDIT_SAMPLE_PATH)
    else:
        log.info("Building audit sample from judge outputs…")
        judge_outputs = pd.read_parquet(JUDGE_OUTPUTS_PATH)
        wide = pivot_judgments(judge_outputs)
        disagreements = find_disagreements(wide)
        log.info(
            "Found %d disagreements out of %d sentences (%.1f%%)",
            len(disagreements),
            len(wide),
            100 * len(disagreements) / max(len(wide), 1),
        )
        audit_sample = stratified_sample(disagreements, n=AUDIT_N, seed=SEED)
        audit_sample.to_parquet(AUDIT_SAMPLE_PATH, index=False)
        log.info("Audit sample saved → %s  (%d rows)", AUDIT_SAMPLE_PATH, len(audit_sample))

    # Load sentence text for display
    sentences_df = pd.read_parquet(SAMPLE_PATH)[["sentence_id", "text", "ticker", "quarter"]]
    text_map: dict[str, str] = dict(
        zip(sentences_df["sentence_id"], sentences_df["text"], strict=False)
    )
    ticker_map: dict[str, str] = dict(
        zip(sentences_df["sentence_id"], sentences_df["ticker"], strict=False)
    )
    quarter_map: dict[str, str] = dict(
        zip(sentences_df["sentence_id"], sentences_df["quarter"], strict=False)
    )

    # Merge ticker/quarter into audit_sample if not already present
    for col, mapping in (("ticker", ticker_map), ("quarter", quarter_map)):
        if col not in audit_sample.columns:
            audit_sample[col] = audit_sample["sentence_id"].map(mapping)

    # Load existing decisions (resume support)
    decisions = load_decisions(DECISIONS_PATH)
    done_ids = set(decisions.keys())
    log.info("%d / %d decisions already recorded", len(done_ids), len(audit_sample))

    # Filter to remaining cases
    remaining = audit_sample[~audit_sample["sentence_id"].isin(done_ids)].reset_index(drop=True)
    total_remaining = len(remaining)

    if total_remaining == 0:
        print("\nAll cases reviewed. Run scripts/04_freeze_gold.py to freeze the gold labels.")
        return

    print(f"\nStarting audit: {total_remaining} cases remaining (of {AUDIT_N} total).")
    print("Enter b / s / k / q after each case.\n")

    reviewed_this_session = 0
    skipped_this_session = 0

    for _, row in remaining.iterrows():
        sid = row["sentence_id"]
        case_num = len(done_ids) + reviewed_this_session + skipped_this_session + 1
        sentence_text = text_map.get(sid, "[text not found]")
        print_case(case_num, AUDIT_N, row, sentence_text)

        while True:
            try:
                key = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"

            if key in ("b", "s", "k", "q"):
                break
            print("  Invalid key. Enter b, s, k, or q.")

        if key == "q":
            print(
                f"\nQuitting. Reviewed {reviewed_this_session} new cases this session "
                f"({skipped_this_session} skipped). Resume anytime."
            )
            break

        if key == "k":
            skipped_this_session += 1
            print("  → skipped")
            continue

        david_label = "boilerplate" if key == "b" else "substantive"
        judge_labels = {
            "anthropic": row.get("anthropic_label"),
            "deepseek": row.get("deepseek_label"),
            "llama": row.get("llama_label"),
        }
        save_decision(sid, david_label, judge_labels, DECISIONS_PATH)
        reviewed_this_session += 1
        print(f"  → {david_label}")

    # Session summary
    total_done = len(load_decisions(DECISIONS_PATH))
    print(
        f"\n{'─' * 70}\n"
        f"Session summary: {reviewed_this_session} labeled, {skipped_this_session} skipped.\n"
        f"Total decisions saved: {total_done} / {AUDIT_N}.\n"
        f"Decisions file: {DECISIONS_PATH}\n"
    )
    if total_done >= AUDIT_N:
        print("All cases reviewed! Run scripts/04_freeze_gold.py next.")


if __name__ == "__main__":
    main()
