"""audit.py — Stage 3 disagreement-audit helpers.

Pure utility layer: pivot judgments, detect disagreements, stratified sampling,
and JSONL decision persistence. No I/O side effects except save_decision.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

VALID_LABELS = frozenset({"boilerplate", "substantive"})
JUDGES = ("anthropic", "deepseek", "llama")


# ── Pivot & disagreement detection ───────────────────────────────────────────


def pivot_judgments(judge_outputs: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long-form judge_outputs (7 500 rows) into wide form (2 500 rows).

    Returns a DataFrame with columns:
        sentence_id,
        anthropic_label, deepseek_label, llama_label,
        anthropic_reasoning, deepseek_reasoning, llama_reasoning
    Only rows where all three judges produced a valid label are kept.
    """
    labels = (
        judge_outputs[judge_outputs["judge_name"].isin(JUDGES)][
            ["sentence_id", "judge_name", "label", "reasoning"]
        ]
        .pivot(index="sentence_id", columns="judge_name", values=["label", "reasoning"])
        .reset_index()
    )
    labels.columns = ["sentence_id" if c[1] == "" else f"{c[1]}_{c[0]}" for c in labels.columns]
    # Keep only complete rows (all three judges labeled successfully)
    label_cols = [f"{j}_label" for j in JUDGES]
    labels = labels.dropna(subset=label_cols).reset_index(drop=True)
    return labels


def find_disagreements(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows where the three judges do not unanimously agree.

    Adds a ``minority_judge`` column: the name of the judge whose label
    differs from the other two (e.g. "anthropic").  For a 3-way split
    (impossible with binary labels) the value is "all".
    """
    label_cols = [f"{j}_label" for j in JUDGES]
    mask = wide_df[label_cols].nunique(axis=1) > 1
    dis = wide_df[mask].copy().reset_index(drop=True)

    def _minority(row: pd.Series) -> str:
        counts: dict[str, int] = {}
        for j in JUDGES:
            lbl = row[f"{j}_label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        majority_label = max(counts, key=lambda k: counts[k])
        minority_judges = [j for j in JUDGES if row[f"{j}_label"] != majority_label]
        if len(minority_judges) == 1:
            return minority_judges[0]
        return "all"

    dis["minority_judge"] = dis.apply(_minority, axis=1)
    return dis


# ── Stratified audit sample ───────────────────────────────────────────────────


def stratified_sample(
    disagreements: pd.DataFrame,
    n: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draw ~n/3 rows from each minority_judge group (anthropic / deepseek / llama).

    When a group has fewer rows than its target, takes all available and
    redistributes the shortfall proportionally to remaining groups.
    Returns at most n rows.
    """
    groups = {j: disagreements[disagreements["minority_judge"] == j] for j in JUDGES}
    targets = {j: n // len(JUDGES) for j in JUDGES}
    # Distribute remainder to first groups
    remainder = n - sum(targets.values())
    for i, j in enumerate(JUDGES):
        if i < remainder:
            targets[j] += 1

    pieces: list[pd.DataFrame] = []
    shortfall = 0
    deferred: list[str] = []

    for j in JUDGES:
        grp = groups[j]
        t = targets[j]
        if len(grp) <= t:
            pieces.append(grp)
            shortfall += t - len(grp)
        else:
            deferred.append(j)
            pieces.append(grp.sample(t, random_state=seed))

    if shortfall > 0 and deferred:
        per_extra = shortfall // len(deferred)
        for j in deferred:
            grp = groups[j]
            already = targets[j]
            extra = min(per_extra, len(grp) - already)
            if extra > 0:
                pieces.append(grp.sample(already + extra, random_state=seed).tail(extra))

    sample = (
        pd.concat(pieces).drop_duplicates(subset=["sentence_id"]).head(n).reset_index(drop=True)
    )
    return sample


# ── Decision persistence ──────────────────────────────────────────────────────


def load_decisions(path: Path) -> dict[str, dict]:
    """
    Load existing audit decisions from a JSONL file.
    Returns {sentence_id: decision_dict}. Corrupt lines are skipped.
    """
    decisions: dict[str, dict] = {}
    if not path.exists():
        return decisions
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                decisions[entry["sentence_id"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass
    return decisions


def save_decision(
    sentence_id: str,
    david_label: str,
    judge_labels: dict[str, str],
    path: Path,
) -> None:
    """Append one audit decision to a JSONL file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "sentence_id": sentence_id,
        "david_label": david_label,
        "timestamp_iso": datetime.now(tz=UTC).isoformat(),
        "original_judge_labels": judge_labels,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
