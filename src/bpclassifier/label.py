"""label.py — Stage 3 gold-labeling helpers.

Pure utility layer: sampling, caching, prompt building, JSON parsing, and
the async judge orchestration loop (judge_fn is injected by the script so
that @weave.op-decorated callables stay out of this importable module).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────
SAMPLE_SIZE = 2_500
PROMPT_VERSION = 1
CONCURRENCY = 5
SEED = 42

ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3.1"
LLAMA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Anthropic Sonnet 4.6 token pricing (USD per token)
SONNET_IN_PRICE: float = 3.00 / 1_000_000
SONNET_OUT_PRICE: float = 15.00 / 1_000_000
SONNET_CACHE_WRITE_PRICE: float = 3.75 / 1_000_000
SONNET_CACHE_READ_PRICE: float = 0.30 / 1_000_000
ANTHROPIC_COST_LIMIT_USD: float = 9.0
COST_GUARD_WARMUP: int = 100  # calls before cost projection activates

REQUIRED_OUTPUT_COLUMNS: list[str] = [
    "sentence_id",
    "judge_name",
    "label",
    "reasoning",
    "latency_ms",
    "model_id",
    "prompt_version",
    "error",
]

VALID_LABELS = frozenset({"boilerplate", "substantive"})


# ── Prompt helpers ────────────────────────────────────────────────────────────


def build_user_prompt(row: Any) -> str:
    """Format a single sentence row as the judge user prompt."""
    return (
        f"Ticker: {row.ticker}\n"
        f"Quarter: {row.quarter}\n"
        f"Section: {row.section_type}\n"
        f"Sentence: {row.text}\n\n"
        "Return only the JSON object as specified in the rubric."
    )


def parse_json_response(text: str | None) -> dict | None:
    """
    Extract a valid label JSON object from model output.
    Strips markdown fences if present. Returns None on failure.
    """
    if text is None:
        return None
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            if inner.lower().startswith("json"):
                inner = inner[4:]
            text = inner.strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict) and obj.get("label") in VALID_LABELS:
        return obj
    return None


# ── Sample creation ───────────────────────────────────────────────────────────


def create_labeling_sample(
    sentences_df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Return n sentences stratified by company × section_type.

    Uses sklearn StratifiedShuffleSplit. Falls back to proportional groupby
    sample if any stratum has fewer than 2 members (can't stratify).
    """
    strat_col = sentences_df["company"].astype(str) + "|" + sentences_df["section_type"].astype(str)
    n_strata = strat_col.nunique()
    min_count = strat_col.value_counts().min()

    if min_count < 2 or n >= len(sentences_df) or n < n_strata:
        total = len(sentences_df)
        strat_sizes = strat_col.value_counts()
        pieces = []
        for stratum, count in strat_sizes.items():
            target = max(1, round(count / total * n))
            mask = strat_col == stratum
            pieces.append(sentences_df[mask].sample(min(target, count), random_state=seed))
        sample = pd.concat(pieces).head(n).reset_index(drop=True)
        return sample

    sample, _ = train_test_split(
        sentences_df,
        train_size=n,
        stratify=strat_col,
        random_state=seed,
    )
    return sample.reset_index(drop=True)


# ── Cache I/O ─────────────────────────────────────────────────────────────────


def load_cache(judge_name: str, cache_dir: Path) -> dict[str, dict]:
    """
    Load existing JSONL cache for a judge.
    Returns {sentence_id: result_dict}. Corrupt lines are skipped.
    """
    path = cache_dir / f"{judge_name}.jsonl"
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["sentence_id"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def append_to_cache(judge_name: str, entry: dict, cache_dir: Path) -> None:
    """Append a single result entry to the judge's JSONL cache file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{judge_name}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Async orchestration loop ──────────────────────────────────────────────────


async def run_one_judge(
    judge_name: str,
    rows: list[Any],
    cache: dict[str, dict],
    judge_fn: Callable[[str, str], Awaitable[dict]],
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Process all sentences for one judge, using JSONL cache where available.

    Args:
        judge_name: e.g. "anthropic", "google", "llama"
        rows: list of namedtuple-style rows (must have .sentence_id, .text, etc.)
        cache: pre-loaded {sentence_id: result_dict} from load_cache()
        judge_fn: async callable (sentence_id, user_prompt) → result dict.
                  Injected by the script (usually @weave.op decorated).
        cache_dir: directory for writing JSONL shards
        semaphore: per-judge concurrency limiter

    Returns:
        List of result dicts in the same order as rows.
    """
    results: list[dict | None] = [None] * len(rows)
    cache_hits = 0

    async def _process(idx: int, row: Any) -> None:
        nonlocal cache_hits
        sid = row.sentence_id
        if sid in cache and cache[sid].get("label") is not None and not cache[sid].get("error"):
            results[idx] = cache[sid]
            cache_hits += 1
            return
        async with semaphore:
            result = await judge_fn(sid, build_user_prompt(row))
            append_to_cache(judge_name, result, cache_dir)
            results[idx] = result

    await asyncio.gather(*(_process(i, row) for i, row in enumerate(rows)))
    new_calls = len(rows) - cache_hits
    log.info("%s: %d cache hits, %d new API calls", judge_name, cache_hits, new_calls)
    return results  # type: ignore[return-value]
