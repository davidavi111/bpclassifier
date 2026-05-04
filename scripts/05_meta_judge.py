"""05_meta_judge.py — Stage 3: Meta-judge the 332 disagreement cases via Qwen3.5-27B.

Inputs:
  data/interim/judge_outputs.parquet      (7,500-row long form from Stage 3)
  data/interim/labeling_sample.parquet    (2,500-row sample; needed for text + section_type)
  data/gold/rubric.md                     (fed verbatim to the meta-judge as system prompt)

Outputs:
  data/interim/judge_cache/qwen_meta.jsonl   (resume cache, keyed by sentence_id)
  data/interim/meta_judge_outputs.parquet    (332 rows: sentence_id, meta_label,
                                              meta_reasoning, model_id)
  W&B run: project=Boilerplate_Classifier,
            tags=[stage:3-gold, purpose:label, model:qwen-meta, split:disagreements]
  W&B Artifact: meta-judge-outputs:v<n>

Usage:
  conda activate Boilerplate_Classifier
  python scripts/05_meta_judge.py

Resume-safe: completed calls are cached in qwen_meta.jsonl and skipped on restart.
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import weave

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from api_clients import wandb_inference_client  # noqa: E402
from bpclassifier.audit import find_disagreements, pivot_judgments  # noqa: E402
from bpclassifier.label import append_to_cache, load_cache, parse_json_response  # noqa: E402
from gpu_runner import init_wandb_run  # noqa: E402

random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("05_meta_judge")

# ── Constants ──────────────────────────────────────────────────────────────────
META_MODEL = "Qwen/Qwen3.5-27B"
CACHE_KEY = "qwen_meta"
CONCURRENCY = 5

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_INTERIM = _ROOT / "data" / "interim"
CACHE_DIR = DATA_INTERIM / "judge_cache"
JUDGE_OUTPUTS_PATH = DATA_INTERIM / "judge_outputs.parquet"
SAMPLE_PATH = DATA_INTERIM / "labeling_sample.parquet"
OUTPUT_PATH = DATA_INTERIM / "meta_judge_outputs.parquet"
RUBRIC_PATH = _ROOT / "data" / "gold" / "rubric.md"


# ── Prompt builder ─────────────────────────────────────────────────────────────


def build_meta_prompt(row: Any, wide_row: pd.Series) -> str:
    """Build the meta-judge user prompt for one disagreement row."""
    judge_block = "\n".join(
        f"  {judge.upper()}: label={wide_row[f'{judge}_label']!r}  "
        f"reasoning={wide_row[f'{judge}_reasoning']!r}"
        for judge in ("anthropic", "deepseek", "llama")
    )
    return (
        f"Ticker: {row.ticker}\n"
        f"Quarter: {row.quarter}\n"
        f"Section: {row.section_type}\n"
        f"Sentence: {row.text}\n\n"
        f"Three judges disagreed on this sentence:\n{judge_block}\n\n"
        "Apply the rubric strictly and pick the single correct label. "
        "Return only the JSON object as specified in the rubric."
    )


# ── Weave-traced API call ──────────────────────────────────────────────────────


@weave.op()
async def _call_qwen_meta(
    sentence_id: str,
    user_prompt: str,
    rubric: str,
    client: Any,
) -> dict:
    """Call Qwen3.5-27B on W&B Inference as meta-judge; retry once on bad JSON."""
    label = reasoning = raw = None
    error = None
    latency_ms = 0

    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=META_MODEL,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if attempt == 1:
                error = f"{type(exc).__name__}: {exc}"
            continue

        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed = parse_json_response(raw)
        if parsed:
            label = parsed["label"]
            reasoning = parsed.get("reasoning", "")
            error = None
            break
        if attempt == 0:
            log.warning("sentence %s: malformed Qwen meta-judge JSON, retrying", sentence_id)

    if label is None and error is None:
        error = f"Malformed JSON after 2 attempts. Raw: {(raw or '')[:200]!r}"

    return {
        "sentence_id": sentence_id,
        "meta_label": label,
        "meta_reasoning": reasoning,
        "model_id": META_MODEL,
        "latency_ms": latency_ms,
        "error": error,
    }


# ── Async orchestration loop ───────────────────────────────────────────────────


async def _run_meta_judge(
    rows: list[tuple[Any, pd.Series]],
    cache: dict[str, dict],
    client: Any,
    rubric: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Process all disagreement rows, skipping cache hits."""
    results: list[dict | None] = [None] * len(rows)
    cache_hits = 0

    async def _process(idx: int, sample_row: Any, wide_row: pd.Series) -> None:
        nonlocal cache_hits
        sid = sample_row.sentence_id
        cached = cache.get(sid)
        if cached and cached.get("meta_label") is not None and not cached.get("error"):
            results[idx] = cached
            cache_hits += 1
            return
        async with semaphore:
            user_prompt = build_meta_prompt(sample_row, wide_row)
            result = await _call_qwen_meta(sid, user_prompt, rubric, client)
            append_to_cache(CACHE_KEY, result, CACHE_DIR)
            results[idx] = result

    await asyncio.gather(*(_process(i, sr, wr) for i, (sr, wr) in enumerate(rows)))
    log.info("meta-judge: %d cache hits, %d new API calls", cache_hits, len(rows) - cache_hits)
    return results  # type: ignore[return-value]


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    weave.init("Boilerplate_Classifier")

    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Load inputs and derive the 332 disagreement rows
    log.info("Loading judge outputs → %s", JUDGE_OUTPUTS_PATH)
    judge_outputs = pd.read_parquet(JUDGE_OUTPUTS_PATH)
    wide = pivot_judgments(judge_outputs)
    disagreements = find_disagreements(wide)
    log.info("Disagreement rows: %d", len(disagreements))

    log.info("Loading labeling sample → %s", SAMPLE_PATH)
    sample_df = pd.read_parquet(SAMPLE_PATH)
    # Index sample by sentence_id for O(1) lookup
    sample_idx = sample_df.set_index("sentence_id")

    # Build aligned (sample_row, wide_row) pairs — only disagreements
    dis_ids = list(disagreements["sentence_id"])
    wide_idx = disagreements.set_index("sentence_id")
    rows: list[tuple[Any, pd.Series]] = []
    for sid in dis_ids:
        sample_row = sample_idx.loc[sid]
        wide_row = wide_idx.loc[sid]
        # namedtuple-style access needed by build_meta_prompt; wrap in a simple object
        rows.append((_SampleRow(sid, sample_row), wide_row))

    # W&B run
    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage3-meta-judge-qwen",
        config={
            "model": META_MODEL,
            "n_disagreements": len(rows),
            "concurrency": CONCURRENCY,
            "seed": 42,
            "max_tokens": 512,
            "temperature": 0,
        },
        tags=["stage:3-gold", "purpose:label", "model:qwen-meta", "split:disagreements"],
        job_type="label",
    )

    client = wandb_inference_client()
    cache = load_cache(CACHE_KEY, CACHE_DIR)
    log.info("Cache: %d existing entries (will skip)", len(cache))

    semaphore = asyncio.Semaphore(CONCURRENCY)

    log.info("Dispatching %d disagreement sentences to %s…", len(rows), META_MODEL)
    results = await _run_meta_judge(rows, cache, client, rubric, semaphore)

    # Assemble output
    output_df = pd.DataFrame(
        [
            {
                "sentence_id": r["sentence_id"],
                "meta_label": r["meta_label"],
                "meta_reasoning": r["meta_reasoning"],
                "model_id": r["model_id"],
            }
            for r in results
        ]
    )
    output_df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved meta-judge outputs → %s  (%d rows)", OUTPUT_PATH, len(output_df))

    # W&B metrics + artifact
    import wandb

    n_labeled = int(output_df["meta_label"].notna().sum())
    n_errors = int(pd.DataFrame(results)["error"].notna().sum())
    dist = output_df["meta_label"].value_counts().to_dict()

    wandb.log(
        {
            "total_meta_judgments": len(output_df),
            "labeled_count": n_labeled,
            "error_count": n_errors,
            "meta_boilerplate": dist.get("boilerplate", 0),
            "meta_substantive": dist.get("substantive", 0),
        }
    )

    artifact = wandb.Artifact(
        "meta-judge-outputs",
        type="dataset",
        description=f"Qwen3.5-27B meta-judge over {len(output_df)} disagreement cases",
        metadata={"model": META_MODEL, "n_disagreements": len(output_df)},
    )
    artifact.add_file(str(OUTPUT_PATH))
    run.log_artifact(artifact)
    run.finish()

    log.info("Done. Labeled: %d  Errors: %d", n_labeled, n_errors)


# ── Thin wrapper so build_meta_prompt can use attribute access ─────────────────


class _SampleRow:
    """Wrap a pandas Series as an attribute-accessible object for prompt building."""

    __slots__ = ("sentence_id", "ticker", "quarter", "section_type", "text")

    def __init__(self, sentence_id: str, series: pd.Series) -> None:
        self.sentence_id = sentence_id
        self.ticker = series["ticker"]
        self.quarter = series["quarter"]
        self.section_type = series["section_type"]
        self.text = series["text"]


if __name__ == "__main__":
    asyncio.run(main())
