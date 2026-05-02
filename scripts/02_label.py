"""02_label.py — Stage 3: Gold-label 2,500 sentences via 3 LLM judges.

Inputs:
  data/raw/sentences.parquet          (54,923 rows — Stage 2 output)
  data/gold/rubric.md                 (loaded verbatim as judge system prompt)

Outputs:
  data/interim/labeling_sample.parquet   (2,500-row reproducible sample)
  data/interim/judge_cache/{judge}.jsonl (per-judge JSONL resume cache)
  data/interim/judge_outputs.parquet     (7,500 rows: 2,500 × 3 judges)
  W&B run: project=Boilerplate_Classifier, tags=[stage:3-gold, purpose:label, split:none]
  W&B Artifact: judge-outputs:v<n>

Usage:
  conda activate Boilerplate_Classifier
  python scripts/02_label.py

The script is safe to interrupt and restart: completed calls are cached
in JSONL shards and skipped on the next run.
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

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from api_clients import anthropic_client, google_client, wandb_inference_client  # noqa: E402
from bpclassifier.label import (  # noqa: E402
    ANTHROPIC_COST_LIMIT_USD,
    ANTHROPIC_MODEL,
    CONCURRENCY,
    GOOGLE_MODEL,
    LLAMA_MODEL,
    PROMPT_VERSION,
    REQUIRED_OUTPUT_COLUMNS,
    SAMPLE_SIZE,
    SEED,
    SONNET_CACHE_READ_PRICE,
    SONNET_CACHE_WRITE_PRICE,
    SONNET_IN_PRICE,
    SONNET_OUT_PRICE,
    create_labeling_sample,
    load_cache,
    parse_json_response,
    run_one_judge,
)
from gpu_runner import init_wandb_run  # noqa: E402

random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("02_label")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW = _ROOT / "data" / "raw"
DATA_INTERIM = _ROOT / "data" / "interim"
CACHE_DIR = DATA_INTERIM / "judge_cache"
SAMPLE_PATH = DATA_INTERIM / "labeling_sample.parquet"
OUTPUT_PATH = DATA_INTERIM / "judge_outputs.parquet"
RUBRIC_PATH = _ROOT / "data" / "gold" / "rubric.md"


# ── Weave-traced judge call functions ─────────────────────────────────────────
# Each wraps one sync API call in asyncio.to_thread and retries once on bad JSON.
# @weave.op() is applied here (not in the package module) so the package stays
# importable without a live Weave session.


@weave.op()
async def _call_anthropic(
    sentence_id: str,
    user_prompt: str,
    rubric: str,
    client: Any,
    cost_tracker: dict,
) -> dict:
    """Call Anthropic Claude with prompt caching; retry once on malformed JSON."""
    # System message with cache_control caches the rubric after the first call
    system = [
        {
            "type": "text",
            "text": rubric,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    label = reasoning = raw = None
    error = None
    latency_ms = 0

    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                client.messages.create,
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if attempt == 1:
                error = f"{type(exc).__name__}: {exc}"
            continue

        latency_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.content[0].text
        usage = resp.usage

        # Dollar cost for this call
        call_cost = (
            getattr(usage, "cache_creation_input_tokens", 0) * SONNET_CACHE_WRITE_PRICE
            + getattr(usage, "cache_read_input_tokens", 0) * SONNET_CACHE_READ_PRICE
            + usage.input_tokens * SONNET_IN_PRICE
            + usage.output_tokens * SONNET_OUT_PRICE
        )
        cost_tracker["total_usd"] += call_cost
        cost_tracker["calls"] += 1

        # Project remaining spend and abort before the bill gets too large
        remaining = cost_tracker["total_sentences"] - cost_tracker["calls"]
        if remaining > 0 and cost_tracker["calls"] > 0:
            avg_cost = cost_tracker["total_usd"] / cost_tracker["calls"]
            projected = cost_tracker["total_usd"] + avg_cost * remaining
            if projected > ANTHROPIC_COST_LIMIT_USD:
                raise OSError(
                    f"Projected Anthropic spend ${projected:.2f} exceeds the "
                    f"${ANTHROPIC_COST_LIMIT_USD} limit. "
                    f"Spent ${cost_tracker['total_usd']:.4f} over "
                    f"{cost_tracker['calls']} calls. Aborting. "
                    f"Set a higher limit or reduce sample size."
                )

        parsed = parse_json_response(raw)
        if parsed:
            label = parsed["label"]
            reasoning = parsed.get("reasoning", "")
            error = None
            break
        if attempt == 0:
            log.warning("sentence %s: malformed Anthropic JSON, retrying", sentence_id)

    if label is None and error is None:
        error = f"Malformed JSON after 2 attempts. Raw: {(raw or '')[:200]!r}"

    return {
        "sentence_id": sentence_id,
        "judge_name": "anthropic",
        "label": label,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "model_id": ANTHROPIC_MODEL,
        "prompt_version": PROMPT_VERSION,
        "error": error,
    }


@weave.op()
async def _call_google(
    sentence_id: str,
    user_prompt: str,
    rubric: str,
    client: Any,
) -> dict:
    """Call Google Gemini; retry once on malformed JSON."""
    from google.genai import types as genai_types  # type: ignore[import]

    label = reasoning = raw = None
    error = None
    latency_ms = 0

    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=GOOGLE_MODEL,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=rubric,
                    max_output_tokens=256,
                    response_mime_type="application/json",
                ),
            )
            raw = resp.text
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
            log.warning("sentence %s: malformed Gemini JSON, retrying", sentence_id)

    if label is None and error is None:
        error = f"Malformed JSON after 2 attempts. Raw: {(raw or '')[:200]!r}"

    return {
        "sentence_id": sentence_id,
        "judge_name": "google",
        "label": label,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "model_id": GOOGLE_MODEL,
        "prompt_version": PROMPT_VERSION,
        "error": error,
    }


@weave.op()
async def _call_llama(
    sentence_id: str,
    user_prompt: str,
    rubric: str,
    client: Any,
) -> dict:
    """Call W&B Inference Llama via OpenAI-compatible API; retry once on bad JSON."""
    label = reasoning = raw = None
    error = None
    latency_ms = 0

    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=LLAMA_MODEL,
                messages=[
                    {"role": "system", "content": rubric},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
                response_format={"type": "json_object"},
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
            log.warning("sentence %s: malformed Llama JSON, retrying", sentence_id)

    if label is None and error is None:
        error = f"Malformed JSON after 2 attempts. Raw: {(raw or '')[:200]!r}"

    return {
        "sentence_id": sentence_id,
        "judge_name": "llama",
        "label": label,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "model_id": LLAMA_MODEL,
        "prompt_version": PROMPT_VERSION,
        "error": error,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    weave.init("Boilerplate_Classifier")

    rubric = RUBRIC_PATH.read_text(encoding="utf-8")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Load or create the reproducible labeling sample
    if SAMPLE_PATH.exists():
        log.info("Loading existing labeling sample from %s", SAMPLE_PATH)
        sample_df = pd.read_parquet(SAMPLE_PATH)
    else:
        log.info("Building labeling sample (n=%d, seed=%d)…", SAMPLE_SIZE, SEED)
        sentences_df = pd.read_parquet(DATA_RAW / "sentences.parquet")
        sample_df = create_labeling_sample(sentences_df)
        sample_df.to_parquet(SAMPLE_PATH, index=False)
        log.info("Sample saved → %s", SAMPLE_PATH)

    rows = list(sample_df.itertuples(index=False))
    log.info("Sample size: %d sentences", len(rows))

    # W&B run
    run = init_wandb_run(
        project="Boilerplate_Classifier",
        name="stage3-gold-labeling",
        config={
            "sample_size": len(sample_df),
            "prompt_version": PROMPT_VERSION,
            "concurrency": CONCURRENCY,
            "anthropic_model": ANTHROPIC_MODEL,
            "google_model": GOOGLE_MODEL,
            "llama_model": LLAMA_MODEL,
            "seed": SEED,
            "cost_limit_usd": ANTHROPIC_COST_LIMIT_USD,
        },
        tags=["stage:3-gold", "purpose:label", "split:none"],
        job_type="label",
    )

    # Build clients (each factory checks its env var and raises if missing)
    clients = {
        "anthropic": anthropic_client(),
        "google": google_client(),
        "llama": wandb_inference_client(),
    }

    # Load JSONL caches (resume support)
    caches = {j: load_cache(j, CACHE_DIR) for j in ("anthropic", "google", "llama")}
    for j, c in caches.items():
        log.info("%s cache: %d existing entries (will skip)", j, len(c))

    # Cost tracker for Anthropic (only judge with per-token billing that could exceed budget)
    cost_tracker: dict = {
        "total_usd": 0.0,
        "calls": 0,
        "total_sentences": max(0, len(rows) - len(caches["anthropic"])),
    }

    # Per-judge semaphores (5 concurrent calls each, independent)
    sems = {j: asyncio.Semaphore(CONCURRENCY) for j in ("anthropic", "google", "llama")}

    # Bind clients and cost_tracker into judge_fn callables for run_one_judge
    _rubric = rubric  # local alias for closures

    async def anthropic_fn(sid: str, prompt: str) -> dict:
        return await _call_anthropic(sid, prompt, _rubric, clients["anthropic"], cost_tracker)

    async def google_fn(sid: str, prompt: str) -> dict:
        return await _call_google(sid, prompt, _rubric, clients["google"])

    async def llama_fn(sid: str, prompt: str) -> dict:
        return await _call_llama(sid, prompt, _rubric, clients["llama"])

    # Dispatch all 3 judges concurrently
    log.info("Dispatching %d sentences × 3 judges…", len(rows))
    anthropic_results, google_results, llama_results = await asyncio.gather(
        run_one_judge(
            "anthropic", rows, caches["anthropic"], anthropic_fn, CACHE_DIR, sems["anthropic"]
        ),
        run_one_judge("google", rows, caches["google"], google_fn, CACHE_DIR, sems["google"]),
        run_one_judge("llama", rows, caches["llama"], llama_fn, CACHE_DIR, sems["llama"]),
    )

    log.info(
        "Anthropic spend: $%.4f across %d calls", cost_tracker["total_usd"], cost_tracker["calls"]
    )

    # Assemble output DataFrame
    all_results = anthropic_results + google_results + llama_results
    output_df = pd.DataFrame(all_results)[REQUIRED_OUTPUT_COLUMNS]
    output_df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Saved judge outputs → %s  (%d rows)", OUTPUT_PATH, len(output_df))

    # W&B metrics and artifact
    import wandb

    n_labeled = int(output_df["label"].notna().sum())
    n_errors = int(output_df["error"].notna().sum())
    dist = output_df["label"].value_counts().to_dict()

    wandb.log(
        {
            "total_judgments": len(output_df),
            "labeled_count": n_labeled,
            "error_count": n_errors,
            "anthropic_cost_usd": cost_tracker["total_usd"],
            "label_boilerplate": dist.get("boilerplate", 0),
            "label_substantive": dist.get("substantive", 0),
        }
    )

    artifact = wandb.Artifact(
        "judge-outputs",
        type="dataset",
        description=f"LLM judge outputs: {len(sample_df)} sentences × 3 judges",
        metadata={"prompt_version": PROMPT_VERSION, "sample_size": len(sample_df)},
    )
    artifact.add_file(str(OUTPUT_PATH))
    run.log_artifact(artifact)
    run.finish()

    log.info("Done. Labeled: %d  Errors: %d", n_labeled, n_errors)


if __name__ == "__main__":
    asyncio.run(main())
