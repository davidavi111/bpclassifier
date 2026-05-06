"""07_gui_smoke_test.py — Stage 7 headless GUI smoke test.

1. Loads winner artifact + threshold from W&B.
2. Runs the full pipeline on 1 random test-set transcript (seed=42).
3. Prints: total sentences, % per class, runtime.
4. Writes data/eval/eyeball_picks.json:
   - one boilerplate-heavy transcript (highest BP% in test set)
   - one Q&A-heavy transcript (highest Q&A sentence ratio in test set)
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS.resolve()))
sys.path.insert(0, str((_ROOT / "src").resolve()))

from bpclassifier.extract import parse_transcript, transcript_to_sentences  # noqa: E402
from gpu_runner import get_wandb_key  # noqa: E402

random.seed(42)

_ECT_DIR = _ROOT / "ECT"
_SPLITS_DIR = _ROOT / "data" / "splits"
_EVAL_DIR = _ROOT / "data" / "eval"


def load_winner():
    """Download FinBERT winner from W&B, return (model, tokenizer, threshold)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    get_wandb_key()
    import wandb  # noqa: PLC0415

    with open(_EVAL_DIR / "confusion_matrix_test.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    winner_name = cfg["winner"]
    threshold = float(cfg["threshold"])

    api = wandb.Api()
    art = api.artifact(
        f"david-avichzer-hebrew-university-of-jerusalem/"
        f"Boilerplate_Classifier/model-{winner_name}:v0"
    )
    model_dir = Path(art.download())

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    return model, tokenizer, threshold, winner_name


def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    import torch
    import torch.nn.functional as F  # noqa: N812

    all_probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.clip(np.array(all_probs, dtype=float), 0.0, 1.0)


def run_on_transcript(path: Path, model, tokenizer, threshold: float) -> list[dict]:
    transcript = parse_transcript(path)
    sentences = transcript_to_sentences(transcript)
    if not sentences:
        return []
    texts = [s.text for s in sentences]
    probs = predict_proba(model, tokenizer, texts)
    return [
        {
            "text": s.text,
            "section_type": s.section_type,
            "label": "substantive" if p >= threshold else "boilerplate",
        }
        for s, p in zip(sentences, probs, strict=False)
    ]


def get_test_ticker_quarters() -> list[str]:
    """Return unique 'ticker_quarter' strings from the test split sentence IDs."""
    test_path = _SPLITS_DIR / "test.json"
    if not test_path.exists():
        return []
    rows = json.loads(test_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for row in rows:
        sid = row["sentence_id"]
        parts = sid.rsplit("_", 1)
        seen.add(parts[0])
    return sorted(seen)


def pick_eyeball_transcripts(
    ticker_quarters: list[str], model, tokenizer, threshold: float
) -> tuple[str, str]:
    """
    Run pipeline on all test transcripts, pick:
    - bp_pick:  highest boilerplate ratio
    - qa_pick:  highest Q&A (question+answer) sentence ratio
    Returns (bp_filename, qa_filename).
    """
    bp_best = ("", -1.0)
    qa_best = ("", -1.0)

    for tq in ticker_quarters:
        ect_path = _ECT_DIR / f"{tq}.txt"
        if not ect_path.exists():
            continue
        results = run_on_transcript(ect_path, model, tokenizer, threshold)
        if not results:
            continue

        total = len(results)
        n_bp = sum(1 for r in results if r["label"] == "boilerplate")
        n_qa = sum(1 for r in results if r["section_type"] in ("question", "answer"))

        bp_ratio = n_bp / total
        qa_ratio = n_qa / total

        if bp_ratio > bp_best[1]:
            bp_best = (ect_path.name, bp_ratio)
        if qa_ratio > qa_best[1]:
            qa_best = (ect_path.name, qa_ratio)

    return bp_best[0], qa_best[0]


def main() -> None:
    print("=== Stage 7 GUI Smoke Test ===\n")

    print("Loading winner model…")
    model, tokenizer, threshold, winner_name = load_winner()
    print(f"Winner: {winner_name}  threshold={threshold:.2f}\n")

    # Pick 1 random test transcript for smoke run
    ticker_quarters = get_test_ticker_quarters()
    if not ticker_quarters:
        print("ERROR: test.json not found")
        sys.exit(1)

    rng = random.Random(42)  # noqa: S311
    smoke_tq = rng.choice(ticker_quarters)
    smoke_path = _ECT_DIR / f"{smoke_tq}.txt"
    if not smoke_path.exists():
        # fallback: first available
        for tq in ticker_quarters:
            p = _ECT_DIR / f"{tq}.txt"
            if p.exists():
                smoke_path = p
                break

    print(f"Smoke transcript: {smoke_path.name}")

    t0 = time.perf_counter()
    results = run_on_transcript(smoke_path, model, tokenizer, threshold)
    elapsed = time.perf_counter() - t0

    total = len(results)
    n_bp = sum(1 for r in results if r["label"] == "boilerplate")
    n_sub = total - n_bp
    pct_bp = 100.0 * n_bp / total if total else 0.0
    pct_sub = 100.0 * n_sub / total if total else 0.0

    print(f"Total sentences : {total}")
    print(f"Boilerplate     : {n_bp}  ({pct_bp:.1f}%)")
    print(f"Substantive     : {n_sub}  ({pct_sub:.1f}%)")
    print(f"Runtime         : {elapsed:.1f}s")
    print()

    # Section breakdown
    sec_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"boilerplate": 0, "substantive": 0}
    )
    for r in results:
        sec_counts[r["section_type"]][r["label"]] += 1
    print("Section breakdown:")
    for sec in ["prepared_remarks", "question", "answer"]:
        if sec in sec_counts:
            bp = sec_counts[sec]["boilerplate"]
            sub = sec_counts[sec]["substantive"]
            print(f"  {sec:<20} bp={bp}  sub={sub}")
    print()

    # Eyeball picks
    print("Selecting eyeball picks across all test transcripts…")
    bp_pick, qa_pick = pick_eyeball_transcripts(ticker_quarters, model, tokenizer, threshold)

    eyeball = {
        "description": "2 test-set transcripts for eyeball check",
        "boilerplate_heavy": str(_ECT_DIR / bp_pick) if bp_pick else "",
        "qa_heavy": str(_ECT_DIR / qa_pick) if qa_pick else "",
    }
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _EVAL_DIR / "eyeball_picks.json"
    out_path.write_text(json.dumps(eyeball, indent=2), encoding="utf-8")
    print(f"Eyeball picks saved → {out_path}")
    print(f"  boilerplate-heavy : {bp_pick}")
    print(f"  Q&A-heavy         : {qa_pick}")
    print()
    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
