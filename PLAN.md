# PLAN.md — bpclassifier master plan

Single source of truth for the project. Read first by Claude Code on every session.

> **Current state:** Stages 1 + 2 COMPLETE (tagged + pushed). All Stage 3–7
> architectural decisions LOCKED. Stage 3 first execution step: rubric drafted
> at `data/gold/rubric.md` — awaiting David's review and API-key setup, then
> the labeling pipeline runs.
> **Stages 8 (writeup) and 9 (submission) deferred — not in current scope.**

---

## The big picture

**Business problem.** Earnings-call transcripts contain a lot of scripted noise (safe-harbor disclaimers, operator housekeeping, generic thanks) and a smaller amount of genuinely informative content (numbers, guidance, strategy commentary). Downstream analyst pipelines want only the latter.

**Asymmetric cost.** Losing a substantive sentence is much worse than forwarding a boilerplate one. The rubric encodes this: substantive recall ≥ 0.96 is a HARD floor; among recall-feasible thresholds, maximize macro-F1.

**The four deliverables (one zip on Moodle):**

1. Notebook / scripts that reproduce the pipeline end-to-end.
2. Trained best model (loaded by the GUI).
3. GUI application that displays inline tagging on a transcript.
4. Project write-up PDF (5–10 pages).

---

## Hard rubric constraints

| Constraint | Pts | Rule |
|---|---|---|
| Substantive recall ≥ 0.96 on test | 15 | Hard floor; failing caps line at 0 |
| Macro-F1 on test | 20 | Target ~0.90; among recall-feasible, maximize |
| Leaderboard breadth | 15 | ≥5 families, same features/splits, speed reported |
| Gold-standard quality | 25 | Multi-source labels, rubric with anchors, audited disagreements |
| GUI | 10 | File picker, inline tagging, statistics panel, runs in <1 min |
| Write-up | 15 | 5–10 pages, 8 specific sections |

---

## Architecture (locked)

| Layer | Choice |
|---|---|
| Env manager | Conda (matches David's global default) |
| Python | 3.12 |
| Project layout | `src/` package |
| Tracking | W&B Academic plan (unlimited runs, 200GB artifacts, sweeps, registry) |
| Compute (CPU) | Local Conda env (rules, LogReg, trees, FastText, embeddings, eval) |
| Compute (GPU) | Google Colab free T4 (FinBERT, SetFit) |
| W&B managed compute | NOT USED (Qwen3-only catalog) |
| LLM judges | Anthropic Sonnet 4.6 + DeepSeek-V3.1 + W&B Inference Llama 3.3 70B |
| GUI | Streamlit |
| Git | `main` only, atomic commits, stage tags |

---

## All locked decisions across stages

### Stage 3 — Gold Labels (9 decisions)

| # | Decision |
|---|---|
| 3.1 | Judges: Anthropic Sonnet 4.6 + DeepSeek-V3.1 (W&B Inference) + Llama 3.3 70B (W&B Inference). Gemini swapped due to free-tier rate limits making 2,500-sentence run impractical; DeepSeek preserves diversity (third distinct model lineage) and runs free on W&B academic credit. |
| 3.2 | 2,500 sentences |
| 3.3 | Stratify by `company × section_type` (42 cells × ~60 each) |
| 3.4 | Supervisor drafts rubric v1; David reviews before judge calls |
| 3.5 | Judge output: label + reasoning (`{"label": "...", "reasoning": "..."}`) |
| 3.6 | Cache key: `(sentence_id, judge_name)` |
| 3.7 | 5 concurrent calls per judge |
| 3.8 | 50 disagreement cases audited by hand (~25 min) |
| 3.9 | Weave on for all judge calls |

### Stage 4 — Splits + Features (5 decisions)

| # | Decision |
|---|---|
| 4.1 | Embedding model: `all-mpnet-base-v2` |
| 4.2 | 60/20/20 splits, stratified by label, group-by-transcript, seed=42 |
| 4.3 | No per-company holdout (default scope) |
| 4.4 | Supervisor proposes ~25 regex flags; David reviews |
| 4.5 | Cache: parquet local + W&B Artifact `embeddings:v0` |

### Stage 5 — Classifier Zoo (3 decisions)

| # | Decision |
|---|---|
| 5.1 | 6 families: Rules + LogReg-on-embeddings + HistGBM + FastText + FinBERT + SetFit |
| 5.2 | 5-fold stratified group-aware CV for OOF probabilities |
| 5.3 | Colab uses W&B Artifacts as the bridge (no manual file shuffling) |

### Stage 6 — Ensembles + Winner (3 decisions)

| # | Decision |
|---|---|
| 6.1 | Both ensembles: mean-probability + rank-averaged of top-5 non-transformer |
| 6.2 | No calibration (skip Platt/isotonic) |
| 6.3 | Winner: substantive recall ≥ 0.96 → highest macro-F1 wins (auto-enforced) |

### Stage 7 — GUI (3 decisions)

| # | Decision |
|---|---|
| 7.1 | Streamlit |
| 7.2 | Statistics panel: counts + percentages + per-section breakdown |
| 7.3 | Inputs: file uploader + paste textarea + dropdown of `ECT/` files |

---

## Stage progress

### Stage 1 — Project Foundation [DONE — tagged stage-1-foundation]

Conda env, project structure, secure W&B wrapper, judge-LLM scaffolding, pre-commit hooks (detect-secrets + gitleaks + ruff + nbstripout), 31 tests, private GitHub repo, tag pushed.

### Stage 2 — Data Extraction [DONE — tagged stage-2-extraction]

131 transcripts → 54,923 sentences with full metadata. Section breakdown: 21,797 prepared_remarks / 7,644 question / 25,482 answer. Length stats: mean 121, median 108, p95 242, max 1,004, min 40. Output at `data/raw/sentences.parquet` + W&B Artifact `sentence-pool:v0`. 21 extraction tests passing.

**Known issue:** `speaker_name` is sparsely populated (6 unique values across 54K rows). Raw `speaker_role` field has 141 clean unique values and is the reliable column. Person-name parsing can be improved post-hoc if needed.

### Stage 3 — Gold Labels [IN PROGRESS]

**Already done:**
- All 9 architectural decisions locked
- Rubric v1 drafted at `data/gold/rubric.md` (awaiting David's review)

**Remaining:**
- David reviews rubric, provides edits, approves
- David sets `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` as Windows User env vars
- David sets $10 spend cap at console.anthropic.com
- Build `scripts/02_label.py` (sample → 3 judges concurrent, cached, Weave-traced → judge_outputs.parquet)
- Build `scripts/03_audit.py` (50 disagreements for human review)
- Build `scripts/04_freeze_gold.py` (majority vote + audit corrections → `data/gold/labeled.parquet` + W&B Artifact `gold-labels:v0` + `data/gold/audit_log.md`)
- Tests: `test_gold_class_balance.py`, `test_gold_no_duplicates.py`, `test_gold_judges_complete.py`
- Commit + tag `stage-3-gold-labels`, push

### Stage 4 — Splits + Features

- 60/20/20 stratified group-aware split (frozen seed = 42), saved as `data/splits/{train,val,test}.json`
- Compute embeddings via `all-mpnet-base-v2`, cache as parquet + W&B Artifact `embeddings:v0`
- Apply ~25 regex feature flags (David reviews list)
- Tests: leakage checks, shape checks, NaN checks
- Commit + tag `stage-4-features`

### Stage 5 — Classifier Zoo

- 6 families on identical features/splits
- 5-fold OOF probabilities for threshold tuning
- FinBERT + SetFit on Colab (W&B-mediated handoff)
- Tests: each family trains without error
- Commit + tag `stage-5-zoo`

### Stage 6 — Ensembles + Winner

- Mean-prob and rank-averaged ensembles of top-5 non-transformer
- Apply recall floor (≥ 0.96 on test, touched ONCE)
- Among eligible, max macro-F1 wins
- Save winner + threshold to W&B Model Registry, alias `production`
- Commit + tag `stage-6-winner`

### Stage 7 — GUI

- Streamlit app: file uploader + paste textarea + ECT/ dropdown
- Inline rendering with red boilerplate background
- Stats panel: counts + percentages + per-section
- Eyeball test on 2 unseen transcripts
- Commit + tag `stage-7-gui`

### Stages 8 + 9 — Out of scope

Deferred. The supervising chat should not bring these up unless David explicitly initiates.

---

## Cost & rate-limit summary

- **Anthropic Sonnet 4.6:** ~$8 total for the project. $10 hard cap.
- **Google Gemini 2.5 Flash:** free tier (15 RPM, 1,500 RPD). Caching handles daily-cap interruptions.
- **W&B Inference Llama 3.3 70B:** free under $250/mo academic credit.
- **All other compute:** free (local CPU, Colab T4).

---

## Required user interventions (already minimized)

| Touchpoint | Time |
|---|---|
| Set Anthropic + Google API keys | 10 min, one-time |
| Review rubric draft | 10 min |
| Review the ~25 regex flags | 10 min |
| Audit 50 disagreement cases | 25 min |
| Run 2 Colab notebooks (FinBERT + SetFit) | 10 min total |
| GUI eyeball test | 10 min |
| Approve commits/tags | quick |

**Total: ~75 minutes of focused human time across the rest of the project.**

---

## Improvements deferred (could revisit if time allows)

- Probability calibration (Platt/isotonic) — only if threshold variance > 0.05
- Per-company holdout for cross-company generalization check
- Active learning loop after first classifier
- Conformal prediction uncertainty bands
- Speaker / section context features
