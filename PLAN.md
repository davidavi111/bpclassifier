# PLAN.md — bpclassifier master plan

Single source of truth for the project. Updated at the end of every stage.

> **Current state:** Stage 2 — Data Extraction COMPLETE. 54,923 sentences from
> 131 transcripts (14 tickers) extracted to `data/raw/sentences.parquet` and
> published as W&B Artifact `sentence-pool:v0`. Awaiting David's go-ahead on
> the Stage 2 commit + tag, then proceed to Stage 3 (Gold-Label Pipeline).

---

## The big picture

**Business problem.** Earnings-call transcripts contain a lot of scripted noise
(safe-harbor disclaimers, operator housekeeping, generic thanks) and a smaller amount
of genuinely informative content (numbers, guidance, strategy commentary). Downstream
analyst pipelines want only the latter. We build a sentence-level classifier and ship
a GUI that tags transcripts inline.

**Asymmetric cost.** Losing a substantive sentence is much worse than forwarding a
boilerplate one. The rubric encodes this: substantive recall ≥ 0.96 is a HARD floor;
among recall-feasible thresholds, maximize macro-F1.

**The four deliverables (one zip on Moodle):**

1. Notebook / scripts that reproduce the pipeline end-to-end.
2. Trained best model (loaded by the GUI).
3. GUI application that displays inline tagging on a transcript.
4. Project write-up PDF (5–10 pages).

---

## Architecture decisions (locked)

| Layer | Choice | Why |
|---|---|---|
| Env manager | Conda | Matches David's global default. |
| Python | 3.12 | ML ecosystem fully caught up by April 2026. |
| Project layout | `src/` package | Clean imports, real pip-installable. |
| Tracking | W&B Academic plan | Unlimited runs, 200GB artifacts, sweeps, registry, Reports. |
| Compute (CPU) | Local Conda env | Rules, LogReg, trees, FastText, embeddings, all eval. |
| Compute (GPU) | Google Colab free T4 | FinBERT fine-tune (~10 min), SetFit fine-tune (~5 min). |
| W&B Training | NOT USED | Closed catalog (Qwen3 only); incompatible with our models. |
| Gold labels | Multi-LLM majority vote | Required for rubric's 25-pt gold-standard criterion. |
| GUI | Streamlit (default) | Shortest path; can swap to Gradio if it shines for inline highlighting. |
| Git | `main` only, atomic commits, stage tags | Solo project, simple rollback. |

---

## The 9 stages

### Stage 1 — Project Foundation [DONE - tagged stage-1-foundation]

Build the secure, reproducible scaffolding.

**Deliverables:**
- [x] Project skeleton (`data/`, `src/`, `scripts/`, `tests/`, `notebooks/`, `artifacts/`, `reports/`)
- [x] `.gitignore` with project-specific overrides
- [x] `.claudeignore` blocking secrets
- [x] `.gitattributes` for cross-platform line endings
- [x] `environment.yml` for the Conda env
- [x] `pyproject.toml` with ruff + pytest config
- [x] `.pre-commit-config.yaml` (detect-secrets, gitleaks, ruff, nbstripout)
- [x] `scripts/gpu_runner.py` — secure W&B wrapper
- [x] `scripts/api_clients.py` — judge-LLM factories
- [x] Test suite (5 files): `test_secrets.py`, `test_no_secrets_in_repo.py`,
      `test_env_smoke.py`, `test_data_paths.py`, `test_wandb_auth.py`
- [x] `README.md` (grader-facing run instructions)
- [x] `CLAUDE.md` (project rules)
- [x] `PLAN.md` (this file)
- [x] `WANDB_API_KEY` set as Windows User env var, verified
- [x] `conda env create -f environment.yml`
- [x] `conda activate Boilerplate_Classifier && pip install -e .`
- [x] `pre-commit install` and `.secrets.baseline` generated
- [x] `git init`, first commit, private GitHub repo at `github.com/davidavi111/bpclassifier`
- [x] `pytest` passes (31 tests + smoke W&B auth)
- [x] Tag `stage-1-foundation` pushed

**Definition of Done:** All checkboxes above are checked. Tag exists. Repo is on
GitHub (private). David has confirmed `pytest -q` is green.

---

### Stage 2 — Data Extraction [DONE - awaiting commit]

Turn 131 raw transcripts into a clean sentence pool.

**Outcome:**
- 131 transcripts parsed (14 tickers: AMD, AVGO, BLK, C, FAST, GS, INTC, JNJ,
  JPM, MSFT, NVDA, PLTR, V, WFC).
- 54,923 sentences extracted (substantially higher than the original 8K–15K
  estimate; the original was based on a truncated file listing — no quality
  concern, just a bigger pool to draw from).
- Section breakdown: 21,797 prepared_remarks / 7,644 question / 25,482 answer.
- Length stats: mean 121, median 108, p95 242, max 1,004, min 40 chars.
- `data/raw/sentences.parquet` written (regenerable, gitignored).
- W&B Artifact `sentence-pool:v0` published.

**Implementation notes:**
- Layer 1 parser adapted from David's prior earnings-NLP project; full
  attribution comment in `src/bpclassifier/extract.py`.
- Bug fix during Stage 2: when transcripts skip the role line and start a
  Question/Answer/Presenter Speech section directly with content, the parser
  was capturing the first sentence as `role`. Fixed via `_resolve_role_and_body`
  (covers both empty-body AVGO case and non-empty-body cases).
- `speaker_name` is best-effort and sparsely populated (6 unique values across
  the corpus). The role-line format in this dataset is mostly 2-part
  (`Executives - Title`) without an embedded person name. Raw `speaker_role`
  field has 141 clean unique values and is the reliable column. Person-name
  parsing can be improved post-hoc if it becomes a wanted feature.
- Tests in place: `test_extract_*.py` — 21 tests, all passing.

**Definition of Done:** All checkboxes met. Tag `stage-2-extraction` pending.

---

### Stage 3 — Gold-Label Pipeline

Build the rubric, run multi-judge labeling, audit disagreements, freeze the gold set.

**Tasks:**
- Write `data/gold/rubric.md`: definitions of boilerplate vs. substantive, with 8–12
  anchor examples per class, plus rules for edge cases (analyst intros, generic thanks,
  one-word answers, mixed sentences).
- Choose 3 judges. Default plan: local Ollama (qwen3:14b or similar), Anthropic Claude,
  OpenAI GPT. (Will defer final choice; all clients ready in `api_clients.py`.)
- Stratified-random-sample 2,500 sentences from the pool, balanced across companies and
  estimated section.
- Run each judge with caching (parquet, content-hash key) and resume-on-interrupt.
- Compute per-judge agreement and pairwise Cohen's kappa.
- Majority vote. Hand-audit a stratified sample of the disagreements (~100 sentences).
- Freeze: `data/gold/labeled.parquet` (committed), `data/gold/audit_log.md` (committed).
- Log W&B Run with disagreement stats. Save W&B Artifact `gold-labels:v0`.
- Use Weave (`@weave.op()` decorators) on the judge call functions for free traceability.

**Tests to add:**
- `test_gold_class_balance.py` — class proportions within expected range.
- `test_gold_no_duplicates.py` — sentence_id is unique.
- `test_gold_judges_complete.py` — every row has 3 judge labels.

**Definition of Done:** `data/gold/labeled.parquet` exists, frozen. Disagreement rate
documented. Audit log committed.

---

### Stage 4 — Splits + Feature Engineering

**Tasks:**
- 60/20/20 stratified split (label-stratified, with company-grouping check to avoid
  leakage). Frozen seed = 42.
- Save split indices as `data/splits/{train,val,test}.json` (committed, tiny).
- Build 20–30 regex feature flags (safe-harbor, operator cues, generic thanks, dollar/percent
  counts, digit density, length, punctuation ratios, speaker-intro patterns, analyst firms).
- Generate sentence embeddings via `sentence-transformers` (default: `all-mpnet-base-v2` or
  `mxbai-embed-large`) for the entire pool. Cache as W&B Artifact `embeddings:v0`.
- Build the feature matrix `X = [embeddings | regex_flags]`.

**Tests:** `test_splits_no_leakage.py`, `test_features_shape.py`, `test_features_no_nan.py`.

---

### Stage 5 — Classifier Zoo

Six families on the same features and splits. Same OOF protocol for threshold tuning.

| Family | Library | Approx cost |
|---|---|---|
| Rules + regex baseline | pure Python | seconds |
| LogReg on frozen embeddings | sklearn | seconds |
| HistGradientBoosting on embeddings + flags | sklearn | minutes |
| FastText n-gram | fasttext-wheel | seconds |
| FinBERT fine-tune | transformers (Colab T4) | 10–15 min |
| SetFit contrastive fine-tune | setfit (Colab T4) | 5–15 min |

For each: train, log to W&B (run config + metrics + curves), compute 5-fold OOF
probabilities on train+val, save model.

**Tests:** `test_zoo_each_family_trains.py` (smoke).

---

### Stage 6 — Threshold Tuning, Ensembles, Winner Selection

**Tasks:**
- Pool OOF probabilities. For each model: find the lowest threshold that achieves
  substantive recall ≥ 0.96 on OOF. Report fold-to-fold std of the chosen threshold.
- Mean-prob ensemble of top-5 non-transformer members. Rank-averaged ensemble of same.
- All models eligible if and only if they have a feasible threshold.
- Among eligible models, pick the one with highest macro-F1 on the *test* set.
  THIS IS THE ONLY TIME THE TEST SET IS TOUCHED.
- Save winner + threshold as W&B Artifact `winning-model:v0`. Promote in registry to
  alias `production`.

---

### Stage 7 — GUI

**Tasks:**
- Streamlit app that:
  - Lets user upload or paste a transcript.
  - Sentence-tokenizes (same code as Stage 2).
  - Loads winning model from `artifacts/` or W&B registry.
  - Renders inline: boilerplate sentences with red background, substantive plain.
  - Statistics panel: counts, percentages.
  - Run instruction in the README.

**Tests:** `test_gui_loads_model.py`, manual eyeball test on 2 unseen transcripts.

---

### Stage 8 — Write-up PDF

5–10 pages, with figures. Sections:

- Introduction (1 paragraph)
- Gold-standard methodology (sources, rubric, disagreement rate, audit, class balance)
- Feature engineering (regex flags + why)
- Classifier-zoo results (leaderboard table, paragraph per family)
- Recall-constrained threshold selection (procedure, per-class P/R/F1, confusion matrices)
- Error analysis (~10 misclassifications by direction, with commentary)
- GUI screenshot (full page)
- Reproducibility (commands)
- LLM-usage disclosure (per ground rules)

Use the `docx` skill to author professional output, export to PDF.

---

### Stage 9 — Submission Packaging

- Test the run-from-clean-checkout commands on a fresh clone.
- Zip everything per Moodle requirements.
- Final commit + tag `submission`.

---

## Improvements beyond the rubric (David asked for these)

Discussed in the relevant stage; not yet committed:

1. **Calibration** (Stage 6): Platt or isotonic on top of LogReg/HistGBM probabilities. Helps
   threshold selection when raw probabilities are skewed.
2. **Conformal prediction** (Stage 6 add-on): produce uncertainty bands so the GUI can flag
   "low confidence" sentences in a third color.
3. **Per-company holdout** (Stage 4 add-on): in addition to the random test set, hold out one
   entire company (e.g. NVDA) and report transfer numbers. Tests robustness to new issuers.
4. **Active learning loop** (Stage 3 add-on): after the audit, train a quick LogReg, find
   cases of high model uncertainty in the unlabeled pool, label those, retrain. 200 extra
   labels typically yields more F1 than 1000 random ones.
5. **Speaker / section context features** (Stage 4 add-on): the previous sentence's class
   and the section header (Q&A vs. prepared) are highly predictive features.

---

## Open questions to resolve later (not blocking now)

- Final choice of judge LLMs (locked at start of Stage 3).
- Embedding model choice between `all-mpnet-base-v2` (smaller, faster) and `mxbai-embed-large`
  (larger, possibly stronger). Decide at Stage 4 by running both on a small sample.
- Streamlit vs. Gradio for the GUI — Streamlit is the default, revisit if it can't render
  inline highlighting cleanly.
