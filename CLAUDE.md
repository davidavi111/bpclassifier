# CLAUDE.md — Project Rules for `bpclassifier`

These rules are READ FIRST by Claude Code (CLI) at session start, alongside the global
`~/.claude/CLAUDE.md`. The global file's universal rules (Conventional Commits, no blind
`git add .`, scan diffs before commit, never auto-edit CLAUDE.md, etc.) apply here in full.
This file ADDS project-specific rules.

---

## On every session start

Before doing ANY work in this project, perform these steps in order:

1. Read this file (`CLAUDE.md`) fully.
2. Read `PLAN.md` fully.
3. Tell David which stage we are in (per `PLAN.md`'s "Current state" line) and what the
   next task is. Do not modify any files until David confirms the next task.
4. Run `pytest -q` to confirm the existing test suite still passes. If anything fails,
   STOP and report — do not start new work on a broken baseline.

---

## The "Stop and Ask" rule (NON-NEGOTIABLE)

During execution, you MUST stop and ask David rather than guess, whenever you encounter:

- An ambiguous labeling case not covered by the rubric in `data/gold/rubric.md`
- An unexpected file format, encoding, or character that breaks parsing
- A metric or output that looks suspicious (e.g., a classifier that scores 1.00 — likely a leak)
- A library API that doesn't match what `PLAN.md` or the code suggests
- Any choice not pre-decided in `PLAN.md`
- Any deviation from the suggested workflow in `BPClassifier_Student_Handout.pdf`

When in doubt, stop. Asking costs five minutes; guessing wrong costs hours.

---

## Hard constraints from the assignment

These are graded directly. Treat them as load-bearing.

| Constraint | Source | What it means in practice |
|---|---|---|
| Substantive recall ≥ 0.96 on test | Handout, p. 2 | Tune threshold on validation/OOF, evaluate on test ONCE. If no threshold meets the floor, the classifier fails — pick another or collect more labels. |
| Test set is frozen | Handout, p. 2 | The held-out test split is touched EXACTLY ONCE, on the final run. Never use it for threshold tuning, model selection, or "let me just check." |
| Multi-source gold labels | Handout, p. 2 | Trio-of-judges majority vote (e.g., three LLMs, or LLM + human passes). Report disagreement rate. |
| Audited disagreements | Handout, p. 2 | Hand-audit a stratified sample of judge disagreements; document ambiguity rules. |
| ≥ 5 classifier families benchmarked | Handout, p. 3 | Same features, same splits — fairness matters. |
| Speed numbers reported | Handout, p. 3 | Training time + inference throughput (sentences/sec) for every entry. |
| Macro-F1 on test ~0.90 target | Handout, p. 5 | Among recall-feasible thresholds, maximize macro-F1. |

---

## Architecture

- **Environment:** Conda env `Boilerplate_Classifier`, Python 3.12. See `environment.yml`.
- **Tracking layer:** W&B project `Boilerplate_Classifier` for runs, artifacts, sweeps, model registry.
- **Compute layer:**
  - Local CPU (David's Windows machine) for: rules, LogReg-on-embeddings, tree ensembles,
    FastText, embedding generation, all evaluation.
  - Google Colab free T4 for: FinBERT fine-tune, SetFit fine-tune.
- **W&B Training is NOT used** — its catalog (Qwen3 family only, LoRA adapters only) does
  not support our model architectures.

---

## Security rules (project specific; complement global)

1. `WANDB_API_KEY` and all judge-LLM keys live ONLY in OS environment variables.
2. ALL W&B auth routes through `scripts/gpu_runner.get_wandb_key()`. Do not read
   `os.environ["WANDB_API_KEY"]` anywhere else.
3. ALL judge-LLM auth routes through `scripts/api_clients.py` factory functions. Do not
   instantiate `Anthropic()`, `OpenAI()`, etc. directly elsewhere.
4. NEVER print, log, return, or commit any API key value — not even masked, not even
   "to verify it works." The masked preview function in `gpu_runner.py` is for one-time
   manual smoke checks only.
5. NEVER read `.env`, `.envrc`, `~/.netrc`, or `~/.config/wandb/`. The `.claudeignore`
   file blocks these; do not work around it.
6. If a key is missing, raise `EnvironmentError` with shell setup instructions. NEVER
   prompt David to paste a key into chat.

---

## Pre-commit checklist (Claude runs through this MANUALLY before every commit)

Even though pre-commit hooks will catch most issues, do this BEFORE staging:

1. Run `git status` and `git diff` — review what changed.
2. Scan the diff for any of: long random strings, fields named `key`/`token`/`secret`/
   `password`/`api_key`/`auth`/`credential`. If anything looks like a secret, STOP and ask David.
3. Stage files explicitly by name (never `git add .` or `git add -A`).
4. Run `git diff --staged` and re-scan.
5. Run `pytest -q` — must pass.
6. Run `pre-commit run --all-files` — all hooks must pass.
7. Commit with Conventional Commits format. Examples:
   - `feat(extract): add NLTK punkt sentence tokenization`
   - `fix(judges): handle empty Anthropic response`
   - `docs(plan): mark stage 2 complete`
   - `test(judges): cover Ollama timeout path`

---

## Git workflow

- One branch: `main`. No feature branches for this project (solo, short timeline).
- Atomic commits: one logical change per commit.
- Tag every completed stage: `stage-1-foundation`, `stage-2-extraction`, ...
- Never force-push to `main` without explicit David approval.
- Never create a remote, change visibility, or merge a PR without explicit David approval
  (per global CLAUDE.md).

---

## Data discipline

- The contents of `ECT/` are read-only. Never write into that folder.
- The frozen test split (`data/splits/test.json`) is touched ONCE for final evaluation.
  Any code that loads it for any other purpose is a bug.
- Cache expensive operations (LLM judge calls, embeddings) as W&B Artifacts AND local
  parquet files, with content-hash names so re-runs deduplicate naturally.
- Set `np.random.seed(42)` and `random.seed(42)` at the top of every script. Document the
  seed in W&B run config.

---

## Tagging strategy for W&B runs

Every `wandb.init()` call must include meaningful tags:

- `stage:<n>` — the pipeline stage, e.g. `stage:5-zoo`
- `purpose:<x>` — `extract`, `label`, `train`, `tune`, `eval`, `final`
- `model:<x>` — for training runs only: `logreg`, `setfit`, `finbert`, `fasttext`,
  `histgbm`, `rules`, `ensemble-mean`, `ensemble-rank`
- `split:<x>` — `train`, `val`, `test`, `oof`

Tags make the dashboard filterable. Without them, 100+ runs become unusable.

---

## Definition of Done — per-stage gates

Stage N is COMPLETE only when:

1. All scripts for the stage run end-to-end without errors.
2. Stage-specific tests in `tests/` pass (e.g. `test_stage_2_extraction.py`).
3. Stage-specific W&B run is visible in the dashboard.
4. The "Definition of Done" checklist for that stage in `PLAN.md` is fully checked.
5. A commit has been made with message `feat(stage-N): <one-line summary>` and a tag
   `stage-N-<short-name>` has been pushed.

Do not start Stage N+1 until Stage N is fully Done.
