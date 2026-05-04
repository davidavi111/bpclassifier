# CLAUDE.md — Project Rules for `bpclassifier`

Read at session start, alongside global `~/.claude/CLAUDE.md`. Reference-only material (W&B tagging conventions, per-stage Definition of Done checklists, commit-message examples) lives in `CLAUDE_REFERENCE.md` — load it only when relevant.

---

## On every session start

1. Read this file fully.
2. Read `PLAN.md` fully (locked Stage 3-7 decisions).
3. Tell David which stage we're in (per `PLAN.md` "Current state") and what's next. Don't modify files until David confirms.
4. Run `pytest -q`. If failing, STOP and report — don't start work on a broken baseline.

`HANDOFF.md` is for the Cowork-mode supervisor, not Claude Code. Skip it.

---

## "Stop and Ask" rule (NON-NEGOTIABLE)

Stop and ask David rather than guess, when you hit:

- Ambiguous labeling case not covered by `data/gold/rubric.md`
- Unexpected file format / encoding that breaks parsing
- Suspicious metric (e.g., classifier scoring 1.00 — likely a leak)
- Library API mismatch with `PLAN.md` or existing code
- Any choice not pre-decided in `PLAN.md`
- Any deviation from `BPClassifier_Student_Handout.pdf`

Asking costs 5 min; guessing wrong costs hours.

---

## Hard rubric constraints (graded — never violate)

| Constraint | Practical meaning |
|---|---|
| Substantive recall ≥ 0.96 on test | Tune threshold on val/OOF; evaluate on test ONCE. If no threshold meets the floor, the classifier fails — pick another or collect more labels. |
| Test set frozen | `data/splits/test.json` touched EXACTLY ONCE, on final eval. Never for tuning, model selection, or "let me just check." |
| Multi-source gold labels + audited disagreements | Trio-of-judges majority vote; hand-audit a stratified disagreement sample; document ambiguity rules. |
| ≥ 5 classifier families | Same features, same splits — fairness matters. |
| Speed numbers reported | Training time + inference throughput (sentences/sec) per model. |
| Macro-F1 ~0.90 on test | Among recall-feasible thresholds, maximize macro-F1. |

---

## Architecture (skim — full detail in `PLAN.md`)

- Conda env `Boilerplate_Classifier`, Python 3.12 (`environment.yml`)
- W&B project `Boilerplate_Classifier` (runs, artifacts, sweeps, registry)
- Local CPU: rules, LogReg, trees, FastText, embeddings, eval
- Colab T4: FinBERT + SetFit fine-tunes
- W&B Training NOT used (Qwen3-only catalog won't fit our models)

---

## Security rules (project-specific; complement global)

1. `WANDB_API_KEY` and judge-LLM keys live ONLY in OS env vars.
2. ALL W&B auth routes through `scripts/gpu_runner.get_wandb_key()`. No raw `os.environ["WANDB_API_KEY"]` elsewhere.
3. ALL judge-LLM auth routes through `scripts/api_clients.py` factories. Don't instantiate `Anthropic()` / `OpenAI()` directly elsewhere.
4. NEVER print, log, return, or commit any API key — even masked, even "to verify it works."
5. NEVER read `.env`, `.envrc`, `~/.netrc`, `~/.config/wandb/`. `.claudeignore` blocks these — don't work around it.
6. If a key is missing, raise `EnvironmentError` with shell setup instructions. NEVER prompt David to paste a key into chat.

---

## Pre-commit checklist (manual, before every commit)

1. `git status` + `git diff` — review changes.
2. Scan diff for secrets (long random strings; fields named `key`/`token`/`secret`/`password`/`auth`/`credential`). Suspicious → STOP, ask David.
3. Stage explicitly by name (never `git add .` or `-A`).
4. `git diff --staged` — re-scan.
5. `pytest -q` must pass.
6. `pre-commit run --all-files` must pass.
7. Conventional Commits format. Examples in `CLAUDE_REFERENCE.md`.

---

## Git workflow

- Single branch `main`. Atomic commits.
- Tag every completed stage: `stage-N-<short-name>`.
- Never force-push to `main` without David's approval.
- Never create remote / change visibility / merge a PR without David's explicit approval (per global).

---

## Data discipline

- `ECT/` is read-only. Never write into that folder.
- `data/splits/test.json` touched ONCE for final eval. Any other use is a bug.
- Cache expensive ops (LLM judge calls, embeddings) as W&B Artifacts + local parquet, content-hash named.
- `np.random.seed(42)` + `random.seed(42)` at top of every script. Document seed in W&B run config.
