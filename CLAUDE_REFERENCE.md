# CLAUDE_REFERENCE.md — Reference material for `bpclassifier`

Material that doesn't need to be in Claude Code's context every session — load only when about to call `wandb.init()`, write a commit, or close out a stage.

---

## W&B run tagging

Every `wandb.init()` call must include meaningful tags:

- `stage:<n>` — pipeline stage, e.g. `stage:5-zoo`
- `purpose:<x>` — `extract`, `label`, `train`, `tune`, `eval`, `final`
- `model:<x>` — training runs only: `logreg`, `setfit`, `finbert`, `fasttext`, `histgbm`, `rules`, `ensemble-mean`, `ensemble-rank`
- `split:<x>` — `train`, `val`, `test`, `oof`

Without tags, 100+ runs become unfilterable.

---

## Conventional Commits — examples

- `feat(extract): add NLTK punkt sentence tokenization`
- `fix(judges): handle empty Anthropic response`
- `docs(plan): mark stage 2 complete`
- `test(judges): cover Ollama timeout path`
- `feat(stage-3): freeze gold labels with majority vote + audit corrections`

Stage-completion commits use scope `stage-N`.

---

## Definition of Done — per-stage gate

Stage N is COMPLETE only when:

1. All scripts for the stage run end-to-end without errors.
2. Stage-specific tests pass (e.g. `tests/test_stage_2_extraction.py`).
3. Stage-specific W&B run is visible in the dashboard.
4. PLAN.md "Definition of Done" checklist for that stage is fully checked.
5. Commit `feat(stage-N): <one-line summary>` made and tag `stage-N-<short-name>` pushed.

Don't start Stage N+1 until Stage N is fully Done.
