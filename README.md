# bpclassifier

Boilerplate-vs-substantive sentence classifier for earnings-call transcripts.

Course assignment, NLP track. See `BPClassifier_Student_Handout.pdf` for the full specification.

## What this does

Given a quarterly earnings-call transcript, classify each sentence as either:

- **boilerplate** — scripted intros, safe-harbor language, operator/analyst housekeeping, generic thanks
- **substantive** — material numbers, guidance, segment commentary, strategy, Q&A specifics

The deliverable is a small GUI that loads a transcript and renders the tagging inline.

## Reproducibility — exact commands from a clean checkout

Tested on Windows 11 + Conda + Python 3.12.

```powershell
# 1. Clone
git clone <repo-url>
cd "second exercise"

# 2. Create the environment
conda env create -f environment.yml
conda activate Boilerplate_Classifier

# 3. Install the project itself in editable mode
pip install -e .

# 4. Install pre-commit hooks (one time per checkout)
pre-commit install

# 5. Set your W&B API key as a Windows User env var (one time, replace placeholder)
# [System.Environment]::SetEnvironmentVariable("WANDB_API_KEY", "<your-key>", "User")
# Then close and reopen all terminals.

# 6. Verify setup
python scripts/gpu_runner.py     # should print: WANDB_API_KEY loaded successfully: ****...****
pytest                            # should run 5 test files, all pass

# 7. Run the full pipeline (Stages 2 - 7)
python scripts/01_extract.py
python scripts/02_label.py
python scripts/03_train_zoo.py
python scripts/04_pick_winner.py

# 8. Launch the GUI
streamlit run scripts/05_gui.py
```

## Project layout

```
.
├── ECT/                   raw earnings-call transcripts (provided)
├── data/
│   ├── raw/               extracted sentence pool (regenerable, gitignored)
│   ├── interim/           cached judge outputs / embeddings (gitignored)
│   ├── gold/              frozen labeled set + rubric (committed)
│   └── splits/            frozen train/val/test indices (committed)
├── src/bpclassifier/      library code (importable as `bpclassifier`)
├── scripts/               runnable stage scripts and the secure W&B wrapper
├── tests/                 pytest suite
├── artifacts/             saved models (small ones; large ones live in W&B Artifacts)
├── notebooks/             exploration only, not pipeline-of-record
└── reports/               figures + final write-up PDF
```

## Security

- The W&B API key lives ONLY in the OS environment variable `WANDB_API_KEY`. It is never in any tracked file.
- All W&B authentication routes through `scripts/gpu_runner.py`. No other code reads `WANDB_API_KEY` directly.
- Pre-commit hooks (`detect-secrets` + `gitleaks`) scan every commit for secrets.
- `.claudeignore` blocks Claude Code from reading any credential-shaped file.

See `CLAUDE.md` for the full security and workflow rules.
