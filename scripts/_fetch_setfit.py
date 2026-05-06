"""Fetch oof-setfit:v0 parquets from W&B to data/interim/."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gpu_runner import get_wandb_key  # noqa: E402

get_wandb_key()
import pandas as pd  # noqa: E402
import wandb  # noqa: E402

ENTITY = "david-avichzer-hebrew-university-of-jerusalem"
PROJECT = "Boilerplate_Classifier"
INTERIM = Path(__file__).parent.parent / "data" / "interim"

api = wandb.Api()
art = api.artifact(f"{ENTITY}/{PROJECT}/oof-setfit:v0")
dl = Path(art.download())

for fname in ["oof_train_setfit.parquet", "pred_val_setfit.parquet"]:
    shutil.copy2(dl / fname, INTERIM / fname)
    print(f"Copied {fname}")

oof = pd.read_parquet(INTERIM / "oof_train_setfit.parquet")
val = pd.read_parquet(INTERIM / "pred_val_setfit.parquet")
print(f"OOF: {len(oof)} rows  std={oof.prob_substantive.std():.3f}")
print(f"Val: {len(val)} rows  std={val.prob_substantive.std():.3f}")
print("Done.")
