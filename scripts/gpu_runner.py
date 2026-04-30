"""
gpu_runner.py — secure W&B authentication wrapper.

This is the SINGLE place in the project that touches WANDB_API_KEY.
All other code MUST import from here, never read os.environ directly.

The key value is read from the OS environment exactly once, on demand,
and is never logged, printed, or returned in any string representation.

Run this file directly to verify your W&B setup:
    python scripts/gpu_runner.py
"""

from __future__ import annotations

import os
import sys
from typing import Any


class WandbAuthError(EnvironmentError):
    """Raised when W&B authentication is misconfigured."""


def get_wandb_key() -> str:
    """
    Read the W&B API key from the OS environment.

    Returns:
        The API key value.

    Raises:
        WandbAuthError: if WANDB_API_KEY is unset or empty.

    The returned value MUST NOT be logged, printed, or written to any file.
    """
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise WandbAuthError(
            "WANDB_API_KEY is not set in your OS environment.\n\n"
            "On Windows (PowerShell, one time only):\n"
            "  [System.Environment]::SetEnvironmentVariable("
            '"WANDB_API_KEY", "<your-key>", "User")\n'
            "Then close every terminal/editor and re-open them.\n\n"
            "Verify with:\n"
            '  if ($env:WANDB_API_KEY) { "set, length=$($env:WANDB_API_KEY.Length)" } '
            'else { "NOT SET" }\n\n'
            "Do NOT paste the key into any file in this project."
        )
    return key


def masked_key_preview(key: str | None = None) -> str:
    """
    Return a masked preview of a W&B-shaped key.

    Shows only the first 4 and last 4 characters, with the rest replaced
    by asterisks. Use ONLY for human-eyeball verification that a key is
    loaded. Never log this to any persistent location.
    """
    k = key if key is not None else get_wandb_key()
    if len(k) < 12:
        return "****"
    return f"{k[:4]}{'*' * (len(k) - 8)}{k[-4:]}"


def init_wandb_run(
    project: str,
    name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    job_type: str | None = None,
    notes: str | None = None,
) -> Any:
    """
    Initialize a W&B run without exposing the API key.

    The key is checked up front (via get_wandb_key()); wandb itself
    reads WANDB_API_KEY from the environment internally.

    Args:
        project: W&B project name (e.g., "Boilerplate_Classifier").
        name: Optional run name. Auto-generated if None.
        config: Hyperparameter / config dict to log.
        tags: Filterable tags, e.g. ["stage:training", "model:setfit"].
        job_type: Logical job type, e.g. "extract", "label", "train", "eval".
        notes: Free-text notes about the run.

    Returns:
        The wandb.Run object.

    Raises:
        WandbAuthError: if WANDB_API_KEY is unset.
        ImportError: if wandb is not installed.
    """
    # Trigger key check up front for a clean error
    get_wandb_key()

    import wandb  # local import: keep this file usable even if wandb isn't installed

    return wandb.init(
        project=project,
        name=name,
        config=config or {},
        tags=tags or [],
        job_type=job_type,
        notes=notes,
    )


if __name__ == "__main__":
    # Manual sanity check: prints ONLY a masked key. Never the value.
    try:
        preview = masked_key_preview()
        print(f"WANDB_API_KEY loaded successfully: {preview}")
        print("Ready to use W&B.")
    except WandbAuthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
