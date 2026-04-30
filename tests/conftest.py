"""Shared pytest fixtures and configuration."""

import pytest

ENV_VARS_TO_GUARD = [
    "WANDB_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
]


@pytest.fixture
def env_clean(monkeypatch):
    """
    Test fixture that removes all sensitive env vars from the process env.
    Use this when testing missing-key error paths so tests don't accidentally
    pass because the developer happens to have a real key set.
    """
    for var in ENV_VARS_TO_GUARD:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch
