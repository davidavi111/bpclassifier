"""
api_clients.py — judge-LLM client factories with env-based authentication.

Like gpu_runner.py for W&B, this is the ONLY place in the project that
reads judge-LLM API keys from the environment. All other modules MUST
import from here, never call os.environ directly for these keys.

All keys are optional: missing keys raise a clean error only when that
specific judge is requested. This means David can set up the project
with just WANDB_API_KEY and add other judges later without code changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Allow `from gpu_runner import ...` when this file is run from project root
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class JudgeAuthError(EnvironmentError):
    """Raised when a judge LLM's API key is required but unset."""


def _require_env(var: str, judge_name: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise JudgeAuthError(
            f"{var} is not set. Required to use {judge_name} as a judge.\n"
            f"Set it as a User-level OS environment variable in PowerShell:\n"
            f"  [System.Environment]::SetEnvironmentVariable("
            f'"{var}", "<key>", "User")\n'
            f"Then close every terminal/editor and re-open them.\n"
            f"Do NOT paste the key into any file in this project."
        )
    return val


def anthropic_client() -> Any:
    """Return an authenticated Anthropic (Claude) client."""
    from anthropic import Anthropic

    return Anthropic(api_key=_require_env("ANTHROPIC_API_KEY", "Anthropic Claude"))


def openai_client() -> Any:
    """Return an authenticated OpenAI (GPT) client."""
    from openai import OpenAI

    return OpenAI(api_key=_require_env("OPENAI_API_KEY", "OpenAI"))


def google_client() -> Any:
    """Return an authenticated Google Gemini client."""
    from google import genai

    return genai.Client(api_key=_require_env("GOOGLE_API_KEY", "Google Gemini"))


def ollama_host() -> str:
    """Return the configured Ollama host (default: http://localhost:11434)."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def wandb_inference_client() -> Any:
    """
    Return an OpenAI-compatible client pointed at W&B Inference.

    W&B Inference exposes hosted open-source LLMs through an
    OpenAI-compatible API; auth is via WANDB_API_KEY as bearer token.
    Useful as a fourth judge LLM without paying Anthropic/OpenAI.
    """
    from openai import OpenAI

    from gpu_runner import get_wandb_key

    return OpenAI(
        api_key=get_wandb_key(),
        base_url="https://api.inference.wandb.ai/v1",
    )


def list_available_judges() -> dict[str, bool]:
    """
    Return {judge_name: env_var_set} for diagnostics.
    Does NOT touch the actual key values, only checks presence.
    """
    return {
        "ollama": True,  # local, no key required
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY")),
        "wandb_inference": bool(os.environ.get("WANDB_API_KEY")),
    }


if __name__ == "__main__":
    print("Judge availability (key-presence check only, no values touched):")
    for name, available in list_available_judges().items():
        marker = "[ok]" if available else "[--]"
        print(f"  {marker} {name}")
