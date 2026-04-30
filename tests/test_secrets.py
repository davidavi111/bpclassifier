"""Test that secrets are handled safely.

These tests run WITHOUT touching the user's actual API keys — they
exercise the error paths and the masking logic with dummy values.
"""

import sys
from pathlib import Path

import pytest

# Add scripts/ to import path so we can test gpu_runner.py
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS.resolve()))

from gpu_runner import (  # noqa: E402
    WandbAuthError,
    get_wandb_key,
    masked_key_preview,
)


class TestGetWandbKey:
    def test_raises_when_unset(self, env_clean):
        with pytest.raises(WandbAuthError):
            get_wandb_key()

    def test_raises_when_empty(self, env_clean):
        env_clean.setenv("WANDB_API_KEY", "")
        with pytest.raises(WandbAuthError):
            get_wandb_key()

    def test_returns_value_when_set(self, env_clean):
        env_clean.setenv("WANDB_API_KEY", "dummy_test_value_for_unit_test_only_xx")
        assert get_wandb_key() == "dummy_test_value_for_unit_test_only_xx"

    def test_error_message_does_not_print_value(self, env_clean):
        """The error must not leak any actual key, even if one was previously set."""
        env_clean.setenv("WANDB_API_KEY", "should_not_appear_in_error")
        env_clean.delenv("WANDB_API_KEY")
        try:
            get_wandb_key()
        except WandbAuthError as e:
            assert "should_not_appear_in_error" not in str(e)


class TestMaskedKeyPreview:
    def test_preserves_first_and_last_four(self):
        masked = masked_key_preview("abcd1234567890wxyz")
        assert masked.startswith("abcd")
        assert masked.endswith("wxyz")
        assert "1234567890" not in masked

    def test_short_key_fully_masked(self):
        assert masked_key_preview("short") == "****"

    def test_no_full_key_in_output(self):
        full = "abcdefghijklmnopqrstuvwxyz1234567890"
        masked = masked_key_preview(full)
        assert masked != full
        # The middle should be masked
        assert masked.count("*") >= len(full) - 8
