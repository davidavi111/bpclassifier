"""tests/test_stage_7_gui.py — Stage 7 GUI contract tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_EVAL_DIR = _ROOT / "data" / "eval"
_ECT_DIR = _ROOT / "ECT"


def test_streamlit_app_imports() -> None:
    """app/streamlit_app.py must be importable without launching the server."""
    import sys

    sys.path.insert(0, str(_ROOT / "scripts"))
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT / "app"))

    # Patch streamlit so no server starts during import
    import unittest.mock as mock

    with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(
            "streamlit_app", _ROOT / "app" / "streamlit_app.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        # Import should not raise
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception as exc:
            pytest.fail(f"streamlit_app.py import raised: {exc}")


def test_eyeball_picks_exist() -> None:
    path = _EVAL_DIR / "eyeball_picks.json"
    if not path.exists():
        pytest.skip("eyeball_picks.json not found — run 07_gui_smoke_test.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "boilerplate_heavy" in data
    assert "qa_heavy" in data


def test_eyeball_picks_are_valid_ect_paths() -> None:
    path = _EVAL_DIR / "eyeball_picks.json"
    if not path.exists():
        pytest.skip("eyeball_picks.json not found — run 07_gui_smoke_test.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("boilerplate_heavy", "qa_heavy"):
        ect_path = Path(data[key])
        assert ect_path.exists(), f"{key} path does not exist: {ect_path}"
        assert ect_path.suffix == ".txt", f"{key} is not a .txt file"


def test_smoke_test_output() -> None:
    """The smoke-test output files exist after running 07_gui_smoke_test.py."""
    path = _EVAL_DIR / "eyeball_picks.json"
    if not path.exists():
        pytest.skip("eyeball_picks.json not found — run 07_gui_smoke_test.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 2, "eyeball_picks.json must have at least 2 entries"
