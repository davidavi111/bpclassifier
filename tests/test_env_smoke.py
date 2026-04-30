"""Environment smoke test: every project dependency imports successfully."""

import sys

import pytest


def test_python_version():
    assert sys.version_info >= (3, 12), f"Python 3.12+ required, got {sys.version}"


@pytest.mark.parametrize(
    "module_name",
    [
        "sklearn",
        "nltk",
        "sentence_transformers",
        "setfit",
        "transformers",
        "datasets",
        "torch",
        "pandas",
        "numpy",
        "pyarrow",
        "plotly",
        "matplotlib",
        "streamlit",
        "wandb",
        "weave",
        "pytest",
    ],
)
def test_dependency_imports(module_name):
    """Every required dependency must import without error."""
    __import__(module_name)


def test_project_package_imports():
    """The bpclassifier package must be importable (i.e., pip install -e . has been run)."""
    import bpclassifier

    assert hasattr(bpclassifier, "__version__")
