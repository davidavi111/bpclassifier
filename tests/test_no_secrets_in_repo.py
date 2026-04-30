"""
Repo-wide guard against accidentally committed secrets.

This is a backstop. Pre-commit hooks (detect-secrets + gitleaks) should
have already rejected the commit. This test catches anything that
slipped through.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / Anthropic style
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r"WANDB_API_KEY\s*=\s*['\"][^'\"]{30,}"),  # hardcoded W&B
    re.compile(r"api[_-]?key\s*=\s*['\"][A-Za-z0-9]{30,}"),
]

DIRS_TO_SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "wandb",
    ".secrets",
    "data",
    "ECT",
    "artifacts",
    "tests",  # tests can contain dummy secret-shaped strings intentionally
}

EXTENSIONS_TO_SCAN = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".ini",
    ".json",
    ".txt",
    ".sh",
    ".ps1",
}


def _iter_text_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in DIRS_TO_SKIP for part in path.parts):
            continue
        if path.suffix not in EXTENSIONS_TO_SCAN:
            continue
        yield path


def test_no_env_files_tracked():
    """No .env or other credential files should be present in the tree."""
    forbidden_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        "secrets.json",
        "credentials.json",
    }
    found = []
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in DIRS_TO_SKIP for part in path.parts):
            continue
        if path.name in forbidden_names:
            found.append(path)
    assert not found, f"Forbidden credential files present: {found}"


def test_no_secret_patterns_in_text_files():
    """Heuristic scan: no secret-shaped strings in any tracked text file."""
    matches = []
    for path in _iter_text_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                matches.append((path.relative_to(PROJECT_ROOT), pattern.pattern))
    assert not matches, "Possible secrets detected:\n" + "\n".join(
        f"  {p}: matched /{pat}/" for p, pat in matches
    )
