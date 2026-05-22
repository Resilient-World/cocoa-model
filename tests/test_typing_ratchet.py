"""CI guard: strict_enabled modules pass mypy --strict."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mypy_typing_config import load_typing_config

EXPECTED_STRICT = frozenset(
    {
        "api.config",
        "api.schemas",
        "analysis._report",
        "models.conformal.cqr",
    }
)


def test_strict_enabled_registry() -> None:
    cfg = load_typing_config()
    enabled = set(cfg["strict_enabled"])
    assert enabled, "strict_enabled must be non-empty"
    assert enabled >= EXPECTED_STRICT
    overlap = enabled & set(cfg["gradual_modules"])
    assert not overlap, f"modules must not be both strict and gradual: {overlap}"


@pytest.mark.parametrize("module", sorted(EXPECTED_STRICT))
def test_mypy_strict_module(module: str) -> None:
    env = {**os.environ, "MYPYPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "-p", module],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"mypy --strict -p {module} failed:\n{proc.stdout}\n{proc.stderr}"
