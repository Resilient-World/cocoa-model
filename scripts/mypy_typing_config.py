"""Load [tool.cocoa.typing] from pyproject.toml."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def load_typing_config() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    section = data.get("tool", {}).get("cocoa", {}).get("typing", {})
    return {
        "strict_enabled": list(section.get("strict_enabled", [])),
        "gradual_modules": list(section.get("gradual_modules", [])),
    }
