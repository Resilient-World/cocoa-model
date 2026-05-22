#!/usr/bin/env python3
"""Fail CI if bare print() calls exist under src/ (use structlog instead)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _find_prints(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            hits.append((node.lineno, "print()"))
    return hits


def main() -> int:
    violations: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        for lineno, kind in _find_prints(py):
            rel = py.relative_to(REPO)
            violations.append(f"{rel}:{lineno}: {kind}")
    if violations:
        print("Bare print() found in src/ — use structlog.get_logger(__name__):", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("OK: no print() in src/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
