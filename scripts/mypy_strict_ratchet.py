#!/usr/bin/env python3
"""Run mypy --strict on [tool.cocoa.typing].strict_enabled modules."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from mypy_typing_config import load_typing_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = load_typing_config()
    strict = cfg["strict_enabled"]
    if not strict:
        print("No strict_enabled modules in pyproject.toml [tool.cocoa.typing]", file=sys.stderr)
        return 1

    env = {**os.environ, "MYPYPATH": str(SRC)}
    pkg_counts: dict[str, int] = defaultdict(int)
    failed = 0

    for module in strict:
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", "-p", module],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        pkg = module.split(".", 1)[0]
        err_count = sum(1 for line in proc.stdout.splitlines() if ": error:" in line)
        err_count += sum(1 for line in proc.stderr.splitlines() if ": error:" in line)
        pkg_counts[pkg] += err_count
        status = "ok" if proc.returncode == 0 else f"FAIL ({err_count} errors)"
        print(f"  {module}: {status}")
        if proc.returncode != 0:
            failed += 1
            if proc.stdout:
                print(proc.stdout, file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)

    print("\nPer-package failure counts:")
    for pkg in sorted(pkg_counts):
        print(f"  {pkg}: {pkg_counts[pkg]}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
