#!/usr/bin/env python3
"""Discover mypy strict failures and emit [tool.cocoa.typing] + override stanzas."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PYPROJECT = REPO / "pyproject.toml"

# Outermost packages first (plan: api / analysis report before inner torch/data).
PACKAGE_RANK: dict[str, int] = {
    "api": 0,
    "analysis": 1,
    "compliance": 2,
    "finance": 3,
    "monitoring": 4,
    "registry": 5,
    "validation": 6,
    "counterfactual": 7,
    "hazards": 8,
    "training": 9,
    "data": 10,
    "models": 11,
    "common": 12,
}

ERROR_RE = re.compile(r"^(?P<path>(?:src/)?[\w./-]+\.py):\d+: error:")


def _module_from_path(path: str) -> str | None:
    p = path.replace("\\", "/")
    if not p.startswith("src/"):
        return None
    rel = Path(p).relative_to("src").with_suffix("")
    return ".".join(rel.parts)


def _all_src_modules() -> list[str]:
    mods: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(SRC).with_suffix("")
        mods.append(".".join(rel.parts))
    return mods


def _rank_key(module: str) -> tuple[int, int, str]:
    top = module.split(".", 1)[0]
    depth = module.count(".")
    return (PACKAGE_RANK.get(top, 99), depth, module)


def _run_mypy_strict() -> str:
    env = {**dict(**__import__("os").environ), "MYPYPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "src/"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def _parse_modules(log: str) -> tuple[dict[str, int], dict[str, int]]:
    """Return per-module error counts and per top-level package totals."""
    per_mod: dict[str, int] = defaultdict(int)
    per_pkg: dict[str, int] = defaultdict(int)
    for line in log.splitlines():
        m = ERROR_RE.match(line.strip())
        if not m:
            continue
        mod = _module_from_path(m.group("path"))
        if mod is None:
            continue
        per_mod[mod] += 1
        per_pkg[mod.split(".", 1)[0]] += 1
    return dict(per_mod), dict(per_pkg)


def _emit_toml(
    gradual: list[str],
    *,
    strict_enabled: list[str],
) -> str:
    strict_set = set(strict_enabled)
    gradual = [m for m in gradual if m not in strict_set]
    lines = [
        "[tool.cocoa.typing]",
        "strict_enabled = [",
    ]
    for m in strict_enabled:
        lines.append(f'  "{m}",')
    lines.append("]")
    lines.append("gradual_modules = [")
    for m in gradual:
        lines.append(f'  "{m}",')
    lines.append("]")
    lines.append("")
    if gradual:
        lines.append("[[tool.mypy.overrides]]")
        lines.append("module = [")
        for m in gradual:
            lines.append(f'  "{m}",')
        lines.append("]")
        lines.append('disable_error_code = ["misc", "arg-type", "return-value"]')
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logfile",
        nargs="?",
        help="Optional mypy log; if omitted, runs mypy --strict src/ (requires ignore_errors removed)",
    )
    parser.add_argument(
        "--strict-enabled",
        nargs="*",
        default=[],
        help="Modules to exclude from gradual list",
    )
    parser.add_argument("--update-pyproject", action="store_true")
    args = parser.parse_args(argv)

    log = Path(args.logfile).read_text(encoding="utf-8") if args.logfile else _run_mypy_strict()
    per_mod, per_pkg = _parse_modules(log)

    if not per_mod:
        gradual = _all_src_modules()
    else:
        gradual = sorted(per_mod.keys(), key=_rank_key)

    fragment = _emit_toml(gradual, strict_enabled=args.strict_enabled)
    print(fragment)
    print("# Per-package failure counts:", file=sys.stderr)
    for pkg, n in sorted(per_pkg.items(), key=lambda x: (_rank_key(x[0] + ".x")[0], -x[1])):
        print(f"  {pkg}: {n}", file=sys.stderr)

    if args.update_pyproject:
        text = PYPROJECT.read_text(encoding="utf-8")
        start = text.find("[tool.cocoa.typing]")
        end = text.find("[tool.ruff]")
        if start == -1 or end == -1:
            print("Could not find insertion markers in pyproject.toml", file=sys.stderr)
            return 1
        new_block = fragment + "\n"
        PYPROJECT.write_text(text[:start] + new_block + text[end:], encoding="utf-8")
        print(f"Updated {PYPROJECT}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
