"""Shared markdown report helpers for external validation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    """Single benchmark outcome with explicit PASS/FAIL gate."""

    name: str
    passed: bool
    metrics: dict[str, Any]
    gate_description: str
    notes: list[str]

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# {self.name}",
            "",
            f"**Gate:** {status}",
            "",
            f"**Criterion:** {self.gate_description}",
            "",
            "## Metrics",
            "",
        ]
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.4f}")
            else:
                lines.append(f"- `{key}`: {value}")
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)


def write_report(result: ValidationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_markdown(), encoding="utf-8")


def combine_summary(results: list[ValidationResult]) -> str:
    overall = all(r.passed for r in results)
    status = "PASS" if overall else "FAIL"
    lines = [
        "# External validation summary",
        "",
        f"**Overall:** {status}",
        "",
        "| Benchmark | Gate |",
        "|-----------|------|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)
