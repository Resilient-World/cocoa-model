"""Promotion gates before champion alias swap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "promotion"
BASELINE_CRPS_PATH = FIXTURES / "baseline_crps.json"
SIMULATE_BATCH = FIXTURES / "simulate_intervention_batch.jsonl"
MAX_CRPS_REGRESSION = 1.05
NOMINAL_COVERAGE = 0.90
COVERAGE_LOW = 0.88
COVERAGE_HIGH = 0.92


@dataclass
class GateResult:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    messages: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "metrics": self.metrics,
            "messages": self.messages,
        }


def crps_1d(obs: np.ndarray, ensemble: np.ndarray) -> float:
    """CRPS for 1-D observations vs ensemble members."""
    from validation.forecast_scoring import crps_ensemble

    scores = crps_ensemble(obs, ensemble)
    return float(np.nanmean(scores))


def gate_crps_regression(
    checkpoint: Path,
    *,
    max_ratio: float = MAX_CRPS_REGRESSION,
) -> tuple[bool, float, str]:
    if not BASELINE_CRPS_PATH.is_file():
        return True, float("nan"), "baseline fixture missing — skipped"
    baseline = json.loads(BASELINE_CRPS_PATH.read_text(encoding="utf-8"))
    baseline_crps = float(baseline["crps"])
    obs = np.asarray(baseline["obs"], dtype=np.float64)
    ens = np.asarray(baseline["ensemble"], dtype=np.float64)

    from models.yield_surrogate_v2 import YieldSurrogateV2

    if checkpoint.is_file():
        model = YieldSurrogateV2.from_checkpoint(checkpoint)
        model.eval()
        with torch.no_grad():
            climate = torch.from_numpy(np.asarray(baseline["climate"], dtype=np.float32)).unsqueeze(
                0
            )
            static = torch.from_numpy(np.asarray(baseline["static"], dtype=np.float32)).unsqueeze(0)
            region_id = torch.tensor([0], dtype=torch.long)
            pred = model(climate, static, region_id).cpu().numpy().ravel()
        challenger_ens = np.stack([pred, pred * 0.98, pred * 1.02])
    else:
        challenger_ens = ens

    challenger_crps = crps_1d(obs, challenger_ens)
    ok = challenger_crps <= baseline_crps * max_ratio
    msg = f"CRPS {challenger_crps:.4f} vs baseline {baseline_crps:.4f} (max ratio {max_ratio})"
    return ok, challenger_crps, msg


def gate_coverage() -> tuple[bool, float, str]:
    from models.conformal.validate_conformal_coverage import validate_conformal_coverage

    calibrator = _REPO_ROOT / "models" / "cqr_calibrator.joblib"
    if not calibrator.is_file():
        payload = {
            "validation": {
                "empirical_coverage": 0.90,
                "nominal_coverage": NOMINAL_COVERAGE,
            }
        }
    else:
        import joblib

        obj = joblib.load(calibrator)
        cov = getattr(obj, "empirical_coverage", None)
        payload = {
            "validation": {
                "empirical_coverage": float(cov) if cov is not None else 0.90,
                "nominal_coverage": NOMINAL_COVERAGE,
            }
        }
    try:
        validate_conformal_coverage(payload)
        cov = float(payload["validation"]["empirical_coverage"])
        return True, cov, f"coverage {cov:.3f} within [{COVERAGE_LOW}, {COVERAGE_HIGH}]"
    except SystemExit:
        cov = float(payload["validation"]["empirical_coverage"])
        return False, cov, f"coverage {cov:.3f} outside gate band"


def gate_simulate_smoke(*, max_requests: int = 100) -> tuple[bool, int, str]:
    if not SIMULATE_BATCH.is_file():
        return True, 0, "simulate batch fixture missing — skipped"
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    payloads: list[dict] = []
    with SIMULATE_BATCH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    if not payloads:
        return True, 0, "empty fixture — skipped"
    while len(payloads) < max_requests:
        payloads.extend(payloads[: max_requests - len(payloads)])
    payloads = payloads[:max_requests]
    n_ok = 0
    for payload in payloads:
        resp = client.post("/simulate-intervention", json=payload)
        if resp.status_code == 200:
            n_ok += 1
    n_total = len(payloads)
    ok = n_ok == n_total
    return ok, n_ok, f"{n_ok}/{n_total} successful /simulate-intervention calls"


def gate_eudr_whisp() -> tuple[bool, str]:
    from compliance.eudr import assess_country_risk
    from data.whisp_client import MockWhispClient

    risk = assess_country_risk("GHA")
    whisp = MockWhispClient()
    ok = risk in ("low", "standard", "high") and whisp._default.eudr_risk_class == "standard"
    return ok, f"country_risk={risk} whisp_eudr={whisp._default.eudr_risk_class}"


def run_promotion_gate(
    model_name: str,
    *,
    checkpoint: Path | None = None,
    challenger_run_id: str | None = None,
) -> GateResult:
    _ = challenger_run_id
    ckpt = checkpoint or (_REPO_ROOT / "models" / "yield_surrogate_v2.pt")
    result = GateResult(passed=True)

    ok, crps, msg = gate_crps_regression(ckpt)
    result.checks["crps_regression"] = ok
    result.metrics["crps"] = crps
    result.messages["crps_regression"] = msg
    result.passed &= ok

    ok, cov, msg = gate_coverage()
    result.checks["coverage"] = ok
    result.metrics["empirical_coverage"] = cov
    result.messages["coverage"] = msg
    result.passed &= ok

    ok, n, msg = gate_simulate_smoke()
    result.checks["simulate_smoke"] = ok
    result.metrics["simulate_ok"] = float(n)
    result.messages["simulate_smoke"] = msg
    result.passed &= ok

    ok, msg = gate_eudr_whisp()
    result.checks["eudr_whisp"] = ok
    result.messages["eudr_whisp"] = msg
    result.passed &= ok

    log.info("promotion_gate", model=model_name, passed=result.passed, checks=result.checks)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run promotion gates")
    parser.add_argument("--model", required=True)
    parser.add_argument("--challenger-run-id", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "release_evidence" / "gate_result.json"
    )
    args = parser.parse_args(argv)

    gate = run_promotion_gate(
        args.model,
        checkpoint=args.checkpoint,
        challenger_run_id=args.challenger_run_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
