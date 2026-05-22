#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:8000}"
TOKEN="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/reports/loadtest"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUT_DIR/k6_summary_${DATE_TAG}.json"

mkdir -p "$OUT_DIR"
export URL TOKEN

if [[ "${K6_SMOKE:-}" == "1" ]]; then
  echo "Running k6 smoke profile against $URL"
fi

run_one() {
  local script="$1"
  local name
  name="$(basename "$script" .js)"
  echo "==> k6 $name"
  k6 run --summary-export="$OUT_DIR/k6_${name}_${DATE_TAG}.json" \
    -e URL="$URL" -e TOKEN="$TOKEN" -e K6_SMOKE="${K6_SMOKE:-}" \
    "$script"
}

FAIL=0
for script in "$ROOT/tests/loadtest/k6/simulate_intervention.js" \
  "$ROOT/tests/loadtest/k6/simulate_scenario.js" \
  "$ROOT/tests/loadtest/k6/concurrent_endpoints.js"; do
  if ! run_one "$script"; then
    FAIL=1
  fi
done

python3 - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
parts = sorted(out.glob("k6_*_${DATE_TAG}.json"))
summary = {"url": "$URL", "smoke": "${K6_SMOKE:-}", "runs": []}
for p in parts:
    summary["runs"].append({"file": p.name, "data": json.loads(p.read_text())})
Path("$SUMMARY").write_text(json.dumps(summary, indent=2))
print("Wrote", "$SUMMARY")
PY

exit "$FAIL"
