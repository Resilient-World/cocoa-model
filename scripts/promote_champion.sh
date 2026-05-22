#!/usr/bin/env bash
# Run promotion gates then atomically promote MLflow challenger → champion.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${1:-yield_surrogate_v2}"
RUN_ID="${2:-}"

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <model_name> <challenger_run_id>" >&2
  exit 1
fi

echo "==> Promotion gates for ${MODEL} (run ${RUN_ID})"
python -m registry.promotion_gate \
  --model "$MODEL" \
  --challenger-run-id "$RUN_ID" \
  --out release_evidence/gate_result.json

echo "==> Promoting challenger to champion"
python -m registry.promote --model "$MODEL" --gate-result release_evidence/gate_result.json --env staging

echo "==> Promotion complete"
