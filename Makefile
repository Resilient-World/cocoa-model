.PHONY: help train-all benchmark report ingest-gee ci lint typecheck test dvc-dag dvc-repro hpo promote validate-spatial validate-temporal validate-calibration plot-reliability loadtest

REGION ?= ghana

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(REPO_ROOT)src

MODEL ?= yield
N ?= 50
PROMOTE_MODEL ?= yield_surrogate_v2
PROMOTE_RUN_ID ?=

help:
	@echo "Targets:"
	@echo "  make ingest-gee   - Write data/raw ingest manifest + AOI"
	@echo "  make train-all    - DVC repro: AEF head, CASEJ surrogate, joint head"
	@echo "  make dvc-repro    - Full mock-GEE DVC pipeline (params.mock_gee=true)"
	@echo "  make hpo MODEL=yield N=50 - Optuna HPO → MLflow"
	@echo "  make promote MODEL=yield_surrogate_v2 - Promotion gate + champion alias"
	@echo "  make benchmark    - Backbone benchmark → reports/backbones/benchmark_latest.md"
	@echo "  make report       - Causal PDF report (synthetic panel)"
	@echo "  make lint         - ruff check"
	@echo "  make typecheck    - mypy src/"
	@echo "  make test         - pytest (fast subset)"
	@echo "  make dvc-dag      - Print DVC pipeline graph"
	@echo "  make validate-spatial REGION=ghana - Spatial block CV report"
	@echo "  make validate-temporal            - Forward-chain temporal CV report"
	@echo "  make validate-calibration         - CRPS/ECE/PIT/sharpness calibration gate"
	@echo "  make plot-reliability             - Reliability figure from calibration_latest.json"
	@echo "  make loadtest URL=http://localhost:8000 - k6 load tests (optional TOKEN=)"
	@echo "  make ci           - lint + test + dvc-dag (local CI)"

ingest-gee:
	python scripts/ingest_gee.py

train-all:
	dvc repro stage_ingest_gee stage_train_aef_head stage_train_casej_surrogate stage_train_joint

benchmark:
	dvc repro stage_benchmark

report:
	python scripts/generate_causal_report.py --synthetic --n-farms 500 --out reports/causal/causal_report_latest.pdf

lint:
	ruff check src tests scripts

typecheck:
	mypy src

test:
	pytest tests/test_joint_exposure_yield.py tests/test_api_simulate.py tests/test_feature_resolver.py tests/test_yield_surrogate.py tests/validation/ -q

dvc-dag:
	dvc dag

dvc-repro:
	dvc repro

hpo:
	python scripts/run_hpo.py --model $(MODEL) --n-trials $(N)

promote:
	scripts/promote_champion.sh $(PROMOTE_MODEL) $(PROMOTE_RUN_ID)

validate-spatial:
	python scripts/validate_spatial_holdout.py --region $(REGION) --block-size-km 50

validate-temporal:
	python scripts/validate_temporal_holdout.py

validate-calibration:
	python -m models.conformal.validate_conformal_coverage --calibration-gate --synthetic --cv-strategy spatial_block --out reports/validation

plot-reliability:
	python scripts/plot_reliability.py --model cqr_yield --scores reports/validation/calibration_latest.json

loadtest:
	@command -v k6 >/dev/null || (echo "install k6: https://k6.io" && exit 1)
	chmod +x scripts/run_k6_loadtest.sh
	./scripts/run_k6_loadtest.sh "$(URL)" "$(TOKEN)"

benchmark-olmoearth:
	python scripts/write_backbone_comparison_reports.py --stub-only

fit-ensemble-v4:
	python scripts/fit_ensemble_v4_weights.py --synthetic

validate-neuralgcm:
	python scripts/validate_neuralgcm_scenario.py

run-tcav:
	python scripts/run_tcav_analysis.py

compare-dml-nuisances:
	python scripts/compare_dml_nuisances.py

ci: lint typecheck test dvc-dag
