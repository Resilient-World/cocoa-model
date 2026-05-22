.PHONY: help train-all benchmark report ingest-gee ci lint typecheck test dvc-dag

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
export PYTHONPATH := $(REPO_ROOT)src

help:
	@echo "Targets:"
	@echo "  make ingest-gee   - Write data/raw ingest manifest + AOI"
	@echo "  make train-all    - DVC repro: AEF head, CASEJ surrogate, joint head"
	@echo "  make benchmark    - Backbone benchmark → reports/backbones/benchmark_latest.md"
	@echo "  make report       - Causal PDF report (synthetic panel)"
	@echo "  make lint         - ruff check"
	@echo "  make typecheck    - mypy src/"
	@echo "  make test         - pytest (fast subset)"
	@echo "  make dvc-dag      - Print DVC pipeline graph"
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
	pytest tests/test_joint_exposure_yield.py tests/test_api_simulate.py tests/test_feature_resolver.py tests/test_yield_surrogate.py -q

dvc-dag:
	dvc dag

ci: lint typecheck test dvc-dag
