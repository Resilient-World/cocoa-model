# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Synthetic DiD (Arkhangelsky et al. 2021) and DiD method comparison harness (`analysis.synthdid`, `analysis.did_comparison_harness`).
- Callaway-Sant'Anna staggered DiD and Borusyak-Jaravel-Spiess imputation estimators for valid causal estimation under staggered shade-tree/agroforestry rollouts (`analysis.csdid`, `analysis.bjs_imputation`, `did_estimator` dispatcher).
- Adaptive Conformal Inference (ACI, Conformal PID, ECI per Wu et al. 2025 ICLR), MultiStepACI for `/simulate-scenario` horizon stratification; online CQR wrapper and `validate_conformal_coverage --benchmark-online` report.
- `scripts/calibrate_online_conformal.py`, `scripts/validate_scenario_coverage.py`, and [`docs/conformal_calibration.md`](docs/conformal_calibration.md) for scenario conformal bootstrap and 48-strata coverage gates.
- `src/api/online_conformal_store.py`, `src/api/scenario_conformal.py`, and `tests/test_api_scenario_online.py`.
- [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md): full GPU/HPC checklist for exposure, yield, CQR, CASEJ, and avoided-loss API artifacts; notes on stopping incomplete laptop runs.
- AgriFM (Video Swin) backbone (arXiv:2505.21357, RSE 2026): MIT reimplementation in `src/models/agrifm_*`, S2 weight download script, `--backbone agrifm` benchmark, and tests.
- AgriFM cocoa fine-tuning (`training.train_agrifm_cocoa`, `models/agrifm_cocoa_seg.pt`) with BCE+Dice loss and hard-example mining.
- Ensemble v2: per-region weights in `config/ensemble_weights.yaml`, `scripts/fit_ensemble_v2_weights.py`, and `benchmark_ensemble_v2_*` reports.

### Changed

- `/simulate-scenario` uses ECI-Integral online conformal calibration by default; static split-CQR remains available via `CONFORMAL_METHOD` env var.
- Production exposure backbone now Galileo + AEF + AgriFM weighted ensemble per region (`ensemble_v2`, default via `COCOA_EXPOSURE_BACKEND=ensemble_v2`).

## [0.2.0] - 2026-05-20

### Added

- **Regional FDP pipeline**: `REGIONS` presets in `cocoa_exposure.py` (Ghana, CIV, Cameroon, Nigeria, Indonesia, Ecuador, Peru, Colombia); `--region` on `era5_ingest` and `sentinel_composite`; per-region backbone benchmarks (`benchmark_<region>_<date>.md`).
- **Global exposure fallback**: AEF + Galileo blend outside native FDP coverage (`sample_cocoa_probability_at_point`).
- **End-to-end demo**: `scripts/demo_end_to_end.py` — Whisp EUDR, FDP/AEF exposure, ERA5 + ATTRICI counterfactual, CMIP6 SSP5-8.5 2050 scenario (CASEJ), climate attribution, and JSON report with source citations.
- **Tests**: `tests/test_demo_end_to_end.py`, `tests/test_regions.py`.

### Changed

- `api/feature_resolver` routes cocoa probability through region-aware FDP vs global fallback.
- Package version **0.2.0** (`pyproject.toml`, FastAPI app metadata).

## [0.1.0] - 2026-05

### Added

- Initial release: ERA5 ingest, Sentinel composites, FDP cocoa exposure, yield surrogate API, causal analysis utilities, EUDR/Whisp integration, CMIP6 scenarios, and ATTRICI counterfactual boundary.

[0.2.0]: https://github.com/Resilient-World/cocoa-model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Resilient-World/cocoa-model/releases/tag/v0.1.0
