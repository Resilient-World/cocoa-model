# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- AgriFM (Video Swin) backbone (arXiv:2505.21357, RSE 2026): MIT reimplementation in `src/models/agrifm_*`, S2 weight download script, `--backbone agrifm` benchmark, and tests.

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
