# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- BSSAL active learning + FixMatch SSL added for cocoa exposure label expansion in sparse-label regions (Xu et al. 2024 ISPRS-J; Kaijage et al. 2024 Sensors; Huang et al. 2025 arXiv:2511.18058).
- Causal discovery (PC, NOTEARS-MLP, GES via causal-learn/gCastle) and PCMCI+ teleconnection discovery (Tigramite subprocess) added to validate the mediation DAG and teleconnection-GNN structure.
- GEDI L4A + L3 canopy 3D features added; HuggingFace PEFT LoRA per-region adapters replace full fine-tunes (default for new region onboarding); new `/exposure-canopy` endpoint.
- Cocoa quality model (premium-pricing layer), CMIP7-readiness factory, and parametric-insurance pricing math added; new `/price-parametric` endpoint exposes basis-risk and DVDS-loaded premiums.
- Aurora 1.5 (Bodnar et al., Nature 2025) added as opt-in scenario backend; LoRA per-region adapters supported via `models/aurora_lora_<region>.safetensors`.
- `downscaling_method=aurora` on `POST /simulate-scenario` with `source_attributions` (`aurora_model_version`, `aurora_lora_id`); optional extra `[aurora]` (`microsoft-aurora`, `peft`).
- `scripts/validate_aurora_scenario.py` and `reports/scenario/aurora_vs_neuralgcm_vs_corrdiff.md` benchmark template.

### Changed

- Mypy strict typing enabled for api.config, api.schemas, analysis._report, models.cqr (ratchet initialized; remaining modules tracked in docs/TYPING_PLAYBOOK.md).

## [0.3.0] - 2026-05-21

### Added

- **Causal mediation (NDE/NIE):** `analysis.mediation`, optional `decompose_mediators` on `POST /simulate-intervention`, `api/mediation.py`, ρ sensitivity and multi-mediator path table; see [`docs/mediation_analysis.md`](docs/mediation_analysis.md).
- **Round-5 end-to-end demo v5:** `scripts/demo_end_to_end.py` → `reports/demo/e2e_civ_v5.json` + `.md` and stakeholder [`reports/demo/README.md`](reports/demo/README.md) covering TerraMind+TiM, WCTM drift, DVDS, CorrDiff (env-gated), policy tree, and mediation.
- Honest DR policy tree/forest targeting (`learn_policy_tree`, `learn_policy_forest`, `POST /learn-policy-rules`, `scripts/learn_targeting_rules.py`); see [`docs/policy_targeting.md`](docs/policy_targeting.md).
- CorrDiff-CMIP6 optional scenario downscaling (`downscaling_method=corrdiff`); see [`docs/corrdiff_compute.md`](docs/corrdiff_compute.md).
- WCTM drift monitoring and `drift_status` on `/simulate-scenario`.
- DVDS sensitivity bounds on `POST /simulate-intervention` (`include_sensitivity`); see [`docs/sensitivity.md`](docs/sensitivity.md).
- TerraMind 1.0 + TiM exposure path (`terramind_tim` backend) and ensemble v3 opt-in.

### Changed

- Package and FastAPI version **0.3.0**; demo default output paths use `e2e_civ_v5.*`.

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

[0.3.0]: https://github.com/Resilient-World/cocoa-model/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Resilient-World/cocoa-model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Resilient-World/cocoa-model/releases/tag/v0.1.0
